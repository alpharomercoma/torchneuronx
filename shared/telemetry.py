#!/usr/bin/env python3
"""1 Hz accelerator telemetry sampler. Runs on the HOST, stdlib only.

Every benchmark in this repo is wrapped by this sampler. A result without a
matching telemetry CSV does not go in the report -- that is what lets us say
"the accelerator was saturated" as a measurement rather than an assumption,
and it is what backs the tokens-per-joule numbers.

Output is one unified CSV schema for every vendor (the column is still named
gpu_util_pct on Neuron so downstream parsing is identical across repos):

    t_s,gpu_util_pct,mem_used_mib,power_w,temp_c,sclk_mhz,mclk_mhz

Neuron reads a single long-lived `neuron-monitor` process emitting one JSON
object per second; AMD reads amdgpu sysfs; NVIDIA shells a fast nvidia-smi
query. In all three cases the design goal is the same: sampling must not
distort the 1 Hz cadence or steal host CPU from the benchmark.

Usage:
    telemetry.py --out run.csv -- <command to run and measure>
    telemetry.py --out run.csv --duration 600        # sample without a command
"""

import argparse
import atexit
import csv
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

SCHEMA = ["t_s", "gpu_util_pct", "mem_used_mib", "power_w", "temp_c", "sclk_mhz", "mclk_mhz"]

# Logical NeuronCores per instance, for the per-core telemetry columns. Kept
# here rather than imported from sft_lora.py so telemetry stays dependency-free
# and importable on a box with no training stack installed.
DEVICE_CORES = {"trn1": 2, "inf2": 2, "trn2": 4}
DEFAULT_CORES = 2


def neuron_core_count(env=None):
    """How many util_nc<i> columns to emit. Pure given env, so it is tested.

    NP_TELEMETRY_CORES wins (the trn2 TP ladder may run at LNC=1 with 8 cores,
    which no static table can predict), then NP_DEVICE, then 2 -- the value
    every Phase-1/2 CSV was captured with.
    """
    env = os.environ if env is None else env
    raw = (env.get("NP_TELEMETRY_CORES") or "").strip()
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return DEVICE_CORES.get((env.get("NP_DEVICE") or "").strip().lower(),
                            DEFAULT_CORES)


def _read(path, scale=1.0, default=None):
    try:
        with open(path) as fh:
            return float(fh.read().strip()) / scale
    except Exception:
        return default


class AmdSampler:
    """Reads amdgpu sysfs directly.

    The MI300X on this droplet is an SR-IOV virtual function, so the card
    number is not stable at 0 and several nodes are the virtio display device.
    We therefore locate the card whose vendor is AMD (0x1002) and which
    actually exposes gpu_busy_percent.
    """

    vendor = "AMD"

    def __init__(self):
        self.dev = None
        for card in sorted(glob.glob("/sys/class/drm/card*/device")):
            vend = None
            try:
                with open(os.path.join(card, "vendor")) as fh:
                    vend = fh.read().strip()
            except Exception:
                continue
            if vend == "0x1002" and os.path.exists(os.path.join(card, "gpu_busy_percent")):
                self.dev = card
                break
        if self.dev is None:
            raise RuntimeError("no amdgpu card with gpu_busy_percent found")

        hw = glob.glob(os.path.join(self.dev, "hwmon", "hwmon*"))
        self.hwmon = hw[0] if hw else None

    def sample(self):
        d = self.dev
        util = _read(os.path.join(d, "gpu_busy_percent"))
        used = _read(os.path.join(d, "mem_info_vram_used"), scale=1024 * 1024)
        power = temp = sclk = mclk = None
        if self.hwmon:
            h = self.hwmon
            # power1_average is the meaningful one where present; fall back to input
            power = _read(os.path.join(h, "power1_average"), scale=1e6)
            if power is None:
                power = _read(os.path.join(h, "power1_input"), scale=1e6)
            # temp2 = junction on MI300; temp1 = edge
            temp = _read(os.path.join(h, "temp2_input"), scale=1000.0)
            if temp is None:
                temp = _read(os.path.join(h, "temp1_input"), scale=1000.0)
            sclk = _read(os.path.join(h, "freq1_input"), scale=1e6)
            mclk = _read(os.path.join(h, "freq2_input"), scale=1e6)
        return [util, used, power, temp, sclk, mclk]


class NvidiaSampler:
    vendor = "NVIDIA"
    QUERY = ("utilization.gpu,memory.used,power.draw,temperature.gpu,"
             "clocks.sm,clocks.mem")

    def __init__(self):
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, timeout=30)

    def sample(self):
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self.QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        vals = []
        for part in out.split(","):
            part = part.strip()
            try:
                vals.append(float(part))
            except ValueError:
                vals.append(None)
        return vals


class NeuronSampler:
    """Reads a long-lived `neuron-monitor` process (one JSON object per line).

    Column mapping into the unified schema:
        gpu_util_pct  <- mean NeuronCore utilization across cores in use
        mem_used_mib  <- device (HBM) bytes used by all Neuron runtimes
        power_w/temp_c/sclk_mhz/mclk_mhz <- not exposed by neuron-monitor on
            trn1/inf2; left empty rather than fabricated. summarize.py reports
            tokens_per_joule as null when the power column is empty.
    Extra columns: util_nc0..util_nc<N-1>, host_mem_used_mib, where N is the
    logical NeuronCore count. N is 2 (Trainium1/Inferentia2) unless the driver
    exports NP_TELEMETRY_CORES or NP_DEVICE -- see neuron_core_count(). The
    class attribute below stays the 2-core form so a CSV captured on trn1 in
    Phase 1 still has exactly the columns it had then.

    When no runtime is attached, neuron-monitor emits no neuron_runtime_data
    entries; utilization is then recorded as 0.0 -- an idle accelerator is a
    measurement, not a missing value.
    """

    vendor = "NEURON"
    extra = ["util_nc0", "util_nc1", "host_mem_used_mib"]

    CONFIG = {
        "period": "1s",
        "neuron_runtimes": [{
            "tag_filter": ".*",
            "metrics": [
                {"type": "neuroncore_counters"},
                {"type": "memory_used"},
                {"type": "neuron_runtime_vcpu_usage"},
            ],
        }],
        "system_metrics": [
            {"type": "memory_info"},
            {"type": "vcpu_usage"},
            {"type": "neuron_hw_counters"},
        ],
    }

    def __init__(self):
        self.cores = neuron_core_count()
        self.extra = ([f"util_nc{i}" for i in range(self.cores)]
                      + ["host_mem_used_mib"])

        exe = "/opt/aws/neuron/bin/neuron-monitor"
        if not os.access(exe, os.X_OK):
            exe = shutil.which("neuron-monitor")
        if not exe:
            raise RuntimeError("neuron-monitor not found")

        cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="neuron-monitor-", delete=False)
        json.dump(self.CONFIG, cfg)
        cfg.close()

        self.proc = subprocess.Popen(
            [exe, "-c", cfg.name],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        atexit.register(self.proc.terminate)

        self._lock = threading.Lock()
        self._latest = None
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        for line in self.proc.stdout:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                self._latest = obj

    @staticmethod
    def parse(obj, cores=2):
        """neuron-monitor JSON -> (util_mean, mem_used_mib, util_nc0 ..
        util_nc<cores-1>, host_mem_used_mib). Split out from sample() so it is
        testable against a committed fixture without hardware.

        `cores` defaults to 2 so the trn1/inf2 call shape and the Phase-1 CSV
        columns are unchanged; util_mean is always the mean over the cores
        actually reporting, never over `cores`, so a wrong count cannot inflate
        the headline utilization figure."""
        utils = {}
        dev_bytes = 0
        host_bytes = 0
        for rt in obj.get("neuron_runtime_data") or []:
            report = rt.get("report") or {}
            in_use = (report.get("neuroncore_counters") or {}).get(
                "neuroncores_in_use") or {}
            for idx, counters in in_use.items():
                u = (counters or {}).get("neuroncore_utilization")
                if u is not None:
                    utils[int(idx)] = utils.get(int(idx), 0.0) + float(u)
            used = (report.get("memory_used") or {}).get(
                "neuron_runtime_used_bytes") or {}
            dev_bytes += used.get("neuron_device") or 0
            host_bytes += used.get("host") or 0
        util_mean = (round(sum(utils.values()) / len(utils), 2)
                     if utils else 0.0)
        return ((util_mean, round(dev_bytes / (1024 * 1024), 1))
                + tuple(round(utils.get(i, 0.0), 2) for i in range(cores))
                + (round(host_bytes / (1024 * 1024), 1),))

    def sample(self):
        with self._lock:
            obj = self._latest
        if obj is None:
            # neuron-monitor has not emitted its first line yet
            return [None] * (len(SCHEMA) - 1 + len(self.extra))
        util, dev_mib, *per_core, host_mib = self.parse(obj, self.cores)
        return ([util, dev_mib, None, None, None, None]
                + per_core + [host_mib])


def make_sampler():
    for cls in (NeuronSampler, AmdSampler, NvidiaSampler):
        try:
            return cls()
        except Exception:
            continue
    raise RuntimeError(
        "no telemetry source found (tried neuron-monitor, amdgpu sysfs, nvidia-smi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="sample for N seconds instead of wrapping a command")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    sampler = make_sampler()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd

    stop = threading.Event()
    rc = {"code": 0}

    def run_cmd():
        try:
            rc["code"] = subprocess.call(cmd)
        finally:
            stop.set()

    worker = None
    if cmd:
        worker = threading.Thread(target=run_cmd, daemon=True)
        worker.start()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    header = SCHEMA + list(getattr(sampler, "extra", []))
    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        while not stop.is_set():
            try:
                row = sampler.sample()
            except Exception:
                row = [None] * (len(header) - 1)
            w.writerow([round(time.time() - t0, 2)] + row)
            fh.flush()
            if args.duration is not None and time.time() - t0 >= args.duration:
                break
            stop.wait(args.interval)

    if worker:
        worker.join(timeout=30)
    sys.exit(rc["code"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)
