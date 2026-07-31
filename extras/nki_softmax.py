"""Track E: a custom NKI kernel (row softmax) vs the framework op.

The point of this lane is ecosystem depth: NKI is Neuron's answer to
"can I write my own kernels?" (the Triton/CUDA-kernel moment of this stack).
We hit NKI from the outside when vLLM 0.21's kernels asserted on our silicon;
this lane goes to the inside: write one, prove it correct, race it.

Kernel: numerically-stable softmax over the free axis of a (128, N) tile --
one row per partition, max/exp/sum on the Vector/Scalar engines. Chosen
because it exercises reductions + elementwise + broadcast without touching
the tensor engine, and a numpy reference fits in four lines.

Discipline (nki 0.5.0, per the on-box inventory):
  * dev/correctness via `nki.simulate` on CPU -- zero NeuronCore time;
  * device timing via wall-clock around synced calls (the ncc_driver
    benchmark helper is attempted first; its API is recorded as observed);
  * `mode=`/`baremetal`/`platform_target` are deprecated -- never used here.

    python3 extras/nki_softmax.py --mode simulate --out /tmp/nki_sim.json
    python3 extras/nki_softmax.py --mode device --out nki_softmax.json
"""
import argparse
import json
import math
import os
import sys
import time

PARTITIONS = 128          # NeuronCore-v2 SBUF partition count
FREE_DIM = 512
BENCH_ITERS = 200
WARMUP = 20


def numpy_reference(x):
    import numpy as np
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


# Module-level on purpose: the device compiler resolves `nl.*` names against
# the kernel's MODULE globals; with a function-local import the first device
# compile died with "failed to resolve name 'nl.load'" (simulate, which
# interprets the python directly, was fine). Import guarded so --help works
# off-box.
try:
    import nki
    import nki.language as nl
except ImportError:                       # receipts cover this on-box
    nki = nl = None


def make_kernel():
    """Build the @nki.jit kernel (imports resolved at module scope above)."""

    @nki.jit
    def row_softmax(x_in):
        # NkiTensor does not overload python arithmetic (measured: simulate
        # raised "unsupported operand type(s) for -" on `x - row_max`);
        # everything goes through nl.* ops explicitly.
        x = nl.load(x_in)
        row_max = nl.max(x, axis=1, keepdims=True)
        e = nl.exp(nl.subtract(x, row_max))
        denom = nl.sum(e, axis=1, keepdims=True)
        out = nl.divide(e, denom)
        out_t = nl.ndarray(x.shape, dtype=x_in.dtype, buffer=nl.shared_hbm)
        nl.store(out_t, out)
        return out_t

    return row_softmax


def run_simulate(out_path, seed=0):
    import numpy as np
    import nki

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((PARTITIONS, FREE_DIM), dtype=np.float32)
    kernel = make_kernel()
    t0 = time.perf_counter()
    # nki 0.5.0's public simulate is a wrapper-factory: simulate(kernel)
    # returns a CPU-simulated callable (first attempt failed with
    # "simulate() takes 1 positional argument but 3 were given" -- receipt
    # kept in git history). Fall back to the internal simulate_kernel
    # signature the inventory found if the wrapper convention drifts again.
    try:
        got = nki.simulate(kernel)(x)
    except TypeError:
        from nki.simulator import simulate_kernel
        got = simulate_kernel(kernel, (x,), {})
    sim_s = time.perf_counter() - t0
    want = numpy_reference(x)
    max_err = float(abs(np.asarray(got) - want).max())
    ok = max_err < 1e-5
    payload = {
        "mode": "simulate", "shape": [PARTITIONS, FREE_DIM],
        "max_abs_err_vs_numpy": max_err, "correct": ok,
        "simulate_wall_s": round(sim_s, 2), "seed": seed,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write(out_path, payload)
    print(f"simulate: correct={ok} max_err={max_err:.2e} -> {out_path}")
    return 0 if ok else 1


def run_device(out_path, seed=0):
    """Benchmark on a NeuronCore: NKI kernel vs torch softmax, same tile."""
    import numpy as np
    import torch
    import torch_xla

    device = torch_xla.device()
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((PARTITIONS, FREE_DIM), dtype=np.float32)
    x = torch.from_numpy(x_np).to(device)
    kernel = make_kernel()

    def timed(fn):
        for _ in range(WARMUP):
            fn()
        torch_xla.sync()
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            fn()
        torch_xla.sync()
        return (time.perf_counter() - t0) / BENCH_ITERS * 1e6  # us/iter

    # Correctness on device first (one call, compared to numpy).
    got = kernel(x)
    torch_xla.sync()
    max_err = float((got.cpu().numpy() - numpy_reference(x_np)).__abs__().max())

    nki_us = timed(lambda: kernel(x))
    torch_us = timed(lambda: torch.nn.functional.softmax(x, dim=1))

    payload = {
        "mode": "device", "shape": [PARTITIONS, FREE_DIM],
        "iters": BENCH_ITERS, "warmup": WARMUP,
        "max_abs_err_vs_numpy": max_err, "correct": max_err < 1e-5,
        "nki_us_per_call": round(nki_us, 2),
        "torch_softmax_us_per_call": round(torch_us, 2),
        "nki_vs_torch_ratio": round(nki_us / torch_us, 3) if torch_us else None,
        "timing_note": ("wall-clock around torch_xla.sync'd loops; includes "
                        "XLA dispatch overhead identically for both sides -- "
                        "a relative number, honest for comparison, not an "
                        "absolute kernel latency"),
        "seed": seed,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write(out_path, payload)
    print(f"device: nki={nki_us:.1f}us torch={torch_us:.1f}us "
          f"ratio={payload['nki_vs_torch_ratio']} correct={payload['correct']} "
          f"-> {out_path}")
    return 0 if payload["correct"] else 1


def write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def receipt(path, stage, exc):
    """API drift or runtime failure -> structured receipt, exit 0 pattern."""
    write(path, {"status": f"{stage}_failed", "reason": repr(exc)[:400],
                 "nki_note": "nki 0.5.0 API attempted per on-box inventory; "
                             "receipt recorded for the report either way",
                 "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"{stage} FAILED (receipt written): {exc}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulate", "device"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        if args.mode == "simulate":
            return run_simulate(args.out, args.seed)
        return run_device(args.out, args.seed)
    except Exception as exc:  # drift-tolerant: the receipt IS a result
        receipt(args.out, args.mode, exc)
        return 0


if __name__ == "__main__":
    sys.exit(main())
