#!/usr/bin/env python3
"""Streaming serving benchmark client. Stdlib only, on purpose.

This is the PRIMARY bench client on the inference box: inf2.xlarge has
4 vCPU / 16 GiB host RAM, and installing a second full vLLM just to use
`vllm bench serve` as a client would cost more host memory than the server
can spare. This client emits a result JSON whose keys are a superset of what
`vllm bench serve --save-result` writes and exactly what summarize.py's
parse_serve reads, so downstream tooling cannot tell which client produced a
row -- that property is enforced by tests/test_fallback_client.py.

Measurement notes (also in METHODOLOGY "Known limits"):
  * input length is controlled by building a prompt of N common short words
    (~1 token/word for Llama-family tokenizers) -- reported input tokens are
    CONFIGURED, not measured;
  * output tokens are counted as received stream chunks, with max_tokens +
    ignore_eos pinning the true count to --output-len;
  * TPOT = (E2EL - TTFT) / (tokens - 1), the same definition vllm bench uses.

Usage:
    fallback_client.py --model <served-name> --num-prompts 32 \
        --max-concurrency 8 --input-len 1024 --output-len 1024 \
        --out isl1024_osl1024_c8.json          # summary line on stdout
"""
import argparse
import http.client
import json
import math
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

WORDS = ("time year people way day man thing woman life child world school "
         "state family student group country problem hand part place case "
         "week company system program question work government number night "
         "point home water room mother area money story fact month lot right "
         "study book eye job word business issue side kind head house service "
         "friend father power hour game line end member law car city name").split()

PCTS = (50, 90, 99)


def build_prompt(input_len, seed):
    # Overprovision by 32 words: the request pins the exact prompt length via
    # truncate_prompt_tokens, so the text only has to be LONGER than the
    # target. Without truncation the first sweep failed on every request with
    # HTTP 400 "1025 input tokens" -- 1024 words tokenized to 1024 + BOS
    # (measured 2026-07-30). One token over the window, all points dead.
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(max(1, input_len) + 32))


def draw_poisson_arrivals(rate_per_s, duration_s, seed, cap=None):
    """Arrival timestamps (s from start) for a Poisson process. Pure.

    Open-loop load (MLPerf Server-scenario style): arrivals are drawn from
    exponential inter-arrival gaps and do NOT wait for completions, so the
    server sees offered load rather than the closed-loop feedback of a
    fixed-concurrency pool. Deterministic per seed for reproducibility.
    """
    rng = random.Random(seed)
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate_per_s)
        if t >= duration_s or (cap is not None and len(out) >= cap):
            break
        out.append(t)
    return out


def compute_goodput(results, duration_s, slo_ttft_ms, slo_tpot_ms):
    """Goodput = output tokens/s from requests meeting BOTH SLOs. Pure.

    The SLO thresholds are declared in the result JSON, never implied --
    goodput without its SLO definition is a number with no meaning.
    """
    good = [r for r in results
            if r["ttft"] <= slo_ttft_ms
            and (r["tpot"] is None or r["tpot"] <= slo_tpot_ms)]
    total = len(results)
    return {
        "slo_ttft_ms": slo_ttft_ms,
        "slo_tpot_ms": slo_tpot_ms,
        "slo_attainment_pct": round(100.0 * len(good) / total, 2) if total else None,
        "goodput_output_tokens_per_s":
            round(sum(r["ntok"] for r in good) / duration_s, 2) if duration_s else 0,
    }


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def dist(name, vals):
    """-> {mean,median,std,p50,p90,p99}_<name>_ms keys."""
    if not vals:
        return {f"{s}_{name}_ms": None
                for s in ("mean", "median", "std", "p50", "p90", "p99")}
    sv = sorted(vals)
    out = {
        f"mean_{name}_ms": round(statistics.fmean(vals), 3),
        f"median_{name}_ms": round(statistics.median(vals), 3),
        f"std_{name}_ms": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
    }
    for p in PCTS:
        out[f"p{p}_{name}_ms"] = round(pct(sv, p), 3)
    return out


class Bench:
    def __init__(self, args):
        self.args = args
        u = urlparse(args.base_url)
        self.host, self.port = u.hostname, u.port or 80
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.second_buckets = {}
        self.results = []
        self.failed = 0
        self.t0 = None

    def _track(self, delta):
        with self.lock:
            self.active += delta
            self.max_active = max(self.max_active, self.active)

    def _bucket(self, stamp, n=1):
        b = int(stamp - self.t0)
        with self.lock:
            self.second_buckets[b] = self.second_buckets.get(b, 0) + n

    def one_request(self, idx):
        body = json.dumps({
            "model": self.args.model,
            "prompt": build_prompt(self.args.input_len, self.args.seed + idx),
            "max_tokens": self.args.output_len,
            "temperature": 0.0,
            "seed": self.args.seed,
            "ignore_eos": True,
            # vLLM extension: server-side truncation to EXACTLY input_len
            # tokens, BOS included -- makes the reported ISL exact rather
            # than approximate, and keeps prompt+max_tokens inside
            # max_model_len by construction.
            "truncate_prompt_tokens": self.args.input_len,
            "stream": True,
        })
        self._track(+1)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        try:
            start = time.perf_counter()
            conn.request("POST", "/v1/completions", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {resp.read(200)!r}")
            ttft = None
            stamps = []
            buf = b""
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        buf = b""
                        break
                    try:
                        text = json.loads(payload)["choices"][0].get("text")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if not text:
                        continue
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now
                    stamps.append(now)
                    self._bucket(now)
            end = time.perf_counter()
            if ttft is None:
                raise RuntimeError("stream produced no tokens")
            ntok = len(stamps)
            e2el_ms = (end - start) * 1000
            ttft_ms = (ttft - start) * 1000
            itls = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
            tpot_ms = ((e2el_ms - ttft_ms) / (ntok - 1)) if ntok > 1 else None
            with self.lock:
                self.results.append(
                    {"ttft": ttft_ms, "e2el": e2el_ms, "tpot": tpot_ms,
                     "itls": itls, "ntok": ntok})
        except Exception as exc:  # a failed request is a counted result
            with self.lock:
                self.failed += 1
            print(f"  request {idx} FAILED: {exc}", file=sys.stderr)
        finally:
            self._track(-1)
            conn.close()

    def run(self):
        a = self.args
        mode = getattr(a, "arrival_mode", "closed")
        print(f"# fallback_client  model={a.model}  mode={mode}  "
              f"prompts={a.num_prompts}  conc={a.max_concurrency}  "
              f"isl={a.input_len}  osl={a.output_len}  seed={a.seed}"
              + (f"  rate={a.request_rate}/s window={a.duration_s}s"
                 if mode == "poisson" else ""))
        self.t0 = time.perf_counter()
        if mode == "poisson":
            # Open loop: a wide pool so admission is never gated by
            # completions; the schedule, not the pool, sets concurrency.
            arrivals = draw_poisson_arrivals(
                a.request_rate, getattr(a, "duration_s", 60.0), a.seed,
                cap=a.num_prompts)
            with ThreadPoolExecutor(max_workers=max(64, a.max_concurrency)) as pool:
                futures = []
                for i, t_arr in enumerate(arrivals):
                    delay = t_arr - (time.perf_counter() - self.t0)
                    if delay > 0:
                        time.sleep(delay)
                    futures.append(pool.submit(self.one_request, i))
                for f in futures:
                    f.result()
            self.offered = len(arrivals)
        else:
            with ThreadPoolExecutor(max_workers=a.max_concurrency) as pool:
                list(pool.map(self.one_request, range(a.num_prompts)))
            self.offered = a.num_prompts
        duration = time.perf_counter() - self.t0

        ok = self.results
        total_out = sum(r["ntok"] for r in ok)
        total_in = a.input_len * len(ok)      # configured, not measured
        itl_all = [x for r in ok for x in r["itls"]]
        out = {
            "date": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
            "backend": "fallback_client",
            "model_id": a.model,
            "num_prompts": a.num_prompts,
            "max_concurrency": a.max_concurrency,
            "input_len_configured": a.input_len,
            "output_len_configured": a.output_len,
            "duration": round(duration, 3),
            "completed": len(ok),
            "failed": self.failed,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "output_throughput": round(total_out / duration, 2) if duration else 0,
            "total_token_throughput":
                round((total_in + total_out) / duration, 2) if duration else 0,
            "max_output_tokens_per_s":
                max(self.second_buckets.values()) if self.second_buckets else 0,
            "max_concurrent_requests": self.max_active,
            "arrival_mode": mode,
            "request_rate_offered":
                (getattr(self.args, "request_rate", None)
                 if mode == "poisson" else None),
            "requests_offered": self.offered,
            "completed_rate_per_s":
                round(len(ok) / duration, 3) if duration else 0,
        }
        out.update(compute_goodput(
            ok, duration,
            getattr(self.args, "slo_ttft_ms", 3000.0),
            getattr(self.args, "slo_tpot_ms", 80.0)))
        out.update(dist("ttft", [r["ttft"] for r in ok]))
        out.update(dist("tpot", [r["tpot"] for r in ok if r["tpot"] is not None]))
        out.update(dist("itl", itl_all))
        out.update(dist("e2el", [r["e2el"] for r in ok]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    # Open-loop (MLPerf Server-style) mode + SLO goodput. Defaults keep the
    # closed-loop behavior of every existing lane byte-identical.
    ap.add_argument("--arrival-mode", choices=["closed", "poisson"],
                    default="closed")
    ap.add_argument("--request-rate", type=float, default=None,
                    help="poisson: offered req/s")
    ap.add_argument("--duration-s", type=float, default=60.0,
                    help="poisson: admission window")
    ap.add_argument("--slo-ttft-ms", type=float, default=3000.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=80.0)
    args = ap.parse_args()
    if args.arrival_mode == "poisson" and not args.request_rate:
        ap.error("--arrival-mode poisson requires --request-rate")

    result = Bench(args).run()
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    import os
    os.replace(tmp, args.out)

    print(f"out_tok/s={result['output_throughput']}  "
          f"ttft_p50={result['p50_ttft_ms']}ms  p99={result['p99_ttft_ms']}ms  "
          f"tpot_p50={result['p50_tpot_ms']}ms  "
          f"completed={result['completed']}/{result['num_prompts']}  "
          f"duration={result['duration']}s -> {args.out}")
    return 1 if (result["completed"] == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
