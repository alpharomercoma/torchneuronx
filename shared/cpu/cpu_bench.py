#!/usr/bin/env python3
"""Lane 4: host CPU.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS LANE.

This is a HOST comparison, not a chip comparison. Neither AMD nor NVIDIA chose
these CPUs; they are whatever the two cloud providers paired with the GPU:

    MI300X box (DigitalOcean) : Xeon Platinum 8568Y+, 20 cores / 20 threads, 235 GB
    H200 box   (Nebius)       : Xeon Platinum 8468,    8 cores / 16 threads, 196 GB

The MI300X host has 2.5x the physical cores. Any CPU result here is therefore
about the rented instance, and it is reported only because it answers a
question that DOES matter for the GPU comparison: can the host feed the GPU?
A 16-thread host driving a vLLM frontend at concurrency 256 can become the
bottleneck, which would show up as GPU utilisation dipping below 100% in the
telemetry rather than as a GPU deficiency.

Three measurements:
  1. memory bandwidth  -- STREAM-style triad, threaded
  2. tokenizer throughput -- the actual per-request CPU work vLLM does
  3. dense matmul      -- generic FP32 compute, threads swept
"""

import argparse
import json
import multiprocessing
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def bench_stream(size_mib=512, reps=20):
    """STREAM triad: a[i] = b[i] + scalar * c[i]. 3 arrays touched."""
    n = size_mib * 1024 * 1024 // 8
    a = torch.ones(n, dtype=torch.float64)
    b = torch.full((n,), 2.0, dtype=torch.float64)
    c = torch.full((n,), 3.0, dtype=torch.float64)
    for _ in range(3):
        torch.add(b, c, alpha=3.0, out=a)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.add(b, c, alpha=3.0, out=a)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    return (3 * n * 8) / med / 1e9  # GB/s


def bench_matmul(threads, n=4096, reps=10):
    torch.set_num_threads(threads)
    a = torch.randn(n, n, dtype=torch.float32)
    b = torch.randn(n, n, dtype=torch.float32)
    torch.matmul(a, b)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    return (2.0 * n ** 3) / med / 1e12  # TFLOP/s


def bench_tokenizer(model_dir, n_docs=2000):
    """Tokenizer throughput -- real per-request CPU work in a serving path."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return {"error": f"transformers unavailable: {exc}"}
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:
        return {"error": f"tokenizer load failed: {exc}"}

    doc = ("The quick brown fox jumps over the lazy dog. " * 100)
    docs = [doc] * n_docs
    tok(docs[:50])  # warm
    t0 = time.perf_counter()
    out = tok(docs)
    dt = time.perf_counter() - t0
    ntok = sum(len(x) for x in out["input_ids"])
    return {"docs": n_docs, "tokens": ntok, "seconds": round(dt, 3),
            "tokens_per_s": round(ntok / dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-dir", default=None)
    args = ap.parse_args()

    ncpu = multiprocessing.cpu_count()
    thread_sweep = sorted({1, 2, 4, 8, max(1, ncpu // 2), ncpu})

    print(f"  host: {platform.processor() or platform.machine()}  ({ncpu} logical CPUs)")

    torch.set_num_threads(ncpu)
    stream_gbs = bench_stream()
    print(f"  STREAM triad: {stream_gbs:.1f} GB/s ({ncpu} threads)")

    mm = []
    for t in thread_sweep:
        tf = bench_matmul(t)
        mm.append({"threads": t, "tflops": round(tf, 4)})
        print(f"  fp32 matmul  t={t:<3d} {tf:7.3f} TFLOP/s")
    torch.set_num_threads(ncpu)

    tokres = {"skipped": "no --tokenizer-dir"}
    if args.tokenizer_dir:
        tokres = bench_tokenizer(args.tokenizer_dir)
        if "tokens_per_s" in tokres:
            print(f"  tokenizer: {tokres['tokens_per_s']:,.0f} tok/s")
        else:
            print(f"  tokenizer: {tokres}")

    payload = {
        "note": ("HOST comparison, not a chip comparison -- the two cloud "
                 "providers chose these CPUs, not the GPU vendors."),
        "logical_cpus": ncpu,
        "cpu_model": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "stream_triad_gb_s": round(stream_gbs, 1),
        "fp32_matmul": mm,
        "tokenizer": tokres,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
