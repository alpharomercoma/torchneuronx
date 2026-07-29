#!/usr/bin/env python3
"""Turn the sustained-loop iteration files into a retention curve.

Kept deliberately separate from the loop that produces them: an earlier repo in
this series lost an entire soak run because the driver parsed inline and threw
the raw numbers away. The loop writes files; this reads them afterwards.
"""

import argparse
import glob
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results/sustained directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "iter_*.json")))
    iters = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if "output_throughput" not in d:
            continue
        iters.append({
            "file": os.path.basename(f),
            "output_tok_s": round(d["output_throughput"], 1),
            "ttft_p99_ms": round(d.get("p99_ttft_ms", 0), 2),
            "tpot_p50_ms": round(d.get("p50_tpot_ms", 0), 3),
            "duration_s": round(d.get("duration", 0), 2),
        })

    if not iters:
        print("no iterations found")
        return

    first = iters[0]["output_tok_s"]
    last = iters[-1]["output_tok_s"]
    peak = max(x["output_tok_s"] for x in iters)
    tail = [x["output_tok_s"] for x in iters[-3:]]
    tail_mean = statistics.fmean(tail)

    summary = {
        "iterations": len(iters),
        "first_tok_s": first,
        "peak_tok_s": peak,
        "last_tok_s": last,
        "tail3_mean_tok_s": round(tail_mean, 1),
        "retention_vs_first_pct": round(100.0 * tail_mean / first, 1),
        "retention_vs_peak_pct": round(100.0 * tail_mean / peak, 1),
        "per_iteration": iters,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"iterations         : {summary['iterations']}")
    print(f"first / peak / last: {first} / {peak} / {last} tok/s")
    print(f"retention vs peak  : {summary['retention_vs_peak_pct']}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
