#!/usr/bin/env python3
"""Render the headline tables from analysis/comparison.json in a terminal.

Every number shown here is read from comparison.json -- the same file that
feeds REPORT.md -- so what an audience sees live is what the report says.

    python3 demo/headline.py            # full set
    python3 demo/headline.py --serve    # serving tables only
"""
import argparse
import json
import os
import sys

CMP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "comparison.json")


def fmt(v, suffix="", nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    show_all = not (args.serve or args.train)

    with open(CMP) as fh:
        cmp = json.load(fh)

    if args.train or show_all:
        lanes = cmp["trn1"]["lanes"]
        rows = []
        for key in ("train_llama31_lora", "train_qwen3_lora"):
            lane = lanes.get(key)
            if not lane:
                continue
            r = lane["result"]
            tel = lane.get("telemetry") or {}
            util = (tel.get("gpu_util_pct") or {}).get("mean")
            rows.append([r.get("model", key),
                         fmt(r.get("median_step_ms"), " ms"),
                         fmt(r.get("tokens_per_s")),
                         fmt(r.get("mfu_pct"), "%"),
                         fmt(util, "%")])
        print("\n== Training on trn1.2xlarge (LoRA SFT, TP=2, bf16) ==")
        print(table(rows, ["model", "median step", "tok/s", "MFU", "NC util"]))
        comp = lanes.get("compile_llama31_train")
        if comp:
            c = comp["result"]
            print(f"\nprecompile: {fmt(c.get('wall_s'), ' s', 0)} wall, "
                  f"cache {fmt(c.get('cache_dir_size_mb'), ' MB', 0)}")

    if args.serve or show_all:
        print("\n== Serving on inf2.xlarge (vLLM Neuron, TP=2) ==")
        for run, d in sorted(cmp["inf2"].get("serve", {}).items()):
            boot = d.get("boot") or {}
            print(f"\n{run}  (boot {fmt(boot.get('boot_wall_s'), ' s', 0)}, "
                  f"warm={boot.get('warm', '?')})")
            rows = []
            for tag, row in sorted(d["points"].items()):
                r = row["result"]
                rows.append([tag,
                             fmt(r.get("output_throughput")),
                             fmt(r.get("p50_ttft_ms"), " ms", 0),
                             fmt(r.get("p99_ttft_ms"), " ms", 0),
                             fmt(r.get("p50_tpot_ms"), " ms"),
                             fmt(row["telemetry"]["gpu_util_pct"]["mean"], "%")])
            print(table(rows, ["point", "out tok/s", "ttft p50", "ttft p99",
                               "tpot p50", "NC util"]))
        for run, f in sorted(cmp["inf2"].get("failures", {}).items()):
            print(f"\n{run}: RECORDED FAILURE -- {f.get('reason', '')[:90]}")

    print(f"\n(generated {cmp['captured']}; "
          f"dropped_no_telemetry={cmp['dropped_no_telemetry']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
