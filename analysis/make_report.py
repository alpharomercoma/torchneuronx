#!/usr/bin/env python3
"""Aggregate raw result triplets -> analysis/comparison.json (+ .txt render).

Every number quoted in REPORT.md is regenerated from comparison.json by this
script; nothing in the report is hand-computed. The two enforced invariants
are inherited from the GPU study:

  1. A benchmark JSON without its telemetry CSV is dropped and counted in
     dropped_no_telemetry -- it never contributes a number.
  2. This repo's lanes are one-sided by design (trn1 trains, inf2 serves), so
     everything is emitted under per-box keys; there is no head-to-head math
     for a renderer to invent.

Exit status is nonzero if any expected lane is missing entirely, so CI or a
human notices an incomplete suite before quoting it.

    python3 analysis/make_report.py            # writes analysis/comparison.{json,txt}
    python3 analysis/make_report.py --strict   # also fail on dropped rows
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
import summarize  # noqa: E402  (load_telemetry + busy-window rule)

EXPECTED = {
    "trn1": ["train/smoke_tinyllama.json", "compile/llama31_train.json",
             "train/llama31_lora.json", "train/qwen3_lora.json",
             "train/merge_llama31.json", "cpu/cpu.json"],
    "inf2": ["serve/llama31_base_short/grid.json",
             "serve/llama31_base_long/grid.json",
             "serve/llama31_dolly_short/grid.json",
             "sustained/sustained.json", "cpu/cpu.json"],
}


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def collect_serve_dir(full):
    """One sweep directory -> {tag: {result, telemetry}}, dropped count."""
    rows, dropped = {}, 0
    for name in sorted(os.listdir(full)):
        if not name.endswith(".json") or name in ("grid.json", "boot.json",
                                                  "load_failure.json"):
            continue
        tag = name[:-5]
        result = load_json(os.path.join(full, name))
        tel = summarize.load_telemetry(
            os.path.join(full, f"{tag}.telemetry.csv"))
        if result is None:
            continue
        if tel is None:
            dropped += 1          # invariant 1: no telemetry, no number
            continue
        rows[tag] = {"result": result, "telemetry": tel}
    return rows, dropped


def collect_box(box):
    res_root = os.path.join(ROOT, box, "results")
    out = {"lanes": {}, "serve": {}, "failures": {}}
    dropped = 0

    for rel in ("cpu/cpu.json", "sustained/sustained.json",
                "compile/llama31_train.json", "train/smoke_tinyllama.json",
                "train/llama31_lora.json", "train/qwen3_lora.json",
                "train/merge_llama31.json"):
        d = load_json(os.path.join(res_root, rel))
        if d is None:
            continue
        key = rel.replace("/", "_")[:-5]
        tel = summarize.load_telemetry(
            os.path.join(res_root, rel[:-5] + ".telemetry.csv"))
        # cpu/merge/compile lanes have no accelerator telemetry by design --
        # the triplet rule applies to lanes that CLAIM accelerator work.
        needs_tel = rel.startswith("train/") and "merge" not in rel
        if needs_tel and tel is None:
            dropped += 1
            continue
        out["lanes"][key] = {"result": d, "telemetry": tel}

    serve_root = os.path.join(res_root, "serve")
    if os.path.isdir(serve_root):
        for run_dir in sorted(os.listdir(serve_root)):
            full = os.path.join(serve_root, run_dir)
            if not os.path.isdir(full):
                continue
            # Failures are results -- but they don't suppress a successful
            # sweep sitting in the same directory (sync-without-delete can
            # resurrect a stale failure record next to fresh points, and a
            # sweep can succeed mechanically while generation fails, which
            # generation_failure.json records).
            for fname, kind in (("load_failure.json", "load"),
                                ("generation_failure.json", "generation")):
                fail = load_json(os.path.join(full, fname))
                if fail:
                    out["failures"][f"{run_dir}:{kind}"] = fail
            grid = load_json(os.path.join(full, "grid.json"))
            rows, drp = collect_serve_dir(full)
            dropped += drp
            if grid or rows:
                out["serve"][run_dir] = {
                    "grid": grid,
                    "boot": load_json(os.path.join(full, "boot.json")),
                    "points": rows,
                }

    quality_root = os.path.join(res_root, "quality")
    if os.path.isdir(quality_root):
        out["quality"] = {f[:-5]: load_json(os.path.join(quality_root, f))
                          for f in sorted(os.listdir(quality_root))
                          if f.endswith(".json")}
    return out, dropped


def render(cmp):
    lines = ["neuron-pipelines comparison  (generated %s)" % cmp["captured"],
             "=" * 64]
    for box in ("trn1", "inf2"):
        b = cmp[box]
        lines.append(f"\n[{box}] lanes: {sorted(b['lanes'])}")
        for run, d in sorted(b.get("serve", {}).items()):
            n = len(d["points"])
            boot = d.get("boot") or {}
            lines.append(f"  serve/{run}: {n} points, boot {boot.get('boot_wall_s', '?')}s"
                         f" (warm={boot.get('warm', '?')})")
            for tag, row in sorted(d["points"].items()):
                r = row["result"]
                lines.append(
                    f"    {tag}: out_tok/s={r.get('output_throughput')}"
                    f" ttft_p50={r.get('p50_ttft_ms')}ms"
                    f" tpot_p50={r.get('p50_tpot_ms')}ms"
                    f" util={row['telemetry']['gpu_util_pct']['mean']}%")
        for run, f in sorted(b.get("failures", {}).items()):
            lines.append(f"  serve/{run}: RECORDED FAILURE ({f.get('status')})")
    lines.append(f"\ndropped_no_telemetry: {cmp['dropped_no_telemetry']}")
    lines.append(f"missing_expected: {cmp['missing_expected']}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="nonzero exit if any row was dropped for telemetry")
    args = ap.parse_args()

    cmp, dropped = {"captured": datetime.now(timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")}, 0
    missing = []
    for box in ("trn1", "inf2"):
        cmp[box], drp = collect_box(box)
        dropped += drp
        for rel in EXPECTED[box]:
            if not os.path.exists(os.path.join(ROOT, box, "results", rel)) \
               and not os.path.exists(os.path.join(
                   ROOT, box, "results", os.path.dirname(rel),
                   "load_failure.json")):
                missing.append(f"{box}/{rel}")
    cmp["dropped_no_telemetry"] = dropped
    cmp["missing_expected"] = missing

    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, "comparison.json"), "w") as fh:
        json.dump(cmp, fh, indent=1, sort_keys=True)
    txt = render(cmp)
    with open(os.path.join(HERE, "comparison.txt"), "w") as fh:
        fh.write(txt)
    print(txt)

    if missing:
        print(f"INCOMPLETE SUITE: {len(missing)} expected lane(s) missing",
              file=sys.stderr)
        return 1
    if args.strict and dropped:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
