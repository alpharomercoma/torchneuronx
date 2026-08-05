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

# The trn2 list is a copy of trn1's, not a subset: the Trainium1-vs-Trainium2
# comparison is only meaningful if both chips completed the identical lanes, so
# a partial trn2 suite must fail the completeness check like any other gap.
_TRAINING_LANES = ["train/smoke_tinyllama.json", "compile/llama31_train.json",
                   "train/llama31_lora.json", "train/qwen3_lora.json",
                   "train/merge_llama31.json", "cpu/cpu.json"]

EXPECTED = {
    "trn1": list(_TRAINING_LANES),
    "trn2": list(_TRAINING_LANES),
    "inf2": ["serve/llama31_base_short/grid.json",
             "serve/llama31_base_long/grid.json",
             "serve/llama31_dolly_short/grid.json",
             "sustained/sustained.json", "cpu/cpu.json"],
}

# Iteration order for every per-box loop. A box with no results/ directory is
# skipped rather than reported missing, so this file stays runnable while the
# trn2 lane is still in flight.
BOXES = ("trn1", "trn2", "inf2")


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
        if not name.endswith(".json") or name in (
                "grid.json", "boot.json", "load_failure.json",
                "generation_failure.json", "roundtrip_hashes.json"):
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
    for box in BOXES:
        b = cmp.get(box)
        if b is None:
            continue
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


def end_to_end_fields(d):
    """Return (tokens/s, MFU%, source) end-to-end, deriving when not recorded.

    sft_lora.py only began emitting these on 2026-08-05T10:30Z. Every published
    Phase-1/2 result predates that, and so does the trn1 rerun (its box pulled
    at 08:42Z); trn2 launched afterwards and records them natively. Without a
    derivation the report would show the column filled for one generation and
    blank for the other -- the asymmetry that makes a comparison unusable. The
    three inputs exist in every result JSON this project has ever written.
    """
    tps = d.get("tokens_per_s_end_to_end")
    mfu = d.get("mfu_pct_end_to_end")
    recorded = tps is not None
    steps, tok = d.get("steps_recorded"), d.get("tokens_per_optimizer_step")
    wall, fpt = d.get("train_wall_s"), d.get("flops_per_token")
    peak = d.get("peak_bf16_flops")
    if tps is None and steps and tok and wall:
        tps = steps * tok / wall
    if mfu is None and tps and fpt and peak:
        mfu = 100.0 * fpt * tps / peak
    return (round(tps, 1) if tps else None,
            round(mfu, 3) if mfu else None,
            "recorded" if recorded else "derived")


def collect_trn1_reruns():
    """The same-silicon, same-software control runs.

    Not part of EXPECTED: these are a CONTROL, not a lane of the study, and a
    box that never ran them is not incomplete. But leaving them uncollected
    meant no claim about run-to-run variance or version confounding could be
    regenerated from comparison.json, which is the file the report is supposed
    to be reproducible from.

    NOTE ON COMPARABILITY: seed 42 reproduces the published configuration, but
    its mfu_pct is NOT comparable to the published one -- the denominator moved
    from 210e12 to 190e12 on 2026-08-05. Compare tokens_per_s or tflops, which
    are denominator-free. Both denominators are carried here so the ratio can be
    recomputed either way.
    """
    root = os.path.join(ROOT, "trn1", "results", "rerun")
    if not os.path.isdir(root):
        return None
    out = {"note": ("same trn1 box, same software (version_delta identical), "
                    "warm v2 NEFF cache. Compare tokens_per_s / tflops, NOT "
                    "mfu_pct: the peak denominator changed after publication."),
           "lanes": {}, "dropped_no_telemetry": []}
    for name in ("rerun_seed42", "rerun_seed43", "rerun_seed44"):
        d = load_json(os.path.join(root, name + ".json"))
        if d is None:
            continue
        tel = summarize.load_telemetry(os.path.join(root, name + ".telemetry.csv"))
        if tel is None:
            out["dropped_no_telemetry"].append(name)   # invariant 1
            continue
        e2e_tps, e2e_mfu, e2e_src = end_to_end_fields(d)
        out["lanes"][name] = {
            "tokens_per_s": d.get("tokens_per_s"),
            "tflops": d.get("tflops"),
            "median_step_ms": d.get("median_step_ms"),
            "train_wall_s": d.get("train_wall_s"),
            "mfu_pct": d.get("mfu_pct"),
            "peak_bf16_flops": d.get("peak_bf16_flops"),
            "mfu_pct_alt": d.get("mfu_pct_alt"),
            "peak_bf16_flops_alt": d.get("peak_bf16_flops_alt"),
            "tokens_per_s_end_to_end": e2e_tps,
            "mfu_pct_end_to_end": e2e_mfu,
            "end_to_end_source": e2e_src,
            "telemetry": tel,
        }
    version_delta = load_json(os.path.join(root, "version_delta.json"))
    if version_delta is not None:
        out["version_delta"] = version_delta
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="nonzero exit if any row was dropped for telemetry")
    args = ap.parse_args()

    cmp, dropped = {"captured": datetime.now(timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")}, 0
    missing = []
    recorded_failures = []
    for box in BOXES:
        # A box whose results/ has never been populated is not yet part of the
        # study; reporting six "missing" lanes for it would bury real gaps.
        if not os.path.isdir(os.path.join(ROOT, box, "results")):
            continue
        cmp[box], drp = collect_box(box)
        dropped += drp
        for rel in EXPECTED[box]:
            base = os.path.join(ROOT, box, "results")
            # A lane is ACCOUNTED FOR if it produced a result, a per-lane
            # failure receipt, or (for serving sweeps) a directory-level
            # load_failure. Only the per-lane receipt was missing here, so a
            # recorded failure such as compile/llama31_train.failure.json was
            # reported as a MISSING lane -- making `make report` exit nonzero
            # against a suite the report calls complete. A recorded failure is
            # an outcome, not a gap; that is the whole point of receipts.
            candidates = [
                os.path.join(base, rel),
                os.path.join(base, rel[:-5] + ".failure.json"),
                os.path.join(base, os.path.dirname(rel), "load_failure.json"),
            ]
            if not any(os.path.exists(c) for c in candidates):
                missing.append(f"{box}/{rel}")
            elif not os.path.exists(candidates[0]):
                recorded_failures.append(f"{box}/{rel}")
    cmp["dropped_no_telemetry"] = dropped
    cmp["missing_expected"] = missing
    # Distinct from missing: the lane ran and its failure was captured.
    cmp["recorded_failures"] = recorded_failures
    reruns = collect_trn1_reruns()
    if reruns:
        cmp["trn1_reruns"] = reruns

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
