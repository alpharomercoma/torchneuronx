#!/usr/bin/env python3
"""Roofline placement for every training lane: is this workload compute- or
memory-bound, and does the answer change between Trainium generations?

WHY THIS EXISTS
---------------
MFU alone cannot tell you WHY a number is what it is. A lane at 40% MFU might
be leaving compute on the table, or it might be running flat out against
bandwidth and physically unable to do better. The roofline separates those.

Ridge point = peak FLOP/s / peak bandwidth = the arithmetic intensity (FLOPs
per byte of HBM traffic) above which a kernel is compute-bound.

    trn1 BF16   190e12 / 0.82e12 = 232 FLOP/byte
    trn2 BF16   667e12 / 2.9e12  = 230 FLOP/byte
    trn2 FP8   1299e12 / 2.9e12  = 448 FLOP/byte

The striking result is that the BF16 ridge point barely moved across a
two-year generation gap: AWS scaled compute (3.51x) and bandwidth (3.54x)
together. So Trainium2 does not change WHICH SIDE of the roofline a workload
sits on -- it moves the whole roof up. Anything memory-bound on trn1 is still
memory-bound on trn2, just faster.

FP8 is the exception, and not in the direction people assume: doubling peak
FLOP/s while bandwidth is unchanged nearly DOUBLES the ridge point to 448.
FP8 therefore only pays for workloads already above ~230 FLOP/byte. Below
that, FP8 buys nothing because you were never compute-limited -- you just get
a chip that is idle in a more expensive way.

HONESTY ABOUT THE MEASUREMENT
-----------------------------
Achieved FLOP/s is measured (LoRA-corrected FLOPs/token x tokens/s). Achieved
BANDWIDTH is not directly observable from our telemetry, so arithmetic
intensity is reported as a LOWER BOUND derived from the minimum traffic any
implementation must move: read every parameter once per forward, and for
trainable parameters read/write gradients and optimizer state. Real traffic is
higher (activations, recompute, spills), so real intensity is LOWER than this
bound -- meaning a lane this bound places as memory-bound is definitely
memory-bound, while one it places as compute-bound may not be. Stated as a
bound rather than dressed up as a measurement.
"""
import argparse
import glob
import json
import os

# (peak_flops, bandwidth_bytes_per_s) per device, from the Neuron arch pages.
DEVICES = {
    "trn1": {"bf16": 190e12, "bw": 0.82e12},
    "trn2": {"bf16": 667e12, "bw": 2.9e12, "fp8": 1299e12},
}
BYTES_BF16 = 2
BYTES_OPT_STATE = 12   # fp32 master + 2 Adam moments, per trainable param


def ridge_points():
    out = {}
    for dev, d in DEVICES.items():
        out[f"{dev}_bf16"] = d["bf16"] / d["bw"]
        if "fp8" in d:
            out[f"{dev}_fp8"] = d["fp8"] / d["bw"]
    return out


def min_bytes_per_token(trainable, frozen, gradient_checkpointing,
                        tokens_per_forward):
    """Lower bound on HBM traffic per token.

    CRITICAL: weight traffic is per FORWARD PASS, not per token. A weight is
    read once and reused by every token in the batch, so it amortises over
    tokens_per_forward = seq_len x micro_batch. Charging a full weight read to
    each token (an easy mistake) inflates traffic by ~2000x at seq 2048 and
    flips every verdict from compute-bound to memory-bound.

    Optimizer-state traffic amortises even further -- it is paid once per
    OPTIMIZER step, i.e. over grad_accum forwards -- but it is left at the
    per-forward rate here, which only makes the bound more conservative.
    """
    if not tokens_per_forward:
        return None
    per_forward = (trainable + frozen) * BYTES_BF16
    if gradient_checkpointing:
        per_forward += (trainable + frozen) * BYTES_BF16
    per_forward += trainable * (BYTES_BF16 + BYTES_OPT_STATE)
    return per_forward / float(tokens_per_forward)


def end_to_end(d):
    """Backfill end-to-end throughput/MFU for results that predate the fields.

    sft_lora.py only started emitting tokens_per_s_end_to_end and
    mfu_pct_end_to_end at 2026-08-05T10:30Z. Every published Phase-1/2 result
    predates that, and so does the trn1 rerun currently in flight -- it pulled
    its code at 08:42Z. trn2 launches after the change and WILL have them.

    Without this backfill the report would show an end-to-end column populated
    for one generation and blank for the other, which is exactly the kind of
    asymmetry that makes a comparison unusable. The three inputs needed are all
    present in every result JSON ever written here.
    """
    tps = d.get("tokens_per_s_end_to_end")
    mfu = d.get("mfu_pct_end_to_end")
    steps = d.get("steps_recorded")
    tok = d.get("tokens_per_optimizer_step")
    wall = d.get("train_wall_s")
    fpt = d.get("flops_per_token")
    peak = d.get("peak_bf16_flops")
    if tps is None and steps and tok and wall:
        tps = steps * tok / wall
    if mfu is None and tps and fpt and peak:
        mfu = 100.0 * fpt * tps / peak
    # End-to-end is only interpretable for FULL-LENGTH lanes. On a 20-step
    # probe the cold compile dominates the wall clock, so the "end-to-end MFU"
    # is really a compile-cost measurement wearing a throughput label -- e.g.
    # ctx_4096 reads 82.7% steady but 9.3% end-to-end purely because 20 steps
    # cannot amortise a compile. Emit the compile share so a reader can see
    # when that is happening instead of comparing a probe to a full run.
    compile_s = d.get("compile_s")
    compile_share = (round(compile_s / wall, 3)
                     if compile_s and wall else None)
    return (round(tps, 1) if tps else None,
            round(mfu, 3) if mfu else None,
            d.get("tokens_per_s_end_to_end") is not None,
            compile_share,
            (steps or 0) >= 100)


def analyse(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    cfg = d.get("config") or {}
    # Phase-1/2 results predate device_profile: fall back to the denominator
    # they recorded (210e12 or 190e12 == trn1, 667e12 == trn2), then to trn1.
    dev = cfg.get("device_profile")
    if dev not in DEVICES:
        peak = d.get("peak_bf16_flops")
        dev = "trn2" if peak and peak > 400e12 else "trn1"
    if d.get("tokens_per_s") in (None, 0):
        return None
    fpt = d.get("flops_per_token")
    tr, fr = d.get("params_trainable"), d.get("params_frozen")
    if not fpt or tr is None or fr is None:
        return None
    # Top-level key exists but is null on older lanes; config is authoritative.
    gc = d.get("gradient_checkpointing")
    if gc is None:
        gc = cfg.get("gradient_checkpointing")
    gc = bool(gc)
    tokens_per_forward = (cfg.get("seq_len") or 0) * (cfg.get("micro_batch") or 1)
    bpt = min_bytes_per_token(tr, fr, gc, tokens_per_forward)
    if not bpt:
        return None
    e2e_tps, e2e_mfu, e2e_recorded, compile_share, e2e_meaningful = end_to_end(d)
    intensity = fpt / bpt
    ridge = DEVICES[dev]["bf16"] / DEVICES[dev]["bw"]
    achieved = fpt * d["tokens_per_s"]
    return {
        "tag": d.get("tag") or os.path.basename(path)[:-5],
        "device": dev,
        "model": cfg.get("model") or d.get("model"),
        "seq_len": cfg.get("seq_len"),
        "flops_per_token": fpt,
        "min_bytes_per_token": round(bpt, 1),
        "tokens_per_forward": tokens_per_forward,
        "arithmetic_intensity_bound": round(intensity, 1),
        "ridge_point": round(ridge, 1),
        # Below ridge => memory-bound is CERTAIN (real intensity is lower still).
        "verdict": "memory-bound" if intensity < ridge else "compute-bound (bound only)",
        "achieved_tflops": round(achieved / 1e12, 2),
        "mfu_pct": d.get("mfu_pct"),
        "mfu_pct_alt": d.get("mfu_pct_alt"),
        "tokens_per_s_end_to_end": e2e_tps,
        "mfu_pct_end_to_end": e2e_mfu,
        "end_to_end_source": "recorded" if e2e_recorded else "derived",
        "compile_share_of_wall": compile_share,
        # False on short probes, where compile dominates the wall clock.
        "end_to_end_comparable": e2e_meaningful,
        "peak_bf16_flops": d.get("peak_bf16_flops"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*",
                    default=["trn1/results", "trn2/results"])
    ap.add_argument("--out", default="analysis/roofline.json")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        # Archive directories must be EXCLUDED. `results/` accumulates
        # invalidated/, deferred/, superseded/ and original_chip/ subtrees, and
        # a recursive glob happily reads all of them -- so an analysis would mix
        # a lane's live result with the very artifact that was invalidated for
        # being wrong. This was caught with residency_pair_b, whose invalidated
        # copy (an experiment that never actually ran) was being scored here.
        for path in sorted(glob.glob(os.path.join(root, "**", "*.json"),
                                     recursive=True)):
            if any(x in path for x in ("/invalidated/", "/deferred/",
                                      "/superseded/", "/original_chip/")):
                continue
            r = analyse(path)
            if r:
                rows.append(r)

    report = {
        "ridge_points_flop_per_byte": {k: round(v, 1)
                                       for k, v in ridge_points().items()},
        "note": ("Arithmetic intensity is a LOWER BOUND on traffic, so it is an "
                 "UPPER bound on intensity: a lane placed memory-bound is "
                 "certainly memory-bound; one placed compute-bound may not be. "
                 "The BF16 ridge point barely moves between generations (232 -> "
                 "230) because AWS scaled compute and bandwidth together; FP8 "
                 "nearly doubles it to 448, so FP8 only pays above ~230."),
        "lanes": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report["ridge_points_flop_per_byte"], indent=2))
    for r in rows:
        print(f"{r['device']:5} {r['tag']:24} "
              f"AI<={r['arithmetic_intensity_bound']:8.1f}  "
              f"ridge={r['ridge_point']:6.1f}  {r['verdict']}")
    print(f"\n{len(rows)} lanes -> {args.out}")


if __name__ == "__main__":
    main()
