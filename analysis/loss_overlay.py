#!/usr/bin/env python3
"""Overlay the training loss trajectories of two boxes on the same lane.

WHY THIS EXISTS
---------------
Every headline number in this study measures SPEED. The quality gate answers
"did the model learn?" with held-out loss at two points, before and after. This
answers a different and cheaper question: **did the two chips take the SAME
PATH?**

That matters because the trn1/trn2 final-loss gap (1.2063 vs 1.1489) is not
seed noise -- three trn1 seeds produced BIT-IDENTICAL loss, so this stack is
deterministic and packing pins the data order. Something structural differs.
The leading hypothesis is TP=2 vs TP=4: a different tensor-parallel width
changes the order in which partial sums are reduced, and floating-point
addition is not associative. If that is the cause, the curves should be
indistinguishable early and drift apart slowly as tiny per-step differences
accumulate.

The alternative -- that one chip is training a materially different model --
would look completely different: an early split, or a persistent offset from
the first steps.

The shape of the divergence therefore distinguishes the two, and it costs
nothing: `loss_trace` is already recorded in every result JSON (645 entries per
lane here). No box time, no compile, no new run.

WHAT IT DOES NOT SHOW
---------------------
Two curves overlapping does not prove the models are equally good; loss on the
training distribution is not quality. That claim belongs to the quality gate
and its held-out split. This is a diagnostic of numerical agreement, nothing
more.

USAGE
    python3 analysis/loss_overlay.py                       # defaults to the primary lane
    python3 analysis/loss_overlay.py --lane qwen3_lora
    python3 analysis/loss_overlay.py --a trn1 --b trn2 --lane llama31_lora
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_trace(box: str, lane: str) -> tuple[list[dict], dict]:
    p = ROOT / box / "results" / "train" / f"{lane}.json"
    if not p.is_file():
        sys.exit(f"missing {p}")
    d = json.loads(p.read_text())
    trace = d.get("loss_trace") or []
    if not trace:
        sys.exit(f"{p} has no loss_trace")
    return trace, d


def align(a: list[dict], b: list[dict]) -> list[tuple[int, float, float]]:
    """Pair the two traces on step number.

    We align on `step`, never on list position. A lane that logged a different
    number of points -- different logging_steps, a resumed run -- would silently
    compare step 12 against step 30 if we zipped positionally, and the resulting
    "divergence" would be an artifact of the join.
    """
    ma = {int(r["step"]): float(r["loss"]) for r in a if r.get("loss") is not None}
    mb = {int(r["step"]): float(r["loss"]) for r in b if r.get("loss") is not None}
    return [(s, ma[s], mb[s]) for s in sorted(set(ma) & set(mb))]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def first_exceed(pairs, thresh: float) -> int | None:
    """First step where |a-b| exceeds thresh AND stays exceeded.

    A single spike crossing the threshold is noise. Requiring the excursion to
    persist to the end of the run is what makes this a "divergence point"
    rather than a "loudest step".
    """
    for i, (s, x, y) in enumerate(pairs):
        if abs(x - y) > thresh and all(abs(p - q) > thresh for _, p, q in pairs[i:]):
            return s
    return None


def sparkline(vals: list[float], width: int = 60) -> str:
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(vals) // width)
    sampled = vals[::step][:width]
    lo, hi = min(sampled), max(sampled)
    if hi - lo < 1e-12:
        return blocks[0] * len(sampled)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in sampled)


def window_stats(pairs, lo: int, hi: int) -> dict:
    w = [(s, x, y) for s, x, y in pairs if lo <= s <= hi]
    if not w:
        return {}
    return {
        "steps": [w[0][0], w[-1][0]],
        "n": len(w),
        "mean_a": round(sum(x for _, x, _ in w) / len(w), 6),
        "mean_b": round(sum(y for _, _, y in w) / len(w), 6),
        "mean_abs_delta": round(sum(abs(x - y) for _, x, y in w) / len(w), 6),
        "max_abs_delta": round(max(abs(x - y) for _, x, y in w), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="trn1")
    ap.add_argument("--b", default="trn2")
    ap.add_argument("--lane", default="llama31_lora")
    ap.add_argument("--out", default=None, help="write JSON here (default analysis/loss_overlay_<lane>.json)")
    args = ap.parse_args()

    ta, da = load_trace(args.a, args.lane)
    tb, db = load_trace(args.b, args.lane)
    pairs = align(ta, tb)
    if not pairs:
        sys.exit("no overlapping steps between the two traces")

    xs = [x for _, x, _ in pairs]
    ys = [y for _, _, y in pairs]
    deltas = [abs(x - y) for x, y in zip(xs, ys)]
    last = pairs[-1][0]

    # The warmup flag marks steps whose ms is dominated by compile, not compute.
    # It does not affect loss, but excluding them from the "early" window keeps
    # the comparison to steady-state optimisation.
    warm_a = {int(r["step"]) for r in ta if r.get("warmup")}
    warm_b = {int(r["step"]) for r in tb if r.get("warmup")}
    warm = sorted(warm_a | warm_b)

    report = {
        "lane": args.lane,
        "boxes": [args.a, args.b],
        "n_steps_compared": len(pairs),
        "tp": {
            args.a: (da.get("config") or {}).get("tensor_parallel_size"),
            args.b: (db.get("config") or {}).get("tensor_parallel_size"),
        },
        "final_loss": {args.a: da.get("final_loss"), args.b: db.get("final_loss")},
        "warmup_steps_excluded_from_early_window": warm,
        "delta": {
            "mean_abs": round(sum(deltas) / len(deltas), 6),
            "max_abs": round(max(deltas), 6),
            "max_abs_at_step": pairs[deltas.index(max(deltas))][0],
            "final": round(xs[-1] - ys[-1], 6),
        },
        "pearson_r": round(pearson(xs, ys) or float("nan"), 6),
        "sustained_divergence_step": {
            "0.01": first_exceed(pairs, 0.01),
            "0.05": first_exceed(pairs, 0.05),
        },
        "windows": {
            "first_10pct": window_stats(pairs, max(warm) + 1 if warm else 1, max(1, last // 10)),
            "middle": window_stats(pairs, last * 4 // 10, last * 6 // 10),
            "tail_50": window_stats(pairs, last - 49, last),
        },
        "sparkline": {args.a: sparkline(xs), args.b: sparkline(ys)},
    }

    out = pathlib.Path(args.out) if args.out else ROOT / "analysis" / f"loss_overlay_{args.lane}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    d = report["delta"]
    w = report["windows"]
    print(f"lane {args.lane}: {args.a} (TP={report['tp'][args.a]}) vs {args.b} (TP={report['tp'][args.b]})")
    print(f"  steps compared      {report['n_steps_compared']}")
    print(f"  final loss          {report['final_loss'][args.a]} vs {report['final_loss'][args.b]}"
          f"  (delta {d['final']:+.4f})")
    print(f"  mean |delta|        {d['mean_abs']:.4f}    max {d['max_abs']:.4f} @ step {d['max_abs_at_step']}")
    print(f"  pearson r           {report['pearson_r']:.6f}")
    print(f"  sustained >0.01 at  step {report['sustained_divergence_step']['0.01']}")
    print(f"  sustained >0.05 at  step {report['sustained_divergence_step']['0.05']}")
    for name in ("first_10pct", "middle", "tail_50"):
        s = w.get(name) or {}
        if s:
            print(f"  {name:<12} steps {s['steps'][0]}-{s['steps'][1]}: "
                  f"mean {s['mean_a']:.4f} vs {s['mean_b']:.4f}, mean|d| {s['mean_abs_delta']:.4f}")
    print(f"  {args.a} {report['sparkline'][args.a]}")
    print(f"  {args.b} {report['sparkline'][args.b]}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
