#!/usr/bin/env python3
"""Summarise the Phase-4 pretraining and post-training lanes as markdown.

    python3 analysis/phase4_summary.py [--results trn1/results/phase4]

Reads the lane JSONs and receipts and emits the tables that go into
REPORT-EXTENSIONS. Generated rather than hand-transcribed, because a number
retyped into prose is a number that can silently disagree with the file it came
from -- and this study's whole claim is that every figure is traceable.

Receipts are first-class here. A lane that hit a wall is reported in the same
table as one that produced a throughput figure, because "this does not work on
this hardware, and here is the exact error" is a result a practitioner needs
just as much as a tok/s number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(d: Path):
    """Return (successes, receipts), each as [(name, dict)] sorted by name."""
    ok, bad = [], []
    for p in sorted(d.glob("*.json")):
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        name = p.name[: -len(".failure.json")] if p.name.endswith(".failure.json") \
            else p.stem
        (bad if p.name.endswith(".failure.json") or blob.get("status") == "failed"
         else ok).append((name, blob))
    return ok, bad


def fmt(v, nd=1, dash="--"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def throughput_table(ok):
    if not ok:
        return "_No lane produced a throughput figure._\n"
    head = ("| lane | stage | model | tok/s (steady) | tok/s (e2e) | TFLOP/s | "
            "MFU % | MFU % alt | median step ms | compile s |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for name, d in ok:
        rows.append(
            f"| `{name}` | {d.get('stage','--')} | {d.get('model','--')} | "
            f"{fmt(d.get('tokens_per_s'))} | {fmt(d.get('tokens_per_s_end_to_end'))} | "
            f"{fmt(d.get('tflops'),2)} | {fmt(d.get('mfu_pct'),2)} | "
            f"{fmt(d.get('mfu_pct_alt'),2)} | {fmt(d.get('median_step_ms'))} | "
            f"{fmt(d.get('compile_s'))} |")
    return head + "\n".join(rows) + "\n"


def receipt_table(bad):
    if not bad:
        return "_No walls recorded._\n"
    head = "| lane | stage | wall |\n|---|---|---|\n"
    rows = []
    for name, d in bad:
        reason = (d.get("reason") or "").replace("\n", " ").strip()
        if len(reason) > 180:
            reason = reason[:177] + "..."
        rows.append(f"| `{name}` | {d.get('stage','--')} | {reason} |")
    return head + "\n".join(rows) + "\n"


def loss_summary(ok):
    """First and last recorded loss -- did the lane actually learn anything?

    A throughput number from a run whose loss never moved is a measurement of
    arithmetic, not of training.
    """
    lines = []
    for name, d in ok:
        trace = d.get("loss_trace") or []
        losses = [r["loss"] for r in trace if r.get("loss") is not None]
        if len(losses) >= 2:
            # last - first, so the sign reads the way a reader expects: negative
            # means the loss went DOWN. The first version of this table computed
            # first - last and labelled it "delta", which inverted the sign and
            # showed a lane whose loss ROSE as "-0.4844" -- a number a reader
            # would take as descent, in a table whose entire job is to say
            # whether descent happened.
            delta = losses[-1] - losses[0]
            lines.append(f"| `{name}` | {losses[0]:.4f} | {losses[-1]:.4f} | "
                         f"{delta:+.4f} | {'yes' if delta < 0 else 'NO'} | "
                         f"{len(losses)} |")
    if not lines:
        return "_No loss traces recorded._\n"
    return ("| lane | first loss | last loss | last - first | descended? | steps |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n"
            + "\n_A negative `last - first` means the loss fell. A lane marked "
              "`NO` produced a valid THROUGHPUT measurement and nothing more: "
              "its optimisation did not visibly progress over the recorded "
              "steps, so it is not evidence that the model learned anything._\n")


def ceiling_series(bad):
    """The graph-size / HBM ceiling series, if those receipts are present.

    These were measured across shapes rather than asserted once, so they locate
    the limit instead of just warning about it.
    """
    rows = []
    for name, d in bad:
        ev = d.get("evidence") or {}
        cfg = ev.get("config") or d.get("config") or {}
        reason = (d.get("reason") or "") + " " + (d.get("detail") or "")
        code = next((c for c in ("NCC_EVRF007", "NCC_EXTP004", "NCC_EOOM001")
                     if c in reason), None)
        if not code and "recompile" not in reason:
            continue
        rows.append(
            f"| `{name}` | {cfg.get('seq_len','--')} | {cfg.get('micro_batch','--')} | "
            f"{cfg.get('grad_accum','--')} | {code or 'per-step recompile'} |")
    if not rows:
        return ""
    return ("\n### Measured ceilings\n\n"
            "| receipt | seq | micro-batch | grad-accum | limit hit |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="trn1/results/phase4")
    args = ap.parse_args()
    d = Path(args.results)
    if not d.is_dir():
        raise SystemExit(f"no such results directory: {d}")

    ok, bad = load(d)
    print(f"# Phase 4 — pretraining and post-training on trn1\n")
    print(f"_{len(ok)} lane(s) produced numbers; {len(bad)} recorded a wall._\n")
    print("## Throughput\n")
    print(throughput_table(ok))
    print("\n## Did the loss move?\n")
    print(loss_summary(ok))
    print("\n## Walls\n")
    print(receipt_table(bad))
    print(ceiling_series(bad))


if __name__ == "__main__":
    main()
