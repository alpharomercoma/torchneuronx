#!/usr/bin/env python3
"""Summarise a speculative-decoding k-sweep into a throughput-vs-k table.

Reads either the structured --benchmark-report-path JSON or, as a fallback, the
three metric blocks NxDI's benchmark_sampling prints into the run log:

    e2e_model               end-to-end (the honest headline)
    context_encoding_model  prefill
    token_generation_model  decode

WHY e2e IS THE HEADLINE
-----------------------
Under speculation, token_generation_model measures TARGET-MODEL FORWARDS, not
emitted tokens. A larger k means fewer, fatter target forwards, so that number
can move for reasons unrelated to user-visible speed. e2e is the only block
whose denominator is wall-clock for the whole request, so it is the one that
answers "is this actually faster".

Speedup is reported against the no-speculation baseline measured on the SAME
box and the SAME NxDI build. Cross-box comparison is refused: the inf2 lane ran
vllm_0_16 and trn1 runs pytorch_2_9_nxd_inference.
"""
import argparse, json, re, sys
from pathlib import Path

BLOCKS = ("e2e_model", "context_encoding_model", "token_generation_model")


def parse_log(text):
    """Pull the three blocks out of a benchmark log."""
    out = {}
    for blk in BLOCKS:
        m = re.search(re.escape(f'"{blk}"') + r'.*?"latency_ms_p50":\s*([0-9.]+).*?'
                      r'"latency_ms_avg":\s*([0-9.]+).*?"throughput":\s*([0-9.]+)',
                      text, re.S)
        if m:
            out[blk] = {"latency_ms_p50": float(m.group(1)),
                        "latency_ms_avg": float(m.group(2)),
                        "throughput": float(m.group(3))}
    return out


def load_case(d: Path, tag: str):
    rep = d / f"{tag}.report.json"
    if rep.is_file():
        try:
            raw = json.loads(rep.read_text())
            if any(b in raw for b in BLOCKS):
                return {b: raw[b] for b in BLOCKS if b in raw}
        except json.JSONDecodeError:
            pass
    log = d / f"{tag}.log"
    if log.is_file():
        return parse_log(log.read_text(errors="replace"))
    return {}


def k_of(tag):
    m = re.match(r"spec_k(\d+)$", tag)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--json-out")
    ap.add_argument("--gen-tokens", type=int, default=256,
                    help="emitted tokens per run = seq_len - max_context_length")
    args = ap.parse_args()
    d = Path(args.results_dir)
    if not d.is_dir():
        sys.exit(f"no such directory: {d}")

    tags = sorted({p.name.split(".")[0] for p in d.glob("*.log")})
    cases, failures = {}, {}
    for t in tags:
        if (d / f"{t}.failure.json").is_file():
            failures[t] = json.loads((d / f"{t}.failure.json").read_text()).get("reason", "")[:110]
        c = load_case(d, t)
        if c:
            cases[t] = c

    GEN = args.gen_tokens          # emitted tokens per run (seq_len - max_context)

    def blocks(tag):
        c = cases.get(tag, {})
        e = c.get("e2e_model", {}).get("latency_ms_avg")
        p_ = c.get("context_encoding_model", {}).get("latency_ms_avg")
        return e, p_

    b_e2e, b_pre = blocks("baseline")
    if not b_e2e:
        print("baseline missing -- no speedups can be computed")
        return
    b_decode = b_e2e - (b_pre or 0.0)
    b_per_tok = b_decode / GEN
    print(f"baseline: e2e {b_e2e:.1f} ms | prefill {b_pre:.2f} ms | "
          f"{b_per_tok:.3f} ms/token | {1000/b_per_tok:.2f} real tok/s")

    # Draft cost, if measured. Without it r is an assumption and E[accepted]
    # cannot be separated from it.
    d_e2e, d_pre = blocks("draft_only")
    d_per_tok = ((d_e2e - (d_pre or 0.0)) / GEN) if d_e2e else None
    if d_per_tok:
        print(f"draft   : {d_per_tok:.3f} ms/token  ->  r = {d_per_tok/b_per_tok:.4f} (MEASURED)")
    else:
        print("draft   : NOT MEASURED -- E[accepted] and acceptance rate withheld")
    print()

    hdr = (f"{'k':>4} {'e2e ms':>9} {'prefill':>8} {'ms/tok':>8} {'speedup':>8} "
           f"{'prefill%':>9} {'E[acc]':>7} {'accept a':>9}")
    print(hdr); print("-" * len(hdr))

    def solve_a(target, k):
        lo, hi = 1e-6, 0.999999
        for _ in range(200):
            mid = (lo + hi) / 2
            if sum(mid**i for i in range(k + 1)) < target: lo = mid
            else: hi = mid
        return (lo + hi) / 2

    rows = []
    order = sorted(cases, key=lambda x: (k_of(x) is None, k_of(x) or 0))
    for t in order:
        if t in ("baseline", "draft_only"):
            continue
        k = k_of(t)
        e, p_ = blocks(t)
        if not e:
            continue
        dec = e - (p_ or 0.0)
        per = dec / GEN
        sp = b_per_tok / per
        pre_pct = ((p_ - b_pre) / b_pre * 100) if (p_ and b_pre) else float("nan")
        eacc = acc = float("nan")
        if d_per_tok and k:
            eacc = (b_per_tok + k * d_per_tok) / per   # iteration cost / cost-per-token
            acc = solve_a(eacc, k)
        print(f"{(k if k is not None else t):>4} {e:9.1f} {p_ or float('nan'):8.2f} "
              f"{per:8.3f} {sp:7.3f}x {pre_pct:8.1f}% {eacc:7.3f} {acc:9.3f}")
        rows.append({"k": k, "e2e_ms": e, "prefill_ms": p_, "ms_per_token": per,
                     "speedup": sp, "prefill_penalty_pct": pre_pct,
                     "expected_accepted": None if eacc != eacc else eacc,
                     "acceptance_rate": None if acc != acc else acc})

    if failures:
        print("\nFAILED (a wall is a result):")
        for t in sorted(failures, key=lambda x: (k_of(x) is None, k_of(x) or 0)):
            print(f"  {t:>12}: {failures[t]}")

    if rows:
        best = max(rows, key=lambda r: r["speedup"])
        ks = [r["k"] for r in rows if r["k"] is not None]
        print(f"\npeak: k={best['k']} at {best['speedup']:.3f}x  ({best['ms_per_token']:.3f} ms/token)")
        print(f"measured k: {ks}")
        if ks and best["k"] == max(ks):
            print("NOTE: peak sits at the largest k measured -- optimum may lie beyond the sweep.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gen_tokens": GEN,
            "baseline_ms_per_token": b_per_tok,
            "draft_ms_per_token": d_per_tok,
            "r_measured": (d_per_tok / b_per_tok) if d_per_tok else None,
            "rows": rows, "failures": failures}, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
