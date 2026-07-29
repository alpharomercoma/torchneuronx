#!/usr/bin/env python3
"""Merge both machines' raw results into one comparison.

Runs on the Mac after results are pulled back:

    python3 shared/summarize.py --mi300x mi300x/results --h200 h200/results \
        --out analysis

Two rules are enforced here rather than left to discipline:

  * NO ROW WITHOUT TELEMETRY. Every serving data point must have a matching
    telemetry CSV, otherwise the "the GPU was saturated" claim behind it is
    unverifiable. Such rows are dropped and counted in `dropped_no_telemetry`.
  * Head-to-head tables contain only lanes both machines completed. Anything
    one-sided (the MI300X capacity lane, a model that failed to serve on one
    side) is emitted separately under `one_sided` so it can never be quietly
    averaged into a summary number.
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

SIDES = ("mi300x", "h200")
LABEL = {"mi300x": "MI300X", "h200": "H200"}


# --------------------------------------------------------------------------- io
def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


BUSY_UTIL_THRESHOLD = 90.0


def load_telemetry(path):
    """Return summary stats for a telemetry CSV, or None if unusable.

    Whole-window means are reported but are NOT the saturation metric. The
    sampler wraps the entire client invocation, and `vllm bench serve` spends
    its first seconds sampling and tokenising the dataset with the GPU idle.
    On short runs that prologue can be half the samples and drags the mean to
    ~45% even when the GPU is pinned at 100% for every second of actual work.

    So saturation and energy are reported over the BUSY WINDOW: the span from
    the first to the last sample at >=90% utilisation. Reporting the
    whole-window mean instead would understate both GPUs, and by different
    amounts depending on how long each took to warm up.
    """
    if not os.path.exists(path):
        return None
    cols = {k: [] for k in
            ("gpu_util_pct", "power_w", "mem_used_mib", "temp_c", "sclk_mhz")}
    try:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                for key, acc in cols.items():
                    v = row.get(key)
                    if v not in (None, "", "None"):
                        try:
                            acc.append(float(v))
                        except ValueError:
                            acc.append(None)
                    else:
                        acc.append(None)
    except Exception:
        return None

    util = cols["gpu_util_pct"]
    if not any(x is not None for x in util) and \
       not any(x is not None for x in cols["power_w"]):
        return None

    def stats(xs, lo=None, hi=None):
        vals = [x for x in (xs[lo:hi] if lo is not None else xs) if x is not None]
        if not vals:
            return None
        return {"mean": round(statistics.fmean(vals), 2),
                "max": round(max(vals), 2),
                "p50": round(statistics.median(vals), 2)}

    busy_idx = [i for i, x in enumerate(util)
                if x is not None and x >= BUSY_UTIL_THRESHOLD]
    lo, hi = (busy_idx[0], busy_idx[-1] + 1) if busy_idx else (None, None)

    out = {
        "samples": len(util),
        "busy_samples": len(busy_idx),
        "busy_window": [lo, hi] if busy_idx else None,
        "whole_window": {k: stats(v) for k, v in cols.items()},
    }
    out["busy"] = ({k: stats(v, lo, hi) for k, v in cols.items()}
                   if busy_idx else None)
    # convenience aliases used by the renderer / energy maths
    src = out["busy"] or out["whole_window"]
    out["gpu_util_pct"] = src["gpu_util_pct"]
    out["power_w"] = src["power_w"]
    out["mem_used_mib"] = src["mem_used_mib"]
    out["temp_c"] = src["temp_c"]
    out["sclk_mhz"] = src["sclk_mhz"]
    return out


# ------------------------------------------------------------------ lane parsers
def parse_micro(root):
    out = {}
    for name in ("gemm", "membw", "attn"):
        d = load_json(os.path.join(root, "micro", f"{name}.json"))
        if d:
            out[name] = d
    return out


def parse_serve(root):
    """-> ({(model_key, mode): {tag: row}}, dropped, failures)"""
    serve_root = os.path.join(root, "serve")
    runs = defaultdict(dict)
    failures = {}
    if not os.path.isdir(serve_root):
        return runs, 0, failures

    dropped = 0
    for run_dir in sorted(os.listdir(serve_root)):
        full = os.path.join(serve_root, run_dir)
        if not os.path.isdir(full):
            continue
        if "_" not in run_dir:
            continue
        model_key, mode = run_dir.rsplit("_", 1)

        # a recorded load failure IS a result (the capacity lane depends on it)
        fail = load_json(os.path.join(full, "load_failure.json"))
        if fail:
            failures[f"{model_key}_{mode}"] = fail

        for fn in sorted(os.listdir(full)):
            if not fn.endswith(".json") or fn in ("server.id", "load_failure.json"):
                continue
            tag = fn[:-5]
            d = load_json(os.path.join(full, fn))
            if not d or "output_throughput" not in d:
                continue

            tel = load_telemetry(os.path.join(full, f"{tag}.telemetry.csv"))
            if tel is None:
                dropped += 1
                continue

            tpot = d.get("mean_tpot_ms") or 0.0
            per_user = (1000.0 / tpot) if tpot else None
            dur = d.get("duration") or 0.0
            tot_tok = (d.get("total_output_tokens") or 0)
            pw = (tel.get("power_w") or {}).get("mean")
            tok_per_j = (tot_tok / (pw * dur)) if (pw and dur) else None

            runs[(model_key, mode)][tag] = {
                "concurrency": d.get("max_concurrency"),
                "isl": _tag_num(tag, "isl"), "osl": _tag_num(tag, "osl"),
                "output_tok_s": round(d["output_throughput"], 1),
                "total_tok_s": round(d.get("total_token_throughput", 0), 1),
                "tok_s_per_user": round(per_user, 2) if per_user else None,
                "ttft_p50_ms": _r(d.get("p50_ttft_ms")),
                "ttft_p99_ms": _r(d.get("p99_ttft_ms")),
                "tpot_p50_ms": _r(d.get("p50_tpot_ms")),
                "tpot_p99_ms": _r(d.get("p99_tpot_ms")),
                "itl_p99_ms": _r(d.get("p99_itl_ms")),
                "completed": d.get("completed"), "failed": d.get("failed"),
                "duration_s": _r(dur),
                "tokens_per_joule": round(tok_per_j, 3) if tok_per_j else None,
                "telemetry": tel,
            }
    return runs, dropped, failures


def _tag_num(tag, prefix):
    for part in tag.split("_"):
        if part.startswith(prefix):
            try:
                return int(part[len(prefix):])
            except ValueError:
                return None
    return None


def _r(x, nd=2):
    return round(x, nd) if isinstance(x, (int, float)) else None


def parse_simple(root, rel):
    return load_json(os.path.join(root, rel))


# ---------------------------------------------------------------- comparison
def cmp_ratio(a, b, higher_better=True):
    """Return (winner, ratio) where ratio >= 1 describes the winner's margin."""
    if a is None or b is None or a == 0 or b == 0:
        return None, None
    if higher_better:
        return ("MI300X", a / b) if a >= b else ("H200", b / a)
    return ("MI300X", b / a) if a <= b else ("H200", a / b)


def build(mi_root, h_root):
    data = {}
    dropped = {}
    load_failures = {}
    for side, root in (("mi300x", mi_root), ("h200", h_root)):
        serve, drop, fails = parse_serve(root)
        data[side] = {
            "micro": parse_micro(root),
            "serve": serve,
            "train": {
                k: parse_simple(root, f"train/{k}.json")
                for k in ("eager", "compiled")
            },
            "cpu": parse_simple(root, "cpu/cpu.json"),
            "quality": {
                k: parse_simple(root, f"quality/{k}.json")
                for k in ("8b", "70b-fp8")
            },
            "sustained": parse_simple(root, "sustained/sustained.json"),
        }
        dropped[side] = drop
        load_failures[side] = fails

    out = {"dropped_no_telemetry": dropped,
           "load_failures": load_failures,
           "head_to_head": {}, "one_sided": {}}

    # ---- microbenchmarks
    micro = {}
    for name, keyfn in (
        ("gemm", lambda r: (r["dtype"], r["shape"])),
        ("membw", lambda r: (r["kernel"],)),
        ("attn", lambda r: (r["case"],)),
    ):
        rows = {}
        for side in SIDES:
            d = data[side]["micro"].get(name)
            if not d:
                continue
            for r in d["results"]:
                if "error" in r:
                    continue
                rows.setdefault(keyfn(r), {})[side] = r
        merged = []
        for k, per in sorted(rows.items()):
            if len(per) != 2:
                continue
            entry = {"key": "/".join(k)}
            for side in SIDES:
                r = per[side]
                for f in ("tflops", "gb_s", "pct_of_peak", "ms",
                          "fwd_tflops", "bwd_tflops",
                          "fwd_pct_of_peak", "bwd_pct_of_peak"):
                    if f in r:
                        entry[f"{side}_{f}"] = r[f]
            main = ("tflops" if "mi300x_tflops" in entry
                    else "gb_s" if "mi300x_gb_s" in entry else "fwd_tflops")
            w, ratio = cmp_ratio(entry.get(f"mi300x_{main}"),
                                 entry.get(f"h200_{main}"))
            entry["winner"], entry["ratio"] = w, _r(ratio, 3)
            merged.append(entry)
        micro[name] = merged
    out["head_to_head"]["micro"] = micro

    # ---- achieved efficiency: the attribution result
    #
    # Each GPU is scored against ITS OWN published peak, so this asks "how much
    # of the silicon you paid for does the software actually reach?" rather
    # than "which is faster". It is the number that explains the serving lane:
    # if one part reaches a far higher fraction of its own peak, a paper FLOPs
    # advantage can be entirely erased before any model is loaded.
    eff = {}
    gm = data["mi300x"]["micro"].get("gemm")
    gh = data["h200"]["micro"].get("gemm")
    if gm and gh:
        def index(d):
            return {(r["dtype"], r["shape"]): r
                    for r in d["results"] if "pct_of_peak" in r}
        M, H = index(gm), index(gh)
        groups = {
            "square_all": lambda s: s.startswith("square"),
            "square_ge4096": lambda s: (s.startswith("square")
                                        and int(s.split("_")[1]) >= 4096),
            "llm_prefill": lambda s: s.startswith("llama") and "decode" not in s,
            "llm_decode": lambda s: "decode" in s,
        }
        for dtype in ("bf16", "fp8"):
            for gname, pred in groups.items():
                keys = [k for k in M if k[0] == dtype and pred(k[1]) and k in H]
                if not keys:
                    continue
                a = statistics.fmean(M[k]["pct_of_peak"] for k in keys)
                b = statistics.fmean(H[k]["pct_of_peak"] for k in keys)
                eff[f"{dtype}/{gname}"] = {
                    "mi300x_pct_of_own_peak": round(a, 1),
                    "h200_pct_of_own_peak": round(b, 1),
                    "gap_points": round(b - a, 1),
                    "n_shapes": len(keys),
                }
        for dtype in ("bf16", "fp8"):
            am = max((M[k]["tflops"] for k in M if k[0] == dtype), default=None)
            ah = max((H[k]["tflops"] for k in H if k[0] == dtype), default=None)
            if am and ah:
                eff[f"{dtype}/best_sustained"] = {
                    "mi300x_tflops": round(am, 1),
                    "mi300x_peak": gm["peak"][f"{dtype}_tflops"],
                    "h200_tflops": round(ah, 1),
                    "h200_peak": gh["peak"][f"{dtype}_tflops"],
                    "h200_over_mi300x": round(ah / am, 3),
                    "paper_peak_ratio_mi300x_over_h200": round(
                        gm["peak"][f"{dtype}_tflops"] / gh["peak"][f"{dtype}_tflops"], 3),
                }
    mm = data["mi300x"]["micro"].get("membw")
    mh = data["h200"]["micro"].get("membw")
    if mm and mh:
        am = max(r["gb_s"] for r in mm["results"])
        ah = max(r["gb_s"] for r in mh["results"])
        eff["hbm/best_sustained"] = {
            "mi300x_gb_s": round(am, 1), "mi300x_peak_tbs": mm["peak_tbs"],
            "h200_gb_s": round(ah, 1), "h200_peak_tbs": mh["peak_tbs"],
            "h200_over_mi300x": round(ah / am, 3),
            "paper_peak_ratio_mi300x_over_h200": round(
                mm["peak_tbs"] / mh["peak_tbs"], 3),
        }
    out["achieved_efficiency"] = eff

    # ---- repeatability: the baseline any tuned-vs-stock claim is judged against
    #
    # Two runs of the IDENTICAL stock configuration.
    #
    # This turned out to measure something more specific than noise. The
    # deviations reproduce almost exactly across two independent repeats -- the
    # same points, the same magnitudes -- which random variance would not do.
    # The likely cause is sweep composition: the full stock sweep walks
    # concurrency 1,4,8,16,32,... so the server is thoroughly warmed by the time
    # it reaches 32, while the coarse grid used for tuned and repeat runs jumps
    # straight from 1 to 32.
    #
    # Consequence, and the reason this matters: the tuned lanes use the COARSE
    # grid, so they must be compared against `8b_repeat` (same grid) rather than
    # `8b_stock` (different grid). Comparing across grids would attribute a
    # warm-up artefact to the vendor's tuning. `tuned_vs_repeat` below does the
    # like-for-like comparison; anything smaller than the spread here is not a
    # result either way.
    repeat = {}
    for side in SIDES:
        base = data[side]["serve"].get(("8b", "stock"), {})
        rep = data[side]["serve"].get(("8b", "repeat"), {})
        tags = sorted(set(base) & set(rep))
        if not tags:
            continue
        ratios = []
        points = []
        for t in tags:
            a = base[t]["output_tok_s"]
            b = rep[t]["output_tok_s"]
            if not a:
                continue
            ratios.append(b / a)
            points.append({"tag": t, "run1": a, "run2": b, "ratio": round(b / a, 4)})
        if ratios:
            worst = max(abs(1 - r) for r in ratios)
            repeat[side] = {
                "measures": ("CROSS-GRID deviation, not repeatability: 8b_stock "
                             "used the full 8-point grid and 8b_repeat the "
                             "coarse 3-point grid, so this quantifies the "
                             "sweep-composition/warm-up artefact"),
                "n_points": len(ratios),
                "mean_ratio": round(statistics.fmean(ratios), 4),
                "min_ratio": round(min(ratios), 4),
                "max_ratio": round(max(ratios), 4),
                "worst_deviation_pct": round(100 * worst, 1),
                "points": points,
            }
    out["cross_grid_deviation"] = repeat

    # The genuine same-grid noise floor comes from the H200 control: its
    # tuned.env sets only a sampler flag, which does nothing to serving
    # throughput, so h200 tuned-vs-repeat is effectively the same configuration
    # run twice on the same grid. That is the bar a real tuning effect must clear.
    out["same_grid_noise_floor"] = {
        "source": ("h200/8b tuned vs repeat -- NVIDIA's tuned.env sets only "
                   "VLLM_USE_FLASHINFER_SAMPLER, inert for serving throughput, "
                   "so this is one configuration measured twice on one grid"),
        "note": "populated from tuned_vs_baseline below once computed",
    }

    # ---- tuned vs the SAME-GRID baseline (the defensible tuning comparison)
    tuned_cmp = {}
    for side in SIDES:
        for model in ("8b", "70b-fp8"):
            tuned = data[side]["serve"].get((model, "tuned"), {})
            if not tuned:
                continue
            # prefer the same-grid repeat baseline; fall back to stock, but say so
            baseline = data[side]["serve"].get((model, "repeat"), {})
            basis = "repeat (same coarse grid)"
            if not baseline:
                baseline = data[side]["serve"].get((model, "stock"), {})
                basis = "stock (DIFFERENT grid -- warm-up confound possible)"
            tags = sorted(set(tuned) & set(baseline))
            if not tags:
                continue
            ratios, points = [], []
            for t in tags:
                a = baseline[t]["output_tok_s"]
                b = tuned[t]["output_tok_s"]
                if not a:
                    continue
                ratios.append(b / a)
                points.append({"tag": t, "baseline": a, "tuned": b,
                               "ratio": round(b / a, 4)})
            if not ratios:
                continue
            # judge against the SAME-GRID control, never the cross-grid figure
            noise = None
            mean_r = statistics.fmean(ratios)
            tuned_cmp[f"{side}/{model}"] = {
                "baseline_basis": basis,
                "n_points": len(ratios),
                "mean_ratio": round(mean_r, 4),
                "min_ratio": round(min(ratios), 4),
                "max_ratio": round(max(ratios), 4),
                "measured_noise_floor_pct": noise,
                "exceeds_noise_floor": (
                    None if noise is None
                    else abs(100 * (mean_r - 1.0)) > noise),
                "points": points,
            }
    # derive the same-grid noise floor from the H200 control, then re-judge
    ctrl = tuned_cmp.get("h200/8b")
    if ctrl and ctrl["baseline_basis"].startswith("repeat"):
        floor = max(abs(100 * (r["ratio"] - 1.0)) for r in ctrl["points"])
        out["same_grid_noise_floor"] = {
            "source": ("h200/8b tuned vs repeat -- NVIDIA's tuned.env sets only "
                       "VLLM_USE_FLASHINFER_SAMPLER, inert for serving "
                       "throughput, so this is one configuration measured twice "
                       "on one grid"),
            "worst_deviation_pct": round(floor, 2),
            "mean_ratio": ctrl["mean_ratio"],
        }
        for k, v in tuned_cmp.items():
            if not v["baseline_basis"].startswith("repeat"):
                continue
            v["measured_noise_floor_pct"] = round(floor, 2)
            v["exceeds_noise_floor"] = abs(100 * (v["mean_ratio"] - 1.0)) > floor
    out["tuned_vs_baseline"] = tuned_cmp

    # ---- serving
    serve_cmp, one_sided_serve = {}, {}
    all_runs = set(data["mi300x"]["serve"]) | set(data["h200"]["serve"])
    for run in sorted(all_runs):
        mi = data["mi300x"]["serve"].get(run, {})
        h2 = data["h200"]["serve"].get(run, {})
        run_name = f"{run[0]}_{run[1]}"
        shared_tags = sorted(set(mi) & set(h2))
        if not shared_tags:
            if mi or h2:
                one_sided_serve[run_name] = {
                    "present_on": [LABEL[s] for s, v in
                                   (("mi300x", mi), ("h200", h2)) if v],
                    "points": {LABEL["mi300x"]: mi, LABEL["h200"]: h2},
                }
            continue
        rows = []
        for tag in shared_tags:
            a, b = mi[tag], h2[tag]
            w, ratio = cmp_ratio(a["output_tok_s"], b["output_tok_s"])
            rows.append({
                "tag": tag, "isl": a["isl"], "osl": a["osl"],
                "concurrency": a["concurrency"],
                "mi300x": a, "h200": b,
                "winner": w, "throughput_ratio": _r(ratio, 3),
            })
        serve_cmp[run_name] = rows
    out["head_to_head"]["serve"] = serve_cmp
    out["one_sided"]["serve"] = one_sided_serve

    # ---- training
    train = {}
    for variant in ("eager", "compiled"):
        a = data["mi300x"]["train"].get(variant)
        b = data["h200"]["train"].get(variant)
        if not a or not b:
            if a or b:
                out["one_sided"].setdefault("train", {})[variant] = {
                    "present_on": [LABEL[s] for s, v in
                                   (("mi300x", a), ("h200", b)) if v]}
            continue
        w, ratio = cmp_ratio(a["tokens_per_s"], b["tokens_per_s"])
        # loss traces must overlay, else throughput is not comparable
        la = [x["loss"] for x in a["loss_trace"]]
        lb = [x["loss"] for x in b["loss_trace"]]
        n = min(len(la), len(lb))
        max_dl = max((abs(la[i] - lb[i]) for i in range(n)), default=None)
        train[variant] = {
            "mi300x": {k: a[k] for k in
                       ("tokens_per_s", "tflops", "mfu_pct",
                        "median_step_ms", "peak_mem_gib")},
            "h200": {k: b[k] for k in
                     ("tokens_per_s", "tflops", "mfu_pct",
                      "median_step_ms", "peak_mem_gib")},
            "winner": w, "throughput_ratio": _r(ratio, 3),
            "max_abs_loss_delta": _r(max_dl, 5),
            "loss_traces_agree": (max_dl is not None and max_dl < 0.05),
        }
    out["head_to_head"]["train"] = train

    # ---- cpu (host comparison, labelled)
    out["host_cpu"] = {
        "note": ("HOST comparison, not a chip comparison: the cloud providers "
                 "chose these CPUs, not the GPU vendors."),
        "mi300x": data["mi300x"]["cpu"], "h200": data["h200"]["cpu"],
    }

    # ---- quality
    qual = {}
    for m in ("8b", "70b-fp8"):
        a = data["mi300x"]["quality"].get(m)
        b = data["h200"]["quality"].get(m)
        if not a or not b:
            continue
        exact = sum(1 for x, y in zip(a["results"], b["results"])
                    if x["text"] == y["text"])
        qual[m] = {
            "n_prompts": min(len(a["results"]), len(b["results"])),
            "exact_match": exact,
            "exact_match_pct": round(100.0 * exact / max(1, len(a["results"])), 1),
            "mi300x_corpus_mean_logprob": a.get("corpus_mean_logprob"),
            "h200_corpus_mean_logprob": b.get("corpus_mean_logprob"),
            "logprob_delta": _r(
                (a.get("corpus_mean_logprob") or 0)
                - (b.get("corpus_mean_logprob") or 0), 6),
        }
    out["head_to_head"]["quality"] = qual

    out["sustained"] = {"mi300x": data["mi300x"]["sustained"],
                        "h200": data["h200"]["sustained"]}
    return out


# --------------------------------------------------------------------- render
def render(cmp):
    L = []
    add = L.append
    add("=" * 78)
    add("MI300X vs H200 -- summary")
    add("=" * 78)

    drop = cmp.get("dropped_no_telemetry", {})
    if any(drop.values()):
        add(f"\n! rows dropped for missing telemetry: {drop}")

    eff = cmp.get("achieved_efficiency") or {}
    if eff:
        add("\n-- Achieved efficiency: % of each GPU's OWN dense peak " + "-" * 20)
        add("   (the attribution result -- how much of the silicon the software reaches)")
        add(f"{'group':<24}{'MI300X':>10}{'H200':>10}{'gap':>10}")
        for k in sorted(x for x in eff if "best_sustained" not in x):
            v = eff[k]
            add(f"{k:<24}{v['mi300x_pct_of_own_peak']:>9.1f}%"
                f"{v['h200_pct_of_own_peak']:>9.1f}%{v['gap_points']:>9.1f}pt")
        for k in sorted(x for x in eff if "best_sustained" in x):
            v = eff[k]
            if "mi300x_tflops" in v:
                add(f"  {k}: MI300X {v['mi300x_tflops']:.0f} / {v['mi300x_peak']:.0f} peak"
                    f"  |  H200 {v['h200_tflops']:.0f} / {v['h200_peak']:.0f} peak"
                    f"  -> H200 {v['h200_over_mi300x']:.2f}x on"
                    f" {v['paper_peak_ratio_mi300x_over_h200']:.2f}x LOWER paper peak")
            else:
                add(f"  {k}: MI300X {v['mi300x_gb_s']:.0f} GB/s / {v['mi300x_peak_tbs']} TB/s"
                    f"  |  H200 {v['h200_gb_s']:.0f} / {v['h200_peak_tbs']} TB/s"
                    f"  -> H200 {v['h200_over_mi300x']:.2f}x on"
                    f" {v['paper_peak_ratio_mi300x_over_h200']:.2f}x LOWER paper peak")

    micro = cmp["head_to_head"]["micro"]
    if micro.get("gemm"):
        add("\n-- GEMM (TFLOP/s, % of that GPU's own dense peak) " + "-" * 26)
        add(f"{'shape':<28}{'MI300X':>18}{'H200':>18}{'winner':>10}{'x':>7}")
        for r in micro["gemm"]:
            a = f"{r.get('mi300x_tflops', 0):.0f} ({r.get('mi300x_pct_of_peak', 0):.0f}%)"
            b = f"{r.get('h200_tflops', 0):.0f} ({r.get('h200_pct_of_peak', 0):.0f}%)"
            add(f"{r['key']:<28}{a:>18}{b:>18}{str(r['winner']):>10}"
                f"{r['ratio'] or 0:>7.2f}")

    if micro.get("membw"):
        add("\n-- HBM bandwidth (GB/s) " + "-" * 50)
        add(f"{'kernel':<28}{'MI300X':>18}{'H200':>18}{'winner':>10}{'x':>7}")
        for r in micro["membw"]:
            a = f"{r.get('mi300x_gb_s', 0):.0f} ({r.get('mi300x_pct_of_peak', 0):.0f}%)"
            b = f"{r.get('h200_gb_s', 0):.0f} ({r.get('h200_pct_of_peak', 0):.0f}%)"
            add(f"{r['key']:<28}{a:>18}{b:>18}{str(r['winner']):>10}"
                f"{r['ratio'] or 0:>7.2f}")

    if micro.get("attn"):
        add("\n-- Attention fwd / bwd (TFLOP/s) " + "-" * 41)
        add(f"{'case':<28}{'MI300X f/b':>20}{'H200 f/b':>20}{'win(fwd)':>10}")
        for r in micro["attn"]:
            a = f"{r.get('mi300x_fwd_tflops', 0):.0f}/{r.get('mi300x_bwd_tflops', 0):.0f}"
            b = f"{r.get('h200_fwd_tflops', 0):.0f}/{r.get('h200_bwd_tflops', 0):.0f}"
            add(f"{r['key']:<28}{a:>20}{b:>20}{str(r['winner']):>10}")

    for run, rows in cmp["head_to_head"]["serve"].items():
        add(f"\n-- Serving: {run} " + "-" * (60 - len(run)))
        add(f"{'isl/osl':>10}{'conc':>6}{'MI300X tok/s':>15}{'H200 tok/s':>13}"
            f"{'winner':>9}{'x':>7}{'MI tok/J':>10}{'H2 tok/J':>10}")
        for r in rows:
            m, h = r["mi300x"], r["h200"]
            add(f"{str(r['isl']) + '/' + str(r['osl']):>10}{r['concurrency']:>6}"
                f"{m['output_tok_s']:>15.1f}{h['output_tok_s']:>13.1f}"
                f"{str(r['winner']):>9}{r['throughput_ratio'] or 0:>7.2f}"
                f"{(m['tokens_per_joule'] or 0):>10.2f}"
                f"{(h['tokens_per_joule'] or 0):>10.2f}")

    for variant, t in cmp["head_to_head"].get("train", {}).items():
        add(f"\n-- Training ({variant}) " + "-" * 48)
        add(f"{'':<14}{'tok/s':>12}{'TFLOP/s':>11}{'MFU%':>8}{'step ms':>10}{'peak GiB':>10}")
        for side in ("mi300x", "h200"):
            v = t[side]
            add(f"{LABEL[side]:<14}{v['tokens_per_s']:>12,.0f}{v['tflops']:>11.1f}"
                f"{v['mfu_pct']:>8.1f}{v['median_step_ms']:>10.1f}"
                f"{v['peak_mem_gib']:>10.1f}")
        add(f"  winner: {t['winner']} ({t['throughput_ratio']}x)   "
            f"loss traces agree: {t['loss_traces_agree']} "
            f"(max |dloss| = {t['max_abs_loss_delta']})")

    # saturation evidence: measured over the busy window, not whole-window mean
    add("\n-- Saturation during serving (busy window, util >= 90%) " + "-" * 18)
    add(f"{'run':<20}{'MI300X util/power':>24}{'H200 util/power':>24}")
    for run, rows in cmp["head_to_head"]["serve"].items():
        cells = []
        for side in SIDES:
            util = [r[side]["telemetry"]["gpu_util_pct"]["mean"]
                    for r in rows if r[side]["telemetry"].get("gpu_util_pct")]
            power = [r[side]["telemetry"]["power_w"]["mean"]
                     for r in rows if r[side]["telemetry"].get("power_w")]
            u = statistics.fmean(util) if util else 0.0
            p = statistics.fmean(power) if power else 0.0
            cells.append(f"{u:.0f}% / {p:.0f} W")
        add(f"{run:<20}{cells[0]:>24}{cells[1]:>24}")

    rep = cmp.get("repeatability") or {}
    if rep:
        add("\n-- Serving repeatability (identical config, run twice) " + "-" * 19)
        for side, v in rep.items():
            add(f"  {LABEL[side]:<8} mean {v['mean_ratio']:.4f}  "
                f"range {v['min_ratio']:.3f}-{v['max_ratio']:.3f}  "
                f"worst deviation {v['worst_deviation_pct']:.1f}%  (n={v['n_points']})")
        add("  -> any tuning effect smaller than this is not a result")

    tv = cmp.get("tuned_vs_baseline") or {}
    if tv:
        add("\n-- Vendor tuning effect on serving " + "-" * 39)
        for key, v in sorted(tv.items()):
            verdict = ("BELOW NOISE FLOOR" if v["exceeds_noise_floor"] is False
                       else "exceeds noise floor" if v["exceeds_noise_floor"]
                       else "no noise floor measured")
            add(f"  {key:<18} mean {v['mean_ratio']:.4f} "
                f"({v['min_ratio']:.3f}-{v['max_ratio']:.3f}, n={v['n_points']})  {verdict}")
            add(f"{'':<20}baseline: {v['baseline_basis']}")

    q = cmp["head_to_head"].get("quality", {})
    for m, v in q.items():
        add(f"\n-- Quality cross-check: {m} " + "-" * 40)
        add(f"  greedy exact match : {v['exact_match']}/{v['n_prompts']} "
            f"({v['exact_match_pct']}%)")
        add(f"  mean logprob       : MI300X {v['mi300x_corpus_mean_logprob']}  "
            f"H200 {v['h200_corpus_mean_logprob']}  "
            f"delta {v['logprob_delta']}")

    one = cmp.get("one_sided", {}).get("serve", {})
    if one:
        add("\n-- One-sided lanes (NOT head-to-head) " + "-" * 36)
        for run, v in one.items():
            add(f"  {run}: ran on {', '.join(v['present_on'])} only")

    add("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mi300x", required=True)
    ap.add_argument("--h200", required=True)
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    cmp = build(args.mi300x, args.h200)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "comparison.json"), "w") as fh:
        json.dump(cmp, fh, indent=2)
    text = render(cmp)
    with open(os.path.join(args.out, "comparison.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
