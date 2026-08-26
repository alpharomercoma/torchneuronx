"""Turn the accuracy-validity receipts into the two tables a talk needs.

    python3 analysis/accuracy_summary.py trn1/results/accuracy inf2/results/accuracy
    python3 analysis/accuracy_summary.py --markdown ...      # slide-ready

Reads only receipts. Computes nothing it cannot show the provenance of, and
refuses to print a delta whose two receipts disagree about the protocol --
the same guard the lanes apply, applied again at presentation time, because
a table is where an unpaired comparison would do the most damage.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "shared", "eval"))
import accuracy as A  # noqa: E402


KNOWN_BOXES = ("trn1", "trn2", "inf1", "inf2")


def box_of(d):
    """Name the box from any path shape, absolute or relative.

    Taking the first path component gave "opt" for
    /opt/np/repo/trn1/results/accuracy -- a cosmetic bug, but one that would
    have put "opt" in a slide column twice and made two boxes' rows
    indistinguishable.
    """
    parts = [x for x in os.path.abspath(d).split(os.sep) if x]
    for x in reversed(parts):
        if x in KNOWN_BOXES:
            return x
    return parts[-2] if len(parts) > 1 else "?"


def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f.endswith((".images.json", ".utterances.json")):
            continue
        tag = f[:-5]
        if tag.endswith(".failure"):
            tag = tag[:-len(".failure")] + " (failed)"
        try:
            out[tag] = json.load(open(os.path.join(d, f)))
        except Exception as exc:
            out[tag] = {"status": "unreadable", "reason": str(exc)}
    return out


def pct(x, nd=2):
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def per_split_wer(d, tag):
    """WER broken out by LibriSpeech split, from the per-utterance records.

    Pooled dev-all is what MLCommons score, but every LibriSpeech number in the
    literature is reported per split -- and the two differ enough to matter
    (dev-other is roughly twice dev-clean's error rate). Reporting only the
    pooled figure would make our number look bad against a test-clean anchor
    and good against a test-other one, for reasons that are entirely about
    which corpus was scored.

    Recomputed from the stored counts rather than re-scoring, so it cannot
    drift from the headline number.
    """
    path = os.path.join(d, tag + ".utterances.json")
    if not os.path.exists(path):
        return {}
    recs = json.load(open(path)).get("records", [])
    agg = {}
    for r in recs:
        c = r.get("counts")
        if not c:
            continue
        a = agg.setdefault(r.get("split", "?"), {"err": 0, "ref": 0, "n": 0})
        a["err"] += c["sub"] + c["ins"] + c["del"]
        a["ref"] += c["ref_words"]
        a["n"] += 1
    return {k: {"wer": v["err"] / v["ref"] if v["ref"] else float("nan"),
                "utterances": v["n"], "ref_words": v["ref"]}
            for k, v in sorted(agg.items())}


def row_accuracy(box, tag, r):
    """One measured lane."""
    if r.get("status") in ("lane_failed", "lane_skipped"):
        return [box, tag, "—", "—", "—", f"FAILED: {r.get('reason','')[:60]}"]
    if "wer" in r:
        note = []
        if r.get("truncated"):
            note.append(f"{r['truncated']} unfinished")
        anchor = (r.get("published_anchor") or {}).get("test_clean_wer")
        if anchor:
            note.append(f"published {anchor*100:.1f}% (test-clean)")
        return [box, tag, pct(r["wer"], 3), pct(r["word_accuracy"], 3),
                r["engine_meta"].get("dtype", "?"), "; ".join(note)]
    if "top1" in r:
        note = []
        if r.get("nonfinite_rows"):
            note.append(f"**{r['nonfinite_rows']}/{r['n_samples']} NaN**")
        if r.get("degenerate_rows"):
            note.append(f"{r['degenerate_rows']} constant rows")
        anchor = (r.get("published_anchor") or {}).get("imagenet_zeroshot_top1")
        if anchor:
            note.append(f"published {anchor*100:.1f}%")
        return [box, tag, pct(r["top1"]), pct(r["top5"]),
                r["engine_meta"].get("dtype", "?"), "; ".join(note)]
    return None


def row_delta(box, tag, r):
    """One paired verdict. The gate and the relative error increase travel
    together on purpose: MLPerf's rule is stated on accuracy, so at a 4% WER it
    passes a 4.96% WER -- a 24% relative rise in errors reading as PASS."""
    g = r.get("mlperf_gate", {})
    if "wer_delta_pp" in r:
        d = f"{r['wer_delta_pp']:+.3f} pp"
        ci = r["wer_delta_ci95_pp"]
        agree = f"{r['transcript_disagreements']}/{r['n_samples']} differ"
    else:
        d = f"{r['top1_delta_pp']:+.2f} pp"
        ci = r["top1_delta_ci95_pp"]
        agree = (f"{r['prediction_disagreements']}/{r['n_samples']} differ"
                 f" ({r.get('disagreements_on_near_ties', 0)} near-tie)")
    rel = g.get("relative_error_increase")
    return [box, tag, d, f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]", agree,
            {True: "PASS", False: "FAIL", None: "n/a"}[g.get("passed")],
            "—" if rel is None else f"{rel*100:+.1f}%"]


def table(headers, rows, markdown):
    if not rows:
        return "  (none)"
    if markdown:
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(out)
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    out = ["  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)),
           "  " + "  ".join("-" * w[i] for i in range(len(headers)))]
    out += ["  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r))
            for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    measured, deltas, problems, splits = [], [], [], []
    for d in args.dirs:
        if not os.path.isdir(d):
            problems.append(f"{d}: not a directory")
            continue
        box = box_of(d)
        for tag, r in load(d).items():
            if tag.endswith("_delta"):
                # Re-verify the pairing AT PRESENTATION TIME. A receipt pair
                # that drifted is most dangerous here, where it becomes a row
                # in a slide that nobody re-derives.
                deltas.append(row_delta(box, tag, r))
                continue
            row = row_accuracy(box, tag, r)
            if row:
                for split, v in per_split_wer(d, tag).items():
                    splits.append([box, tag, split, pct(v["wer"], 3),
                                   v["utterances"], v["ref_words"]])
                measured.append(row)
                # BOTH silent-failure shapes flag, not just NaN. A graph
                # emitting a CONSTANT row is just as dead and scores at chance
                # rather than at zero, so it reads as a plausible bad result
                # instead of an obvious broken one.
                if r.get("nonfinite_rows"):
                    problems.append(
                        f"{box}/{tag}: {r['nonfinite_rows']}/{r['n_samples']} "
                        "non-finite rows -- the graph ran and returned NaN. "
                        "Speed alone would have called this a success.")
                if r.get("degenerate_rows"):
                    problems.append(
                        f"{box}/{tag}: {r['degenerate_rows']}/{r['n_samples']} "
                        "constant rows -- the graph emitted an identical score "
                        "for every class. Scores at chance, so it looks like a "
                        "bad model rather than a dead graph.")
                if r.get("truncated"):
                    problems.append(
                        f"{box}/{tag}: {r['truncated']} utterances hit the "
                        "length cap without finishing -- a static-shape "
                        "artefact of this harness, not a hardware property.")

    print("\n## Measured accuracy (each lane, each engine)\n")
    print(table(["box", "lane", "primary", "secondary", "dtype", "note"],
                measured, args.markdown))
    if splits:
        print("\n## ASR, broken out by LibriSpeech split\n")
        print(table(["box", "lane", "split", "WER", "utterances", "ref words"],
                    splits, args.markdown))
    print("\n## Paired verdicts (accelerator vs same-box float32 CPU)\n")
    print(table(["box", "comparison", "delta", "95% CI", "disagreements",
                 "MLPerf gate", "rel. error"], deltas, args.markdown))
    if problems:
        print("\n## Flags\n")
        for p in problems:
            print(f"  * {p}")
    print("\nPrimary column: WER for ASR, top-1 for zero-shot. Secondary: word "
          "accuracy / top-5.\nThe gate is MLPerf's >=99%-of-reference rule with "
          "the CPU float32 run as reference,\nnot with anyone's published "
          "number -- see shared/eval/accuracy.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
