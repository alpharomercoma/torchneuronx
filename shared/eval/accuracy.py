"""Industry-standard accuracy metrics -- ONE implementation, shared by every
engine so the metric can never drift between the thing under test and its
control.

WHY THIS FILE EXISTS
--------------------
Every inference lane in this study reported speed only: RTF 0.047, 188.48
images/s, 121 tok/s. Speed alone cannot tell a working compile from a broken
one. The CLIP lane proved it -- on trn1 the traced graph compiled cleanly,
ran at full speed, and returned NaN probabilities. A throughput number would
have called that a success.

So accuracy here is not a headline. It is the VALIDITY CHECK on the hardware
path: does compiling for Neuron (static shapes, bf16 auto-cast, traced graph,
no KV cache) change the answer? A speed win bought by a silently wrong answer
is not a speed win.

THE FAIRNESS CONTRACT
---------------------
The primary statistic is a PAIRED DELTA between two engines that differ in
exactly one respect -- where the forward pass runs:

    engine=cpu     stock HF model, float32, eager, on the instance's own vCPUs
    engine=neuron  the compiled artifact, on a NeuronCore

Everything else is held byte-identical by construction:

  * the same sample IDs, drawn once with a fixed seed and persisted with a
    SHA-256 so a later run cannot quietly score a different subset
    (`sample_digest`, `assert_paired`);
  * the same preprocessing, from the same processor snapshot;
  * the same decode parameters (greedy, same max length);
  * the same scoring functions -- literally the functions in this module,
    imported by both engines.

That leaves the compile as the only free variable, which is the whole point.

The SECONDARY statistic is the absolute number against a published reference
(Whisper's LibriSpeech WER, CLIP's ImageNet zero-shot top-1). That one is a
harness sanity check: it catches "my evaluation loop is broken" but it CANNOT
adjudicate hardware, because a published number carries someone else's
preprocessing, decoding and normalisation choices.

Deliberate consequence: any ambiguity in the normalisation recipe (see
`mlperf_normalize` step 3) shifts BOTH arms identically and therefore cancels
out of the paired delta. It only perturbs the secondary anchor.

    uv run --with pytest python -m pytest tests/test_accuracy.py -q

Pure functions only. No torch, no transformers, no network -- so the metric
is testable on a laptop and identical on every box.
"""
import hashlib
import json
import math
import re
import unicodedata

# ---------------------------------------------------------------- text norm

_NON_ALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def basic_normalize(text):
    """MLPerf Whisper normalisation step 1: lowercase, keep only alphabetic
    characters and whitespace, collapse runs of whitespace.

    Applied to hypothesis and reference alike. Digits are DROPPED rather than
    spelled out -- that is what "retain only alphabetic characters" means and
    it is symmetric, so a model that writes "1990" and a reference that writes
    "nineteen ninety" both lose the token rather than one being punished.
    LibriSpeech references are already verbalised (no digits), so on this
    corpus the rule is close to a no-op on the reference side.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _NON_ALPHA.sub(" ", text)
    return _WS.sub(" ", text).strip()


def split_compounds(text):
    """MLPerf Whisper normalisation step 3: split compound words.

    MLCommons describe the step but do not publish the code in the writeup we
    sourced this from, so this is an INTERPRETATION and is labelled as such in
    every receipt (`normalize_recipe.step3`): split on hyphen-like joiners
    into separate words.

    Safe by construction for the primary claim -- it is deterministic and is
    applied to hypothesis and reference, on both engines, so it cancels out of
    the paired delta. It moves only the absolute WER, i.e. the secondary
    published-reference anchor.

    Runs BEFORE `basic_normalize` in `mlperf_normalize`, because step 1 would
    otherwise turn the hyphen into a space and make this a no-op.
    """
    return _WS.sub(" ", re.sub(r"[-‐-―_/]+", " ", str(text))).strip()


_PUNCT = re.compile(r"[^a-z0-9\s]")


def strip_punctuation(text):
    """Remove punctuation, KEEP digits, collapse whitespace.

    This is our reading of MLCommons' step 1 ("convert to lowercase and retain
    only alphabetic characters, removing punctuation"). We deliberately keep
    DIGITS, and the reason is not a preference -- a literal alphabetic-only
    filter is destructive in a way that silently disarms the benchmark. See
    `RECIPES` below.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", str(text).lower())).strip()


def normalize(text, english_normalizer=None, recipe="whisper"):
    """Normalise one string under a NAMED recipe.

    Two recipes, both reported in every receipt, because they answer different
    questions and picking one silently would be picking the flattering one.

    recipe="whisper"  -- EnglishTextNormalizer alone, then whitespace collapse.
        THE PRIMARY. It is the protocol OpenAI used to publish Whisper's
        LibriSpeech WER and the protocol every published Whisper WER uses, so
        it is the only recipe under which our absolute number is comparable to
        the published anchor.

    recipe="mlperf"   -- compound split -> EnglishTextNormalizer -> punctuation
        stripped, DIGITS KEPT. MLCommons' three steps, with one documented
        deviation.

    THE DEVIATION, AND WHY (found by adversarial review, 2026-08-20):
        MLCommons' step 1 says "retain only alphabetic characters". Applied
        literally AFTER step 2, it is destructive: EnglishTextNormalizer maps
        "nineteen hundred and ten" -> "1910", and an alphabetic-only filter
        then deletes it. Reference and hypothesis alike. Worked example --
            ref  "THE CHAPTER WAS HEADED NINETEEN HUNDRED AND TEN"
            hyp  "the chapter was headed 1920"        (WRONG YEAR)
        both normalise to "the chapter was headed", scoring 0 errors. Every
        mistake on a date, quantity, price or address becomes invisible, and a
        hallucinating model is rewarded. So we keep digits.

    The deviation is SYMMETRIC -- identical on reference and hypothesis, on
    both engines -- so it cannot move the paired CPU-vs-Neuron delta. It moves
    only the absolute WER.
    """
    if recipe == "whisper":
        out = str(text)
        if english_normalizer is not None:
            out = english_normalizer(out)
        return _WS.sub(" ", out.lower()).strip()
    if recipe == "mlperf":
        out = split_compounds(text)
        if english_normalizer is not None:
            out = english_normalizer(out)
        return strip_punctuation(out)
    if recipe == "mlperf_literal":
        # MLCommons' step 1 taken at its word, kept ONLY so the receipt can
        # show what the literal reading costs. Never the primary.
        out = split_compounds(text)
        if english_normalizer is not None:
            out = english_normalizer(out)
        return basic_normalize(out)
    raise ValueError(f"unknown normalisation recipe: {recipe!r}")


RECIPES = ("whisper", "mlperf", "mlperf_literal")


def mlperf_normalize(text, english_normalizer=None):
    """Back-compat alias for the MLPerf recipe (digits kept)."""
    return normalize(text, english_normalizer, recipe="mlperf")


# ---------------------------------------------------------------------- WER

def edit_counts(ref_words, hyp_words):
    """Levenshtein alignment over WORD tokens -> (sub, ins, del).

    Standard DP with a rolling row plus a rolling backtrace of the three
    counts, so memory is O(len(hyp)) and a 200-word utterance is free.

    TIE-BREAKING, stated because the receipt publishes the decomposition:
    when substitution, insertion and deletion reach the same minimal cost, the
    order below prefers substitution, then insertion, then deletion (the
    Wagner-Fischer convention jiwer and SCLITE also use). The TOTAL is the
    true edit distance and is unaffected; only which of several equally
    minimal (sub, ins, del) decompositions gets reported depends on this.
    """
    n, m = len(ref_words), len(hyp_words)
    # row[j] = (cost, sub, ins, dele) for ref[:i] vs hyp[:j]
    row = [(j, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        prev = row
        row = [(i, 0, 0, i)] + [None] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                c, s, ins, d = prev[j - 1]
                cand = (c, s, ins, d)
            else:
                c, s, ins, d = prev[j - 1]
                cand = (c + 1, s + 1, ins, d)
            c, s, ins, d = row[j - 1]                 # insertion
            if c + 1 < cand[0]:
                cand = (c + 1, s, ins + 1, d)
            c, s, ins, d = prev[j]                    # deletion
            if c + 1 < cand[0]:
                cand = (c + 1, s, ins, d + 1)
            row[j] = cand
    _, sub, ins, dele = row[m]
    return sub, ins, dele


def corpus_wer(pairs):
    """Corpus-level WER over [(reference, hypothesis), ...] of NORMALISED text.

    Corpus-level -- total edits divided by total reference words -- NOT the
    mean of per-utterance WERs. The mean-of-ratios form lets one three-word
    utterance with two errors (0.667) outweigh a fifty-word utterance with two
    errors (0.04), and it is not what LibriSpeech, Whisper's paper, or MLPerf
    report. Getting this wrong is the single most common way two "WER" numbers
    turn out to be incomparable.

    Returns a dict carrying the raw counts, so a caller can re-aggregate,
    bootstrap, or merge two runs without re-scoring.
    """
    sub = ins = dele = ref_n = hyp_n = 0
    per_utt = []
    for ref, hyp in pairs:
        r = ref.split()
        h = hyp.split()
        s, i, d = edit_counts(r, h)
        sub += s
        ins += i
        dele += d
        ref_n += len(r)
        hyp_n += len(h)
        per_utt.append({"ref_words": len(r), "hyp_words": len(h),
                        "sub": s, "ins": i, "del": d})
    errors = sub + ins + dele
    wer = errors / ref_n if ref_n else float("nan")
    return {
        "wer": wer,
        "word_accuracy": 1.0 - wer if ref_n else float("nan"),
        "errors": errors, "sub": sub, "ins": ins, "del": dele,
        "ref_words": ref_n, "hyp_words": hyp_n, "utterances": len(per_utt),
        "per_utterance": per_utt,
    }


# --------------------------------------------------------------- top-k acc

def topk_accuracy(logit_rows, labels, ks=(1, 5)):
    """Top-k accuracy from per-sample score rows. Ties broken by lowest index,
    which matches numpy/torch argsort's stable order -- stated because a
    NaN row would otherwise silently "win" whichever class sorts first.

    A row containing any non-finite value is counted WRONG for every k and
    tallied separately in `nonfinite_rows`. That is the CLIP-on-trn1 failure
    mode: the graph compiled, ran at full speed, and emitted NaN. Scoring NaN
    as a miss rather than crashing means the receipt reports 0.1% top-1 --
    a number a reader can act on -- instead of a traceback.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    hits = {k: 0 for k in ks}
    nonfinite = 0
    degenerate = 0
    n = 0
    for row, label in zip(logit_rows, labels):
        n += 1
        vals = list(row)
        if any((v != v) or (v in (float("inf"), float("-inf"))) for v in vals):
            nonfinite += 1
            continue
        if len(set(vals)) <= 1:
            # A DEAD graph emitting a constant (all-zero, all-equal) row is not
            # a 0.1%-chance guess -- with lowest-index tie-breaking it would
            # score 100% on class 0 and 100% top-5 on classes 0-4, i.e. a
            # broken compile would report a plausible-looking accuracy on a
            # stratified sample. Counted as degenerate, and always a miss.
            degenerate += 1
            continue
        order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
        for k in ks:
            if label in order[:k]:
                hits[k] += 1
    return {
        "n": n,
        "nonfinite_rows": nonfinite,
        "degenerate_rows": degenerate,
        **{f"top{k}": (hits[k] / n if n else float("nan")) for k in ks},
        **{f"top{k}_hits": hits[k] for k in ks},
    }


# ------------------------------------------------------- pairing guarantees

def sample_digest(sample_ids):
    """SHA-256 over the ORDERED sample id list.

    The engines run in separate processes, often hours apart, sometimes on
    different boxes. This digest is what makes "they scored the same data" a
    checked fact rather than an assumption -- it goes in every receipt and is
    compared by `assert_paired` before any delta is computed.
    """
    h = hashlib.sha256()
    for s in sample_ids:
        h.update(str(s).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# Every field here must be IDENTICAL for a delta between two receipts to mean
# anything. Adversarial review (2026-08-20) found compare() would happily
# subtract an fp32 run from a bf16 run, or a 224-token run from a 192-token
# one, and present the difference as a hardware effect.
PROTOCOL_FIELDS = (
    "model", "n_samples", "sample_digest",
    ("decode", "max_length"), ("decode", "strategy"),
    ("normalize", "primary"),
    ("decode", "forced_decoder"), ("decode", "decoder_prefix"),
    ("dataset", "templates"), ("dataset", "per_class"), ("dataset", "seed"),
    ("dataset", "splits"), ("dataset", "n_per_split"),
    ("engine_meta", "auto_cast"),
    ("dataset", "text_attention_mask"),
)


def _dig(d, key):
    if isinstance(key, tuple):
        for k in key:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d
    return d.get(key) if isinstance(d, dict) else None


def assert_same_protocol(a, b, ignore=()):
    """Refuse to compare two receipts whose EXPERIMENT differed, not just
    whose hardware did.

    `engine` and `engine_meta.dtype` are expected to differ -- that is the
    treatment. Everything in PROTOCOL_FIELDS is a control, and a control that
    silently drifted turns a delta into a number about nothing. Fields absent
    from BOTH receipts are skipped (lanes carry different key sets); a field
    present in one and absent in the other is a mismatch, because that is what
    a schema change between two runs looks like.
    """
    bad = []
    for key in PROTOCOL_FIELDS:
        if key in ignore:
            continue
        va, vb = _dig(a, key), _dig(b, key)
        if va is None and vb is None:
            continue
        if va != vb:
            bad.append((key, va, vb))
    if bad:
        raise ValueError(
            "receipts differ in the EXPERIMENT, not just the engine -- "
            "refusing to call this a hardware delta: "
            + "; ".join(f"{k}: {x!r} vs {y!r}" for k, x, y in bad))
    return True


def assert_paired(a, b):
    """Refuse to compare two receipts that did not see the same samples.

    Raises rather than warns. A paired delta computed across different subsets
    is not a weaker result, it is a WRONG one, and the study has already been
    bitten once by a comparison that looked fine and was not.
    """
    for key in ("sample_digest", "n_samples"):
        if a.get(key) != b.get(key):
            raise ValueError(
                f"unpaired receipts: {key} differs "
                f"({a.get(key)!r} vs {b.get(key)!r}) -- refusing to score a "
                "delta across different samples")
    if not a.get("sample_digest"):
        raise ValueError("receipts carry no sample_digest -- cannot verify pairing")
    return True


# ------------------------------------------------------------------- gates

# MLPerf Inference closed-division rule for Whisper: a submission must reach
# at least 99% of the reference model's accuracy. Re-anchored here to the
# SAME-BOX float32 CPU run rather than to MLCommons' published large-v3
# number, because the question this study asks is "did compiling for Neuron
# cost accuracy", not "is whisper-small as good as whisper-large-v3".
MLPERF_ACCURACY_FLOOR = 0.99


def accuracy_gate(candidate, reference, floor=MLPERF_ACCURACY_FLOOR,
                  error_rate=None):
    """MLPerf's >=99%-of-reference rule, as a reusable verdict.

    `candidate` / `reference` are accuracy-like (higher is better) in [0, 1]:
    word accuracy for ASR, top-1 for zero-shot. Returns the ratio and a
    pass/fail so the receipt states the verdict rather than leaving a reader
    to eyeball two decimals.
    """
    if reference is None or reference <= 0 or not math.isfinite(reference):
        return {"passed": None, "reason": "reference accuracy unavailable",
                "floor": floor}
    if not math.isfinite(candidate):
        return {"passed": False, "ratio": None, "floor": floor,
                "reason": "candidate accuracy is not finite"}
    ratio = candidate / reference
    out = {
        "passed": bool(ratio >= floor),
        "ratio": ratio,
        "floor": floor,
        "candidate": candidate,
        "reference": reference,
        "rule": "MLPerf Inference closed division: >=99% of reference accuracy",
    }
    if error_rate is not None:
        # WHY THIS FIELD EXISTS. MLPerf's rule is stated on ACCURACY, and for
        # ASR accuracy is 1 - WER, so at a 4.0% reference WER the 99% floor
        # permits a candidate WER of 4.96% -- a 24% RELATIVE increase in
        # transcription errors passing as "MLPerf PASS". At a 63% top-1 the
        # same rule permits a 0.63pp drop. The rule is not wrong, but a reader
        # hears "PASS" as "no degradation", so the relative error increase is
        # published next to the verdict rather than left to be derived.
        cand_err, ref_err = error_rate
        out["relative_error_increase"] = (
            (cand_err - ref_err) / ref_err if ref_err else None)
        out["error_rates"] = {"candidate": cand_err, "reference": ref_err}
    return out


# --------------------------------------------------------------- bootstrap

def paired_bootstrap_ci(units, statistic, n_boot=2000, seed=1234, alpha=0.05):
    """Percentile CI for a paired statistic, resampling UNITS (utterances or
    images) with replacement.

    `units` is a list of opaque per-sample records; `statistic(sample)` maps a
    resampled list back to a scalar. Resampling the unit -- not the arm --
    is what makes the interval paired: both engines' results for a unit move
    together, so the interval covers sampling error in WHICH utterances were
    drawn, which is the only randomness in a deterministic greedy decode.

    Deterministic: seeded `random.Random`, so the published interval is
    reproducible from the receipt.
    """
    import random
    if not units:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "seed": seed}
    rng = random.Random(seed)
    n = len(units)
    point = statistic(units)
    draws = []
    for _ in range(n_boot):
        resample = [units[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(resample))
    draws.sort()
    # Nearest-rank percentile on the sorted draws. The previous form,
    # floor(alpha/2 * B) - 1, selected the 2.45th percentile at B=2000 rather
    # than the 2.5th -- a half-percentile bias in every published interval.
    def _pct(q):
        return draws[min(n_boot - 1, max(0, int(round(q * (n_boot - 1)))))]
    lo, hi = _pct(alpha / 2), _pct(1 - alpha / 2)
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot,
            "seed": seed, "alpha": alpha}


def wer_delta_units(paired_records):
    """Build bootstrap units for a WER delta from per-utterance edit counts.

    Each unit carries BOTH engines' counts for one utterance, which is what
    keeps the resample paired.
    """
    return [{"ref_words": r["ref_words"],
             "a_err": r["a"]["sub"] + r["a"]["ins"] + r["a"]["del"],
             "b_err": r["b"]["sub"] + r["b"]["ins"] + r["b"]["del"]}
            for r in paired_records]


def wer_delta_statistic(units):
    """WER(a) - WER(b), corpus-aggregated over a resample. Positive means the
    first engine made more errors per reference word."""
    ref = sum(u["ref_words"] for u in units)
    if not ref:
        return float("nan")
    return (sum(u["a_err"] for u in units) - sum(u["b_err"] for u in units)) / ref


def topk_delta_statistic(units):
    """top1(a) - top1(b) over a resample of per-image hit indicators."""
    if not units:
        return float("nan")
    return (sum(u["a_hit"] for u in units) - sum(u["b_hit"] for u in units)) / len(units)


# ---------------------------------------------------------------- receipts

def atomic_json(path, payload):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    os.replace(tmp, path)
