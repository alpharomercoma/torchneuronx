"""LibriSpeech WER for Whisper -- the MLPerf ASR benchmark, run as a validity
check on the Neuron compile.

WHAT THIS MEASURES AND WHY
--------------------------
`whisper_lane.py` transcribes one 11-second JFK clip and reports RTF. That
number says the graph is fast. It does not say the graph is RIGHT. This lane
answers the second question with the benchmark the industry actually uses.

  benchmark   MLPerf Inference v5.1 ASR
  dataset     LibriSpeech dev-clean + dev-other ("dev-all"), MLCommons' own
              split choice -- both qualities, so a clean-only score cannot
              flatter the result
  metric      corpus Word Error Rate after MLCommons' three normalisation
              steps (see shared/eval/accuracy.mlperf_normalize)
  gate        MLPerf closed-division rule: accuracy >= 99% of reference

THE COMPARISON THAT CARRIES THE CLAIM
-------------------------------------
Not "our WER vs MLCommons' published 97.9329% word accuracy" -- that is
whisper-large-v3 on their harness, and this is whisper-small on ours; the gap
would be dominated by model size, not silicon.

The claim is a PAIRED delta on the SAME box, same samples, same preprocessing,
same decode parameters:

    engine=cpu     float32 eager HF model on the instance's vCPUs
    engine=neuron  the compiled artifact on a NeuronCore

MLPerf's >=99% rule is then applied with the CPU float32 run as the reference.
That isolates the one thing that differs -- the compile: static decoder shape,
bf16 auto-cast, no KV cache -- which is exactly the "is it cutting corners"
question. The published number is kept as a secondary anchor that catches a
broken harness.

FAIRNESS CONTROLS, ALL RECORDED IN THE RECEIPT
----------------------------------------------
  * identical sample list: seeded draw, persisted, SHA-256'd (`sample_digest`)
  * identical decode: greedy (num_beams=1, do_sample=False), same max length
  * identical audio: the same FLAC files, read the same way, 16 kHz native
  * identical scoring: shared/eval/accuracy.py, imported by both engines
  * `--cpu-nocache-probe`: Neuron whisper is forced use_cache=False while CPU
    defaults to True. KV caching is mathematically exact, but this study does
    not take "mathematically exact" on trust -- the probe re-decodes N
    utterances on CPU with use_cache=False and records whether the transcripts
    are identical, turning an assumption into evidence.
  * truncation is counted, not hidden: an utterance that hits the length cap
    without emitting EOS was cut off by OUR static shape, not by the hardware.
    `truncated` > 0 flags the run rather than quietly inflating WER.

    python3 extras/asr_wer_lane.py --engine cpu    --out .../asr_cpu.json
    python3 extras/asr_wer_lane.py --engine neuron --out .../asr_neuron.json
    python3 extras/asr_wer_lane.py --compare .../asr_neuron.json .../asr_cpu.json \
                                   --out .../asr_delta.json
"""
import argparse
import json
import os
import random
import sys
import tarfile
import time
import traceback
from urllib.request import urlretrieve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "shared", "eval"))
import accuracy as A  # noqa: E402

MODEL_ID = "openai/whisper-small"
# Canonical LibriSpeech distribution. Deliberately NOT the HuggingFace mirror:
# openslr.org is the source of record, needs no auth, and its on-disk layout
# gives real utterance IDs (1272-128104-0000) that make the sample list
# auditable by hand.
OPENSLR = "https://www.openslr.org/resources/12/{}.tar.gz"
# MLCommons combine dev-clean and dev-other into "dev-all" so the score spans
# a range of audio quality.
MLPERF_SPLITS = ("dev-clean", "dev-other")
# TOTAL decoder length -- the traced graph's static axis, and the identical
# generate(max_length=...) cap on CPU. It must be max_LENGTH on both arms, not
# max_length on one and max_new_tokens on the other: Whisper prepends a 4-token
# forced prefix, so max_new_tokens=224 grants CPU 228 total while max_length=224
# grants Neuron 220. Adversarial review caught that asymmetry before it ran; it
# would have charged 4 tokens of headroom to the accelerator.
# LibriSpeech dev utterances top out near 33 s (~130 tokens); 224 leaves
# headroom while staying well under Whisper's 448 limit, so a truncation is a
# real signal rather than a chosen ceiling.
MAX_LENGTH = 224
# MLCommons' published whisper-large-v3 reference, kept for context only.
MLPERF_REFERENCE = {
    "model": "openai/whisper-large-v3",
    "dataset": "LibriSpeech dev-all (dev-clean + dev-other)",
    "word_accuracy": 0.979329,
    "wer": 1.0 - 0.979329,
    "numeric_format": "bfloat16",
    "source": "https://mlcommons.org/2025/09/whisper-inferencev5-1/",
    "note": "different model size and harness -- context, not a target",
}
# Whisper paper (Radford et al. 2022) Table 2, greedy, LibriSpeech test-clean.
PUBLISHED_ANCHOR = {"openai/whisper-small": {"test_clean_wer": 0.034}}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["cpu", "neuron"], default="cpu")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--splits", default=",".join(MLPERF_SPLITS))
    ap.add_argument("--n-per-split", type=int, default=250,
                    help="0 = the whole split (dev-all is 5567 utterances)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH,
                    help="TOTAL decoder length (prefix included) on BOTH arms")
    ap.add_argument("--data-dir", default="/opt/np/models/librispeech")
    ap.add_argument("--compiled-dir", default=None,
                    help="default is keyed by model AND sequence length; an "
                         "unkeyed path lets a 192-token trace be reloaded for "
                         "a 224-token run")
    ap.add_argument("--normalize", choices=list(A.RECIPES), default="whisper",
                    help="primary scoring recipe; ALL recipes are reported")
    ap.add_argument("--cpu-nocache-probe", type=int, default=8,
                    help="CPU only: re-decode N utterances with use_cache=False "
                         "and record whether transcripts match")
    ap.add_argument("--compare", nargs=2, metavar=("CANDIDATE", "REFERENCE"),
                    help="score a paired delta from two finished receipts")
    ap.add_argument("--progress-every", type=int, default=25)
    return ap


# ----------------------------------------------------------------- dataset

def ensure_split(data_dir, split):
    """Download + extract one LibriSpeech split. Idempotent."""
    os.makedirs(data_dir, exist_ok=True)
    root = os.path.join(data_dir, "LibriSpeech", split)
    if os.path.isdir(root):
        return root
    tgz = os.path.join(data_dir, f"{split}.tar.gz")
    if not os.path.exists(tgz):
        print(f"downloading LibriSpeech {split} -> {tgz}", flush=True)
        urlretrieve(OPENSLR.format(split), tgz)
    print(f"extracting {tgz}", flush=True)
    with tarfile.open(tgz) as tf:
        # OpenSLR is trusted and HTTPS, but an archive member is still untrusted
        # input: refuse anything that would land outside data_dir. `filter=`
        # arrived in 3.12; the explicit check keeps older interpreters safe too.
        # realpath, not abspath: a symlink already sitting under data_dir
        # would otherwise let a member named "link/x" resolve inside the root
        # on paper and land outside it on disk.
        base = os.path.realpath(data_dir)
        for m in tf.getmembers():
            dest = os.path.realpath(os.path.join(base, m.name))
            if dest != base and not dest.startswith(base + os.sep):
                raise RuntimeError(f"unsafe tar member escapes data_dir: {m.name!r}")
            if m.issym() or m.islnk():
                raise RuntimeError(f"unsafe tar member is a link: {m.name!r}")
        tf.extractall(data_dir)
    return root


def index_split(root, split):
    """Every utterance in a split as {id, flac, text, split}, sorted by id.

    Sorted so the enumeration order is a property of the corpus, not of the
    filesystem -- two boxes with different directory orders must draw the same
    seeded subset.
    """
    items = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".trans.txt"):
                continue
            with open(os.path.join(dirpath, name)) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, text = line.split(" ", 1)
                    flac = os.path.join(dirpath, utt_id + ".flac")
                    if os.path.exists(flac):
                        items.append({"id": utt_id, "flac": flac,
                                      "text": text, "split": split})
    items.sort(key=lambda r: r["id"])
    return items


def select_samples(data_dir, splits, n_per_split, seed):
    """The seeded, per-split-stratified sample list.

    Stratified rather than pooled: dev-clean and dev-other differ in
    difficulty, and a pooled draw would let the harder half drift between
    runs. Fixed `seed` + sorted index means this function is a pure map from
    (splits, n, seed) to a list of utterance IDs.
    """
    chosen = []
    for split in splits:
        root = ensure_split(data_dir, split)
        items = index_split(root, split)
        if not items:
            raise SystemExit(f"no utterances found under {root}")
        if n_per_split and n_per_split < len(items):
            rng = random.Random(f"{seed}:{split}")
            chosen.extend(sorted(rng.sample(items, n_per_split),
                                 key=lambda r: r["id"]))
        else:
            chosen.extend(items)
    return chosen


def read_flac(path):
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz, got {sr} Hz ({path})")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, len(audio) / 16000.0


# ------------------------------------------------------------------ engines

def load_cpu(args):
    """Stock float32 eager HF Whisper -- the reference arm."""
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32).eval()
    meta = {"dtype": "float32", "framework": "transformers eager",
            "threads": torch.get_num_threads()}

    def transcribe(features, use_cache=True):
        with torch.no_grad():
            ids = model.generate(features, max_length=args.max_length,
                                 num_beams=1, do_sample=False,
                                 use_cache=use_cache,
                                 **FORCED)
        return ids
    return processor, transcribe, meta


def load_neuron(args):
    """The compiled artifact -- the arm under test.

    Compiled at sequence_length == --max-new-tokens so the static decoder
    length and the CPU cap are the SAME number. A traced graph shorter than
    the control's cap would truncate only one arm and manufacture a WER gap
    that had nothing to do with the hardware.
    """
    from transformers import AutoProcessor
    from optimum.neuron import NeuronWhisperForConditionalGeneration
    # Keyed by model AND length. The old unkeyed default let a run compiled at
    # 192 be silently reloaded by a run asking for 224: generate() would fall
    # back to the SAVED generation_config's shorter max_length, truncate the
    # tail of every long utterance, and charge the resulting WER to the chip.
    compiled = os.path.abspath(args.compiled_dir or os.path.join(
        "/opt/np/models/neuron-compiled",
        "whisper-wer-" + args.model.split("/")[-1] + f"-L{args.max_length}"))
    cached = os.path.exists(os.path.join(compiled, "config.json"))
    compile_s = 0.0
    if not cached:
        print(f"--- compiling {args.model} seq={args.max_length} "
              f"-> {compiled} ---", flush=True)
        t0 = time.perf_counter()
        model = NeuronWhisperForConditionalGeneration.from_pretrained(
            args.model, export=True, inline_weights_to_neff=False,
            auto_cast="all", auto_cast_type="bf16",
            batch_size=1, sequence_length=args.max_length)
        model.save_pretrained(compiled)
        AutoProcessor.from_pretrained(args.model).save_pretrained(compiled)
        del model
        compile_s = time.perf_counter() - t0
    model = NeuronWhisperForConditionalGeneration.from_pretrained(compiled)
    # The processor comes from the HUB SNAPSHOT, not from the compiled dir.
    # Both arms must feed the model log-mel features produced by the same
    # code: a processor pickled at compile time under an older transformers
    # would apply a different mel filterbank or padding rule, and the delta
    # would absorb a library-version skew as if it were silicon.
    processor = AutoProcessor.from_pretrained(args.model)
    meta = {"dtype": "bfloat16 (auto_cast=all)", "framework": "optimum-neuron",
            "static_decoder_seq_len": args.max_length,
            "use_cache": False, "compile_s": round(compile_s, 1),
            "compiled_from_cache": cached, "compiled_dir": compiled}

    def transcribe(features, use_cache=True):
        # The traced graph's decoder axis IS max_length, and the CPU arm is
        # capped at the same total -- one number, both arms.
        return model.generate(features, max_length=args.max_length,
                              num_beams=1, do_sample=False, **FORCED)
    return processor, transcribe, meta


# Forced decoder prompt, identical on both engines. LibriSpeech is English
# read speech; leaving language detection on would let the two arms take
# different decoder prefixes on a borderline utterance and charge the
# difference to the hardware. Populated in main() once we know the engine
# accepts it.
FORCED = {}


# -------------------------------------------------------------------- run

def run_engine(args):
    samples = select_samples(args.data_dir, args.splits.split(","),
                             args.n_per_split, args.seed)
    ids = [s["id"] for s in samples]
    digest = A.sample_digest(ids)
    print(f"# asr_wer engine={args.engine} model={args.model} "
          f"n={len(samples)} splits={args.splits} seed={args.seed} "
          f"max_length={args.max_length} digest={digest[:16]}",
          flush=True)

    processor, transcribe, engine_meta = (
        load_cpu(args) if args.engine == "cpu" else load_neuron(args))

    global FORCED
    forced_mode = "language=en,task=transcribe"
    FORCED = {"language": "en", "task": "transcribe"}
    probe_audio, _ = read_flac(samples[0]["flac"])
    probe_feats = processor(probe_audio, sampling_rate=16000,
                            return_tensors="pt").input_features
    # A kwarg that RAISES is easy. The dangerous case, found by adversarial
    # review, is a traced generate() that silently DROPS language=/task= and
    # falls back to language autodetect: nothing raises, both receipts claim
    # "language=en", and one arm quietly decodes a noisy dev-other utterance
    # as German. So the probe reads back the tokens actually emitted and
    # records the real decoder prefix, which compare() then requires to match.
    try:
        probe_ids = transcribe(probe_feats)
    except Exception as exc:
        print(f"forced decoder ids rejected ({type(exc).__name__}: {exc}); "
              "falling back to autodetect", flush=True)
        FORCED = {}
        forced_mode = "autodetect (forced ids rejected by this engine)"
        probe_ids = transcribe(probe_feats)
    decoder_prefix = describe_prefix(processor, probe_ids)
    print(f"decoder prefix actually emitted: {decoder_prefix}", flush=True)

    records, pairs, wall = [], [], 0.0
    truncated = 0
    audio_s_total = 0.0
    t_start = time.perf_counter()
    for i, s in enumerate(samples):
        audio, dur = read_flac(s["flac"])
        audio_s_total += dur
        feats = processor(audio, sampling_rate=16000,
                          return_tensors="pt").input_features
        t0 = time.perf_counter()
        ids_out = transcribe(feats)
        wall += time.perf_counter() - t0
        text = processor.batch_decode(ids_out, skip_special_tokens=True)[0].strip()
        # TRUNCATION, third attempt -- the first two were both wrong and the
        # pilot is what caught the second.
        #   v1  n_tok >= max_length          -> flags a naturally finished
        #       utterance whose 4-token prefix pushes it to the cap, and flags
        #       100% of a static graph's full-buffer returns. (adversarial
        #       review caught this before it ran)
        #   v2  EOS absent from the ids       -> transformers' Whisper generate
        #       STRIPS the trailing EOS on short-form output, so every
        #       utterance looked truncated: measured 8/8 on BOTH engines with
        #       buffers of 6, 17, 27 and 51 tokens against a cap of 224.
        #   v3  EOS present, OR the sequence stopped short of the cap.
        # Both clauses are needed: the length clause covers EOS-stripping, and
        # the EOS clause covers a traced graph that returns its full static
        # buffer padded with 50257 (Whisper's pad token IS its EOS token).
        toks = ids_out[0].tolist()
        eos = getattr(processor.tokenizer, "eos_token_id", None)
        stopped_early = len(toks) < args.max_length
        has_eos = eos is not None and eos in toks
        finished = has_eos or stopped_early
        n_tok = toks.index(eos) + 1 if has_eos else len(toks)
        if not finished:
            truncated += 1
        records.append({"id": s["id"], "split": s["split"], "audio_s": dur,
                        "gen_tokens": n_tok, "buffer_tokens": len(toks),
                        "finished": finished, "hyp_raw": text,
                        "ref_raw": s["text"]})
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"  {i+1}/{len(samples)}  {wall:.1f}s decode", flush=True)
    total_wall = time.perf_counter() - t_start

    # Score under EVERY recipe, publish all of them, and name which is
    # primary. Picking one silently is picking the flattering one, and the
    # three differ by more than rounding -- the literal MLCommons reading
    # deletes every number from both sides (see accuracy.normalize).
    norm = make_normalizer()
    all_scores = {}
    for recipe in A.RECIPES:
        rp = [(A.normalize(r["ref_raw"], norm.fn, recipe),
               A.normalize(r["hyp_raw"], norm.fn, recipe)) for r in records]
        sc = A.corpus_wer(rp)
        per = sc.pop("per_utterance")
        if recipe == args.normalize:
            scored, pairs = sc, rp
            for r, u, (ref_t, hyp_t) in zip(records, per, rp):
                r["counts"], r["ref"], r["hyp"] = u, ref_t, hyp_t
        all_scores[recipe] = {"wer": sc["wer"], "word_accuracy": sc["word_accuracy"],
                              "errors": sc["errors"], "ref_words": sc["ref_words"]}

    payload = {
        "lane": "asr_wer", "engine": args.engine, "model": args.model,
        "benchmark": "MLPerf Inference v5.1 ASR (LibriSpeech dev-all)",
        "dataset": {"source": "openslr.org/resources/12",
                    "splits": args.splits, "n_per_split": args.n_per_split,
                    "seed": args.seed, "audio_s_total": round(audio_s_total, 1)},
        "n_samples": len(samples), "sample_digest": digest,
        "decode": {"strategy": "greedy", "num_beams": 1, "do_sample": False,
                   "max_length": args.max_length,
                   "forced_decoder": forced_mode,
                   "decoder_prefix": decoder_prefix},
        "normalize": {
            "primary": args.normalize,
            "english_normalizer": ("OpenAI EnglishTextNormalizer" if norm.available
                                   else "UNAVAILABLE -- not substituted"),
            "english_normalizer_reason": norm.reason,
            "recipes": {
                "whisper": "EnglishTextNormalizer only -- the protocol the "
                           "published Whisper WER numbers use, so the only one "
                           "under which our absolute number is comparable",
                "mlperf": "MLCommons' three steps, DIGITS KEPT (documented "
                          "deviation: the literal alphabetic-only filter "
                          "deletes every normalised number from both sides)",
                "mlperf_literal": "MLCommons step 1 taken at its word -- shown "
                                  "only to expose what the literal reading costs",
            },
        },
        "wer_by_recipe": all_scores,
        "engine_meta": engine_meta,
        "wer": scored["wer"], "word_accuracy": scored["word_accuracy"],
        "counts": {k: scored[k] for k in
                   ("errors", "sub", "ins", "del", "ref_words", "hyp_words")},
        "truncated": truncated,
        "n_unfinished": truncated,
        "timing": {"decode_wall_s": round(wall, 2),
                   "total_wall_s": round(total_wall, 2),
                   "rtf": round(wall / audio_s_total, 4) if audio_s_total else None,
                   "s_per_utterance": round(wall / len(samples), 3)},
        "mlperf_reference_context": MLPERF_REFERENCE,
        "published_anchor": PUBLISHED_ANCHOR.get(args.model),
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.engine == "cpu" and args.cpu_nocache_probe:
        payload["kv_cache_equivalence_probe"] = nocache_probe(
            args, samples, processor, transcribe, records)

    A.atomic_json(args.out, payload)
    A.atomic_json(args.out.replace(".json", ".utterances.json"),
                  {"sample_digest": digest, "records": records})
    print(f"\nWER {scored['wer']*100:.3f}%  word_accuracy "
          f"{scored['word_accuracy']*100:.3f}%  "
          f"({scored['errors']} errors / {scored['ref_words']} ref words, "
          f"{truncated} truncated)  ->  {args.out}", flush=True)
    return 0


def nocache_probe(args, samples, processor, transcribe, records):
    """Evidence, not assumption, that KV caching does not change the answer.

    Neuron's Whisper runs use_cache=False; CPU defaults to True. Caching is an
    exact algebraic rewrite, so the transcripts SHOULD be identical -- this
    re-decodes a handful without the cache and reports whether they were.
    """
    n = min(args.cpu_nocache_probe, len(samples))
    mismatches = []
    for s, r in zip(samples[:n], records[:n]):
        audio, _ = read_flac(s["flac"])
        feats = processor(audio, sampling_rate=16000,
                          return_tensors="pt").input_features
        ids_out = transcribe(feats, use_cache=False)
        text = processor.batch_decode(ids_out, skip_special_tokens=True)[0].strip()
        if text != r["hyp_raw"]:
            mismatches.append({"id": s["id"], "cached": r["hyp_raw"],
                               "uncached": text})
    return {"n_probed": n, "identical": len(mismatches) == 0,
            "mismatches": mismatches,
            "why": "Neuron Whisper forces use_cache=False; CPU defaults True. "
                   "This checks the two decode paths agree before the delta "
                   "attributes any difference to the hardware."}


def describe_prefix(processor, ids, n=5):
    """The decoder prefix ACTUALLY emitted, as token strings.

    This is the evidence that both arms took the same decoding path. A traced
    generate() that silently drops language="en" produces a different prefix
    (<|de|> instead of <|en|>, or a timestamp token instead of
    <|notimestamps|>) while raising nothing at all, so the receipt records
    what came back rather than what was asked for.
    """
    try:
        toks = ids[0].tolist()[:n]
        return " ".join(processor.tokenizer.convert_ids_to_tokens(toks))
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


class _Norm:
    __slots__ = ("fn", "available", "reason")


def make_normalizer():
    """OpenAI's EnglishTextNormalizer, or a recorded absence.

    Never silently substituted. A different normaliser produces a different
    WER, and an unrecorded substitution is how two numbers stop being
    comparable without anyone noticing.
    """
    n = _Norm()
    try:
        from transformers.models.whisper.english_normalizer import \
            EnglishTextNormalizer
        n.fn = EnglishTextNormalizer({})
        n.available = True
        n.reason = None
    except Exception as exc:
        n.fn = None
        n.available = False
        n.reason = f"{type(exc).__name__}: {exc}"
    return n


# ---------------------------------------------------------------- compare

def compare(args):
    """Paired delta + MLPerf gate from two finished receipts."""
    cand_p, ref_p = args.compare
    cand = json.load(open(cand_p))
    ref = json.load(open(ref_p))
    A.assert_paired(cand, ref)
    # The engine is the treatment; EVERYTHING else is a control. Without this
    # guard compare() would subtract a 192-token run from a 224-token one, or
    # an autodetect run from a forced-English one, and print the difference as
    # a hardware effect.
    A.assert_same_protocol(cand, ref)
    cu = json.load(open(cand_p.replace(".json", ".utterances.json")))["records"]
    ru = json.load(open(ref_p.replace(".json", ".utterances.json")))["records"]
    by_id = {r["id"]: r for r in ru}
    paired = [{"ref_words": c["counts"]["ref_words"],
               "a": c["counts"], "b": by_id[c["id"]]["counts"]}
              for c in cu if c["id"] in by_id]
    if len(paired) != len(cu):
        raise ValueError(f"utterance files disagree: {len(paired)} of {len(cu)} matched")
    units = A.wer_delta_units(paired)
    ci = A.paired_bootstrap_ci(units, A.wer_delta_statistic)
    gate = A.accuracy_gate(cand["word_accuracy"], ref["word_accuracy"],
                           error_rate=(cand["wer"], ref["wer"]))
    # The MORE SENSITIVE detector, and the one the CPU-only KV-cache probe
    # cannot provide: how many utterances the two engines transcribed
    # DIFFERENTLY AT ALL. WER can cancel -- one engine fixing an error while
    # breaking another nets to zero -- but a graph whose numerics have drifted
    # shows up here immediately, and this compares the two engines to each
    # other rather than one engine to itself.
    disagree = [c["id"] for c in cu
                if c["hyp_raw"] != by_id[c["id"]]["hyp_raw"]]
    unfinished = {"candidate": cand.get("truncated"), "reference": ref.get("truncated")}
    payload = {
        "lane": "asr_wer_delta",
        "candidate": {"engine": cand["engine"], "wer": cand["wer"],
                      "word_accuracy": cand["word_accuracy"],
                      "truncated": cand["truncated"], "receipt": cand_p},
        "reference": {"engine": ref["engine"], "wer": ref["wer"],
                      "word_accuracy": ref["word_accuracy"],
                      "truncated": ref["truncated"], "receipt": ref_p},
        "n_samples": cand["n_samples"], "sample_digest": cand["sample_digest"],
        "wer_delta_pp": (cand["wer"] - ref["wer"]) * 100,
        "wer_delta_ci95_pp": {"lo": ci["lo"] * 100, "hi": ci["hi"] * 100,
                              "point": ci["point"] * 100,
                              "n_boot": ci["n_boot"], "seed": ci["seed"]},
        "transcript_disagreements": len(disagree),
        "transcript_disagreement_rate": len(disagree) / len(cu) if cu else None,
        "transcript_disagreement_ids": disagree[:50],
        "unfinished_utterances": unfinished,
        "wer_by_recipe": {"candidate": cand.get("wer_by_recipe"),
                          "reference": ref.get("wer_by_recipe")},
        "mlperf_gate": gate,
        "interpretation": (
            "wer_delta_pp > 0 means the candidate engine made MORE errors per "
            "reference word than the same model on the same samples in float32 "
            "on CPU. A CI straddling zero means this run cannot distinguish the "
            "two, at this sample size."),
        "caveats": [
            "transcript_disagreements == 0 is the strong result: byte-identical "
            "transcripts mean the compile changed nothing at all. A nonzero WER "
            "delta with ZERO disagreements is impossible and would indicate a "
            "harness bug, not a hardware one.",
            "Both receipts must carry truncated == 0; a truncated utterance is "
            "a static-shape artefact of this harness, not a hardware property.",
            "Absolute WER is not comparable to MLCommons' published figure: "
            "different model size (small vs large-v3) and harness.",
        ],
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    A.atomic_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in
                      ("wer_delta_pp", "wer_delta_ci95_pp", "mlperf_gate")},
                     indent=1))
    return 0


def _record_failure(out, stage):
    reason = traceback.format_exc().strip().splitlines()[-1]
    path = out[:-5] + ".failure.json" if out.endswith(".json") else out + ".failure.json"
    A.atomic_json(path, {"lane": "asr_wer", "status": "lane_failed",
                         "stage": stage, "reason": reason,
                         "traceback": traceback.format_exc(),
                         "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime())})
    print(f"asr_wer FAILED at {stage}: {reason}", file=sys.stderr)
    return 1


def main():
    args = build_parser().parse_args()
    if args.compare:
        try:
            return compare(args)
        except Exception:
            return _record_failure(args.out, "compare")
    try:
        return run_engine(args)
    except Exception:
        return _record_failure(args.out, "run")


if __name__ == "__main__":
    sys.exit(main())
