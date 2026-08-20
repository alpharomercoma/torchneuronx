"""Accuracy-validity track: the metric and the fairness invariants.

These tests exist because the accuracy numbers are the study's ONLY defence
against a fast-but-wrong compile, so a silent bug here would be worse than
having no accuracy lane at all -- it would launder a broken graph as verified.

Pure functions only; no torch, no transformers, no network.

    uv run --with pytest python -m pytest tests/test_accuracy.py -q
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "shared", "eval"))
sys.path.insert(0, os.path.join(ROOT, "extras"))

import accuracy as A                 # noqa: E402
import asr_wer_lane as ASR           # noqa: E402
import zeroshot_imagenet_lane as ZS  # noqa: E402


# ------------------------------------------------------------ normalisation
def test_basic_normalize_matches_mlperf_step1():
    assert A.basic_normalize("Hello, World!") == "hello world"
    # "retain only alphabetic characters" -- digits go, symmetrically
    assert A.basic_normalize("in 1990 we") == "in we"
    assert A.basic_normalize("  A   B  ") == "a b"


def test_compound_split_runs_before_lowercasing():
    # If step 3 ran after step 1 the hyphen would already be a space and the
    # step would be a no-op; this pins the order.
    assert A.split_compounds("world-wide web") == "world wide web"
    assert A.mlperf_normalize("World-Wide") == "world wide"


class _FakeEnglishNormalizer:
    """Stands in for OpenAI's EnglishTextNormalizer: its defining behaviour
    for our purposes is that it turns spoken numbers into DIGITS."""

    def __call__(self, text):
        return (text.lower()
                .replace("nineteen hundred and ten", "1910")
                .replace("nineteen hundred and twenty", "1920"))


def test_literal_mlperf_step1_deletes_every_number_from_both_sides():
    # THE DEFECT adversarial review found, pinned so it cannot come back.
    # EnglishTextNormalizer maps spoken numbers to digits; an alphabetic-only
    # filter applied after it deletes them from reference AND hypothesis, so a
    # wrong year scores zero errors and a hallucinating model is rewarded.
    n = _FakeEnglishNormalizer()
    ref = "THE CHAPTER WAS HEADED NINETEEN HUNDRED AND TEN"
    wrong_year = "the chapter was headed 1920"
    lit = [A.normalize(t, n, "mlperf_literal") for t in (ref, wrong_year)]
    assert lit[0] == lit[1]                                  # indistinguishable
    assert A.corpus_wer([tuple(lit)])["errors"] == 0         # a free pass

    # The shipped recipes keep the number and score the error.
    for recipe in ("whisper", "mlperf"):
        pair = tuple(A.normalize(t, n, recipe) for t in (ref, wrong_year))
        assert pair[0] != pair[1], recipe
        assert A.corpus_wer([pair])["errors"] >= 1, recipe


def test_mlperf_recipe_keeps_digits_but_drops_punctuation():
    assert A.strip_punctuation("Room 101, please!") == "room 101 please"
    assert A.basic_normalize("Room 101") == "room"           # the literal one


def test_whisper_recipe_is_the_default_and_is_the_published_protocol():
    args = ASR.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.normalize == "whisper"
    assert set(A.RECIPES) == {"whisper", "mlperf", "mlperf_literal"}


def test_unknown_recipe_raises_rather_than_falling_back():
    with pytest.raises(ValueError):
        A.normalize("x", None, "made_up")


def test_normalizer_is_idempotent():
    once = A.mlperf_normalize("The Quick, Brown-Fox!")
    assert A.mlperf_normalize(once) == once


def test_english_normalizer_is_never_silently_substituted():
    n = ASR.make_normalizer()
    # Either it is genuinely available, or its absence is RECORDED with a
    # reason. What must never happen is a different normaliser standing in.
    assert n.available is (n.fn is not None)
    if not n.available:
        assert n.reason and ":" in n.reason


# --------------------------------------------------------------------- WER
def test_edit_counts_each_error_type():
    assert A.edit_counts("a b c".split(), "a b c".split()) == (0, 0, 0)
    assert A.edit_counts("a b c".split(), "a x c".split()) == (1, 0, 0)   # sub
    assert A.edit_counts("a b c".split(), "a b c d".split()) == (0, 1, 0)  # ins
    assert A.edit_counts("a b c".split(), "a c".split()) == (0, 0, 1)      # del


def test_edit_counts_is_minimal_not_greedy():
    # Greedy left-to-right alignment would call this 3 substitutions.
    sub, ins, dele = A.edit_counts("a b c d".split(), "x a b c d".split())
    assert (sub, ins, dele) == (0, 1, 0)


def test_wer_is_corpus_aggregated_not_mean_of_ratios():
    # 1 error in a 3-word utterance, 1 error in a 50-word utterance.
    short = ("a b c", "a x c")
    long_ref = " ".join(f"w{i}" for i in range(50))
    long_hyp = " ".join(["zz"] + [f"w{i}" for i in range(1, 50)])
    got = A.corpus_wer([short, (long_ref, long_hyp)])
    assert got["wer"] == pytest.approx(2 / 53)          # corpus: 2 errors / 53
    mean_of_ratios = (1 / 3 + 1 / 50) / 2               # ~0.177 -- 4.7x larger
    assert got["wer"] < mean_of_ratios / 4
    assert got["word_accuracy"] == pytest.approx(1 - 2 / 53)


def test_wer_can_exceed_one():
    # A runaway hallucination inserts more words than the reference has.
    got = A.corpus_wer([("a", "a b c d e")])
    assert got["wer"] == pytest.approx(4.0)
    assert got["word_accuracy"] == pytest.approx(-3.0)


def test_corpus_wer_reports_per_utterance_counts_for_pairing():
    got = A.corpus_wer([("a b", "a x"), ("c", "c")])
    assert [p["ref_words"] for p in got["per_utterance"]] == [2, 1]
    assert got["per_utterance"][0]["sub"] == 1


# ---------------------------------------------------------------- top-k
def test_topk_basic():
    rows = [[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]
    got = A.topk_accuracy(rows, [1, 0], ks=(1, 2))
    assert got["top1"] == 1.0 and got["top2"] == 1.0


def test_nan_row_is_a_miss_and_is_counted():
    # The trn1 CLIP failure: the graph ran at full speed and emitted NaN.
    # It must score as WRONG, and it must be visible in the receipt.
    nan = float("nan")
    got = A.topk_accuracy([[nan, nan, nan], [0.0, 1.0, 0.0]], [0, 1])
    assert got["nonfinite_rows"] == 1
    assert got["top1"] == 0.5


def test_inf_row_is_also_rejected():
    got = A.topk_accuracy([[float("inf"), 0.0]], [0])
    assert got["nonfinite_rows"] == 1 and got["top1"] == 0.0


def test_topk_ties_break_by_lowest_index_deterministically():
    got = A.topk_accuracy([[1.0, 1.0]], [1], ks=(1,))
    assert got["top1"] == 0.0          # index 0 wins the tie, label 1 misses


# ------------------------------------------------------- pairing guarantees
def test_sample_digest_is_order_sensitive():
    assert A.sample_digest(["a", "b"]) != A.sample_digest(["b", "a"])
    assert A.sample_digest(["a", "b"]) == A.sample_digest(["a", "b"])


def test_sample_digest_has_no_delimiter_collision():
    # Without the NUL separator these two lists would hash identically.
    assert A.sample_digest(["ab", "c"]) != A.sample_digest(["a", "bc"])


def test_assert_paired_refuses_mismatched_receipts():
    a = {"sample_digest": "x" * 64, "n_samples": 10}
    assert A.assert_paired(a, dict(a)) is True
    with pytest.raises(ValueError):
        A.assert_paired(a, {"sample_digest": "y" * 64, "n_samples": 10})
    with pytest.raises(ValueError):
        A.assert_paired(a, {"sample_digest": "x" * 64, "n_samples": 9})


def test_assert_paired_refuses_receipts_with_no_digest():
    with pytest.raises(ValueError):
        A.assert_paired({"sample_digest": None, "n_samples": 1},
                        {"sample_digest": None, "n_samples": 1})


# ---------------------------------------------------------------- the gate
def test_mlperf_gate_boundary_is_inclusive_at_99pct():
    assert A.accuracy_gate(0.99, 1.0)["passed"] is True
    assert A.accuracy_gate(0.9899, 1.0)["passed"] is False
    assert A.MLPERF_ACCURACY_FLOOR == 0.99


def test_gate_reports_unknown_rather_than_passing_without_a_reference():
    for bad in (None, 0.0, -1.0, float("nan")):
        assert A.accuracy_gate(0.9, bad)["passed"] is None


def test_gate_fails_a_nonfinite_candidate():
    assert A.accuracy_gate(float("nan"), 0.9)["passed"] is False


# ------------------------------------------------------------- delta + CI
def test_wer_delta_sign_convention():
    units = [{"ref_words": 10, "a_err": 3, "b_err": 1}]
    assert A.wer_delta_statistic(units) == pytest.approx(0.2)   # a is worse


def test_topk_delta_sign_convention():
    units = [{"a_hit": 0, "b_hit": 1}, {"a_hit": 1, "b_hit": 1}]
    assert A.topk_delta_statistic(units) == pytest.approx(-0.5)  # a is worse


def test_bootstrap_is_deterministic_and_brackets_the_point():
    units = [{"ref_words": 10, "a_err": i % 3, "b_err": i % 2} for i in range(60)]
    one = A.paired_bootstrap_ci(units, A.wer_delta_statistic, n_boot=200, seed=7)
    two = A.paired_bootstrap_ci(units, A.wer_delta_statistic, n_boot=200, seed=7)
    assert one == two
    assert one["lo"] <= one["point"] <= one["hi"]
    assert A.paired_bootstrap_ci(units, A.wer_delta_statistic, n_boot=200,
                                 seed=8) != one


def test_bootstrap_ci_is_zero_width_when_engines_are_identical():
    units = [{"ref_words": 10, "a_err": 2, "b_err": 2} for _ in range(30)]
    ci = A.paired_bootstrap_ci(units, A.wer_delta_statistic, n_boot=200, seed=1)
    assert ci["lo"] == ci["hi"] == 0.0


def test_bootstrap_handles_empty_units():
    ci = A.paired_bootstrap_ci([], A.wer_delta_statistic)
    assert ci["n_boot"] == 0


# ------------------------------------------------- lane-level determinism
def test_asr_static_length_matches_the_cpu_cap():
    # The traced decoder axis and the CPU max_new_tokens are ONE number. If
    # they diverged, only one arm would truncate and the WER gap would be an
    # artefact of the harness.
    args = ASR.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.max_length == ASR.MAX_LENGTH == 224
    assert ASR.MAX_LENGTH < 448              # Whisper's own ceiling


def test_asr_uses_mlperf_dev_all_not_just_dev_clean():
    args = ASR.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert set(args.splits.split(",")) == {"dev-clean", "dev-other"}


def test_asr_defaults_to_greedy_paired_decode():
    args = ASR.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.cpu_nocache_probe > 0        # the KV-cache equivalence check


def test_published_reference_is_context_not_a_target():
    assert ASR.MLPERF_REFERENCE["model"] != ASR.MODEL_ID
    assert "context, not a target" in ASR.MLPERF_REFERENCE["note"]


def test_zeroshot_reads_all_shards_because_they_are_class_ordered():
    # Verified against the real tarball: shard 0 opens with consecutive
    # class-0 images. A prefix of shards is a few hundred classes, not
    # ImageNet.
    assert ZS.N_SHARDS == 7 and ZS.N_CLASSES == 1000


def test_zeroshot_defaults_to_fp32_cast():
    args = ZS.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.auto_cast == "none"          # bf16 measured to produce NaN


def test_zeroshot_uses_the_dataset_repos_own_prompts():
    src = open(os.path.join(ROOT, "extras", "zeroshot_imagenet_lane.py")).read()
    assert "classnames.txt" in src
    assert "zeroshot_classification_templates.txt" in src


def test_zeroshot_image_selection_is_seeded_and_stratified(tmp_path):
    root = tmp_path / "images"
    for c in range(3):
        d = root / f"{c:04d}"
        d.mkdir(parents=True)
        for i in range(8):
            (d / f"s{c}{i}.jpg").write_bytes(b"x")
    orig = ZS.N_CLASSES
    try:
        ZS.N_CLASSES = 3
        a = ZS.select_images(str(root), 4, seed=1234)
        b = ZS.select_images(str(root), 4, seed=1234)
        c = ZS.select_images(str(root), 4, seed=99)
        assert [r["key"] for r in a] == [r["key"] for r in b]     # reproducible
        assert [r["key"] for r in a] != [r["key"] for r in c]     # seed matters
        assert len(a) == 12                                       # 4 per class
        assert sorted(r["label"] for r in a) == [0]*4 + [1]*4 + [2]*4
    finally:
        ZS.N_CLASSES = orig


def test_zeroshot_selection_fails_loudly_on_an_incomplete_dataset(tmp_path):
    (tmp_path / "images" / "0000").mkdir(parents=True)
    orig = ZS.N_CLASSES
    try:
        ZS.N_CLASSES = 2
        with pytest.raises(SystemExit):
            ZS.select_images(str(tmp_path / "images"), 1, seed=1)
    finally:
        ZS.N_CLASSES = orig


# --------------------------------------------------------- driver hygiene
def test_inf2_extras_no_longer_points_lanes_at_the_vllm_venv():
    src = open(os.path.join(ROOT, "extras", "run_extras_inf2.sh")).read()
    assert 'NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"' in src
    # the serve lane must still get the vLLM venv, explicitly
    assert 'NP_VENV="$SERVE_VENV"' in src
    assert "optimum-neuron must never be installed INTO the vLLM venv" in src


def test_drivers_are_syntactically_valid():
    for sh in ("run_extras_inf2.sh", "run_accuracy.sh"):
        subprocess.run(["bash", "-n", os.path.join(ROOT, "extras", sh)],
                       check=True)


def test_accuracy_driver_runs_the_cpu_reference_arm_first():
    src = open(os.path.join(ROOT, "extras", "run_accuracy.sh")).read()
    assert "for eng in cpu neuron" in src
    assert "preflight" in src


# =====================================================================
# Regressions for defects found by adversarial review (agy + kiro),
# 2026-08-20, BEFORE any accelerator time was spent. Each test names the
# wrong number the defect would have produced.
# =====================================================================

def test_dead_graph_emitting_constant_logits_scores_zero_not_100pct():
    # Tie-breaking by lowest index handed class 0 a free win on an all-zero
    # row: a compile emitting constant logits would have reported 100% top-1
    # on class 0 and 100% top-5 on classes 0-4 -- on a class-stratified
    # sample, a plausible-looking accuracy from a completely dead model.
    dead = [[0.0] * 5 for _ in range(5)]
    got = A.topk_accuracy(dead, [0, 1, 2, 3, 4], ks=(1, 5))
    assert got["top1"] == 0.0 and got["top5"] == 0.0
    assert got["degenerate_rows"] == 5


def test_degenerate_and_nonfinite_are_counted_separately():
    got = A.topk_accuracy([[0.0, 0.0], [float("nan"), 1.0], [1.0, 0.0]],
                          [0, 0, 0], ks=(1,))
    assert got["degenerate_rows"] == 1 and got["nonfinite_rows"] == 1
    assert got["top1"] == pytest.approx(1 / 3)


def test_bootstrap_percentiles_are_the_nominal_ones():
    # The old index, floor(alpha/2 * B) - 1, selected the 2.45th percentile at
    # B=2000 instead of the 2.5th -- a half-percentile bias in every interval.
    units = [{"ref_words": 1, "a_err": i, "b_err": 0} for i in range(100)]
    ci = A.paired_bootstrap_ci(units, A.wer_delta_statistic, n_boot=1000,
                               seed=3, alpha=0.05)
    assert ci["lo"] < ci["point"] < ci["hi"]
    wide = A.paired_bootstrap_ci(units, A.wer_delta_statistic, n_boot=1000,
                                 seed=3, alpha=0.5)
    assert wide["hi"] - wide["lo"] < ci["hi"] - ci["lo"]   # alpha is honoured


def test_protocol_guard_refuses_a_cross_experiment_delta():
    base = {"model": "m", "n_samples": 5, "sample_digest": "d" * 64,
            "decode": {"max_length": 224, "strategy": "greedy",
                       "forced_decoder": "language=en,task=transcribe",
                       "decoder_prefix": "<|startoftranscript|> <|en|>"},
            "engine_meta": {"auto_cast": "none"}}
    assert A.assert_same_protocol(base, dict(base)) is True
    import copy
    for path, value in ((("engine_meta", "auto_cast"), "matmul"),
                        (("decode", "max_length"), 192),
                        (("decode", "decoder_prefix"), "<|startoftranscript|> <|de|>"),
                        (("decode", "forced_decoder"), "autodetect")):
        other = copy.deepcopy(base)
        other[path[0]][path[1]] = value
        with pytest.raises(ValueError, match="differ in the EXPERIMENT"):
            A.assert_same_protocol(base, other)


def test_protocol_guard_allows_the_treatment_to_differ():
    a = {"model": "m", "n_samples": 1, "sample_digest": "x" * 64,
         "engine": "cpu", "engine_meta": {"dtype": "float32"}}
    b = dict(a, engine="neuron", engine_meta={"dtype": "bfloat16"})
    assert A.assert_same_protocol(a, b) is True


def test_gate_publishes_relative_error_increase_next_to_the_pass():
    # MLPerf's rule is on ACCURACY, so at a 4.0% reference WER it passes a
    # candidate at 4.96% -- a 24% relative surge in transcription errors
    # reading as "MLPerf PASS". The relative figure ships with the verdict.
    g = A.accuracy_gate(1 - 0.0496, 1 - 0.04, error_rate=(0.0496, 0.04))
    assert g["passed"] is True
    assert g["relative_error_increase"] == pytest.approx(0.24, abs=0.005)


def test_asr_length_budget_is_one_number_for_both_arms():
    # max_new_tokens=224 on CPU grants 228 total (4-token forced prefix);
    # max_length=224 on Neuron grants 220. The asymmetry would have charged
    # 4 tokens of headroom to the accelerator.
    src = open(os.path.join(ROOT, "extras", "asr_wer_lane.py")).read()
    assert "max_new_tokens=args" not in src
    assert src.count("max_length=args.max_length") >= 2


def test_asr_compiled_dir_is_keyed_by_sequence_length():
    # An unkeyed dir let a 192-token trace be reloaded by a 224-token run;
    # generate() then fell back to the SAVED config's shorter max_length and
    # truncated the tail of every long utterance.
    args = ASR.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.compiled_dir is None
    src = open(os.path.join(ROOT, "extras", "asr_wer_lane.py")).read()
    assert 'f"-L{args.max_length}"' in src


def test_asr_truncation_needs_both_the_eos_and_the_length_clause():
    # Two wrong versions preceded this one, and the second was caught by the
    # pilot rather than by review:
    #   v1  shape >= cap        -> 100% false positives on a static graph
    #   v2  EOS absent          -> 100% false positives too, because
    #       transformers' Whisper generate STRIPS the trailing EOS (measured:
    #       8/8 flagged on both engines with 6-51 token buffers vs a 224 cap)
    # Only the disjunction is right, and each clause covers the other's blind
    # spot -- EOS-stripping on one side, a padded static buffer on the other
    # (Whisper's pad token IS its EOS token, 50257).
    src = open(os.path.join(ROOT, "extras", "asr_wer_lane.py")).read()
    assert "stopped_early = len(toks) < args.max_length" in src
    assert "finished = has_eos or stopped_early" in src
    assert "n_tok >= args.max_new_tokens" not in src


def test_asr_records_the_decoder_prefix_actually_emitted():
    # A traced generate() that silently DROPS language="en" raises nothing.
    # The only evidence both arms decoded the same way is the emitted prefix.
    assert hasattr(ASR, "describe_prefix")
    assert ("decode", "decoder_prefix") in A.PROTOCOL_FIELDS


def test_both_arms_load_the_processor_from_the_same_source():
    # A processor pickled at compile time under a different transformers could
    # resize with a different resample filter (~0.3pp top-1 on ImageNet) or a
    # different mel filterbank, and the delta would absorb a library skew.
    for lane in ("asr_wer_lane.py", "zeroshot_imagenet_lane.py"):
        src = open(os.path.join(ROOT, "extras", lane)).read()
        assert "AutoProcessor.from_pretrained(compiled)" not in src, lane
        assert "AutoProcessor.from_pretrained(args.model)" in src, lane


def test_zeroshot_refuses_a_partially_extracted_dataset(tmp_path):
    # A staging run that died after shard 2 left ~2,400 images across 1000
    # classes. Both arms scored the same 2,400, the digest matched, and the
    # study would have published "ImageNet-1k top-1" from a 24% slice.
    root = tmp_path / "images"
    for c in range(3):
        d = root / f"{c:04d}"
        d.mkdir(parents=True)
        for i in range(10 if c < 2 else 2):
            (d / f"s{c}{i:03d}.jpg").write_bytes(b"x")
    orig = ZS.N_CLASSES
    try:
        ZS.N_CLASSES = 3
        with pytest.raises(SystemExit, match="dataset incomplete"):
            ZS.select_images(str(root), 10, seed=1234)
        assert len(ZS.select_images(str(root), 2, seed=1234)) == 6
    finally:
        ZS.N_CLASSES = orig


def test_zeroshot_sample_keys_are_class_qualified(tmp_path):
    # WebDataset basenames are unique only within a shard. compare() pairs the
    # two engines through a dict on this key; an unqualified basename shared
    # by two classes would pair a shark against a volcano.
    root = tmp_path / "images"
    for c in (0, 1):
        d = root / f"{c:04d}"
        d.mkdir(parents=True)
        (d / "s0000001.jpg").write_bytes(b"x")     # SAME basename, two classes
    orig = ZS.N_CLASSES
    try:
        ZS.N_CLASSES = 2
        got = ZS.select_images(str(root), 1, seed=1)
        keys = [r["key"] for r in got]
        assert len(set(keys)) == 2, keys
        assert keys == ["0000/s0000001", "0001/s0000001"]
    finally:
        ZS.N_CLASSES = orig


def test_zeroshot_refuses_a_trace_built_for_another_protocol():
    src = open(os.path.join(ROOT, "extras", "zeroshot_imagenet_lane.py")).read()
    assert "trace_meta.json" in src
    assert "refusing to reload a mismatched trace" in src


def test_throughput_field_names_say_what_they_exclude():
    src = open(os.path.join(ROOT, "extras", "zeroshot_imagenet_lane.py")).read()
    assert "image_tower_images_per_s" in src
    assert "end_to_end_images_per_s" in src
    assert "images_per_s_forward" not in src        # the ambiguous old name


def test_compare_paths_are_guarded_by_the_protocol_check():
    for lane in ("asr_wer_lane.py", "zeroshot_imagenet_lane.py"):
        src = open(os.path.join(ROOT, "extras", lane)).read()
        assert "A.assert_same_protocol(cand, ref)" in src, lane


def test_asr_compare_reports_cross_engine_transcript_disagreements():
    # The CPU-only KV-cache probe checks CPU-cache vs CPU-nocache, which is
    # not the claim. This compares the two ENGINES to each other and is the
    # more sensitive detector: WER can cancel, byte-identical text cannot.
    src = open(os.path.join(ROOT, "extras", "asr_wer_lane.py")).read()
    assert "transcript_disagreements" in src


def test_preflight_covers_the_sentencepiece_tokenizer_dependency():
    src = open(os.path.join(ROOT, "extras", "run_accuracy.sh")).read()
    assert "sentencepiece" in src and "protobuf" in src


def test_clip_text_mask_defaults_off_and_is_symmetric_across_arms():
    # BISECTED on inf2, 2026-08-20: passing CLIP's attention mask into a
    # torch_neuronx trace returns NaN for every prompt (2048/2048 non-finite)
    # while the compiler reports PASS and the graph runs at 1,165 images/s.
    # Without the mask: 0/2048. CPU is clean either way.
    #   causal mask (finfo.min) + padding mask (finfo.min) -> -inf -> NaN
    # off is also the open_clip / clip_benchmark reference recipe.
    args = ZS.build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.text_attention_mask == "off"
    src = open(os.path.join(ROOT, "extras", "zeroshot_imagenet_lane.py")).read()
    # one switch drives BOTH the traced signature and the tokenised inputs, so
    # an arm cannot quietly get a different one
    assert 'wants_mask = args.text_attention_mask == "on"' in src
    assert 'use_mask = args.text_attention_mask == "on"' in src
    assert ("dataset", "text_attention_mask") in A.PROTOCOL_FIELDS


def test_clip_text_mask_is_part_of_the_trace_cache_key():
    # A mask-on trace reloaded by a mask-off run would report NaN under a
    # receipt claiming the reference recipe.
    src = open(os.path.join(ROOT, "extras", "zeroshot_imagenet_lane.py")).read()
    assert '-m{args.text_attention_mask}"' in src
    assert '"text_attention_mask": args.text_attention_mask}' in src   # trace_meta
