"""Phase-4 lanes: pretraining, DPO/ORPO re-basing, and the GRPO/RLVR probe.

    uvx --with pytest pytest -q tests/test_phase4.py

CPU-only and torch-free where possible. phase4_lib imports sft_lora, which is
stdlib-only at module scope, so the accounting helpers are testable off-box --
which is the point: the MFU denominator behind a published chart should not
require a $1.34/hr instance to verify.
"""
import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared", "train"))

import phase4_lib as P4  # noqa: E402
import sft_lora  # noqa: E402
from grpo_probe import extract_pred, gsm8k_gold, verifiable_reward  # noqa: E402
from pretrain_fineweb import ARCH, lr_at  # noqa: E402


# ---------------------------------------------------------------------------
# The anti-drift guarantee
# ---------------------------------------------------------------------------
def test_phase4_shares_the_published_mfu_formula_not_a_copy():
    """Phase 4 must reuse sft_lora's accounting, not reimplement it.

    Two implementations of MFU start identical and drift, and the moment they
    drift the Phase-4 lanes stop being comparable to the Phase-1/2/3 numbers
    they sit beside in the report. Identity, not equality, is the assertion.
    """
    assert P4.throughput_metrics is sft_lora.throughput_metrics
    assert P4.attention_flops_per_token is sft_lora.attention_flops_per_token
    assert P4.resolve_versions is sft_lora.resolve_versions
    assert P4.DEVICE_PROFILES is sft_lora.DEVICE_PROFILES


def test_unknown_device_profile_stops_rather_than_defaults():
    """A wrong profile is a wrong peak-FLOPS denominator on a chart."""
    with pytest.raises(SystemExit):
        P4.profile_for("trn9")
    assert P4.profile_for("trn1")["nproc"] == 2


# ---------------------------------------------------------------------------
# StepLog: the warmup convention
# ---------------------------------------------------------------------------
def _log_with(ms_values, warmup=2):
    log = P4.StepLog(warmup=warmup)
    for i, ms in enumerate(ms_values):
        log.rows.append({"step": i + 1, "loss": 1.0, "ms": ms,
                         "warmup": (i + 1) <= warmup})
    log._last = log._t0 + sum(ms_values) / 1000.0
    return log


def test_first_step_is_excluded_from_the_median():
    """A cold compile in step 1 must not be averaged into throughput."""
    log = _log_with([30000.0, 5000.0, 100.0, 100.0, 100.0], warmup=2)
    m = log.metrics()
    assert m["first_step_ms"] == 30000.0
    assert m["median_step_ms"] == 100.0        # warmup steps excluded
    assert m["compile_s"] == pytest.approx(29.9, abs=0.05)
    assert m["steps_recorded"] == 5
    assert m["warmup_steps_excluded"] == 2


def test_compile_seconds_floors_at_zero():
    """A warm cache can make step 1 faster than the median; that is not
    negative compile time."""
    log = _log_with([90.0, 100.0, 100.0, 100.0], warmup=1)
    assert log.metrics()["compile_s"] == 0.0


def test_steady_throughput_uses_the_median_not_the_mean():
    log = _log_with([30000.0, 100.0, 100.0, 100.0], warmup=1)
    # 1000 tokens per 100 ms => 10_000 tok/s, unaffected by the 30 s first step
    assert P4.steady_tokens_per_s(log, 1000) == pytest.approx(10_000.0)


def test_empty_log_reports_none_rather_than_crashing_or_zero():
    log = P4.StepLog(warmup=10)
    m = log.metrics()
    assert m["median_step_ms"] is None and m["first_step_ms"] is None
    assert P4.steady_tokens_per_s(log, 1000) is None


# ---------------------------------------------------------------------------
# Result emission
# ---------------------------------------------------------------------------
def test_emit_result_writes_both_throughput_conventions(tmp_path):
    log = _log_with([30000.0, 100.0, 100.0, 100.0], warmup=1)
    out = tmp_path / "r.json"
    res = P4.emit_result(
        out, tag="t", stage="pretrain", model="m", dataset="d",
        params={"params_total": 1000, "params_trainable": 1000, "params_frozen": 0},
        config={}, log=log, tokens_per_step=1000,
        profile=P4.profile_for("trn1"))
    on_disk = json.loads(out.read_text())
    assert on_disk == res
    # steady-state describes the chip; end-to-end is what a human waits through
    assert res["tokens_per_s"] > res["tokens_per_s_end_to_end"]
    assert res["stage"] == "pretrain"
    assert res["peak_bf16_flops"] == P4.profile_for("trn1")["peak_bf16_flops"]
    # both published Trainium1 denominators are reported, per DEVICE_PROFILES
    assert res["mfu_pct_alt"] is not None
    # device HBM is never estimated
    assert res["peak_device_mem_mib"] is None


def test_pretraining_is_charged_dense_6n_not_the_lora_split():
    """Nothing is frozen in a from-scratch run, so FLOPs/token must be 6N."""
    log = _log_with([100.0, 100.0], warmup=0)
    res = P4.emit_result(
        "/tmp/np-test-6n.json", tag="t", stage="pretrain", model="m", dataset="d",
        params={"params_total": 100, "params_trainable": 100, "params_frozen": 0},
        config={}, log=log, tokens_per_step=1, profile=P4.profile_for("trn1"))
    assert res["flops_per_token"] == pytest.approx(600.0)


def test_failure_receipt_is_a_terminal_result_with_a_reason(tmp_path):
    out = tmp_path / "f.failure.json"
    rec = P4.failure_receipt(out, tag="grpo", stage="grpo", box="trn1",
                             reason="online RL unsupported", detail="trace...")
    d = json.loads(out.read_text())
    assert d["status"] == "failed" and d["reason"] == "online RL unsupported"
    assert d["captured"].endswith("Z") and rec["versions"]


# ---------------------------------------------------------------------------
# Pretraining lane
# ---------------------------------------------------------------------------
def test_arch_is_the_published_smollm2_360m_shape():
    """Pinned so the lane stays reproducible if the Hub config moves."""
    assert ARCH["hidden_size"] == 960
    assert ARCH["num_hidden_layers"] == 32
    assert ARCH["num_attention_heads"] == 15
    assert ARCH["num_key_value_heads"] == 5      # GQA
    assert ARCH["intermediate_size"] == 2560
    assert ARCH["vocab_size"] == 49152
    # head_dim must divide evenly or attention silently reshapes wrong
    assert ARCH["hidden_size"] % ARCH["num_attention_heads"] == 0


def test_vocab_fits_uint16_so_the_token_memmap_is_lossless():
    """The corpus is stored as uint16; a larger vocab would truncate ids."""
    assert ARCH["vocab_size"] < 65536


def test_param_count_is_about_362m():
    a = ARCH
    head = a["hidden_size"] // a["num_attention_heads"]
    per_layer = (
        a["hidden_size"] * a["num_attention_heads"] * head        # q
        + a["hidden_size"] * a["num_key_value_heads"] * head * 2  # k, v
        + a["hidden_size"] * a["hidden_size"]                     # o
        + 3 * a["hidden_size"] * a["intermediate_size"]           # gate, up, down
    )
    total = per_layer * a["num_hidden_layers"] + a["vocab_size"] * a["hidden_size"]
    assert 350e6 < total < 375e6


def _lr_args(**kw):
    d = dict(lr=3e-4, min_lr_ratio=0.1, warmup_steps=100, lr_schedule="cosine")
    d.update(kw)
    return argparse.Namespace(**d)


def test_constant_schedule_never_varies_the_lr_scalar():
    """The LR scalar is traced into the XLA graph as a constant, so any change
    is a new graph and a full neuronx-cc compile. Measured on 2026-08-14: the
    cosine schedule produced 3 graphs in 2 steps and a 53-minute second step.
    'constant' exists to make the throughput lane measurable at all, so it must
    return the SAME value at every step -- including inside the warmup window,
    where the cosine path would otherwise ramp."""
    a = _lr_args(lr_schedule="constant")
    values = {lr_at(s, 1000, a) for s in (0, 1, 50, 99, 100, 500, 999, 5000)}
    assert values == {3e-4}


def test_lr_warms_up_then_cosine_decays_to_the_floor():
    a = _lr_args()
    assert lr_at(0, 1000, a) == pytest.approx(3e-4 / 100)
    assert lr_at(99, 1000, a) == pytest.approx(3e-4)
    assert lr_at(999, 1000, a) == pytest.approx(3e-4 * 0.1, rel=1e-2)
    # monotone decay after warmup
    post = [lr_at(s, 1000, a) for s in range(100, 1000, 50)]
    assert all(x >= y - 1e-12 for x, y in zip(post, post[1:]))


def test_lr_never_dips_below_the_floor_even_past_the_end():
    a = _lr_args()
    assert lr_at(5000, 1000, a) >= 3e-4 * 0.1 - 1e-12


# ---------------------------------------------------------------------------
# RLVR verifier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ans,want", [
    ("Some reasoning.\n#### 42", "42"),
    ("steps\n#### 1,234", "1234"),
    ("no marker here", None),
])
def test_gsm8k_gold_extraction(ans, want):
    assert gsm8k_gold(ans) == want


@pytest.mark.parametrize("text,want", [
    ("The answer is 42.", "42"),
    ("first 7 then finally 13", "13"),        # last number wins, the convention
    ("1,024 widgets", "1024"),
    ("no digits at all", None),
])
def test_prediction_extraction(text, want):
    assert extract_pred(text) == want


def test_reward_is_one_only_for_an_exact_match():
    comps = ["so the answer is 42", "the answer is 43", "no number"]
    assert verifiable_reward(comps, ["42", "42", "42"]) == [1.0, 0.0, 0.0]


def test_reward_handles_missing_gold_without_rewarding_it():
    """A row with no parseable gold must never score 1.0 -- that would reward
    the policy for the dataset being malformed."""
    assert verifiable_reward(["the answer is 42"], [None]) == [0.0]


def test_reward_accepts_chat_style_completions():
    comps = [[{"role": "assistant", "content": "hence 42"}]]
    assert verifiable_reward(comps, ["42"]) == [1.0]


# ---------------------------------------------------------------------------
# Fixed-shape collator (torch only)
# ---------------------------------------------------------------------------
def test_collator_pads_and_truncates_to_one_fixed_length():
    torch = pytest.importorskip("torch")
    from posttrain_align import FixedShapeCollator

    def inner(_features):
        return {
            "chosen_input_ids": torch.ones(2, 5, dtype=torch.long),
            "chosen_attention_mask": torch.ones(2, 5, dtype=torch.long),
            "chosen_labels": torch.ones(2, 5, dtype=torch.long),
            "rejected_input_ids": torch.ones(2, 12, dtype=torch.long),
            "not_a_sequence": torch.ones(2, dtype=torch.long),
        }

    out = FixedShapeCollator(inner, pad_token_id=99, max_length=8)([{}])
    for k in ("chosen_input_ids", "chosen_attention_mask",
              "chosen_labels", "rejected_input_ids"):
        assert out[k].shape == (2, 8), k
    # pad VALUES are not interchangeable
    assert out["chosen_input_ids"][0, -1].item() == 99      # pad token
    assert out["chosen_attention_mask"][0, -1].item() == 0  # masked out
    assert out["chosen_labels"][0, -1].item() == -100       # no loss on padding
    # over-long input is truncated, not left to recompile
    assert out["rejected_input_ids"].shape[1] == 8
    # 1-D tensors are left alone
    assert out["not_a_sequence"].shape == (2,)


# ---------------------------------------------------------------------------
# The transformers.Trainer surface NeuronTrainer omits
# ---------------------------------------------------------------------------
def test_compat_mixin_covers_what_dpo_and_orpo_actually_read():
    """Pin the enumerated gap.

    These four names were found by AST-diffing TRL's reads against
    NeuronTrainer's provides, after 'is_deepspeed_enabled' cost a four-minute
    8B checkpoint load to discover the slow way. If a TRL upgrade needs more,
    this test is where the next name gets added -- deliberately, not by
    another expensive AttributeError.
    """
    from posttrain_align import HFTrainerCompat as C
    for name in ("is_deepspeed_enabled", "is_fsdp_enabled",
                 "_prepare_inputs", "create_model_card"):
        assert hasattr(C, name), name
        assert name in C.COVERS, f"{name} supplied but not declared in COVERS"


def test_neuron_has_no_deepspeed_or_fsdp():
    """Not placeholders: sharding on Neuron is NxD's job and DeepSpeed has no
    Neuron integration, so False is the correct answer rather than a stub."""
    from posttrain_align import HFTrainerCompat as C
    assert C.is_deepspeed_enabled is False
    assert C.is_fsdp_enabled is False
    assert C.is_fsdp_xla_enabled is False


def test_create_model_card_is_a_noop_because_nothing_is_pushed():
    from posttrain_align import HFTrainerCompat
    assert HFTrainerCompat().create_model_card(model_name="x") is None


def test_prepare_inputs_leaves_non_tensors_alone():
    torch = pytest.importorskip("torch")
    from posttrain_align import HFTrainerCompat

    class Fake(HFTrainerCompat):
        def _prepare_inputs(self, inputs):      # CPU stand-in for the XLA move
            return {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()}

    out = Fake()._prepare_inputs({"ids": torch.ones(2), "meta": "keep-me"})
    assert out["meta"] == "keep-me"
    assert isinstance(out["ids"], torch.Tensor)


# ---------------------------------------------------------------------------
# Forward-kwarg stripping
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self):
        self.seen = None

    def forward(self, x, **kw):
        self.seen = kw
        return x


def test_falsy_unsupported_kwargs_are_dropped():
    """torch_xla's checkpoint() rejects unknown kwargs; TRL sends use_cache."""
    from posttrain_align import strip_unsupported_forward_kwargs
    m = strip_unsupported_forward_kwargs(_FakeModel())
    assert m.forward(1, use_cache=False, keep=7) == 1
    assert m.seen == {"keep": 7}


def test_a_truthy_request_raises_instead_of_being_silently_dropped():
    """Discarding a real request would change what the loss reads from while
    still producing a plausible-looking number."""
    from posttrain_align import strip_unsupported_forward_kwargs
    m = strip_unsupported_forward_kwargs(_FakeModel())
    with pytest.raises(ValueError, match="use_cache"):
        m.forward(1, use_cache=True)
    # By DEFAULT output_hidden_states is not droppable either.
    m2 = strip_unsupported_forward_kwargs(_FakeModel(), drop=("output_hidden_states",))
    with pytest.raises(ValueError, match="output_hidden_states"):
        m2.forward(1, output_hidden_states=True)


def test_the_verified_exception_drops_output_hidden_states_even_when_true():
    """Granted only because trl 0.24.0's concatenated_forward -- the path this
    lane takes -- sets output_hidden_states=True and never reads it back. The
    Liger path does read it, which is why the caller asserts Liger is off before
    passing this."""
    from posttrain_align import strip_unsupported_forward_kwargs
    m = strip_unsupported_forward_kwargs(
        _FakeModel(), drop=("use_cache",),
        drop_even_if_true=("output_hidden_states",))
    assert m.forward(1, output_hidden_states=True, keep=3) == 1
    assert m.seen == {"keep": 3}


def test_stripping_leaves_untouched_calls_alone():
    from posttrain_align import strip_unsupported_forward_kwargs
    m = strip_unsupported_forward_kwargs(_FakeModel())
    assert m.forward(5, attention_mask="am") == 5
    assert m.seen == {"attention_mask": "am"}


# ---------------------------------------------------------------------------
# The super() closure-cell bug that the GRPO probe found at stage D
# ---------------------------------------------------------------------------
def test_naive_dict_copy_reproduces_the_super_typeerror():
    """Pin the bug, so the fix below is demonstrably fixing something real.

    This is exactly what killed GRPO stage D on 2026-08-14:
    'TypeError: super(type, obj): obj must be an instance or subtype of type'.

    The match is deliberately loose: 3.12 (the box) says "obj must be an
    instance or subtype of type", 3.13 (this laptop) says "obj (instance of X)
    is not an instance or subtype of type (Y)". Pinning either wording would
    make the test fail on the other interpreter for no real reason.
    """
    from posttrain_align import _clone_rebound

    class OrigBase:
        def __init__(self):
            self.tag = "orig-base"

    class Upstream(OrigBase):          # stands in for trl.DPOTrainer
        def __init__(self):
            super().__init__()         # compiles to super(__class__, self)
            self.tag = "upstream"

    class NewBase:                     # stands in for NeuronTrainer
        def __init__(self):
            self.tag = "new-base"

    naive = type("Naive", (NewBase,), dict(Upstream.__dict__))
    with pytest.raises(TypeError, match="instance or subtype of type"):
        naive()

    fixed = type("Fixed", (NewBase,), dict(Upstream.__dict__))
    for k, v in list(Upstream.__dict__.items()):
        r = _clone_rebound(v, fixed)
        if r is not v:
            setattr(fixed, k, r)
    obj = fixed()                       # super() now resolves to NewBase
    assert obj.tag == "upstream"


def test_clone_rebound_does_not_mutate_the_original_class():
    """Rebinding cells in place would corrupt trl.DPOTrainer process-wide."""
    from posttrain_align import _clone_rebound

    class Base:
        def hello(self):
            return "base"

    class Up(Base):
        def hello(self):
            return "up+" + super().hello()

    class Other:
        def hello(self):
            return "other"

    new = type("New", (Other,), dict(Up.__dict__))
    setattr(new, "hello", _clone_rebound(Up.__dict__["hello"], new))
    assert new().hello() == "up+other"
    assert Up().hello() == "up+base"      # original still intact


def test_clone_rebound_passes_through_functions_without_super():
    from posttrain_align import _clone_rebound

    def plain(self):
        return 1
    assert _clone_rebound(plain, object) is plain
    assert _clone_rebound("not-a-function", object) == "not-a-function"


def test_collator_is_a_noop_when_already_the_right_shape():
    torch = pytest.importorskip("torch")
    from posttrain_align import FixedShapeCollator
    t = torch.arange(16).reshape(2, 8)
    out = FixedShapeCollator(lambda f: {"chosen_input_ids": t.clone()},
                             pad_token_id=0, max_length=8)([{}])
    assert torch.equal(out["chosen_input_ids"], t)


# ---------------------------------------------------------------------------
# The per-step host syncs TRL performs, and when it is safe to skip them
# ---------------------------------------------------------------------------
class _FakeAccelerator:
    def __init__(self):
        self.calls = 0

    def gather_for_metrics(self, tensor, *a, **k):
        self.calls += 1
        return ["gathered", tensor]


class _FakeTrainer:
    def __init__(self):
        self.accelerator = _FakeAccelerator()


def test_metric_gather_is_neutralised_only_under_pure_tensor_parallelism():
    """dp == 1 is the whole justification, so the guard is the thing to test.

    Skipping the gather is arithmetically exact when every rank holds the same
    batch. The moment any real data parallelism exists it would silently report
    rank 0's metrics as if they were the global mean -- a wrong number that
    looks perfectly plausible, which is the worst kind.
    """
    from posttrain_align import neutralise_out_of_graph_gathers

    tp_only = _FakeTrainer()
    assert neutralise_out_of_graph_gathers(tp_only, "dpo", nproc=2, tp=2,
                                           say=lambda *a: None) is True
    assert tp_only.accelerator.gather_for_metrics("x") == "x"
    assert tp_only.accelerator.calls == 0

    with_dp = _FakeTrainer()
    assert neutralise_out_of_graph_gathers(with_dp, "dpo", nproc=8, tp=2,
                                           say=lambda *a: None) is False
    assert with_dp.accelerator.gather_for_metrics("x") == ["gathered", "x"]
    assert with_dp.accelerator.calls == 1


def test_metric_gather_neutralisation_survives_a_trainer_without_an_accelerator():
    from posttrain_align import neutralise_out_of_graph_gathers

    class _Bare:
        pass

    assert neutralise_out_of_graph_gathers(_Bare(), "orpo", nproc=2, tp=2,
                                           say=lambda *a: None) is False


# ---------------------------------------------------------------------------
# The accounting bug the preference lanes shipped with, pinned so it cannot return
# ---------------------------------------------------------------------------
def test_preference_tokens_per_step_uses_dp_not_world_size():
    """Under TP the ranks share one micro-batch, so world size must not multiply.

    The preference lanes computed tokens as max_length * micro_batch *
    grad_accum * nproc * 2 with nproc=2 and tp=2. Data-parallel size is 1 there,
    so that reported exactly twice the real token rate. Paired with a rank-local
    parameter sum (half the model) the two errors cancelled inside MFU, which is
    why nothing looked wrong.
    """
    tps = P4.tokens_per_optimizer_step
    # trn1.2xlarge preference lane: world 2, tp 2 -> dp 1.
    assert 2 * tps(512, 1, 8, 1) == 8192
    # The bug: multiplying by the world size instead.
    assert 2 * tps(512, 1, 8, 2) == 16384
    # And the identity guarantee, so this cannot drift from sft_lora.
    assert P4.tokens_per_optimizer_step is sft_lora.tokens_per_optimizer_step
    assert P4.count_parameters is sft_lora.count_parameters
    assert P4.lora_flops_per_token is sft_lora.lora_flops_per_token


def test_dpo_reference_pass_is_charged_above_the_shared_convention():
    """DPO's adapter-disabled reference forward is real work the 6N/4N misses."""
    trainable, total = 42_000_000, 8_030_000_000
    frozen = total - trainable
    base = P4.lora_flops_per_token(trainable, frozen)
    with_ref = base + 2.0 * frozen
    assert with_ref > base
    # A forward over the frozen base is half the cost of the 4*frozen term the
    # convention already charges for forward+backward-input on frozen weights.
    assert with_ref - base == pytest.approx(0.5 * 4.0 * frozen)


def test_log_shim_absorbs_the_start_time_arity_difference():
    """TRL forwards start_time positionally; NeuronTrainer.log does not take it.

    This killed the ORPO lane inside optimum-neuron's own logging step-closure
    AFTER the model had compiled and steps had run -- the most expensive place
    to discover a two-argument signature.
    """
    from posttrain_align import HFTrainerCompat

    seen = []

    class _TwoArgParent:                      # NeuronTrainer.log(self, logs)
        def log(self, logs):
            seen.append(("two", logs))

    class _ThreeArgParent:                    # a future optimum-neuron
        def log(self, logs, start_time=None):
            seen.append(("three", logs, start_time))

    class Narrow(HFTrainerCompat, _TwoArgParent):
        pass

    class Wide(HFTrainerCompat, _ThreeArgParent):
        pass

    Narrow().log({"loss": 1.0}, 123.0)
    assert seen[-1] == ("two", {"loss": 1.0})       # start_time dropped, no raise

    Wide().log({"loss": 2.0}, 456.0)
    assert seen[-1] == ("three", {"loss": 2.0}, 456.0)   # passed through

    assert "log" in HFTrainerCompat.COVERS


def test_steplog_flags_a_run_that_diverged_to_nan():
    """A NaN run and a healthy run produce identical throughput. Say which is which.

    The 150-step ORPO lane went non-finite at step 23 and reported 1,011.2
    tok/s -- within 0.13% of the clean 30-step run -- because NaN costs the same
    FLOPs as a number. Nothing in a timing harness can notice that, so the
    result record has to carry it explicitly.
    """
    diverged = P4.StepLog(warmup=1)
    for v in (1.0, 2.0, float("nan"), float("nan")):
        diverged.tick(v)
    m = diverged.metrics()
    assert m["loss_numerically_valid"] is False
    assert m["loss_first_nonfinite_step"] == 3
    assert m["loss_nonfinite_steps"] == 2
    assert m["loss_finite_steps"] == 2
    assert "DIVERGED" in m["loss_validity_note"]

    clean = P4.StepLog(warmup=1)
    for v in (1.0, 2.0):
        clean.tick(v)
    cm = clean.metrics()
    assert cm["loss_numerically_valid"] is True
    assert cm["loss_validity_note"] is None
    assert cm["loss_first_nonfinite_step"] is None
