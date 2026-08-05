"""Pure-helper tests for sft_lora.py: MFU math, warmup flags, dolly format.

    python3 -m pytest tests/test_train_scripts.py -q   # 5 passed
"""
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared", "train"))
import sft_lora  # noqa: E402  (heavy imports are lazy; import is cheap)


def test_lora_flops_per_token_hand_computed():
    # 40M trainable at 6 FLOPs + 8B frozen at 4 FLOPs
    assert sft_lora.lora_flops_per_token(40e6, 8e9) == 6 * 40e6 + 4 * 8e9


def test_throughput_metrics_hand_computed():
    m = sft_lora.throughput_metrics(40e6, 8e9, tokens_per_s=1000,
                                    peak_flops=210e12)
    assert m["flops_per_token"] == 3.224e10
    assert abs(m["tflops"] - 32.24) < 1e-9
    assert abs(m["mfu_pct"] - 100 * 3.224e13 / 210e12) < 1e-9
    # No alt denominator asked for -> no second number invented.
    assert m["mfu_pct_alt"] is None


def test_throughput_metrics_reports_both_denominators():
    """AWS's arch page (190) and instance marketing (210) disagree for trn1.

    tflops is denominator-free and must be identical under both; only the MFU
    percentage moves. Reporting one number would hide which source was used.
    """
    m = sft_lora.throughput_metrics(40e6, 8e9, tokens_per_s=1000,
                                    peak_flops=190e12, peak_flops_alt=210e12)
    assert abs(m["tflops"] - 32.24) < 1e-9
    assert abs(m["mfu_pct"] - 100 * 3.224e13 / 190e12) < 1e-9
    assert abs(m["mfu_pct_alt"] - 100 * 3.224e13 / 210e12) < 1e-9
    # The arch-page denominator is smaller, so it reports the HIGHER MFU.
    assert m["mfu_pct"] > m["mfu_pct_alt"]


def test_dense_would_overstate_lora_mfu():
    lora = sft_lora.lora_flops_per_token(40e6, 8e9)
    dense = 6 * (40e6 + 8e9)
    assert dense / lora > 1.45          # the ~50% inflation the docstring cites


def test_tokens_per_step_tp_does_not_multiply():
    # TP=2 shards one model; only DP multiplies tokens. dp=1 on trn1.2xlarge.
    assert sft_lora.tokens_per_optimizer_step(2048, 1, 8, 1) == 16384


def test_mark_warmup_by_position_and_dolly_shape():
    trace = [{"step": s, "loss": 1.0, "ms": 100} for s in range(15)]
    out = sft_lora.mark_warmup(trace, warmup_steps=10)
    assert sum(e["warmup"] for e in out) == 10 and not out[14]["warmup"]
    assert sft_lora.steady_state_step_ms(out) == [100] * 5

    msgs = sft_lora.dolly_messages({
        "instruction": "Summarize.", "context": "Some text.",
        "response": "A summary."})
    roles = [m["role"] for m in msgs]
    assert roles[-2:] == ["user", "assistant"]
    assert msgs[-1]["content"] == "A summary."


def test_kwarg_tolerant_shim_partial_binds_non_tensors():
    calls = {}

    def fake_checkpoint(function, *args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return function(*args)

    def layer(x, y, use_cache=None, reduction="mean"):
        calls["layer"] = (x, y, use_cache, reduction)
        return "out"

    is_tensor = lambda v: isinstance(v, str) and v.startswith("T")  # noqa: E731
    tolerant = sft_lora.make_kwarg_tolerant(fake_checkpoint, is_tensor)

    out = tolerant(layer, "T1", "T2", use_cache=False, reduction="sum")
    assert out == "out"
    assert calls["kwargs"] == {}                       # nothing non-tensor leaks
    assert calls["layer"] == ("T1", "T2", False, "sum")  # semantics preserved
    # idempotent: wrapping twice returns the same shim
    assert sft_lora.make_kwarg_tolerant(tolerant, is_tensor) is tolerant


def test_patch_walks_sys_modules(monkeypatch):
    import sys as _sys
    import types

    def orig_ckpt(fn, *a, **kw):
        return fn(*a)

    fake = types.ModuleType("optimum.neuron.models.training.llama.modeling_llama")
    fake.checkpoint = orig_ckpt
    monkeypatch.setitem(_sys.modules,
                        "optimum.neuron.models.training.llama.modeling_llama",
                        fake)
    patched = sft_lora.patch_optimum_modeling_checkpoint(lambda v: False)
    assert "modeling_llama" in patched
    assert getattr(fake.checkpoint, "_np_kwarg_shim", False)
    # second call: already shimmed, not re-patched
    assert "modeling_llama" not in sft_lora.patch_optimum_modeling_checkpoint(
        lambda v: False)


# --------------------------------------------------- device profiles (Phase 3)
def test_trn1_profile_reproduces_published_phase1_config():
    """Phase-1/2 published MFU (68.3% @2048) came from 2/2/210e12.

    The parallelism is pinned as hard as the FLOPs formula is -- a change to
    2/2/1 would invalidate every published training number. The DENOMINATOR
    moved deliberately in Phase 3: the primary is now the arch page's 190e12,
    and 210e12 survives as the alt so the published numbers stay reproducible
    and re-derivable rather than being quietly restated.
    """
    args = sft_lora.build_parser().parse_args(["--tag", "t", "--out", "/tmp/o"])
    p = sft_lora.resolve_device_profile(args, env={})
    assert p["device_profile"] == "trn1"
    assert (p["nproc"], p["tp"], p["pp"]) == (2, 2, 1)
    assert p["peak_bf16_flops"] == 190e12        # arch page
    assert p["peak_bf16_flops_alt"] == 210e12    # what REPORT.md used
    assert p["logical_nc_config"] is None


def test_published_phase1_mfu_is_still_derivable_from_the_alt_denominator():
    """The 68.318% in REPORT.md must remain reconstructible after the switch.

    Reproduce it from the alt denominator, and show the arch-page denominator
    restates the SAME measurement as 75.5% -- same silicon, same tokens/s,
    different published peak. This is exactly why both are emitted.
    """
    published_mfu, published_peak = 68.318, 210e12
    achieved = published_mfu / 100.0 * published_peak
    arch = 100.0 * achieved / 190e12
    assert abs(arch - 75.51) < 0.05
    assert abs(published_mfu * (210.0 / 190.0) - arch) < 1e-6


def test_peak_ratio_between_the_chips_depends_on_the_source():
    """Mixing sources across boxes is the bug this guards against.

    Consistently-sourced: 667/190 = 3.51x. The old mixed pairing (instance
    figure for trn1, arch figure for trn2) understated it as 3.18x -- in the
    direction that flatters the newer chip.
    """
    trn1 = sft_lora.DEVICE_PROFILES["trn1"]
    trn2 = sft_lora.DEVICE_PROFILES["trn2"]
    arch_ratio = trn2["peak_bf16_flops"] / trn1["peak_bf16_flops"]
    mixed_ratio = trn2["peak_bf16_flops"] / trn1["peak_bf16_flops_alt"]
    assert abs(arch_ratio - 3.51) < 0.01
    assert abs(mixed_ratio - 3.18) < 0.01
    assert arch_ratio > mixed_ratio


def test_trn2_profile_is_four_logical_cores_at_667_tflops():
    args = sft_lora.build_parser().parse_args(
        ["--tag", "t", "--out", "/tmp/o", "--device-profile", "trn2"])
    p = sft_lora.resolve_device_profile(args, env={})
    assert (p["nproc"], p["tp"]) == (4, 4)
    assert p["peak_bf16_flops"] == 667e12
    # Arch and instance specs agree for Trainium2 -- no conflict to report.
    assert p["peak_bf16_flops_alt"] == 667e12
    assert p["logical_nc_config"] == 2      # Neuron default on Trainium2
    assert p["hbm_gib_per_core"] == 24      # vs 16 on trn1 -- the ctx-cliff lever


def test_tp_ladder_overrides_beat_the_profile():
    """The trn2 TP fallback ladder must be a flag change, never a code edit."""
    args = sft_lora.build_parser().parse_args(
        ["--tag", "t", "--out", "/tmp/o", "--device-profile", "trn2",
         "--nproc-per-node", "8", "--tensor-parallel-size", "8"])
    p = sft_lora.resolve_device_profile(args, env={})
    assert (p["nproc"], p["tp"]) == (8, 8)
    assert p["peak_bf16_flops"] == 667e12   # still the same chip


def test_detect_device_key_precedence():
    # explicit instance type beats the environment
    assert sft_lora.detect_device_key(
        env={"NP_DEVICE": "trn1"}, instance_type="trn2.3xlarge") == "trn2"
    # environment beats the default
    assert sft_lora.detect_device_key(env={"NP_DEVICE": "trn2"}) == "trn2"
    # an unknown family falls back rather than crashing a multi-hour lane
    assert sft_lora.detect_device_key(env={}, instance_type="p5.48xlarge") == "trn1"
    assert sft_lora.detect_device_key(env={}) == "trn1"


def test_unknown_profile_fails_fast():
    import pytest
    args = sft_lora.build_parser().parse_args(["--tag", "t", "--out", "/tmp/o"])
    args.device_profile = "trn9"
    with pytest.raises(SystemExit):
        sft_lora.resolve_device_profile(args, env={})


def test_torchrun_argv_carries_the_profile_rank_count():
    argv = sft_lora.torchrun_argv("/x/sft_lora.py", ["--tag", "t"],
                                  nproc_per_node=4)
    assert "--nproc_per_node=4" in argv
    assert argv[1:3] == ["-m", "torch.distributed.run"]


def test_lnc_env_is_set_only_for_trainium2():
    env = {}
    sft_lora.apply_neuron_env(env=env, cache_dir="/tmp/c",
                              profile=sft_lora.DEVICE_PROFILES["trn1"])
    assert "NEURON_LOGICAL_NC_CONFIG" not in env

    env = {}
    sft_lora.apply_neuron_env(env=env, cache_dir="/tmp/c",
                              profile=sft_lora.DEVICE_PROFILES["trn2"])
    assert env["NEURON_LOGICAL_NC_CONFIG"] == "2"

    # setdefault: the ladder can force LNC=1 from the driver
    env = {"NEURON_LOGICAL_NC_CONFIG": "1"}
    sft_lora.apply_neuron_env(env=env, cache_dir="/tmp/c",
                              profile=sft_lora.DEVICE_PROFILES["trn2"])
    assert env["NEURON_LOGICAL_NC_CONFIG"] == "1"


def test_synthetic_data_defaults_off_so_published_lanes_are_unaffected():
    args = sft_lora.build_parser().parse_args(["--tag", "t", "--out", "/tmp/o"])
    assert args.synthetic_data == 0


def test_synthetic_lane_drops_the_formatter():
    """The isolation lane feeds pre-tokenised rows.

    If build_trainer still passed formatting_func, TRL would re-render the
    random ids as text and the lane would measure the very host work it exists
    to remove -- a silent no-op that would look like a real negative result.
    """
    src = pathlib.Path(sft_lora.__file__).read_text()
    block = src.split("def build_trainer", 1)[1]
    assert 'kwargs.pop("formatting_func")' in block
    assert 'getattr(args, "synthetic_data", 0)' in block


def test_synthetic_lane_marks_its_loss_meaningless_in_the_payload():
    """Random tokens teach nothing, so the loss must never be plottable as real."""
    src = pathlib.Path(sft_lora.__file__).read_text()
    assert '"loss_is_meaningless": True' in src
    assert '"synthetic:random-token-ids"' in src


def test_synthetic_lane_skips_the_holdout():
    src = pathlib.Path(sft_lora.__file__).read_text()
    assert "if args.synthetic_data and args.holdout_frac:" in src


def test_impossible_mfu_is_flagged():
    """MFU > 100% means the measurement is broken, not that the chip is fast.

    A grad_accum=4 control reported 146.7% because accumulation is unrolled into
    the compiled graph and the step timer captures a different share of deferred
    XLA work at different accumulation depths.
    """
    out = sft_lora.throughput_metrics(50e6, 8e9, tokens_per_s=10_000_000,
                                      peak_flops=210e12)
    assert out["mfu_pct"] > 100.0
    assert "mfu_impossible" in out
    assert "tokens_per_s_end_to_end" in out["mfu_impossible"]["use_instead"]


def test_plausible_mfu_is_not_flagged():
    out = sft_lora.throughput_metrics(50e6, 8e9, tokens_per_s=3000,
                                      peak_flops=210e12)
    assert out["mfu_pct"] < 100.0
    assert "mfu_impossible" not in out
