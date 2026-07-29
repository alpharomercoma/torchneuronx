"""Pure-helper tests for sft_lora.py: MFU math, warmup flags, dolly format.

    python3 -m pytest tests/test_train_scripts.py -q   # 5 passed
"""
import os
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
