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
