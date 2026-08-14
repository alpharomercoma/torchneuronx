"""Shared plumbing for the Phase-4 pretraining and post-training lanes.

Phase 4 adds four lanes the Phase 1-3 harness never had to express: from-scratch
pretraining, DPO, ORPO, and a GRPO/RLVR probe. They all need the same three
things -- a step timer that separates the compile-bearing first step from steady
state, the published MFU denominators, and a result JSON that
analysis/make_report.py can already read.

WHY THIS IMPORTS sft_lora RATHER THAN RESTATING IT
--------------------------------------------------
sft_lora.py is 81 KB and is the source of every published Phase-1/2/3 number.
Copying its MFU formula here would create a second implementation that starts
identical and drifts, and two lanes computing MFU differently are silently
incomparable -- which is the one failure this study cannot afford, since the
whole point is cross-hardware comparison. sft_lora is import-safe (its only
top-level branch is the __main__ guard), so the pure helpers are imported.

The one thing deliberately NOT inherited is the LoRA FLOPs split. Pretraining,
DPO and ORPO are full-parameter or adapter runs with different frozen/trainable
splits, and each lane states its own split explicitly.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sft_lora  # noqa: E402  (path shim above is required first)

DEVICE_PROFILES = sft_lora.DEVICE_PROFILES
throughput_metrics = sft_lora.throughput_metrics
attention_flops_per_token = sft_lora.attention_flops_per_token
resolve_versions = sft_lora.resolve_versions

# Two more that Phase 4 got WRONG by hand before reusing them, which is the
# whole argument for this module existing:
#
#   tokens_per_optimizer_step -- the preference lanes multiplied tokens by
#     nproc. Under TP the ranks share one micro-batch, so only the
#     DATA-parallel size multiplies tokens; nproc overstated throughput 2x.
#   count_parameters -- the preference lanes summed this rank's shard, which
#     halves the FLOPs/token. Paired with the 2x above the errors cancelled in
#     MFU, so MFU looked right while tok/s was double. Two wrongs cancelling is
#     worse than one wrong, because nothing looks broken.
tokens_per_optimizer_step = sft_lora.tokens_per_optimizer_step
count_parameters = sft_lora.count_parameters
lora_flops_per_token = sft_lora.lora_flops_per_token
PARAM_METHOD_NOTES = sft_lora.PARAM_METHOD_NOTES


def profile_for(name: str) -> dict:
    """Device profile by name, failing loudly rather than defaulting silently.

    A wrong profile means a wrong peak-FLOPS denominator, which means a wrong
    MFU on a chart. Better to stop than to publish a number computed against
    the other chip's peak.
    """
    if name not in DEVICE_PROFILES:
        raise SystemExit(f"unknown device profile {name!r}; "
                         f"have {sorted(DEVICE_PROFILES)}")
    return DEVICE_PROFILES[name]


class StepLog:
    """Per-step loss and wall time, with the warmup convention Phase 1-3 uses.

    The first step is not a training step -- it traces the XLA graph and then
    either compiles it or loads a NEFF from cache, so it runs tens of seconds
    against a steady state of hundreds of milliseconds. Averaging it in would
    corrupt throughput. The established convention is to record every step but
    exclude the first `warmup` from the median, and to report the first step
    separately so the compile cost stays visible instead of being smoothed away.
    """

    def __init__(self, warmup: int = 10):
        self.warmup = warmup
        self.rows: list[dict] = []
        self._t0 = time.time()
        self._last = self._t0

    def tick(self, loss: float | None) -> float:
        now = time.time()
        ms = (now - self._last) * 1000.0
        self._last = now
        step = len(self.rows) + 1
        self.rows.append({
            "step": step,
            "loss": None if loss is None else round(float(loss), 4),
            "ms": round(ms, 2),
            "warmup": step <= self.warmup,
        })
        return ms

    @property
    def wall_s(self) -> float:
        return self._last - self._t0

    def _steady(self) -> list[float]:
        return [r["ms"] for r in self.rows if not r["warmup"]]

    def metrics(self) -> dict:
        steady = self._steady()
        first = self.rows[0]["ms"] if self.rows else None
        median = statistics.median(steady) if steady else None
        losses = [r["loss"] for r in self.rows if r["loss"] is not None]
        return {
            "steps_recorded": len(self.rows),
            "warmup_steps_excluded": min(self.warmup, len(self.rows)),
            "first_step_ms": None if first is None else round(first, 2),
            "first_step_note": (
                "Step 1 traces the XLA graph and then either compiles it or loads "
                "it from the NEFF cache. A warm cache collapses this toward the "
                "median; a cold cache leaves tens of minutes in it."),
            "median_step_ms": None if median is None else round(median, 2),
            "compile_s": (None if (first is None or median is None)
                          else round(max(0.0, (first - median) / 1000.0), 1)),
            "compile_s_note": "first_step_ms - median_step_ms, floored at 0.",
            "train_wall_s": round(self.wall_s, 2),
            "final_loss": losses[-1] if losses else None,
            "loss_trace": self.rows,
        }


def steady_tokens_per_s(log: StepLog, tokens_per_step: int) -> float | None:
    """Throughput from the STEADY-STATE median, not the end-to-end average.

    End-to-end would fold compile time into throughput and make a cold-cache run
    look like slow silicon. Both numbers are emitted; this is the one that
    describes the chip.
    """
    steady = log._steady()
    if not steady:
        return None
    return tokens_per_step / (statistics.median(steady) / 1000.0)


def emit_result(path: str | os.PathLike, *, tag: str, model: str, dataset: str,
                stage: str, params: dict, config: dict, log: StepLog,
                tokens_per_step: int, profile: dict,
                gradient_checkpointing: bool = False,
                extra: dict | None = None) -> dict:
    """Write a result JSON in the schema analysis/make_report.py already reads.

    Emits BOTH throughput conventions. `tokens_per_s` is the steady-state figure
    that characterises the hardware; `tokens_per_s_end_to_end` includes compile
    and is what a practitioner actually waits through. Reporting only one of
    them is how a benchmark becomes either optimistic or unfair.
    """
    m = log.metrics()
    tps = steady_tokens_per_s(log, tokens_per_step)
    trainable = params.get("params_trainable") or 0
    frozen = params.get("params_frozen") or 0

    out: dict = {
        "tag": tag,
        "stage": stage,          # pretrain | dpo | orpo | grpo -- new in Phase 4
        "model": model,
        "dataset": dataset,
        **params,
        "config": config,
        "versions": resolve_versions(),
        "tokens_per_optimizer_step": tokens_per_step,
        **m,
    }

    if tps is not None:
        out["tokens_per_s"] = round(tps, 1)
        out.update({
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in throughput_metrics(
                trainable, frozen, tps,
                peak_flops=profile["peak_bf16_flops"],
                gradient_checkpointing=gradient_checkpointing,
                peak_flops_alt=profile.get("peak_bf16_flops_alt"),
            ).items()
        })
        out["flops_formula"] = "6*params_trainable + 4*params_frozen" + (
            " + 2*(trainable+frozen) for recompute" if gradient_checkpointing else "")
        out["peak_bf16_flops"] = profile["peak_bf16_flops"]
        out["peak_bf16_flops_source"] = profile.get("peak_source")

    if m["train_wall_s"]:
        e2e = (tokens_per_step * m["steps_recorded"]) / m["train_wall_s"]
        out["tokens_per_s_end_to_end"] = round(e2e, 1)
        out["tokens_per_s_note"] = (
            "tokens_per_s is the steady-state median, excluding the "
            "compile-bearing warmup steps -- it describes the chip. "
            "tokens_per_s_end_to_end includes compile and is the wall-clock a "
            "practitioner actually experiences.")

    # Device HBM high-water is not readable from torch on XLA. Phase 1-3 leave
    # it null and point at the telemetry CSV rather than estimating it, and this
    # lane keeps that promise instead of inventing a number.
    out.setdefault("peak_device_mem_mib", None)
    out.setdefault("peak_device_mem_note",
                   "Not available from torch on XLA. Read mem_used_mib from the "
                   "matching telemetry CSV (1 Hz neuron-monitor). Deliberately "
                   "null rather than estimated.")

    if extra:
        out.update(extra)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    return out


def failure_receipt(path: str | os.PathLike, *, tag: str, stage: str, box: str,
                    reason: str, detail: str = "", config: dict | None = None,
                    extra: dict | None = None) -> dict:
    """Record a lane that did not produce a number, and WHY.

    METHODOLOGY treats a documented failure as a terminal outcome rather than a
    hole -- a wall that is measured and explained is a finding. The GRPO/RLVR
    probe is expected to land here, and that receipt is the deliverable.
    """
    out = {
        "tag": tag,
        "stage": stage,
        "box": box,
        "status": "failed",
        "reason": reason,
        "detail": detail[-4000:],
        "config": config or {},
        "versions": resolve_versions(),
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        out.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    return out
