#!/usr/bin/env python
"""DPO and ORPO on Trainium, by re-basing TRL's trainers onto NeuronTrainer.

WHY THIS FILE EXISTS
--------------------
optimum-neuron 0.4.3 ships exactly one alignment trainer: NeuronSFTTrainer. The
DPO/ORPO support AWS documents lives in `neuronx-distributed-training`, a
separate NeMo-style package that is NOT installed here and that expects
HF checkpoints converted to its own sharded format.

But NeuronSFTTrainer is not a bespoke implementation. It is built by stealing
TRL's methods and reparenting them:

    _SFTTrainer = type("_SFTTrainer", (NeuronTrainer,), SFTTrainer.__dict__.copy())

NeuronTrainer is a from-scratch trainer (its base is `object`, not
transformers.Trainer) that deliberately mirrors the transformers.Trainer surface
-- including create_accelerator_and_postprocess, so `self.accelerator` exists.
TRL's DPOTrainer and ORPOTrainer are transformers.Trainer subclasses of exactly
the same shape as SFTTrainer, so the identical trick applies to them. That is
what this file does, and it avoids both the extra install and the checkpoint
conversion.

THE PART THAT IS NOT FREE: FIXED SHAPES
---------------------------------------
Neuron compiles static-shape graphs ahead of time. TRL's preference collators
pad to the longest sequence IN EACH BATCH, so shapes vary batch to batch and
every new shape triggers a recompile -- which on this stack costs tens of
seconds to minutes. optimum-neuron solves this for SFT with a collator that pads
to exactly max_length; there is no shipped equivalent for preference data, so
FixedShapeCollator below wraps whatever collator TRL builds and pads every
sequence tensor to one constant length.

WHY DPO NEEDS NO SEPARATE REFERENCE MODEL HERE
----------------------------------------------
DPO normally holds two copies of the policy -- trainable and frozen reference.
Two 8B models do not fit in 16 GiB per core. With a LoRA policy, TRL computes
reference log-probabilities by DISABLING THE ADAPTER on the same weights
(ref_model=None + a PEFT model), so the base model serves as its own reference
at zero extra memory. That is the single lever that makes 8B DPO expressible on
this chip, and it is why the DPO lane is LoRA rather than full-parameter.

ORPO needs no reference model at all by construction -- its odds-ratio penalty
is computed from the policy alone. The cost delta between the two is itself a
reportable result rather than an implementation detail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase4_lib as P4  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["dpo", "orpo", "cpo"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device-profile", default=os.environ.get("NP_DEVICE", "trn1"))
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO kl_beta / ORPO lambda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--full-finetune", type=int, default=0,
                    help="ORPO only; DPO requires LoRA for the implicit reference")
    # ORPO's whole point is to replace SFT+DPO in one step FROM A BASE MODEL, so
    # this lane deliberately uses meta-llama/Llama-3.1-8B rather than -Instruct.
    # A base model has no chat template, and UltraFeedback's pairs are
    # conversational, so TRL cannot render them:
    #   ValueError: Cannot use chat template functions because
    #   tokenizer.chat_template is not set ...
    # Borrowing the matching Instruct model's template is the standard remedy
    # and keeps the tokenisation identical to what the preference data assumes.
    # It is recorded in the result rather than applied silently, because the
    # template determines what the model is actually trained on.
    ap.add_argument("--precompute-ref-logps", action="store_true",
                    help="DPO only: compute reference log-probs once before "
                         "training so the reference forward leaves the "
                         "training-step graph entirely")
    ap.add_argument("--chat-template-from", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="source model for a chat template when the policy lacks one")
    ap.add_argument("--gradient-checkpointing", type=int, default=1)
    ap.add_argument("--nproc-per-node", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    return ap.parse_args(argv)


def torchrun_argv(script, argv, nproc):
    return [sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}", script, *argv]


# ---------------------------------------------------------------------------
# Fixed-shape collator
# ---------------------------------------------------------------------------
class FixedShapeCollator:
    """Pad every sequence tensor a TRL preference collator produces to ONE length.

    Wraps rather than replaces the TRL collator: TRL owns the semantics of how
    chosen/rejected get tokenised and aligned, and reimplementing that would be
    a second source of truth for what a preference pair means. This only fixes
    the shape afterwards.

    Pad values matter and are not interchangeable. input_ids take the tokenizer
    pad id; attention_mask takes 0 so padding is masked out of attention; labels
    take -100 so padding contributes no loss. Getting labels wrong here would
    train the model to predict pad tokens and the loss would look plausible
    while being meaningless.
    """

    def __init__(self, inner, pad_token_id: int, max_length: int):
        self.inner = inner
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def _pad_value(self, key: str):
        if key.endswith("labels"):
            return -100
        if key.endswith("attention_mask"):
            return 0
        return self.pad_token_id

    def __call__(self, features):
        import torch
        batch = self.inner(features)
        for key, val in list(batch.items()):
            if not isinstance(val, torch.Tensor) or val.dim() != 2:
                continue
            if not key.endswith(("input_ids", "attention_mask", "labels")):
                continue
            cur = val.shape[1]
            if cur == self.max_length:
                continue
            if cur > self.max_length:
                batch[key] = val[:, : self.max_length]
            else:
                pad = torch.full((val.shape[0], self.max_length - cur),
                                 self._pad_value(key), dtype=val.dtype,
                                 device=val.device)
                batch[key] = torch.cat([val, pad], dim=1)
        return batch


def _clone_rebound(fn, new_cls):
    """Copy a function, rebinding its `__class__` closure cell to new_cls.

    Python compiles a bare `super()` into `super(__class__, self)`, where
    `__class__` is a CLOSURE CELL captured at class-creation time and bound to
    the class the method was textually defined in. Copying `__dict__` into a new
    class copies references to the SAME function objects, whose cells still
    point at the original class -- so `super()` searches for trl.DPOTrainer in
    an MRO that does not contain it and raises

        TypeError: super(type, obj): obj must be an instance or subtype of type

    which is exactly what the GRPO probe hit at stage D. optimum-neuron dodges
    this for SFT by writing a bespoke __init__ that calls
    NeuronTrainer.__init__(self, ...) explicitly instead of via super().

    Cloning with a fresh cell fixes it generally, and WITHOUT mutating TRL's
    originals -- rebinding the cells in place would corrupt trl.DPOTrainer for
    every other user in the process.
    """
    import types
    if not isinstance(fn, types.FunctionType):
        return fn
    fv = fn.__code__.co_freevars
    if "__class__" not in fv or not fn.__closure__:
        return fn
    cells = list(fn.__closure__)
    cells[fv.index("__class__")] = types.CellType(new_cls)
    clone = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__,
                               fn.__defaults__, tuple(cells))
    clone.__kwdefaults__ = fn.__kwdefaults__
    clone.__dict__.update(fn.__dict__)
    clone.__qualname__ = fn.__qualname__
    return clone


class HFTrainerCompat:
    """The transformers.Trainer surface NeuronTrainer does not implement.

    TRL was written against transformers.Trainer. NeuronTrainer reimplements
    most of that surface but not all of it, and the gaps only surface deep
    inside training -- the first one (is_deepspeed_enabled) appeared after a
    four-minute 8B checkpoint load, which is an expensive way to learn one
    attribute name.

    So the gap was enumerated STATICALLY rather than one AttributeError at a
    time: AST-diff everything TRL reads against everything NeuronTrainer
    provides (methods, class vars, and self.X assigned in any method). For DPO
    that is exactly four names, for ORPO two. Each has an unambiguous value on
    this hardware, so none of the following is guesswork:

        is_deepspeed_enabled  Neuron has no DeepSpeed integration at all.
        is_fsdp_enabled       Sharding on Neuron is NxD's job, not FSDP.
        _prepare_inputs       Move a batch to the XLA device.
        create_model_card     Writes a Hub README; nothing is pushed here.

    Kept at module scope, and free of any optimum/torch import at class-creation
    time, so the contract is testable without a Trainium instance.
    """

    is_deepspeed_enabled = False
    is_fsdp_enabled = False
    is_fsdp_xla_enabled = False
    hub_model_id = None

    #: Names this mixin exists to supply. Asserted against TRL in the test suite.
    COVERS = ("is_deepspeed_enabled", "is_fsdp_enabled", "is_fsdp_xla_enabled",
              "hub_model_id", "_prepare_inputs", "create_model_card",
              "model_wrapped", "current_gradient_accumulation_steps",
              "compute_loss_context_manager", "log")

    def log(self, logs, start_time=None):
        """Absorb the arity difference between Trainer.log and NeuronTrainer.log.

        transformers.Trainer.log is log(self, logs, start_time=None) and TRL's
        DPO/ORPO overrides forward both positionally:

            trl/trainer/orpo_trainer.py:1020
                return super().log(logs, start_time)

        optimum-neuron's NeuronTrainer.log is log(self, logs) -- no start_time.
        Because the re-based class puts TRL's override in front of
        NeuronTrainer, that super() call lands on a two-argument function with
        three arguments and raises

            TypeError: NeuronTrainer.log() takes 2 positional arguments but 3
            were given

        which on 2026-08-14 killed the ORPO lane INSIDE optimum-neuron's own
        logging step-closure, after the model had compiled and steps had run.

        start_time is only used by transformers to annotate its own
        tokens-per-second estimate. This study times steps itself with StepLog
        and reports that number, so dropping the annotation costs nothing that
        is reported. The signature is inspected rather than assumed, so if a
        future optimum-neuron adds start_time it will be passed through instead
        of silently discarded.
        """
        import inspect
        parent = super().log
        try:
            params = inspect.signature(parent).parameters
        except (TypeError, ValueError):        # C-implemented or unintrospectable
            params = {}
        accepts = ("start_time" in params
                   or any(p.kind is p.VAR_KEYWORD for p in params.values()))
        return parent(logs, start_time) if accepts else parent(logs)

    def _prepare_inputs(self, inputs):
        """Move a batch onto the XLA device, leaving non-tensors alone."""
        import torch as _torch
        import torch_xla.core.xla_model as _xm
        try:
            import torch_xla as _txla
            dev = _txla.device() if hasattr(_txla, "device") else _xm.xla_device()
        except Exception:
            dev = _xm.xla_device()
        if isinstance(inputs, dict):
            return {k: (v.to(dev) if isinstance(v, _torch.Tensor) else v)
                    for k, v in inputs.items()}
        return inputs.to(dev) if isinstance(inputs, _torch.Tensor) else inputs

    def create_model_card(self, *a, **k):
        return None

    # Cheap extras TRL's other trainers read. Harmless here, and they keep this
    # mixin useful beyond DPO/ORPO.
    @property
    def model_wrapped(self):
        return getattr(self, "model", None)

    @model_wrapped.setter
    def model_wrapped(self, value):
        self._model_wrapped = value

    @property
    def current_gradient_accumulation_steps(self):
        return getattr(self.args, "gradient_accumulation_steps", 1)

    def compute_loss_context_manager(self):
        # NeuronTrainer already owns the autocast policy; defer to it.
        return self.autocast_smart_context_manager()


def _make_shim(NeuronTrainer):
    """A class that stands where transformers.Trainer stands in TRL's MRO.

    NeuronTrainer accepts a NARROWER __init__ than transformers.Trainer -- no
    model_init, no compute_metrics, no preprocess_logits_for_metrics. TRL's
    trainers pass those through, so a direct delegation would die on unexpected
    keywords. This filters them.

    Filtering silently would be worse than crashing, so anything dropped with a
    NON-None value is reported: that is a real semantic loss (a compute_metrics
    that never runs is a metric quietly missing from the report), not a
    formality.
    """
    import inspect

    allowed = set(inspect.signature(NeuronTrainer.__init__).parameters) - {"self"}

    class _NeuronTrainerShim(HFTrainerCompat, NeuronTrainer):
        def __init__(self, *args, **kwargs):
            # TRL may pass positionally; map onto transformers.Trainer's order.
            if args:
                try:
                    from transformers import Trainer as _HFTrainer
                    names = [p for p in inspect.signature(_HFTrainer.__init__).parameters
                             if p != "self"]
                    for name, val in zip(names, args):
                        kwargs.setdefault(name, val)
                except Exception:
                    kwargs.setdefault("model", args[0])

            dropped = {k: v for k, v in kwargs.items()
                       if k not in allowed and v is not None}
            if dropped:
                print(f"[align] NeuronTrainer does not accept "
                      f"{sorted(dropped)} -- dropped (non-None, so this is a real "
                      f"behaviour difference from the TRL default)", flush=True)
            NeuronTrainer.__init__(
                self, **{k: v for k, v in kwargs.items() if k in allowed})
            # Set by transformers.Trainer.__init__, read by TRL's shared base.
            self._train_batch_size = getattr(
                self.args, "per_device_train_batch_size", 1)
            self.is_in_train = False

    return _NeuronTrainerShim


def strip_unsupported_forward_kwargs(model, drop=("use_cache",),
                                     drop_even_if_true=()):
    """Drop kwargs optimum-neuron's gradient-checkpointed forward cannot take.

    TRL calls the policy with `use_cache=` and `output_hidden_states=`, which is
    fine against a stock transformers model. optimum-neuron's training Llama
    routes each decoder layer through torch_xla.utils.checkpoint.checkpoint when
    gradient checkpointing is on, and that helper rejects anything it does not
    recognise:

        ValueError: Unexpected keyword arguments: use_cache,output_hidden_states

    Turning gradient checkpointing OFF would also avoid it, by not taking the
    checkpoint path at all -- but that trades away the activation memory an 8B
    DPO run on 16 GiB/core actually needs, to fix a signature mismatch. Dropping
    the kwargs is the cheaper trade.

    `drop` names are removed only when FALSY; a truthy one raises, because
    silently discarding a real request would change what the loss is computed
    from while still producing a plausible number. Training never wants a KV
    cache, so use_cache=False always qualifies.

    `drop_even_if_true` is the deliberate exception, and it is only ever passed
    after checking the call site. TRL sets output_hidden_states=True in two
    places in dpo_trainer.py (verified against trl 0.24.0):

        line 1371, in _compute_loss_liger      -- READS outputs.last_hidden_state
                                                  and feeds it to the fused loss
        line 1599, in concatenated_forward     -- sets it and never reads it;
                                                  the only reference after that
                                                  line is the assignment itself

    concatenated_forward is the path taken here, and the Liger path requires the
    liger-kernel package, which is not installed on this box. So the flag is
    vestigial for this configuration and dropping it changes nothing. The caller
    asserts use_liger_loss is off before granting the exception -- if that ever
    becomes true, the name must move back to `drop` and this lane must fail
    loudly rather than quietly compute a different loss.
    """
    original = model.forward

    def forward(*args, **kwargs):
        for key in drop_even_if_true:
            kwargs.pop(key, None)
        for key in drop:
            if key in kwargs:
                value = kwargs.pop(key)
                if value:
                    raise ValueError(
                        f"{key}={value!r} was requested, but optimum-neuron's "
                        f"gradient-checkpointed forward cannot accept it. "
                        f"Dropping a truthy {key} would silently change the "
                        f"computation, so this lane refuses to continue.")
        return original(*args, **kwargs)

    model.forward = forward
    return model


def guard_orpo_numerics(trainer, algo, say):
    """Stop ORPO's odds-ratio loss from producing NaN on bf16 hardware.

    MEASURED FAILURE THIS PREVENTS
    ------------------------------
    REPORT-EXTENSIONS 32.10: the 150-step ORPO lane "diverged to NaN at step 23"
    and was "non-finite for 128 of its 150 steps", while still reporting an
    entirely ordinary 1,011.2 tok/s -- because NaN costs the same FLOPs as a
    number. 32.10 recorded a HYPOTHESIS (a completion emptied by truncation) and
    explicitly did not test it. Reading trl 0.24's source resolves it, and the
    hypothesis was half right: there are TWO NaN paths, both reachable, and both
    rooted in the same thing -- a pathologically short chosen/rejected
    completion.

    PATH 1 -- orpo_trainer.py:721, the mean itself
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    A completion with zero label tokens divides 0/0. This IS 32.10's hypothesis,
    now confirmed as a real code path rather than a guess.

    PATH 2 -- orpo_trainer.py, odds_ratio_loss
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps)))
    orpo_trainer.py:786 passes average_log_prob=True, so these are MEAN
    per-token log-probs -- small negatives, and exp() of them lands near 1.0.
    bf16 spacing in [0.5,1) is 2**-9, so exp(x) rounds to EXACTLY 1.0 once
    x > -0.000977 (per-token probability > 99.9023%), which then feeds
    log1p(-1.0) = -inf. Measured, there are THREE degradation modes, not one:

        chosen saturates only    -> loss = +0.000000, gradient vanishes SILENTLY
        rejected saturates only  -> loss = -inf
        BOTH saturate            -> -inf - (-inf) = NaN   <- 32.10's step 23

    The single-sided modes are arguably worse than the NaN: they do not show up
    in a loss trace at all. In fp32 the threshold is -2.98e-08, i.e. unreachable,
    so every one of these is a PRECISION artifact rather than an algorithm one.

    A one-or-two-token completion that the model finds obvious clears -0.000977
    easily, which is why the 30-step lanes (320 pairs) stayed finite and the
    150-step lane (1,400 pairs) did not: same seed, same shapes, different draw.

    WHY BOTH ARE FIXED AND NOT JUST ONE
    -----------------------------------
    The filter alone cannot help path 2 -- a 40-token completion can still be
    predicted at 99.9%. The fp32 recast alone cannot help path 1 -- 0/0 is NaN
    in every precision. They are independent failures with a shared cause.

    Counts are RECORDED, not hidden: a lane that silently dropped data would
    reproduce 32.6's "two wrongs cancelling" problem in a new place.
    """
    import torch

    if algo not in ("orpo", "cpo"):
        return {"applied": False, "reason": "guard applies to orpo/cpo only"}

    stats = {"applied": True, "algo": algo,
             "nonfinite_logps": 0, "nonfinite_batches": 0, "clamped_values": 0,
             "_clamped_dev": None}

    # PATH 1 applies to BOTH: cpo_trainer.py:839 sets average_log_prob=True for
    # loss_type in {ipo, simpo}, which reaches the same /loss_mask.sum(-1).
    # CPO carries no log1p(-exp(...)) at all, so PATH 2 is ORPO-only.
    _logps = type(trainer).get_batch_logps

    @staticmethod
    def safe_get_batch_logps(*a, **kw):
        out = _logps(*a, **kw)
        if isinstance(out, torch.Tensor):
            bad = ~torch.isfinite(out)
            n = int(bad.sum())
            if n:
                stats["nonfinite_logps"] += n
                out = torch.where(bad, torch.full_like(out, -1e-2), out)
        return out

    type(trainer).get_batch_logps = safe_get_batch_logps

    if algo == "cpo":
        say(f"[{algo}] logp guard installed (0/0 path only; CPO has no log1p term)")
        return stats

    inner = type(trainer).odds_ratio_loss

    # bf16 saturation threshold, with a safety factor. Below this, exp() cannot
    # round to 1.0 even in bf16.
    LOGP_CEIL = -1e-2

    def safe_odds_ratio_loss(self, policy_chosen_logps, policy_rejected_logps):
        # fp32 for the whole odds-ratio computation: path 2 is purely a
        # precision artifact and disappears at fp32's 2**-24 spacing.
        c = policy_chosen_logps.float()
        r = policy_rejected_logps.float()

        # Path 1: a zero-token completion arrives here as NaN already (0/0
        # upstream). Replace with the ceiling so the batch survives; the count
        # is what gets reported.
        bad = ~(torch.isfinite(c) & torch.isfinite(r))
        if bool(bad.any()):
            stats["nonfinite_batches"] += 1
            c = torch.where(bad, torch.full_like(c, LOGP_CEIL), c)
            r = torch.where(bad, torch.full_like(r, LOGP_CEIL), r)

        # Path 2: clamp away from 0 so exp() cannot reach 1.0.
        #
        # THE COUNTER MUST NOT SYNC. The first version of this guard wrote
        #     n_clamp = int((c > LOGP_CEIL).sum() + (r > LOGP_CEIL).sum())
        # and int() on a lazy tensor forces a device->host transfer EVERY STEP,
        # severing the XLA graph -- the exact failure neutralise_out_of_graph_
        # gathers exists to prevent, reintroduced in new code. Measured cost:
        # 1,181 -> 806 tok/s (-32%), median step 13,873 -> 20,327 ms.
        # Accumulate on device instead and read once, after training.
        over = (c > LOGP_CEIL).sum() + (r > LOGP_CEIL).sum()
        stats["_clamped_dev"] = over if stats["_clamped_dev"] is None else stats["_clamped_dev"] + over
        c = c.clamp(max=LOGP_CEIL)
        r = r.clamp(max=LOGP_CEIL)

        return inner(self, c, r)

    type(trainer).odds_ratio_loss = safe_odds_ratio_loss
    say(f"[{algo}] odds-ratio guard installed: fp32 + logp clamp at {LOGP_CEIL} "
        f"(bf16 exp() saturates to 1.0 above -0.000977)")
    return stats


def build_neuron_trainer_class(algo: str):
    """Mirror optimum-neuron's re-basing trick for DPO / ORPO, generalised.

    Returns (TrainerClass, ConfigClass). Kept in one place so the two algorithms
    cannot drift apart in how they are constructed.
    """
    from optimum.neuron.trainers import NeuronTrainer
    from optimum.neuron.trainers.training_args import NeuronTrainingArguments

    if algo == "dpo":
        from trl import DPOConfig as _Cfg, DPOTrainer as _Trainer
    elif algo == "cpo":
        # CPO is reference-free like ORPO, so it needs no second model. Its one
        # .generate() call sits at cpo_trainer.py:940, inside generate_during_eval
        # -- the SAME call site as orpo_trainer.py:895, which the proven ORPO lane
        # never reaches. loss_type="simpo" is reachable from here for free.
        from trl import CPOConfig as _Cfg, CPOTrainer as _Trainer
    else:
        from trl import ORPOConfig as _Cfg, ORPOTrainer as _Trainer

    shim = _make_shim(NeuronTrainer)

    # These are set by type() itself; carrying copies over confuses class
    # creation ("__dict__ slot conflicts with class variable").
    ns = {k: v for k, v in _Trainer.__dict__.items()
          if k not in ("__dict__", "__weakref__", "__classcell__")}
    base = type(f"_Neuron{algo.upper()}Trainer", (shim,), ns)
    for key, val in list(ns.items()):
        rebound = _clone_rebound(val, base)
        if rebound is not val:
            setattr(base, key, rebound)

    @dataclass
    class NeuronAlignConfig(NeuronTrainingArguments, _Cfg):
        def __post_init__(self):
            # Same reason optimum-neuron forces this off for SFT: padding-free
            # flattening produces variable shapes, and variable shapes recompile.
            if hasattr(self, "padding_free"):
                self.padding_free = False
            super().__post_init__()

    return base, NeuronAlignConfig


def neutralise_out_of_graph_gathers(trainer, algo, nproc, tp, say):
    """Stop TRL's per-step metric gathers from cutting the lazy graph.

    WHAT BREAKS
    -----------
    trl 0.24's DPOTrainer.get_batch_loss_metrics ends with eight consecutive
    lines of the form

        metrics["rewards/chosen"] = self.accelerator.gather_for_metrics(t).mean().item()

    On CUDA that is a cheap all-gather. On Neuron it routes through
    optimum-neuron's Accelerator.gather(out_of_graph=True) ->
    _xla_gather_one -> xm.mesh_reduce -> _maybe_convert_to_cpu ->
    _XLAC._xla_sync_multi, i.e. it materialises device tensors to host in the
    middle of a training step. Each one severs the lazy graph at whatever point
    that rank happens to have reached.

    The two TP ranks do not reach those points with identical pending work. In
    the 2026-08-14 run the ranks were compiling two different module hashes
    concurrently -- rank 0 MODULE_3412766081022117232, rank 1
    MODULE_12367629996638363887 -- and the compile that rank 0 was forced into
    died with

        RunNeuronCCImpl: error condition !(error != 400):
        TypeError: stat: path should be string, bytes, os.PathLike or integer,
        not NoneType

    Eleven graphs had already compiled cleanly before this, so neither the
    model nor the data was ever the problem.

    WHY IDENTITY IS EXACT HERE, NOT AN APPROXIMATION
    ------------------------------------------------
    gather_for_metrics exists to collect a statistic computed on *different*
    data on each rank -- that is, across the data-parallel dimension. This lane
    runs pure tensor parallelism: nproc == tp, so the data-parallel size is 1
    and every rank sees the same batch and computes the same reward and logp
    scalars. Gathering therefore concatenates k identical copies, and the
    .mean() that immediately follows returns exactly what the ungathered tensor
    would have. Returning the tensor unchanged is arithmetically identical, and
    it removes eight host syncs per step.

    So this is only applied when dp_size == 1. If a future lane runs any real
    data parallelism the gather is left alone, because then it would actually
    be averaging different numbers and skipping it would silently report only
    rank 0's metrics.
    """
    dp = max(1, nproc // max(1, tp))
    acc = getattr(trainer, "accelerator", None)
    if acc is None or not hasattr(acc, "gather_for_metrics"):
        say(f"[{algo}] no accelerator.gather_for_metrics to neutralise")
        return False
    # A NOTE ON THE PRECOMPUTE PATH, WHICH THIS ONCE REFUSED TO TOUCH.
    #
    # Not every gather in TRL is a metric gather. DPO's precompute_ref_log_probs
    # path calls gather_for_metrics((ref_chosen_logp, ref_rejected_logp)) and
    # then Dataset.add_column()s the result, which needs one value per ROW OF
    # THE DATASET. This function used to bail out whenever that flag was set,
    # on the theory that identity would add a 16-row column to a 32-row dataset.
    #
    # That theory silently assumed data parallelism. It is wrong at dp == 1.
    # The precompute loader (dpo_trainer.py:836) is built over the whole
    # train_dataset and prepared by an accelerator whose world has no data
    # dimension, so EVERY rank iterates ALL rows -- local length == global
    # length. Gathering there concatenates k identical copies of the full
    # column and add_column raises `Length of values (640) doesn't match length
    # of dataset (320)`. Identity is not merely safe on this path, it is the
    # only correct choice, for the same dp == 1 reason as everywhere else.
    if dp != 1:
        say(f"[{algo}] data-parallel size {dp} > 1 -- leaving gather_for_metrics "
            f"alone, its result is not redundant here")
        return False
    acc.gather_for_metrics = lambda tensor, *a, **k: tensor
    say(f"[{algo}] gather_for_metrics -> identity (dp=1, tp={tp}): removes the "
        f"8 out-of-graph host syncs TRL performs per step")
    return True


def resolve_xla_device():
    """The one place that answers "which device". Separate so it is swappable
    in tests that have no torch_xla, and so there is a single definition."""
    import torch_xla.core.xla_model as _xm
    try:
        import torch_xla as _txla
        return _txla.device() if hasattr(_txla, "device") else _xm.xla_device()
    except Exception:
        return _xm.xla_device()


def xla_mark_step():
    """Flush the pending lazy graph. Separate for the same reason as
    resolve_xla_device: one definition, and swappable where torch_xla is absent."""
    import torch_xla.core.xla_model as _xm
    _xm.mark_step()


def device_place_reference_precompute(trainer, algo, say):
    """Move precompute batches onto the XLA device before the reference forward.

    THE TWO FAILURES THIS FIXES
    ---------------------------
    Both are the same root cause seen from two ends, and neither reached the
    compiler -- which matters, because the three earlier DPO attempts died
    *inside* neuronx-cc. These die before it.

    Round 15 (batch on the host):

        dpo_trainer.py:840   compute_ref_log_probs(padded_batch)
        neuronx_distributed/parallel_layers/mappings.py:147  reduce_scatter(...)
        RuntimeError: Expected XLA tensor. Got: CPUBFloat16Type

    Round 16 (MODEL on the host -- `torch.FloatTensor` is the embedding
    weight, not the input, so moving the batch alone just moved the error one
    frame earlier):

        neuronx_distributed/parallel_layers/layers.py:349  F.embedding(weight, input, ...)
        RuntimeError: Expected XLA tensor. Got: torch.FloatTensor

    WHY THE MODEL IS ON THE HOST AT ALL
    -----------------------------------
    An ordering mismatch between the two libraries. TRL hangs its precompute
    pass off `get_train_dataloader()`. optimum-neuron's `_train` calls that at
    transformers.py:1103 -- and only afterwards calls `setup_training`, whose
    `self.model = self.accelerator.prepare_model(self.model)` is what puts the
    weights on the device. So at precompute time the model has never been
    placed. Nothing in either library is wrong on its own; the hook simply
    fires one step too early on this stack.

    So the fix has two halves: place the model, then route the batch through
    the same `_prepare_inputs` the training loop uses. Placing the model early
    is exactly what optimum-neuron's own error message instructs callers to do
    ("Make sure ... `model.to(xm.xla_device())` is performed before the
    optimizer creation"), and `prepare_model` later is idempotent with respect
    to a model already on the device. No optimizer exists yet, so the
    model/optimizer device check at transformers.py:211 cannot fire.
    """
    fn = getattr(trainer, "compute_ref_log_probs", None)
    if fn is None:
        say(f"[{algo}] no compute_ref_log_probs to wrap")
        return False

    placed = {"done": False}

    def on_device(batch, _inner=fn, _t=trainer):
        if not placed["done"]:
            dev = resolve_xla_device()
            _t.model.to(dev)
            xla_mark_step()          # settle the parameter transfers on their own
            placed["done"] = True
            say(f"[{algo}] model placed on {dev} before the reference pass: "
                f"TRL precomputes from get_train_dataloader(), which "
                f"optimum-neuron calls before setup_training() places weights")
        out = _inner(_t._prepare_inputs(batch))
        # Materialise here, on our terms, and hand back host tensors.
        #
        # Round 17 got the reference forward to COMPILE (two `Compiler status
        # PASS` lines) and then died on the very next statement:
        #
        #   dpo_trainer.py:844  ref_chosen_logps.append(ref_chosen_logp.cpu())
        #   RuntimeError: BufferMapAdd: error condition !(((buffer) != nullptr))
        #
        # TRL calls .cpu() on a lazy tensor whose graph has not been flushed.
        # Cutting the graph explicitly and doing the transfer ourselves means
        # TRL's .cpu() lands on a tensor that is already on the host, where it
        # is a no-op. Everything downstream of it -- torch.cat, .float(),
        # .numpy(), Dataset.add_column -- is host-side arithmetic that never
        # wanted an XLA tensor in the first place.
        xla_mark_step()
        if isinstance(out, tuple):
            return tuple(o.cpu() if hasattr(o, "cpu") else o for o in out)
        return out.cpu() if hasattr(out, "cpu") else out

    trainer.compute_ref_log_probs = on_device
    say(f"[{algo}] compute_ref_log_probs -> place model + _prepare_inputs(batch): "
        f"TRL's precompute loader yields host tensors against host weights, and "
        f"the TP collectives inside a sharded model reject both")
    return True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    profile = P4.profile_for(args.device_profile)
    nproc = args.nproc_per_node or profile["nproc"]
    tp = args.tensor_parallel_size or profile["tp"]

    if "LOCAL_RANK" not in os.environ:
        os.execv(sys.executable, torchrun_argv(__file__, argv, nproc))

    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from datasets import load_dataset
    from transformers import AutoTokenizer

    rank = xr.global_ordinal() if hasattr(xr, "global_ordinal") else xm.get_ordinal()
    master = rank == 0

    def say(*a):
        if master:
            print(*a, flush=True)

    out_path = Path(args.out)
    fail_path = out_path.with_suffix(".failure.json") if out_path.suffix == ".json" \
        else Path(str(out_path) + ".failure.json")

    cfg_record = {
        "algo": args.algo, "base_model": args.model, "dataset": args.dataset,
        "split": args.split, "n_samples": args.n_samples,
        "max_length": args.max_length, "max_prompt_length": args.max_prompt_length,
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
        "lr": args.lr, "beta": args.beta, "seed": args.seed,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "tensor_parallel_size": tp, "world_size": nproc,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        # Filled in below if the policy had no template of its own. The chat
        # template decides how a preference pair is rendered into tokens, so a
        # borrowed one is a property of the experiment, not an implementation
        # detail.
        "chat_template_source": None,
    }

    def ensure_accelerate_state(where):
        """Make sure accelerate's shared state exists RIGHT NOW.

        ORPOTrainer.__init__ calls logger.warning() at orpo_trainer.py:289 --
        before it ever reaches super().__init__ and before NeuronTrainer builds
        its accelerator -- and accelerate's logging refuses to run without a
        state:

            RuntimeError: You must initialize the accelerate state by calling
            either `PartialState()` or `Accelerator()` before using the logging
            utility.

        Calling PartialState() once at the top of main() is NOT enough: it
        succeeds there and the state is gone again by the time the trainer is
        built, cleared somewhere in config construction or model loading. So
        this is checked and re-established immediately before each use, and it
        reports what it found rather than silently papering over the ordering.
        """
        try:
            from accelerate.state import PartialState
            existed = bool(PartialState._shared_state)
            if not existed:
                PartialState()
            ok = bool(PartialState._shared_state)
            print(f"[{args.algo}] accelerate state @{where}: "
                  f"{'already present' if existed else 'initialised'} -> {ok}",
                  flush=True)
            return ok
        except Exception as exc:  # noqa: BLE001 -- informative, not fatal
            print(f"[{args.algo}] accelerate state @{where} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return False

    try:
        ensure_accelerate_state("startup")
        TrainerCls, ConfigCls = build_neuron_trainer_class(args.algo)
        say(f"[{args.algo}] re-based trainer MRO: "
            f"{' -> '.join(c.__name__ for c in TrainerCls.__mro__[:4])}")

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        chat_template_source = None
        if getattr(tok, "chat_template", None) in (None, "") and args.chat_template_from:
            donor = AutoTokenizer.from_pretrained(args.chat_template_from)
            if getattr(donor, "chat_template", None):
                tok.chat_template = donor.chat_template
                chat_template_source = args.chat_template_from
                cfg_record["chat_template_source"] = chat_template_source
                say(f"[{args.algo}] {args.model} has no chat template; borrowed "
                    f"one from {args.chat_template_from} (recorded in the result)")

        ds = load_dataset(args.dataset, split=args.split)
        if args.n_samples > 0:
            ds = ds.shuffle(seed=args.seed).select(range(min(args.n_samples, len(ds))))
        keep = {"prompt", "chosen", "rejected"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        say(f"[{args.algo}] {len(ds)} preference pairs, columns {ds.column_names}")

        conf = ConfigCls(
            output_dir=f"/tmp/np-{args.tag}",
            per_device_train_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            bf16=True,
            gradient_checkpointing=bool(args.gradient_checkpointing),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            tensor_parallel_size=tp,
            disable_tqdm=True,
            # ----------------------------------------------------------------
            # The DPO workaround: take the reference forward OUT of the step.
            # ----------------------------------------------------------------
            # DPO failed three times compiling the graph flushed at the first
            # .item() in get_batch_loss_metrics. That graph contains the policy
            # forward+backward, the optimizer, AND the adapter-disabled
            # reference forward, all in one lazy trace. ORPO, whose graph has no
            # reference pass, compiles and runs -- which is what isolated the
            # reference forward as the blocker.
            #
            # precompute_ref_log_probs runs the reference over the whole dataset
            # ONCE, before training, and stores the log-probs as dataset
            # columns. The training step then contains no reference forward at
            # all, so its graph has the same shape as ORPO's -- the shape known
            # to compile on this stack.
            #
            # This does not change the objective. DPO's loss is defined against
            # a FROZEN reference; recomputing identical numbers every step is an
            # implementation convenience, not part of the maths. TRL ships this
            # flag for exactly that reason.
            **({"precompute_ref_log_probs": True}
               if (args.algo == "dpo" and args.precompute_ref_logps) else {}),
        )

        from optimum.neuron.models.training import NeuronModelForCausalLM
        from optimum.neuron.peft import get_peft_model
        from peft import LoraConfig

        model = NeuronModelForCausalLM.from_pretrained(args.model, conf.trn_config)

        use_lora = not (args.algo == "orpo" and args.full_finetune)
        if use_lora:
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]))

        if use_lora:
            # ---------------------------------------------------------------
            # Freeze the parameter counts to remove one mid-step sync.
            # ---------------------------------------------------------------
            # DPO gets its reference model for free by entering
            # peft.disable_adapter(). That context manager opens with a
            # bookkeeping call:
            #
            #   disable_adapter -> get_model_status -> get_nb_trainable_parameters
            #     -> optimum.neuron ... _get_model_param_count -> xm.mark_step()
            #
            # optimum-neuron's override all-reduces the counts across the TP
            # group, so it has to sync, and a mark_step() there cuts the graph
            # at an arbitrary point in the step.
            #
            # NOTE ON A WRONG DIAGNOSIS. This was first written believing it
            # was the cause of the 2026-08-14 DPO failure
            #   RunNeuronCCImpl: error condition !(error != 400):
            #   TypeError: stat: path should be string ... not NoneType
            # It was not. Pinning the counts changed nothing and the identical
            # error recurred; reading the traceback rather than guessing at it
            # showed the sync came from gather_for_metrics (see
            # neutralise_out_of_graph_gathers below), which is a different call
            # site in a different library. The pin is kept because removing a
            # per-step collective is still correct, not because it fixed
            # anything.
            #
            # Parameter counts do not change during training. Computing them
            # once and returning them thereafter is exact, not an approximation,
            # and it removes a sync that only ever existed to print a status.
            _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _all = sum(p.numel() for p in model.parameters())
            model.get_nb_trainable_parameters = lambda: (_trainable, _all)
            say(f"[{args.algo}] pinned param counts "
                f"({_trainable:,} trainable / {_all:,} total) so "
                f"disable_adapter() cannot trigger a mid-step compile")

        # Patch the OUTERMOST model -- the object TRL actually calls -- so the
        # kwargs are gone before PEFT forwards them down to the checkpointed
        # decoder layers.
        #
        # output_hidden_states may only be dropped while the Liger fused loss is
        # off; that is the branch which actually consumes hidden states. Assert
        # it rather than trust it.
        uses_liger = bool(getattr(conf, "use_liger_loss", False))
        if uses_liger:
            raise RuntimeError(
                "use_liger_loss=True consumes outputs.last_hidden_state, which "
                "optimum-neuron's checkpointed forward cannot produce. Refusing "
                "to run rather than silently compute a different loss.")
        model = strip_unsupported_forward_kwargs(
            model, drop=("use_cache",), drop_even_if_true=("output_hidden_states",))

        # The state established at startup does not survive config construction
        # and model loading, so re-establish it here -- immediately before the
        # trainer __init__ that actually logs.
        ensure_accelerate_state("pre-trainer")

        kwargs = dict(model=model, args=conf, train_dataset=ds,
                      processing_class=tok)
        if args.algo == "dpo":
            # None + a PEFT policy => the adapter is disabled to produce the
            # reference log-probs. This is the whole reason 8B DPO fits.
            kwargs["ref_model"] = None
        trainer = TrainerCls(**kwargs)

        neutralise_out_of_graph_gathers(trainer, args.algo, nproc, tp, say)

        orpo_guard = guard_orpo_numerics(trainer, args.algo, say)

        if getattr(conf, "precompute_ref_log_probs", False):
            device_place_reference_precompute(trainer, args.algo, say)

        # Fix the shapes AFTER construction: TRL builds its own collator inside
        # __init__, and we want to wrap exactly the one it chose.
        if getattr(trainer, "data_collator", None) is not None:
            trainer.data_collator = FixedShapeCollator(
                trainer.data_collator, tok.pad_token_id, args.max_length)
            say(f"[{args.algo}] collator wrapped -> fixed {args.max_length}-token shapes")

        log = P4.StepLog(warmup=10)

        # NeuronTrainer drives its own loop, so step timing is captured with a
        # callback rather than by owning the loop as the pretrain lane does.
        from transformers import TrainerCallback

        class Ticker(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    log.tick(logs["loss"])

        trainer.add_callback(Ticker())
        trainer.train()

        # A lane that "succeeds" with no step timings is worse than one that
        # fails: emit_result would happily write a JSON with null tokens_per_s
        # and null MFU, and a null in a results directory reads as "measured
        # nothing" rather than "the instrumentation missed". NeuronTrainer runs
        # its own loop and is not obliged to dispatch on_log the way
        # transformers.Trainer does, so this is a real possibility, not a
        # defensive flourish.
        if not log.rows:
            raise RuntimeError(
                "training completed but the Ticker callback recorded zero "
                "steps -- NeuronTrainer did not dispatch on_log with a 'loss' "
                "key, so there is no step timing to report. Refusing to emit a "
                "result with null throughput.")

        # Counts and token accounting come from sft_lora, not from a local sum.
        # Both were wrong here before: a rank-local parameter sum halves
        # FLOPs/token under TP, and multiplying tokens by nproc doubles
        # throughput under TP. The two errors cancelled in MFU and left tok/s
        # 2x too high -- see phase4_lib's note. Safe to sync here: training has
        # already finished, so the all-reduce cannot perturb a measured step.
        # Safe to sync now: training has finished, so materialising the
        # guard's device-side counter cannot perturb a measured step.
        if orpo_guard.get("_clamped_dev") is not None:
            try:
                orpo_guard["clamped_values"] = int(orpo_guard.pop("_clamped_dev"))
            except Exception as exc:
                orpo_guard["clamped_values"] = f"unreadable: {exc!r}"
        orpo_guard.pop("_clamped_dev", None)

        counts = P4.count_parameters(model, torch, tp)
        trainable, total = counts["trainable"], counts["total"]

        # TP shares one micro-batch across ranks; only DP multiplies tokens.
        dp_size = max(1, nproc // max(1, tp))
        tokens_per_step = 2 * P4.tokens_per_optimizer_step(  # chosen + rejected
            args.max_length, args.micro_batch, args.grad_accum, dp_size)

        if master:
            res = P4.emit_result(
                args.out, tag=args.tag, stage=args.algo, model=args.model,
                dataset=f"{args.dataset}:{args.split}",
                params={
                    "params_total": total,
                    "params_trainable": trainable,
                    "params_frozen": total - trainable,
                    "params_count_method": counts["method"],
                    "params_local_total": counts["local_total"],
                    "params_local_trainable": counts["local_trainable"],
                    "params_note": P4.PARAM_METHOD_NOTES.get(
                        counts["method"],
                        "Whole-model counts, obtained the same way as every "
                        "other LoRA lane in this study."),
                },
                config={**cfg_record,
                        "orpo_numerics_guard": orpo_guard,
                        "orpo_numerics_guard_note": (
                            "32.10 left the ORPO NaN unexplained. trl 0.24 has two "
                            "reachable NaN paths, both from a pathologically short "
                            "completion: orpo_trainer.py:721 divides by "
                            "loss_mask.sum(-1) (0/0 on an empty completion), and "
                            "odds_ratio_loss takes log1p(-exp(mean_logp)) where bf16 "
                            "rounds exp() to exactly 1.0 once mean_logp > -0.000977, "
                            "giving log1p(-1) = -inf and -inf - -inf = NaN. The guard "
                            "recasts to fp32 and clamps; counts above are how often "
                            "each fired. Zero counts mean the run never approached "
                            "either path, NOT that the guard was inactive."),
                        "reference_model": ("implicit: adapter disabled on the "
                                            "policy (ref_model=None + PEFT)")
                        if args.algo == "dpo" else "none (ORPO is reference-free)",
                        "data_parallel_size": dp_size,
                        "tokens_per_step_note": (
                            "max_length * micro_batch * grad_accum * dp_size * 2. "
                            "The 2x counts chosen AND rejected: a preference step "
                            "forwards both sequences, so its token cost is twice an "
                            "SFT step of equal length. dp_size, NOT the world size: "
                            "tensor parallelism shares one micro-batch across ranks "
                            "and multiplies neither batch nor tokens."),
                        "flops_caveat": (
                            "MFU uses the study-wide 6*trainable + 4*frozen. For DPO "
                            "that UNDER-counts: the reference log-probs cost an extra "
                            "forward over both sequences, roughly a further "
                            "2*(trainable+frozen) per token. The DPO MFU below is "
                            "therefore a lower bound on real utilisation. ORPO is "
                            "reference-free and needs no such caveat."
                            if args.algo == "dpo" else
                            "ORPO is reference-free: one forward-backward over chosen "
                            "and rejected, so 6*trainable + 4*frozen applies "
                            "unmodified and this MFU is directly comparable to the "
                            "SFT lanes."),
                        "trainer_construction": (
                            "TRL trainer re-based onto optimum-neuron NeuronTrainer "
                            "via type(name,(NeuronTrainer,),Trainer.__dict__.copy()), "
                            "mirroring how optimum-neuron builds NeuronSFTTrainer"),
                        },
                log=log, tokens_per_step=tokens_per_step, profile=profile,
                gradient_checkpointing=bool(args.gradient_checkpointing))

            # DPO runs a SECOND forward over both sequences, with the adapter
            # disabled, to get the reference log-probs. That work is real and
            # the study-wide 6*trainable + 4*frozen convention does not charge
            # for it, so the primary mfu_pct above is a LOWER BOUND for DPO.
            #
            # The primary figure is kept on the shared convention so it stays
            # comparable to the SFT and pretrain lanes; the reference-inclusive
            # figure is emitted beside it rather than replacing it, so a reader
            # can use whichever comparison they actually want. A forward pass
            # over the frozen base is 2*frozen per token (the adapter is off, so
            # the LoRA parameters do no work).
            if args.algo == "dpo" and res.get("tokens_per_s"):
                base = P4.lora_flops_per_token(trainable, total - trainable)
                if args.gradient_checkpointing:
                    base += 2.0 * total
                ref_fpt = base + 2.0 * (total - trainable)
                tflops = res["tokens_per_s"] * ref_fpt / 1e12
                res["flops_per_token_with_reference_pass"] = ref_fpt
                res["tflops_with_reference_pass"] = round(tflops, 3)
                res["mfu_pct_with_reference_pass"] = round(
                    100.0 * tflops * 1e12 / profile["peak_bf16_flops"], 3)
                res["reference_pass_note"] = (
                    f"mfu_pct ({res.get('mfu_pct')}%) uses the study-wide "
                    f"6*trainable+4*frozen and is comparable to the SFT lanes. "
                    f"mfu_pct_with_reference_pass "
                    f"({res['mfu_pct_with_reference_pass']}%) adds the 2*frozen "
                    f"per token that DPO's adapter-disabled reference forward "
                    f"actually executes, and is the better estimate of how busy "
                    f"the chip really was.")
                out_p = Path(args.out)
                out_p.write_text(json.dumps(res, indent=2) + "\n")

            say(f"[{args.algo}] done: {res.get('tokens_per_s')} tok/s "
                f"MFU {res.get('mfu_pct')}%")
        return 0

    except Exception as exc:  # noqa: BLE001 -- a wall is a finding, record it
        if master:
            P4.failure_receipt(
                fail_path, tag=args.tag, stage=args.algo, box=args.device_profile,
                reason=f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(), config=cfg_record)
            print(f"[{args.algo}] FAILED -> {fail_path}", flush=True)
            traceback.print_exc()
        return 4


if __name__ == "__main__":
    sys.exit(main())
