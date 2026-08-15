#!/usr/bin/env python
"""Pretrain a ~362M Llama FROM SCRATCH through optimum-neuron's NeuronTrainer.

WHY THIS EXISTS ALONGSIDE pretrain_fineweb.py
---------------------------------------------
`pretrain_fineweb.py` owns its training loop. On this stack that loop recompiles
the XLA graph on every optimizer step -- measured, reproducible, and still
unexplained after two attributions were made and both falsified on hardware
(REPORT-EXTENSIONS §32.8). Compiles reached 54 minutes each, which made further
iteration unaffordable.

Every OTHER training lane in this study runs through optimum-neuron's
`NeuronTrainer`, and none of them shows that behaviour: the SFT lane reaches
68.3% MFU and the ORPO lane 30.2%, both with a single steady-state graph. That
is a strong signal that the defect lives in hand-rolled lazy-tensor code rather
than in the hardware -- but §32.8 could only say the framework path was NEVER
TESTED, because it had not been.

This script tests it. Same architecture, same corpus, same sequence length,
micro-batch, grad-accum, LR schedule, seed and accounting.

IT IS NOT A CLEAN ONE-VARIABLE COMPARISON, and saying so was an error worth
correcting: the hand-written lane ran DP=2 across both NeuronCores, while this
one is forced onto ONE core (see PARALLELISM below). Loop ownership AND core
count both change. So a successful run here demonstrates that the FRAMEWORK
path trains; it does not by itself prove the hand-written loop caused the
per-step recompile. Closing that gap would need the hand loop re-run at
nproc=1. Flagged by an independent reviewer, 2026-08-15.

Two outcomes and both are worth the instance time:
  - it trains  -> the recompile is confirmed as a property of the hand-written
                  loop, and the phase gets its pretraining number
  - it recompiles too -> the defect is in the stack, not in my loop, which is a
                  far more important finding than a throughput figure

ACCEPTANCE GATE, fixed before the run so the result cannot be read generously
after the fact. Two independent reviewers converged on this:
  - steps 1-3 may introduce any finite number of graphs (warmup: weight-init
    sharding, optimizer-state creation, the step graph itself)
  - steps 4-10 must introduce ZERO new MODULE hashes and ZERO compile events
  - steps 4-10 median step time stable within +/-20%, all losses finite
"Step 2 was fast" is NOT the criterion; a per-step recompile can hide behind one
quick step.

WHY A RANDOM-INIT MODEL NEEDS A ROUND TRIP THROUGH DISK
-------------------------------------------------------
`NeuronModelForCausalLM` is optimum-neuron's training model class, and its only
constructor is `from_pretrained` -- it maps HF weights onto sharded Neuron
modules as it loads. There is no "build me an empty one" entry point. So rank 0
materialises a randomly-initialised `LlamaForCausalLM` on CPU, saves it, and
every rank loads it back. The round trip costs about a gigabyte of scratch and
a few seconds; it is not a workaround for a bug, it is how you hand this API a
model it did not download.

PARALLELISM: TP=2, AND NOT BY CHOICE
------------------------------------
A 362M model fits on one NeuronCore, so data parallelism is what this lane
wants -- tensor parallelism on a model this small measures the interconnect
rather than the compute. optimum-neuron does not offer it. Setting
`tensor_parallel_size=1` makes it build a world of tp x pp = 1, ignore the
torchrun world, and leave rank 1 outside the collective bootstrap, where the
Neuron runtime aborts it. Measured, with the exact assertion, in main().

So TP=2, which is also what every other lane in this study uses, keeping the
whole-chip MFU convention intact. dp is therefore 1 and
`tokens_per_optimizer_step` must NOT multiply by the world -- the same rule as
the preference lanes. See phase4_lib for the bug that taught this study the
difference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase4_lib as P4  # noqa: E402
from pretrain_fineweb import ARCH  # noqa: E402  -- one architecture, one source


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="uint16 token memmap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="pretrain_nt")
    ap.add_argument("--device-profile", default=os.environ.get("NP_DEVICE", "trn1"))
    ap.add_argument("--nproc-per-node", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scratch", default="/opt/np/models/scratch_init")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    return ap.parse_args(argv)


def torchrun_argv(script, argv, nproc):
    return [sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}", script, *argv]


class PackedTokens:
    """Fixed-length windows over a uint16 memmap, as a map-style torch Dataset.

    Shapes are constant by construction, which is the entire requirement Neuron
    places on a data pipeline: a collator that pads per batch produces a new
    shape and therefore a new graph. Nothing here can vary between steps.

    Labels are the input ids. optimum-neuron's Llama shifts internally, exactly
    as transformers does, so shifting here would train the model to predict the
    token it was just given.
    """

    def __init__(self, path, seq_len, seed, limit_windows=None):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        n = (len(self.arr) - 1) // seq_len
        if n <= 0:
            raise SystemExit(f"{path}: too small for seq_len {seq_len}")
        rng = np.random.default_rng(seed)
        self.order = rng.permutation(n)
        if limit_windows:
            self.order = self.order[:limit_windows]

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        import torch
        s = int(self.order[i]) * self.seq_len
        ids = np.asarray(self.arr[s:s + self.seq_len], dtype=np.int64)
        t = torch.from_numpy(ids)
        return {"input_ids": t, "attention_mask": torch.ones_like(t), "labels": t.clone()}


def collate(features):
    import torch
    return {k: torch.stack([f[k] for f in features]) for k in features[0]}


def materialise_random_init(scratch: str, seed: int, say) -> str:
    """Write a randomly-initialised Llama to disk so from_pretrained can read it."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    p = Path(scratch)
    if (p / "config.json").exists():
        say(f"[pretrain-nt] reusing random init at {p}")
        return str(p)
    torch.manual_seed(seed)
    cfg = LlamaConfig(**ARCH)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    n = sum(x.numel() for x in model.parameters())
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p), safe_serialization=True)
    say(f"[pretrain-nt] materialised {n/1e6:.1f}M-param random init -> {p}")
    del model
    return str(p)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    profile = P4.profile_for(args.device_profile)
    nproc = args.nproc_per_node or profile["nproc"]

    if "LOCAL_RANK" not in os.environ:
        os.execv(sys.executable, torchrun_argv(__file__, argv, nproc))

    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
    from optimum.neuron.models.training import NeuronModelForCausalLM
    from transformers import TrainerCallback

    rank = xr.global_ordinal() if hasattr(xr, "global_ordinal") else xm.get_ordinal()
    master = rank == 0

    def say(*a):
        if master:
            print(*a, flush=True)

    out_path = Path(args.out)
    fail_path = (out_path.with_suffix(".failure.json") if out_path.suffix == ".json"
                 else Path(str(out_path) + ".failure.json"))

    # ----------------------------------------------------------------------
    # TP=2, even though a 362M model fits on one core -- because the framework
    # gives no choice.
    # ----------------------------------------------------------------------
    # The intent was DP=2: tensor parallelism on a model this small measures the
    # interconnect rather than the compute. optimum-neuron does not support it.
    # With tensor_parallel_size=1 and pipeline_parallel_size=1 it builds its
    # world as tp x pp = 1, ignores the torchrun world entirely, and logs
    #
    #   > initializing data parallel with size 1
    #   > initializing world size to 1
    #
    # while torchrun has in fact started two ranks. Rank 1 is then not in the
    # collective bootstrap list and the Neuron runtime aborts it outright:
    #
    #   KaenaRuntime/enc/replica_groups.cc:76: set_my_group(...):
    #   Assertion `it != bootstrap_participants.end()' failed.   (SIGABRT)
    #
    # measured 2026-08-15. So NeuronTrainingArguments has no data-parallel
    # dimension of its own: the only way to use both NeuronCores is tensor
    # parallelism. TP=2 it is, which also keeps this lane on the same whole-chip
    # convention as every other lane in the study. The forced choice is itself
    # the result -- see REPORT-EXTENSIONS.
    tp_size = args.tensor_parallel_size or profile["tp"]
    world = nproc
    dp_size = max(1, world // tp_size)

    # ----------------------------------------------------------------------
    # AND TP=2 IS NOT AVAILABLE EITHER, because of the architecture's head count.
    # ----------------------------------------------------------------------
    # Tensor parallelism shards attention heads across ranks, so the head count
    # must divide the TP degree. SmolLM2-360M has FIFTEEN attention heads (and
    # five KV heads). 15 is odd. neuronx-distributed refuses:
    #
    #   AssertionError: 15 is not divisible by 2
    #
    # measured 2026-08-15. Combined with the missing data-parallel path above,
    # that leaves exactly one usable topology for this architecture: a single
    # NeuronCore. Which is a real constraint worth stating plainly -- a
    # published model's head count decides which AWS silicon topologies are open
    # to it, and an odd head count closes all the multi-core ones.
    #
    # The alternative was to change the architecture until it sharded. This lane
    # exists to measure a REAL published configuration (see pretrain_fineweb's
    # docstring); reshaping the model to flatter the hardware would defeat that,
    # so the lane runs on one core and is labelled a half-chip configuration
    # wherever it appears, per the rule in METHODOLOGY.
    heads, kv = ARCH["num_attention_heads"], ARCH["num_key_value_heads"]
    if tp_size > 1 and (heads % tp_size or kv % tp_size):
        raise SystemExit(
            f"architecture has {heads} attention heads and {kv} KV heads; "
            f"neither divides tensor_parallel_size={tp_size}. Re-run with "
            f"--nproc-per-node 1 --tensor-parallel-size 1 (a half-chip "
            f"configuration, and label it as one) rather than reshaping the "
            f"model to fit the hardware.")

    cfg_record = {
        "architecture": "SmolLM2-360M shape (published config, not tuned here)",
        **ARCH,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "lr_schedule": "constant",
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "world_size": world,
        "tensor_parallel_size": tp_size,
        "data_parallel_size": dp_size,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "loop_owner": "optimum-neuron NeuronTrainer",
        "contrast": ("pretrain_probe_capturable ran the SAME architecture, "
                     "corpus and shapes through a HAND-WRITTEN XLA loop and "
                     "recompiled every step. The only variable changed here is "
                     "who owns the training loop."),
    }

    try:
        # ------------------------------------------------------------------
        # Filesystem barrier, NOT xm.rendezvous().
        # ------------------------------------------------------------------
        # The obvious way to make rank 1 wait for rank 0's checkpoint is
        # xm.rendezvous(). It aborts the process:
        #
        #   KaenaRuntime/enc/replica_groups.cc:76: set_my_group(...):
        #   Assertion `it != bootstrap_participants.end()' failed.   (SIGABRT)
        #
        # A collective issued before optimum-neuron has built its parallel state
        # forces the runtime to initialise a DEFAULT one -- tp=1, dp=1, world=1
        # -- and the second torchrun rank is not in that bootstrap list, so the
        # runtime kills it. The framework's own tp/dp configuration is not
        # established until NeuronTrainingArguments and the model load.
        #
        # Measured 2026-08-15: this fired identically at tp=1 and tp=2, which is
        # what proved it was the barrier and not the parallelism setting.
        #
        # A marker file needs no collective and no runtime state at all.
        ready = Path(args.scratch) / ".ready"
        if master:
            materialise_random_init(args.scratch, args.seed, say)
            ready.write_text("ok\n")
        else:
            for _ in range(600):                      # 10 min, generous
                if ready.exists():
                    break
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"rank {rank}: rank 0 did not produce {ready} within 10 min")
        init_dir = args.scratch

        targs = NeuronTrainingArguments(
            output_dir=f"/tmp/{args.tag}",
            overwrite_output_dir=True,
            do_train=True,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            # Constant, matching the hand-written lane's control run. A per-step
            # LR was the FIRST withdrawn attribution for that lane's recompile;
            # holding it constant here keeps this a one-variable comparison.
            lr_scheduler_type="constant",
            warmup_steps=0,
            bf16=True,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": True}
                if not args.no_gradient_checkpointing else None),
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            disable_tqdm=True,
        )
        model = NeuronModelForCausalLM.from_pretrained(init_dir, targs.trn_config)
        say(f"[pretrain-nt] loaded random init through NeuronModelForCausalLM")

        # Sized with margin: the sampler shards this across dp ranks, and a
        # dataset that runs dry mid-run ends the epoch early and truncates the
        # measurement window rather than failing loudly.
        ds = PackedTokens(args.data, args.seq_len, args.seed,
                          limit_windows=4 * (args.max_steps + 20)
                          * args.micro_batch * args.grad_accum * dp_size)
        say(f"[pretrain-nt] {len(ds)} windows of {args.seq_len} tokens, dp={dp_size}")

        log = P4.StepLog(warmup=args.warmup_steps)

        class Ticker(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    ms = log.tick(logs["loss"])
                    if master and (len(log.rows) <= 3 or len(log.rows) % 10 == 0):
                        print(f"  step {len(log.rows)}/{args.max_steps} "
                              f"loss {logs['loss']:.4f} {ms:.0f}ms", flush=True)

        trainer = NeuronTrainer(model=model, args=targs, train_dataset=ds,
                                data_collator=collate)

        # ------------------------------------------------------------------
        # optimum-neuron 0.4.3: NeuronTrainer sends the model a kwarg the model
        # refuses.
        # ------------------------------------------------------------------
        # trainers/transformers.py:201 decides the model "accepts loss kwargs"
        # by looking for **kwargs in its forward signature. optimum-neuron's own
        # Llama forward HAS **kwargs, so this is True. Then compute_loss does
        #
        #   trainers/transformers.py:978
        #       if num_items_in_batch is not None:
        #           inputs = dict(**inputs, reduction="sum")
        #
        # and that same forward validates its kwargs strictly and raises
        #
        #   ValueError: Unexpected keyword arguments: reduction
        #
        # So a plain NeuronTrainer over a plain causal-LM model cannot complete
        # a single step in 0.4.3. The SFT and preference lanes never hit it
        # because TRL's trainers override compute_loss entirely.
        #
        # Turning the flag off is exact here, not a lossy shortcut. With it on,
        # loss = sum / num_items_in_batch; with it off, loss = mean-per-
        # micro-batch / grad_accum. Those differ only when micro-batches contain
        # different numbers of loss-bearing tokens. Every window in this lane is
        # exactly seq_len tokens with no padding and no -100 labels, and every
        # micro-batch is the same size, so the two are arithmetically identical.
        if getattr(trainer, "model_accepts_loss_kwargs", False):
            trainer.model_accepts_loss_kwargs = False
            say("[pretrain-nt] model_accepts_loss_kwargs -> False: optimum-neuron "
                "0.4.3 would otherwise pass reduction='sum' to a forward that "
                "rejects it (exact here -- every window is a full seq_len)")

        # Cross-check the framework's own view rather than trusting the
        # arithmetic above. NeuronTrainingArguments.world_size returns
        # trn_config.data_parallel_size; if it ever disagrees with world//tp the
        # token accounting must follow the framework, not this script.
        dp_reported = int(getattr(targs, "world_size", 0) or 0)
        if dp_reported and dp_reported != dp_size:
            say(f"[pretrain-nt] dp_size {dp_size} -> {dp_reported} (framework wins)")
            dp_size = dp_reported
        say(f"[pretrain-nt] world={world} tp={tp_size} dp={dp_size}")

        trainer.add_callback(Ticker())
        trainer.train()

        if not log.rows:
            raise RuntimeError(
                "training completed but the Ticker callback recorded zero "
                "steps -- refusing to emit a result with null throughput.")

        counts = P4.count_parameters(model, torch, tp_size)
        tokens_per_step = P4.tokens_per_optimizer_step(
            args.seq_len, args.micro_batch, args.grad_accum, dp_size)

        if master:
            res = P4.emit_result(
                args.out, tag=args.tag, stage="pretrain",
                model=f"llama-arch-{counts['total']/1e6:.0f}M-from-scratch",
                dataset="HuggingFaceFW/fineweb-edu:sample-10BT",
                params={
                    "params_total": counts["total"],
                    "params_trainable": counts["trainable"],
                    "params_frozen": counts["total"] - counts["trainable"],
                    "params_count_method": counts["method"],
                    "params_note": ("Full-parameter pretraining from random "
                                    "initialisation; nothing is frozen, so "
                                    "FLOPs/token is the dense 6N."),
                },
                config={**cfg_record, "data_parallel_size": dp_size,
                        "tokens_per_step_note": (
                            f"seq_len * micro_batch * grad_accum * dp_size, "
                            f"with dp_size={dp_size}. The lane was DESIGNED as "
                            f"DP=2 and could not run that way: "
                            f"NeuronTrainingArguments has no data-parallel "
                            f"dimension, and TP=2 is excluded by the "
                            f"architecture's 15 attention heads. So it ran on "
                            f"one core, dp=1, and the multiplier is 1 -- the "
                            f"same arithmetic as the preference lanes, for a "
                            f"different reason."),
                        "confound_vs_hand_written_loop": (
                            "This lane is NOT a clean one-variable comparison "
                            "against pretrain_probe_capturable. That lane ran "
                            "DP=2 on both NeuronCores under a hand-written "
                            "loop; this one runs on ONE core under "
                            "NeuronTrainer. Loop ownership AND core count both "
                            "changed. A successful run here shows the FRAMEWORK "
                            "path works; it does not by itself prove the "
                            "hand-written loop was the cause of the per-step "
                            "recompile. Closing that would need the hand loop "
                            "re-run at nproc=1. Flagged by an independent "
                            "reviewer; recorded rather than glossed."),
                        "note_plain_torch_xla_does_support_dp2": (
                            "Worth separating: the HAND-WRITTEN lane really did "
                            "run DP=2 across both cores under plain "
                            "torchrun + torch_xla. The missing data-parallel "
                            "dimension is a property of optimum-neuron 0.4.3's "
                            "NeuronTrainingArguments, NOT of Trainium or of "
                            "torch_xla.")},
                log=log, tokens_per_step=tokens_per_step, profile=profile,
                gradient_checkpointing=not args.no_gradient_checkpointing,
                extra={
                "tokens_trained": tokens_per_step * len(log.rows),
                "chip_fraction_used": f"{world}/{profile['nproc']} NeuronCores",
                "half_chip_configuration": world < profile["nproc"],
                "mfu_denominator_note": (
                    "mfu_pct uses the WHOLE-CHIP peak, as every lane in this "
                    "study does, so it stays comparable. When "
                    "half_chip_configuration is true the lane had only "
                    f"{world} of {profile['nproc']} NeuronCores available to "
                    "it, so mfu_pct understates the utilisation of the "
                    "core(s) it actually used by that factor. The per-core "
                    "figure is reported beside it rather than substituted."),
            })
            if res.get("mfu_pct") is not None and world < profile["nproc"]:
                res["mfu_pct_per_core_used"] = round(
                    res["mfu_pct"] * profile["nproc"] / world, 3)
                Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
            say(f"[pretrain-nt] done: {res.get('tokens_per_s')} tok/s "
                f"MFU {res.get('mfu_pct')}%  "
                f"numerically_valid={res.get('loss_numerically_valid')}")
        return 0

    except Exception as exc:  # noqa: BLE001 -- a wall is a finding, record it
        if master:
            P4.failure_receipt(
                fail_path, tag=args.tag, stage="pretrain", box=args.device_profile,
                reason=f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(), config=cfg_record)
            print(f"[pretrain-nt] FAILED -> {fail_path}", flush=True)
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
