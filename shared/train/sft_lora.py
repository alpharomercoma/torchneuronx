#!/usr/bin/env python3
"""Lane 2/4/5: LoRA SFT on AWS Trainium via optimum-neuron's NeuronSFTTrainer.

Usage (exactly how shared/run_all.sh calls it -- plain `python`, no torchrun):

    python shared/train/sft_lora.py \
        --model meta-llama/Llama-3.1-8B-Instruct --tag llama31_lora \
        --out trn1/results/train/llama31_lora.json

    # -> config banner, then one metric line per OPTIMIZER step (first 10 and
    #    then every 10th):
    #      step    1  loss  2.4181  |  148203.7 ms  | warmup
    #      step    2  loss  2.3907  |    1183.2 ms  | warmup
    #      ...
    #      step   20  loss  1.9044  |    1178.9 ms
    #    then a single pasteable line:
    #      SUMMARY tag=llama31_lora model=meta-llama/Llama-3.1-8B-Instruct tp=2 dp=1
    #        median_step_ms=1179.4 first_step_ms=148203.7 compile_s=147.0
    #        tokens_per_s=13897 tflops=31.9 mfu_pct=15.2 peak_host_mem_mib=24817
    #    and trn1/results/train/llama31_lora.json written atomically by rank 0.

    # 20-step plumbing smoke on an ungated 1.1B (lane 2, ~$0.50):
    python shared/train/sft_lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --tag smoke_tinyllama --max-steps 20 --out /tmp/smoke.json

    # under the telemetry sampler, which is how a result becomes reportable:
    python shared/telemetry.py --out run.csv -- python shared/train/sft_lora.py ...

RE-EXEC UNDER TORCHRUN (why this file restarts itself)
------------------------------------------------------
optimum-neuron's distributed training is a torchrun program: every rank is a
separate OS process, ranks are assigned one NeuronCore each, and the tensor
parallel group is built from the torchrun world. There is no in-process way to
ask for TP=2 -- `NeuronTrainingArguments(tensor_parallel_size=2)` only tells the
framework how to SHARD the world torchrun already created. Launched as plain
`python`, WORLD_SIZE is unset, the world is 1, and TP=2 is unsatisfiable.

run_all.sh is byte-identical on both boxes and invokes every lane as
`$PY <script> ...`, so rather than special-casing this lane in the driver, the
script re-execs ITSELF under `python -m torch.distributed.run
--nproc_per_node=2` when WORLD_SIZE is absent. Arguments are parsed BEFORE the
re-exec so a typo fails once, in one process, instead of twice inside torchrun.

WHY world == TP, AND WHY IT IS A PROFILE
----------------------------------------
One rank per logical NeuronCore, and the whole world spent on tensor
parallelism, so DP=1. That is a consequence of the instance, not a tuning
choice, and it is why the token accounting below has no data-parallel
multiplier on either box.

  trn1.2xlarge  1x Trainium1, 2x NeuronCore-v2  -> world 2, TP=2, 16 GiB/core
  trn2.3xlarge  1x Trainium2, 8x NeuronCore-v3 grouped into 4 logical cores
                at LNC=2 (the Neuron default)   -> world 4, TP=4, 24 GiB/core

Llama 3.1 8B in BF16 is ~16 GiB of weights; it would nominally fit on one trn1
core, but activations, gradients and optimizer state would not, so TP is what
makes the primary lane run at all.

These live in DEVICE_PROFILES, selected by --device-profile, else NP_DEVICE,
else the instance type from IMDS, else trn1. The profile also carries the peak
BF16 FLOP/s that MFU is divided by -- so an MFU number can never be silently
compared against the wrong chip. --nproc-per-node and --tensor-parallel-size
override it, which is what makes the trn2 TP ladder a flag change rather than
an edit to this file mid-study.

WHY THE FLOPS FORMULA IS NOT 6N
-------------------------------
See lora_flops_per_token(). Dense training is 6 FLOPs/param/token; LoRA freezes
>99% of the weights and those contribute only 4, not 6. Using 6N everywhere
would overstate MFU by ~50%, which is the difference between "this is a
reasonable result" and "this is a great result". The correction is applied here
rather than in the report so the raw JSON is already honest.

MEASUREMENT CAVEATS THAT ARE RECORDED, NOT HIDDEN
-------------------------------------------------
  * XLA executes lazily and the Neuron runtime pipelines up to
    NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS steps ahead. Individual
    per-step wall times can therefore be skewed by a step or two; the MEDIAN
    over non-warmup steps is the statistic to trust, and the telemetry CSV is
    the independent check that the device was actually busy for that window.
  * peak_device_mem_mib is left null on purpose. Device HBM high-water comes
    from the telemetry CSV (mem_used_mib), which is sampled by neuron-monitor
    outside this process. Fabricating it from a torch API that does not exist
    on XLA would be worse than a null.
  * compile_s is first_step_ms minus median_step_ms: on step 1 XLA traces the
    graph and either compiles it or loads it from the NEFF cache. With a warm
    cache (lane 3 precompile ran) this collapses toward zero, which is the
    point of lane 3. first_step_ms is recorded separately so the two readings
    are never conflated.
  * optimum-neuron ships its own enable_mfu_metrics / enable_throughput_metrics
    training arguments. They are deliberately NOT enabled here -- this repo's
    MFU must be computed the same way on every box in the study -- but they are
    a useful independent cross-check during the on-box smoke run.

Structure note: everything above main() is pure Python with no torch, optimum,
transformers or peft imports, so `import sft_lora` is cheap and testable on a
laptop (tests/test_train_scripts.py). Heavy imports live inside functions.
"""

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import random
import sys
import time

# ------------------------------------------------------------------ defaults
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_SEQ_LEN = 2048
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_MICRO_BATCH = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 3
DEFAULT_SEED = 42
DEFAULT_LR = 1e-4
DEFAULT_WARMUP_STEPS = 5
DEFAULT_ADAPTER_ROOT = "/opt/np/models/adapters"
DEFAULT_CACHE_DIR = "/opt/np/cache/neuron-compile-cache"

# One profile per accelerator generation this study runs on. Everything that
# differs between Trainium1 and Trainium2 lives here and nowhere else, so a
# result JSON always carries the denominator its MFU was computed against.
#
# World = TP on both boxes (one rank per logical NeuronCore, DP = 1): the point
# of this study is per-chip efficiency, so the whole chip goes to one model.
#
# peak_bf16_flops is per CHIP, dense BF16, because TP spreads every token
# across all of the chip's cores.
#
# TWO DENOMINATORS, BOTH REPORTED -- AWS's own sources disagree for Trainium1
# ---------------------------------------------------------------------------
# The Trainium1 ARCHITECTURE page states "190 FP16/BF16/cFP8/TF32 TFLOPS" per
# chip. AWS separately markets trn1.32xlarge at 3.4 PFLOP/s, which over 16
# devices is 212.5 -> the 210e12 this project used for every Phase-1/2 number.
# 190 x 16 = 3.04 PF, so the two AWS figures cannot both be right.
#
# For Trainium2 there is no conflict: the arch page says 667/chip and
# 667 x 16 = 10.7 PF agrees with the instance-level spec.
#
# That asymmetry is the actual bug. Using the INSTANCE figure for trn1 and the
# ARCH figure for trn2 mixes sources in the direction that flatters the newer
# chip -- it understates trn1's MFU and shrinks the peak ratio trn2 has to beat
# from 3.51x to 3.18x. So every lane now reports MFU under BOTH conventions:
#
#   primary ("arch")     trn1 190, trn2 667  -> peak ratio 3.51x
#   alt     ("instance") trn1 210, trn2 667  -> peak ratio 3.18x, and what
#                                               REPORT.md's published 68.3% /
#                                               82.7% were computed against
#
# A reader must be able to see the denominator, not just the ratio. Neither
# figure is "the" truth; what matters is that the two boxes are divided by
# numbers drawn from the same source.
DEVICE_PROFILES = {
    "trn1": {
        "instance_type": "trn1.2xlarge",
        "nproc": 2,
        "tp": 2,
        "pp": 1,
        "peak_bf16_flops": 190e12,
        "peak_source": ("Neuron arch page arch/neuron-hardware/trainium: "
                        "190 FP16/BF16/cFP8/TF32 TFLOPS per chip"),
        "peak_bf16_flops_alt": 210e12,
        "peak_source_alt": ("trn1.32xlarge marketed at 3.4 PFLOP/s / 16 "
                            "devices = 212.5, rounded to 210 -- the "
                            "denominator behind every published Phase-1/2 MFU"),
        "hbm_gib_per_core": 16,
        "hbm_bandwidth_gbs": 820,
        "logical_nc_config": None,   # LNC does not exist on NeuronCore-v2
        "note": ("trn1.2xlarge: 1x Trainium1, 2x NeuronCore-v2, 32 GiB HBM "
                 "(16 GiB per core) at 820 GB/s. 190 TFLOP/s BF16 per chip "
                 "(arch page); 210 under the instance-derived convention."),
    },
    "trn2": {
        "instance_type": "trn2.3xlarge",
        "nproc": 4,
        "tp": 4,
        "pp": 1,
        "peak_bf16_flops": 667e12,
        "peak_source": ("Neuron arch page arch/neuron-hardware/trainium2: "
                        "667 BF16/FP16/TF32 TFLOPS per chip"),
        # No conflicting source for Trainium2: 667 x 16 = 10.7 PF matches the
        # instance spec, so both conventions land on the same number.
        "peak_bf16_flops_alt": 667e12,
        "peak_source_alt": ("same figure; arch and instance-level specs agree "
                            "for Trainium2"),
        "hbm_gib_per_core": 24,
        "hbm_bandwidth_gbs": 2900,
        "logical_nc_config": 2,      # Neuron default on Trainium2
        "note": ("trn2.3xlarge: 1x Trainium2, 8x NeuronCore-v3 grouped into 4 "
                 "logical cores at LNC=2 (the Neuron default), 96 GiB HBM "
                 "(24 GiB per logical core) at 2.9 TB/s. 667 TFLOP/s BF16 "
                 "per chip; 1299 FP8 and 2563 sparse are NOT used here."),
    },
}
DEFAULT_DEVICE = "trn1"

# Module-level aliases kept for the trn1 lanes published in Phase 1/2: those
# numbers must stay byte-reproducible, so the historical names still resolve to
# the historical values. New code reads the profile.
NPROC_PER_NODE = DEVICE_PROFILES[DEFAULT_DEVICE]["nproc"]
TENSOR_PARALLEL_SIZE = DEVICE_PROFILES[DEFAULT_DEVICE]["tp"]
PIPELINE_PARALLEL_SIZE = DEVICE_PROFILES[DEFAULT_DEVICE]["pp"]

# LoRA target modules: every projection in the transformer block. Attention
# only (q,k,v,o) trains ~half as many parameters and reliably underperforms on
# instruction SFT; the MLP projections are where most of the adaptation lands.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Historical alias for the Trainium1 peak; see DEVICE_PROFILES above. Kept so
# throughput_metrics() has the same default it had when Phase-1 numbers were
# recorded, and so a caller that never heard of profiles still gets trn1.
PEAK_BF16_FLOPS = DEVICE_PROFILES[DEFAULT_DEVICE]["peak_bf16_flops"]

# Steps excluded from the median. Step 1 pays graph tracing/compile; the next
# few pay cache warm-up and dataloader spin-up.
WARMUP_STEPS = 10

# Print the first WARMUP_STEPS steps individually (that is where the compile
# signal lives), then every PRINT_EVERY-th step.
PRINT_EVERY = 10

# Set before torch/XLA import in every process, including torchrun children.
# --model-type transformer picks the transformer-specific compiler pipeline;
# --retry_failed_compilation covers the intermittent compiler flakes that
# otherwise kill a multi-hour lane at step 1. MALLOC_ARENA_MAX caps glibc
# per-thread arenas, which is the documented host-OOM mitigation on the 32 GiB
# trn1.2xlarge.
NEURON_ENV = {
    "NEURON_CC_FLAGS": "--model-type transformer --retry_failed_compilation",
    "NEURON_FUSE_SOFTMAX": "1",
    "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS": "3",
    "MALLOC_ARENA_MAX": "64",
}

VERSION_PACKAGES = (
    "torch", "torch-neuronx", "torch-xla", "optimum-neuron", "transformers",
    "trl", "peft", "datasets", "accelerate", "neuronx-cc",
    "neuronx-distributed", "libneuronxla",
)

# The system preamble from optimum-neuron's own Dolly recipe
# (examples/training/llama/finetune_llama.py). Kept verbatim so a loss curve
# from this harness is comparable to the upstream tutorial's.
DOLLY_SYSTEM_PROMPT = (
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 29 Jul 2025\n\n"
    "You are a helpful assistant"
)


# --------------------------------------------------------------------- cli
def build_parser():
    """argparse builder. Top level and import-cheap so tests can introspect it."""
    ap = argparse.ArgumentParser(
        description="LoRA SFT on Trainium via optimum-neuron NeuronSFTTrainer")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HuggingFace model id to fine-tune")
    ap.add_argument("--tag", required=True,
                    help="short lane name; names the adapter dir and the JSON")
    ap.add_argument("--out", required=True, help="metrics JSON path")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps; default None = full --epochs")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help=("Fraction of the dataset held out and evaluated "
                          "BEFORE and AFTER training. Every number this study "
                          "reports otherwise measures speed; none of them "
                          "shows the model actually learned. Held-out loss is "
                          "the minimum credible answer to 'did it train, or "
                          "just run fast?'. 0 disables (the published lanes)."))
    ap.add_argument("--holdout-split-seed", type=int, default=20260805,
                    help="Fixed split seed so both chips hold out the SAME rows.")
    ap.add_argument("--data-seed", type=int, default=None,
                    help=("Separate sampler seed. --seed alone did not change "
                          "the loss trajectory at all on this stack (three "
                          "trn1 seeds produced bit-identical loss), so data "
                          "order is pinned elsewhere -- probably by packing."))
    ap.add_argument("--synthetic-data", type=int, default=0, metavar="N",
                    help=("DATALOADER ISOLATION. Replace the dataset with N "
                          "rows of random token IDs, pre-tokenised to exactly "
                          "--seq-len, with no formatter, no packing and no Hub "
                          "download. Compiled graph shapes are unchanged so "
                          "nothing recompiles. Isolates device throughput from "
                          "host-side dataloader cost -- the leading hypothesis "
                          "for why trn2 is 1.20x at seq 2048 but 1.92x at "
                          "4096. THE LOSS FROM THIS LANE IS MEANINGLESS and "
                          "the result JSON marks it so. 0 disables (every "
                          "published lane)."))
    ap.add_argument("--dataset-fields", default=None,
                    help=("Remap columns onto the Dolly schema the formatter "
                          "expects, e.g. 'context=input,response=output' for "
                          "tatsu-lab/alpaca. Without this a dataset with "
                          "different column names raises KeyError inside the "
                          "formatter, which would look like a training bug "
                          "rather than a schema mismatch."))
    ap.add_argument("--adapter-out", default=None,
                    help=f"default {DEFAULT_ADAPTER_ROOT}/<tag>")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="default: the device profile's tp (trn1 2, trn2 4)")
    ap.add_argument("--device-profile", default=None,
                    choices=sorted(DEVICE_PROFILES),
                    help="accelerator profile: sets ranks, TP and the peak "
                         "FLOP/s that MFU is measured against. Default: "
                         "NP_DEVICE, else the instance type from IMDS, else "
                         f"{DEFAULT_DEVICE}")
    ap.add_argument("--nproc-per-node", type=int, default=None,
                    help="override the profile's rank count (one rank per "
                         "logical NeuronCore). Used by the trn2 TP ladder")
    ap.add_argument("--attn-impl", default="flash_attention_2",
                    help="flash_attention_2 needs seq-len to be a multiple of "
                         "2048; pass eager to disable")
    ap.add_argument("--no-packing", action="store_true",
                    help="disable example packing (packing is ON by default)")
    ap.add_argument("--no-lora", action="store_true",
                    help=("Full fine-tune: update every weight instead of a "
                          "LoRA adapter. Only reachable on Trainium2 -- the "
                          "optimizer state alone (fp32 master + 2 moments = "
                          "12 B/param) exceeds trn1's 16 GiB per core for "
                          "anything past ~1B. Changes the FLOPs accounting to "
                          "dense 6N, which is handled in run()."),)
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="disable activation recomputation (8B models then "
                         "exceed the 16 GB/core HBM limit: NCC_EOOM001)")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="checkpoint every N steps (Track C4 timing lane); "
                         "default: save once at the end only")
    return ap


def resolve_adapter_out(adapter_out, tag, root=DEFAULT_ADAPTER_ROOT):
    """--adapter-out if given, else <root>/<tag>. Pure, so it is unit-tested."""
    return adapter_out if adapter_out else os.path.join(root, tag)


# ------------------------------------------------------------ dolly recipe
def dolly_messages(example, system_prompt=DOLLY_SYSTEM_PROMPT):
    """One Dolly row -> a 3-turn chat.

    This is optimum-neuron's own Dolly recipe: instruction becomes the user
    turn, a non-empty context is appended to it (NOT made a separate turn, so
    the model sees retrieval context the way a RAG caller would send it), and
    response becomes the assistant turn. Pure -- no tokenizer -- so the shape
    is testable without transformers installed.
    """
    user_content = example["instruction"]
    context = example.get("context") or ""
    if len(context) > 0:
        user_content += f"\n\nContext: {context}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["response"]},
    ]


def format_dolly(example, tokenizer, system_prompt=DOLLY_SYSTEM_PROMPT):
    """Dolly row -> one training string, rendered with the MODEL'S OWN template.

    Upstream's tutorial overwrites tokenizer.chat_template with a hand-written
    Llama-3.1 template. This harness deliberately does not: the merged model
    from lane 6 is served on the inf2 box in lane 7, and it must be prompted
    there with the template that ships with the tokenizer. Training on a
    different template than we serve with would make the quality lane measure
    a formatting mismatch instead of the fine-tune.

    Models with no chat template at all (base, non-instruct checkpoints) fall
    back to the plain-text Dolly rendering rather than failing the lane.
    """
    messages = dolly_messages(example, system_prompt=system_prompt)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return plain_dolly_text(example)


def plain_dolly_text(example):
    """Template-free fallback rendering. Pure."""
    parts = [f"### Instruction\n{example['instruction']}"]
    context = example.get("context") or ""
    if len(context) > 0:
        parts.append(f"### Context\n{context}")
    parts.append(f"### Answer\n{example['response']}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- metrics
def lora_flops_per_token(params_trainable, params_frozen):
    """FLOPs per token for a LoRA fine-tune: 6*N_trainable + 4*N_frozen.

    Dense training costs ~6 FLOPs per parameter per token: 2 for the forward
    GEMM, 2 for the gradient w.r.t. the input activations, and 2 for the
    gradient w.r.t. the weights.

    Under LoRA the frozen base weights still pay the forward (2) and still pay
    the activation-gradient GEMM (2, because the backward pass must propagate
    through them to reach the adapters) -- but their weight-gradient GEMM is
    never executed, because they have no gradient. So frozen parameters cost 4,
    not 6, and only the adapter parameters cost the full 6.

    For an 8B model with a rank-16 adapter this is ~4/6 of the dense figure.
    Charging 6N for the frozen base would inflate both TFLOP/s and MFU by ~50%.
    Pure function; unit-tested against hand-computed numbers.
    """
    return 6.0 * params_trainable + 4.0 * params_frozen


def attention_flops_per_token(n_layers, hidden_size, seq_len,
                              gradient_checkpointing=False):
    """Sequence-dependent attention FLOPs per token -- the term 6N omits.

    The parameter-count formula (6*trainable + 4*frozen) covers every GEMM
    whose cost scales with WEIGHTS. It does not cover the two matmuls whose
    cost scales with SEQUENCE LENGTH: scores = Q @ K^T and context = A @ V.
    Per token per layer those are 2*seq*d each, so forward is

        4 * n_layers * seq_len * hidden_size

    and training is ~3x forward (backward is ~2x), plus another forward when
    activations are recomputed.

    WHY THIS MATTERS HERE, AND ONLY HERE
    ------------------------------------
    At seq 2048 on Llama 3.1 8B this is ~6.6% of the parameter term -- small
    enough that Phase 1/2 ignoring it was defensible. It grows LINEARLY with
    sequence length while the parameter term stays flat:

        seq  2048   ~7%      seq  8192   ~27%
        seq  4096   ~13%     seq 16384   ~53%

    The context ladder is the headline of Phase 3, so by seq 16384 the
    published convention would understate FLOPs -- and therefore MFU -- by
    roughly a third. This is reported as an ADDITIONAL field rather than folded
    into flops_per_token, because changing the primary number would silently
    restate every Phase-1/2 result and destroy comparability with trn1.

    GQA note: grouped-query attention shrinks the K/V PROJECTIONS (already in
    the parameter term) but not the score/context matmuls, which still run at
    full head width. So this term is unaffected by the KV-head count.
    """
    if not (n_layers and hidden_size and seq_len):
        return None
    fwd = 4.0 * n_layers * seq_len * hidden_size
    return fwd * (4.0 if gradient_checkpointing else 3.0)


def provenance(args):
    """Everything needed to reconstruct this run's inputs, not just name them."""
    import subprocess
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "git_commit": sh("git -C /opt/np/repo rev-parse HEAD"),
        "git_dirty": bool(sh("git -C /opt/np/repo status --porcelain")),
        "argv": sys.argv,
        "dataset": args.dataset,
        "dataset_fields": args.dataset_fields,
        "holdout_frac": args.holdout_frac,
        "holdout_split_seed": args.holdout_split_seed,
        "model": args.model,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }


def throughput_metrics(params_trainable, params_frozen, tokens_per_s,
                       peak_flops=PEAK_BF16_FLOPS,
                       gradient_checkpointing=False,
                       peak_flops_alt=None):
    """(trainable, frozen, tok/s) -> flops_per_token, TFLOP/s, MFU%. Pure.

    With activation recomputation every parameter pays one extra forward
    GEMM (+2 FLOPs/param/token): frozen 4 -> 6, trainable 6 -> 8. Same
    convention as the GPU study's x4/3 checkpointing adjustment, restated
    for the LoRA-split accounting.

    peak_flops_alt, when given, yields a second MFU under the other published
    denominator. AWS's arch page and instance marketing disagree for Trainium1
    (190 vs 210 TFLOP/s), so a single MFU number would hide which one it used --
    see DEVICE_PROFILES. tflops is denominator-independent and is the number to
    trust when the two disagree.
    """
    fpt = lora_flops_per_token(params_trainable, params_frozen)
    if gradient_checkpointing:
        fpt += 2.0 * (params_trainable + params_frozen)
    achieved = fpt * tokens_per_s
    out = {
        "flops_per_token": fpt,
        "tflops": achieved / 1e12,
        "mfu_pct": 100.0 * achieved / peak_flops,
        "gradient_checkpointing": bool(gradient_checkpointing),
    }
    out["mfu_pct_alt"] = (
        100.0 * achieved / peak_flops_alt if peak_flops_alt else None)

    # MFU ABOVE 100% IS PHYSICALLY IMPOSSIBLE. It means the measurement is
    # wrong, not that the chip is fast, and it must never reach a chart.
    #
    # This fired for real: a control at grad_accum=4 reported 146.7%. The cause
    # is that gradient accumulation is UNROLLED INTO THE COMPILED GRAPH on this
    # stack (changing grad_accum triggers a full recompile), and the per-step
    # timer captures a different fraction of the deferred XLA work at different
    # accumulation depths. The same micro-batch shape measured 1147 ms/micro-
    # batch at accum 8 and 714 ms at accum 4. Steady-state throughput is
    # therefore NOT comparable across grad_accum settings, and the impossible
    # MFU is the only signal that says so out loud.
    if out["mfu_pct"] > 100.0:
        out["mfu_impossible"] = {
            "measured_mfu_pct": round(out["mfu_pct"], 3),
            "why": ("MFU > 100% cannot be physical. The steady-state step timer "
                    "under-measures at this configuration -- most likely a "
                    "grad_accum different from the published lanes, which "
                    "recompiles the graph and changes how much deferred XLA "
                    "work lands inside the timed window."),
            "use_instead": "tokens_per_s_end_to_end / mfu_pct_end_to_end",
        }
    return out


def make_kwarg_tolerant(checkpoint_fn, is_tensor):
    """Wrap a reentrant checkpoint function so non-tensor kwargs survive.

    torch_xla.utils.checkpoint.checkpoint (reentrant, the only variant the
    XLA backend supports) rejects every keyword argument, but transformers'
    GradientCheckpointingLayer forwards layer-config kwargs (use_cache,
    reduction, ...) straight through. Those kwargs are closure state, not
    autograd inputs, so binding them into the function with functools.partial
    is semantically identical -- tensors stay positional so autograd still
    sees them. Pure aside from the wrapped call; unit-tested with a fake
    is_tensor predicate.
    """
    if getattr(checkpoint_fn, "_np_kwarg_shim", False):
        return checkpoint_fn

    @functools.wraps(checkpoint_fn)
    def tolerant(function, *args, **kwargs):
        non_tensor = {k: v for k, v in kwargs.items() if not is_tensor(v)}
        if non_tensor:
            function = functools.partial(function, **non_tensor)
            kwargs = {k: v for k, v in kwargs.items() if is_tensor(v)}
        return checkpoint_fn(function, *args, **kwargs)

    tolerant._np_kwarg_shim = True
    return tolerant


def wrap_checkpoint_funcs(model, is_tensor):
    """Shim every module's bound _gradient_checkpointing_func. Returns count.

    transformers stows the checkpoint callable per-module when
    gradient_checkpointing_enable() runs, so wrapping must happen AFTER that
    call and catches every binding path optimum-neuron uses.
    """
    wrapped = 0
    for module in model.modules():
        fn = getattr(module, "_gradient_checkpointing_func", None)
        if fn is not None and not getattr(fn, "_np_kwarg_shim", False):
            module._gradient_checkpointing_func = make_kwarg_tolerant(
                fn, is_tensor)
            wrapped += 1
    return wrapped


def patch_optimum_modeling_checkpoint(is_tensor):
    """Shim the module-global `checkpoint` in optimum-neuron's modeling files.

    Their custom decoders do `from torch_xla.utils.checkpoint import
    checkpoint` at module scope and call it with layer kwargs (use_cache,
    reduction) -- torch-xla 2.9 made checkpoint() reject all kwargs, so the
    vendor path itself crashes (measured on Llama AND Qwen3, 2026-07-30).
    Patching the per-module _gradient_checkpointing_func attr does nothing
    for these call sites; the imported symbol in each modeling module is the
    only interposition point. Walks sys.modules rather than pkgutil: by the
    time this runs the active model's modeling module is guaranteed imported,
    whereas pkgutil.walk_packages silently skipped the llama subpackage
    (measured: patched list was [granite, utils] while modeling_llama kept
    crashing). Returns the list of patched module basenames.
    """
    patched = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("optimum.neuron.models.training"):
            continue
        fn = getattr(mod, "checkpoint", None) if mod is not None else None
        if callable(fn) and not getattr(fn, "_np_kwarg_shim", False):
            mod.checkpoint = make_kwarg_tolerant(fn, is_tensor)
            patched.append(name.rsplit(".", 1)[-1])
    return patched


def tokens_per_optimizer_step(seq_len, micro_batch, grad_accum, dp_size):
    """Tokens consumed between optimizer steps.

    TP shards ONE model across the cores, so both NeuronCores are working on
    the same micro-batch -- tensor parallelism multiplies neither the batch nor
    the token count. Only data parallelism does, and on trn1.2xlarge dp_size is
    1 because the entire world is spent on TP. Pure.
    """
    return seq_len * micro_batch * grad_accum * dp_size


def median(values):
    """Median of a list, or None if empty. Pure (no statistics import needed)."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mark_warmup(trace, warmup_steps=WARMUP_STEPS):
    """Flag the first `warmup_steps` entries of a loss trace as warmup.

    Flagged by POSITION in the recorded trace, not by global_step, so a resumed
    or step-capped run still excludes its own first steps. Mutates and returns
    the trace. Pure.
    """
    for i, entry in enumerate(trace):
        entry["warmup"] = i < warmup_steps
    return trace


def steady_state_step_ms(trace):
    """Millisecond values from the non-warmup, timed entries of a trace. Pure."""
    return [e["ms"] for e in trace if not e.get("warmup") and e.get("ms") is not None]


def read_peak_host_mem_mib(status_path="/proc/self/status"):
    """Host RSS high-water mark in MiB from VmHWM, or None off Linux.

    VmHWM is the kernel's own peak-RSS accounting, so it survives the fact that
    RSS at exit is lower than RSS during the merge/compile spikes. Deliberately
    psutil-free: the DLAMI venv is not ours to add packages to. Pure enough to
    unit-test against a fixture file.
    """
    try:
        with open(status_path) as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return round(float(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def resolve_versions(packages=VERSION_PACKAGES):
    """Installed versions from importlib.metadata; None when absent. Pure.

    Read from metadata rather than module __version__ attributes so this works
    without importing torch, and so a package that is installed but broken
    still reports its version instead of crashing the run.
    """
    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
        except Exception:
            out[name] = None
    return out


def atomic_write_json(path, payload):
    """Write JSON via tmp+rename so a killed run never leaves half a result.

    run_all.sh treats a non-empty results file as "lane done" and skips it, so
    a truncated JSON would silently poison every later re-run of the suite.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


# ------------------------------------------------------- distributed setup
def is_distributed_launch(env=None):
    """True when we are already inside torchrun. Pure."""
    env = os.environ if env is None else env
    return "WORLD_SIZE" in env or "TORCHELASTIC_RUN_ID" in env


def rank(env=None):
    """This process's global rank (0 when not distributed). Pure."""
    env = os.environ if env is None else env
    try:
        return int(env.get("RANK", "0"))
    except ValueError:
        return 0


def world_size(env=None):
    """torchrun world size (1 when not distributed). Pure."""
    env = os.environ if env is None else env
    try:
        return int(env.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def detect_device_key(env=None, instance_type=None):
    """Pick a DEVICE_PROFILES key. Pure given its inputs, so it is unit-tested.

    Order: explicit instance type (from IMDS, passed in by the caller) beats
    NP_DEVICE in the environment, which beats the trn1 default. The default is
    trn1 rather than "fail loudly" because every Phase-1/2 lane ran without
    this flag existing and must keep reproducing byte-for-byte.
    """
    env = os.environ if env is None else env
    if instance_type:
        family = str(instance_type).split(".")[0].lower()
        if family in DEVICE_PROFILES:
            return family
    requested = (env.get("NP_DEVICE") or "").strip().lower()
    if requested in DEVICE_PROFILES:
        return requested
    return DEFAULT_DEVICE


def read_imds_instance_type(timeout=1.0):
    """Ask IMDSv2 for this box's instance type, or None. Never raises.

    Best-effort only: a wrong answer here would silently change the MFU
    denominator, so every caller also honours --device-profile / NP_DEVICE,
    and the resolved profile is written into the result JSON for audit.
    """
    try:
        import urllib.request

        def _get(url, headers, method="GET"):
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode().strip()

        token = _get("http://169.254.169.254/latest/api/token",
                     {"X-aws-ec2-metadata-token-ttl-seconds": "60"}, "PUT")
        return _get("http://169.254.169.254/latest/meta-data/instance-type",
                    {"X-aws-ec2-metadata-token": token})
    except Exception:
        return None


def resolve_device_profile(args, env=None, instance_type=None):
    """(args, env) -> a DEVICE_PROFILES entry with CLI overrides applied.

    Returns a fresh dict so callers can stash it straight into the result JSON.
    --nproc-per-node and --tensor-parallel-size override the profile; that is
    what makes the trn2 TP ladder (4 -> 2 -> 8 at LNC=1) a flag change rather
    than an edit to this file mid-study.
    """
    key = getattr(args, "device_profile", None)
    if key:
        key = key.strip().lower()
        if key not in DEVICE_PROFILES:
            raise SystemExit(f"unknown --device-profile {key!r}; "
                             f"known: {sorted(DEVICE_PROFILES)}")
    else:
        key = detect_device_key(env=env, instance_type=instance_type)

    profile = dict(DEVICE_PROFILES[key])
    profile["device_profile"] = key
    if getattr(args, "nproc_per_node", None):
        profile["nproc"] = int(args.nproc_per_node)
    if getattr(args, "tensor_parallel_size", None):
        profile["tp"] = int(args.tensor_parallel_size)
    return profile


def torchrun_argv(script_path, argv, nproc_per_node=NPROC_PER_NODE):
    """Build the `python -m torch.distributed.run ...` argv. Pure, so the
    re-exec command is testable without ever executing it.

    `-m torch.distributed.run` rather than the `torchrun` console script: it
    guarantees the children run under the SAME interpreter as the parent, which
    matters on a DLAMI where several Neuron venvs are on the box and PATH order
    decides which `torchrun` you get.
    """
    return [sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            os.path.abspath(script_path)] + list(argv)


def apply_neuron_env(env=None, cache_dir=None, profile=None):
    """Set the Neuron compiler/runtime env vars, without clobbering the user's.

    Must run before torch_xla is imported anywhere, hence before the re-exec:
    the child processes inherit it. Returns the resolved cache dir.
    """
    env = os.environ if env is None else env
    for key, value in NEURON_ENV.items():
        env.setdefault(key, value)
    # Logical NeuronCore config exists only from Trainium2 on. setdefault, so
    # the TP ladder can force LNC=1 from the driver without editing code.
    lnc = (profile or {}).get("logical_nc_config")
    if lnc:
        env.setdefault("NEURON_LOGICAL_NC_CONFIG", str(lnc))
    cache = cache_dir or env.get("NEURON_COMPILE_CACHE_URL") or DEFAULT_CACHE_DIR
    # NEURON_COMPILE_CACHE_URL is the repo-wide convention -- shared/bin/
    # sync_neuron_cache.sh syncs exactly this directory to S3, so a replaced
    # instance never pays the 8B compile twice.
    env["NEURON_COMPILE_CACHE_URL"] = cache
    return cache


# -------------------------------------------------------------------- main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    # IMDS is only worth asking before the re-exec: the children inherit
    # NP_DEVICE below, so exactly one HTTP call happens per lane.
    if not is_distributed_launch() and not args.device_profile \
            and not os.environ.get("NP_DEVICE"):
        detected = read_imds_instance_type()
        if detected:
            print(f"[sft_lora] IMDS instance-type: {detected}", flush=True)
    else:
        detected = None
    profile = resolve_device_profile(args, instance_type=detected)

    cache_dir = apply_neuron_env(profile=profile)
    random.seed(args.seed)

    # Re-exec under torchrun if we were launched as plain `python`. Arguments
    # are already validated at this point, so a bad flag has failed once rather
    # than twice inside the elastic launcher.
    if not is_distributed_launch():
        # Pin the resolved profile for the children so they do not re-detect
        # and cannot disagree with the parent about the MFU denominator.
        os.environ["NP_DEVICE"] = profile["device_profile"]
        cmd = torchrun_argv(__file__, argv, nproc_per_node=profile["nproc"])
        print(f"[sft_lora] profile={profile['device_profile']} "
              f"({profile['instance_type']}) -- re-exec under torchrun "
              f"(--nproc_per_node={profile['nproc']}, one rank per logical "
              f"NeuronCore)", flush=True)
        print(f"[sft_lora] {' '.join(cmd)}", flush=True)
        os.execv(sys.executable, cmd)

    return run(args, cache_dir, profile)


def run(args, cache_dir, profile=None):
    """The distributed body. Every heavy import lives below this line."""
    profile = profile or resolve_device_profile(args)
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback, set_seed

    from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments

    my_rank = rank()
    world = world_size()
    is_main = my_rank == 0

    def say(*parts):
        if is_main:
            print(*parts, flush=True)

    set_seed(args.seed)  # seeds random, numpy and torch in one call

    adapter_out = resolve_adapter_out(args.adapter_out, args.tag)
    packing = not args.no_packing
    tp_size = profile["tp"]
    pp_size = profile["pp"]
    peak_flops = profile["peak_bf16_flops"]
    peak_flops_alt = profile.get("peak_bf16_flops_alt")
    dp_size = max(1, world // tp_size)
    max_steps = args.max_steps if args.max_steps is not None else -1

    # ------------------------------------------------------ training args
    training_args = NeuronTrainingArguments(
        output_dir=adapter_out,
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        lr_scheduler_type="constant",
        bf16=True,
        # Activation recomputation, the vendor-intended way (this also turns
        # on optimum-neuron's trn_config MLP recompute). It only survives
        # because patch_optimum_modeling_checkpoint() shims the torch_xla
        # checkpoint symbol their modeling calls with layer kwargs -- see
        # that helper's docstring for the measured failure.
        gradient_checkpointing=not args.no_gradient_checkpointing,
        # optimum-neuron 0.4.3 crashes on the None default here
        # (sft_trainer.py:263); reentrant is the XLA-supported variant.
        gradient_checkpointing_kwargs=(
            {"use_reentrant": True}
            if not args.no_gradient_checkpointing else None),
        # THE TP KNOB. Verified against optimum-neuron's NeuronTrainingArguments
        # dataclass: the field is `tensor_parallel_size`, and it partitions the
        # torchrun world -- it does not create one.
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        logging_steps=1,          # every optimizer step; the trace needs them all
        # Default: one explicit save at the end. Track C4 overrides to timed
        # periodic saves -- checkpoint cost is a production ops number.
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=1,
        report_to=[],             # no wandb/tensorboard phoning home from the box
        seed=args.seed,
        # HF falls back to `seed` when data_seed is None. Passing it explicitly
        # is the one lever left for varying data order; if the trajectory is
        # STILL bit-identical, packing has pinned the order and that is the
        # finding, not a bug to keep chasing.
        **({"data_seed": args.data_seed} if args.data_seed is not None else {}),
        disable_tqdm=True,        # log files, not a redrawn progress bar
    )

    # dp_size straight from the framework when it exposes it, so the token
    # accounting cannot drift from what actually ran.
    dp_size = int(getattr(training_args, "world_size", dp_size) or dp_size)

    # ------------------------------------------------------------- model
    model = load_neuron_model(args, training_args, torch)
    params = count_parameters(model, torch, tp_size)
    params_total = params["total"]
    params_trainable = params["trainable"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ensure_pad_token(tokenizer)

    # --no-lora -> peft_config=None -> NeuronSFTTrainer full fine-tunes.
    # The LoRA lanes freeze >99% of weights; a full FT updates all of them,
    # which changes both the memory profile and the FLOPs accounting below.
    lora_config = None if args.no_lora else LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.synthetic_data:
        # ---- DATALOADER ISOLATION LANE ------------------------------------
        # WHY: Trainium2 is only 1.20x faster than Trainium1 at seq 2048 but
        # 1.92x at seq 4096. One explanation is that at short context the step
        # is not device-bound at all -- host-side work (tokenise, pack, collate,
        # copy to device) takes a fixed wall-clock toll that the faster chip
        # cannot shrink, so it eats a larger FRACTION of trn2's shorter step.
        #
        # This lane tests that by deleting the host work. Rows are random token
        # IDs generated once, already the exact seq_len, already carrying
        # attention_mask and labels. No Hub download, no tokenisation, no
        # packing, no formatter. The compiled graph sees IDENTICAL shapes, so
        # the NEFF cache hits and nothing recompiles.
        #
        # If tokens/s jumps, host work was the bottleneck and the 1.20x is a
        # dataloader artifact. If it does not move, the chip really was
        # saturated and 1.20x is the honest device-level answer at this shape.
        # Either outcome is publishable; that is why the lane is worth running.
        #
        # The loss from this lane is MEANINGLESS -- random tokens teach nothing.
        # It is a throughput probe only, and the result JSON says so.
        from datasets import Dataset as _HFDataset

        rng = random.Random(args.data_seed if args.data_seed is not None else 20260805)
        vocab = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
        n_rows = int(args.synthetic_data)
        say(f"  SYNTHETIC DATA     : {n_rows} rows x {args.seq_len} random ids "
            f"(vocab {vocab}) -- throughput probe, loss is meaningless")
        rows = []
        for _ in range(n_rows):
            ids = [rng.randrange(vocab) for _ in range(args.seq_len)]
            rows.append({"input_ids": ids,
                         "attention_mask": [1] * args.seq_len,
                         "labels": list(ids)})
        dataset = _HFDataset.from_list(rows)
        packing = False          # rows are already exactly seq_len
    else:
        dataset = load_dataset(args.dataset, split="train")
        if args.dataset_fields:
            # target=source pairs; rename so format_dolly() keeps working
            # unchanged. Doing this here rather than branching the formatter
            # keeps every lane on ONE prompt template -- a different template
            # would confound any cross-dataset comparison with a formatting
            # change.
            for pair in args.dataset_fields.split(","):
                target, _, source = pair.partition("=")
                target, source = target.strip(), source.strip()
                if source and source in dataset.column_names:
                    dataset = dataset.rename_column(source, target)
            if "context" not in dataset.column_names:
                dataset = dataset.add_column("context", [""] * len(dataset))

    # ---- deterministic held-out split -------------------------------------
    # Same split_seed on both chips => byte-identical held-out rows, so the two
    # boxes are scored on the same examples. Split BEFORE packing: packing
    # concatenates examples, so splitting after it would leak train content
    # into eval sequences.
    #
    # Skipped entirely for synthetic data: holding out random tokens would
    # produce a held-out loss with no meaning, and reporting one would be worse
    # than reporting none.
    eval_dataset = None
    if args.synthetic_data and args.holdout_frac:
        say("  holdout SKIPPED    : synthetic rows are random tokens; "
            "a held-out loss on them would be meaningless")
    elif args.holdout_frac and args.holdout_frac > 0:
        split = dataset.train_test_split(test_size=args.holdout_frac,
                                         seed=args.holdout_split_seed)
        dataset, eval_dataset = split["train"], split["test"]
        say(f"  holdout            : {len(eval_dataset)} rows "
            f"({args.holdout_frac:.0%}, split seed {args.holdout_split_seed}); "
            f"train {len(dataset)}")

    sft_config = build_sft_config(NeuronSFTConfig, training_args,
                                  seq_len=args.seq_len, packing=packing)

    # ----------------------------------------------------------- callback
    class StepMetrics(TrainerCallback):
        """Wall time per optimizer step + the loss the trainer logged for it.

        on_step_end fires once per OPTIMIZER step (after grad accumulation), so
        one entry here == one entry of loss_trace == grad_accum micro-batches.
        Timing and loss arrive from two different callbacks in the same
        iteration and are joined on state.global_step.
        """

        def __init__(self):
            self.trace = []
            self.saves = []
            self.step_ms = {}
            self._t0 = None
            self.train_t0 = None

        def on_train_begin(self, a, state, control, **kw):
            self.train_t0 = time.perf_counter()

        def on_step_begin(self, a, state, control, **kw):
            self._t0 = time.perf_counter()

        def on_save(self, a, state, control, **kw):
            # on_save fires after the checkpoint hits disk; the last step
            # ended at _t_step_end, so the delta is the save's wall cost
            # (sharded NxD checkpoint + consolidation work, the ops number).
            now = time.perf_counter()
            if getattr(self, "_t_step_end", None) is not None:
                self.saves.append({"step": state.global_step,
                                   "save_wall_s": round(now - self._t_step_end, 2)})

        def on_step_end(self, a, state, control, **kw):
            self._t_step_end = time.perf_counter()
            if self._t0 is not None:
                self.step_ms[state.global_step] = (
                    self._t_step_end - self._t0) * 1e3

        def on_log(self, a, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            step = state.global_step
            ms = self.step_ms.pop(step, None)
            entry = {"step": step, "loss": round(float(logs["loss"]), 6),
                     "ms": round(ms, 2) if ms is not None else None,
                     "warmup": len(self.trace) < WARMUP_STEPS}
            self.trace.append(entry)
            if not is_main:
                return
            n = len(self.trace)
            if n <= WARMUP_STEPS or n % PRINT_EVERY == 0:
                ms_txt = f"{ms:10.1f} ms" if ms is not None else "         ? ms"
                flag = "  | warmup" if entry["warmup"] else ""
                print(f"  step {step:5d}  loss {entry['loss']:8.4f}  | "
                      f"{ms_txt}{flag}", flush=True)

    metrics_cb = StepMetrics()
    trainer = build_trainer(NeuronSFTTrainer, sft_config, model, lora_config,
                            tokenizer, dataset, args, metrics_cb)
    if args.no_lora:
        say("  full fine-tune     : LoRA disabled, every weight trains "
            "(dense 6N accounting)")

    # The checkpoint-kwargs shim MUST be live before the first forward: patch
    # the imported symbol in every optimum-neuron modeling module, plus any
    # per-module bindings transformers created (belt and suspenders).
    if not args.no_gradient_checkpointing:
        patched = patch_optimum_modeling_checkpoint(torch.is_tensor)
        n_wrapped = wrap_checkpoint_funcs(trainer.model, torch.is_tensor)
        # The ACTIVE architecture's modeling module must be among the patched
        # (or already-shimmed) set -- a generic "something was patched" check
        # let a llama run through with only granite patched.
        arch_mods = [n for n, m in sys.modules.items()
                     if n.startswith("optimum.neuron.models.training")
                     and getattr(m, "checkpoint", None) is not None
                     and not getattr(m.checkpoint, "_np_kwarg_shim", False)]
        if arch_mods:
            raise RuntimeError(
                "gradient checkpointing requested but these loaded modeling "
                f"modules still hold an unshimmed checkpoint: {arch_mods} -- "
                "refusing to run: the forward would die on torch-xla's "
                "kwarg rejection, or worse, NCC_EOOM001 without recompute")
        say(f"  grad ckpt shim     : module globals {patched}; "
            f"{n_wrapped} bound funcs wrapped")

    # Recount AFTER the trainer wrapped the model in PEFT: the pre-wrap count
    # sees every base parameter as trainable (the first smoke run reported
    # params_trainable == params_total -- wrong by 3 orders of magnitude, and
    # it silently poisons MFU). The post-wrap count is what actually trains.
    post = count_parameters(trainer.model, torch, tp_size)
    if 0 < post["trainable"] < post["total"]:
        params = post
        params_total = post["total"]
        params_trainable = post["trainable"]
    else:
        say("  WARNING: post-PEFT recount looks degenerate "
            f"(trainable={post['trainable']:,}/{post['total']:,}); "
            "MFU will be reported as null rather than from a bad count")
        params_trainable = None

    # ------------------------------------------------------------ banner
    versions = resolve_versions()
    say("")
    say("=" * 78)
    say(f"  LoRA SFT on Trainium            tag={args.tag}")
    say("=" * 78)
    say(f"  model              : {args.model}")
    say(f"  params total       : {params_total:,}  "
        f"(rank-local shard {params['local_total']:,}; "
        f"via {params['method']})")
    if params_trainable is not None:
        say(f"  params trainable   : {params_trainable:,}  "
            f"({100.0 * params_trainable / max(1, params_total):.3f}% of total)")
    else:
        say("  params trainable   : unknown (degenerate recount; MFU nulled)")
    say(f"  grad checkpointing : {not args.no_gradient_checkpointing}")
    say(f"  parallelism        : world={world} tp={tp_size} dp={dp_size} pp="
        f"{pp_size}")
    say(f"  lora               : r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout}")
    say(f"  lora targets       : {','.join(LORA_TARGET_MODULES)}")
    say(f"  seq_len            : {args.seq_len}  (packing={packing}, "
        f"attn={args.attn_impl})")
    say(f"  batch              : micro={args.micro_batch} "
        f"grad_accum={args.grad_accum} -> "
        f"{tokens_per_optimizer_step(args.seq_len, args.micro_batch, args.grad_accum, dp_size):,}"
        f" tok/optimizer step")
    say(f"  schedule           : epochs={args.epochs} max_steps={max_steps} "
        f"lr={args.lr}")
    say(f"  seed               : {args.seed}")
    say(f"  dataset            : {args.dataset}")
    say(f"  neuron cache       : {cache_dir}")
    say(f"  adapter out        : {adapter_out}")
    say(f"  versions           : torch={versions.get('torch')} "
        f"optimum-neuron={versions.get('optimum-neuron')} "
        f"neuronx-cc={versions.get('neuronx-cc')}")
    say("=" * 78)
    say("")

    # ------------------------------------------------------------- train
    # ---- held-out loss BEFORE training ------------------------------------
    # Measured before and after so the reported quantity is an IMPROVEMENT on
    # unseen data, not an absolute loss whose scale means little on its own.
    # Wrapped: if NeuronSFTTrainer.evaluate() is unsupported on this stack we
    # record that as a receipt and still run the lane, rather than losing the
    # throughput measurement to a quality feature.
    eval_before = eval_after = eval_error = None
    eval_method = None
    eval_wall_s = 0.0
    nonlocal_tb = []

    def holdout_loss(phase):
        """Mean forward loss on the held-out split.

        optimum-neuron 0.4.3's NeuronSFTTrainer has NO .evaluate() -- confirmed
        on trn1: AttributeError. So the primary path is a ZERO-LEARNING-RATE
        pass over the held-out rows using the SAME trainer machinery: same
        collator, same packing, same compiled graph shapes, so it needs no new
        API and no new compile. With lr=0 and a constant schedule the optimizer
        cannot move a weight, so the logged per-step losses are exactly forward
        losses on unseen data.

        Gradients are still computed and thrown away. That is wasted work, but
        it buys correctness through supported APIs instead of hand-rolling an
        XLA eval loop whose numerics we would then have to defend.
        """
        nonlocal eval_error, eval_method
        try:
            if hasattr(trainer, "evaluate"):
                eval_method = "trainer.evaluate"
                out = trainer.evaluate(eval_dataset=eval_dataset,
                                       metric_key_prefix=phase)
                return out.get(f"{phase}_loss")
            eval_method = "zero_lr_forward_pass"
            cb = StepMetrics()
            import dataclasses as _dc
            # dataclasses.replace keeps every field the real config already
            # resolved (parallelism, dtype, packing, seq len) and overrides
            # only what makes this a scoring pass instead of a training one.
            cfg = _dc.replace(sft_config,
                              learning_rate=0.0,
                              num_train_epochs=1,
                              max_steps=-1,
                              warmup_steps=0,
                              lr_scheduler_type="constant",
                              # ZeRO-1 clips gradients before stepping, and on
                              # a frozen scoring pass there ARE no gradients:
                              # neuronx_distributed grads.get_grad_norm([])
                              # raises IndexError on the empty list. Clipping a
                              # gradient we never intend to apply is pointless
                              # anyway, so switch it off rather than fabricate
                              # gradients to keep the optimizer happy.
                              max_grad_norm=0.0,
                              save_strategy="no",
                              output_dir=f"/tmp/holdout_{phase}")
            # After training the model is already PEFT-wrapped. Passing a
            # peft_config again (or letting trl re-wrap) is the most likely
            # source of the post-pass failure, so use the trainer's own model
            # when it exists and never re-apply an adapter.
            base = getattr(trainer, "model", model)
            ev = build_trainer(NeuronSFTTrainer, cfg, base, None,
                               tokenizer, eval_dataset, args, cb)
            ev.train()
            losses = [e["loss"] for e in cb.trace if e.get("loss") is not None]
            return round(sum(losses) / len(losses), 6) if losses else None
        except Exception as exc:                      # noqa: BLE001
            # Capture the TRACEBACK, not just the message. The first version
            # recorded "IndexError: list index out of range" with no frame
            # information, which made the failure undiagnosable and cost a
            # whole round trip. A receipt that cannot be acted on is not a
            # receipt.
            import traceback
            eval_error = f"{type(exc).__name__}: {exc}"[:200]
            eval_traceback = traceback.format_exc()[-1800:]
            say(f"  holdout eval FAILED ({eval_method}): {eval_error}")
            say("  --- traceback ---")
            say(eval_traceback)
            nonlocal_tb.append(eval_traceback)
            return None

    if eval_dataset is not None:
        _t = time.perf_counter()
        eval_before = holdout_loss("holdout_pre")
        say(f"  holdout loss (pre) : {eval_before}  [{eval_method}]")
        eval_wall_s += time.perf_counter() - _t

    t_train0 = time.perf_counter()
    trainer.train()
    train_wall_s = time.perf_counter() - t_train0

    # ---- held-out loss AFTER training -------------------------------------
    if eval_dataset is not None and eval_error is None:
        _t = time.perf_counter()
        eval_after = holdout_loss("holdout_post")
        say(f"  holdout loss (post): {eval_after}  [{eval_method}]")
        eval_wall_s += time.perf_counter() - _t

    # save_model() writes the adapter (TP-sharded when tp>1 -- merge_adapter.py
    # consolidates). It is collective, so every rank calls it; only rank 0
    # writes the metrics JSON.
    trainer.save_model(adapter_out)

    if not is_main:
        return 0

    # ----------------------------------------------------------- metrics
    trace = mark_warmup(metrics_cb.trace)
    steady = steady_state_step_ms(trace)
    med_ms = median(steady)
    first_ms = trace[0]["ms"] if trace and trace[0].get("ms") is not None else None
    compile_s = None
    if first_ms is not None and med_ms is not None:
        compile_s = max(0.0, (first_ms - med_ms) / 1e3)

    tok_per_step = tokens_per_optimizer_step(
        args.seq_len, args.micro_batch, args.grad_accum, dp_size)
    tok_s = tok_per_step / (med_ms * 1e-3) if med_ms else None

    perf = None
    if tok_s is not None and params_trainable is not None:
        perf = throughput_metrics(
            params_trainable, params_total - params_trainable, tok_s,
            peak_flops=peak_flops,
            peak_flops_alt=peak_flops_alt,
            gradient_checkpointing=not args.no_gradient_checkpointing)

    payload = {
        "tag": args.tag,
        "model": args.model,
        "dataset": ("synthetic:random-token-ids" if args.synthetic_data
                    else args.dataset),
        # A throughput probe on random tokens produces a loss number, and a
        # loss number in a results file WILL eventually be plotted by someone.
        # Marking it in the payload is the only thing that stops that.
        "synthetic_data": ({"rows": int(args.synthetic_data),
                            "seq_len": args.seq_len,
                            "loss_is_meaningless": True,
                            "purpose": "isolate device throughput from host "
                                       "dataloader cost; graph shapes unchanged"}
                           if args.synthetic_data else None),
        "params_total": params_total,
        "params_trainable": params_trainable,
        "params_frozen": (params_total - params_trainable
                          if params_trainable is not None else None),
        "params_local_shard_total": params["local_total"],
        "params_local_shard_trainable": params["local_trainable"],
        "params_method": params["method"],
        "params_note": param_method_note(params["method"]),
        "config": {
            "seq_len": args.seq_len,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "epochs": args.epochs,
            "max_steps": max_steps,
            "lr": args.lr,
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "seed": args.seed,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": list(LORA_TARGET_MODULES),
            "packing": packing,
            "attn_implementation": args.attn_impl,
            "dtype": "bfloat16",
            "world_size": world,
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
            "data_parallel_size": dp_size,
            "neuron_compile_cache_url": cache_dir,
            "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
            # Every number below is only meaningful against this denominator.
            "device_profile": profile["device_profile"],
            "instance_type": profile["instance_type"],
            "nproc_per_node": profile["nproc"],
            "logical_nc_config": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            "hbm_gib_per_core": profile["hbm_gib_per_core"],
            "device_note": profile["note"],
        },
        "versions": resolve_versions(),
        # PROVENANCE. load_dataset(name, split=...) pins nothing: the Hub is
        # mutable, so "dolly-15k" alone does not let anyone reconstruct the
        # corpus this trained on. A reviewer must be able to rebuild the exact
        # inputs, including which rows were held out.
        "provenance": provenance(args),
        # THE QUALITY GATE. Throughput without this cannot distinguish "trained
        # correctly, fast" from "produced garbage, fast". eval_wall_s is kept
        # OUT of train_wall_s so no throughput number is diluted by evaluation.
        "holdout": {
            "enabled": bool(args.holdout_frac),
            "frac": args.holdout_frac,
            "split_seed": args.holdout_split_seed,
            "loss_before": eval_before,
            "loss_after": eval_after,
            "loss_delta": (round(eval_after - eval_before, 6)
                           if (eval_before is not None and eval_after is not None)
                           else None),
            "eval_wall_s": round(eval_wall_s, 2),
            "method": eval_method,
            "traceback": (nonlocal_tb[-1] if nonlocal_tb else None),
            "error": eval_error,
            "note": ("Held-out in-domain token loss on rows never trained on. "
                     "Supports 'the fine-tune learned, comparably on both "
                     "chips'. Does NOT support any claim about instruction "
                     "quality or downstream benchmarks."),
        },
        "tokens_per_optimizer_step": tok_per_step,
        "steps_recorded": len(trace),
        "warmup_steps_excluded": WARMUP_STEPS,
        "first_step_ms": first_ms,
        "first_step_note": (
            "Step 1 traces the XLA graph and then either compiles it or loads "
            "it from the NEFF cache. A warm cache (lane 3 precompile) collapses "
            "this toward the median; a cold cache leaves tens of minutes in it."),
        "median_step_ms": round(med_ms, 3) if med_ms is not None else None,
        "compile_s": round(compile_s, 2) if compile_s is not None else None,
        "compile_s_note": "first_step_ms - median_step_ms, floored at 0.",
        "train_wall_s": round(train_wall_s, 2),
        "tokens_per_s": round(tok_s, 1) if tok_s is not None else None,
        # ---- STEADY-STATE vs END-TO-END -----------------------------------
        # tokens_per_s above is tokens_per_optimizer_step / MEDIAN STEP TIME:
        # a steady-state, compute-window rate with warmup excluded. That is the
        # standard convention and the one the GPU study this is compared
        # against also uses -- but it is NOT what an audience hears when you
        # say "the chip was N% utilised".
        #
        # On the published trn1 lane, 645 steps x 5.55 s = 59.7 min inside a
        # 122.2 min wall: the measured window is only ~49% of the run. Some of
        # the remainder is legitimately excluded (compile, model load,
        # tokenisation), but optimum-neuron's own per-step telemetry reports
        # ~46% overhead_time_percent, so most of it is BETWEEN steps, not
        # startup. End-to-end throughput is therefore ~2x lower than the
        # steady-state figure, and end-to-end MFU correspondingly so.
        #
        # Both are emitted so the report can state the convention explicitly
        # instead of picking whichever flatters. Neither is wrong; they answer
        # different questions, and the gap between them IS the optimisation
        # headroom.
        "tokens_per_s_end_to_end": (
            round(len(trace) * tok_per_step / train_wall_s, 1)
            if trace and tok_per_step and train_wall_s else None),
        "measured_window_fraction": (
            round(len(trace) * (med_ms / 1000.0) / train_wall_s, 4)
            if trace and med_ms and train_wall_s else None),
        "throughput_note": (
            "tokens_per_s is steady-state (median step, warmup excluded); "
            "tokens_per_s_end_to_end divides ALL tokens by the full training "
            "wall clock. measured_window_fraction is what share of wall time "
            "the median-step window actually covers. Compare like with like."),
        "flops_per_token": perf["flops_per_token"] if perf else None,
        "flops_formula": "6*params_trainable + 4*params_frozen",
        "tflops": round(perf["tflops"], 3) if perf else None,
        "mfu_pct": round(perf["mfu_pct"], 3) if perf else None,
        "peak_bf16_flops": peak_flops,
        "peak_bf16_flops_source": profile.get("peak_source"),
        # Second denominator: AWS's arch page and instance marketing disagree
        # for Trainium1 (190 vs 210 TFLOP/s). Reporting one MFU would hide
        # which was used, and mixing sources across boxes would bias the
        # comparison. tflops above is denominator-free.
        "mfu_pct_alt": (round(perf["mfu_pct_alt"], 3)
                        if perf and perf.get("mfu_pct_alt") is not None else None),
        "peak_bf16_flops_alt": peak_flops_alt,
        "peak_bf16_flops_alt_source": profile.get("peak_source_alt"),
        # Same measurement, end-to-end denominator. On the published trn1 lane
        # this turns 68.3% into roughly 33%.
        "mfu_pct_end_to_end": (
            round(100.0 * perf["flops_per_token"] * len(trace) * tok_per_step
                  / train_wall_s / peak_flops, 3)
            if perf and trace and tok_per_step and train_wall_s else None),
        "peak_host_mem_mib": read_peak_host_mem_mib(),
        "peak_device_mem_mib": None,
        "peak_device_mem_note": (
            "Device HBM high-water is not available from torch on XLA. Read it "
            "from the matching telemetry CSV (mem_used_mib column, sampled at "
            "1 Hz by neuron-monitor). Deliberately null rather than estimated."),
        "final_loss": trace[-1]["loss"] if trace else None,
        "loss_trace": trace,
        "checkpoint_saves": metrics_cb.saves,
        "adapter_out": adapter_out,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = atomic_write_json(args.out, payload)

    say("")
    say(f"  median step {med_ms:.1f} ms | {tok_s:,.0f} tok/s | "
        f"{perf['tflops']:.1f} TFLOP/s | MFU {perf['mfu_pct']:.1f}%"
        if perf and med_ms else "  (no steady-state steps recorded)")
    say("")
    say(f"SUMMARY tag={args.tag} model={args.model} tp={tp_size} dp={dp_size} "
        f"steps={len(trace)} "
        f"median_step_ms={_fmt(med_ms, 1)} first_step_ms={_fmt(first_ms, 1)} "
        f"compile_s={_fmt(compile_s, 1)} "
        f"tokens_per_s={_fmt(tok_s, 0)} "
        f"tflops={_fmt(perf['tflops'] if perf else None, 2)} "
        f"mfu_pct={_fmt(perf['mfu_pct'] if perf else None, 2)} "
        f"final_loss={_fmt(payload['final_loss'], 4)} "
        f"peak_host_mem_mib={_fmt(payload['peak_host_mem_mib'], 0)} "
        f"train_wall_s={_fmt(train_wall_s, 1)}")
    say(f"wrote {out_path}")
    say(f"adapter -> {adapter_out}  (TP-sharded; merge_adapter.py consolidates)")
    return 0


def _fmt(value, places):
    """Summary-line formatter that renders a missing number as `null`. Pure."""
    if value is None:
        return "null"
    return f"{value:.{places}f}"


def count_parameters(model, torch, tp_size):
    """Whole-model (total, trainable) parameter counts, plus how we got them.

    With TP=2 each rank holds roughly half of every sharded matrix, so a
    rank-local sum understates the model by ~2x and would halve the reported
    FLOPs and MFU. Three ways to recover the whole-model figure, in descending
    order of trustworthiness:

      1. xm.all_reduce -- the collective the Neuron/XLA backend actually
         supports. Correct even if the sharding is uneven.
      2. torch.distributed.all_reduce -- works if a gloo/CPU group exists.
      3. local * tp_size -- an estimate. Exact for sharded matrices (which are
         >99.9% of the weights) and over-counts replicated ones (norms, and
         LoRA A/B on replicated modules).

    Paths 1 and 2 also over-count replicated parameters, by counting them once
    per rank. For an 8B Llama that is ~266k of 8.03e9 -- under 0.01%, well
    inside the run-to-run spread of MFU. Whichever path was used is recorded in
    the JSON so no reader has to guess.

    Returns {"total", "trainable", "method", "local_total", "local_trainable"}.
    """
    local_total = sum(p.numel() for p in model.parameters())
    local_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base = {"local_total": local_total, "local_trainable": local_trainable}

    try:
        import torch_xla.core.xla_model as xm
        buf = torch.tensor([float(local_total), float(local_trainable)],
                           device=xm.xla_device())
        buf = xm.all_reduce(xm.REDUCE_SUM, buf)
        xm.mark_step()
        return {**base, "total": int(buf[0].item()),
                "trainable": int(buf[1].item()), "method": "xla_all_reduce"}
    except Exception:
        pass

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            buf = torch.tensor([float(local_total), float(local_trainable)])
            dist.all_reduce(buf)
            return {**base, "total": int(buf[0].item()),
                    "trainable": int(buf[1].item()), "method": "dist_all_reduce"}
    except Exception:
        pass

    return {**base, "total": local_total * tp_size,
            "trainable": local_trainable * tp_size,
            "method": f"local_x_tp{tp_size}_estimate"}


PARAM_METHOD_NOTES = {
    "xla_all_reduce":
        "Summed across the TP group with xm.all_reduce, so these are "
        "whole-model counts rather than this rank's shard. Parameters that are "
        "replicated rather than sharded (norms, LoRA A/B on replicated "
        "modules) are counted once per rank and so are marginally "
        "over-counted -- under 0.01% for an 8B model.",
    "dist_all_reduce":
        "Summed across ranks with torch.distributed.all_reduce (the XLA "
        "collective was unavailable). Same small over-count of replicated "
        "parameters as the XLA path.",
}


def param_method_note(method):
    """Human-readable provenance for the parameter counts. Pure."""
    if method in PARAM_METHOD_NOTES:
        return PARAM_METHOD_NOTES[method]
    return (f"ESTIMATE ({method}): no collective was available, so the "
            "rank-local shard counts were scaled by the tensor-parallel "
            "degree. Exact for sharded matrices, over-counts replicated ones. "
            "Treat flops_per_token and mfu_pct as approximate.")


def ensure_pad_token(tokenizer):
    """Give the tokenizer a pad token without stealing EOS when we can avoid it.

    Packing pads the tail of the last chunk. If pad == eos the collator's
    padding is indistinguishable from a real end-of-turn, which teaches the
    model to emit EOS early. Llama 3.x ships a dedicated pad token; anything
    else falls back to EOS, which is what upstream does.
    """
    if tokenizer.pad_token is not None:
        return tokenizer.pad_token
    for candidate in ("<|finetune_right_pad_id|>", "<|pad|>", "<pad>"):
        vocab = tokenizer.get_vocab()
        if candidate in vocab:
            tokenizer.pad_token = candidate
            return candidate
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.pad_token


def load_neuron_model(args, training_args, torch):
    """Load the base model with optimum-neuron's Trainium training modeling.

    NeuronModelForCausalLM is NOT transformers' AutoModelForCausalLM: it maps
    the architecture onto optimum-neuron's own Trainium implementations
    (Llama/Qwen3/Granite) and consumes the TrainingNeuronConfig that
    NeuronTrainingArguments built, which is what actually shards the weights
    across the TP group at load time. Loading via transformers instead would
    materialise the whole model on every rank and OOM the 32 GiB host.

    The `torch_dtype` -> `dtype` kwarg rename landed in transformers 4.56 and
    optimum-neuron's examples still use both spellings, so we try the modern
    name and fall back. Same story for attn_implementation on architectures
    whose Neuron modeling has no flash-attention kernel.
    """
    from optimum.neuron.models.training import NeuronModelForCausalLM

    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    attempts = [
        {"dtype": dtype, "attn_implementation": args.attn_impl},
        {"torch_dtype": dtype, "attn_implementation": args.attn_impl},
        {"dtype": dtype},
        {"torch_dtype": dtype},
    ]
    last = None
    for kwargs in attempts:
        try:
            return NeuronModelForCausalLM.from_pretrained(
                args.model, trn_config, **kwargs)
        except TypeError as exc:
            last = exc
            continue
    raise RuntimeError(
        "NeuronModelForCausalLM.from_pretrained rejected every known kwarg "
        f"spelling; last error: {last}")


def build_sft_config(NeuronSFTConfig, training_args, seq_len, packing):
    """NeuronTrainingArguments + SFT knobs -> NeuronSFTConfig.

    Upstream's pattern exactly: NeuronSFTConfig subclasses BOTH
    NeuronTrainingArguments and trl's SFTConfig, so the training arguments are
    splatted back in as a dict rather than passed as an object.

    trl renamed max_seq_length -> max_length; NeuronSFTConfig keeps a
    compatibility shim, but which spelling is a real dataclass field depends on
    the installed trl. We pick by inspecting the dataclass instead of guessing.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(NeuronSFTConfig)}
    length_kwarg = "max_length" if "max_length" in field_names else "max_seq_length"
    kwargs = training_args.to_dict()
    kwargs[length_kwarg] = seq_len
    kwargs["packing"] = packing
    return NeuronSFTConfig(**kwargs)


def build_trainer(NeuronSFTTrainer, sft_config, model, lora_config, tokenizer,
                  dataset, args, callback):
    """Construct the trainer, tolerating the tokenizer -> processing_class rename.

    transformers 4.46 deprecated the `tokenizer` kwarg on Trainer in favour of
    `processing_class`, and optimum-neuron followed. Older pinned DLAMI venvs
    still expect `tokenizer`. Inspecting the signature is cheaper than
    discovering it as a TypeError two hours into a lane.
    """
    import inspect

    params = inspect.signature(NeuronSFTTrainer.__init__).parameters
    kwargs = {
        "args": sft_config,
        "model": model,
        "peft_config": lora_config,
        "train_dataset": dataset,
        "formatting_func": lambda example: format_dolly(example, tokenizer),
        "callbacks": [callback],
    }
    # Synthetic rows arrive ALREADY tokenized (input_ids/attention_mask/labels),
    # which is the whole point -- no formatter, no tokenizer, no packing. Passing
    # a formatting_func anyway would make TRL re-render them as text and the
    # isolation would measure nothing.
    if getattr(args, "synthetic_data", 0):
        kwargs.pop("formatting_func")
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return NeuronSFTTrainer(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
