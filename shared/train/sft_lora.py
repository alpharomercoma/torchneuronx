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

WHY nproc_per_node=2 and tensor_parallel_size=2
-----------------------------------------------
trn1.2xlarge carries one Trainium1 device = 2 NeuronCores. One rank per core
gives world=2; spending the whole world on tensor parallelism gives TP=2, DP=1.
Llama 3.1 8B in BF16 is ~16 GiB of weights against 32 GiB of device HBM -- it
would nominally fit on one core, but activations, gradients and optimizer state
would not, so TP=2 is what makes the primary lane run at all. DP=1 is therefore
a consequence of the instance, not a tuning choice, and it is why the token
accounting below has no data-parallel multiplier on this box.

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

# trn1.2xlarge: one Trainium1 = 2 NeuronCores. World = TP, so DP = 1.
NPROC_PER_NODE = 2
TENSOR_PARALLEL_SIZE = 2
PIPELINE_PARALLEL_SIZE = 1

# LoRA target modules: every projection in the transformer block. Attention
# only (q,k,v,o) trains ~half as many parameters and reliably underperforms on
# instruction SFT; the MLP projections are where most of the adaptation lands.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# One Trainium1 device, dense BF16. AWS quotes trn1.32xlarge at 3.4 PFLOP/s
# BF16 across 16 Trainium devices -> 212.5 TFLOP/s per device; 210e12 is the
# rounded per-device figure used consistently across this study. MFU is
# measured against the WHOLE device because TP=2 means both NeuronCores
# cooperate on every token.
PEAK_BF16_FLOPS = 210e12

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
    ap.add_argument("--adapter-out", default=None,
                    help=f"default {DEFAULT_ADAPTER_ROOT}/<tag>")
    ap.add_argument("--tensor-parallel-size", type=int,
                    default=TENSOR_PARALLEL_SIZE)
    ap.add_argument("--attn-impl", default="flash_attention_2",
                    help="flash_attention_2 needs seq-len to be a multiple of "
                         "2048; pass eager to disable")
    ap.add_argument("--no-packing", action="store_true",
                    help="disable example packing (packing is ON by default)")
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="disable activation recomputation (8B models then "
                         "exceed the 16 GB/core HBM limit: NCC_EOOM001)")
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


def throughput_metrics(params_trainable, params_frozen, tokens_per_s,
                       peak_flops=PEAK_BF16_FLOPS,
                       gradient_checkpointing=False):
    """(trainable, frozen, tok/s) -> flops_per_token, TFLOP/s, MFU%. Pure.

    With activation recomputation every parameter pays one extra forward
    GEMM (+2 FLOPs/param/token): frozen 4 -> 6, trainable 6 -> 8. Same
    convention as the GPU study's x4/3 checkpointing adjustment, restated
    for the LoRA-split accounting.
    """
    fpt = lora_flops_per_token(params_trainable, params_frozen)
    if gradient_checkpointing:
        fpt += 2.0 * (params_trainable + params_frozen)
    achieved = fpt * tokens_per_s
    return {
        "flops_per_token": fpt,
        "tflops": achieved / 1e12,
        "mfu_pct": 100.0 * achieved / peak_flops,
        "gradient_checkpointing": bool(gradient_checkpointing),
    }


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


def apply_neuron_env(env=None, cache_dir=None):
    """Set the Neuron compiler/runtime env vars, without clobbering the user's.

    Must run before torch_xla is imported anywhere, hence before the re-exec:
    the child processes inherit it. Returns the resolved cache dir.
    """
    env = os.environ if env is None else env
    for key, value in NEURON_ENV.items():
        env.setdefault(key, value)
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

    cache_dir = apply_neuron_env()
    random.seed(args.seed)

    # Re-exec under torchrun if we were launched as plain `python`. Arguments
    # are already validated at this point, so a bad flag has failed once rather
    # than twice inside the elastic launcher.
    if not is_distributed_launch():
        cmd = torchrun_argv(__file__, argv)
        print(f"[sft_lora] no WORLD_SIZE in env -- re-exec under torchrun "
              f"(--nproc_per_node={NPROC_PER_NODE}, one rank per NeuronCore)",
              flush=True)
        print(f"[sft_lora] {' '.join(cmd)}", flush=True)
        os.execv(sys.executable, cmd)

    return run(args, cache_dir)


def run(args, cache_dir):
    """The distributed body. Every heavy import lives below this line."""
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
    tp_size = args.tensor_parallel_size
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
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        logging_steps=1,          # every optimizer step; the trace needs them all
        save_strategy="no",       # the adapter is saved once, explicitly, at the end
        save_total_limit=1,
        report_to=[],             # no wandb/tensorboard phoning home from the box
        seed=args.seed,
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

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(args.dataset, split="train")

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
            self.step_ms = {}
            self._t0 = None
            self.train_t0 = None

        def on_train_begin(self, a, state, control, **kw):
            self.train_t0 = time.perf_counter()

        def on_step_begin(self, a, state, control, **kw):
            self._t0 = time.perf_counter()

        def on_step_end(self, a, state, control, **kw):
            if self._t0 is not None:
                self.step_ms[state.global_step] = (
                    time.perf_counter() - self._t0) * 1e3

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
        f"{PIPELINE_PARALLEL_SIZE}")
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
    t_train0 = time.perf_counter()
    trainer.train()
    train_wall_s = time.perf_counter() - t_train0

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
            gradient_checkpointing=not args.no_gradient_checkpointing)

    payload = {
        "tag": args.tag,
        "model": args.model,
        "dataset": args.dataset,
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
            "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
            "data_parallel_size": dp_size,
            "neuron_compile_cache_url": cache_dir,
            "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
        },
        "versions": resolve_versions(),
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
        "flops_per_token": perf["flops_per_token"] if perf else None,
        "flops_formula": "6*params_trainable + 4*params_frozen",
        "tflops": round(perf["tflops"], 3) if perf else None,
        "mfu_pct": round(perf["mfu_pct"], 3) if perf else None,
        "peak_bf16_flops": PEAK_BF16_FLOPS,
        "peak_host_mem_mib": read_peak_host_mem_mib(),
        "peak_device_mem_mib": None,
        "peak_device_mem_note": (
            "Device HBM high-water is not available from torch on XLA. Read it "
            "from the matching telemetry CSV (mem_used_mib column, sampled at "
            "1 Hz by neuron-monitor). Deliberately null rather than estimated."),
        "final_loss": trace[-1]["loss"] if trace else None,
        "loss_trace": trace,
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
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return NeuronSFTTrainer(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
