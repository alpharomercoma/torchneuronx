#!/usr/bin/env python
"""GRPO / RLVR probe on Trainium: find and document exactly where the stack stops.

WHY THIS IS A PROBE AND NOT A LANE
----------------------------------
DPO and ORPO are OFFLINE: the preference pairs already exist, so training is a
forward/backward over fixed data and looks like SFT with a different loss. GRPO
and RLVR are ONLINE: at every step the policy must SAMPLE G completions per
prompt, score them with a reward, and train on the advantage. That sampling is
autoregressive decoding inside the training loop.

That is the hard part on this hardware, and it is architectural rather than a
missing import:

  * Neuron compiles static-shape graphs ahead of time. Autoregressive decode is
    a different graph from a training step, with a KV cache and a sequence
    dimension that grows by one token at a time.
  * The training graph and the decode graph would have to be co-resident on the
    same two NeuronCores and alternate every step.
  * TRL's answer to this cost on GPUs is vLLM (GRPOConfig carries use_vllm,
    vllm_mode, vllm_server_base_url). On Neuron, vLLM is a SERVING stack -- in
    this very study it runs on inf2 -- not something a trn1 training process
    hosts alongside itself.
  * neuronx-distributed-training's model_alignment_strategy enumerates
    dpo / sft / orpo / peft. There is no grpo, ppo, rloo or kto.

So the expected outcome is a wall. METHODOLOGY treats a documented wall as a
terminal result rather than a hole, and a practitioner choosing hardware needs
to know that the offline half of post-training works here and the online half
does not.

WHY IT STAGES THE DIAGNOSIS
---------------------------
"GRPO failed" is nearly useless. The probe therefore advances through four
stages and records the first one that breaks, so the report can say WHICH
capability is missing:

  A  construct  -- can TRL's GRPOTrainer even be re-based onto NeuronTrainer?
  B  generate   -- can the TRAINING-mode model produce tokens at all?
  C  reward     -- does the verifiable-reward function work on real GSM8K?
  D  train      -- does a single GRPO optimizer step complete?

Stage B is the interesting one. If a model built for training cannot generate,
every online RL algorithm is blocked for the same reason, and the finding
generalises well beyond GRPO.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase4_lib as P4  # noqa: E402

ANSWER_RE = re.compile(r"-?[\d,]*\.?\d+")


def gsm8k_gold(answer: str) -> str | None:
    """GSM8K puts the final answer after '#### '."""
    if "####" not in answer:
        return None
    return answer.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> str | None:
    """Last number in the completion -- the standard GSM8K convention."""
    nums = ANSWER_RE.findall(text.replace(",", ""))
    return nums[-1].strip() if nums else None


def verifiable_reward(completions, answer, **kw):
    """RLVR reward: 1.0 for an exactly-correct final answer, else 0.0.

    This is the whole point of 'verifiable' -- no reward model, no preference
    annotation, just a checkable fact. It is also why GSM8K is the canonical
    RLVR task.
    """
    out = []
    for comp, gold in zip(completions, answer):
        text = comp if isinstance(comp, str) else comp[-1]["content"]
        pred = extract_pred(text)
        out.append(1.0 if (pred is not None and gold is not None and pred == gold) else 0.0)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="grpo_rlvr_probe")
    ap.add_argument("--device-profile", default=os.environ.get("NP_DEVICE", "trn1"))
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nproc-per-node", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    profile = P4.profile_for(args.device_profile)
    nproc = args.nproc_per_node or profile["nproc"]

    if "LOCAL_RANK" not in os.environ:
        os.execv(sys.executable, [sys.executable, "-m", "torch.distributed.run",
                                  f"--nproc_per_node={nproc}", __file__, *argv])

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

    stages: list[dict] = []

    def record(name, ok, note="", err=""):
        stages.append({"stage": name, "ok": ok, "note": note,
                       "error": err[-1500:] if err else ""})
        say(f"  [{'PASS' if ok else 'WALL'}] {name}: {note or err[:200]}")
        return ok

    cfg_record = {
        "model": args.model, "task": "openai/gsm8k:main",
        "reward": "exact-match on the final integer (RLVR, no reward model)",
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps, "world_size": nproc,
    }
    t0 = time.time()

    # ---- stage A: construct -------------------------------------------------
    TrainerCls = ConfigCls = None
    try:
        from optimum.neuron.trainers import NeuronTrainer
        from optimum.neuron.trainers.training_args import NeuronTrainingArguments
        from trl import GRPOConfig as _Cfg, GRPOTrainer as _Trainer

        TrainerCls = type("_NeuronGRPOTrainer", (NeuronTrainer,),
                          _Trainer.__dict__.copy())

        @dataclass
        class NeuronGRPOConfig(NeuronTrainingArguments, _Cfg):
            def __post_init__(self):
                if hasattr(self, "padding_free"):
                    self.padding_free = False
                super().__post_init__()

        ConfigCls = NeuronGRPOConfig
        record("A_construct", True,
               "GRPOTrainer re-based onto NeuronTrainer: "
               + " -> ".join(c.__name__ for c in TrainerCls.__mro__[:3]))
    except Exception as exc:
        record("A_construct", False, err=f"{type(exc).__name__}: {exc}\n"
               + traceback.format_exc())

    # ---- stage C first (cheap, host-only): does the verifier work? ----------
    ds = None
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n_samples, len(ds))))
        ds = ds.map(lambda ex: {"prompt": ex["question"],
                                "answer": gsm8k_gold(ex["answer"])})
        golds = [r for r in ds["answer"] if r]
        fake = ["The answer is 42.", f"So the answer is {golds[0]}."]
        rw = verifiable_reward(fake, [golds[0], golds[0]])
        ok = rw == [0.0, 1.0] or (rw[1] == 1.0)
        record("C_reward", ok,
               f"{len(golds)}/{len(ds)} gold answers parsed; "
               f"verifier on (wrong,right) -> {rw}")
    except Exception as exc:
        record("C_reward", False, err=f"{type(exc).__name__}: {exc}")

    # ---- stage B: can a TRAINING-mode model generate? -----------------------
    # This is the crux. If it cannot, every online RL method is blocked here for
    # the same reason and the finding is far broader than GRPO.
    gen_note = ""
    try:
        from optimum.neuron.models.training import NeuronModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        trn_cfg = ConfigCls(output_dir="/tmp/np-grpo-probe",
                            tensor_parallel_size=profile["tp"],
                            bf16=True, report_to=[]).trn_config \
            if ConfigCls else None
        model = NeuronModelForCausalLM.from_pretrained(args.model, trn_cfg)
        has_generate = hasattr(model, "generate")
        gen_note = f"NeuronModelForCausalLM.generate present={has_generate}"
        if not has_generate:
            record("B_generate", False,
                   note=gen_note + " -- the training model class exposes no "
                        "generate(); online RL needs sampling inside the loop")
        else:
            enc = tok(["What is 2+2?"], return_tensors="pt",
                      padding="max_length", max_length=args.max_prompt_length)
            enc = {k: v.to(xm.xla_device()) for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=8, do_sample=False)
            record("B_generate", True,
                   f"{gen_note}; produced {out.shape[-1] - args.max_prompt_length} tokens")
    except Exception as exc:
        record("B_generate", False, note=gen_note,
               err=f"{type(exc).__name__}: {exc}\n" + traceback.format_exc())

    # ---- stage D: a real GRPO step ------------------------------------------
    if TrainerCls is not None and ds is not None:
        try:
            from optimum.neuron.models.training import NeuronModelForCausalLM
            tok = AutoTokenizer.from_pretrained(args.model)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            conf = ConfigCls(
                output_dir="/tmp/np-grpo-probe",
                per_device_train_batch_size=args.num_generations,
                num_generations=args.num_generations,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                max_steps=args.max_steps,
                use_vllm=False,          # no vLLM on a trn1 training process
                bf16=True, logging_steps=1, save_strategy="no",
                report_to=[], seed=args.seed,
                tensor_parallel_size=profile["tp"], disable_tqdm=True)
            model = NeuronModelForCausalLM.from_pretrained(args.model, conf.trn_config)
            trainer = TrainerCls(model=model, args=conf, train_dataset=ds,
                                 processing_class=tok,
                                 reward_funcs=[verifiable_reward])
            trainer.train()
            record("D_train", True, f"{args.max_steps} GRPO steps completed")
        except Exception as exc:
            record("D_train", False, err=f"{type(exc).__name__}: {exc}\n"
                   + traceback.format_exc())
    else:
        record("D_train", False, note="skipped: prerequisite stage failed")

    if master:
        passed = [s["stage"] for s in stages if s["ok"]]
        walls = [s for s in stages if not s["ok"]]
        common = dict(tag=args.tag, stage="grpo", box=args.device_profile,
                      config=cfg_record,
                      extra={"stages": stages,
                             "stages_passed": passed,
                             "first_wall": walls[0]["stage"] if walls else None,
                             "probe_wall_s": round(time.time() - t0, 1),
                             "interpretation": (
                                 "Offline preference optimisation (DPO/ORPO) needs "
                                 "only forward/backward over fixed data and is "
                                 "expressible on this stack. Online RL (GRPO, PPO, "
                                 "RLVR) additionally needs autoregressive sampling "
                                 "inside the training loop, which requires a decode "
                                 "graph co-resident with the training graph on the "
                                 "same two NeuronCores. TRL's standard escape hatch "
                                 "is vLLM, which on Neuron is an inference-only "
                                 "stack running on separate hardware.")})
        if walls:
            P4.failure_receipt(
                Path(args.out).with_suffix(".failure.json"),
                reason=f"online RL unsupported; first wall at {walls[0]['stage']}",
                detail=walls[0]["error"] or walls[0]["note"], **common)
            say(f"[grpo] WALL at {walls[0]['stage']} -- receipt written")
        else:
            import json
            p = Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {**common, "status": "passed",
                 "versions": P4.resolve_versions()}, indent=2) + "\n")
            say("[grpo] all stages passed -- GRPO IS viable here, which "
                "contradicts the documented alignment surface; verify before "
                "reporting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
