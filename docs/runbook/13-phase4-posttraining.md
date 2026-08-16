# 13 — Phase-4: pretraining and post-training on trn1

Phase 1–3 covered supervised fine-tuning and serving. Phase 4 asks the two
questions the report could not previously answer:

1. Can you **pretrain from scratch** on a Trainium1 chip, and what does it cost?
2. How far up the **post-training ladder** (SFT → DPO/ORPO → GRPO/RLVR) does
   the Neuron stack actually go?

Assumes runbooks 00–05 have run at least once (stack exists, trn1 box exists,
NEFF cache warm, HF token in SSM).

## What the stack supports, measured rather than assumed

Audited on the box on 2026-08-14 against optimum-neuron 0.4.3 /
neuronx-distributed 0.19.28492 / TRL 0.24.0 / transformers 4.57.6.

This table was written from a **static audit** before any lane ran. Two of its
five rows turned out to be wrong once the lanes were executed. Both the
prediction and the outcome are kept, because the gap between them is the point:
an import-and-signature audit tells you what a stack *offers*, not what it
*does*.

| Stage | Predicted (static audit) | **MEASURED** | Route |
|---|---|---|---|
| Pretraining | works, small models only | **works via NeuronTrainer** — 4,573 tok/s, 13.9% per-core MFU, single steady graph; the HAND-WRITTEN loop remains unresolved. SmolLM2-360M is confined to ONE core (no DP in 0.4.3, 15 heads will not shard across TP=2), but a TP-divisible architecture uses the whole chip for **+7.7%** | `NeuronTrainer`, TP=1 or TP=2 |
| SFT | works | **works** — 2,952 tok/s, 68.3% MFU @ seq 2048 | `optimum.neuron.NeuronSFTTrainer` (Phase 1) |
| DPO | works via re-basing | **unresolved** — the adapter-disabled forward COMPILES once moved out of the training step; the lane then dies in a host transfer | TRL `DPOTrainer` reparented onto `NeuronTrainer` |
| ORPO | works via re-basing | **works** — 1,181 tok/s, 30.2% MFU @ max_length 1024 | TRL `ORPOTrainer`, same route |
| GRPO / RLVR | **blocked** | **blocked**, confirmed at stage B_generate | no generation in the training-model class |

Full analysis in REPORT-EXTENSIONS §32. Two practical takeaways for anyone
repeating this:

- **Prefer ORPO to DPO on a 16 GiB-per-core device.** ORPO is reference-free by
  construction, so it never needs the second forward pass that blocks DPO here,
  and an explicit `ref_model` does not fit at 8B.
- **The DPO wall was misattributed for two days, and the correction matters
  more than the lane.** Three attempts died inside neuronx-cc on the
  adapter-disabled reference forward, and that was written up as "the forward
  cannot compile". It compiles. Setting `precompute_ref_log_probs=True` runs it
  once before training instead of inside every step, and neuronx-cc returns
  `Compiler status PASS`. The blocker was the forward's **placement in the
  training-step graph**, not the forward. Three device-placement fixes were
  needed to get that far — TRL precomputes from `get_train_dataloader()`, which
  optimum-neuron calls at `transformers.py:1103`, *before* `setup_training()`
  places the model — and the lane still fails afterwards on `.cpu()` against an
  unflushed lazy tensor (`BufferMapAdd`). Unresolved, but for a different and
  much smaller reason than the report claimed.
- **Do not hand-roll the training loop.** The one lane in this phase that used a
  hand-written loop is the one lane still unresolved. Everything that went
  through `NeuronTrainer` reached a number -- including pretraining, once it was
  actually tried there (§32.11).
- **Check your head count before you plan a topology.** optimum-neuron 0.4.3
  has no data-parallel dimension, so tensor parallelism is the only way to use
  more than one NeuronCore, and TP requires `num_attention_heads % tp == 0`.
  An odd head count (SmolLM2-360M has 15) strands half the chip.
- **A plain `NeuronTrainer` over a plain causal-LM model needs
  `model_accepts_loss_kwargs = False`** in 0.4.3, or it dies on
  `Unexpected keyword arguments: reduction` before step 1.

**`neuronx-distributed-training` is NOT installed** on the trn1 DLAMI, though
1.7.0 is available from the Neuron pip repo. That package is where AWS's
documented `model_alignment_strategy: {dpo, orpo, sft, peft}` lives. It is a
NeMo-style YAML stack that wants HF checkpoints converted to its own sharded
format. Phase 4 deliberately does **not** use it: installing it would mutate the
venv that produced every published Phase-1/2/3 number, and the checkpoint
conversion is a second pipeline to validate. The re-basing route below reaches
DPO and ORPO without either cost.

## Running it

```bash
# on the box, disconnect-proof
setsid nohup bash extras/run_phase4_trn1.sh smoke >> /opt/np/phase4_smoke.log 2>&1 &
# then, once pretrain_probe.json reports a real tokens_per_s:
setsid nohup bash extras/run_phase4_trn1.sh full  >> /opt/np/phase4_full.log  2>&1 &
```

`smoke` runs every lane at tiny scale first. Three of the four lanes are new
code against an unproven combination, and committing ~25 hours of paid Trainium
to code that has never executed is a bad trade. The smoke stage costs about an
hour and its `pretrain_probe` throughput is what sizes the full run.

Results land in `trn1/results/phase4/` as the usual triplet
(`.json` + `.log` + `.telemetry.csv`) and are pushed to
`s3://neuron-pipelines-artifacts-600627330911/results/trn1/` after every lane.

Budget knobs for `full`:

| env var | default | effect |
|---|---|---|
| `NP_PRETRAIN_TOKENS` | `1e9` | token budget for the pretrain lane |
| `NP_PRETRAIN_HOURS` | `8` | wall-clock cap; whichever binds first wins |
| `NP_PREF_SAMPLES` | `4000` | UltraFeedback pairs for DPO and ORPO |

The result records `stopped_because` so a truncated run is never mistaken for a
completed one.

## Data prep

`prep_data.sh` (run once) fetches UltraFeedback-binarized, GSM8K, and the
Llama-3.1-8B **base** weights, and pre-tokenises FineWeb-Edu into a flat uint16
memmap at `/opt/np/data/pretrain/fineweb_edu_1B.uint16.bin` (~2.2 GB, about six
minutes at 1.68M tok/s).

Two deliberate choices:

- **Pre-tokenised, not streamed.** The Phase-3 dataloader-isolation lane showed
  host-side input pipelines can dominate step time on an 8-vCPU box. Streaming
  would fold that into the MFU number and make the pretraining lane a
  measurement of the dataloader. A memmap makes input cost ~0.
- **On `/opt` (EBS), not `/scratch`.** Instance store is wiped on stop/start,
  and re-tokenising is 30 minutes of paid time to buy back.

## Gotchas that cost real time

**`libneuronpjrt` path.** Invoking a venv's python directly without putting its
`bin` first on `PATH` makes `torch_xla`, `peft`, `accelerate` and
`neuronx_distributed` all fail with `FileNotFoundError`. This looks exactly like
a broken box and is not. Always `export PATH="$NP_VENV/bin:$PATH"` — the same
trap that cost seven lanes in Phase 2.

**Graph size, not HBM, binds the pretraining lane.** A 362M model has memory to
spare, so micro_batch 8 looks free. It is not: the largest tensor is the logits,
`micro_batch × seq_len × vocab`. At mb=8 that is 805M elements and the compile
dies with

```
NCC_EVRF007 Instructions generated by compiler 37,536,776
            exceeds the typical limit of 5,000,000
```

mb=1 gives 100M elements and compiles. **Do not compensate by raising
grad_accum** — accumulation is unrolled into the compiled graph on this stack
(the finding behind the 146.7% MFU control), so a bigger grad_accum makes the
graph bigger too. Let tokens-per-step fall instead. Receipt:
`pretrain_probe_mb8.failure.json`.

**Bare `super()` does not survive `__dict__` copying.** optimum-neuron builds
its SFT trainer as

```python
_SFTTrainer = type("_SFTTrainer", (NeuronTrainer,), SFTTrainer.__dict__.copy())
```

and then writes a bespoke `__init__` that calls `NeuronTrainer.__init__(self, …)`
explicitly. That is not a stylistic choice. Python compiles a bare `super()` to
`super(__class__, self)` where `__class__` is a closure cell bound to the class
the method was defined in; copying `__dict__` keeps the old cell, so `super()`
searches an MRO that no longer contains its target and raises
`TypeError: super(type, obj): obj must be an instance or subtype of type`.
`posttrain_align._clone_rebound` fixes this generally by cloning each method
with a fresh cell — without mutating TRL's originals, which would corrupt
`trl.DPOTrainer` for the rest of the process.

**`NeuronTrainer` is not a `transformers.Trainer` subclass.** Its base is
`object`; it reimplements most of the surface but not all. TRL reads attributes
it does not set. The gap is small and was enumerated statically (AST-diff TRL's
reads against NeuronTrainer's provides) rather than one `AttributeError` at a
time, because each rediscovery costs a four-minute 8B checkpoint load:

| needed by | names |
|---|---|
| DPO | `_prepare_inputs`, `create_model_card`, `is_deepspeed_enabled`, `is_fsdp_enabled` |
| ORPO | `_prepare_inputs`, `create_model_card` |

`posttrain_align.HFTrainerCompat` supplies them. `is_deepspeed_enabled` and
`is_fsdp_enabled` are `False` as facts, not stubs: Neuron has no DeepSpeed
integration and sharding here is NxD's job.

**Preference collators pad per-batch.** TRL pads to the longest item in each
batch, so shapes vary and Neuron recompiles on every new shape.
`FixedShapeCollator` wraps whatever collator TRL chooses and pads everything to
one constant length. Pad values are not interchangeable: `input_ids` take the
pad id, `attention_mask` takes 0, `labels` take -100 — get the last one wrong
and the model trains to predict padding while the loss still looks plausible.

**DPO would need two 8B models.** It does not fit in 16 GiB per core. With a
LoRA policy and `ref_model=None`, TRL computes reference log-probabilities by
disabling the adapter on the same weights, so the base model is its own
reference at zero extra memory. This is why the DPO lane is LoRA and not
full-parameter.

## Why GRPO and RLVR are blocked

The probe stages its diagnosis so the report can say *which* capability is
missing rather than just "it failed":

```
A_construct  PASS   GRPOTrainer re-bases onto NeuronTrainer fine
C_reward     PASS   64/64 GSM8K gold answers parsed; verifier correct
B_generate   WALL   NeuronModelForCausalLM.generate present=False
```

The training-model class exposes no `generate()` at all. Generation on Neuron
lives in a separate inference class backed by an ahead-of-time-compiled decode
graph. Online RL needs to sample completions *inside* the training loop, which
would require the decode graph and the training graph co-resident on the same
two NeuronCores. TRL's standard escape hatch is vLLM, which on Neuron is an
inference-only stack — in this study it runs on inf2, a different box.

So the boundary is not "GRPO is unimplemented". It is that **offline
post-training works and online post-training does not**, and the same reason
blocks PPO, RLOO and RLVR. Receipt: `grpo_rlvr_probe.failure.json`.

## Verification

```bash
cd tests && uvx --with pytest --with numpy pytest -q test_phase4.py
```

Runs off-box and torch-free where possible — the MFU denominator behind a
published chart should not need a $1.34/hr instance to check. Includes a
regression test that reproduces the `super()` closure-cell bug and proves the
fix, and one asserting Phase 4 reuses `sft_lora`'s accounting *by identity*
rather than by a copy that would drift.
