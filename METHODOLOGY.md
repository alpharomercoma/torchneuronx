# Methodology

Read this before quoting anything from REPORT.md.

## The question

Can AWS's own AI silicon — Trainium1 for training, Inferentia2 for inference —
run *modern* open models end to end, at production quality, on the smallest
instances money can rent? Not "does a tutorial pass": a LoRA fine-tune of
Llama 3.1 8B Instruct trained on a trn1.2xlarge, merged, shipped through S3,
compiled and served by vLLM on an inf2.xlarge, with the same serving metrics
(TTFT/TPOT/ITL/E2EL percentiles, throughput, sustained retention) used in this
repo's GPU sibling, [MI300X-vs-H200](https://github.com/alpharomercoma/MI300X-vs-H200).

This is a *capability and maturity* study of one vendor's stack, not a
head-to-head. Where the GPU study compared two chips against each other, this
one compares the Neuron stack against the claim implicit in its marketing:
that specialized silicon plus an open-source serving stack is a production
path, not a science project.

## The rules

Each rule states the decision and the reasoning, written down before the
first measured run. Changes after that point go in REPORT.md §Corrections.

### 1. Smallest instances, deliberately

trn1.2xlarge and inf2.xlarge each carry one accelerator (2 NeuronCores v2,
32 GB HBM) — the smallest rentable unit of each family. Production Neuron
deployments scale by fleet, not by box, so per-chip behavior is the honest
unit of measurement, and the numbers here are reproducible for roughly a
dollar an hour, which is the point of the exercise.

### 2. Stock DLAMI stack, pinned by provenance

Everything runs in the AWS-published Neuron DLAMI venvs — no source builds,
no version mixing. `specs.txt` (lane 0) records every Neuron package version,
and no lane runs before lane 0 has. The question is what the stack does out
of the box, because that is what a newcomer gets.

### 3. Compile time is a result, not overhead to hide

Ahead-of-time compilation is the Neuron experience's biggest difference from
the CUDA/ROCm world, so it is measured and reported first-class: precompile
wall time (`compile/*.json`), cold-boot vs warm-boot server wall time
(`serve/*/boot.json`), and compile-cache size. The warm path exists (persistent
`NEURON_COMPILE_CACHE_URL` on EBS, synced to S3) and its effectiveness is a
headline number, not an apology.

### 4. Precompile numerics are invalid

`neuron_parallel_compile` executes graphs with garbage numerics by design.
No loss value produced during a precompile lane is recorded anywhere; the
precompile metrics JSON deliberately has no loss field, and its transient
output goes to /tmp. This rule exists because a plausible-looking loss curve
from a precompile run is the single easiest way to publish a false training
result on this platform.

### 5. MFU is LoRA-corrected

Model FLOPs utilization uses FLOPs/token = 6·N_trainable + 4·N_frozen against
Trainium1's published 210 TFLOP/s dense BF16 peak. Reasoning: frozen base
weights do forward (2·N) and activation-gradient (2·N) work but skip the
weight-gradient pass (2·N); trainable LoRA parameters do all three (6·N). The
usual 6·N-for-everything convention would overstate LoRA MFU by ~50%. The
torchtitan-style convention from the GPU study is kept for everything else.

### 6. KV-cache arithmetic drives the serving grid, and reductions are declared

Llama 3.1 8B: 32 layers × 8 KV heads × 128 head-dim × 2 (K+V) × 2 bytes =
128 KiB per token of context. ~15 GiB of weights in 32 GiB HBM leaves
~12–14 GiB for KV, and the NxD Inference backend preallocates approximately
`max_num_seqs × max_model_len`. One server config therefore cannot cover both
short-context/high-concurrency and long-context lanes; there are two configs
per model (A: 2048 ctx × 32 seqs, B: 9216 ctx × 8 seqs — exactly the max the long shapes need; 10240 crossed an internal neuronx-cc bound-check crash, NCC_INLA001), each a separate
compile, both warmed. The concurrency grid tops out at 32 — not 256 as in the
GPU study — because beyond the resident-sequence bound the client would be
measuring its own queue, not the accelerator. Every sweep directory carries a
`grid.json` declaring its exact grid, `reduced: true`, and the reasoning. A
reduced grid presented as a full one is the failure mode this machinery exists
to prevent.

### 7. No result without telemetry, and nothing fabricated

Inherited unchanged from the GPU study: a result row without its telemetry
CSV is dropped and counted, and saturation is judged over the busy window
(first to last sample ≥90% utilization). One Neuron-specific amendment:
`neuron-monitor` exposes NeuronCore utilization and device memory but not
board power on these instances, so the power column is empty and
`tokens_per_joule` is reported as null with a note — an absent sensor is
reported as absent, never interpolated.

### 8. Failures are results

If the vLLM Neuron backend cannot boot a model (the Qwen3 8B serve lane is
explicitly an *attempt*), the outcome is a structured `load_failure.json`
carrying the verbatim error line, committed like any other result. Declared
exclusions beat silent gaps; negative results are still results.

### 9. Weights are verified, including the round trip

SHA-256 of every model artifact is recorded at download, after the LoRA merge,
and after the S3 round trip onto the inference box (`model_hashes.txt`,
`merge_llama31.json`). "The model we served is the model we trained" is a
checkable claim, not an assumption.

### 10. The host CPU lane is context, not headline

inf2.xlarge has 4 vCPUs. The CPU lane measures the host's tokenization and
memory bandwidth floor so that any client-side ceiling at high concurrency is
attributable *before* it can masquerade as an accelerator limit. Host numbers
never headline; they bound the serving numbers' interpretation.

## What is measured

| Lane | Box | What | Why it is fair |
|---|---|---|---|
| 0 provenance | both | specs, package versions, model hashes | no lane before it |
| 1 host CPU | both | STREAM triad, matmul scaling, tokenizer throughput | bounds client-side ceilings (rule 10) |
| 2 smoke | both | TinyLlama-1.1B through the full plumbing | ~$1 gate before 8B hours |
| 3–5 training | trn1 | precompile; LoRA SFT Llama 3.1 8B; Qwen3 8B | compile first-class (rule 3); MFU rule 5 |
| 6 merge | trn1 | adapter merge + SHA-256 + S3 | rule 9 |
| 3–4 serving | inf2 | two-config sweeps, TTFT/TPOT/ITL/E2EL × p50/p90/p99 | grid math declared (rule 6) |
| 5 sustained | inf2 | 30 min at fixed load, retention vs first/peak | thermal/stability claims measured |
| 6 quality | inf2 | greedy determinism + mean logprob, 16 fixed prompts | temp-0 seed-0, same harness as GPU study |
| 7 fine-tune serve | inf2 | the trn1-trained model, served | the end-to-end claim itself |
| 8 Qwen3 attempt | inf2 | boot attempt, outcome recorded either way | rule 8 |

## Known limits

- One box per role, one seed, one run per lane (plus the smoke lanes). The
  GPU study's repeatability lane is out of scope at this budget; step-time
  and TTFT distributions within runs are reported instead.
- The primary bench client is this repo's `fallback_client.py` (schema-
  identical to `vllm bench serve --save-result`). Input length is exact:
  the prompt over-provisions words and the request pins the token count via
  vLLM's `truncate_prompt_tokens` (adopted after the BOS token pushed
  1024-word prompts to 1025 tokens and 400'd an entire sweep). Output length
  is pinned by `max_tokens` + `ignore_eos`. Where the host allows,
  `vllm bench serve` cross-checks a subset of points.
- Dolly-15k × 3 epochs of LoRA is a demonstration-scale SFT chosen to make
  behavior change visible in the quality lane — it is not a production
  alignment recipe, and no claim about fine-tune *quality* beats that bar.
- No per-dollar or per-watt headlines (power is unavailable; prices are
  listed once for context: trn1.2xlarge $1.34/hr, inf2.xlarge $0.76/hr,
  us-west-2 on-demand, July 2026).
- Trainium1 and Inferentia2 are the *previous* Neuron generation (Trn2/Inf2
  successors exist). That is deliberate: these are the instances whose quotas,
  prices, and availability make them the realistic first contact with the
  platform. Phase 3 adds a **training** lane on one Trainium2
  (trn2.3xlarge, sa-east-1) so the generational comparison is measured rather
  than assumed — but the *serving* results here still do not extend to
  Trainium2, and the inf2 vLLM 0.16 AMI pin exists precisely because the
  Trn2-targeted DLAMI cannot boot on NeuronCore-v2 at all.
- Phase 3 rule, inherited from rule 6: **the trn2 box runs the identical lane
  list and identical hyperparameters as trn1, through the same driver branch.**
  A tuned Trainium2 measured against an untuned Trainium1 is not a comparison.
  Efficiency levers (longer sequences, recompute off) are separate declared
  lanes with their own triplets, never edits to the primary lane. And because
  the parallelism degree is decided by a probe rather than assumed, any result
  that ran on fewer than all four logical cores is labelled as a partial-chip
  configuration wherever it appears.
- **Phase 4 limits (pretraining and post-training).** The ORPO lanes measure
  throughput and nothing else: both 30-step runs had their loss *rise*, so no
  claim is made that preference optimisation improved any model. The ORPO
  numbers (25.9% MFU at max_length 512, 30.2% at 1024) must not be read against
  the SFT lane's 68.3% at seq 2048 — the sequence lengths differ, and a
  preference step forwards two sequences where an SFT step forwards one, so the
  shapes differ too. The gap between them has **not** been separated into
  "shorter sequences" and "heavier objective".
- **Phase 4 non-claim.** The pretraining lane recompiles its XLA graph every
  step and the cause is not established (two attributions made, both falsified
  by measurement). That lane is a **hand-written** training loop. Every other
  training lane in this study runs through optimum-neuron's `NeuronTrainer` and
  shows no such behaviour, and pretraining *through* that framework path was
  never tested. Nothing here supports the claim "pretraining from scratch does
  not work on Trainium1"; what is supported is "a hand-rolled lazy-tensor
  training loop recompiles per step on this stack."
- **Phase 4 update (§32.11): the framework path WAS then tested, and it
  trains.** 362M from random init through `NeuronTrainer`, single steady graph
  held for 58 consecutive steps, loss 11.02 -> 7.76. Two caveats travel with
  that number and must not be dropped. (1) It ran on ONE of two NeuronCores:
  optimum-neuron 0.4.3 exposes no data-parallel dimension, and SmolLM2-360M's
  15 attention heads cannot shard across TP=2, so the architecture is confined
  to a half-chip configuration. (2) It is NOT a clean one-variable comparison
  against the hand-written lane, which ran DP=2 on both cores -- loop ownership
  and core count both changed, so this does not prove the hand loop caused its
  own recompile. Closing that would need the hand loop re-run at nproc=1.
- **Phase 4 preference-lane FLOPs.** MFU for preference lanes uses the same
  `6*trainable + 4*frozen` convention as every other lane, so it stays
  comparable. For ORPO that is exact — it is reference-free. For DPO it would
  under-count by roughly `2*frozen` per token (the adapter-disabled reference
  forward), and the code emits `mfu_pct_with_reference_pass` beside the
  comparable figure rather than silently choosing one. As everywhere else in
  this study, the convention omits sequence-dependent attention work.
