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
  identical to `vllm bench serve --save-result`). It approximates input
  length via characters-per-token rather than exact tokenization; input token
  counts are reported as configured, not measured. Where the host allows,
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
  platform — but claims here do not extend to Trainium2.
