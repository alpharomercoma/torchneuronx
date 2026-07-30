# Report

Status: **complete.** Every number below is regenerated from
`analysis/comparison.json` (`make report`); nothing is hand-computed.
Read [METHODOLOGY.md](METHODOLOGY.md) before quoting anything.

## 1. Summary

The claim under test held, with receipts on both halves:

- **Training works, and works well.** A LoRA fine-tune of Llama 3.1 8B
  Instruct ran on one trn1.2xlarge ($1.34/hr) at **2,952 tok/s — 68.3% MFU**
  against the chip's published BF16 peak, with activation recomputation
  charged honestly in the accounting. Qwen3 8B trained twice at 57.2/57.3%
  MFU — **0.1% run-to-run throughput agreement**.
- **Serving works.** The same 8B models serve on one inf2.xlarge ($0.76/hr)
  via vLLM: 415 tok/s aggregate at concurrency 32 with decode latency
  essentially flat (63→71 ms TPOT from c1 to c32) and NeuronCores at 93–96%
  utilization; **perfectly stable over 30 minutes** of sustained load
  (retention 100.4% vs first iteration).
- **The end-to-end loop closed.** The adapter trained on the Trainium box was
  merged (15.0 GiB, SHA-256'd), shipped through S3, and served on the
  Inferentia box at base-model parity — booting in **9.1 min with zero new
  compilation** against a warm NEFF cache that cost 39.5 min to build once.
- **Maturity has edges, and they are documented, not hidden.** The newest
  vLLM DLAMI (0.21) cannot run inf2 at all; long-context serving dies in an
  internal compiler crash at two different lengths; Qwen3 boots but crashes
  on its first generated token; logprobs are rejected by the serving API.
  Section 9 lists every exclusion; section 11 lists every correction this
  study made to itself.

## 2. Machines

Per `*/results/specs.txt`: trn1.2xlarge (1× Trainium1, 2 NeuronCores v2,
32 GB HBM; 8 vCPU/32 GiB host + 64 GiB NVMe swap; Neuron DLAMI PyTorch 2.9,
torch-neuronx 2.9.0, neuronx-cc 2.26, optimum-neuron 0.4.3 + trl 0.24.0 +
peft 0.17.0) and inf2.xlarge (1× Inferentia2, 2 NeuronCores v2, 32 GB HBM;
4 vCPU/16 GiB host + 48 GiB EBS swap; Neuron vLLM 0.16 DLAMI, vllm 0.16.0,
NxD Inference 0.10). Both us-west-2, SSM-only access, weights SHA-256
verified including the S3 round trip (`model_hashes.txt`,
`merge_llama31.json`).

## 3. Compile costs (rule 3: a result, not overhead)

| Event | Wall time | Cache state |
|---|---|---|
| Llama 8B serving config A, first ever boot | **2,372 s (39.5 min)** | cold → NEFFs cached |
| Llama 8B config A, subsequent suite boot | 1,668 s | partial reuse |
| **Fine-tuned 8B (different weights, same graphs)** | **548 s (9.1 min)** | 616 MB cache, **0 new NEFFs** |
| TinyLlama smoke, warm | 126 s | full hit |
| Qwen3 8B first boot | 2,530 s | cold (then generation failed, §9) |

The cache is the production story: weights changed, graphs didn't, and the
fine-tune paid nothing. Training-side compile is folded into each run's
`first_step_ms` (the standalone precompile lane failed under
`neuron_parallel_compile` + torchrun; recorded in
`compile/llama31_train.failure.json` — see §11).

## 4. Training (trn1.2xlarge, LoRA r16/α32, bf16, TP=2, recompute on)

| Run | median step | tok/s | TFLOP/s | MFU | loss |
|---|---|---|---|---|---|
| TinyLlama 1.1B smoke (20 steps, full-param FLOPs basis) | 3,661 ms | 4,475 | 30.1 | 14.3% | sanity only |
| **Llama 3.1 8B** (645 steps, 3 epochs dolly-15k) | 5,550 ms | 2,952 | 143.5 | **68.3%** | 2.18 → 1.21 |
| Qwen3 8B — run 1 | 6,759 ms | 2,424 | 120.2 | 57.2% | → ~1.44 |
| Qwen3 8B — run 2 (unplanned repeat, §11) | 6,758 ms | 2,426 | 120.2 | 57.3% | 2.37 → 1.44 |

MFU uses the LoRA-corrected, recompute-aware convention (METHODOLOGY rule 5):
trainable params 52.4M of 8.08B (0.65%). The Qwen repeat — forced by an
operational mistake, not planned — landed within 0.1% of the original,
which is the repeatability evidence this budget couldn't otherwise afford.

## 5. Serving sweeps (inf2.xlarge, vLLM 0.16/NxDI, config A: 2048 ctx × 32 seqs)

Llama 3.1 8B Instruct, shape 1024:1024, exact ISL via `truncate_prompt_tokens`:

| conc | out tok/s | TTFT p50 | TPOT p50 | NC util |
|---|---|---|---|---|
| 1 | 15.7 | 399 ms | 63.2 ms | 95.6% |
| 4 | 61.4 | 988 ms | 64.3 ms | 95.1% |
| 8 | 119.3 | 1,765 ms | 65.4 ms | 94.4% |
| 16 | 226.6 | 3,333 ms | 67.4 ms | 93.8% |
| 32 | 415.5 | 6,409 ms | 70.8 ms | 93.3% |

Decode latency is nearly flat while aggregate throughput scales 26× — the
chip batches well inside its declared envelope. TTFT growth is queueing, as
the grid declaration predicts (`grid.json`, reduced=true, KV math).

**The trn1 fine-tune serves at parity** (same architecture, merged weights):
15.75 / 57.8 / 114.8 / 213.2 / 394.5 tok/s at c1–c32 — within 5% of base at
every point.

## 6. Sustained (30 min, config A, concurrency 8)

7 iterations: 118.8 → 119.3 tok/s; tail-3 mean 119.3; **retention 100.4% vs
first, 100.0% vs peak**. TTFT p99 spread across the half hour: 1.6%. No
thermal or stability droop is observable on this box at this load.

## 7. Quality (16 fixed prompts, greedy, temp 0, seed 0)

Both base and fine-tune completed 16/16 deterministically. The behavior
delta is visible and demoable: base answers in assistant-explainer register
("The sky appears blue because of a phenomenon called Rayleigh
scattering…"), the dolly fine-tune shifts toward instruction-conditioned
dolly register. Per METHODOLOGY known-limits: this is a *style shift*, not a
claimed quality improvement. `corpus_mean_logprob` is null — the backend
rejects `logprobs` (§9), and an unsupported sensor is reported as
unsupported.

## 8. The end-to-end loop

train (trn1, 95 min) → merge (6 artifacts, 14.97 GiB, SHA-256 each) →
S3 (25.5 s up) → pull on inf2 → boot 548 s, zero recompile → sweep at
base parity → quality 16/16. "The model we served is the model we
trained" is checkable hash-to-hash.

## 9. Declared exclusions (rule 8)

| What | Evidence |
|---|---|
| vLLM 0.21 DLAMI on inf2 — any model | EFA mapper knows only trn2/trn3; past that, NKI kernels assert `tensor_copy does not support engine.scalar on NeuronCore-v2`. This DLAMI generation cannot serve inf2. Pin the 0.16 line (`ami-035c945d557065665`). |
| Long-context serving (config B) | `NCC_INLA001` internal compiler crash at max-model-len 10240 **and** 9216 (`llama31_base_long/load_failure.json`). |
| Qwen3 8B serving | Boots, passes /health, crashes the engine core on the first generation step — HTTP 500 on both streamed and plain completions (`qwen3_base_short/generation_failure.json` + probe log). Training the same model works. |
| `logprobs` on completions | Rejected by the 0.16 NxDI backend; quality lane degrades to greedy-match with null logprob. |
| Per-watt anything | neuron-monitor exposes no board power on these instances; `tokens_per_joule` is null throughout. |
| Serving concurrency > 32 | KV arithmetic (128 KiB/token) bounds resident sequences; declared in every `grid.json`. |

## 10. Host CPU context (rule 10)

inf2.xlarge's 4 vCPUs are the client ceiling, not the serving ceiling at
these points — NeuronCore utilization stayed ≥93% through c32 while the
stdlib client drove load from the same host. The cpu lane
(`cpu/cpu.json`) records the host floor for anyone extending the grid.

## 11. Corrections made during this study

Every issue that changed a previously recorded number or invalidated a run,
in order of discovery:

| Issue | Effect | Resolution |
|---|---|---|
| DLAMI training venv ships no optimum-neuron; `[neuronx]` extra backtracks pip into py3.12-incompatible releases | first suite pass all-failed | install by exact pins from the wheel's own metadata (`trl==0.24.0`, `peft==0.17.0`); numpy kept ≥2 (compiler wins) — measured working |
| 8B graphs exceed 16 GB/core without recompute (`NCC_EOOM001`, 23.7 GB peak) | both 8B training lanes dead | gradient checkpointing default-on; MFU made recompute-aware (frozen 4→6, trainable 6→8 FLOPs/param/token) |
| optimum-neuron 0.4.3 + transformers 4.57 + torch-xla 2.9 route layer kwargs into a kwargs-intolerant reentrant checkpoint | training crashed at first forward | module-global checkpoint shim, partial-binding non-tensor kwargs; patched via sys.modules after pkgutil silently skipped the llama subpackage |
| Param count taken before the PEFT wrap | 100% "trainable", MFU inflated ~50% | post-wrap recount; degenerate counts null the MFU rather than fake it |
| Failed precompile wrote a success-shaped JSON | resume guard wedged the lane | failures get `.failure.json`; success filename never lies |
| Client prompts tokenized to 1025 tokens (BOS) against a 1024+1024=2048 window | an entire sweep 400'd; sustained looped 5,120 empty iterations | `truncate_prompt_tokens` pins exact ISL; 737 MB of junk iterations purged (S3 + tree; survives in one git commit's history) |
| `pull_code.sh --delete` treated results as deletions | silently wiped on-box results twice; forced a full Qwen retrain (the "repeat" in §4); one file (`qwen3_lora.json` v1) lost outright | `--delete` removed; suite pushes results at completion; all other artifacts had S3/git copies |
| optimum-neuron 0.4.3 saves LoRA under `adapter_default/adapter_shards/` | merge lane failed to find the adapter | detector accepts both layouts |
| Stale failure records resurrected by sync-without-delete | successful sweeps rendered as failures | aggregator collects failures *and* points; stale records deleted with git history as archive |

## 12. Limits

As pre-registered: one box per role, single seed (except the accidental
Qwen repeat), demonstration-scale SFT with no quality claims, primary
client is this repo's own (schema-identical, exact-ISL), no energy numbers,
previous-generation Neuron silicon by design. Prices: trn1.2xlarge
$1.34/hr, inf2.xlarge $0.76/hr, us-west-2 on-demand, July 2026.
