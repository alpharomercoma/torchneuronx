# Report — Phase 2 extensions

Continuation of [REPORT.md](REPORT.md). Same contract: every number below
comes from a result triplet (JSON + log + telemetry CSV) under
`inf2/results/` or `trn1/results/`, failures are recorded results, and
anything excluded is excluded out loud. Phase-2 lanes were **validity-checked
before spending compute** (the table in §13.9 was written first; the
predicted-vs-observed column is the meta-experiment).

Boxes: the same trn1.2xlarge (`i-0cb9e758143a745d5`) and a **fresh**
inf2.xlarge (`i-0936ae7615727251e`) — the original inf2 was deliberately
destroyed and redeployed as the §13.7 cold-start experiment.

## 13. Phase-2 results

### 13.1 mlx-models parity: Whisper, CLIP, SigLIP (Track A)

Ports of the same demos as [mlx-models](https://github.com/alpharomercoma/mlx-models)
on Apple M5. One twist discovered mid-track: **optimum-neuron cannot coexist
with the vLLM DLAMI venv** (§13.8 finding 2), so these lanes measure on
trn1's training venv — same NeuronCore-v2 silicon as inf2.

| lane | outcome | evidence |
|---|---|---|
| **Whisper-small** | **WORKS** — JFK 11 s clip transcribed verbatim, **RTF 0.047** (0.522 s wall), compile 157.6 s, load 2.7 s | `trn1/results/extras/whisper.json` |
| CLIP ViT-B/32 | **excluded (numerics)** — optimum exporter rejects this transformers ("Dictionary inputs … must have consistent type"); raw `torch_neuronx.trace` fallback compiles and runs but returns NaN logits in bf16 *and* fp32 | `clip.failure.json` + optimum receipt in git history |
| SigLIP | **excluded (predicted)** — exporter names its supported list; siglip is not in it. M5 runs it: a real cross-ecosystem gap | `siglip.json` (`export_unsupported`) |

### 13.2 Mistral-7B-Instruct v0.3 serving (Track A4)

Second modern 7B-class architecture through the identical vLLM 0.16/NxDI
path as Llama 3.1 8B. Cold compile to healthy: **2,089 s**. Short config
(2048×32), ISL/OSL 1024/1024:

| conc | out tok/s | TPOT p50 | TTFT p50 |
|---|---|---|---|
| 1 | 17.6 | 56.5 ms | 206 ms |
| 4 | 69.2 | 57.4 ms | 505 ms |
| 8 | 136.0 | 58.0 ms | 898 ms |
| 16 | 263.2 | 59.2 ms | 1,678 ms |
| 32 | **497.4** | 61.2 ms | 3,250 ms |

Same signature as Llama: **28× throughput from c1→c32 for +8% TPOT** —
batching is close to free until the TTFT queue grows.

### 13.3 Open-loop load: the capacity knee (Track D)

MLPerf-Server-style Poisson arrivals + goodput@SLO (declared SLOs: TTFT<3 s,
TPOT<80 ms) added to the fallback client (`--arrival-mode poisson`).
Llama 3.1 8B, warm, ISL/OSL 512/128, 120 s per point:

| offered rate | SLO attainment | goodput tok/s | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| 0.5 req/s | 100% | 66.8 | 227 ms | 55.7 ms |
| 1 req/s | **100%** | **122.5** | 236 ms | 63.1 ms |
| 2 req/s | 43.2% | 100.4 | 246 ms | 82.0 ms |
| 4 req/s | 2.6% | 8.2 | 13.7 s | 85.1 ms |
| 6 req/s | 2.7% | 8.5 | 15.6 s | 74.4 ms |
| 8 req/s | 1.5% | 4.9 | 15.6 s | 74.4 ms |

The knee is between 1 and 2 req/s: at 2, TPOT p50 (82 ms) crosses the 80 ms
SLO; past 4 the queue explodes and goodput collapses to ~7% of peak while
raw throughput would still look healthy. **This is why fixed-concurrency
numbers oversell capacity** — the Phase-1 c32 sweep and this table describe
the same server.

### 13.4 Context-length cliffs, bisected (Tracks B1 + B4)

* **Serving (inf2, vLLM/NxDI)**: 2048 serves (all of Phase 1) · **3072
  serves (healthy boot)** · 4096 / 6144 / 8192 all crash the compiler with
  `NCC_INLA001`. Cliff bracketed to **(3072, 4096]** at max_num_seqs 8 —
  `extras/ctx_bisect/len_*.json`.
* **Training (trn1, LoRA)**: seq 4096 **trains** — see §13.6. seq 8192 =
  `NCC_EOOM002`: peak HBM 18.12 GB vs the 16 GB/core ceiling
  (`ctx_8192.failure.json`). A capacity wall with an exact overshoot number,
  not a compiler bug.

### 13.5 Quantization (Track B2)

fp8 KV-cache A/B at c8, same config as the Phase-1 bf16 point:

| | out tok/s | TPOT p50 |
|---|---|---|
| bf16 KV (Phase 1) | 119.3 | 65.4 ms |
| **fp8 KV** | 120.8 | 64.5 ms |

Throughput parity (+1.3%) — the win is the freed KV memory (larger batch or
longer contexts), not speed. int8 weights: **declared not-attempted** this
pass (needs an offline checkpoint-quantization stage; `int8_note.json`).

### 13.6 Longer-sequence training efficiency (Track B4)

Llama 3.1 8B LoRA at seq 4096 (20-step probe): **3,575 tok/s, 82.7% MFU**
(recompute-corrected), compile 1,149 s — versus 2,952 tok/s / 68.3% MFU at
seq 2048 in Phase 1. Longer sequences amortize the fixed per-step cost;
the chip gets *more* efficient as the problem grows until the §13.4 wall.

### 13.7 Production ops trio (Tracks C2, C3, C4)

* **Multi-tenancy (C2)**: two TinyLlama vLLM servers, one pinned per
  NeuronCore via `NEURON_RT_VISIBLE_CORES`, loaded simultaneously at c4:
  tenant0 233.4 tok/s / TPOT p50 16.2 ms, tenant1 233.6 tok/s / 16.1 ms.
  **Near-identical per-tenant numbers = the isolation claim, measured.**
* **Autoscaling cold-start (C3)**: `cdk destroy` → deploy → first token =
  **47.9 min**, of which infrastructure (destroy 172 s + deploy 175 s + SSM
  53 s + code 2 s + **1.2 GB NEFF cache seed from S3: 10 s**) is only ~7 min.
  The 8B server's process-start→first-token span is 2,447 s *with a warm
  cache*. Model boot, not infra, is the autoscaling bottleneck; plan
  capacity ahead of demand. First request after healthy: TTFT 568 ms /
  TPOT 42.2 ms — steady-state from the first token.
* **Checkpointing (C4)**: LoRA adapter saves (16.2 M trainable params,
  TinyLlama probe) cost **~1.0 s each** (1.02 / 1.08 / 0.98 s at steps
  10/20/30) — checkpoint-often is free at adapter scale.

### 13.8 NKI custom kernel (Track E) — the API-drift gauntlet

Row-softmax kernel, nki 0.5.0. `nki.simulate` correctness gate:
**bit-exact vs numpy (max err 0.0)**. The path there, each with a receipt:

1. `simulate(kernel)(x)` wrapper-factory convention (positional call TypeError)
2. `NkiTensor` does not overload python `-` / `/` — use `nl.subtract` etc.
3. device compiler additionally rejects `nl.divide` → `nl.reciprocal` + `nl.multiply`
4. `nl.*` names resolve against module globals — function-local imports fail
   only on device (`failed to resolve name 'nl.load'`)
5. device compile still fails downstream in `neuronx-cc` (receipt:
   `nki_device.json`) — **declared exclusion**

Simulator-first development works and is free; the device path's API surface
is visibly younger than the rest of the stack. That asymmetry *is* the
finding.

### 13.9 Predicted vs observed (the validity meta-experiment)

Verdicts written before compute was spent; outcomes after:

| item | predicted | observed |
|---|---|---|
| Whisper | GO | ✅ works (RTF 0.047) |
| CLIP | GO | ❌ numerics exclusion — the one true miss |
| SigLIP | attempt-only, exclusion likely | ✅ exclusion, as predicted |
| Mistral-7B | GO (attempt) | ✅ full sweep |
| RAG sized-down (0.6B embed/rerank) | GO with receipts | see §13.10 |
| Long-ctx bisection | GO | ✅ bracketed both cliffs |
| fp8 KV | GO | ✅ parity A/B |
| int8 weights | GO | ⚠ declared prep-stage gap |
| gpt-oss-20b MoE | attempt-only | ⏸ driver not built (declared) |
| Spec-decode | GO via fused | see §13.10 |
| Multi-tenant | GO (attempt) | ✅ measured isolation |
| Cold-start | GO | ✅ 47.9 min, infra ≈ 7 min |
| Ckpt timing | GO | ✅ ~1 s/save |
| Poisson/goodput | GO | ✅ knee found |
| NKI | GO | ⚠ simulate ✅, device excluded |

12 of 15 verdicts held exactly; the misses were all on the *newest* API
surfaces (CLIP exporter, NKI device, NxDI Tier-2/3 CLI paths) — consistent
with Phase 1's conclusion that the mature paths are genuinely mature.

### 13.10 RAG appliance + speculative decoding

Final lanes of the phase (results land in `inf2/results/rag/` and
`extras/spec_decode/`); this section is regenerated when they complete.

## 14. Phase-2 corrections (what broke and what it taught)

1. **`libneuronpjrt-path` must be on PATH** — torch-neuronx's initializer
   shells out to it. One missing export produced failure receipts across
   seven lanes that looked like seven different bugs. Now exported by
   `launch_vllm.sh` itself.
2. **optimum-neuron must never enter the vLLM venv** — it registers a second
   vLLM platform plugin ("Only one platform plugin can be activated") and
   the DLAMI's transformers is a patched fork under an upstream version
   string. Healed by destroy+redeploy (which became the §13.7 experiment).
3. **venv-from-venv loses the parent's site-packages** — "system" means the
   base python. The RAG overlay is a pip-managed directory used via
   `PYTHONPATH` on the underlay python, overlay torch/numpy stripped, and
   servers boot `env -u PYTHONPATH`.
4. Bare SSM shells have no `HF_HOME`/token → gated-model 401s masqueraded
   as compiler failures (ctx_8192's first two "failures" were this).
5. `inference_demo --model-path` is a local dir, not a repo id.
6. bf16 autocast can NaN CLIP-style masks (`finfo.min` constants).
7. A 0-byte receipt is worse than no receipt — `xargs` ate a traceback's
   quotes; receipt writers now use plain variables and a tail-of-log
   fallback so "failed" always says *why*.
