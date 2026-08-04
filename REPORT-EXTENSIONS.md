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
| RAG sized-down (0.6B embed/rerank) | GO with receipts | ✅ retrieval 7/7; reranker + co-residency receipted |
| Long-ctx bisection | GO | ✅ bracketed both cliffs |
| fp8 KV | GO | ✅ parity A/B |
| int8 weights | GO | ⚠ declared prep-stage gap |
| gpt-oss-20b MoE | attempt-only | ⏸ driver not built (declared) |
| Spec-decode | GO via fused | ✅ 2.4× fused speedup |
| Multi-tenant | GO (attempt) | ✅ measured isolation |
| Cold-start | GO | ✅ 47.9 min, infra ≈ 7 min |
| Ckpt timing | GO | ✅ ~1 s/save |
| Poisson/goodput | GO | ✅ knee found |
| NKI | GO | ⚠ simulate ✅, device excluded |

12 of 15 verdicts held exactly; the misses were all on the *newest* API
surfaces (CLIP exporter, NKI device, NxDI Tier-2/3 CLI paths) — consistent
with Phase 1's conclusion that the mature paths are genuinely mature.

### 13.10 RAG appliance (Track F)

Port of [local-agentic-rag-with-qwen3](https://github.com/alpharomercoma/local-agentic-rag-with-qwen3)
(3×8B Qwen on ≥48 GB VRAM) sized for a 32 GB Inferentia2: Postgres 16 +
pgvector (HNSW, 1024-dim) on the host CPU, **Qwen3-Embedding-0.6B and
Qwen3-Reranker-0.6B compiled to NeuronCores via raw `torch_neuronx.trace`**
(571 s / 909 s compiles), Llama 3.1 8B as the generator. Corpus: this
repo's own docs (72 chunks, ingested at 1.5 chunks/s end-to-end including
on-chip embedding).

* **Retrieval quality: 7/7 verification probes pass** (cosine top-3
  contains the expected fact for every REPORT-derived question) —
  `probes_nollm.json`.
* Per-stage query timings (fresh process; embed ~0.7–1.9 s at static
  batch 8 × seq 512 fp32, pgvector retrieve 0.2–0.5 s). The `e2e_ms`
  fields include a ~56 s traced-module load per probe process — a harness
  artifact, declared, not query latency.
* **Reranker: runs on-chip (~1.9 s/batch of 4) but degrades quality 7/7 →
  4/7** (`probes_nollm_rerank.json`). Root cause is visible in the compile
  log: the static trace **ignores the attention mask**, so left-padding
  attends to pad tokens and corrupts the yes-logit scores. No-rerank mode
  is the production configuration; the reranker stays an honest attempt
  receipt.
* **Co-residency wall, measured**: with the TP=2 8B server owning both
  NeuronCores, the co-resident embedder cannot initialize the runtime —
  all 7 LLM-stage probes errored (`probes_llm.json`). On a 2-core
  inf2.xlarge, generation + on-chip embedding need either a quantized
  TP=1 LLM (the declared int8 gap, §13.5) or a second box. The
  retrieval-first probe ordering was designed for exactly this outcome, so
  clean retrieval numbers exist regardless.

### 13.11 Speculative decoding (Track C1)

NxDI `inference_demo` fused speculation, Llama 3.1 8B target + Llama 3.2 1B
draft, TP=2, greedy, 1024-token context, 256 generated:

| | e2e latency (avg) | e2e throughput |
|---|---|---|
| baseline 8B | 8,587 ms | 149.1 tok/s |
| **fused spec (k=5)** | **3,569 ms** | **358.6 tok/s** |

**2.4× end-to-end speedup** on identical greedy output. Two receipts on the
way there: the fused path hard-requires on-device sampling
(`AttributeError` on `on_device_sampling_config` without it — the CLI help
dump is archived next to the results), and `--model-path` wants a local
snapshot directory, not an HF repo id.

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

---

## 15. Trainium1 vs Trainium2, one chip each (Phase 3)

**Status: infrastructure and harness complete, measurements not taken.** The
lane is blocked on AWS capacity, not on work. This section is scaffolded with
the declared design so that when a box lands, the numbers drop into a shape
that was fixed *before* any result was seen.

### 15.1 What is being compared, and why it is fair

One Trainium1 chip against one Trainium2 chip, on the identical workload:
Llama 3.1 8B, LoRA r16/α32, micro-batch 1, grad-accum 8, dolly-15k, 3 epochs,
seed 42, seq 2048 — the same lane list through the same `shared/run_all.sh`
branch, byte-for-byte. World = TP on both boxes, so the whole chip works on
every token and MFU is measured against the whole chip.

| | trn1.2xlarge | trn2.3xlarge |
|---|---|---|
| NeuronCores | 2 × v2 | 8 × v3 → 4 logical at LNC=2 |
| HBM / chip | 32 GiB | 96 GiB |
| HBM / logical core | 16 GiB | 24 GiB |
| BF16 dense peak | 210 TFLOP/s | 667 TFLOP/s |
| Ratio | — | **3.18×** |

3.18× is the ceiling the measurement is judged against. If achieved speedup
lands materially below it, that gap is the finding and gets profiled — not
buried.

### 15.2 The parallelism question was open, and was measured

AWS documentation does **not** state whether tensor-parallel/world size 4 is
valid on Trainium2. The often-cited "world size limited to 1, 2, 8, 32" line
appears on a page tagged for Trn2 — so it cannot be waved away as stale — but
it is worded as a performance *placement* heuristic and its examples
(0/8/16/24) are trn1.32xlarge-shaped. Meanwhile AWS's own `neuronx-distributed`
documentation uses `tensor_parallel_size=4` freely, no document states a
power-of-two or divisibility rule, and **no AWS example anywhere runs a single
Trainium2 chip at any TP**.

That absence is why `extras/tp_probe_trn2.sh` descends a declared ladder on
TinyLlama before any 8B lane is allowed to run, and keeps every rung's outcome:

| rung | LNC | world | TP | consequence if it wins |
|---|---|---|---|---|
| 1 | 2 | 4 | 4 | the whole chip; a clean 1:1 comparison |
| 2 | 2 | 2 | 2 | **half the chip idle** — every downstream number is labelled a partial-chip configuration |
| 3 | 1 | 8 | 8 | 8 physical v3 cores; per-rank HBM to be measured, not assumed |

### 15.3 The context cliff (Track B4)

The strongest single prediction this phase makes. On trn1, seq 4096 passed at
82.7% MFU and seq 8192 died:

```
NCC_EOOM002] Maximum peak HBM usage of 18.12GB exceeds HBM limit of 16.00GB for Trn1
```

24 GiB per logical core should clear 18.12 GiB. The ladder therefore extends to
**16384**, to locate the new cliff rather than merely confirm the old one moved.
Either outcome is a receipt.

### 15.4 Declared efficiency levers

Run only after the untuned baseline exists, each as its own lane with its own
triplet:

- **E1** seq 4096 as the efficient operating point (mirrors trn1's own
  2048→4096 gain of 68.3% → 82.7% MFU).
- **E2** activation recomputation off — worth +2·N FLOPs/token in the MFU
  accounting; more HBM may make it unnecessary.

### 15.5 How you actually get a Trainium2, and what it costs

A granted quota tells you nothing about whether you can launch. `L-2C3B7624`
was granted at 12 vCPU in sa-east-1 — exactly one trn2.3xlarge — and on
2026-08-04 all three AZs returned `InsufficientInstanceCapacity`, as did
us-east-2 for trn2.48xlarge. Spot placement scores were 1/10 and 3/10
respectively. Roughly ten hours of polling with `create-capacity-reservation`
(which fails in seconds and costs nothing, unlike `cdk deploy`) never once
found a free slot.

The signal that explained it was in the pricing API: **no on-demand record for
trn2.48xlarge at all**, only a Capacity Block line item. Trainium2 is sold
primarily through **Capacity Blocks**, which is precisely why the on-demand
pools are bare. The unblock was the corresponding quota — `L-64569A79`,
"Concurrent TRN2 Capacity Blocks per account", moving 0 → 192. Before that
grant `describe-capacity-block-offerings` returned
`CapacityBlockDescribeLimitExceeded`; after it, real inventory appeared
immediately. (The per-*organization* twin `L-24E8B4C0` remained 0 throughout and
gated nothing.)

**This is the first published price for trn2.3xlarge that we are aware of.**
A 24-hour block in sa-east-1b cost **$53.64**, i.e. **$2.235/hr**:

| | trn1.2xlarge | trn2.3xlarge |
|---|---|---|
| Rate | $1.34/hr | **$2.235/hr** (1.67×) |
| BF16 dense peak | 210 TFLOP/s | 667 TFLOP/s (3.18×) |
| **Peak TFLOP/s per dollar-hour** | 157 | **298 (1.90×)** |

So on *list* terms Trainium2 is a ~1.9× better deal per peak FLOP. Whether it
is a better deal per *delivered* token is the entire point of §15.1–15.4: peak
FLOPs are the denominator of MFU, not a result. If the measured speedup lands
materially below 3.18×, the price advantage narrows accordingly, and that is
the number a buyer actually needs.

Three procurement facts worth carrying into any Trainium2 plan:

1. **24 h is the minimum block**; 6 h and 12 h are rejected as
   `InvalidParameterValue`. You buy a fixed *window*, not a duration, and
   cancellation is not permitted.
2. **The launch must target the reservation.** `InstanceMatchCriteria` is
   `targeted`, so an ordinary launch does not fall into the block — it fails on
   capacity exactly like an unreserved attempt, while the block bills for its
   whole window regardless. Both `InstanceMarketOptions.MarketType =
   capacity-block` and the reservation ID are required.
3. **Instances are terminated, not stopped.** Blocks end at 11:30 UTC and AWS
   begins terminating at **11:00 UTC** on the final day. EBS goes with the
   instance, taking the warm NEFF cache with it — so a 24 h block is 23.5 h of
   usable time and the last half hour is a hard deadline, not a grace period.

For a talk about *production* LLMs on Trainium, this is more practically useful
than a benchmark: the newest accelerator generation is capacity-rationed and
sold in fixed pre-paid windows, while the previous generation — which the rest
of this report measures end to end — is the one you can get on demand.

## 16. Phase-3 corrections

1. **Trainium2 HBM is 96 GiB per chip, not 512.** `describe-instance-types`
   reports `TotalNeuronDeviceMemoryInMiB = 524288` for trn2.3xlarge, a
   single-chip instance. The Neuron architecture docs and the NKI architecture
   guide independently state 96 GiB @ 2.9 TB/s, consistent with trn2.48xlarge's
   advertised 1,536 GiB across 16 chips. An earlier note in this project
   repeated the API figure; it was wrong. Trust the docs over the API for
   device memory, and confirm with `neuron-ls` on the box.

2. **"There is no small trn2" was wrong.** An earlier claim in this project
   held that Trainium2 ships only as a 48xlarge. `describe-instance-type-
   offerings` disproves it: trn2.3xlarge exists, in sa-east-1 only. Check
   offerings per region before quoting instance sizes — AWS added the small
   size without it appearing in any US region.

3. **Do not assume 12 GiB per core at LNC=1.** The LNC documentation says both
   physical NeuronCores "have access to the entire 24GB HBM bank"; it does not
   say the bank is halved. The rung-3 design was corrected to measure rather
   than assume.

4. **"trn2.3xlarge is sa-east-1 only" is true for on-demand, not for Capacity
   Blocks.** `describe-instance-type-offerings` across all 17 enabled regions
   returned sa-east-1 alone, and correction 2 above records that. But the
   Capacity Blocks supported-regions table lists trn2.3xlarge in
   **ap-southeast-4 (Melbourne) as well as sa-east-1**. Availability is per
   *purchase mechanism*, not merely per region: an offerings scan can therefore
   understate where a type can actually be obtained.

5. **A capacity poller that discards stderr is not polling.** The on-demand
   watcher ran `create-capacity-reservation` with `2>/dev/null`, which made an
   expired AWS session indistinguishable from absent capacity. It ran ~9 hours
   and ~204 rounds in a state where it could not have succeeded even if a slot
   had opened; only the first hour's attempts are trustworthy evidence. Fixed by
   classifying the error text and exiting loudly on auth failures. The general
   rule: an unattended loop must treat "the answer I expected" and "I could not
   ask the question" as different outcomes.
