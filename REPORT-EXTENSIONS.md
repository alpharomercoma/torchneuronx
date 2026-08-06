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

6. **Neuron caches FAILED compiles, and a poisoned entry is silent.** Both
   Phase-3 serving grids failed at server boot with nothing useful at the
   caller — just `Engine core initialization failed`. The real cause was buried
   in the server log:

       [ERROR]: Got a cached failed neff at /opt/np/cache/.../model.neff.
       Will skip compilation, please set --retry_failed_compilation
       [NCC_INLA001] ... checkDMATranspose ...   2026-07-29T22:37:54Z

   A compile that died on 2026-07-29 left a **failed** NEFF in the cache. Every
   later run with the same cache key skips recompilation and inherits that
   failure, indefinitely, until someone passes `--retry_failed_compilation` or
   deletes the entry. Nothing surfaces at the call site.

   The upstream mistake was ours and is the more instructive half: the phase
   grids were pointed at the `long` geometry (9216) on the reasoning that it
   needed no recompile — **without checking whether `long` works**. It does
   not. `llama31_base_long: server_failed_to_start` is a recorded failure in
   this project's own report, and its cache is exactly where the poisoned NEFF
   lives. The evidence was already in `analysis/comparison.json` and went
   unread. *"Same geometry requested" is not "same working executable."*

7. **An undefined TPOT is not a satisfied TPOT.** `compute_goodput` counted
   `tpot is None` as meeting the decode SLO. That is defensible in the ordinary
   grids, where sub-2-token requests are a small minority, and it is the
   convention behind every published inf2 number. In the prefill grid it is
   fatal: `OSL=1` makes **every** request `tpot=None`, so a two-SLO goodput
   silently degenerates into a TTFT-only one and reports 100% attainment
   against a decode SLO that was never evaluated once.

   Fixed without restating published numbers: the pass-through convention is
   unchanged, and only the **degenerate** case — no evaluable TPOT anywhere in
   the run — now reports `null` goodput with an explicit reason, alongside a
   new always-defined `ttft_only_attainment_pct`. A first attempt that made
   undefined TPOT *fail* was caught by the existing test suite precisely
   because it would have moved published results.

## 17. Prefill vs decode on Inferentia2 (Phase 3)

Every serving grid in Phase 1 and 2 mixed the two phases of a request, so none
of them could show the mechanism the roofline analysis predicts. Two additive
grids fix that; nothing published is restated.

    prefill   input-length sweep at OSL=1   -> one output token, so end-to-end
                                               latency is essentially prefill
    decode    128 in / 1900 out             -> ~94% of tokens are decode steps

Both run on the **short** server geometry (`MAX_MODEL_LEN=2048`,
`MAX_NUM_SEQS=32`) — the same compiled graph as the published short lane — so
the only variable is request shape. Llama 3.1 8B Instruct, inf2.xlarge, one
Inferentia2, TP=2.

### 17.1 Prefill is compute-bound

Concurrency 1, p50 TTFT, throughput derived as `ISL / TTFT`:

| input tokens | TTFT p50 (ms) | prefill tokens/s |
|---|---|---|
| 256 | 116.2 | 2,204 |
| 512 | 199.8 | 2,563 |
| 1024 | 397.5 | 2,576 |
| 1792 | 422.3 | **4,244** |

Throughput **rises with prompt length** — the compute-bound signature. A longer
prompt raises arithmetic intensity and amortises fixed per-request cost, so the
accelerator gets *more* efficient the more work it is handed at once.

Stated rather than buried: TTFT includes admission, queueing, parsing, the
prefill pass, and emission of the first token. At concurrency 1 queueing is
negligible, so this is a close **lower bound** on true prefill throughput, not
a server-side phase timing.

### 17.2 Decode is memory-bandwidth-bound

| concurrency | output tokens/s | TPOT p50 (ms) | TTFT p50 (ms) |
|---|---|---|---|
| 1 | 17.83 | 56.02 | 66.0 |
| 4 | 70.89 | 56.37 | 157.1 |
| 8 | 140.68 | 56.75 | 279.1 |

**TPOT is flat to within 1.3% while aggregate throughput scales 1 : 3.98 :
7.89.** The flatness is the evidence, not the throughput. Each decode step must
stream the entire 8B weight set out of HBM to emit one token, so the weights
are already in flight; a second, fourth or eighth concurrent stream fills
otherwise-idle compute at almost no per-token cost. Were decode compute-bound,
TPOT would climb with batch size. It does not move. TTFT does climb
(66 -> 279 ms) — that is queueing, which should grow with concurrency.

### 17.3 The contrast, on one chip

| | tokens/s | improves with | bound by |
|---|---|---|---|
| prefill | 2,204 -> 4,244 | prompt LENGTH | compute |
| decode | 17.8 per stream (flat) | CONCURRENCY | memory bandwidth |

Roughly a **124-238x per-token gap between the two phases of the same
request** — same chip, same model, same server process — and the two phases
reward opposite levers. This is the clearest illustration in the study of why
an inference accelerator is specified the way it is, and why a single
"tokens/second" figure for a serving system means little without saying which
phase produced it.

### 17.4 An anomaly reported, not smoothed

1792 input tokens took only **6% longer** than 1024 (422.3 vs 397.5 ms) despite
75% more input, which is what produces the 4,244 tok/s outlier. The likely
cause is NxD Inference's compile-time **bucketing**: both lengths probably fall
in the same bucket, making the larger prompt nearly free. If so, prefill cost
is a **step function, not a line** — which matters directly to anyone sizing
prompts or padding batches. Recorded as an open question; confirming it needs
the bucket boundaries from the compiled artifacts.

### 17.5 What these grids could NOT do

The planned prefill sweep reached 8192 input tokens on the `long` geometry. It
could not run: see correction 6. The sweep therefore stops at 1792 and the
long-context half of the prefill curve is missing. That is the honest cost of
running on a server that starts.

8. **optimum-neuron 0.4.3 has NO evaluation API.** `NeuronSFTTrainer` does not
   implement `.evaluate()` -- confirmed on trn1:
   `AttributeError: 'NeuronSFTTrainer' object has no attribute 'evaluate'`.
   Held-out scoring, early stopping, and any eval-during-training workflow are
   simply unavailable through the supported training path. For a study whose
   every other number measures speed, this is the single biggest gap in the
   stack: it makes "did it train correctly?" harder to answer than "how fast
   did it run?".

   **Workaround that does work:** a zero-learning-rate pass over the held-out
   split using the same trainer machinery -- same collator, same packing, same
   compiled graph shapes, so no new API and no new compile. With `lr=0` and a
   constant schedule the optimizer cannot move a weight, so the logged per-step
   losses are forward losses on unseen rows. Gradients are still computed and
   discarded; that waste buys correctness through supported APIs instead of a
   hand-rolled XLA eval loop whose numerics would then need defending.

   Two things had to be fixed to make it run:

   - `max_grad_norm=0.0`. ZeRO-1 clips gradients before stepping, and on a
     frozen scoring pass there are none:
     `neuronx_distributed/parallel_layers/grads.py get_grad_norm` raises
     `IndexError: list index out of range` on the empty list. Clipping a
     gradient that will never be applied is pointless anyway.
   - Never re-apply a PEFT adapter to an already-wrapped model on the
     post-training pass.

9. **A receipt that cannot be acted on is not a receipt.** The first version of
   the held-out lane caught the exception and recorded only
   `"IndexError: list index out of range"` with no frame information. That is
   unactionable: it cost a full round trip to the box to learn nothing. Adding
   `traceback.format_exc()` to the receipt identified the ZeRO-1 clipping path
   on the very next run. METHODOLOGY rule 7 says failures are receipts; this
   sharpens it -- a receipt must carry enough context to fix the failure.

10. **`--seed` does not perturb this stack.** Three trn1 runs at seeds 42, 43
    and 44 produced BIT-IDENTICAL loss (tail-50 mean 1.102654, stdev 0.0),
    even though `seed=args.seed` is passed to the trainer. Data order is pinned
    elsewhere -- most likely by packing, which builds a fixed packed-example
    sequence. Consequences:

    - The variance lanes measure system TIMING only. That is still useful: the
      tokens/s spread across three identical runs is **2.4%**, which is the
      run-to-run noise floor any ratio in this report must clear.
    - Because runs are deterministic, the trn1/trn2 final-loss difference
      (1.2063 vs 1.1489) is **not** seed noise. It is real, and the most likely
      cause is TP=2 vs TP=4 changing collective accumulation order -- the same
      root as the `params_trainable` discrepancy in 15.x.
    - Repeating trn2 at seeds 43/44 is therefore redundant.

---

## 18. Do the two chips take the same training path? (Phase 3)

Section 15 shows Trainium2 is faster. The quality gate (§19) shows the
fine-tune learned. Neither answers a third question a reviewer will ask
immediately: the two chips ended at **different final losses** — 1.2063 on
trn1, 1.1489 on trn2 — so are they even training the same model?

This section answers it from data already on disk. `loss_trace` is recorded in
every result JSON (645 entries for the primary lane), so the comparison costs
no box time, no compile, and no new run. `analysis/loss_overlay.py` aligns the
two traces **on step number**, never on list position, and reports how the gap
evolves.

### 18.1 Why the gap cannot be seed noise

Three trn1 runs at seeds 42/43/44 produced **bit-identical** loss (tail-50 mean
1.102654, stdev 0.0). `--seed` does nothing on this stack because packing pins
the data order (§16, correction 6). The trajectory is deterministic, so a
0.0574 difference between the boxes is *structural*. Something about the two
configurations genuinely differs.

The candidate: the boxes run at different tensor-parallel widths. trn1 is TP=2
(one Trainium1, two NeuronCore-v2). trn2 is TP=4 (one Trainium2, eight
NeuronCore-v3 at LNC=2). TP width sets the order in which partial sums are
reduced across cores, and floating-point addition is not associative — so
identical mathematics, executed in a different reduction order, produces
slightly different bits, and gradient descent amplifies the difference over
hundreds of steps.

### 18.2 The divergence shape distinguishes the hypotheses

Accumulation order and "a materially different model" predict *different
shapes*, which is what makes this diagnostic rather than decorative:

- **Accumulation order** → curves indistinguishable early, gap growing slowly
  and monotonically as rounding differences compound.
- **A different model** (wrong weights, wrong data, a broken adapter) → an
  early split or a constant offset from the first steps.

Measured, primary lane, Llama 3.1 8B, 645 steps:

| window | steps | trn1 mean | trn2 mean | mean abs delta |
|---|---|---|---|---|
| first 10% (post-warmup) | 11–64 | 1.3848 | 1.3858 | **0.0080** |
| middle | 258–387 | 1.1772 | 1.1627 | 0.0145 |
| tail 50 | 596–645 | 1.1027 | 1.0748 | **0.0278** |

Pearson r = **0.996923** across all 645 steps. Mean absolute delta 0.0164;
largest single-step delta 0.0988 at step 62, early and transient. Sustained
divergence — the first step after which the gap never again falls below the
threshold — arrives at **step 544 of 645** for 0.01, and only at the final step
for 0.05.

That is the accumulation-order shape exactly: the gap **grows 3.5× from the
first tenth of training to the last fifty steps**, and the curves are
statistically inseparable early.

### 18.3 It replicates on a second architecture

Qwen3 8B, 624 steps, independently:

| window | trn1 mean | trn2 mean | mean abs delta |
|---|---|---|---|
| first 10% | 1.4335 | 1.4351 | 0.0088 |
| middle | 1.2817 | 1.2723 | 0.0094 |
| tail 50 | 1.2209 | 1.1991 | 0.0218 |

Pearson r = 0.997309, final gap 0.0252, sustained >0.05 divergence **never
occurs**. Same monotone growth, same early agreement, a different model family.
One lane showing this pattern is an anecdote; two independent architectures
showing it is a property of the configuration difference.

### 18.4 What this does and does not license

**Supports:** the two chips train the same model along the same trajectory, and
the final-loss gap is consistent with tensor-parallel reduction order rather
than a defect in either lane. It is the same root cause as the
`params_trainable` discrepancy in §16.

**Does not support:** any claim that one chip trains a *better* model. Loss on
the training distribution is not quality, and a 0.0574 difference is far too
small to interpret as such. Whether the fine-tune actually learned is the
quality gate's question, answered on held-out rows with a shared split seed.

**Does not isolate the variable.** TP width is confounded with the chip itself:
we cannot run trn1 at TP=4 (it has two cores) and did not run trn2 at TP=2. The
hypothesis is consistent with every measurement we have and no other
explanation fits the shape, but it remains inference from a correlated pair,
not a controlled experiment. Running trn2 at TP=2 would settle it and is
recorded here as the obvious follow-up.

---

## 19. The quality gate: did the fine-tune actually learn? (Phase 3)

Every other number in this study measures speed. A practitioner choosing
hardware will ask the obvious follow-up — *did the model actually train?* —
and until this lane existed the honest answer was "we don't know".

### 19.1 Why this is harder than it sounds on Neuron

`NeuronSFTTrainer` has **no working `.evaluate()`** on this stack. The
substitute is a zero-learning-rate forward pass over the held-out rows, reusing
the same collator, packing, and compiled graph shapes as training, so it needs
no recompile. Two things had to be discovered the hard way:

1. The frozen pass produces **no gradients**, and ZeRO-1's gradient clipping
   raises `IndexError` on the empty list. `max_grad_norm=0.0` disables the
   clipping and the pass runs.
2. The first receipt for that failure recorded only the exception *message*,
   which was not enough to diagnose anything. Receipts now carry
   `traceback.format_exc()`. That change is what turned an opaque failure into
   a one-line fix.

The held-out split uses a **fixed seed (20260805) applied before packing**, so
both chips score the identical rows. Splitting after packing would leak
training content into evaluation sequences, since packing concatenates
examples.

### 19.2 Result (Trainium1)

| lane | model | held-out loss before | after | delta | eval wall |
|---|---|---|---|---|---|
| `llama31_lora_holdout` | Llama 3.1 8B | 2.1491 | **1.2510** | **−0.898** | 843.8 s |
| `quality_smoke` | TinyLlama 1.1B | 1.7252 | **1.3962** | −0.329 | 291.7 s |

The primary lane's held-out loss falls by **0.898 nats on rows the model never
saw**. This converts "it ran fast" into "it trained correctly".

Training throughput during the quality lane was 2933.3 tok/s against 2951.8
tok/s published — a 0.6% difference, well inside the measured 2.4% noise floor,
so **adding evaluation did not perturb the training measurement**. `eval_wall_s`
is recorded separately from `train_wall_s` throughout, so no evaluation time
contaminates any throughput number.

### 19.3 What it does not license

This supports exactly one claim: the fine-tune learned, on held-out data. It
says nothing about instruction quality, MMLU, or any downstream benchmark.
Dolly SFT is not designed to move those and evaluator noise would dominate the
signal — which is why this study does not report them, on the explicit advice
of every adversarial reviewer consulted.

### 19.4 Both chips, byte-identical held-out rows

Trainium2 has now run the same gate, same script, same split seed, so both
chips were scored on the identical rows:

| lane | chip | before | after | delta | eval wall |
|---|---|---|---|---|---|
| Llama 3.1 8B | trn1 | 2.1491 | 1.2510 | **−0.8981** | 843.8 s |
| Llama 3.1 8B | trn2 | 2.1481 | 1.2652 | **−0.8829** | 610.3 s |
| TinyLlama | trn1 | 1.7252 | 1.3962 | −0.3290 | 291.7 s |
| TinyLlama | trn2 | 1.7250 | 1.3924 | −0.3327 | 506.0 s |

Two things worth stating.

**The starting losses agree to four decimal places** (2.1491 vs 2.1481;
1.7252 vs 1.7250). That is the split seed doing its job — the same untrained
model scored on the same rows — and it is the check that makes the *after*
column comparable at all. Had they differed, the deltas would be measuring
different examples.

**Both chips learn the same amount.** −0.8981 against −0.8829 on the 8B lane, a
difference of 0.015 nats. Read against §18 — where the training-loss gap grows
monotonically from accumulation order — this is the expected outcome: a small
numerical divergence that does not change what the model learned. Trainium2
reached it in 610 s of evaluation against trn1's 844 s.

The asymmetry this section previously disclosed is now closed. Both chips have
held-out loss before and after, from the same script, on the same rows.

### 19.5 The `--data-seed` lane failed on both chips

`quality_smoke_dataseed` produced a failure receipt on trn1 *and* trn2. It is
recorded rather than hidden: the lane was a probe of whether `--data-seed`
moves a trajectory that `--seed` does not, and it did not survive on either
chip. It failing identically on both is itself weak evidence that the cause is
the stack rather than the hardware.

---

## 20. Replication on a second physical chip (Phase 3)

An instance failure produced an experiment we would not otherwise have run. The
original Trainium2 box (`i-00e7b6117eac3a122`) was terminated after a host OOM
and the ASG replaced it with `i-0a3f33482fa319c76`, which started with an empty
results directory and re-walked the whole suite. The primary lane therefore ran
twice, same code, same hyperparameters, on **two different physical Trainium2
chips in the same availability zone**.

Almost no published benchmark reports this. It is the difference between "we
measured a chip" and "we measured *this* chip".

| | original chip | replacement chip | delta |
|---|---|---|---|
| tokens/s (steady) | 3532.8 | 3618.0 | **+2.41%** |
| median step | 4637.7 ms | 4528.5 ms | −2.35% |
| train wall | 3322.1 s | 3220.1 s | −3.07% |
| MFU (/667) | 25.85% | 26.47% | +0.62 pp |
| compile | 19.4 s | 18.3 s | −5.7% |
| **final loss** | **1.1489** | **1.1489** | **0.000000** |

Two independent findings, and the second is the more interesting one:

**Timing varies by 2.4% between physical chips.** That is the same magnitude as
the noise floor measured *within* a single box across three seeds. So the
2.4% figure is not an artifact of one machine — it holds across hardware, and
it is the resolution limit for every throughput number in this study. Any claim
of a difference smaller than ~2.4% is not supported by this methodology, and
§15 should be read with that band applied.

**The final loss is bit-identical across different silicon.** Not close —
identical to four decimal places, the same 1.1489. Combined with the
bit-identical results across three seeds on trn1, this establishes that the
stack is deterministic *end to end*: same code and same configuration produce
the same weights regardless of which physical chip executes them. That is a
strong and genuinely useful property for reproducibility, and it is what makes
§18's argument possible — because runs are bitwise reproducible, the trn1/trn2
loss gap must come from the *configuration* difference (TP width), since
nothing else varies.

It also retires a possible objection to §18: someone could argue the trn1/trn2
loss gap is just hardware variation. It is not. Hardware variation here is
exactly zero.

---

## 21. Killing our own best explanation: the dataloader isolation lane (Phase 3)

§15's headline is that Trainium2 is **1.20× faster than Trainium1 at seq 2048
but 1.92× at seq 4096**. The weaker number is the one we lead with, so we owe
the reader an explanation of why the same two chips produce two such different
ratios.

The explanation we believed, and which two independent adversarial reviewers
proposed unprompted, was **host-side dataloader cost**. Tokenising, packing,
collating, and copying to the device costs roughly the same wall clock on both
boxes, because both have ordinary CPUs. A fixed host cost is a larger fraction
of trn2's shorter step than of trn1's longer one, so it would compress the
observed ratio — and doubling the sequence length doubles device work while
leaving host work alone, which would make the ratio open up. That is exactly
the shape of 1.20× → 1.92×. It is a good hypothesis. It is also wrong.

### 21.1 The test

`--synthetic-data N` replaces the dataset with pre-tokenised random token IDs
of exactly `seq_len`: no Hub download, no tokeniser, no packing, no formatter.
The compiled graph sees identical shapes, so the NEFF cache hits and nothing
recompiles. **The only thing removed is host-side data preparation.**

The discriminating comparison is not synthetic-vs-real at one shape — it is how
the uplift *changes* with sequence length. So each shape gets its own real-data
control at a matched 40 steps. Comparing a 40-step synthetic run against the
published 645-step lane would confound the uplift with warmup amortisation.

### 21.2 Result (Trainium1)

| shape | real data | synthetic | uplift |
|---|---|---|---|
| seq 2048 | 2940.1 tok/s | 2936.5 tok/s | **0.999×** |
| seq 4096 | 3571.1 tok/s | 3570.5 tok/s | **1.000×** |

Both real-data controls validate against the published lanes — 2940.1 vs
2951.8 (0.4%) and 3571.1 vs 3575.0 (0.1%) — so these are sound measurements,
not a broken comparison.

Removing **all** host-side data preparation changed throughput by −0.12% and
−0.02%. That is not merely inside the 2.4% noise floor of §20, it is inside a
twentieth of it. Host dataloader cost is not a measurable share of a training
step on this stack, at either shape.

**The hypothesis is dead.** It was our best explanation, it was independently
proposed by two reviewers, and it is not what is happening.

### 21.3 What the answer actually looks like

With the host ruled out, the 1.20×/1.92× split has to come from the device, and
the MFU column already showed it in plain sight:

| seq | trn2 MFU | trn2 vs trn1 |
|---|---|---|
| 2048 | 25.9% | 1.20× |
| 4096 | 50.3% | 1.92× |
| 8192 | 60.8% | trn1 cannot run it |

Trainium2 is not being slowed down at seq 2048 — it is **not being filled**. At
micro-batch 1 and seq 2048 there is not enough work in a step to occupy a chip
with 3.5× the peak FLOPs and 3× the HBM, so most of it idles and the ratio
collapses toward parity. Trainium1, at 75.2% MFU on the same shape, is close to
saturated. The gap between the chips is therefore not a gap in speed but a gap
in **how much work you must bring to make the bigger chip worth its price**.

That reframes the buying advice. Trainium2 is not "1.20× a Trainium1" — that
number describes a chip running someone else's problem size. It is 1.92× at
4096, 2.21× end-to-end on the same job, and it runs seq 8192 at all, which
trn1 cannot at any speed.

### 21.4 Trainium2 ran it too, and the null holds

trn1 was the weaker direction of the test — it has the longer step, so a fixed
host cost should matter least there. Trainium2 is the chip whose headline the
hypothesis was invented to explain, so its result is the one that counts:

| shape | real | synthetic | uplift |
|---|---|---|---|
| seq 2048 | 3712.9 tok/s | 3795.6 tok/s | **1.022×** |
| seq 4096 | 7339.8 tok/s | 7317.5 tok/s | **0.997×** |

The uplift at 2048 is **+2.2%**, and it does shrink to nothing at 4096 — which
is the *direction* the hypothesis predicts. But this study's own resolution
limit is **2.4%** (§20, measured both across seeds and across two physical
chips). The effect is smaller than the smallest difference the methodology can
resolve, so it cannot be reported as an effect.

**A correction worth recording.** The lane's own summary script initially
declared this "a real and shape-dependent host cost", because its threshold was
2% — below the study's 2.4% noise floor. That is exactly how a noise reading
becomes a finding: a threshold chosen without reference to the measurement's
resolution. The threshold now derives from the noise floor, and the stored
summary carries both the corrected reading and the superseded one.

The honest conclusion across both chips: host dataloader cost is bounded
**below 2.4%** of a training step at both shapes. That is not zero, and at
seq 2048 on trn2 it may be a real ~2%, but it cannot account for a gap between
1.20× and 1.92×. The occupancy explanation in 21.3 stands.

### 21.5 Limits, stated plainly
- A null result bounds host cost below the noise floor; it does not prove it is
  exactly zero.
- The occupancy explanation in 21.3 is consistent with every measurement we
  have — the MFU ladder, the null here, and the context ladder — but it is an
  inference from throughput, not a profiler trace. Confirming it would need
  `neuron-profile` on both shapes, which this window did not have room for.

We are reporting a hypothesis we liked, the experiment that killed it, and the
explanation we now believe with its evidence and its remaining uncertainty. The
alternative was to leave a plausible-sounding story in the report that we had
the means to test and did not.

---

## 22. The micro-batch ladder: an experiment that could not run (Phase 3)

§21 concluded that Trainium2 at seq 2048 is not slowed but **starved** — 25.9%
MFU against Trainium1's 75.2% on the identical shape. That was an inference
from throughput. This lane was built to test it directly with the one lever not
yet pulled: bring more work per step, and see which chip absorbs it.

**It could not be done.** Neither chip will compile a micro-batch above 1 at
seq 4096. The lane's value is therefore not the answer it was designed to give,
but the constraint it discovered, and the honest report of a hypothesis left
untested.

### 22.1 Result

| micro-batch | trn1 | trn2 |
|---|---|---|
| 1 | 3570.4 tok/s, MFU 91.3% | **7004.7 tok/s**, MFU 51.3% |
| 2 | `NCC_EXTP003` compiler limit | `NCC_EXSP001` device HBM |
| 4 | `NCC_EXTP003` | `NCC_EXSP001` |
| 8 | `NCC_EXTP003` | `NCC_EXSP001` |

The baseline rung is a useful independent check: **1.96×**, from a lane with
its own step count and its own session, against the published 1.92× at seq 4096
(§15). Two independent measurements 2.2% apart — inside the 2.4% noise floor of
§20.

### 22.2 The two chips wall in different subsystems *at this shape*

This is the substantive finding, and it is not one we went looking for.

> **Correction, added after the trn1 symmetry pass (§26.1).** The statement
> below is true of the seq-4096 micro-batch ladder and **must not be
> generalised**. At seq 2048, Trainium1 fails with `NCC_EXSP001` — the same
> *device memory* code as Trainium2 — not with the compiler-instruction error.
> Trainium1 is not "the chip that hits compiler limits"; it hits whichever
> ceiling the shape reaches first, and so does Trainium2.

- **Trainium1 hits a *compiler* ceiling.** `NCC_EXTP003`: 2,064,384 instructions
  generated against a stated limit of 150,000. The silicon is never consulted;
  the toolchain refuses to emit the graph.
- **Trainium2 hits a *device memory* ceiling.** `NCC_EXSP001`: 64.13 GB of HBM
  required against 25.77 GB available per logical core at LNC=2 / TP=4. The
  graph compiles conceptually; it will not fit.

A practitioner sizing a job needs to know these are different problems. The
Trainium1 wall might move with a compiler release. The Trainium2 wall moves only
with sharding, a smaller shape, or more silicon.

### 22.3 What is NOT established, and what was ruled out

On Trainium2 the reported memory requirement **changes** across rungs — 64.13,
64.15, 64.18 GB — so those were genuinely independent compiles.

On Trainium1 the instruction count is **identical** at every failed rung:
2,064,384 at micro-batch 2, 4 and 8. Graph size must grow with micro-batch, so
this is not explicable as three measurements of three graphs. Two candidate
explanations were tested and both failed:

1. **A cached failure.** Ruled out. The re-run used a different compiler-flag
   hash (`+f7f529f3` → `+e30acd3a`) and each rung emitted its own distinct
   module hash. The compiles were fresh.
2. **The ladder holding the varied quantity constant.** The first design set
   `grad_accum = 8 / micro_batch` to hold global batch fixed, which — since
   gradient accumulation is unrolled into the compiled graph here — made every
   rung compile the same total work. Plausible, and ruled out: the corrected
   design fixes `grad_accum` at 8, so unrolled work varies 8× across the rungs,
   and the count is **still identical**.

We do not know why. The summary carries a `SUSPECT` flag saying so, and this
section will not offer a third theory it has not tested. What is measured is
that at seq 4096 on Trainium1, micro-batch 1 compiles and nothing above it does.

### 22.4 Three defects in this lane, all found and all disclosed

Recorded because the failure modes generalise to any parameter sweep:

1. **No `--retry_failed_compilation`.** Neuron caches failed compiles, so the
   first failure would have been reported for every rung above it. Fixed.
2. **The validity check ran after the summary was written**, so it flagged an
   object nobody reads. The first corrected run duly produced three identical
   counts and a summary that said nothing. Fixed.
3. **The design varied two levers at once** and, worse, held their product
   constant — so the ladder never varied the thing it existed to vary. Fixed by
   holding `grad_accum` constant and letting global batch grow.

The general rule this produced: **a parameter sweep must record a quantity that
is expected to change with the swept parameter.** Without it, a cache hit, a
degenerate design, and a real measurement are indistinguishable. The instruction
count is now recorded in every receipt for exactly that reason — and it is what
exposed all three defects above.

### 22.5 Consequence for §21

The occupancy explanation is **not confirmed and not refuted**. It remains the
only hypothesis consistent with every measurement — the MFU ladder, the
dataloader null, the context ladder — but the direct test is unavailable on this
stack, because the toolchain and the HBM both refuse the larger shapes it would
require. Varying sequence length (§15) stays the only working instrument for
work-per-step on these chips, and a `neuron-profile` trace remains the way to
settle it.

---

## 23. Provenance repair: the replacement instance overwrote 13 published files

An audit of every number in §15–§22 against the stored JSON found **no incorrect
value** — but it did find that six of them were no longer traceable to the file
they came from. That is a defect in its own right, and this section records it
rather than quietly fixing it.

### 23.1 What happened

The original Trainium2 instance (`i-00e7b6117eac3a122`) ran the full suite and
pushed its results to S3. It was later terminated after a host OOM and the ASG
replaced it with `i-0a3f33482fa319c76`, which started with an empty results
directory and **re-ran the whole suite, pushing to the same S3 keys**. Thirteen
result files were overwritten:

```
compile/llama31_train.json      extras/tp_probe.json
cpu/cpu.json                    extras/tp_probe_lnc2_tp4.json
extras/ctx_4096.json            extras/tp_probe.failure.json
extras/ctx_8192.json            train/llama31_lora.json
extras/ctx_16384.failure.json   train/qwen3_lora.json
extras/tp_preflight_8b.json     train/merge_llama31.json
                                train/smoke_tinyllama.json
```

So §15 cited 3532.8 tok/s while the file at that path had come to contain
3618.0 — the *replacement* chip's number. Both are real measurements of a real
Trainium2. They are simply from **different physical chips**, which is precisely
the distinction §20 is about.

### 23.2 Recovery

S3 bucket versioning was enabled, so the original chip's versions were still
retrievable. All thirteen were recovered by version ID and are now stored under
a path the suite never writes to:

```
results/trn2/original_chip/...        (S3, and trn2/results/original_chip/ locally)
```

Every number in §15 was then re-verified against the recovered files:

| cited in §15 | recovered original | match |
|---|---|---|
| 3532.8 tok/s | 3532.8 | ✅ |
| 4637.7 ms median step | 4637.72 | ✅ |
| 3322.1 s train wall | 3322.13 | ✅ |
| 3181.0 tok/s end-to-end | 3181.0 | ✅ |
| 25.9% MFU | 25.851 | ✅ |
| 1.1489 final loss | 1.1489 | ✅ |
| Qwen3 1.24× / 2.04× | 1.237 / 2.045 | ✅ |
| ctx 4096: 6878.4, 50.3% | 6878.4, 50.332 | ✅ |
| ctx 8192: 8310.5, 60.8% | 8310.5, 60.812 | ✅ |

**No published number changed.** Every one was correct as written; they had
simply become unverifiable from the current files.

### 23.3 Which chip each section cites

- **§15 and the context ladder** — the ORIGINAL chip, `original_chip/`.
- **§20 (replication)** — both, deliberately: that section exists to compare
  them, and its 2.41% timing spread and bit-identical loss are exactly this
  pair.
- **§19, §21, §22** (quality gate, isolation, batch ladder) — the REPLACEMENT
  chip only. Those lanes were built after the original instance was gone, so
  there is no ambiguity and nothing was overwritten.

### 23.4 Derived metrics, and which numbers are not stored anywhere

The same audit found a second, smaller traceability gap: **two frequently
quoted figures are computed, not stored**, and the report had not said so.

| figure | how it is derived | worked example |
|---|---|---|
| trn1 end-to-end **1441 tok/s** (§15) | `steps x tokens_per_optimizer_step / train_wall_s` | 645 x 16,384 / 7333.66 s = 1441.0 |
| Inferentia2 prefill **2204 -> 4244 tok/s** (§17) | `prompt_tokens / p50_ttft` | 256 / 0.11616 s = 2204; 1792 / 0.422285 s = 4244 |

Both are correct and both are reproducible from stored fields, but neither
appears as a key in any result JSON. `tokens_per_s_end_to_end` was added to the
schema partway through Phase 3, so the older trn1 lanes predate it and
`analysis/make_report.py` backfills the value with `end_to_end_fields()`. The
Inferentia2 prefill throughput is not a vLLM metric at all — vLLM reports
*output* throughput, which for a one-token completion is ~2.5 tok/s and says
nothing about prefill. The prefill rate has to be reconstructed from TTFT, and
that reconstruction is the whole reason §17 could separate the two phases.

Stating the derivation matters because a reader checking `output_throughput` in
the prefill JSONs would find 2.51 tok/s and reasonably conclude the report was
wrong.

### 23.5 The lesson

A results path that a re-run will write to again is not an archive. Publishing a
number pins it to a file, and any process that can rewrite that file can break
the citation without changing the claim. Versioning saved this; without it the
original measurements would have been unrecoverable and §15 would have had to be
re-derived from a different chip.

---

## 24. Lanes that ran on both chips but were not yet written up (Phase 3)

A coverage audit found nine lanes with results on disk and no section citing
them. Every number below comes from the stored JSON.

### 24.1 Checkpoint save time is identical on both chips

TinyLlama, 30 steps, saving every 10:

| chip | save 1 | save 2 | save 3 | training tok/s |
|---|---|---|---|---|
| trn1 | 1.02 s | 1.08 s | 0.98 s | 4435.2 |
| trn2 | 1.05 s | 0.94 s | 0.99 s | 5616.3 |

Roughly **1 second per checkpoint on both**, with no trend across saves. This is
a useful negative result: checkpointing is not a differentiator between the
generations, because it is bound by host I/O to the local NVMe rather than by
anything on the accelerator. Anyone budgeting checkpoint overhead can use the
same figure for either chip.

### 24.2 Efficiency sweeps: which levers actually move Trainium2

Each lever applied *individually* to the primary lane, never silently folded
into it:

| lever | tok/s | MFU | vs primary lane (3532.8) |
|---|---|---|---|
| baseline (seq 2048, mb 1, recompute on) | 3532.8 | 25.9% | — |
| **seq 4096** | **7277.4** | **53.3%** | **2.06×** |
| gradient checkpointing OFF | 5265.5 | 25.7% | 1.49× |
| micro-batch 2 | `NCC_EXSP001` device HBM | — | fails |
| micro-batch 4 | `NCC_EXSP001` device HBM | — | fails |

Sequence length is the only lever that produces a large gain, and it more than
doubles throughput. Disabling activation recomputation gives 1.49× in tokens/s
but **MFU barely moves** (25.9% → 25.7%) — because switching recompute off also
removes FLOPs from the numerator, so the chip is doing less work per token, not
using itself better. That distinction is exactly why MFU and tokens/s are both
reported.

Micro-batch is unavailable at all: both rungs hit the device HBM ceiling, which
is the same wall §22 found at seq 4096.

### 24.3 Small models: where these chips are worst

MNIST and CIFAR-10, three architectures each, final test accuracy:

| task | trn1 | trn2 |
|---|---|---|
| MNIST MLP | 97.93% | 97.80% |
| MNIST CNN | 98.94% | 99.00% |
| MNIST ViT | 97.59% | 97.88% |
| CIFAR MLP | 52.16% | 52.24% |
| CIFAR CNN | 78.93% | 78.65% |
| CIFAR ViT | 80.64% | *not run* |

Accuracies match within 0.3 points everywhere, which is the point: **the chips
compute the same answers**, and these lanes exist to show the harness is honest
on workloads where an accelerator this large has no advantage at all. A
Trainium2 running MNIST is almost entirely idle. Nobody should buy one for this,
and the accuracy parity is the evidence that a difference elsewhere is a
performance difference rather than a correctness one.

### 24.4 Non-LLM architectures: NeuronCore-v3 fixes a v2 failure

| model | trn1 (v2) | trn2 (v3) |
|---|---|---|
| Whisper | ✅ | ✅ |
| SigLIP | ✅ | ✅ |
| **CLIP** | **✖ recorded failure** | **✅ passes** |

CLIP fails to compile on Trainium1 and compiles and runs on Trainium2. This is
the second case in the study of a recorded Phase-2 failure turning into a pass
on new silicon (the first being seq 8192), and it is the kind of result that
only appears if failures are kept as artifacts rather than discarded.

### 24.5 Mixture-of-Experts is not supported for training on Neuron

`Qwen3-30B-A3B` fails immediately with:

```
ValueError: Model type qwen3_moe is not supported for task text-generation
in neuron in training mode. Supported types are: ['llama', 'granite', 'qwen3']
```

This is a **capability limit, not a resource limit** — the architecture is
rejected by an allowlist before any compilation or memory allocation is
attempted. For a practitioner evaluating Trainium for MoE fine-tuning, this is
the single most decisive fact in the study, and it costs nothing to discover.

Note the receipt for this lane originally captured only the torchrun
`ChildFailedError` banner, which hid the real cause. It was rewritten from the
log. A receipt that records the wrapper's error rather than the program's error
is nearly worthless — the same lesson as adding `traceback.format_exc()` in §19.

---

## 25. Maximum utilisation, and four results that came with caveats (Phase 3)

### 25.1 The best configuration is 2.30x faster and finishes no sooner

The maxutil lane takes the best operating point the efficiency sweeps found and
runs the **full 3 epochs** — not a step-capped probe — so its wall clock is
directly comparable to the primary lane on the same chip. The selector chose
seq 8192, micro-batch 1, gradient checkpointing on: one lever changed from the
published configuration.

| | primary (seq 2048) | maxutil (seq 8192) | ratio |
|---|---|---|---|
| steady-state tokens/s | 3618.0 | **8336.6** | **2.30×** |
| MFU | 26.5% | **61.0%** | 2.30× |
| tokens processed | 10,567,680 | 10,813,440 | 1.02× |
| compile | 18.3 s | 42.6 s | 2.33× |
| **train wall** | **3220.1 s** | **3310.9 s** | **1.03× SLOWER** |
| **end-to-end tokens/s** | **3281.8** | **3266.0** | **0.995×** |

Reading only the first two rows, seq 8192 looks like a decisive win: 2.30×
throughput, MFU from 26.5% to 61.0%. Reading the wall clock, it processed 2%
more tokens in 3% more time and finished **no sooner**.

Both are true, and the reconciliation is in a field this harness records for
exactly this purpose:

| | steps measured | summed step time | wall | **measured fraction** |
|---|---|---|---|---|
| primary | 635 | 2869.5 s | 3220.1 s | **89.1%** |
| maxutil | 155 | 1212.2 s | 3310.9 s | **36.6%** |

At seq 2048, 89% of the job is inside the timed steps. At seq 8192, only **37%**
is. The device is genuinely 2.30× faster while it is computing, and it spends
nearly two thirds of the job not computing. Longer sequences mean far fewer,
much larger optimizer steps, and everything between them — data preparation,
packing to 8192, the dataloader, checkpoint and logging boundaries — is
unchanged in absolute terms while the step count collapses from 635 to 155.

**This is the sharpest illustration in the study of why MFU is a poor
purchasing signal.** MFU rose 2.3×. Time-to-result did not move. A practitioner
who tuned on MFU alone would have declared a large win and shipped a job that
finishes at the same time.

**It is not a Trainium2 quirk.** Trainium1 was given the same treatment at ITS
best shape (seq 4096, 91.4% MFU), full 3 epochs, and behaves identically:

| | trn1 @ seq 4096 | trn2 @ seq 8192 |
|---|---|---|
| steady-state vs its own primary lane | **1.21×** | **2.30×** |
| MFU | 91.4% | 61.0% |
| in-window fraction | **40.0%** | **40.3%** |
| untimed wall clock per step | 14.33 s | 12.76 s |
| **end-to-end vs its own primary lane** | **0.98×** | **0.995×** |

Two different chips, two different sequence lengths, two different MFU regimes —
and the same outcome: the in-window fraction collapses to **40%** and end-to-end
throughput does not improve at all. trn1's best configuration is 2% *slower*
end to end than its published lane, as trn2's is 3% slower in wall clock.

The effect therefore belongs to the shape of the work, not to the silicon:
fewer, larger optimizer steps amortise a fixed per-step overhead over far fewer
steps — and §21 and §28 already established that overhead is **not** host data
preparation. Anyone tuning sequence length upward on either generation should
expect the throughput metric to improve while the job finishes at the same time.

It also bounds §21's null. Host cost was below the 2.4% noise floor at seq 2048
and 4096; at seq 8192 the non-step fraction is 63%. We did not run the isolation
lane at 8192 and cannot attribute that gap to the dataloader specifically, but
it is clearly not negligible at that shape, and the honest statement is that
§21's bound applies **only to the two shapes it tested**.

### 25.2 Full fine-tuning of a 1.7B model requires Trainium2

Same script, same model, no LoRA:

```
trn1: NCC_EOOM001 -- peak HBM 19.46 GB exceeds the 16.00 GB limit for Trn1
                     (14.42 GB of it I/O tensors)
trn2: 6172.9 tok/s
```

This is a **capability** difference, not a performance one, and it is the third
recorded Trainium1 failure that becomes a Trainium2 pass — after seq 8192 and
CLIP. It also sharpens the case for LoRA on the older part: on trn1, LoRA is not
merely the faster choice for a model this size, it is the only one that fits.

TinyLlama full fine-tuning runs on both: 7209.1 tok/s on trn1, 8811.6 on trn2.

### 25.3 FP8: one chip cannot use it, the other cannot trust it

`--auto-cast-type=fp8_e4m3`, TinyLlama, 20 steps, on both chips, against the
BF16 lane of the same model and step count:

| | tokens/s | vs BF16 | final loss |
|---|---|---|---|
| **trn1** BF16 | 4486.0 | — | 1.3962 |
| **trn1** FP8 | 4420.0 | **−1.5%** | **1.5041** (finite) |
| **trn2** BF16 | 5412.7 | — | 1.2899 |
| **trn2** FP8 | **5699.6** | **+5.3%** | **NaN** |

Two different failures, and neither is the one we expected.

**Trainium1 computes FP8 correctly and gains nothing.** The loss is finite and
in the normal range; throughput moves −1.5%, inside the noise floor. This is
exactly what the hardware documentation predicts, and it is worth having as a
measurement rather than an inference: the Trainium1 architecture page lists
cFP8 at the **same 190 TFLOP/s as BF16**, so there is no FP8 throughput to
capture on that part. The flag is honoured and the silicon has nothing extra to
give.

**Trainium2 gains ~5% and produces NaN from the very first step.** Not a
divergence part-way through — every one of the 20 logged steps is NaN. The lane
"succeeded" in the sense that it ran to completion and wrote a result file, and
that result file describes the speed of computing nothing usable.

The 5.3% is above the 2.4% noise floor, so it is a real throughput difference.
It is also **not a benefit**, because no model was trained. Reporting "FP8 is
5.3% faster on Trainium2" without the loss column would be one of the most
misleading sentences this study could produce.

It is also nowhere near what the FLOPs table implies. Trainium2 lists 667
TFLOP/s BF16 against **1299 FP8** — close to 2× — and the measured difference
on a lane that does not even produce finite numbers is 5%. Whatever the flag is
doing, it is not putting the FP8 datapath to work.

**What this replaces.** §16 recorded, on the advice of an adversarial reviewer,
that the FP8 gap is headroom an H100 could exploit and we could not measure. We
can now say something better and more specific: **on this software stack, at
this Neuron version, a one-flag FP8 autocast delivers no speedup on Trainium1
and no usable numerics on Trainium2.** Getting real FP8 training value on
NeuronCore-v3 evidently requires more than the autocast flag — quantisation
recipes, loss scaling, or per-layer control that this study did not attempt.

That is a narrower claim than "FP8 is unreachable" and a far more useful one: it
tells a practitioner the easy path does not work and roughly where the
difficulty lies, rather than leaving the whole subject unmeasured.

**Caveat.** One model, 20 steps, one flag. A NaN this immediate usually means a
numerics or scaling problem rather than a hardware defect, and a longer run or a
different model might behave differently. What is established is that the
simplest available route to FP8 training does not work on either chip today.

### 25.4 The v3 compiler needs more host memory than v2 for the same model

`cifar_vit` — a small Vision Transformer on CIFAR-10 — compiles and trains on
Trainium1 and **cannot be compiled on Trainium2**:

```
trn2: [F137] neuronx-cc was forcibly killed -- This most commonly occurs
             due to insufficient system memory
```

| | host RAM | swap | cifar_vit |
|---|---|---|---|
| trn1.2xlarge | 30 GiB | 63 GiB | ✅ passes |
| trn2.3xlarge | 124 GiB | 63 GiB | ✖ compiler killed |

The chip with **four times the host memory fails**, and it failed again after a
64 GiB swapfile was added. The same source, the same script, the same dataset.

We are not able to prove the mechanism from the outside. The compile targets
differ, and at the LNC=2 default one logical v3 core spans two physical cores,
so the graph being compiled is not identical even though the model is. What is
measured is that the v3 toolchain's host-memory appetite for this model exceeds
a 124 GiB machine while the v2 toolchain fits in 30 GiB. Anyone sizing a build
host for Trainium2 should not assume the instance's own RAM is sufficient.

This is also the second host-memory compile failure on trn2, after `ctx_16384`.
Two independent lanes hitting the same wall makes it a property of the
toolchain rather than an accident of one graph.

### 25.5 The no-packing lane is NOT comparable, and we are reporting it anyway

| lane | trn1 loss | trn2 loss | trn1 tok/s | trn2 tok/s |
|---|---|---|---|---|
| `data_alpaca` (packed) | 1.0088 | 1.0088 | 2834.0 | 3696.4 |
| `data_dolly_nopack` | **5.4633** | **5.4631** | 2892.5 | 3708.6 |

The alpaca loss is **identical to four decimals on both chips**, which is
another instance of the determinism established in §20.

The no-packing loss of ~5.46 against ~1.15 packed is far too large to read as
"packing improves quality", and the throughput figure is worse than useless:
`tokens_per_optimizer_step` is computed as `seq_len x micro_batch x grad_accum`,
which assumes every position carries a real token. That holds when packing is
on. **With packing off, Dolly examples are short and padded out to 2048, so most
counted positions are padding** — the lane's 2892.5 and 3708.6 tok/s are
counting padding as throughput, and its loss is computed over a very different
mix of real tokens per sequence.

So this lane measures neither quality nor throughput in a way that can be set
beside the packed lanes. It is reported because it ran on both chips and the
numbers exist; it is flagged here so that nobody plots it. Fixing it would mean
counting non-padding tokens per batch, which the harness does not currently do.

---

## 26. What the Trainium1 symmetry pass changed (Phase 3)

Several lanes had run on Trainium2 only. Each was therefore a single-chip
observation that could not be attributed to the *generation* rather than to the
workload. trn1 is on credits and idle, so the counterparts were run. Three of
them changed what the report can claim.

### 26.1 A correction: Trainium1 is not "the compiler-limit chip"

§22 found that at seq 4096 the micro-batch ladder failed with `NCC_EXTP003`
(compiler instruction count) on trn1 and `NCC_EXSP001` (device HBM) on trn2, and
described the two chips as walling in different subsystems. The efficiency
sweeps at **seq 2048** show that framing was too broad:

| lane (trn1, seq 2048) | error | detail |
|---|---|---|
| `eff_mb2` | **`NCC_EXSP001`** | needs 94.39 GB vs 17.17 GB available |
| `eff_mb4` | **`NCC_EXSP001`** | needs 94.41 GB vs 17.17 GB available |
| `eff_norecompute` | `NCC_EOOM001` | peak 22.19 GB vs 16.00 GB limit |

Trainium1 hits the *device memory* error here — the same class as Trainium2.
The correct statement is that **each chip hits whichever ceiling the shape
reaches first**, and which ceiling that is depends on the configuration, not on
the generation. §22.2 now carries this correction inline.

Note also that trn1's requirement barely moves between micro-batch 2 and 4
(94.39 → 94.41 GB), the same batch-insensitivity seen in trn2's HBM figures and
in trn1's instruction counts. Whatever dominates these estimates is not the
activation memory that scales with batch. We still cannot explain it, and say so.

### 26.2 A fourth capability gap: disabling recompute requires Trainium2

| | result |
|---|---|
| trn1 `eff_norecompute` | ✖ `NCC_EOOM001`, 22.19 GB vs 16.00 GB |
| trn2 `eff_norecompute` | ✅ 5265.5 tok/s |

Turning gradient checkpointing off — the classic trade of memory for speed —
is simply unavailable on Trainium1 for this model. It is a lever that exists
only on the newer part.

This is the fourth recorded Trainium1 failure that becomes a Trainium2 pass,
after seq 8192 (§15), CLIP (§24.4) and full fine-tuning of Qwen3-1.7B (§25.2).
Taken together they are a more concrete argument for the newer chip than any
throughput ratio: four things that cannot be done at all on Trainium1.

### 26.3 The one lever that works, on both chips

| lane | trn1 | trn2 | ratio |
|---|---|---|---|
| `eff_seq4096` | 3572.3 tok/s, **91.4% MFU** | 7277.4 tok/s, 53.3% MFU | **2.04×** |

Sequence length is the only efficiency lever that both chips accept, and it is
the one that pays: a clean 2.04×, independently consistent with the 1.92× from
the context ladder (§15) and the 1.96× from the batch-ladder baseline (§22.1).
Three separate lanes, three measurements of the same quantity, spread 2.04 /
1.96 / 1.92 — a range of 6%, against a 2.4% noise floor, which is about as much
agreement as this methodology can demonstrate.

Note the MFU inversion once more: trn1 is at **91.4%** and trn2 at **53.3%**
while trn2 does twice the work per second. A reader optimising MFU would prefer
the slower chip.

### 26.4 What the symmetry pass confirmed rather than changed

- **MoE is rejected identically on both chips** — the `qwen3_moe` allowlist
  error is byte-identical on v2 and v3, so §24.5's limit is Neuron-wide rather
  than v3-specific. This is the clearest case of why the pass was worth running.
- **`data_alpaca` loss is identical to four decimals on both chips** (1.0088),
  another instance of the determinism in §20.
- **`data_dolly_nopack` reproduces its anomaly on both** (5.4633 / 5.4631),
  confirming §25.5's conclusion that the lane's accounting is broken rather
  than that one chip behaved oddly.

---

## 27. Can one Trainium2 host two training jobs? (Phase 3)

96 GiB of HBM and four logical cores make co-tenancy plausible on Trainium2 in
a way it never was on Trainium1's 32 GiB and two cores. This is a production
question rather than a benchmark one — it decides whether a team shares a chip
or buys two.

**The answer, on this stack, is no.**

### 27.1 What was measured

Two independent 2-core LoRA jobs (TinyLlama, 50 steps, identical), pinned to
disjoint halves of the chip with `NEURON_RT_VISIBLE_CORES=0,1` and `2,3`, each
with its own rendezvous port:

| lane | cores | run | tokens/s |
|---|---|---|---|
| `residency_solo_a` | 0,1 | alone | 5946.8 |
| `residency_solo_b` | 2,3 | alone | 6068.2 |
| `residency_pair_a` | 0,1 | concurrent | 5906.8 |
| `residency_pair_b` | 2,3 | concurrent | **failed to start** |

```
ERROR NRT:nrt_allocate_neuron_cores
  Logical Neuron Core(s) not available - Requested:lnc1-lnc1
  Available:0 Logical Core size:2 (cores busy, ret=-16)
```

The second job could not claim its cores while the first held the device.

### 27.2 Why this is a finding and not a misconfiguration

The obvious objection is that the core indices were wrong. They were not:
**`solo_a` on cores 0,1 and `solo_b` on cores 2,3 each ran successfully** with
exactly those indices. Both halves are individually addressable and both work.
What fails is claiming the second half while the first is in use.

The second objection is that this is the earlier port collision again. It is
not. That bug was real — both jobs defaulted to `torch.distributed` port 29500 —
and it was fixed; the logs confirm 29511 and 29512. The failure then moved from
`EADDRINUSE` to this Neuron-runtime error, which is a different and more
informative wall. **Fixing the first bug is what made the second one visible.**

### 27.3 What it does and does not license

**Supports:** two concurrent training jobs cannot be placed on one
trn2.3xlarge by partitioning with `NEURON_RT_VISIBLE_CORES` at the LNC=2
default. A team wanting two isolated training workloads needs two instances.

**Does not support** any claim about inference co-tenancy, which is a different
runtime path and is how multi-model serving is normally done on Neuron; about
other partitioning mechanisms we did not try; or about behaviour at LNC=1, where
the eight physical cores are addressed directly and the allocation arithmetic
differs. It also says nothing about *performance* under sharing, because sharing
never began — `pair_a` at 5906.8 tok/s against its 5946.8 solo baseline is a
0.7% difference measured while it had the chip to itself.

### 27.4 The earlier version of this lane produced a plausible, wrong answer

Worth recording because of how it failed. In the first run both jobs used port
29500, `pair_a` died instantly, and `pair_b` ran **alone**. Its throughput was
99.1% of its solo baseline, and the natural reading — "two jobs share one chip
with 1% interference" — is exactly the answer a reader hopes for. It was
measuring an idle chip.

Nothing in the result file marked it: `pair_b.json` was complete and
well-formed. It was caught only by asking why `pair_a.json` was **absent**.
That is the third lane in this study whose failure showed up as a missing file
rather than a recorded error, and it is the reason §26 now insists an audit diff
intended lanes against present artifacts rather than reading what happens to be
there.

---

## 28. Closing the loop: serving the Trainium2 fine-tune, and the context cliff

### 28.1 A Trainium2 fine-tune serves on Inferentia2, byte-verified

Phase 2 closed the train-then-serve loop for Trainium1: train, merge, push to
S3, pull onto Inferentia2, serve, with sha256 at both ends. Phase 3 trained the
same model on Trainium2 and, until now, **never served it** — so every serving
number in this study described trn1-trained weights.

The trn2 fine-tune now serves, and the provenance check passes:

```
all_match = True   (config.json, 4 safetensors shards, tokenizer.json)
```

sha256 of the weights on the *serving* box against the digests
`merge_adapter.py` recorded on the *training* box. Same grid as the trn1 lane:

| concurrency | trn1 fine-tune | trn2 fine-tune | ratio | TPOT trn1 | TPOT trn2 |
|---|---|---|---|---|---|
| 1 | 15.75 tok/s | 15.82 | 1.004 | 62.63 ms | 62.84 ms |
| 4 | 57.81 | 61.92 | 1.071 | 63.74 | 63.68 |
| 8 | 114.78 | 120.21 | 1.047 | 65.18 | 64.84 |
| 16 | 213.17 | 228.30 | 1.071 | 67.31 | 66.86 |
| 32 | 394.51 | 418.17 | 1.060 | 71.61 | 70.26 |

Mean ratio **1.051**, range 1.004–1.071.

**What this supports:** a fine-tune trained on Trainium2 deploys on Inferentia2
through the identical path, at the same serving performance. Which chip trained
the adapter is invisible at serving time — as it should be, since the merged
artefact is an ordinary set of safetensors. That is the useful, boring result: a
practitioner can train on whichever Trainium generation they can get and serve
on the same Inferentia fleet.

**What it does NOT support:** the 5.1% mean difference is above the 2.4% noise
floor and consistent in sign, but the two lanes ran hours apart on the same box
with different cold compiles (boot 548 s vs 602 s), and nothing in this study
isolates a mechanism by which the *weights* would change decode throughput. The
architecture, shapes, and dtype are identical. Treating 5.1% as a property of
trn2-trained weights would be over-reading a single paired run; it is recorded
and left unexplained.

### 28.2 The context cliff is between 8192 and 12288 — and it is the HOST

The context ladder ends where the *compiler's host memory* ends, not where HBM
does:

| seq | outcome | compiler peak host memory |
|---|---|---|
| 4096 | ✅ 7433.2 tok/s | compiles |
| 8192 | ✅ 8335.8 tok/s, 61.0% MFU | compiles — the study's best MFU |
| **10240** | ✖ | **104 GB**, killed by watchdog |
| **12288** | ✖ | **114 GB**, killed by watchdog |
| 16384 | ✖ | 48.6 GB RSS / 80.8 GB VM at the kernel's OOM-kill, twice |

The cliff therefore sits between **8192 and 10240**, and the failures are
ordered the way compilation cost should be: 104 GB at 10240, 114 GB at 12288.

**And there is nothing legal in between.** An attempt to bisect at 9216 failed
instantly with

```
NotImplementedError: Only support sequence as multiples of 2K
```

9216 = 4.5 x 2048, so it is not a valid sequence length on this stack at all —
the lane was invalid by construction and says nothing about memory (its receipt
records `kernel_oom_events: 0` and no watchdog trigger). But the constraint it
exposed settles the question: **valid sequence lengths are multiples of 2048**,
so 10240 is the very next legal step above 8192.

**The trainable maximum for Llama 3.1 8B LoRA on a trn2.3xlarge is therefore
exactly 8192**, and it is bounded not by the accelerator but by the compiler's
host memory at the next legal shape. That is a precise answer rather than an
interval, and it arrived only because the failing lane's actual error was read
instead of its exit status.

The 16384 figure is **not** comparable to the other two and must not be read as
"16384 needs less memory than 10240". Those two were killed by our watchdog at a
70 GB threshold, so their reported peak is where the watchdog caught them on the
way up. 16384 was killed by the *kernel*, which fires when the whole machine is
exhausted — a different trigger, sampled at a different moment. All three say
the same thing: the compiler's working set exceeds what a 124 GiB host can
give it.

Every failure above 8192 is `walrus_driver` — the Neuron compiler backend —
exhausting a **124 GiB host**. Compilation never completes, so **no HBM figure
is ever produced**: this study cannot say whether seq 12288 or 16384 would fit
in the chip's 96 GiB, only that they cannot be compiled on this instance.

63 GiB of swap was free and **entirely unused** at the 16384 kill. The kernel
chose to OOM-kill rather than swap the compiler's working set, so adding swap
did not help — worth knowing before anyone tries the same mitigation.

**Practical statement:** on a trn2.3xlarge, Llama 3.1 8B LoRA trains at up to
**seq 8192**, and the binding constraint above that is host RAM for compilation,
not accelerator memory. A larger build host, or ahead-of-time compilation on a
memory-rich machine, is the direction to try — not a bigger accelerator.

### 28.3 The watchdog that made 12288 reportable

The 16384 lane was OOM-killed three separate times, and every time the kernel
took the whole systemd unit with it — including the shell that would have
written the failure receipt. Each attempt left **no artifact at all**, and was
recoverable only by noticing an absence and reading `journalctl`.

The 12288 lane ran under a watchdog that polls `walrus_driver`'s RSS and kills
it deliberately at 70 GB. It fired at **114 GB** and the receipt was written
normally, carrying the peak memory figure — which is the number that makes the
result interpretable rather than merely negative.

The general lesson, now recorded in three places in this study: **a process that
can be killed by the kernel cannot be relied upon to record its own death.**
Either watch it from outside, or reconcile intended lanes against produced
artifacts afterwards. A missing file is silent in a way a failure receipt is not.

---

## 29. What this study did NOT accomplish

A final audit against the Phase-3 plan and against every lane attempted. The
sections above report what was measured; this one reports what was not, so the
gaps are on the record rather than left for a reader to discover.

### 29.1 Questions asked and not answered

| question | why it is unanswered |
|---|---|
| Does seq 12288 / 16384 **fit** in Trainium2's 96 GiB HBM? | Compilation never completed — the compiler exhausted a 124 GiB host first. No HBM figure was ever produced, so the accelerator's capacity at those shapes is untested (§28.2). |
| Why is trn1's compiler instruction count identical at micro-batch 2, 4 and 8? | Two explanations were tested and both falsified: cached failures (ruled out by distinct module hashes) and the design holding the product constant (ruled out by the corrected ladder). No third explanation was tested (§22.3). |
| Is the 5.1% serving difference between trn1- and trn2-trained weights real? | Above the 2.4% noise floor, consistent in sign, and with no mechanism by which identical-architecture weights would change decode throughput. One paired run, hours apart, different cold compiles. Not claimed (§28.1). |
| Can FP8 training work on NeuronCore-v3? | The one-flag autocast produces NaN from step 1. Whether a proper quantisation recipe, loss scaling or per-layer control fixes it was not attempted (§25.3). |
| Is the seq-8192 overhead device-side or framework-side? | The isolation lane ruled out host data preparation at three shapes, which is a bound, not a mechanism. Confirming it needs a `neuron-profile` trace, which no window had room for (§25.1). |

### 29.2 Lanes that were planned and did not run

- **Trainium2 seed variance** (`llama31_lora_seed43/44`, `qwen3_lora_seed43`) —
  deliberately deferred, then skipped by their own budget guards when under
  70 minutes remained. `--seed` is a no-op on this stack, so they would have
  re-measured timing noise that §20 already bounds at 2.4% using two *physical*
  chips. Their receipts say exactly this.
- **`ctx_32768`** — gated on 16384 passing, which it did not.
- **`eff_combined` on trn1** — never attempted; its trn2 counterpart failed on
  HBM, and §26.1 predicts the same outcome.
- **`cifar_vit` on trn2** — attempted twice, compiler OOM-killed both times.
  Recorded as a receipt (§25.4).
- **Multi-tenant residency on trn1** — physically impossible: the lane needs
  four logical cores and Trainium1 has two.
- **Three of five adversarial-review models** (qwen3.8-max, deepseek-v4-pro,
  minimax-m3) never returned answers. Two did, and their recommendations were
  implemented or explicitly rejected.

### 29.3 A methodological weakness this study could not resolve

`analysis/roofline.py` reports **arithmetic intensity as an upper bound only**,
so every lane classifies as "compute-bound (bound only)". That is not a finding
— it is the analysis being unable to discriminate. Weight traffic is amortised
analytically rather than measured, because Neuron exposes no HBM-traffic
counter. A real roofline needs hardware counters this platform does not provide,
the same gap that makes `power_w`/`temp_c` empty and perf-per-watt unmeasurable.

### 29.4 Artifact-integrity issues found in the final audit

Three classes, all repaired, all recorded because the failure mode generalises:

1. **Zombie artifacts.** `push_results.sh` uses `aws s3 sync` **without
   `--delete`**, so a file invalidated on the box stays live in S3 at its
   original path. Seven tags carried both a result and a failure receipt.
2. **A zero-byte "receipt."** `spec_decode/baseline.failure.json` was 0 bytes —
   an artifact that looks like a record and contains nothing.
3. **Archive contamination in analysis.** `roofline.py` globbed `**/*.json` with
   no exclusions, so it scored `invalidated/residency_pair_b.json` — the very
   artifact from an experiment that never ran (§27.4).

Stale copies were moved to `superseded/` rather than deleted, the empty file was
removed, and `roofline.py` now skips `invalidated/`, `deferred/`, `superseded/`
and `original_chip/`. Three tags legitimately retain both a failure and a
success — `whisper`, `tp_probe`, `fused_spec` — where a lane failed and a retry
succeeded; that history is real and is kept.
