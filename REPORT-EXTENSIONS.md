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
