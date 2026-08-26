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
longer contexts), not speed. int8 weights were **declared not-attempted** at
the time of this section (needs an offline checkpoint-quantization stage;
`int8_note.json`). That is superseded: int8 *was* attempted later, and the
quantised model shards, loads and warms up — it is the perplexity harness that
dies. See §36.4.

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
| int8 weights | GO | ⚠ prep-stage gap — the quantised model *loads and warms up*; the eval harness dies (§36.4) |
| gpt-oss-20b MoE | attempt-only | ❌ resolved — blocked by a MoE kernel *shape* constraint (§36.3), not memory |
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

**Status: MEASURED.** A 24-hour Capacity Block ran 2026-08-05/06 on a
trn2.3xlarge in sa-east-1 ($53.64, non-refundable). Every number below comes
from that run and is traceable to a stored result JSON.

This section was originally scaffolded with the declared design *before* any
result was seen — the grid, the denominators and the fairness argument were
fixed in advance so the numbers could only drop into a shape already committed
to. That scaffold text survived into the measured version by mistake and, until
this correction, said the lane had not run while §§18–28 cited its results. The
pre-registration was real; the "not taken" status was stale.

**Read §15 with three caveats that later sections establish and that this
section originally did not carry:**

- **MFU here is PROVISIONAL** — the FLOP model omits attention (§30.1).
- **The two chips ran at different tensor-parallel widths** (trn1 TP=2, trn2
  TP=4). This is a one-chip *system* comparison at each chip's working default,
  not an isolated silicon comparison (§30.2).
- **The $/token figure amortises a non-refundable block** as if fully utilised.
  The single-job bill was $53.64 (§30.3).

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
2,064,384 at micro-batch 2, 4 and 8. Two candidate explanations were tested and
both failed:

1. **A cached failure.** Ruled out. The re-run used a different compiler-flag
   hash (`+f7f529f3` → `+e30acd3a`) and each rung emitted its own distinct
   module hash. The compiles were fresh.
2. **The ladder holding the varied quantity constant.** The first design set
   `grad_accum = 8 / micro_batch` to hold global batch fixed, which — since
   gradient accumulation is unrolled into the compiled graph here — made every
   rung compile the same total work. Plausible, and ruled out: the corrected
   design fixes `grad_accum` at 8, so unrolled work varies 8× across the rungs,
   and the count is **still identical**.

**Resolved 2026-08-12: the number is not a measurement of the graph.**

An earlier version of this section said "we do not know why" and declined to
offer a third theory. The third theory is now tested, and it is that the premise
was wrong — nothing about these graphs needs explaining, because the count never
described them.

The compiler prints the shape of the operator it rejects. It is a
flash-attention backward kernel, and its tensor scales exactly as expected:

| rung | rejected kernel `dy_ref` | elements | instructions |
|---|---|---:|---:|
| mb2 | (2, 16, 128, 4096) | 16.8M | 2,064,384 |
| mb4 | (4, 16, 128, 4096) | 33.6M | 2,064,384 |
| mb8 | (8, 16, 128, 4096) | 67.1M | 2,064,384 |
| `eff_combined` (mb2, **recompute off**) | (2, 16, 128, 4096) | 16.8M | 2,064,384 |

A 4× range in the volume of the very operator that triggered the error, plus a
fourth configuration toggling a lever no rung varied, and the figure does not
move by one instruction. Across the whole study there are **59 occurrences in 10
lane logs over three independent runs and two compiler-flag hashes, and every
one reads 2,064,384**. No configuration anywhere produces a different value.

An instruction count cannot be invariant to a 4× change in the operator being
counted. 2,064,384 is a constant — note its shape, 63 × 2¹⁵, or 2016 × 1024 —
and it carries no information about the submitted graph.

**What this invalidates.** The ladder previously flagged itself `SUSPECT`
whenever failed rungs shared a count, on the reasoning that independent compiles
must differ. That check was unsound and fired on a sound ladder; it has been
replaced by a module-hash comparison, which is what "compiled independently"
actually means. Both ladders now record
`independent_compiles_verified: 3 failed rungs, 3 distinct module-hash sets`.
The `compiler_instructions` field is retained for provenance and explicitly
marked `compiler_instructions_is_diagnostic: false`.

**What remains unknown** is what 2,064,384 denotes — a capacity, a sentinel, an
artifact of the error path. Settling that needs neuronx-cc internals rather than
another lane, and no claim in this study depends on it.

What is measured is unchanged: at seq 4096 on Trainium1, micro-batch 1 compiles
and nothing above it does. The wall is real; only the number printed beside it
was uninformative.

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
| What does trn1's `NCC_EXTP003` instruction count of 2,064,384 denote? | **The question of why it is identical across rungs is closed** — the count is invariant across a 4× change in the rejected operator's tensor volume, across recompute on/off, and across all 59 occurrences in the study, so it does not describe the submitted graph (§22.3). What the constant itself denotes needs neuronx-cc internals, not another lane. No claim depends on it. |
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

---

## 30. Adversarial review: what an independent reviewer broke

Before publication this study was handed to an independent model (codex-cli
0.146.0) with instructions to attack it rather than praise it. Seven findings
came back. **Two were critical and both are confirmed.** Every claim below was
re-verified against the code and the stored JSON rather than accepted.

### 30.1 CRITICAL — MFU is provisional and must not be read as utilisation

`shared/train/sft_lora.py` defines `attention_flops_per_token()` — a
sequence-dependent term the parameter-count FLOP model cannot capture — and
**never calls it**. Every MFU in this study is parameter-only.

That is not a rounding detail. The attention term, computed with the study's own
formula, is:

| seq | attention as % of the parameter term |
|---|---|
| 2048 | 8.9% |
| 4096 | 17.7% |
| 8192 | 35.5% |

And applying it produces impossible numbers:

| lane | stored MFU | + attention |
|---|---|---|
| trn1 `eff_seq4096` (/190) | 91.38% | **107.53%** ← impossible |
| trn1 `mb1_seq4096` (/190) | 91.33% | **107.47%** ← impossible |
| trn2 `ctx_8192` (/667) | 61.00% | 82.47% |

**A utilisation above 100% is proof the model is wrong somewhere**, not proof of
a fast chip. Candidates: the attention formula overestimates; the 190 TFLOP/s
trn1 denominator is too low (the alternative 210 figure gives 97.3%, still
implausible); or — most likely — `tokens_per_s` derives from the *median timed
step*, which §25.1 showed covers only 40–89% of wall clock, so it is an
instantaneous steady-state rate being compared against a sustained peak.

**Consequences applied throughout:**

- **MFU in this study is a PROVISIONAL, parameter-only lower bound.** It is
  useful for comparing lanes computed the same way; it is not an absolute
  utilisation figure and should not be cited as one.
- §26.3's "91.4% MFU" is **not** "the study's best MFU" in any absolute sense.
- §21.3's occupancy argument leaned on the MFU ladder. That ladder is
  directionally informative and quantitatively unreliable — see §30.4.
- **`tokens_per_s` and wall clock are the trustworthy metrics.** They are
  measured, not modelled, and every headline ratio in §15 rests on them.

The fix is not to silently wire the attention term in — that would publish
107% utilisation. It needs a reconciled FLOP model and denominator, which this
study did not have time to establish.

### 30.2 CRITICAL — §15's status line was stale, and the comparison is TP-confounded

§15 opened with "measurements not taken… blocked on AWS capacity" while §§18–28
cited its results throughout. That was pre-registration scaffold text left in
place after the block ran. Corrected in §15.

The reviewer's substantive point stands: **trn1 ran TP=2 and trn2 ran TP=4.**
That changes collective topology, shard size, HBM per rank, compiler graph and
reduction order simultaneously with the silicon. §18 already conceded this for
the loss gap, but §§21 and 26 phrase 1.20×, 1.92×, 2.04× and 2.21× as
Trainium2 results.

**The defensible claim is a one-chip SYSTEM comparison at each chip's working
default — trn1 at TP=2 against trn2 at TP=4 — not an isolated silicon effect.**
A matched-width comparison is structurally impossible here: trn1 has two cores
and cannot reach TP=4. Running trn2 at TP=2 was possible and was not done; that
is the missing control, and it is now the single most valuable follow-up.

### 30.3 The cost headline needs both numbers, always

The arithmetic is correct — trn1 $0.2583/M, trn2 $0.1952/M, 24.4% lower — but
"occupied cost" amortises a **non-refundable** Capacity Block as though it were
fully utilised. Allocating the whole 24-hour block to this one job gives
**$5.0759/M**, and the actual bill was **$53.64** regardless of how much of the
block was used.

Both numbers are already computed by `cost_metrics()` and the study's rule was
never to blend them. The correction is that the 24.4% figure must appear
**beside** the block-allocated figure wherever it is quoted, and must never be
called "what the job cost".

### 30.4 Three claims downgraded from finding to hypothesis

- **§21.3 device occupancy.** "The chip is not being filled" is an inference
  from an unreliable MFU ladder plus a null host result. The null bounds host
  cost below the noise floor; it does not license a positive claim about device
  idling. **Downgraded to a hypothesis requiring `neuron-profile`.**
- **§25.1 end-to-end generality.** The end-to-end ratios (0.995× and 0.98×) are
  *inside* the 2.4% noise floor, so "finishes no sooner" is at the edge of what
  the data supports. The **in-window fraction** collapse (89% → 40%) is a large,
  real effect measured on both chips; the end-to-end consequence is consistent
  with it but rests on differences too small to resolve. Two shapes, unequal
  step counts, and no fixed real-token or fixed-update budget.
- **§28.1 serving equivalence.** "At the same serving performance" overstates a
  single unpaired run with different cold compiles. The defensible claim: *the
  Trainium2-trained artifact deploys through the identical Inferentia2 path and
  is byte-verified; this study does not establish a serving-performance
  difference between the two artifacts.*

### 30.5 What survived

The reviewer identified the byte-verified train-to-serve compatibility (§28.1)
as the strongest valid result, and did not dispute: the held-out quality gate
(§19), the bit-identical cross-chip determinism (§20), the MoE allowlist
rejection (§24.5), the host-memory compile cliff (§28.2), the residency runtime
limit (§27), or any of the recorded failure receipts. The arithmetic audit
confirmed `tokens_per_optimizer_step`, steady-state tokens/s, end-to-end rates
and the §20 replication figures all reproduce exactly.

**This section exists because a study that only publishes what survived review
is not reporting the review.**

---

## 31. Biases this study carries

A second independent model (kimi-k3) was asked what biases a study built this
way inevitably carries. Four came back. All four apply, three are only partly
mitigated, and none is fully solved. They are recorded here because a reader
cannot discount what a study does not disclose.

### 31.1 Survivorship bias in the reruns

**The charge:** failed lanes got extra rolls of the dice until they behaved, so
the published results are the lucky runs and the true failure rate is buried.

**It applies.** Lanes that were re-run after failing: the batch ladder (three
times — missing retry flag, misplaced validity check, broken design), the
quality gate, the dataloader isolation smoke, the residency lane, and
`ctx_16384` (three attempts). The primary lane itself ran twice, on two
different chips.

**Partial defence.** Most re-runs corrected a *defect in the harness* rather
than re-rolling a stochastic outcome — a missing compiler flag, a port
collision, a stale lock. Those are not extra dice; they are the same
measurement finally taken correctly, and each is documented with what changed.

**Where the charge lands.** `ctx_16384` genuinely was three attempts at the same
thing hoping for a different result, and the earlier attempts are recorded as
receipts. The honest statement is that this study reports **first-attempt
failure rates poorly**: the report shows which lanes eventually produced a
number, not how often a lane failed before it did.

### 31.2 Informative censoring — the missing lanes are missing *because* they were troublesome

**The charge:** lanes that never ran are absent precisely because they were slow
or difficult, which biases every average optimistic.

**It applies, and this is the sharpest one.** The lanes that never ran:
`cifar_vit` on trn2 (compiler OOM, twice), `ctx_16384` (never resolved),
`ctx_32768` (gated on 16384), `eff_combined` on trn1, three seed lanes (skipped
by budget guards), and the entire micro-batch ladder above 1 (both chips).

Every one of those is missing **because it was expensive or it failed** — the
textbook definition of not-missing-at-random. A reader who averages what is
present is averaging the workloads that were cheap enough to finish.

**Mitigation:** every absence is a receipt, not a silence. §29 lists all of them
explicitly. That does not remove the bias; it makes it visible.

### 31.3 The metrics and the report have the same author

**The charge:** one party choosing the yardstick *and* grading the result is
advocacy, not measurement.

**It applies.** The metric set — MFU, tokens/s, end-to-end throughput, $/token —
was chosen by the same process that wrote the conclusions.

**Mitigation, and it is real but partial.** Two independent models were given
the finished study with instructions to attack it. They produced §30 (two
critical findings, both confirmed: MFU is parameter-only and unusable as
absolute utilisation; §15's status line was stale) and this section. Several
claims were downgraded from finding to hypothesis as a direct result.

**Where it still lands.** The reviewers audited what was written; they could not
audit what was never measured. The choice to report cost-per-token — a metric
on which Trainium2 wins — while MFU, on which it loses, required an explanatory
paragraph, is exactly the kind of framing decision an adversarial reader should
weigh for themselves. Both numbers are published; the emphasis was still ours.

### 31.4 One window is weather, not climate

**The charge:** a single rented 24-hour window confounds every finding with that
day's instance noise, thermal state and neighbour load.

**It applies, with one genuine and unplanned mitigation.** An instance failure
forced the ASG to replace the box mid-study, so the primary lane ran on **two
different physical Trainium2 chips**. Timing differed by 2.41% and the final
loss was **bit-identical** (§20). That is real replication across hardware, and
it is why 2.4% is used as the resolution floor throughout.

**Where the charge still lands.** Two chips in one availability zone on one day
is not a distribution. Nothing here samples across regions, times of day, or
Neuron releases. The Trainium1 side has no such replication at all beyond three
bit-identical seeds. **Any single number in this report should be read as one
day's measurement on one pair of machines, ±2.4%.**

### 31.5 What would actually fix these

Not more analysis of the same data. Repeated windows on independent instances,
randomised lane order, a pre-registered metric set fixed before any hardware
lands, and a reviewer who sees the design *before* the results. This study
pre-registered §15's design and denominators, which is why that section could be
checked against its own scaffold — but it pre-registered only that section.

## 32. Pretraining and post-training on Trainium1 (Phase 4)

The question this phase asked was blunt: **can a Trainium1 chip do the training
stages that come before and after supervised fine-tuning — pretraining from
scratch, preference optimisation, and verifiable-reward RL — and if so, how
fast?**

One of the three now has a number. One is unresolved, with an earlier
"terminal" verdict retracted in 32.5.1. One is architecturally impossible on this path. The
pretraining lane is unresolved, and this section says so rather than dressing a
wall as a finding.

Every figure below is read from `trn1/results/phase4/*.json` by
`analysis/phase4_summary.py`. Nothing here is hand-transcribed.

### 32.1 What ran, and what it cost

| stage | verdict | evidence |
|---|---|---|
| SFT | **works** (Phase 1/2) | 2,952 tok/s, 68.3% MFU at seq 2048 |
| ORPO | **works** | 1,181 tok/s, 30.2% MFU at max_length 1024 |
| DPO | **unresolved** (was "terminal"; see 32.5.1) | the reference forward COMPILES out of the step; the lane dies later in a host transfer |
| GRPO / RLVR | **architecturally blocked** | no `generate()` on the training model class |
| pretraining from scratch | **unresolved** | per-step XLA recompilation in a hand-written loop |

### 32.2 ORPO works, and here is the number

ORPO was chosen over DPO as the primary preference lane for a reason that
turned out to be load-bearing: it is **reference-free by construction**. The
ORPO paper's whole claim is that the odds-ratio penalty removes the need for a
separate reference model, and on a 16 GiB-per-core device that stops being an
elegance argument and becomes a feasibility one.

| lane | max_length | tok/s (steady) | TFLOP/s | MFU % | MFU % alt | median step | compile s |
|---|---|---|---|---|---|---|---|
| `orpo_llama31_8b` | 512 | 1,012.5 | 49.21 | **25.90** | 23.43 | 8,091 ms | 67.1 |
| `orpo_llama31_8b_len1024` | 1024 | 1,181.0 | 57.40 | **30.21** | 27.33 | 13,873 ms | 568.8 |

Llama-3.1-8B base, LoRA r16/α32 on all seven projections, `ultrafeedback_binarized:train_prefs`,
micro-batch 1, grad-accum 8, TP=2, DP=1, gradient checkpointing on, 30 steps,
seed 42. Whole-model parameter counts (8,082,956,288 total / 52,428,800
trainable) obtained by `xm.all_reduce` across the TP group, the same way every
other LoRA lane in this study obtains them.

`tokens_per_optimizer_step` is `max_length × micro_batch × grad_accum × dp_size × 2`.
The ×2 counts chosen **and** rejected: a preference step forwards both
sequences. `dp_size`, not world size — tensor parallelism shares one micro-batch
across ranks and multiplies neither batch nor tokens. §32.6 records what
happened when that distinction was got wrong.

### 32.3 The number you should not quote against SFT

25.9% and 30.2% sit beside 68.3% for SFT, and the temptation is to say
preference optimisation costs 38 points of utilisation. **It does not, and the
comparison as stated is invalid**, for two reasons that pull in the same
direction.

**Sequence length.** This study has already measured a strong sequence-length
dependence in exactly this setting: the same SFT lane goes 68.3% at seq 2048 to
82.7% at seq 4096. The ORPO points are at 512 and 1024 — below the bottom of
that range. The ORPO ladder itself moves the right way, +4.3 points from 512 to
1024.

**Shape, not just length.** An ORPO step at max_length 1024 forwards **two**
1024-token sequences. An SFT step at seq 2048 forwards **one** 2048-token
sequence. Those are the same token count arranged differently, and attention
cost is quadratic in the length of a single sequence, so they are not the same
work even before the objective differs. The 6N convention this study uses does
not charge attention (§METHODOLOGY), so the denominators do not absorb the
difference either.

The honest reading is that **preference optimisation on this chip runs at
roughly half the utilisation of supervised fine-tuning at the sequence lengths
measured, and the gap has not been separated into "because the sequences are
shorter" and "because the objective is heavier."**

### 32.4 What the ORPO lanes do NOT show

Both 30-step lanes had their loss **rise**:

| lane | first loss | last loss | last − first | descended? |
|---|---|---|---|---|
| `orpo_llama31_8b` | 14.2031 | 14.6875 | +0.4844 | **NO** |
| `orpo_llama31_8b_len1024` | 13.0977 | 13.7109 | +0.6132 | **NO** |

Thirty steps at lr 5e-6 over 240 sequences is not a training run and was never
meant to be one; these lanes were sized to measure throughput. But a throughput
figure from a run whose loss did not descend measures arithmetic, not training,
and this report will not let the first table imply the second. **These are
hardware measurements. They are not evidence that ORPO improved the model.**

### 32.5 DPO fails, and ORPO is why that statement is worth something

> **Retracted in part.** This subsection concluded that the adapter-disabled
> reference forward *cannot compile*. It can. See 32.5.1. The controlled
> comparison against ORPO below is still valid and still isolates the reference
> forward as the blocker — what was wrong is the mechanism assigned to it.

DPO failed three times with the same compiler error:

```
RunNeuronCCImpl: error condition !(error != 400):
TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
```

raised while compiling the graph flushed by the first `.item()` in
`trl/trainer/dpo_trainer.py:1782`.

A stack trace alone would leave that a mystery. What makes it a result is that
**ORPO ran to completion through the same script, the same re-based
`NeuronTrainer`, the same chip, the same dataset, the same fixed-shape collator,
and the identical per-metric `.item()` pattern DPO dies in.** The two lanes
differ in exactly one structural way: DPO obtains its reference log-probs by
entering `peft.disable_adapter()` and running a second forward over the frozen
base. ORPO has no such pass.

That isolates the adapter-disabled reference forward as the blocker by
controlled comparison rather than by inference.

Two attributions were made along the way and both were **withdrawn by
measurement**, recorded in `dpo_smoke.failure.json`:

1. *`disable_adapter()` → `get_nb_trainable_parameters()` → `xm.mark_step()`
   forces the compile.* Pinned the parameter counts so the bookkeeping call
   cannot sync. **Rejected** — identical error recurred.
2. *TRL's eight per-step `gather_for_metrics` calls cut the graph where the two
   TP ranks disagree.* Evidence was real: the ranks were compiling different
   module hashes concurrently. Replacing the gather with the identity function
   (exact at dp=1) fixed the divergence. **Insufficient** — the bare `.item()`
   at the same line still flushes the step. An independent reviewer predicted
   this before the run.

An explicit `ref_model` cannot sidestep the blocker at this size: policy and
reference are ~8.03e9 parameters each, ~4.04e9 per core in bf16, before
optimizer state, activations, and the ~2× preference batch. The implicit
reference is the only configuration that fits, which is precisely why the
failure is terminal rather than a tuning problem.

**Not done, and recorded as not done:** DPO with an explicit `ref_model` at a
scale where two copies fit — TinyLlama-1.1B — would *demonstrate* the
attribution instead of isolating it. It was not run.

### 32.5.1 Retraction: the forward compiles, and placement was the blocker

32.5 called DPO terminal on the grounds that the adapter-disabled reference
forward cannot be compiled by neuronx-cc. **That is wrong**, and the correction
is worth more than the lane.

TRL ships `precompute_ref_log_probs`, which runs the reference over the dataset
**once, before training**, and stores the log-probs as columns. The objective is
unchanged — DPO is defined against a *frozen* reference, so recomputing
identical numbers every step was only ever an implementation convenience. With
the flag on, the training step contains no reference forward and the compiler
sees the shape ORPO already proved compiles.

On the fourth attempt neuronx-cc emitted, on the adapter-disabled forward
itself:

```
Compiler status PASS
Compilation Successfully Completed for model.MODULE_13739246366155782745
```

So the blocker was the forward's **placement inside the training-step graph**,
not the forward. Three device-placement defects had to be cleared to get there,
all from one ordering mismatch — TRL hangs precompute off
`get_train_dataloader()`, which optimum-neuron calls at
`trainers/transformers.py:1103`, *before* `setup_training()` places the model:

1. the precompute batch stays on the host → `Expected XLA tensor. Got: CPUBFloat16Type`
2. the *model* is still on the host → `Expected XLA tensor. Got: torch.FloatTensor`
   (the embedding **weight**, not the input)
3. a guard of ours was wrong: `neutralise_out_of_graph_gathers` bailed out
   whenever precompute was set, reasoning that `add_column` needs the global
   length. That only holds under data parallelism. At dp=1 every rank iterates
   the whole dataset, so gathering hands `add_column` a 2×-too-long column. dp,
   not the flag, is the correct guard.

**Still unresolved.** After the compile passes, the lane dies at
`dpo_trainer.py:844`, `ref_chosen_logps.append(ref_chosen_logp.cpu())`, with
`RuntimeError: BufferMapAdd: error condition !(((buffer) != nullptr))` — a
host-transfer failure on a lazy tensor, reproduced twice including once with an
explicit `mark_step()` flush and a host-side handoff. DPO still has no
throughput number on this stack.

**Why this is recorded rather than quietly fixed.** The original verdict
predicted the right *outcome* — DPO does not run — from the wrong *mechanism*.
That is the hardest class of error to catch, because nothing downstream
contradicts it. It survived three attempts and an independent review, and was
only exposed by trying the one configuration that would have been pointless if
the original claim were true.

### 32.6 An accounting bug that cancelled itself

The preference lanes originally computed

```python
tokens_per_step = max_length * micro_batch * grad_accum * nproc * 2   # WRONG
```

Under TP=2 the data-parallel size is 1, so multiplying by the world size
reported **exactly twice** the real token rate. The same code summed this rank's
parameter shard (4.04e9 of an 8.03e9 model), halving FLOPs/token.

The two errors cancel inside MFU. Reported MFU would have been right; reported
tok/s would have been double. **Two wrongs cancelling is worse than one wrong,
because nothing looks broken.**

Both were caught before any preference number was published — independently by
the author and by an external reviewer, in the same hour. The fix was not to
correct the arithmetic but to delete it: `phase4_lib` now re-exports
`sft_lora.tokens_per_optimizer_step` and `sft_lora.count_parameters` **by
identity**, so the preference lanes and the published SFT lanes cannot diverge.
`tests/test_phase4.py` asserts the identity and pins the bug.

The proven lane already carried the correct rule, in a docstring:

> TP shards ONE model across the cores, so both NeuronCores are working on the
> same micro-batch — tensor parallelism multiplies neither the batch nor the
> token count.

The bug was written by ignoring a helper that existed to prevent it.

### 32.7 GRPO / RLVR: architecturally blocked

`grpo_probe.py` diagnoses in four independent stages so the wall's location is
recorded rather than guessed:

| stage | result |
|---|---|
| A_construct | **OK** — `_NeuronGRPOTrainer -> NeuronTrainer -> object` builds |
| C_reward | **OK** — 64/64 GSM8K gold answers parsed; verifier returns [0.0, 1.0] on (wrong, right) |
| B_generate | **FAILS** — `NeuronModelForCausalLM.generate present=False` |
| D_train | not reached |

The trainer assembles and the verifiable reward works. What does not exist is
sampling: **optimum-neuron's training model class exposes no `generate()`**, and
online RL requires rollouts inside the training loop.

This is the most durable finding in the phase because it is an API fact, not a
memory or compiler fact. It does not change with model size, sequence length, or
a newer chip. Any online-RL method — GRPO, PPO, RLOO, RLVR — is out of reach on
this training path until that method exists. Offline preference optimisation
(§32.2) is the ceiling today.

### 32.8 Pretraining: three measured ceilings and one unresolved defect

A 362M SmolLM2-shaped Llama, trained from random initialisation on 1.1B
FineWeb-Edu tokens via a **hand-written** XLA loop with DP=2.

Three ceilings were located by walking the grid, not asserted:

| configuration | limit hit |
|---|---|
| micro-batch 8 | `NCC_EVRF007` — 37,536,776 instructions vs the 5,000,000 limit |
| micro-batch 1, grad-accum 4 | `NCC_EXTP004` — 5,919,820 instructions |
| seq 2048 | `NCC_EOOM001` — 16.20 GB peak vs the 16.00 GB per-core limit |

Two ceilings that pull against each other: gradient checkpointing relieves HBM
and worsens the instruction count. Only sequence length relieves both. The
compiler's suggested `--optlevel=1` was **declined**, because a lane compiled at
a different optimisation level is not comparable to every other lane in this
study.

The unresolved defect is that the loop **recompiles the XLA graph every step**
— 3 distinct module hashes in 3 steps. Two attributions were made and both
falsified on hardware:

| hypothesis | test | result |
|---|---|---|
| the linear-warmup/cosine LR schedule writes a new Python float into `param_groups` each step | constant-LR control | **falsified** — still 3 graphs in 2 steps |
| `AdamW`'s non-capturable bias correction `.item()`s the step counter | `capturable=True` | **falsified** — still 3 graphs, and compiles got 116× slower (step 2: 28,153 ms → 3,259,387 ms) |

An independent reviewer read the loop line by line and found **no third
candidate**, adding: *dump and compare the HLOs, especially their parameter
lists and literals; do not invent a third Python-scalar theory.* That is the
correct instrument — `PT_XLA_DEBUG_LEVEL=2` names the frame that severs the
graph — and it was **not run**, because by then each cold compile cost up to 54
minutes of paid instance time.

**What is claimed:** a hand-written XLA training loop on this stack recompiles
per step, reproducibly, and the cause is not established.

**What is NOT claimed:** that pretraining from scratch is impossible on
Trainium1. Every other training lane in this study — including the 68.3% MFU SFT
lane — runs through optimum-neuron's `NeuronTrainer` and shows no such
behaviour. **Pretraining through the framework path was never tested.** The
defect demonstrated here is in hand-rolled lazy-tensor code, not in the
hardware, and the practitioner's lesson is to use the framework's trainer rather
than to avoid the chip.

### 32.9 Making TRL's trainers run on NeuronTrainer

optimum-neuron ships exactly one alignment trainer, and builds it by stealing
TRL's methods and reparenting them:

```python
type("_SFTTrainer", (NeuronTrainer,), SFTTrainer.__dict__.copy())
```

`posttrain_align.py` generalises that to DPO and ORPO. Eight distinct blockers
had to be cleared, each found by running rather than reading, and all eight are
carried by the working ORPO lane:

| blocker | fix |
|---|---|
| `TypeError: super(type, obj)` — `__dict__` copying keeps a stale `__class__` closure cell | `_clone_rebound()` rebinds the cell |
| `NeuronTrainer` is not a `transformers.Trainer` subclass (its base is `object`) and omits attributes TRL reads | `HFTrainerCompat` mixin, gap found by AST-diffing TRL's reads against NeuronTrainer's provides |
| `NeuronTrainer.log(logs)` vs `Trainer.log(logs, start_time)` — TRL forwards both positionally | signature-inspecting `log` shim |
| `ValueError: Unexpected keyword arguments: use_cache, output_hidden_states` | `strip_unsupported_forward_kwargs`, with the Liger-loss path asserted off first |
| accelerate state cleared between startup and trainer construction | `ensure_accelerate_state()` at both points |
| TRL's preference collators pad per batch; Neuron needs fixed shapes | `FixedShapeCollator` |
| base models have no chat template | borrow from the matching Instruct model, recorded in the result |
| 8 per-step out-of-graph host syncs, ranks diverging | `neutralise_out_of_graph_gathers`, exact at dp=1, guarded against dp>1 and `precompute_ref_log_probs` |

The `log` shim is worth singling out. It cost a full lane-run to find because it
fires **inside optimum-neuron's own logging step-closure — after the model has
compiled and steps have run.** A two-argument signature mismatch discovered at
the most expensive possible moment.

### 32.10 Three more measurements, one of which invalidates a training claim

Three lanes ran after §32.2 was written. Each changes something.

**The ORPO throughput figure reproduces to 0.13%.** The same configuration
measured twice, at 30 steps and at 150:

| lane | steps | tok/s | MFU % | median step |
|---|---|---|---|---|
| `orpo_llama31_8b` | 30 | 1,012.5 | 25.90 | 8,091.0 ms |
| `orpo_llama31_8b_long` | 150 | 1,011.2 | 25.87 | 8,101.2 ms |

**0.13% apart across a 5× longer run.** That is the same order of agreement as
the Qwen3 SFT pair (57.2% / 57.3%) and it is well inside the 2.4% resolution
floor this study adopted in §31. The ORPO throughput number is stable.

**And the 150-step run diverged to NaN at step 23.** It was non-finite for 128
of its 150 steps, and it still reported a completely ordinary-looking 1,011.2
tok/s — because **NaN costs exactly the same FLOPs as a number.** A throughput
harness cannot tell the difference, and this one did not, until the loss trace
was read.

That is worth stating plainly: the reproducibility result above is real, and it
was produced by a run that was numerically broken for 85% of its length. Both
facts are true simultaneously. Throughput remained a valid measurement of the
hardware; it stopped being a measurement of training at step 23.

Two consequences, both now enforced in code rather than in prose:

- `phase4_lib.StepLog.metrics()` emits `loss_numerically_valid`,
  `loss_first_nonfinite_step`, `loss_nonfinite_steps` and an explicit
  `loss_validity_note` on every future lane.
- `analysis/phase4_summary.py` prints a **Numerical divergence** table and
  computes its loss columns over finite steps only.

The two published ORPO figures (25.9% at 512, 30.2% at 1024) come from 30-step
runs with **zero** non-finite losses, verified across all lanes. They stand.

**What caused the NaN is not established.** The one variable that distinguishes
the diverged run from the two clean ones is the data draw — 1,400 preference
pairs versus 320, same seed, same shapes, same learning rate. A plausible
mechanism is that a longer draw eventually contains an example whose completion
is emptied by truncation to `max_prompt_length`, making ORPO's NLL term a mean
over zero label tokens. **That is a hypothesis and it was not tested.** After
two falsified attributions in the pretraining lane (§32.8), this section records
the observation and the distinguishing variable and stops there.

**ORPO at max_length 2048 does not fit.**

```
RuntimeError: AllocBuffer: error condition NRT_RESOURCE == rt_status:
Not enough Neuron memory on core 0 for size=262540032
```

with HBM at 15.596 GB (core 0) and 15.614 GB (core 1) against the 16 GB
per-core budget when a further 250 MiB was refused.

This is the cleanest practical finding in the phase. **The SFT lane runs this
same 8B model at seq 2048 on this same chip at 68.3% MFU.** ORPO at 2048 does
not, because a preference step forwards chosen *and* rejected — a 2048-token
preference step has the activation footprint of a 4096-token supervised step.
**Preference training runs out of memory one rung of the sequence ladder earlier
than supervised fine-tuning on identical hardware.**

Note this is a **runtime** refusal by the Neuron runtime, distinct from the
pretraining lane's **compile-time** `NCC_EOOM001`. Both are the same 16 GiB
per-core ceiling, reported by different layers, with different error text.

The completed ORPO ladder:

| max_length | result |
|---|---|
| 512 | 1,012.5 tok/s, 25.90% MFU |
| 1024 | 1,181.0 tok/s, 30.21% MFU |
| 2048 | does not fit |

Which is also why §32.3's question — how much of the ORPO-versus-SFT gap is
sequence length and how much is the objective — **cannot be answered on this
hardware.** The sequence length at which the comparison would be fair is the
sequence length at which the preference lane runs out of memory.

### 32.11 Pretraining does work — through the framework, on half a chip

§32.8 left the pretraining lane unresolved and made a narrow non-claim: the
recompile was demonstrated for a **hand-written** XLA loop, and the framework
path had never been tested. This section tests it.

Same architecture, corpus, sequence length, micro-batch, grad-accum, LR
schedule and seed. `optimum-neuron`'s `NeuronTrainer` owns the loop instead of
this repo.

**It trains.**

| lane | micro-batch | tok/s | MFU % (chip) | MFU % (core used) | median step | compile s | loss |
|---|---|---|---|---|---|---|---|
| `pretrain_nt_362m` | 1 | 2,750.5 | 4.19 | 8.38 | 744.6 ms | 539.2 | 11.016 → 7.755 |
| `pretrain_nt_362m_mb8` | 8 | 4,573.2 | 6.97 | 13.93 | 3,582.6 ms | 913.1 | 11.049 → 7.736 |

362M parameters (SmolLM2-360M's exact published shape), random initialisation,
FineWeb-Edu, seq 1024, constant LR 3e-4, seed 42, ONE NeuronCore. Parameter
counts by `xm.all_reduce`. All losses finite.

**The loss descends.** 11.02 → 7.76 from random init. This is the only Phase-4
lane where it did — the ORPO lanes rose over 30 steps and the 150-step one went
NaN (§32.10). It is still only 122,880 tokens: a throughput probe, not a model.

#### The acceptance criterion was fixed before the result existed

Two independent reviewers were asked what would count as proof that the
framework path does **not** have the hand loop's defect, *before* the run
finished, precisely so the log could not be read generously afterwards. They
converged on the same gate, and one of them explicitly rejected the obvious
weaker test: "step 2 was fast" is not the criterion, because a per-step
recompile can hide behind one quick step.

| criterion | required | `mb=1` | `mb=8` |
|---|---|---|---|
| new graph hashes, steps 4–10 | zero | **zero** | **zero** |
| compile events, steps 4–10 | zero | **zero** | **zero** |
| median step stability | ±20% | **1.7%** | **0.1%** |
| losses finite | yes | yes | yes |

Every compile in both lanes happens in steps 0–1. The remaining hashes appear
after the final step, at teardown. Steps 3–59 (`mb=1`) and 3–40 (`mb=8`) run on
a single steady graph.

Against the hand loop's **3 distinct graphs in 3 steps** and its 54-minute
compiles, that is not a marginal difference.

#### What this does NOT establish

The hand-written lane ran **DP=2 across both NeuronCores**; this lane is forced
onto **one**. Loop ownership *and* core count changed, so this result shows the
framework path trains — it does **not** prove the hand loop caused its own
recompile. An independent reviewer flagged the confound after the run was
launched and before its result was read; closing it would need the hand loop
re-run at `nproc=1`, which was not done. Recorded in the result JSON as
`confound_vs_hand_written_loop` rather than left to a careful reader.

Separately, and worth keeping straight: the hand loop really did run DP=2 under
plain `torchrun` + `torch_xla`. The missing data-parallel dimension below is a
property of `optimum-neuron` 0.4.3's training arguments, **not** of Trainium and
not of `torch_xla`.

#### Why one core, and why that is the finding

Reaching both NeuronCores turned out to be impossible for this architecture,
via two constraints that compose:

**No data parallelism.** `NeuronTrainingArguments` builds its world as
`tp × pp`. With `tensor_parallel_size=1` it logs

```
> initializing data parallel with size 1
> initializing world size to 1
```

while `torchrun` has started two ranks. Rank 1 is then absent from the
collective bootstrap and the Neuron runtime aborts it. Tensor parallelism is the
only route to the second core.

**And TP is closed by the head count.** TP shards attention heads, so the head
count must divide the TP degree:

```
AssertionError: 15 is not divisible by 2
```

SmolLM2-360M has 15 attention heads and 5 KV heads. Both odd.

Together: **a real published 360M architecture can use only one of this chip's
two NeuronCores.** A model's head geometry decides which silicon topologies are
open to it, and an odd head count closes every multi-core one on this path.

The alternative was to reshape the model — 15 heads → 16 — until it sharded.
That was declined. The lane exists to measure a published configuration
(§32.8); reshaping the model to flatter the hardware would have destroyed the
thing being measured. So it runs on one core and is labelled a half-chip
configuration wherever it appears, with `mfu_pct` on the whole-chip denominator
for comparability and `mfu_pct_per_core_used` beside it — never substituted.

#### Reading 4.19%, honestly

That number must not be quoted as "pretraining on Trainium runs at 4% MFU". It
confounds two causes, and the batch ladder separates them:

- **Micro-batch.** 1 → 8 buys **+66% throughput** (2,750 → 4,573 tok/s) and
  takes per-core MFU from 8.38% to 13.93%. Real, and roughly what a bigger
  matmul should buy.
- **Model size.** Even at `mb=8`, per-core MFU is 13.9% against the 8B SFT
  lane's 68.3%. A 362M model at seq 1024 simply does not give the tensor engine
  enough work per operation. **This is the dominant term, and it is a property
  of the model, not of the chip.**

The practitioner's reading: Trainium1 is poorly matched to small-model
pretraining not because it is slow but because a 362M model cannot fill it. The
same chip reaches 68.3% MFU on an 8B fine-tune.

#### One more optimum-neuron 0.4.3 defect, found on the way

A plain `NeuronTrainer` over a plain causal-LM model cannot complete a single
step:

```
ValueError: Unexpected keyword arguments: reduction
```

`trainers/transformers.py:201` decides a model "accepts loss kwargs" by finding
`**kwargs` in its forward signature. optimum-neuron's own Llama forward *has*
`**kwargs`, so this is true. `compute_loss` (line 978) then does
`inputs = dict(**inputs, reduction="sum")` — and that same forward validates its
kwargs strictly and rejects `reduction`. The heuristic and the validation
disagree with each other inside one library.

Fix: `trainer.model_accepts_loss_kwargs = False`. Exact rather than lossy here —
with it on, loss is `sum / num_items_in_batch`; with it off,
`mean / grad_accum`. Those differ only when micro-batches hold different numbers
of loss-bearing tokens, and every window in this lane is exactly `seq_len`
tokens with no padding and no `-100` labels.

The SFT and preference lanes never hit this because TRL's trainers override
`compute_loss` entirely.

#### §32.8's non-claim, now resolved

> *"Nothing here supports the claim 'pretraining from scratch does not work on
> Trainium1'; what is supported is 'a hand-rolled lazy-tensor training loop
> recompiles per step on this stack.'"*

That non-claim was correct and is now settled in the direction it hedged toward.
Pretraining from scratch **does** work on Trainium1 — through the framework's
trainer, on one core for this architecture, at a low MFU that the model's size
explains.

## 33. The cost math against GPUs

The talk this study backs promises "the cost math versus GPU instances." This
section is that math, and it is the least flattering section in the report.

### 33.1 Where the GPU numbers come from, and why that limits the claim

No GPU was run for this study. The comparison numbers come from this repo's
sibling, [MI300X-vs-H200](https://github.com/alpharomercoma/MI300X-vs-H200),
which measured the same model, the same request shape and the same metric
schema on single accelerators — that shared lineage is the only reason a
comparison is possible at all.

Four differences are load-bearing and none of them can be removed by arithmetic:

| | this study | the GPU study |
|---|---|---|
| vLLM | 0.16 (AMI-pinned) | **0.26** |
| PyTorch | 2.9.1 | 2.11.0 |
| cloud | AWS | DigitalOcean (MI300X), Nebius (H200) |
| device HBM | 32 GB | 192 GB / 141 GB |

Ten minor versions of vLLM sit between the two serving stacks, and the Neuron
pin is not a choice — the newer DLAMI cannot boot on NeuronCore-v2. So a slower
Neuron result is partly a measurement of an older serving stack, and this
section cannot separate the two.

### 33.2 Serving, at matched concurrency

Llama 3.1 8B Instruct, BF16, 1024 in / 1024 out, one accelerator each.
Concurrency 32 is where the comparison is honest: it is the top of the
Inferentia grid, bounded by KV memory (rule 6), and a rung the GPU sweep also
measured.

| device | $/hr | out tok/s @ c32 | $ per 1M output tokens |
|---|---|---|---|
| inf2.xlarge, 1× Inferentia2 | 0.7582 | 415.5 | **0.507** |
| MI300X, 1 GPU | 2.59 | 3,443 | **0.209** |
| H200, 1 GPU | 4.50 | 4,486 | **0.279** |

**Inferentia2 costs 2.4× more per output token than an MI300X and 1.8× more
than an H200 at matched concurrency.** Prices are on-demand list, us-west-2 for
AWS (pricing API), and the providers' published GPU-hour rates for the other
two.

At each side's *best* operating point the gap widens, because the GPUs keep
scaling past the point Inferentia's KV budget stops at:

| device | best measured | at concurrency | $ per 1M output tokens |
|---|---|---|---|
| inf2.xlarge | 415.5 | 32 | **0.507** |
| MI300X | 8,085 | 256 | **0.089** |
| H200 | 11,337 | 256 | **0.110** |

**5.7× and 4.6× respectively.** A single Inferentia2 is not a cost-competitive
way to serve an 8B model against a single current-generation GPU.

### 33.3 The TTFT comparison that would have been wrong

The obvious next table — TTFT at concurrency 32, 6,414 ms on Inferentia2
against 7 ms on an H200 — is not a prefill comparison and must not be presented
as one. At concurrency 32 the Inferentia figure is dominated by queueing:
`MAX_NUM_SEQS=32` bounds resident sequences, so requests wait.

§17.1 isolates prefill properly, at concurrency 1: a 1024-token prompt reaches
first token in **397.5 ms p50**, and prefill throughput rises from 2,204 to
4,244 tokens/s across the input sweep. That is the number to compare, and this
study does not have the GPU-side equivalent at concurrency 1 isolated the same
way. So the honest statement is: **Inferentia2 prefill is roughly two orders of
magnitude slower than an H200 on a 1024-token prompt, and the 1000× figure a
naive reading of the sweeps would produce is queueing, not silicon.**

### 33.4 Training is NOT compared, deliberately

The two studies did not train the same thing. This one runs **LoRA r16** on
Llama 3.1 8B — 26M of 8B parameters updated. The GPU study runs **full BF16
training** — all 8B updated. Tokens per second across those two objectives are
different quantities, and dividing one by the other produces a number with no
meaning.

For the record, both sides, clearly labelled as non-comparable:

| lane | tok/s @ 4096 | what is updated |
|---|---|---|
| trn1, LoRA r16 | 3,575 | 26M params |
| MI300X, full BF16 | 5,603.3 | 8B params |
| H200, full BF16 | 7,821.6 | 8B params |

Training cost where this study *can* speak: Trainium1 delivers a LoRA
fine-tune at **$0.104 per 1M training tokens** at sequence 4096 ($0.126 at
2048). The GPU study published no prices, so there is no counterpart figure and
none is invented here.

The training economics and the serving economics point in opposite directions,
and that is the finding: **Trainium is the strong half of this platform, and
Inferentia at this instance size is the weak half.**

### 33.5 The gap this section cannot close

The comparison a listener actually wants is against **AWS** GPU instances, and
this study has none. The relevant single-device rentals in us-west-2 are:

| instance | GPU | HBM | $/hr | throughput |
|---|---|---|---|---|
| g5.2xlarge | 1× A10G | 24 GB | 1.212 | **not measured** |
| g6e.xlarge | 1× L40S | 48 GB | 1.861 | **not measured** |

Both are priced in the same band as trn1.2xlarge and inf2.xlarge, which makes
them the honest comparison and makes their absence the biggest open hole in
this report. Running the existing serving harness against a g6e.xlarge would
close it for about two dollars.

## 34. A decision framework: when Neuron silicon fits

Assembled from the measurements above rather than from vendor material. Five
questions, in the order that kills a bad fit fastest.

### 34.1 Is your architecture on the supported list?

Check this before anything else, because it is a hard gate and it is cheap to
check. `optimum-neuron`'s exporter supports a fixed, enumerated set of
architectures. SigLIP is not on it, and the lane failed identically on all
three boxes with `siglip is not supported yet for transformers`. CLIP needed a
trace fallback and produced NaN probabilities on Trainium1 while working on
Trainium2.

A supported architecture is a good day on this platform. An unsupported one is
not a tuning problem, it is a wall.

### 34.2 Is the objective offline or online?

| objective | status |
|---|---|
| SFT, LoRA or full | works |
| ORPO and other reference-free preference methods | works |
| DPO | reference forward compiles only outside the training step (§32) |
| GRPO, PPO, RLOO, RLVR | **architecturally blocked** |

Online RL needs to sample completions inside the training loop. The training
model class exposes no `generate()` at all; generation lives in a separate
ahead-of-time-compiled inference class. This is the single hardest boundary in
the study and no amount of budget moves it.

### 34.3 Are your shapes static?

Every distinct tensor shape is a separate ahead-of-time compilation. The study
paid 1,498 s for a cold 8B training compile and 1,832 s for a cold serving
boot. Warm, those collapse. The corollary is a design rule: pad to fixed
shapes, or pay a compile per shape. Phase 4 needed a `FixedShapeCollator`
precisely because TRL's preference collators pad per batch.

If your workload has genuinely dynamic shapes, the compile cost is not a
one-time tax, it is a per-request tax.

### 34.4 Training or serving?

This is where the answer diverges sharply, and it is the framework's most
useful axis.

**Training — yes.** 82.7% MFU on a Trainium1 at sequence 4096, 61.0% on a
Trainium2 at 8192, and $0.104 per 1M training tokens. These are strong numbers
by any standard, and the platform's ahead-of-time model suits training, where
shapes are fixed and the compile amortises over hours.

**Serving — only if cost is not the deciding factor.** §33 measures 1.8–4.6×
worse cost per output token than a single GPU. Choose Inferentia for serving
when something else dominates: capacity availability when GPUs cannot be got,
data residency, an existing Neuron training pipeline you want to keep on one
stack, or a throughput SLO you can meet inside the KV-bounded concurrency
ceiling.

### 34.5 What is your latency SLO, and at what concurrency?

KV cache costs 128 KiB per token, which bounds resident sequences to roughly 48
at 2048 context and 12 at 9216 on a 32 GB device. Past that the client measures
queue depth. Prefill is ~397 ms p50 for a 1024-token prompt at concurrency 1.

If your SLO is a low TTFT under load, this instance size will not meet it. If
your SLO is aggregate throughput at bounded concurrency, it will.

### 34.6 The short version

| your situation | verdict |
|---|---|
| LoRA or full SFT, supported architecture, static shapes | **strong fit** — the study's best numbers live here |
| Pretraining a small model from scratch | works up to ~400M params on one small instance; optimiser state, not compute, is the ceiling |
| Preference optimisation, reference-free | works |
| Online RL of any kind | **blocked** — do not plan around it |
| Cost-optimised 8B serving | **do not** — a single GPU is 1.8–4.6× cheaper per token |
| Serving where capacity or residency dominates cost | reasonable, inside the concurrency ceiling |
| Unsupported or fast-moving architecture | **do not** — the exporter list is the gate |
| Team without slack for toolchain debugging | budget for it: most of this study's walls were toolchain, not silicon |

## 35. Accuracy as a validity check (Phase 5)

Every inference lane up to this point reported **speed**. Speed cannot tell a
working compile from a broken one. This section is the control that proves it:
the same graphs, scored for correctness against a same-box float32 CPU
reference, using MLPerf's ≥99%-of-reference rule.

The reference is deliberately **the CPU run on the same box**, not anyone's
published number. A published figure differs for dataset, preprocessing and
prompt-template reasons that have nothing to do with the accelerator; only a
paired run isolates the silicon. Regenerate with:

```bash
python3 analysis/accuracy_summary.py trn1/results/accuracy inf2/results/accuracy --markdown
```

### 35.1 Paired verdicts

| box | comparison | delta | 95% CI | disagreements | MLPerf gate | rel. error |
|---|---|---|---|---|---|---|
| trn1 | ASR WER | −0.020 pp | [−0.052, +0.000] | 7/500 differ | **PASS** | −0.4% |
| trn1 | CLIP zero-shot | +0.00 pp | [+0.000, +0.000] | 0/10000 | **PASS** | — |
| trn1 | SigLIP zero-shot | +0.00 pp | [+0.000, +0.000] | 0/10000 | **PASS** | — |
| inf2 | ASR WER | −0.031 pp | [−0.070, +0.000] | 7/500 differ | **PASS** | −0.6% |
| inf2 | CLIP zero-shot | +0.00 pp | [+0.000, +0.000] | 0/10000 | **PASS** | — |
| inf2 | SigLIP zero-shot | +0.00 pp | [+0.000, +0.000] | 0/10000 | **PASS** | — |

Both zero-shot lanes are **bit-exact against CPU across 10,000 ImageNet
validation images** on both boxes — zero label disagreements, zero near-ties.
ASR moves by hundredths of a point and in the *favourable* direction on both
boxes, which is bf16 rounding, not a regression.

### 35.2 The measured numbers

| box | lane | top-1 / WER | top-5 / word acc | dtype |
|---|---|---|---|---|
| trn1 | CLIP CPU | 63.77% | 88.64% | float32 |
| trn1 | CLIP Neuron | 63.77% | 88.64% | float32 |
| trn1 | CLIP CPU bf16 | 63.59% | 88.60% | bfloat16 |
| trn1 | CLIP Neuron bf16 | 63.64% | 88.58% | bf16 matmuls |
| trn1 | SigLIP CPU / Neuron | 76.12% | 94.48% | float32 |
| inf2 | CLIP CPU / Neuron | 63.77% | 88.64% | float32 |
| inf2 | SigLIP CPU / Neuron | 76.12% | 94.48% | float32 |
| trn1 | Whisper CPU | 5.217% WER | 94.783% | float32 |
| trn1 | Whisper Neuron | 5.196% WER | 94.804% | bf16 (auto_cast=all) |
| inf2 | Whisper Neuron | 5.186% WER | 94.814% | bf16 (auto_cast=all) |

ASR broken out by split (250 utterances each): dev-clean 3.436% CPU vs 3.417%
Neuron; dev-other 7.357% CPU vs 7.335% (trn1) / 7.312% (inf2). The pooled 5.2%
is the two splits combined, which is why it sits above the 3.4% test-clean
figure Whisper is usually quoted at — a different split mix, not a worse model.

### 35.3 Why this lane exists: the CLIP text tower returns NaN

Running ImageNet-1k zero-shot on inf2, the traced CLIP **text** tower returned
NaN for all 1000 classes — **512,000 non-finite classifier entries** — while
`neuronx-cc` reported `Compiler status PASS` and the graph ran at 1,165
images/s. A throughput number would have called that a success.

Bisected to one variable (`extras/clip_nan_bisect.py`, six cells):

```
cpu    text +mask   CLEAN  0/2048      neuron text +mask   NaN  2048/2048
cpu    text -mask   CLEAN  0/2048      neuron text -mask   CLEAN   0/2048
cpu    image        CLEAN  0/2048      neuron image        CLEAN   0/2048
```

CLIP's text encoder fills a causal mask with `finfo(float32).min` (−3.4e38) and
then **adds** the padding mask, filled with the same constant. −3.4e38 +
−3.4e38 overflows to −inf, and `softmax(−inf − −inf)` is NaN. On CPU the
addition saturates harmlessly. This reproduces on trn1 and inf2
(NeuronCore-v2) and **not** on trn2 (v3).

`analysis/accuracy_summary.py` therefore flags **two** silent-failure shapes,
not one. A graph emitting a *constant* logit row is equally dead but scores at
chance rather than zero, so it reads as a mediocre model instead of a broken
compile. Both shapes are checked at table-render time, because a slide is where
an unpaired or dead comparison would do the most damage — nobody re-derives a
table during a talk.

### 35.4 Declared failures in this lane

| receipt | what happened |
|---|---|
| `trn1/results/accuracy/zs_clip_neuron_bf16.failure.json` | `RuntimeError: neuronx-cc failed with 2` |
| `trn1/results/accuracy/zs_clip_bf16_delta.failure.json` | refused to pair — `receipts differ in the EXPERIMENT, not just the engine`. The comparator declines to compute a delta across two different experiments rather than emit a meaningless one. |

`extras/clip_lane.py` is **deliberately not fixed**. Its trn2 figure (188.48
images/s, correct probabilities) was measured *with* the attention mask;
silently changing the traced signature now would leave three boxes reporting a
"CLIP parity" number derived from two different graphs.

## 36. Speculative decoding, measured properly (Phase 5)

§13.11 reported a single fused-speculation point from `inference_demo`. This
section replaces it with a full acceptance ladder over **Spec-Bench** — 39
prompts across 13 categories — on trn1.2xlarge: Llama-3.1-8B-Instruct target +
Llama-3.2-1B-Instruct draft, NxDI fused speculation, TP=2, batch 1, seq_len
1280, greedy, EOS active.

```bash
python3 analysis/specdec_summary.py trn1/results/specdec
```

### 36.1 The correction that had to come first

`sb_k0.json` capped every prompt at exactly 128 generated tokens (39×128 =
4,992 tokens, one distinct length) while every k≥2 arm ran uncapped to
`max_length=1280` (~1,120 tokens/prompt). Prefill was therefore amortised
**~8.7× less** in the baseline, inflating every speedup in the table.

`sb_k0_matched.json` re-ran k=0 with an invocation byte-identical to the k≥2
arms except the speculation flags. Baseline moved **30.85 → 31.61 tok/s**, and
output-length distributions now match ([439, 1270] on both arms). Every number
below uses the corrected baseline.

### 36.2 The ladder

| k | tok/s | speedup | agreement | E[accepted] | ms/call |
|---|---|---|---|---|---|
| 0 | 31.61 | 1.000× | — | 1.001 | 31.66 |
| 2 | 44.51 | 1.408× | 96.6% | 1.932 | 43.40 |
| 3 | 56.41 | 1.784× | 92.8% | 2.783 | 49.35 |
| 4 | 64.30 | 2.034× | 88.8% | 3.551 | 55.22 |
| 5 | 70.37 | 2.226× | 85.9% | 4.294 | 61.02 |
| 6 | 73.86 | 2.336× | 82.4% | 4.945 | 66.96 |
| 7 | 77.34 | **2.447×** | 80.4% | 5.628 | 72.77 |
| 10 | 78.47 | **2.482×** | 70.9% | 7.085 | 90.29 |

**Speedup saturates, and the reason is acceptance, not cost.** The cost side is
almost perfectly linear —

```
t_call(k) = 31.711 ms + 5.864 ms × k      R² = 0.999993
```

— fitted over 8 points, with k=0 excluded from the fit and landing on the
intercept to within 0.05 ms. The implied draft/target cost ratio is **0.1849**,
measured from the slope rather than assumed from weight bytes.

What decays is agreement: 96.6% at k=2 down to 70.9% at k=10. Going k=7 → k=10
costs three more draft tokens and buys **+0.036× (1.4%)**, where k=0 → k=2
bought +0.408×. E[accepted] at k=10 came in at 7.085 against a naive
extrapolation of 7.68 — the naive prediction overshoots by 7.6%.

### 36.2.1 The single-prompt lane, and where the optimum actually is

A separate single-prompt lane (`trn1/results/specdec`) runs the same target and
draft against one fixed prompt, and pushes k further than Spec-Bench did:

| k | ms/token | speedup | E[accepted] | accept rate |
|---|---|---|---|---|
| 2 | 21.918 | 1.481× | 2.054 | 0.642 |
| 4 | 14.314 | 2.267× | 4.022 | 0.891 |
| 7 | 11.231 | 2.890× | 6.803 | 0.953 |
| 8 | 10.931 | 2.969× | 7.564 | 0.956 |
| **16** | **9.676** | **3.354×** | 13.735 | 0.973 |

against a 32.455 ms/token (30.81 tok/s) baseline, with draft cost **measured**
at 6.278 ms/token — a draft/target ratio of **0.1934**, independently close to
the 0.1849 the Spec-Bench cost model fitted from its slope.

Two things about that table are easy to over-read, so they are stated here
rather than left to the tool's source:

- **The speedup column is decode-only.** `analysis/specdec_summary.py` subtracts
  prefill from e2e and divides by generated tokens, on both arms. That is the
  same basis as the 2.890× already published, so the column is internally
  consistent — but **end-to-end**, k=16 is **3.190×** (2,646.0 ms against
  8,440.2 ms), not 3.354×. Speculation makes prefill *more* expensive here, by a
  steady ~28% at every k, and the decode-only figure excludes that penalty.
- **E[accepted] is derived, not counted.** It is inferred from the fitted
  iteration cost, `(baseline_per_tok + k × draft_per_tok) / measured_per_tok`,
  and the acceptance rate is then solved from it numerically. Nothing in this
  lane counts accepted drafts directly — the probe built to do exactly that is
  broken (§36.2.2).

**Quote the Spec-Bench numbers, not these.** One prompt is not 39 prompts across
13 categories, and this lane's acceptance rate reaches 0.973 where Spec-Bench's
agreement had fallen to 70.9% by k=10 — a single prompt is exactly the
condition under which a draft model looks best. What the lane does establish is
that the earlier note ("peak sits at the largest k measured — the optimum may
lie beyond the sweep") was right: extending to k=16 kept gaining, and the peak
still sits at the edge of the sweep. Where the true optimum is remains unmeasured,
and now unmeasurable — the box was terminated on 2026-08-26.

`spec_k1` is a recorded failure: `RuntimeError: Error while lowering:
aten::cumsum`.

### 36.2.2 An instrument that failed its own self-check — and what it was worth anyway

The lane also carries a direct acceptance probe (`accept_k*.json`) that hooks
`ModelWrapper.forward` and computes E[accepted] as tokens ÷ decode-graph
invocations. Its own note states the criterion: *"Baseline (k=0) MUST give 1.0 —
that is the self-check."*

It gives **0.2009**, and reports a *negative* number of accepted drafts for
every k below 5:

| k | 0 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| raw E[accepted] | 0.201 | 0.402 | 0.601 | 0.798 | 1.000 | 1.202 | 1.391 |
| accepted drafts | −0.799 | −0.598 | −0.399 | −0.203 | 0.000 | 0.202 | 0.391 |

The failure is a **constant factor, not noise**. At k=0 the hook logs 1,274
invocations for 256 tokens: it fires **4.977× per decode step**, so every entry
in the row is scaled by ~1/5. That is also why the column reads exactly 1.000 at
k=5 — the point where invocations happen to equal generated tokens — which is
precisely the plausible-looking middle value that would have survived review.

Because the distortion is constant, the probe's own self-check calibrates it.
Dividing through by 4.977 recovers E[accepted], and it lands on the
timing-derived column from §36.2.1 — a completely independent path, built from
latency and a fitted draft cost rather than from counting anything:

| k | normalised probe | timing-derived | difference |
|---|---|---|---|
| 2 | 2.000 | 2.054 | −2.6% |
| 3 | 2.991 | 3.052 | −2.0% |
| 4 | 3.969 | 4.022 | −1.3% |
| 5 | 4.977 | 4.928 | +1.0% |
| 6 | 5.981 | 5.899 | +1.4% |
| 7 | 6.924 | 6.803 | +1.8% |

**Two unrelated methods agree to within 2.6% across the ladder.** Counting
invocations and modelling latency have no shared failure mode, so the agreement
is real corroboration of the acceptance figures rather than a consistency check
of one method against itself.

`analysis/specdec_summary.py` still does not consume the probe, and the numbers
in §36.2.1 remain the timing-derived ones — a normalisation derived after the
fact from a broken instrument is not something to publish as a primary
measurement. But the instrument was salvageable precisely because it declared
the invariant that exposed it. An instrument that states what it must return on
a control, and is kept when it violates it, is worth more than one that quietly
returns something plausible.

### 36.3 gpt-oss-20b: a shape wall, not a memory wall

The lane pre-registered a question: does NxDI load MXFP4 as-is (~13.8 GB, fits)
or dequantise to bf16 (~42 GB, does not fit)? **It dequantises** — *"Using MXFP4
quantized models requires a GPU, we will default to dequantizing the model to
bf16"*. But the run never reached a memory limit. It died on:

```
[NCC_INKI016] Kernel validation exception:
H=2880 must be divisible by 128
```

`nkilib/core/utils/kernel_assert.py:17` asserts `dims.H % _pmax == 0` with
`_pmax = 128` — the NeuronCore partition-dimension maximum, the 128 of the
128×128 systolic array. gpt-oss-20b's `hidden_size` is 2880, and 2880 / 128 =
22.5. It is divisible by 64, not by 128.

**There is no OOM evidence at all**: `dmesg` out-of-memory matches = 0, server
log HBM-exceeded matches = 0, host memory ended with 9 GiB free of 15 GiB. The
verdict is that gpt-oss-20b is blocked on inf2 by a **MoE kernel shape
constraint**, not by model size and not by memory. Two secondary packaging gaps
surfaced on the way: the blockwise MoE NKI kernels fail to import
(`No module named 'neuronxcc.nki._private.blockwise_mm'`), and SWIGLU is not yet
supported in the selected blockwise matmul kernel.

This supersedes the earlier `gpt_oss_20b_short` receipt, which died sooner at
`ModuleNotFoundError: accelerate` and never reached the question the lane was
written to answer. §13.9 previously recorded this as "driver not built
(declared)"; that is retracted. Receipt:
`inf2/results/extras/serve/gpt_oss_20b_short_retry/load_failure.json`.

## 37. A second SFT dataset: does the headline replicate? (Phase 5)

The 68.3% MFU headline came from one dataset (dolly-15k). Running the identical
recipe on **allenai/tulu-3-sft-mixture** — same model, same LoRA r16/α32, same
seq 2048, micro-batch 1, grad-accum 8, 645 steps, seed 42 — tests whether the
number is a property of the chip or of the corpus.

| | dolly-15k | Tulu-3 SFT mixture |
|---|---|---|
| median step | 5,550 ms | 5,527.68 ms |
| **tok/s** | **2,952** | **2,964.0** |
| MFU (210 TFLOP/s denom) | 68.3% | 68.60% |
| MFU (190 TFLOP/s denom) | — | 75.82% |
| final loss | 1.21 | 1.067 |

**+0.4% apart.** The throughput headline is a property of the chip and the
shapes, not of the corpus — which is what a throughput claim is supposed to
mean. Receipt: `trn1/results/sft/sft_llama31_8b_tulu3.json`.

Two accounting notes carried from that receipt, both of which keep the number
honest: end-to-end throughput is **1,425.9 tok/s**, not 2,964 — the median-step
window covers only 48.11% of wall time, and `measured_window_fraction` is
recorded precisely so the steady-state figure cannot be mistaken for the
wall-clock one. And `peak_device_mem_mib` is null rather than estimated,
because torch on XLA does not expose an HBM high-water mark; read
`mem_used_mib` from the matching telemetry CSV instead.

## 38. A comparison that did not survive its own control (Phase 5)

The midtrain lane was meant to compare learning-rate schedules on the 362M
from-scratch model: constant vs linear-warmup-then-cosine, 60 steps each,
FineWeb-Edu `sample-10BT`, everything else held fixed. **It is not published as
a comparison, because it failed its own control.**

The two runs differ in exactly one config key, `lr_schedule`. The scheduler
demonstrably ran — the logs show a constant `0.0001` on one arm and a varying
`0.0 → 1.09e-06 → 1.11e-05 → 1.85e-05 …` on the other — and the cosine arm paid
**215.4 s of compile against 12.6 s**, a 17× difference consistent with a
changing LR scalar being retraced into the graph.

And yet **all 60 logged loss values are bit-identical across the two runs**,
first to last, 10.994 → 6.4584, in the logs as well as in the JSON. Throughput
matches too (4,379.1 vs 4,374.4 tok/s).

A schedule that visibly changes the LR, visibly changes compile behaviour, and
changes *nothing at all* about the loss trajectory is not a result. Either the
LR the graph executes is not the LR the trainer logs, or the two arms are not
as independent as their configs claim. The study cannot distinguish those from
here: both boxes were terminated on 2026-08-26 and only EBS snapshots remain
(`docs/preservation/2026-08-26-RECOVERY.md`).

What is recorded, therefore, is the anomaly and not a schedule recommendation.
The lane's throughput and MFU figures (4,379 tok/s, 6.67% whole-chip / 13.34%
per-core-used) stand on their own as a second confirmation of the §32.11
pretraining numbers; the schedule axis does not. Both runs are named
`midtrain_finemath_*` for historical reasons but ran on **FineWeb-Edu**, per
the `dataset` field in both receipts — the filenames are misleading and the
receipts are authoritative.

### 36.4 int8 weights: the model loaded, the ruler did not

§13.9 records int8 as a "declared prep-stage gap". The recovered logs
(`trn1/results/ppl/int8.log`, `bf16.log`) make the boundary precise, and it sits
later than "prep stage" implies. The int8 checkpoint **shards and loads
successfully** — 66.2 s sharding, 76.3 s weight load, 77.5 s total, warmup
completed in 0.81 s — and NxDI reports the expected `Removing redundant keys
from checkpoint` for the per-projection `.scale` tensors. What failed is the
*measurement*:

```
File "/opt/np/ppl_harness.py", line 51, in ppl
    from datasets import load_dataset
ModuleNotFoundError: No module named 'datasets'
```

So there is no int8 perplexity number, and the study does not claim one. But the
gap is an **evaluation-harness packaging** gap, not an int8 support gap: the
quantised graph compiled, loaded and ran. That distinction matters to anyone
deciding whether int8 is viable on this stack — the answer this study can
support is "it loads and runs; we never scored it", not "it does not work".

## 39. What the S3 teardown nearly cost (Phase 5)

Preparing to delete the artifacts bucket surfaced a harness defect worth more
than the cleanup. `make pull-results` syncs `results/{trn1,trn2,inf2}/`. The
bucket held **nineteen** `results/` prefixes: the three canonical ones plus
per-lane and per-hostname prefixes written by on-box drivers
(`results/trn1-specdec/`, `results/final-ip-172-31-20-190-specdec/`,
`results/trn1-ppl/`, and thirteen more).

**736 objects had never been pulled**, and 50 of them existed nowhere else:

| what | why it mattered |
|---|---|
| `spec_k8.*`, `spec_k16.*` | extend the speculative ladder from 2.890× to **3.354×** and turn draft cost from withheld into measured (§36.2.1) |
| `accept_k0..k7.json` | the acceptance probe that fails its own self-check (§36.2.2) |
| `trn1-ppl/{bf16,int8}.log` | the int8 receipts that relocate the gap from "prep stage" to "eval harness" (§36.4) |
| `trn1-sft-failed/*` | dependency-failure receipts for the Tulu-3 lane |

Two files in the committed tree were also **zero bytes** — `draft_only.log` and
`spec_k8.log` — and their real content existed only in S3. A third,
`inf2/results/extras/spec_decode/baseline.failure.json`, is zero bytes in all
four copies: a failure receipt that records no failure. That is a defect in the
receipt writer, and it is left in place rather than deleted, because under this
study's own rules the presence of a `.failure.json` is the signal and its
emptiness is the bug to report.

The lesson generalises past this repo: **a sync target that names its prefixes
is a sync target that will silently miss the ones nobody remembered to add.**
`make pull-results-all` now mirrors the whole prefix so reconciliation is
possible at all.
