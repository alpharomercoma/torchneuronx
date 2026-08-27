# extras/rag — Track F: RAG appliance on one inf2.xlarge

Port of [local-agentic-rag-with-qwen3](https://github.com/alpharomercoma/local-agentic-rag-with-qwen3)
(Qwen3 LLM + Qwen3 embedding + Qwen3 reranker + Postgres 16/pgvector) onto
AWS Neuron, sized to the chip by arithmetic, not vibes. Corpus is this repo's
own docs, so the probe suite asks the stack questions whose answers are
verbatim facts from [REPORT.md](../../REPORT.md).

## Two-command reproduce (on the inf2 box)

```bash
sudo bash extras/rag/setup_pg.sh     # Postgres 16 + pgvector + schema (idempotent)
bash extras/rag/run_rag.sh           # compile -> embed lane -> ingest -> probes -> LLM probes -> push
```

`run_rag.sh` re-runs `setup_pg.sh` itself (both are rerun-safe), boots the
LLM via `shared/serve/launch_vllm.sh`, and pushes results to S3 at the end.
Results land in `inf2/results/rag/`. `FORCE=1` redoes completed stages;
`RAG_LLM_KEY=<key>` picks the LLM ladder rung.

## Files

| File | What |
|---|---|
| `setup_pg.sh` | Idempotent Postgres 16 + pgvector install, role/db `np_rag`, applies `schema.sql`, verify echo |
| `schema.sql` | `chunks` table (`id` in the source repo's namespace scheme, `vector(1024)`), HNSW cosine index |
| `compile_models.py` | Compile-if-absent: Qwen3-Embedding-0.6B (b8 x s512) + Qwen3-Reranker-0.6B (b4 x s1024, ATTEMPT) with structured receipts |
| `embed_lane.py` | Embeddings/s at batch 1 and 8 (200 texts), p50/p99, dim==1024 assert; home of the shared `NeuronEmbedder` |
| `ingest.py` | Heading-aware markdown chunker (max_chars 1200) -> embed -> `psql` COPY upsert. Zero-dep: no psycopg, stdlib + psql subprocess only |
| `query.py` | embed -> pgvector cosine top-12 -> rerank to top-3 (declared skip if reranker absent) -> LLM answer; per-stage timings JSON |
| `probes.py` | 7 REPORT.md-grounded factual probes, substring-checked; `--no-llm` retrieval-hit mode |
| `run_rag.sh` | Driver (academic-track pattern): setup -> compile -> lanes -> LLM boot -> probes -> stop -> push |
| `tests/test_rag.py` | Local pure gate: chunker/namespace, probe checker, schema string sanity, pgvector literal, py_compile |
| `README.md` | This exec-spec |

## Co-residency arithmetic (why the models are sized down)

The source repo's own README declares a **>= 48 GB VRAM floor** for its three
8B models resident in BF16. One Inferentia2 has **32 GB HBM** (2 x 16 GB
NeuronCore banks). Three co-resident 8B models are therefore impossible on
this chip by subtraction — that triggers the sizing rule (plan: "iff not
possible"), and the receipt is the arithmetic itself:

- **Embedding**: Qwen3-Embedding-**0.6B** (official optimum-neuron tutorial
  class `NeuronModelForEmbedding`, last-token pooling, 1024-dim Matryoshka).
  Bonus: 1024 <= pgvector's 2000-dim index cap, so the **HNSW index works** —
  the source repo's 2048-dim embeddings forced a documented seq-scan
  workaround.
- **Reranker**: Qwen3-Reranker-**0.6B** via `NeuronModelForCausalLM` yes/no
  token scoring (HF model-card pattern). **Attempt-only** per the validity
  table: on compile or forward failure a structured receipt is written and
  the stack degrades to no-rerank mode, declared in every query JSON.
- **LLM ladder** (all-Qwen attempt first, each rung receipted):
  1. `qwen3_base` — Qwen3-8B via NxDI (known Phase-1 §9 outcome: boots,
     crashes on first generated token; re-running merely re-receipts it),
  2. Qwen2.5-7B via vLLM (qwen2 is a different Tier-1 path than the crashed
     qwen3; the `qwen25_7b` catalogue entry it needs now exists at
     `shared/serve/launch_vllm.sh:217`, alongside `qwen25_1_5b` at :242),
  3. `llama31_base` — Llama-3.1-8B-Instruct, the proven fallback and the
     driver default (`RAG_LLM_KEY=llama31_base`).

**Scheduling**: the primary Phase-2 plan (int8 LLM TP=1 on core 0 +
embed/rerank pinned to core 1) depends on Track B2's quantized artifact. This
driver ships the declared fallback: **batch ingest + retrieval probes run
offline before the LLM boots** (both cores free), then the LLM serves TP=2
and the end-to-end probe pass co-schedules query-time embedding next to it —
that co-residency is the declared risk; if the runtime refuses, the
retrieval-only numbers already exist and the failure is recorded.

## Zero-dependency database path

`psycopg` is **not** installed on the inf2 box, and this repo's client-side
style contract is stdlib-only (`shared/serve/fallback_client.py`). All SQL
goes through `psql` subprocesses: bulk load is `COPY FROM STDIN` (CSV with
pgvector `'[f1,f2,...]'` literals) into a TEMP staging table, merged with
`INSERT ... ON CONFLICT (id) DO UPDATE` — the source repo's upsert without
its driver dependency.

## Latency table (measured; every cell traces to a receipt in `inf2/results/rag/`)

| Stage | p50 | Source |
|---|---|---|
| embed (query, batch-1 graph call) | **737.1 ms** retrieval-only · **597.0 ms** with the 3B LLM resident | `probes_nollm.json`, `probes_llm.llama32_3b.json` → `timings.embed_ms` |
| pgvector retrieve (top-12, HNSW) | **316.5 ms** · 227.1 ms in the 3B run | `timings.retrieve_ms` |
| rerank (12 → 3, batch 4 × 3 calls) | **1,870.2 ms** | `probes_nollm_rerank.json` → `timings.rerank_ms`; null in no-rerank mode |
| LLM TTFT (3 contexts + question) | **1,528.1 ms** (Llama-3.2-3B) | `probes_llm.llama32_3b.json` → `timings.ttft_ms` |
| end-to-end query | **101.3 s** retrieval-only · **126.0 s** with the 3B | `timings.e2e_ms` — **dominated by per-process model load**, not by the query. Each probe process loads its models from cold; the stage rows above are the query cost |
| ingest | **1.49 chunks/s** (14 docs → 72 chunks, 48.2 s wall, embed p50 513.0 ms) | `ingest.json` |
| embeddings/s batch 1 / batch 8 | **1.96 / 15.65** | `embed_lane.json` — same 511 ms graph call either way, so batch 8 is a straight 8× on a static-shape graph; batch 1 pads 7 slots away |
| embedder load | **56.5 s** | `embed_lane.json` → `load_s`, Qwen3-Embedding-0.6B, 1024-dim, seq 512, compiled batch 8 |
| probes passed | **7/7 retrieval-only**, 4/7 with rerank, **0/7 end-to-end on Llama-3.1-8B**, 1/7 on Llama-3.2-3B | `probes_nollm.json`, `probes_nollm_rerank.json`, `probes_llm.json`, `probes_llm.llama32_3b.json` |

Read the last row honestly: **retrieval works and generation does not.** The
vector stage answers all seven probes; the generative stage on top of it
answers at most one. Reranking makes retrieval *worse* on this corpus (7/7 →
4/7). Those are the results, and they are why the RAG lane is reported as an
appliance that retrieves rather than one that answers.

## Local gate

```bash
cd extras/rag && uv run --with pytest python -m pytest tests/test_rag.py -q
bash -n extras/rag/setup_pg.sh extras/rag/run_rag.sh
```

No AWS, no postgres, no Neuron needed locally — the tests exercise only the
pure parts (chunker, namespace ids, probe checker, literals, compile checks).
