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
     qwen3; needs a `qwen25_7b` catalogue entry in
     `shared/serve/launch_vllm.sh` before it can run — TODO-VERIFY),
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

## Latency table (placeholders — filled from `inf2/results/rag/` when measured)

| Stage | p50 | Notes |
|---|---|---|
| embed (query, batch-1 graph call) | TBD ms | `probes_*.json` -> `timings.embed_ms` |
| pgvector retrieve (top-12, HNSW) | TBD ms | `timings.retrieve_ms` |
| rerank (12 -> 3, batch 4 x 3 calls) | TBD ms | `timings.rerank_ms`; null in no-rerank mode |
| LLM TTFT (3 contexts + question) | TBD ms | `timings.ttft_ms` |
| end-to-end query | TBD ms | `timings.e2e_ms` (includes per-process model load, broken out) |
| ingest | TBD chunks/s | `ingest.json` |
| embeddings/s batch 1 / batch 8 | TBD / TBD | `embed_lane.json` |
| probes passed | TBD/7 end-to-end, TBD/7 retrieval-only | `probes_llm.json` / `probes_nollm.json` |

## Local gate

```bash
cd extras/rag && uv run --with pytest python -m pytest tests/test_rag.py -q
bash -n extras/rag/setup_pg.sh extras/rag/run_rag.sh
```

No AWS, no postgres, no Neuron needed locally — the tests exercise only the
pure parts (chunker, namespace ids, probe checker, literals, compile checks).
