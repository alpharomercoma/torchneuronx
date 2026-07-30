-- extras/rag/schema.sql -- Track F RAG appliance: the one table.
--
-- Design mirrored from the source repo's rag_preprocess/vector_store.py
-- (id text pk, source doc, content, vector column, created timestamp,
-- HNSW cosine index with m=16 / ef_construction=64), with one deliberate
-- change: embedding is vector(1024), not vector(2048).
--
-- Why 1024: pgvector caps indexed vectors at 2000 dims, so the source
-- repo's 2048-dim Qwen3-Embedding-8B output could NOT be HNSW-indexed and
-- it shipped a documented sequential-scan workaround. Qwen3-Embedding-0.6B
-- emits 1024-dim (Matryoshka) embeddings natively -- sized down per the
-- co-residency arithmetic (their own >=48 GB floor vs this chip's 32 GB HBM,
-- see extras/rag/README.md) -- which keeps HNSW available instead of
-- inheriting the seq-scan workaround. A sizing constraint that IMPROVES the
-- index story is worth recording.
--
-- Rerun-safe: every statement is IF NOT EXISTS; setup_pg.sh applies this
-- file through the same DSN the lanes use (ingest.py / query.py).

-- setup_pg.sh already created the extension as superuser; this line is a
-- self-documenting no-op on the np_rag role when the extension exists.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,          -- {stem}::{section}-{NNNN}-{slug} (source repo namespace scheme)
    doc       text NOT NULL,             -- source document stem (e.g. REPORT)
    section   text NOT NULL,             -- markdown heading the chunk came from
    content   text NOT NULL,             -- raw chunk text, <= ~1200 chars (chunker contract)
    embedding vector(1024) NOT NULL,     -- Qwen3-Embedding-0.6B, L2-normalized
    created   timestamptz NOT NULL DEFAULT now()
);

-- Cosine HNSW: same operator class and build parameters as the source
-- repo's config (index_type hnsw, hnsw_m 16, hnsw_ef_construction 64) --
-- only here it can actually be created, because 1024 <= 2000.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON COLUMN chunks.embedding IS
    '1024-dim (Matryoshka) Qwen3-Embedding-0.6B: keeps HNSW available vs the source repo''s 2048-dim seq-scan workaround (pgvector index cap = 2000 dims)';
