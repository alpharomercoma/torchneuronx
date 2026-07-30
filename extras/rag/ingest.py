#!/usr/bin/env python3
"""Track F / F4: ingest this repo's own docs into pgvector.

Corpus is self-referential by design (plan F4): README.md, METHODOLOGY.md,
REPORT.md and docs/runbook/*.md -- no PDF dependencies, and the probe suite
(probes.py) can then ask the stack questions whose answers are known facts
from REPORT.md.

Pipeline: heading-aware markdown chunking (max_chars ~= 1200, the source
repo's chunker.max_chars) -> chunk ids in the source repo's namespace scheme
{stem}::{section}-{NNNN}-{slug} -> Qwen3-Embedding-0.6B on Neuron (batch 8)
-> Postgres upsert.

ZERO-DEP DATABASE PATH, on purpose: psycopg is NOT installed on the inf2 box
and this repo's style contract is stdlib-only clients (see
shared/serve/fallback_client.py). All SQL goes through a `psql` subprocess:
bulk load is COPY FROM STDIN with CSV rows whose embedding column is a
pgvector literal '[f1,f2,...]', staged into a TEMP table and merged with
INSERT ... ON CONFLICT (id) DO UPDATE (the source repo's upsert, moved from
psycopg executemany to a single COPY -- faster and dependency-free).

Metrics JSON: {docs, chunks, chunks_per_s, embed_ms_p50, wall_s, captured}.
embed_ms_p50 is the p50 over per-batch-call embed latencies (batch 8).
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys
import time

from compile_models import EMBED_DIRNAME, DEFAULT_MODELS_DIR

DEFAULT_DSN = os.environ.get(
    "PG_DSN", "postgresql://np_rag:np_rag@localhost:5432/np_rag")

# Default corpus globs, relative to --repo-root.
DEFAULT_GLOBS = ["README.md", "METHODOLOGY.md", "REPORT.md",
                 "docs/runbook/*.md"]

# ------------------------------------------------------------------ chunker
# Namespace scheme mirrored from the source repo's rag_preprocess/chunker.py:
#   {normalized_stem}::{section}-{NNNN}-{slug}
# stem normalization keeps [A-Za-z0-9_-]; slugs are lowercase [a-z0-9-],
# capped at 40 chars; NNNN is a zero-padded per-section index.
_NAMESPACE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def normalize_stem(stem):
    cleaned = _NAMESPACE_RE.sub("_", stem).strip("_-")
    return cleaned or "doc"


def slug(text, max_len=40):
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "item"


def _hard_split(block, max_chars):
    """Split a single oversized block (e.g. a long markdown table) at line,
    then word, boundaries so every piece is <= max_chars."""
    if len(block) <= max_chars:
        return [block]
    out, rest = [], block
    while len(rest) > max_chars:
        cut = rest.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = rest.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def chunk_markdown(text, stem, max_chars=1200):
    """Heading-aware chunker -> [{id, doc, section, content}].

    Headings (#..######) open a new section; blocks (paragraphs, tables,
    code fences read as plain blocks) accumulate into a chunk until adding
    one would exceed max_chars. Single blocks longer than max_chars are
    hard-split. Every chunk content is <= max_chars by construction
    (tested in tests/test_rag.py).
    """
    ns = normalize_stem(stem)
    # Pass 1: split into (kind, text) blocks on headings and blank lines.
    blocks, cur = [], []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if cur:
                blocks.append(("text", "\n".join(cur).strip()))
                cur = []
            blocks.append(("heading", m.group(2).strip()))
        elif not line.strip():
            if cur:
                blocks.append(("text", "\n".join(cur).strip()))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(("text", "\n".join(cur).strip()))

    # Pass 2: assemble chunks per section with the max_chars budget.
    chunks = []
    section = "intro"                 # preamble before any heading
    sec_counts = {}
    buf, buf_len = [], 0

    def flush():
        nonlocal buf, buf_len
        content = "\n\n".join(buf).strip()
        buf, buf_len = [], 0
        if not content:
            return
        sec = slug(section)
        i = sec_counts.get(sec, 0)
        sec_counts[sec] = i + 1
        chunks.append({
            "id": f"{ns}::{sec}-{i:04d}-{slug(content[:80])}",
            "doc": stem,
            "section": section,
            "content": content,
        })

    for kind, block in blocks:
        if kind == "heading":
            flush()
            section = block or "intro"
            continue
        for piece in _hard_split(block, max_chars):
            joined = len(piece) + (2 if buf else 0)
            if buf and buf_len + joined > max_chars:
                flush()
                joined = len(piece)
            buf.append(piece)
            buf_len += joined
    flush()
    return chunks


# ------------------------------------------------------------------ pgvector
def to_pgvector(vec):
    """Python floats -> pgvector input literal '[f1,f2,...]' (no spaces).

    6 decimals: the embeddings are L2-normalized so |x| <= 1; 1e-6 absolute
    precision is far below any retrieval-relevant angle. Tested in
    tests/test_rag.py.
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def run_psql(dsn, sql_script, timeout=600):
    """Run a SQL script (optionally with inline COPY data) through psql.

    psql ships with postgresql-16 (setup_pg.sh); no python driver needed.
    """
    proc = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-q"],
        input=sql_script.encode(), capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"psql failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:800]}")
    return proc.stdout.decode(errors="replace")


def upsert_chunks(dsn, rows):
    """COPY rows into a TEMP staging table, then merge with ON CONFLICT.

    rows: [(id, doc, section, content, embedding_literal), ...]

    One psql session carries the whole thing: TEMP tables are per-session,
    so staging never leaks into the schema. Inline COPY data ends at a line
    that is exactly `\\.` -- CSV-quoted content with embedded newlines is
    fine (psql only terminates on the bare marker), but a content LINE that
    is literally `\\.` would truncate the stream, so it is rejected loudly
    below rather than corrupting the load.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for r in rows:
        writer.writerow(r)
    csv_data = buf.getvalue()
    if "\n\\.\n" in ("\n" + csv_data):
        raise ValueError(r"corpus contains a bare '\.' line; refusing COPY")

    script = (
        "CREATE TEMP TABLE chunks_staging "
        "(LIKE chunks INCLUDING DEFAULTS);\n"
        "COPY chunks_staging (id, doc, section, content, embedding) "
        "FROM STDIN WITH (FORMAT csv);\n"
        + csv_data + "\\.\n" +
        "INSERT INTO chunks (id, doc, section, content, embedding)\n"
        "  SELECT id, doc, section, content, embedding FROM chunks_staging\n"
        "ON CONFLICT (id) DO UPDATE SET\n"
        "  doc = EXCLUDED.doc, section = EXCLUDED.section,\n"
        "  content = EXCLUDED.content, embedding = EXCLUDED.embedding,\n"
        "  created = now();\n"
    )
    run_psql(dsn, script)


# ---------------------------------------------------------------------- main
def collect_files(repo_root, globs):
    files = []
    for g in globs:
        pat = g if os.path.isabs(g) else os.path.join(repo_root, g)
        files.extend(sorted(glob.glob(pat)))
    files = [f for f in dict.fromkeys(files) if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"no corpus files matched {globs} under {repo_root}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Ingest repo docs into pgvector")
    ap.add_argument("--repo-root", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--corpus-glob", action="append", default=None,
                    help=f"repeatable; default: {DEFAULT_GLOBS}")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--out", required=True, help="metrics JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk only: no embedding, no database")
    args = ap.parse_args()

    t_wall = time.perf_counter()
    files = collect_files(args.repo_root, args.corpus_glob or DEFAULT_GLOBS)
    print(f"# corpus: {len(files)} files", file=sys.stderr)

    chunks = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as fh:
            chunks.extend(chunk_markdown(fh.read(), stem,
                                         max_chars=args.max_chars))
    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        # Two corpus files with the same stem would collide in the
        # namespace; the default corpus is collision-free, so this is a
        # loud guard, not a silent dedupe.
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise SystemExit(f"duplicate chunk ids (same stem twice?): {dupes}")
    print(f"# chunks: {len(chunks)}", file=sys.stderr)

    embed_lat_ms = []
    if not args.dry_run:
        from embed_lane import NeuronEmbedder   # heavy path stays lazy
        embedder = NeuronEmbedder(
            os.path.join(args.models_dir, EMBED_DIRNAME))
        vectors = []
        bs = embedder.batch_size
        for i in range(0, len(chunks), bs):
            batch = chunks[i:i + bs]
            # Documents are embedded raw (no Instruct wrapper -- asymmetry
            # per model card), with a light "doc / section" header so a
            # chunk carries its own provenance into the vector.
            texts = [f"{c['doc']} / {c['section']}\n{c['content']}"
                     for c in batch]
            t0 = time.perf_counter()
            vectors.extend(embedder.embed(texts))
            embed_lat_ms.append((time.perf_counter() - t0) * 1000)
        rows = [(c["id"], c["doc"], c["section"], c["content"],
                 to_pgvector(v)) for c, v in zip(chunks, vectors)]
        upsert_chunks(args.dsn, rows)
        total = run_psql(args.dsn, "SELECT count(*) FROM chunks;")
        print(f"# chunks table now: {total.split()[-1] if total.split() else '?'} rows",
              file=sys.stderr)

    wall_s = time.perf_counter() - t_wall
    sorted_lat = sorted(embed_lat_ms)
    payload = {
        "docs": len(files),
        "chunks": len(chunks),
        "chunks_per_s": round(len(chunks) / wall_s, 2) if wall_s else None,
        "embed_ms_p50": (round(sorted_lat[len(sorted_lat) // 2], 2)
                         if sorted_lat else None),   # per batch-8 call
        "wall_s": round(wall_s, 2),
        "max_chars": args.max_chars,
        "dry_run": args.dry_run,
        "files": [os.path.relpath(f, args.repo_root) for f in files],
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"docs={payload['docs']} chunks={payload['chunks']} "
          f"chunks/s={payload['chunks_per_s']} "
          f"embed_p50={payload['embed_ms_p50']}ms wall={payload['wall_s']}s "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
