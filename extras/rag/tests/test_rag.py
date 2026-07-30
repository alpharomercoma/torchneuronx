"""Track F local gate: pure/offline tests, no postgres, no Neuron, no torch.

    cd extras/rag && uv run --with pytest python -m pytest tests/test_rag.py -q
"""
import os
import py_compile
import re
import sys

import pytest

RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAG_DIR)

from ingest import chunk_markdown, normalize_stem, slug, to_pgvector  # noqa: E402
from probes import PROBES, probe_pass  # noqa: E402

FIXTURE_MD = """\
Preamble paragraph before any heading lands in the intro section.

# Report

## Summary

Training reached 68.3% MFU at 2,952 tok/s on one trn1.2xlarge box.

The cold boot took 39.5 min while a warm NEFF cache booted in 9.1 minutes.

## Serving results

| conc | tok/s |
|---|---|
| 1 | 15.7 |
| 32 | 415.5 |

Long-context serving died with NCC_INLA001 on the inf2.xlarge box.
"""

NAMESPACE_PAT = re.compile(r"^[A-Za-z0-9_-]+::[a-z0-9-]+-\d{4}-[a-z0-9-]+$")


# ------------------------------------------------------------------ chunker
def test_chunker_namespace_and_max_chars():
    chunks = chunk_markdown(FIXTURE_MD, "REPORT", max_chars=200)
    assert chunks
    for c in chunks:
        assert NAMESPACE_PAT.match(c["id"]), c["id"]
        assert c["id"].startswith("REPORT::")
        assert len(c["content"]) <= 200, c["id"]
        assert c["doc"] == "REPORT"
    assert len({c["id"] for c in chunks}) == len(chunks)  # ids unique


def test_chunker_heading_aware_sections():
    chunks = chunk_markdown(FIXTURE_MD, "REPORT", max_chars=1200)
    sections = {c["section"] for c in chunks}
    assert "intro" in sections            # preamble before first heading
    assert "Summary" in sections
    assert "Serving results" in sections
    # facts land in their own section's chunks
    mfu = [c for c in chunks if "68.3" in c["content"]]
    assert mfu and mfu[0]["section"] == "Summary"


def test_chunker_hard_splits_oversized_block():
    long_block = ("word " * 400).strip()          # ~2000 chars, no blank lines
    chunks = chunk_markdown("# T\n" + long_block, "doc", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c["content"]) <= 300 for c in chunks)
    assert all(NAMESPACE_PAT.match(c["id"]) for c in chunks)


def test_stem_and_slug_normalization():
    assert normalize_stem("06-deploy inferentia!") == "06-deploy_inferentia"
    assert normalize_stem("///") == "doc"
    assert slug("Declared exclusions (rule 8)") == "declared-exclusions-rule-8"
    assert len(slug("x" * 100)) <= 40
    assert slug("!!!") == "item"


# ------------------------------------------------------------------- probes
def test_probe_substring_checker():
    assert probe_pass("the run reached 68.3% MFU", ["68.3"])
    assert probe_pass("crash was ncc_INLA001 again", ["NCC_INLA001"])  # ci
    assert probe_pass("2952 tok/s sustained", ["2,952", "2952"])       # variants
    assert not probe_pass("no numbers here", ["68.3"])
    assert not probe_pass(None, ["68.3"])                              # no answer


def test_probe_table_shape():
    assert len(PROBES) == 7
    for p in PROBES:
        assert p["q"].strip()
        assert p["expect_any"]


# ------------------------------------------------------------------- schema
def test_schema_sql_sanity():
    with open(os.path.join(RAG_DIR, "schema.sql")) as fh:
        text = fh.read().lower()
    assert "vector(1024)" in text
    assert "hnsw" in text
    assert "vector_cosine_ops" in text
    assert "chunks" in text
    assert "if not exists" in text        # rerun-safe


# ----------------------------------------------------------------- pgvector
def test_pgvector_literal_format():
    lit = to_pgvector([0.5, -1.0, 0.125])
    assert lit.startswith("[") and lit.endswith("]")
    assert " " not in lit
    vals = [float(x) for x in lit[1:-1].split(",")]
    assert vals == pytest.approx([0.5, -1.0, 0.125])


def test_pgvector_literal_dim_1024():
    lit = to_pgvector([0.001] * 1024)
    assert lit.count(",") == 1023


# ------------------------------------------------------------------ scripts
@pytest.mark.parametrize("name", ["compile_models.py", "embed_lane.py",
                                  "ingest.py", "query.py", "probes.py"])
def test_lane_scripts_py_compile(name):
    py_compile.compile(os.path.join(RAG_DIR, name), doraise=True)


def test_lazy_heavy_imports():
    """Importing the lane modules must not drag in torch/optimum/transformers
    -- the pure parts (chunker, formatters, probe checker) have to work on a
    laptop with none of the Neuron stack installed."""
    before = set(sys.modules)
    import compile_models  # noqa: F401
    import embed_lane      # noqa: F401
    import probes          # noqa: F401
    import query           # noqa: F401
    new = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not new & {"torch", "optimum", "transformers"}
