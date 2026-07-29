"""grid.json schema: every sweep must self-declare, reductions included.

    python3 -m pytest tests/test_grids.py -q   # 2 passed
"""
import json
import os
import subprocess

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "grid_example.json")
REQUIRED = {"model_key": str, "config": str, "shapes": list,
            "concurrency": list, "reduced": bool, "reduction_note": str,
            "client": str, "num_prompts_rule": str}


def test_fixture_matches_schema():
    g = json.load(open(FIXTURE))
    for key, typ in REQUIRED.items():
        assert isinstance(g.get(key), typ), f"{key} missing or wrong type"
    assert g["reduced"] is True          # this repo's grids are always declared-reduced
    assert "KV" in g["reduction_note"]   # the reasoning must travel with the data


def test_bench_serve_emits_same_field_names():
    # The emitting heredoc lives in bench_serve.sh; keep it honest by string
    # inspection (running it needs a live server).
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "shared", "serve", "bench_serve.sh")).read()
    for key in REQUIRED:
        assert f'"{key}"' in src, f"bench_serve.sh grid.json lost field {key}"
    assert subprocess.run(["bash", "-n", os.path.join(
        os.path.dirname(__file__), "..", "shared", "serve", "bench_serve.sh")],
        capture_output=True).returncode == 0
