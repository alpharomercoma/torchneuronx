"""Extras track: pure helpers only -- no Neuron, torch, or optimum needed.

The lane modules keep every heavy import lazy (inside main()/helpers), so
importing them here in a bare environment is itself part of the contract.

    uv run --with pytest python -m pytest tests/test_extras.py -q
"""
import json
import os
import subprocess
import sys

import pytest

EXTRAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extras"))
sys.path.insert(0, EXTRAS)

import clip_lane      # noqa: E402
import siglip_lane    # noqa: E402
import whisper_lane   # noqa: E402

MLX_LABELS = [
    "two cats lying on a couch",
    "a dog playing in a park",
    "a plate of food on a table",
    "a car driving on a road",
    "a person riding a bicycle",
]
MLX_DEMO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


# ------------------------------------------------------------ whisper: RTF
def test_rtf_math():
    assert whisper_lane.compute_rtf(5.5, 11.0) == pytest.approx(0.5)
    assert whisper_lane.compute_rtf(11.0, 11.0) == pytest.approx(1.0)
    assert whisper_lane.compute_rtf(0.0, 11.0) == 0.0


def test_rtf_rejects_nonpositive_audio():
    with pytest.raises(ValueError):
        whisper_lane.compute_rtf(1.0, 0.0)
    with pytest.raises(ValueError):
        whisper_lane.compute_rtf(1.0, -3.0)


def test_reference_head_fuzzy_match():
    # Punctuation/casing drift must not break the match...
    assert whisper_lane.matches_reference_head(
        "And so, my fellow Americans: ask not what your country can do for you")
    assert whisper_lane.matches_reference_head(
        "and so my fellow americans ask not")
    # ...but actual mistranscription must.
    assert not whisper_lane.matches_reference_head("ask not what your country")
    assert not whisper_lane.matches_reference_head("")


def test_whisper_sample_url_matches_mlx():
    assert whisper_lane.SAMPLE_URL == (
        "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/"
        "samples/jfk.wav")


# ----------------------------------------------------- clip: mlx demo parity
def test_clip_labels_match_mlx():
    assert clip_lane.DEFAULT_LABELS == MLX_LABELS
    assert clip_lane.DEMO_URL == MLX_DEMO_URL
    assert clip_lane.MODEL_ID == "openai/clip-vit-base-patch32"


def test_siglip_demo_matches_mlx():
    assert siglip_lane.DEFAULT_LABELS == MLX_LABELS
    assert siglip_lane.DEMO_URL == MLX_DEMO_URL
    assert siglip_lane.MODEL_ID == "google/siglip-base-patch16-224"


# ------------------------------------------------- siglip: receipt writer
def test_siglip_receipt_shape(tmp_path):
    out = tmp_path / "siglip.json"
    payload = siglip_lane.write_receipt(str(out), reason="KeyError: 'siglip'")
    data = json.loads(out.read_text())
    assert data == payload
    assert data["status"] == "export_unsupported"
    assert data["reason"] == "KeyError: 'siglip'"
    assert data["mirror_note"] == (
        "mlx-models 5_siglip runs this on Apple M5 -- ecosystem gap")
    assert set(data) >= {"status", "model", "reason", "mirror_note", "captured"}
    assert data["captured"].endswith("Z")


def test_siglip_receipt_status_override(tmp_path):
    out = tmp_path / "siglip.json"
    siglip_lane.write_receipt(str(out), reason="export succeeded",
                              status="unexpected_success")
    data = json.loads(out.read_text())
    assert data["status"] == "unexpected_success"


# --------------------------------------------------- lane scripts compile
@pytest.mark.parametrize(
    "script", ["whisper_lane.py", "clip_lane.py", "siglip_lane.py"])
def test_lane_scripts_py_compile(script):
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", os.path.join(EXTRAS, script)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
