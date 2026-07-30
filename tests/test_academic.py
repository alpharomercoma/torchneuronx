"""Academic track: model shapes + LR schedule, CPU-only.

    uv run --with torch --with torchvision --with pytest python -m pytest tests/test_academic.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "academic"))
torch = pytest.importorskip("torch")
from models import build, wants_flat  # noqa: E402
from train_academic import cosine_warmup, resolve_cfg, build_parser  # noqa: E402


@pytest.mark.parametrize("ds,img,ch", [("mnist", 28, 1), ("cifar", 32, 3)])
@pytest.mark.parametrize("arch", ["mlp", "cnn", "vit"])
def test_forward_shapes(ds, img, ch, arch):
    m = build(ds, arch).eval()
    x = torch.randn(4, ch, img, img)
    if wants_flat(arch):
        x = x.flatten(1)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (4, 10)
    assert sum(p.numel() for p in m.parameters()) > 10_000


def test_cosine_warmup_shape():
    total, warm = 1000, 100
    assert cosine_warmup(0, total, warm) == pytest.approx(0.01)
    assert cosine_warmup(99, total, warm) == pytest.approx(1.0)
    assert cosine_warmup(100, total, warm) == pytest.approx(1.0)
    assert cosine_warmup(999, total, warm) < 0.001
    assert cosine_warmup(0, total, 0) == pytest.approx(1.0)   # mnist: no warmup


def test_defaults_mirror_mlx():
    args = build_parser().parse_args(
        ["--dataset", "cifar", "--arch", "vit", "--out", "/tmp/x.json"])
    cfg = resolve_cfg(args)
    assert cfg == dict(epochs=40, lr=6e-4, weight_decay=5e-2, adamw=True)
