"""MNIST/CIFAR-10 models: MLP, CNN, ViT -- faithful torch ports of mlx-models.

Every architecture mirrors github.com/alpharomercoma/mlx-models (1_mnist,
2_cifar) layer for layer and dimension for dimension, so accuracy is
comparable across silicon: the MLX repo measured these on an Apple M5, this
repo measures the same nets on one Trainium1 NeuronCore. Port notes:

  * torch is NCHW where MLX is NHWC -- reshapes/flattens land identically.
  * MLX nn.MultiHeadAttention has no bias on qkv by default; torch
    nn.MultiheadAttention has bias. ~0.1% of ViT params; noted, not chased.
  * Weight init differs between frameworks (both default schemes). The MLX
    README reports accuracy RANGES for exactly this reason; ranges are the
    comparison unit here too.

Pick with `build(dataset, arch)`; `wants_flat(arch)` matches the MLX helper.

    python3 -c "from academic.models import build; \
        print(sum(p.numel() for p in build('cifar','vit').parameters()))"
"""
import torch
import torch.nn as nn

NUM_CLASSES = 10
SPECS = {
    "mnist": dict(img=28, ch=1),
    "cifar": dict(img=32, ch=3),
}


def wants_flat(arch):
    """True if this arch expects flattened input rather than images."""
    return arch == "mlp"


class MLP(nn.Module):
    """mnist: 784->256->256->10 (no dropout). cifar: 3072->512->512->10, drop 0.2."""

    def __init__(self, in_dim, hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.fc3(x)


class MnistCNN(nn.Module):
    """LeNet-style: 32/64 3x3 conv + two 2x2 pools -> fc128 -> 10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        spatial = 28 // 4
        self.fc1 = nn.Linear(64 * spatial * spatial, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class CifarCNN(nn.Module):
    """VGG-ish with BatchNorm: 32/64/128 + three 2x2 pools -> fc256 -> 10."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        spatial = 32 // 8
        self.fc1 = nn.Linear(128 * spatial * spatial, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x)


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: MHSA + MLP, residuals on both (as in MLX)."""

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """From-scratch ViT; mnist: p7/d64/x4/h4/mlp128, cifar: p4/d128/x6/h8/mlp256."""

    def __init__(self, img, ch, patch, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        assert img % patch == 0
        num_patches = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(ch, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_CLASSES)

    def forward(self, x):
        n = x.shape[0]
        x = self.patch_embed(x)                    # (N, dim, h, w)
        x = x.flatten(2).transpose(1, 2)           # (N, patches, dim)
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def build(dataset, arch):
    s = SPECS[dataset]
    flat = s["img"] * s["img"] * s["ch"]
    if dataset == "mnist":
        if arch == "mlp":
            return MLP(flat, hidden=256)
        if arch == "cnn":
            return MnistCNN()
        if arch == "vit":
            return ViT(s["img"], s["ch"], patch=7, dim=64, depth=4,
                       heads=4, mlp_dim=128)
    if dataset == "cifar":
        if arch == "mlp":
            return MLP(flat, hidden=512, dropout=0.2)
        if arch == "cnn":
            return CifarCNN()
        if arch == "vit":
            return ViT(s["img"], s["ch"], patch=4, dim=128, depth=6,
                       heads=8, mlp_dim=256)
    raise ValueError(f"unknown {dataset}/{arch}")
