"""Academic-dataset training on one Trainium NeuronCore, mlx-models parity.

Same recipes as github.com/alpharomercoma/mlx-models so accuracy is
comparable across silicon (Apple M5 there, Trainium1 here):

  mnist: mlp/cnn Adam 5 epochs lr 1e-3; vit AdamW 10 epochs lr 1e-3
  cifar: full recipe -- random crop(pad 4)+hflip, cosine LR with 10% linear
         warmup, label smoothing 0.1, AdamW; mlp/cnn 30 epochs lr 1e-3
         wd 5e-4; vit 40 epochs lr 6e-4 wd 5e-2
  batch 128, seed 0, fp32, single NeuronCore (this is a small-model lane;
  TP would be overhead theater).

XLA notes (the reason this file is not just the MLX script with s/mx/torch/):
fixed shapes everywhere (drop_last on train, exact-divisor eval batches),
one mark_step per optimizer step, and the first step's wall time is the
graph compile -- reported as compile_s, never averaged into step stats.

    python3 academic/train_academic.py --dataset mnist --arch cnn \
        --out /tmp/m.json                       # ~99% test acc, minutes
    python3 academic/train_academic.py --dataset cifar --arch vit \
        --out /tmp/c.json                       # ~78-82% (from scratch)
"""
import argparse
import json
import os
import random
import sys
import time

DEFAULTS = {
    ("mnist", "mlp"): dict(epochs=5, lr=1e-3, weight_decay=0.0, adamw=False),
    ("mnist", "cnn"): dict(epochs=5, lr=1e-3, weight_decay=0.0, adamw=False),
    ("mnist", "vit"): dict(epochs=10, lr=1e-3, weight_decay=0.0, adamw=True),
    ("cifar", "mlp"): dict(epochs=30, lr=1e-3, weight_decay=5e-4, adamw=True),
    ("cifar", "cnn"): dict(epochs=30, lr=1e-3, weight_decay=5e-4, adamw=True),
    ("cifar", "vit"): dict(epochs=40, lr=6e-4, weight_decay=5e-2, adamw=True),
}
EVAL_BATCH = 1000          # 10k test sets divide exactly: fixed XLA shapes
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    ap.add_argument("--arch", choices=["mlp", "cnn", "vit"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.environ.get(
        "NP_DATA_DIR", "/opt/np/models/academic-data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None,
                    help="override (cpu for local smoke; default: XLA if available)")
    return ap


def resolve_cfg(args):
    cfg = dict(DEFAULTS[(args.dataset, args.arch)])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    return cfg


def cosine_warmup(step, total, warmup):
    """LR multiplier: linear warmup then cosine to zero. Pure; unit-tested."""
    import math
    if warmup and step < warmup:
        return (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))


def main():
    args = build_parser().parse_args()
    cfg = resolve_cfg(args)

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision as tv
    import torchvision.transforms as T

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
        is_xla = False
    else:
        import torch_xla
        device = torch_xla.device()
        is_xla = True

    def sync():
        if is_xla:
            import torch_xla
            torch_xla.sync()

    # ------------------------------------------------------------- data
    if args.dataset == "mnist":
        base = [T.ToTensor()]                      # /255, no further normalize (MLX parity)
        train_tf = T.Compose(base)
        test_tf = T.Compose(base)
        ds = tv.datasets.MNIST
    else:
        norm = [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
        aug = [] if args.no_augment else [
            T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        train_tf = T.Compose(aug + norm)
        test_tf = T.Compose(norm)
        ds = tv.datasets.CIFAR10

    train_set = ds(args.data_dir, train=True, download=True, transform=train_tf)
    test_set = ds(args.data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=4,
                              persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=EVAL_BATCH, shuffle=False,
                             drop_last=False, num_workers=2)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import build, wants_flat

    model = build(args.dataset, args.arch).to(device)
    params = sum(p.numel() for p in model.parameters())
    flat = wants_flat(args.arch)

    opt_cls = torch.optim.AdamW if cfg["adamw"] else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"])
    smoothing = args.label_smoothing if args.dataset == "cifar" else 0.0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = int(args.warmup_frac * total_steps) if args.dataset == "cifar" else 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_warmup(s, total_steps, warmup)) \
        if args.dataset == "cifar" else None

    print(f"# academic  dataset={args.dataset} arch={args.arch} "
          f"params={params:,} epochs={cfg['epochs']} lr={cfg['lr']} "
          f"wd={cfg['weight_decay']} adamw={cfg['adamw']} "
          f"batch={args.batch_size} smoothing={smoothing} "
          f"augment={not args.no_augment} seed={args.seed} "
          f"device={'xla' if is_xla else args.device}", flush=True)

    # ------------------------------------------------------------ train
    step_ms = []
    first_step_ms = None
    epochs_out = []
    t_train0 = time.perf_counter()
    global_step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t_ep = time.perf_counter()
        running = 0.0
        for x, y in train_loader:
            t0 = time.perf_counter()
            x = x.to(device)
            y = y.to(device)
            if flat:
                x = x.flatten(1)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            if sched is not None:
                sched.step()
            sync()
            ms = (time.perf_counter() - t0) * 1e3
            if first_step_ms is None:
                first_step_ms = ms
            else:
                step_ms.append(ms)
            running += loss.item()
            global_step += 1
        ep_wall = time.perf_counter() - t_ep

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                if flat:
                    x = x.flatten(1)
                pred = model(x).argmax(1).cpu()
                correct += (pred == y).sum().item()
        acc = 100.0 * correct / len(test_set)
        epochs_out.append({"epoch": epoch + 1,
                           "train_loss": round(running / steps_per_epoch, 4),
                           "test_acc": round(acc, 2),
                           "wall_s": round(ep_wall, 1)})
        print(f"  epoch {epoch + 1:3d}/{cfg['epochs']}  "
              f"loss {epochs_out[-1]['train_loss']:7.4f}  "
              f"test_acc {acc:5.2f}%  {ep_wall:6.1f}s", flush=True)

    total_wall = time.perf_counter() - t_train0
    steady = sorted(step_ms)[len(step_ms) // 2] if step_ms else None
    best = max(e["test_acc"] for e in epochs_out)
    samples_s = (args.batch_size / (steady / 1e3)) if steady else None

    payload = {
        "dataset": args.dataset, "arch": args.arch, "params": params,
        "config": {**cfg, "batch_size": args.batch_size,
                   "label_smoothing": smoothing,
                   "augment": not args.no_augment, "seed": args.seed,
                   "device": "xla-neuroncore" if is_xla else args.device,
                   "precision": "fp32"},
        "best_test_acc": best,
        "final_test_acc": epochs_out[-1]["test_acc"],
        "epochs": epochs_out,
        "first_step_ms": round(first_step_ms, 1) if first_step_ms else None,
        "median_step_ms": round(steady, 2) if steady else None,
        "compile_s_estimate": round(max(0.0, (first_step_ms - steady) / 1e3), 1)
                              if steady and first_step_ms else None,
        "samples_per_s_steady": round(samples_s, 1) if samples_s else None,
        "total_wall_s": round(total_wall, 1),
        "mlx_reference": "github.com/alpharomercoma/mlx-models (Apple M5)",
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, args.out)
    print(f"best test_acc {best:.2f}%  |  steady {steady:.1f} ms/step  |  "
          f"{samples_s:.0f} samples/s  |  compile ~{payload['compile_s_estimate']}s"
          f"  ->  {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
