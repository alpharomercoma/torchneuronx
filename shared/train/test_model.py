#!/usr/bin/env python3
"""CPU self-test for the training model. No GPU required.

Run this before burning GPU hours on the training lane:

    python3 shared/train/test_model.py

It exercises the parts that are easy to get silently wrong -- GQA head
expansion, RoPE broadcasting, gradient flow through gradient checkpointing,
and seed determinism -- on a tiny config where a mistake is obvious. A shape
bug found here costs seconds; the same bug found on the box costs an hour of
rented H200 time.

Determinism is the important one: the whole training comparison rests on both
machines consuming identical batches so their loss traces can be overlaid. If
the same seed does not reproduce the same loss here, that premise is broken
before either GPU is involved.
"""

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_bench", os.path.join(_here, "train_bench.py"))
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)

# tiny stand-in for Llama 3.1 8B: same structure, trivial dimensions
V, D, L, H, KV, FF, S, B = 512, 128, 2, 8, 2, 256, 64, 2


def main():
    failures = []

    model = tb.Llama(V, D, L, H, KV, FF, ckpt=True).to(torch.float32)
    model.train()
    cos, sin = tb.rope_cache(S, D // H, torch.device("cpu"))

    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, V, (B, S), generator=gen)
    tgt = torch.roll(idx, -1, 1)

    out = model(idx, cos, sin)
    if tuple(out.shape) != (B, S, V):
        failures.append(f"logits shape {tuple(out.shape)} != {(B, S, V)}")

    loss = F.cross_entropy(out.float().view(-1, V), tgt.view(-1))
    loss.backward()

    n_par = sum(1 for _ in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    if n_grad != n_par:
        failures.append(f"only {n_grad}/{n_par} params received gradients")

    blk = model.blocks[0]
    if blk.n_heads // blk.n_kv_heads != H // KV:
        failures.append("GQA repeat factor wrong")

    # same seed must give the same loss, twice
    def run_once():
        torch.manual_seed(0)
        m = tb.Llama(V, D, L, H, KV, FF, ckpt=False).to(torch.float32)
        m.train()
        g = torch.Generator().manual_seed(0)
        i = torch.randint(0, V, (B, S), generator=g)
        o = m(i, cos, sin)
        return F.cross_entropy(o.float().view(-1, V),
                               torch.roll(i, -1, 1).view(-1)).item()

    a, b = run_once(), run_once()
    if a != b:
        failures.append(f"non-deterministic: {a} != {b}")

    print(f"logits          : {tuple(out.shape)}")
    print(f"loss            : {loss.item():.4f}")
    print(f"gradients       : {n_grad}/{n_par}")
    print(f"GQA             : {blk.n_heads} heads / {blk.n_kv_heads} kv "
          f"(repeat {blk.n_heads // blk.n_kv_heads})")
    print(f"determinism     : {a:.6f} == {b:.6f}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
