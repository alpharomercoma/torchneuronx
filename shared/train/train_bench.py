#!/usr/bin/env python3
"""Lane 3: single-GPU training throughput, TFLOP/s and MFU.

A self-contained Llama-3-shaped transformer rather than a HuggingFace model, so
that both machines provably execute the same graph with no version-dependent
modelling code in between. Architecture defaults match Llama 3.1 8B exactly
(32 layers, d_model 4096, 32 heads / 8 KV heads, SwiGLU 14336, RoPE, RMSNorm).

MEMORY CHOICE, and why it matters for fairness:
weights, grads and optimiser state are all BF16. Mixed precision with FP32
master weights would need 16 + 32 + 64 = 112 GB for an 8B model, which fits the
MI300X's 192 GB but not the H200's 141 GB -- so choosing it would hand AMD a
win by configuration rather than by silicon. Pure BF16 costs ~64 GB and fits
both with room for activations, keeping this a compute comparison.

Determinism: the same seed produces the same synthetic batches on both boxes,
so the loss traces must overlay. If they do not, the throughput numbers are
meaningless and the report says so instead of quoting them.

MFU uses the torchtitan/PaLM convention:
    flops_per_token = 6 * N_nonembed + 12 * L * n_heads * head_dim * seqlen
and is multiplied by 4/3 when gradient checkpointing recomputes the forward.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gpuspec  # noqa: E402


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def rope_cache(seqlen, head_dim, device, base=500000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seqlen, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (b, h, s, d)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.hd = d // n_heads
        self.attn_norm = RMSNorm(d)
        self.wq = nn.Linear(d, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d, bias=False)
        self.ffn_norm = RMSNorm(d)
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(b, s, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(h).view(b, s, self.n_kv_heads, self.hd).transpose(1, 2)
        v = self.wv(h).view(b, s, self.n_kv_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, s, -1)
        x = x + self.wo(a)
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Llama(nn.Module):
    def __init__(self, vocab, d, layers, n_heads, n_kv_heads, d_ff, ckpt=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, n_kv_heads, d_ff) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ckpt = ckpt

    def forward(self, idx, cos, sin):
        x = self.tok(idx)
        for blk in self.blocks:
            if self.ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.head(self.norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=14336)
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    key, vendor = gpuspec.detect()
    pk = gpuspec.peak(key)
    dev = torch.device("cuda")
    ckpt = not args.no_checkpoint

    model = Llama(args.vocab, args.d_model, args.layers, args.heads,
                  args.kv_heads, args.d_ff, ckpt=ckpt).to(dev).to(torch.bfloat16)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.tok.weight.numel() + model.head.weight.numel()
    n_nonembed = n_params - n_embed

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1, foreach=True)

    step_fn = model
    if args.compile:
        step_fn = torch.compile(model)

    cos, sin = rope_cache(args.seqlen, args.d_model // args.heads, dev)
    cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    tokens_per_step = args.batch * args.seqlen

    flops_per_token = (6 * n_nonembed
                       + 12 * args.layers * args.heads
                       * (args.d_model // args.heads) * args.seqlen)
    if ckpt:
        flops_per_token *= 4.0 / 3.0   # checkpointing recomputes the forward

    losses, step_ms = [], []
    torch.cuda.reset_peak_memory_stats()

    total = args.warmup + args.steps
    for i in range(total):
        # deterministic synthetic batch: identical on both machines
        idx = torch.randint(0, args.vocab, (args.batch, args.seqlen),
                            generator=gen).to(dev)
        tgt = torch.roll(idx, shifts=-1, dims=1)

        if i == args.warmup:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = step_fn(idx, cos, sin)
        loss = F.cross_entropy(logits.float().view(-1, args.vocab), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        dt = (time.perf_counter() - t0) * 1e3
        lv = loss.item()
        if i >= args.warmup:
            step_ms.append(dt)
        losses.append({"step": i, "loss": round(lv, 6),
                       "ms": round(dt, 2), "warmup": i < args.warmup})
        if i % 5 == 0 or i == total - 1:
            print(f"  step {i:3d}  loss {lv:8.4f}  {dt:8.1f} ms", flush=True)

    step_ms.sort()
    med_ms = step_ms[len(step_ms) // 2]
    tok_s = tokens_per_step / (med_ms * 1e-3)
    tflops = flops_per_token * tokens_per_step / (med_ms * 1e-3) / 1e12
    mfu = 100.0 * tflops / pk["bf16_tflops"]
    peak_gib = torch.cuda.max_memory_allocated() / (1 << 30)

    print(f"\n  median step {med_ms:.1f} ms | {tok_s:,.0f} tok/s | "
          f"{tflops:.1f} TFLOP/s | MFU {mfu:.1f}% | peak {peak_gib:.1f} GiB")

    payload = {
        "gpu": key, "vendor": vendor, "torch": torch.__version__,
        "device_name": gpuspec.device_name(key),
        "config": {
            "layers": args.layers, "d_model": args.d_model, "heads": args.heads,
            "kv_heads": args.kv_heads, "d_ff": args.d_ff, "vocab": args.vocab,
            "seqlen": args.seqlen, "batch": args.batch, "seed": args.seed,
            "compile": args.compile, "grad_checkpoint": ckpt,
            "dtype": "bfloat16", "optimizer": "AdamW(foreach)",
        },
        "params_total": n_params, "params_nonembed": n_nonembed,
        "flops_per_token": flops_per_token,
        "median_step_ms": round(med_ms, 3),
        "tokens_per_s": round(tok_s, 1),
        "tflops": round(tflops, 2),
        "mfu_pct": round(mfu, 2),
        "peak_mem_gib": round(peak_gib, 2),
        "peak_bf16_tflops": pk["bf16_tflops"],
        "loss_trace": losses,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
