#!/usr/bin/env python3
"""Track F / F2 sub-lane: standalone embedding throughput + the shared embedder.

Measures embeddings/s at batch 1 and batch 8 over 200 short deterministic
texts, plus p50/p99 per-call latency, and asserts the 1024-dim contract.
This is the survivor of the old flat "A4 embeddings lane" (plan Track F note).

Also home of NeuronEmbedder, imported by ingest.py and query.py -- one
implementation of tokenize -> forward -> last-token pool -> Matryoshka slice
-> L2-normalize, so the ingest-side and query-side vectors are computed by
the same code path (asymmetric-encoding bugs are a classic RAG failure).

Pooling per the official optimum-neuron qwen_embedding tutorial and the
Qwen3-Embedding model card: LAST-token pooling with right padding, then
normalize. Matryoshka: slice to the first 1024 dims BEFORE normalizing
(MRL contract); for the 0.6B model 1024 is the native width, so the slice
is a no-op guard that keeps the code correct if a wider model is swapped in.

Static shapes: the artifact is compiled at batch 8 x seq 512, so every call
pads to exactly that shape. "Batch 1" therefore measures one useful item per
batch-8 graph invocation -- that is the honest per-call number on Neuron, and
the batch-8 lane shows what the same graph does when full. Declared here and
in the metrics JSON, not hidden.

Module import stays torch/optimum-free (heavy imports live inside methods)
so the local test gate can import and py_compile this file anywhere.
"""
import argparse
import json
import math
import os
import random
import sys
import time

from compile_models import EMBED_DIM, EMBED_DIRNAME, DEFAULT_MODELS_DIR

# Task instruction for QUERIES only -- Qwen3 embedding asymmetry per model
# card and source repo config (query_instruction): queries get an
# "Instruct: ...\nQuery: ..." wrapper, documents are embedded raw.
DEFAULT_TASK = ("Given a question about the neuron-pipelines benchmark "
                "repository, retrieve documentation passages that answer it.")

WORDS = ("neuron trainium inferentia compile cache serving latency token "
         "throughput adapter merge checkpoint tensor parallel batch sequence "
         "vector index chunk retrieval rerank answer benchmark lane result "
         "telemetry sustained retention quality greedy probe report").split()


def make_texts(n=200, seed=0):
    """Deterministic short texts (8-16 words) for the throughput lane."""
    rng = random.Random(seed)
    return [f"passage {i}: " + " ".join(rng.choice(WORDS)
                                        for _ in range(rng.randint(8, 16)))
            for i in range(n)]


def pct(vals, p):
    if not vals:
        return None
    sv = sorted(vals)
    k = (len(sv) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sv[lo]
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


class NeuronEmbedder:
    """Qwen3-Embedding-0.6B on one NeuronCore, static batch 8 x seq 512."""

    def __init__(self, model_dir, batch_size=8, seq_len=512, dim=EMBED_DIM):
        import torch                                   # heavy: lazy on purpose
        import torch.nn.functional as F
        from transformers import AutoTokenizer

        self._torch, self._F = torch, F
        self.batch_size, self.seq_len, self.dim = batch_size, seq_len, dim
        # compile_models.py writes a raw torch_neuronx trace (traced_embed.pt)
        # -- optimum-neuron 0.4.3 has no embedding class (receipt). The traced
        # module takes (input_ids, attention_mask) positionally and returns
        # last_hidden_state.
        traced_pt = os.path.join(model_dir, "traced_embed.pt")
        if os.path.exists(traced_pt):
            traced = torch.jit.load(traced_pt)
            self.model = lambda input_ids, attention_mask: traced(
                input_ids, attention_mask)
        else:                                          # legacy optimum artifact
            from optimum.neuron import NeuronModelForEmbedding
            self.model = NeuronModelForEmbedding.from_pretrained(model_dir)
        # Right padding: the tutorial's last_token_pool handles either side,
        # but right is what it loads and what our pooling below assumes.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, padding_side="right")

    def _last_token_pool(self, hidden, attention_mask):
        # Tutorial code, right-padding branch: index of last real token.
        torch = self._torch
        seq_lens = attention_mask.sum(dim=1) - 1
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[idx, seq_lens]

    def embed(self, texts):
        """-> list of 1024-dim L2-normalized python float lists."""
        torch, F = self._torch, self._F
        out_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            n = len(batch)
            # Static batch: fill short batches by repeating the last text,
            # slice the fills back off after pooling.
            batch = batch + [batch[-1]] * (self.batch_size - n)
            toks = self.tokenizer(batch, padding="max_length", truncation=True,
                                  max_length=self.seq_len, return_tensors="pt")
            # Keep only what the traced graph was compiled with.
            inputs = {k: v for k, v in toks.items()
                      if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                out = self.model(**inputs)
            # TODO-VERIFY on-box: tutorial passes model output straight to
            # last_token_pool (i.e. it is the hidden-states tensor); accept
            # the object/tuple forms too in case the wrapper differs.
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out[0] if isinstance(out, (tuple, list)) else out
            pooled = self._last_token_pool(hidden, inputs["attention_mask"])
            emb = pooled[:, :self.dim].float()      # Matryoshka slice, then...
            emb = F.normalize(emb, p=2, dim=1)      # ...L2-normalize (MRL order)
            out_vecs.extend(emb[:n].tolist())
        return out_vecs

    def embed_queries(self, queries, task=DEFAULT_TASK):
        """Instruct-wrapped query encoding (asymmetric, per model card)."""
        return self.embed([f"Instruct: {task}\nQuery: {q}" for q in queries])


def timed_lane(embedder, texts, call_batch):
    """Embed `texts` in calls of `call_batch`; return metrics dict."""
    lat_ms = []
    t_lane = time.perf_counter()
    for i in range(0, len(texts), call_batch):
        t0 = time.perf_counter()
        embedder.embed(texts[i:i + call_batch])
        lat_ms.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - t_lane
    return {
        "call_batch": call_batch,
        "calls": len(lat_ms),
        "embeddings_per_s": round(len(texts) / wall, 2) if wall else None,
        "p50_ms": round(pct(lat_ms, 50), 2),
        "p99_ms": round(pct(lat_ms, 99), 2),
        "wall_s": round(wall, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Embedding throughput lane")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-texts", type=int, default=200)
    args = ap.parse_args()

    model_dir = os.path.join(args.models_dir, EMBED_DIRNAME)
    texts = make_texts(args.n_texts)

    t0 = time.perf_counter()
    embedder = NeuronEmbedder(model_dir)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"# embedder loaded from {model_dir} in {load_s}s", file=sys.stderr)

    # Contract check first: dim == 1024 and unit norm, or the lane dies loudly.
    probe = embedder.embed([texts[0]])[0]
    assert len(probe) == EMBED_DIM, f"dim {len(probe)} != {EMBED_DIM}"
    norm = math.sqrt(sum(x * x for x in probe))
    assert abs(norm - 1.0) < 1e-3, f"not L2-normalized (|v|={norm})"

    lanes = {"batch1": timed_lane(embedder, texts, 1),
             "batch8": timed_lane(embedder, texts, embedder.batch_size)}
    payload = {
        "model_dir": model_dir,
        "dim": EMBED_DIM,
        "n_texts": len(texts),
        "seq_len": embedder.seq_len,
        "compiled_batch": embedder.batch_size,
        "load_s": load_s,
        "note": ("static shapes: batch1 = 1 useful item per batch-8 graph "
                 "call (padded); batch8 = same graph, full"),
        "lanes": lanes,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    for name, lane in lanes.items():
        print(f"{name}: {lane['embeddings_per_s']} emb/s  "
              f"p50={lane['p50_ms']}ms p99={lane['p99_ms']}ms")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
