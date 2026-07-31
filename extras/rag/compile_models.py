#!/usr/bin/env python3
"""Track F / F2: compile-if-absent the two small Qwen3 models to Neuron.

Targets (sized-down per the co-residency arithmetic in README.md -- the
source repo's own >=48 GB floor cannot fit a 32 GB Inferentia2):

  * Qwen/Qwen3-Embedding-0.6B  -> NeuronModelForEmbedding, batch 8, seq 512,
    TP=1. Per the official optimum-neuron inference tutorial
    (inference_tutorials/qwen_embedding): get_neuron_config() + export() +
    save_pretrained(). Pooling is LAST-TOKEN (model card), applied at
    inference time in embed_lane.NeuronEmbedder, then a slice to 1024 dims
    (Matryoshka; native dim of the 0.6B is already 1024, so the slice is a
    declared contract that also survives a model swap) and L2-normalize.
  * Qwen/Qwen3-Reranker-0.6B   -> NeuronModelForCausalLM, batch 4, seq 1024,
    TP=1. ATTEMPT (validity table: causal yes-logit scoring, attempt-only).
    On failure a structured receipt is written and downstream degrades to
    no-rerank mode -- declared, never silent.

Compile-if-absent: a successful compile drops an OK_MARKER json inside the
output dir; its presence short-circuits the next run (FORCE via --force).
A failed/partial output dir is removed so the next run retries cleanly.

Receipts (always written, one per model, load_failure.json style):
  {model_id, out_dir, status: cached|compiled|failed, wall_s, error, captured}
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_BATCH = 8
EMBED_SEQ = 512
EMBED_DIM = 1024          # Matryoshka slice target == native dim of the 0.6B
EMBED_DIRNAME = "qwen3-embedding-0.6b-b8-s512"

RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANK_BATCH = 4
RERANK_SEQ = 1024
RERANK_DIRNAME = "qwen3-reranker-0.6b-b4-s1024"

OK_MARKER = "COMPILE_OK.json"   # presence == artifact usable; query.py keys off this

DEFAULT_MODELS_DIR = "/opt/np/models/neuron-compiled"


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def compile_embedding(out_dir):
    """Raw torch_neuronx.trace of the embedding trunk -> last_hidden_state.

    v1 tried the optimum-neuron tutorial class; 0.4.3 exports no
    NeuronModelForEmbedding at all (receipt: compile_embed.json in git
    history), and the decoder classes drag in neuronx_distributed, which the
    inference DLAMI does not ship. The AWS-docs canonical encoder path -- a
    plain trace returning hidden states, pooling done by the caller -- has no
    such dependency and is the same fallback that carried CLIP. fp32 on
    purpose: the CLIP lane measured bf16 autocast NaNs on this silicon.
    """
    import torch
    import torch_neuronx
    from transformers import AutoModel, AutoTokenizer

    class Trunk(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask,
                         return_dict=False)
            return out[0]                    # last_hidden_state (B, S, H)

    base = AutoModel.from_pretrained(EMBED_ID).eval()
    base.config.use_cache = False            # no KV outputs in the trace
    dummy = (torch.ones((EMBED_BATCH, EMBED_SEQ), dtype=torch.int64),
             torch.ones((EMBED_BATCH, EMBED_SEQ), dtype=torch.int64))
    traced = torch_neuronx.trace(Trunk(base), dummy,
                                 compiler_args=["--auto-cast", "none"])
    os.makedirs(out_dir, exist_ok=True)
    torch.jit.save(traced, os.path.join(out_dir, "traced_embed.pt"))
    AutoTokenizer.from_pretrained(EMBED_ID).save_pretrained(out_dir)


def compile_reranker(out_dir):
    """Raw trace of the reranker returning ONLY last-position logits (ATTEMPT).

    Same raw-trace rationale as compile_embedding (the optimum decoder path
    needs neuronx_distributed -- absent on this DLAMI; receipt in git
    history). Returning logits[:, -1, :] keeps the output at (B, vocab)
    instead of a multi-GB (B, S, vocab) tensor; query.NeuronReranker only
    reads the yes/no logits at the last position (left padding guarantees
    position -1 is real).
    """
    import torch
    import torch_neuronx
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class LastLogits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask,
                         return_dict=False)
            return out[0][:, -1, :]          # (B, vocab)

    base = AutoModelForCausalLM.from_pretrained(RERANK_ID).eval()
    base.config.use_cache = False
    dummy = (torch.ones((RERANK_BATCH, RERANK_SEQ), dtype=torch.int64),
             torch.ones((RERANK_BATCH, RERANK_SEQ), dtype=torch.int64))
    traced = torch_neuronx.trace(LastLogits(base), dummy,
                                 compiler_args=["--auto-cast", "none"])
    os.makedirs(out_dir, exist_ok=True)
    torch.jit.save(traced, os.path.join(out_dir, "traced_rerank.pt"))
    AutoTokenizer.from_pretrained(RERANK_ID).save_pretrained(out_dir)


def build_one(name, model_id, out_dir, fn, receipts_dir, force):
    """Compile one model with cache short-circuit + structured receipt."""
    receipt_path = os.path.join(receipts_dir, f"compile_{name}.json")
    marker = os.path.join(out_dir, OK_MARKER)

    if os.path.exists(marker) and not force:
        print(f"--- {name}: cached at {out_dir} (marker present) ---")
        write_json(receipt_path, {
            "model_id": model_id, "out_dir": out_dir, "status": "cached",
            "wall_s": 0, "error": None, "captured": utcnow()})
        return True

    # A dir without the marker is a partial artifact from an interrupted
    # run -- remove it so the compile starts clean (rerun-safe).
    if os.path.isdir(out_dir):
        print(f"--- {name}: removing partial dir {out_dir} ---")
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"--- {name}: compiling {model_id} -> {out_dir} ---")
    t0 = time.time()
    try:
        fn(out_dir)
    except BaseException as exc:  # BaseException: compiler SIGKILL surfaces as SystemExit
        wall = round(time.time() - t0, 1)
        err_line = f"{type(exc).__name__}: {exc}"[:500]
        print(f"    {name} FAILED after {wall}s: {err_line}", file=sys.stderr)
        # Structured receipt (launch_vllm load_failure style): status + the
        # verbatim error line, full traceback alongside for the post-mortem.
        write_json(receipt_path, {
            "model_id": model_id, "out_dir": out_dir, "status": "failed",
            "wall_s": wall, "error": err_line, "captured": utcnow()})
        with open(os.path.join(receipts_dir, f"compile_{name}.traceback.txt"),
                  "w") as fh:
            fh.write(traceback.format_exc())
        shutil.rmtree(out_dir, ignore_errors=True)   # never leave a half-artifact
        return False

    wall = round(time.time() - t0, 1)
    receipt = {"model_id": model_id, "out_dir": out_dir, "status": "compiled",
               "wall_s": wall, "error": None, "captured": utcnow()}
    write_json(marker, receipt)          # marker inside the artifact dir
    write_json(receipt_path, receipt)    # copy in the results dir
    print(f"--- {name}: compiled in {wall}s ---")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Compile Qwen3 embedding + reranker (0.6B) for Neuron, if absent")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--receipts-dir", required=True,
                    help="where compile_{embed,rerank}.json receipts land")
    ap.add_argument("--only", choices=["embed", "rerank", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    ok = True
    if args.only in ("embed", "all"):
        ok = build_one("embed", EMBED_ID,
                       os.path.join(args.models_dir, EMBED_DIRNAME),
                       compile_embedding, args.receipts_dir, args.force)
    if args.only in ("rerank", "all"):
        # Reranker failure does NOT fail the pipeline: no-rerank mode is a
        # declared degrade (validity table: attempt-only). The receipt is
        # the record; query.py keys off the missing OK_MARKER.
        build_one("rerank", RERANK_ID,
                  os.path.join(args.models_dir, RERANK_DIRNAME),
                  compile_reranker, args.receipts_dir, args.force)
    return 0 if ok else 1   # nonzero only when the REQUIRED embedder failed


if __name__ == "__main__":
    sys.exit(main())
