#!/usr/bin/env python3
"""Track F / F5: one RAG query, every stage timed.

embed(query) -> pgvector cosine top-k (k=12) -> rerank to top-3 -> prompt the
LLM at localhost:8000 with the contexts -> answer.

Per-stage timings JSON on STDOUT (stdout carries ONLY the JSON; logs go to
stderr so probes.py can parse the output):

  {question, embed_ms, retrieve_ms, rerank_ms, ttft_ms, llm_ms, e2e_ms,
   answer, ids, retrieved_ids, contexts, reranked, rerank_note, ...}

Degrades, declared, never silent:
  * reranker artifact absent (compile receipt / OK marker missing) -> skip
    rerank, top-3 by cosine, rerank_note says why, rerank_ms null;
  * --no-llm -> retrieval-only pass: answer/ttft_ms/llm_ms null.

Retrieval SQL is injection-safe by construction: the only value interpolated
is the pgvector literal, which this code builds from floats -- the user's
question text never touches SQL, it is only embedded.

LLM access is the same stdlib streaming pattern as
shared/serve/fallback_client.py (TTFT = first streamed content chunk).
"""
import argparse
import csv
import http.client
import io
import json
import os
import subprocess
import sys
import time
from urllib.parse import urlparse

from compile_models import (DEFAULT_MODELS_DIR, EMBED_DIRNAME, OK_MARKER,
                            RERANK_BATCH, RERANK_DIRNAME, RERANK_SEQ)
from embed_lane import DEFAULT_TASK, NeuronEmbedder
from ingest import DEFAULT_DSN, to_pgvector


# ---------------------------------------------------------------- retrieval
def retrieve(dsn, vec_literal, k, timeout=120):
    """pgvector cosine top-k through psql --csv (stdlib csv parses newlines
    inside quoted content fields correctly)."""
    sql = (
        "SELECT id, doc, section, content, "
        f"(embedding <=> '{vec_literal}'::vector) AS distance "
        "FROM chunks "
        f"ORDER BY embedding <=> '{vec_literal}'::vector LIMIT {int(k)};"
    )
    proc = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "--csv", "-c", sql],
        capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"psql retrieval failed: {proc.stderr.decode(errors='replace')[:400]}")
    rows = list(csv.DictReader(io.StringIO(proc.stdout.decode(errors="replace"))))
    for r in rows:
        r["distance"] = float(r["distance"])
        r["cosine_sim"] = round(1.0 - r["distance"], 6)
    return rows


# ------------------------------------------------------------------ reranker
class NeuronReranker:
    """Qwen3-Reranker-0.6B causal-LM scoring, HF model-card pattern.

    The reranker is a plain causal LM: the pair prompt asks it to answer
    "yes"/"no" and the score is P(yes) from log_softmax over the two token
    logits at the last position. Prefix/suffix strings are the model card's,
    verbatim. Left padding so position -1 is always the real last token.
    Static shapes: batch 4 x seq 1024 (compile_models.py).
    """
    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the '
              'requirements based on the Query and the Instruct provided. '
              'Note that the answer can only be "yes" or "no".'
              '<|im_end|>\n<|im_start|>user\n')
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_dir, batch_size=RERANK_BATCH, seq_len=RERANK_SEQ):
        import torch                                    # heavy: lazy
        from transformers import AutoTokenizer

        self._torch = torch
        self.batch_size, self.seq_len = batch_size, seq_len
        # compile_models.py writes a raw trace returning last-position logits
        # (B, vocab) -- see compile_reranker for why optimum is out.
        traced_pt = os.path.join(model_dir, "traced_rerank.pt")
        if os.path.exists(traced_pt):
            self.model = torch.jit.load(traced_pt)
        else:                                           # legacy optimum artifact
            from optimum.neuron import NeuronModelForCausalLM
            self.model = NeuronModelForCausalLM.from_pretrained(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.prefix_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self.true_id = self.tok.convert_tokens_to_ids("yes")
        self.false_id = self.tok.convert_tokens_to_ids("no")
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id

    def score(self, query, docs, task=DEFAULT_TASK):
        """-> [P(yes)] per doc, higher = more relevant."""
        torch = self._torch
        budget = self.seq_len - len(self.prefix_ids) - len(self.suffix_ids)
        seqs = []
        for d in docs:
            body = f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {d}"
            ids = self.tok.encode(body, add_special_tokens=False,
                                  truncation=True, max_length=budget)
            seqs.append(self.prefix_ids + ids + self.suffix_ids)

        scores = []
        for i in range(0, len(seqs), self.batch_size):
            batch = seqs[i:i + self.batch_size]
            n = len(batch)
            batch = batch + [batch[-1]] * (self.batch_size - n)  # static fill
            input_ids = torch.full((self.batch_size, self.seq_len),
                                   self.pad_id, dtype=torch.long)
            attn = torch.zeros((self.batch_size, self.seq_len),
                               dtype=torch.long)
            for j, s in enumerate(batch):
                input_ids[j, -len(s):] = torch.tensor(s, dtype=torch.long)
                attn[j, -len(s):] = 1
            with torch.no_grad():
                # TODO-VERIFY on-box: NxDI-backed NeuronModelForCausalLM may
                # not support a full-sequence forward() (its decoders are
                # built around incremental generate()); if forward rejects
                # this call, the exception is caught in main() and the query
                # degrades to no-rerank with the verbatim error in
                # rerank_note -- that outcome IS the attempt receipt.
                # positional: torch.jit.trace mangles arg names to
                # argument_1/argument_2, so kwargs raise "missing value for
                # argument 'argument_1'" (round-4 receipt)
                out = self.model(input_ids, attn)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out                 # raw trace: already (B, vocab)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            pair = torch.stack(
                [logits[:, self.false_id], logits[:, self.true_id]], dim=1)
            probs = torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].exp()
            scores.extend(probs[:n].tolist())
        return scores


# ---------------------------------------------------------------------- LLM
def build_prompt(question, contexts):
    blocks = "\n\n".join(f"[{i + 1}] ({c['id']})\n{c['content']}"
                         for i, c in enumerate(contexts))
    return (
        "You are answering a question about the neuron-pipelines benchmark "
        "repository using ONLY the context passages below. Quote exact "
        "figures verbatim from the context. If the answer is not in the "
        "context, say \"not in context\".\n\n"
        f"Context passages:\n{blocks}\n\n"
        f"Question: {question}\nAnswer:"
    )


def ask_llm(base_url, model, prompt, max_tokens=256, timeout=600):
    """Streamed /v1/completions; returns (answer, ttft_ms, llm_ms).

    Same SSE parsing as fallback_client.one_request: TTFT is the first
    non-empty streamed text chunk.
    """
    u = urlparse(base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "seed": 0, "stream": True,
    })
    try:
        t0 = time.perf_counter()
        conn.request("POST", "/v1/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read(300)!r}")
        ttft_ms, parts, buf = None, [], b""
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    buf = b""
                    break
                try:
                    text = json.loads(payload)["choices"][0].get("text")
                except (ValueError, KeyError, IndexError):
                    continue
                if not text:
                    continue
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                parts.append(text)
        llm_ms = (time.perf_counter() - t0) * 1000
        if ttft_ms is None:
            raise RuntimeError("stream produced no tokens")
        return "".join(parts).strip(), round(ttft_ms, 2), round(llm_ms, 2)
    finally:
        conn.close()


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="One timed RAG query")
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=12, help="pgvector top-k")
    ap.add_argument("--rerank-k", type=int, default=3, help="contexts kept")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--llm-model", default=None,
                    help="served model name for /v1/completions")
    ap.add_argument("--no-llm", action="store_true",
                    help="retrieval(+rerank)-only timing pass")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--out", default=None, help="also write the JSON here")
    args = ap.parse_args()
    if not args.no_llm and not args.llm_model:
        ap.error("--llm-model is required unless --no-llm")

    t_e2e = time.perf_counter()

    # Stage 0 (not counted in embed_ms): model loads, reported separately.
    t0 = time.perf_counter()
    embedder = NeuronEmbedder(os.path.join(args.models_dir, EMBED_DIRNAME))
    reranker, rerank_note = None, None
    rr_dir = os.path.join(args.models_dir, RERANK_DIRNAME)
    if args.no_rerank:
        rerank_note = "disabled by --no-rerank"
    elif not os.path.exists(os.path.join(rr_dir, OK_MARKER)):
        rerank_note = ("reranker compile receipt absent (see "
                       "compile_rerank.json) -- no-rerank mode, top-3 by "
                       "cosine; declared degrade")
    else:
        try:
            reranker = NeuronReranker(rr_dir)
        except Exception as exc:
            rerank_note = (f"reranker load failed: "
                           f"{type(exc).__name__}: {exc}"[:300]
                           + " -- no-rerank mode")
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"# models loaded in {load_ms:.0f}ms "
          f"(reranker={'yes' if reranker else 'no'})", file=sys.stderr)

    # Stage 1: embed the query (Instruct-wrapped, asymmetric encoding).
    t0 = time.perf_counter()
    qvec = embedder.embed_queries([args.question], task=args.task)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    # Stage 2: pgvector cosine top-k.
    t0 = time.perf_counter()
    rows = retrieve(args.dsn, to_pgvector(qvec), args.k)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    retrieved_ids = [r["id"] for r in rows]

    # Stage 3: rerank to top rerank_k (or declared skip).
    rerank_ms = None
    if reranker and rows:
        try:
            t0 = time.perf_counter()
            scores = reranker.score(args.question,
                                    [r["content"] for r in rows],
                                    task=args.task)
            rerank_ms = (time.perf_counter() - t0) * 1000
            for r, s in zip(rows, scores):
                r["rerank_score"] = round(s, 6)
            rows = sorted(rows, key=lambda r: r["rerank_score"], reverse=True)
        except Exception as exc:
            reranker = None
            rerank_ms = None
            rerank_note = (f"rerank forward failed: "
                           f"{type(exc).__name__}: {exc}"[:300]
                           + " -- fell back to cosine order")
            print(f"# {rerank_note}", file=sys.stderr)
    contexts = rows[:args.rerank_k]

    # Stage 4: LLM answer with contexts (skipped in --no-llm).
    answer, ttft_ms, llm_ms = None, None, None
    if not args.no_llm:
        answer, ttft_ms, llm_ms = ask_llm(
            args.base_url, args.llm_model,
            build_prompt(args.question, contexts),
            max_tokens=args.max_tokens)

    e2e_ms = (time.perf_counter() - t_e2e) * 1000
    payload = {
        "question": args.question,
        "k": args.k,
        "rerank_k": args.rerank_k,
        "embed_ms": round(embed_ms, 2),
        "retrieve_ms": round(retrieve_ms, 2),
        "rerank_ms": round(rerank_ms, 2) if rerank_ms is not None else None,
        "ttft_ms": ttft_ms,
        "llm_ms": llm_ms,
        "e2e_ms": round(e2e_ms, 2),          # includes model-load overhead...
        "model_load_ms": round(load_ms, 2),  # ...which is also broken out
        "answer": answer,
        "ids": [c["id"] for c in contexts],
        "retrieved_ids": retrieved_ids,
        "contexts": [{"id": c["id"], "cosine_sim": c["cosine_sim"],
                      "rerank_score": c.get("rerank_score"),
                      "content": c["content"]} for c in contexts],
        "reranked": bool(reranker),
        "rerank_note": rerank_note,
        "llm_model": args.llm_model,
        "no_llm": args.no_llm,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = json.dumps(payload, indent=2)
    print(out)                       # stdout: JSON only (probes.py contract)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(out + "\n")


if __name__ == "__main__":
    main()
