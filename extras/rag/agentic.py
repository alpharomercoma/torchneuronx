#!/usr/bin/env python3
"""Agentic RAG on one Inferentia2: an iterative retrieve-judge-refine loop.

WHAT MAKES THIS "AGENTIC", AND WHAT DOES NOT
--------------------------------------------
query.py is a straight pipeline: embed -> retrieve -> rerank -> answer, one
pass, no decisions. This adds the smallest honest increment of agency: the LM
decides, between retrievals, whether the context it has is sufficient, and if
not it writes the next search query itself.

    round 1..N:
        embed(search_query)          Qwen3-Embedding-0.6B on nc1
        retrieve(top-k)              pgvector
        rerank -> top-r              Qwen3-Reranker-0.6B on nc1
        JUDGE: sufficient?           the LM on nc0
          yes -> answer and stop
          no  -> LM writes the next search query, loop

This is NOT tool-calling, NOT planning over a tool catalogue, and NOT
multi-agent. Those need a model that reliably emits structured calls; the LM
here is a 1B. Calling this "agentic RAG" without that qualifier would oversell
it, so the receipt records `agency: "iterative retrieve-judge-refine"` and the
report should use that phrase.

WHY IT IS WORTH RUNNING ANYWAY
The judge step is the cheapest possible probe of whether a small on-chip LM can
make a *control-flow* decision, as opposed to just producing prose. A model
whose generation is numerically corrupt cannot do this at all, so the loop
doubles as a sharper test of generation health than an answer-string match: a
corrupt model cannot even emit YES or NO.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compile_models import DEFAULT_MODELS_DIR, EMBED_DIRNAME, RERANK_DIRNAME
from embed_lane import DEFAULT_TASK, NeuronEmbedder
from ingest import DEFAULT_DSN, to_pgvector
from query import NeuronReranker, ask_llm, retrieve

MAX_ROUNDS = 3

JUDGE = (
    "You are checking whether some context passages are enough to answer a "
    "question. Reply with exactly one word: YES or NO.\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\n"
    "Is the context sufficient to answer? Reply YES or NO.\nAnswer:"
)

REFINE = (
    "The context below did NOT answer the question. Write a better search "
    "query to find the missing information. Reply with the search query only, "
    "no explanation.\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\nSearch query:"
)

ANSWER = (
    "Answer the question using ONLY the context passages. Quote exact figures "
    "verbatim. If the answer is not present, say \"not in context\".\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)


def ctx_block(contexts):
    return "\n\n".join(f"[{i + 1}] {c['content'][:700]}"
                       for i, c in enumerate(contexts))


def printable_fraction(s):
    """A corrupt decode shows up as control bytes long before it shows up as a
    wrong answer. Measure it rather than eyeball it."""
    if not s:
        return 0.0
    return sum(ch.isprintable() or ch in "\n\t " for ch in s) / len(s)


def run_question(q, embedder, reranker, args, trace):
    search = q
    seen, contexts = set(), []
    for rnd in range(1, args.max_rounds + 1):
        t0 = time.perf_counter()
        qvec = embedder.embed_queries([search], task=args.task)[0]
        rows = retrieve(args.dsn, to_pgvector(qvec), args.k)
        fresh = [r for r in rows if r["id"] not in seen]
        for r in fresh:
            seen.add(r["id"])
        if reranker is not None and fresh:
            scored = reranker.score(search, fresh, task=args.task)
            fresh = [d for d, _ in sorted(zip(fresh, scored),
                                          key=lambda p: -p[1])][:args.rerank_k]
        contexts = (contexts + fresh)[: args.rerank_k * 2]
        retrieve_ms = (time.perf_counter() - t0) * 1000

        block = ctx_block(contexts)
        verdict, _, judge_ms = ask_llm(
            args.base_url, args.llm_model,
            JUDGE.format(ctx=block, q=q), max_tokens=6)
        sufficient = bool(re.search(r"\byes\b", verdict, re.I))
        trace.append({
            "round": rnd, "search_query": search,
            "new_chunks": [r["id"] for r in fresh],
            "retrieve_ms": round(retrieve_ms, 1),
            "judge_raw": verdict[:120],
            "judge_printable": round(printable_fraction(verdict), 3),
            "judge_decision": "sufficient" if sufficient else "insufficient",
            "judge_ms": round(judge_ms, 1),
        })
        if sufficient or rnd == args.max_rounds:
            break
        nxt, _, refine_ms = ask_llm(
            args.base_url, args.llm_model,
            REFINE.format(ctx=block, q=q), max_tokens=32)
        nxt = nxt.strip().splitlines()[0][:200] if nxt.strip() else ""
        trace[-1]["refine_raw"] = nxt[:120]
        trace[-1]["refine_ms"] = round(refine_ms, 1)
        # A corrupt or empty refinement must not silently re-search the same
        # thing and look like convergence.
        if not nxt or printable_fraction(nxt) < 0.9:
            trace[-1]["refine_rejected"] = "unusable refinement; loop stopped"
            break
        search = nxt

    answer, ttft_ms, llm_ms = ask_llm(
        args.base_url, args.llm_model,
        ANSWER.format(ctx=ctx_block(contexts), q=q), max_tokens=args.max_tokens)
    return {
        "question": q, "rounds": len(trace), "answer": answer,
        "answer_printable": round(printable_fraction(answer), 3),
        "ttft_ms": round(ttft_ms, 1), "llm_ms": round(llm_ms, 1),
        "context_ids": [c["id"] for c in contexts],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--rerank-k", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    questions = [
        "What MFU did the Llama 3.1 8B LoRA fine-tune reach?",
        "Which dataset was the Llama 3.1 8B model fine-tuned on?",
        "Which EC2 instance type was used as the serving box?",
    ]

    embedder = NeuronEmbedder(os.path.join(args.models_dir, EMBED_DIRNAME))
    reranker = None
    try:
        reranker = NeuronReranker(os.path.join(args.models_dir, RERANK_DIRNAME))
    except Exception as exc:                      # reranker is optional
        print(f"reranker unavailable, continuing without it: {exc}")

    results, t0 = [], time.perf_counter()
    for q in questions:
        trace = []
        try:
            r = run_question(q, embedder, reranker, args, trace)
        except Exception as exc:
            r = {"question": q, "error": f"{type(exc).__name__}: {exc}"[:300]}
        r["trace"] = trace
        results.append(r)
        print(f"[{len(results)}/{len(questions)}] rounds={r.get('rounds')} "
              f"printable={r.get('answer_printable')} :: {q[:48]}")

    ok = [r for r in results if r.get("answer_printable", 0) > 0.98]
    out = {
        "lane": "agentic_rag", "box": "inf2",
        "agency": "iterative retrieve-judge-refine (NOT tool-calling or planning)",
        "llm_model": args.llm_model,
        "embedder": "Qwen/Qwen3-Embedding-0.6B (nc1)",
        "reranker": ("Qwen/Qwen3-Reranker-0.6B (nc1)" if reranker else "disabled"),
        "topology": "LLM nc0 tp=1 | encoders nc1",
        "max_rounds": args.max_rounds,
        "questions": len(questions),
        "answers_printable": len(ok),
        "wall_s": round(time.perf_counter() - t0, 1),
        "results": results,
        "note": ("answers_printable counts answers that are >98% printable "
                 "characters. It measures decode health, NOT answer accuracy: "
                 "a fluent wrong answer counts as printable."),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nagentic: {len(ok)}/{len(questions)} answers printable -> {args.out}")


if __name__ == "__main__":
    main()
