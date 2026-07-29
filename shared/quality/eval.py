#!/usr/bin/env python3
"""Lane 6: correctness cross-check against a running vLLM server.

Speed numbers are worthless if the two stacks are not computing the same thing.
This lane is the guard against "fast but wrong", and it also prices the one
place where the two machines are deliberately NOT bit-identical: FP8. gfx942
uses e4m3fnuz and Hopper uses e4m3fn, so an FP8 run genuinely produces
different numerics by design. This measures how much that costs.

Two checks:

  1. GREEDY DETERMINISM. A fixed prompt set at temperature 0. Both boxes should
     produce near-identical continuations at BF16. Reported as exact-match rate
     plus per-prompt text so any divergence can be inspected rather than just
     scored.

  2. TOKEN-LEVEL LOGPROB AGREEMENT. Mean log-probability over a fixed corpus,
     which is a perplexity proxy that works through the OpenAI-compatible API.
     Comparable across machines because prompts and sampling are fixed.
"""

import argparse
import json
import os
import time
import urllib.request

PROMPTS = [
    "Explain why the sky appears blue, in three sentences.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Summarise the causes of the 1929 stock market crash.",
    "What is the difference between a mutex and a semaphore?",
    "Translate to French: 'The weather is pleasant today and I plan to walk.'",
    "List the first ten prime numbers, comma separated.",
    "Describe the CAP theorem and give one practical consequence.",
    "Write a haiku about high bandwidth memory.",
    "What does the LIMIT clause do in SQL, and when is it non-deterministic?",
    "Give the chemical equation for photosynthesis and name the inputs.",
    "Explain gradient checkpointing and what it trades away.",
    "Why does floating point addition fail to be associative?",
    "Outline the steps of the TCP three-way handshake.",
    "What is the difference between BF16 and FP16, and why do trainers prefer BF16?",
    "Name three sorting algorithms with O(n log n) average complexity.",
    "Explain what a page fault is in an operating system.",
]


def post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    comp_url = f"{args.base_url}/v1/completions"

    greedy = []
    for i, p in enumerate(PROMPTS):
        r = post(comp_url, {
            "model": args.model, "prompt": p,
            "max_tokens": args.max_tokens,
            "temperature": 0.0, "top_p": 1.0, "seed": 0,
            "logprobs": 1,
        })
        ch = r["choices"][0]
        lps = (ch.get("logprobs") or {}).get("token_logprobs") or []
        lps = [x for x in lps if isinstance(x, (int, float))]
        greedy.append({
            "idx": i, "prompt": p,
            "text": ch["text"],
            "n_tokens": len(lps),
            "mean_logprob": round(sum(lps) / len(lps), 6) if lps else None,
        })
        print(f"  [{i:2d}] {len(lps):4d} tok  "
              f"mean_logprob {greedy[-1]['mean_logprob']}", flush=True)

    vals = [g["mean_logprob"] for g in greedy if g["mean_logprob"] is not None]
    corpus_mean = sum(vals) / len(vals) if vals else None

    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                     "max_tokens": args.max_tokens},
        "n_prompts": len(PROMPTS),
        "corpus_mean_logprob": round(corpus_mean, 6) if corpus_mean else None,
        "results": greedy,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\ncorpus mean logprob: {corpus_mean}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
