#!/usr/bin/env python3
"""Stream one completion against a warm endpoint, showing token cadence live.

Stdlib only, so it runs from any laptop or box with no installs. Prints each
token as it arrives, then a pasteable summary line. The point is to make TTFT
and TPOT *visible* rather than quoted.

Usage:
    python3 live_ttft.py --base-url http://localhost:8000 \
        --model meta-llama/Llama-3.1-8B-Instruct            # TTFT=~xxx ms live
    python3 live_ttft.py --prompt "Explain KV caches simply." --max-tokens 200
"""
import argparse
import json
import sys
import time
import urllib.request

DEFAULT_PROMPT = ("In three short sentences, explain why ahead-of-time "
                  "compilation changes how you deploy an LLM.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"# live_ttft  model={args.model}  max_tokens={args.max_tokens} "
          f"seed={args.seed}\n# prompt: {args.prompt!r}\n")

    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": args.seed,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    stamps = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"]
            except (ValueError, KeyError, IndexError):
                continue
            tok = delta.get("content")
            if not tok:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - t0) * 1000
                print(f"[first token after {ttft:.0f} ms]\n")
            stamps.append(now)
            sys.stdout.write(tok)
            sys.stdout.flush()

    if ttft is None:
        print("no tokens received", file=sys.stderr)
        return 1
    n = len(stamps)
    total_ms = (stamps[-1] - t0) * 1000
    tpot = ((stamps[-1] - stamps[0]) / (n - 1) * 1000) if n > 1 else 0.0
    print(f"\n\nTTFT={ttft:.0f}ms  TPOT={tpot:.1f}ms  tokens={n}  "
          f"total={total_ms / 1000:.1f}s  ({1000 / tpot:.1f} tok/s steady)"
          if tpot else f"\n\nTTFT={ttft:.0f}ms  tokens={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
