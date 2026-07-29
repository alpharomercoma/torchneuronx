#!/usr/bin/env python3
"""Streaming serving benchmark client. Stdlib only, on purpose.

This is the PRIMARY bench client on the inference box: inf2.xlarge has
4 vCPU / 16 GiB host RAM, and installing a second full vLLM just to use
`vllm bench serve` as a client would cost more host memory than the server
can spare. This client emits a result JSON whose keys are a superset of what
`vllm bench serve --save-result` writes and exactly what summarize.py's
parse_serve reads, so downstream tooling cannot tell which client produced a
row -- that property is enforced by tests/test_fallback_client.py.

Measurement notes (also in METHODOLOGY "Known limits"):
  * input length is controlled by building a prompt of N common short words
    (~1 token/word for Llama-family tokenizers) -- reported input tokens are
    CONFIGURED, not measured;
  * output tokens are counted as received stream chunks, with max_tokens +
    ignore_eos pinning the true count to --output-len;
  * TPOT = (E2EL - TTFT) / (tokens - 1), the same definition vllm bench uses.

Usage:
    fallback_client.py --model <served-name> --num-prompts 32 \
        --max-concurrency 8 --input-len 1024 --output-len 1024 \
        --out isl1024_osl1024_c8.json          # summary line on stdout
"""
import argparse
import http.client
import json
import math
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

WORDS = ("time year people way day man thing woman life child world school "
         "state family student group country problem hand part place case "
         "week company system program question work government number night "
         "point home water room mother area money story fact month lot right "
         "study book eye job word business issue side kind head house service "
         "friend father power hour game line end member law car city name").split()

PCTS = (50, 90, 99)


def build_prompt(input_len, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(max(1, input_len)))


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def dist(name, vals):
    """-> {mean,median,std,p50,p90,p99}_<name>_ms keys."""
    if not vals:
        return {f"{s}_{name}_ms": None
                for s in ("mean", "median", "std", "p50", "p90", "p99")}
    sv = sorted(vals)
    out = {
        f"mean_{name}_ms": round(statistics.fmean(vals), 3),
        f"median_{name}_ms": round(statistics.median(vals), 3),
        f"std_{name}_ms": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
    }
    for p in PCTS:
        out[f"p{p}_{name}_ms"] = round(pct(sv, p), 3)
    return out


class Bench:
    def __init__(self, args):
        self.args = args
        u = urlparse(args.base_url)
        self.host, self.port = u.hostname, u.port or 80
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.second_buckets = {}
        self.results = []
        self.failed = 0
        self.t0 = None

    def _track(self, delta):
        with self.lock:
            self.active += delta
            self.max_active = max(self.max_active, self.active)

    def _bucket(self, stamp, n=1):
        b = int(stamp - self.t0)
        with self.lock:
            self.second_buckets[b] = self.second_buckets.get(b, 0) + n

    def one_request(self, idx):
        body = json.dumps({
            "model": self.args.model,
            "prompt": build_prompt(self.args.input_len, self.args.seed + idx),
            "max_tokens": self.args.output_len,
            "temperature": 0.0,
            "seed": self.args.seed,
            "ignore_eos": True,
            "stream": True,
        })
        self._track(+1)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        try:
            start = time.perf_counter()
            conn.request("POST", "/v1/completions", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {resp.read(200)!r}")
            ttft = None
            stamps = []
            buf = b""
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
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now
                    stamps.append(now)
                    self._bucket(now)
            end = time.perf_counter()
            if ttft is None:
                raise RuntimeError("stream produced no tokens")
            ntok = len(stamps)
            e2el_ms = (end - start) * 1000
            ttft_ms = (ttft - start) * 1000
            itls = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
            tpot_ms = ((e2el_ms - ttft_ms) / (ntok - 1)) if ntok > 1 else None
            with self.lock:
                self.results.append(
                    {"ttft": ttft_ms, "e2el": e2el_ms, "tpot": tpot_ms,
                     "itls": itls, "ntok": ntok})
        except Exception as exc:  # a failed request is a counted result
            with self.lock:
                self.failed += 1
            print(f"  request {idx} FAILED: {exc}", file=sys.stderr)
        finally:
            self._track(-1)
            conn.close()

    def run(self):
        a = self.args
        print(f"# fallback_client  model={a.model}  prompts={a.num_prompts} "
              f"conc={a.max_concurrency}  isl={a.input_len}  osl={a.output_len} "
              f"seed={a.seed}")
        self.t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.max_concurrency) as pool:
            list(pool.map(self.one_request, range(a.num_prompts)))
        duration = time.perf_counter() - self.t0

        ok = self.results
        total_out = sum(r["ntok"] for r in ok)
        total_in = a.input_len * len(ok)      # configured, not measured
        itl_all = [x for r in ok for x in r["itls"]]
        out = {
            "date": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
            "backend": "fallback_client",
            "model_id": a.model,
            "num_prompts": a.num_prompts,
            "max_concurrency": a.max_concurrency,
            "input_len_configured": a.input_len,
            "output_len_configured": a.output_len,
            "duration": round(duration, 3),
            "completed": len(ok),
            "failed": self.failed,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "output_throughput": round(total_out / duration, 2) if duration else 0,
            "total_token_throughput":
                round((total_in + total_out) / duration, 2) if duration else 0,
            "max_output_tokens_per_s":
                max(self.second_buckets.values()) if self.second_buckets else 0,
            "max_concurrent_requests": self.max_active,
        }
        out.update(dist("ttft", [r["ttft"] for r in ok]))
        out.update(dist("tpot", [r["tpot"] for r in ok if r["tpot"] is not None]))
        out.update(dist("itl", itl_all))
        out.update(dist("e2el", [r["e2el"] for r in ok]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = Bench(args).run()
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    import os
    os.replace(tmp, args.out)

    print(f"out_tok/s={result['output_throughput']}  "
          f"ttft_p50={result['p50_ttft_ms']}ms  p99={result['p99_ttft_ms']}ms  "
          f"tpot_p50={result['p50_tpot_ms']}ms  "
          f"completed={result['completed']}/{result['num_prompts']}  "
          f"duration={result['duration']}s -> {args.out}")
    return 1 if (result["completed"] == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
