#!/usr/bin/env python3
"""Track F / F5: 7 factual probes against the RAG stack.

Mirrors the source repo's verification style (scripts/verify_rag.py: "7
canned probes against the vector store, expect >= 6/7") but with questions
whose ground truth is REPORT.md of THIS repo -- the corpus ingest.py loaded.
Every expected answer below is a verbatim fact from REPORT.md (checked
2026-07-31), so a pass means the whole chain retrieved and surfaced the
right receipt, not that the LLM happened to know something.

Each probe shells out to query.py (stdout = one JSON) and checks the
expected substring case-insensitively:
  * default mode: against the LLM answer (end-to-end pass);
  * --no-llm mode: against the concatenated top-3 retrieved contexts
    (retrieval-hit pass -- doubles as the driver's no-LLM timing lane).

Output: pass/fail table on stdout + JSON to --out
  {mode, probes: [{q, expect_any, pass, ...timings}], passed, total, ...}

Exit code: 0 when every probe EXECUTED (a failed check is a recorded
outcome, not a crash); nonzero only when query.py itself errored.
"""
import argparse
import json
import os
import subprocess
import sys
import time

QUERY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "query.py")

# expect_any: pass if ANY variant appears (case-insensitive) -- number
# formatting ("2,952" vs "2952") must not fail a correct answer.
PROBES = [
    {"q": "What MFU did the Llama 3.1 8B LoRA fine-tune reach?",
     "expect_any": ["68.3"]},
    {"q": ("How long did the first-ever cold boot of the Llama 8B serving "
           "config take, including compilation?"),
     "expect_any": ["39.5", "2,372", "2372"]},
    {"q": ("Which compiler error code crashed long-context serving at "
           "max-model-len 9216 and 10240?"),
     "expect_any": ["ncc_inla001"]},
    {"q": "Which EC2 instance type was used as the serving box?",
     "expect_any": ["inf2.xlarge"]},
    {"q": ("What throughput retention was measured over 30 minutes of "
           "sustained serving load?"),
     "expect_any": ["100.4"]},
    {"q": "Which dataset was the Llama 3.1 8B model fine-tuned on?",
     "expect_any": ["dolly-15k", "dolly 15k", "dolly"]},
    {"q": ("How many tokens per second did the Llama 3.1 8B LoRA training "
           "run sustain?"),
     "expect_any": ["2,952", "2952"]},
]


def probe_pass(text, expect_any):
    """Case-insensitive substring check against any accepted variant. Pure."""
    t = (text or "").lower()
    return any(e.lower() in t for e in expect_any)


def run_probe(probe, args):
    cmd = [sys.executable, QUERY_SCRIPT,
           "--question", probe["q"],
           "--k", str(args.k), "--rerank-k", str(args.rerank_k),
           "--dsn", args.dsn, "--models-dir", args.models_dir]
    if args.no_llm:
        cmd.append("--no-llm")
    else:
        cmd += ["--llm-model", args.llm_model, "--base-url", args.base_url]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=900)
    wall_ms = (time.perf_counter() - t0) * 1000
    if proc.returncode != 0:
        return {"q": probe["q"], "expect_any": probe["expect_any"],
                "pass": False, "error": proc.stderr.decode(
                    errors="replace")[-400:],
                "probe_wall_ms": round(wall_ms, 1)}
    j = json.loads(proc.stdout.decode(errors="replace"))
    if args.no_llm:
        judged = "\n".join(c["content"] for c in j.get("contexts", []))
    else:
        judged = j.get("answer") or ""
    # SPURIOUS-PASS GUARD.
    #
    # probe_pass judges the WHOLE returned string. A small instruct model that
    # runs past its answer re-emits the prompt and the retrieved passages -- and
    # those passages are, by construction, the ones containing the expected
    # figure. So the matcher can score a pass on the model's own echoed context
    # while the answer proper is wrong. Measured 2026-08-15: Llama-3.2-1B
    # answered "trn1.2xlarge" (wrong) and passed the inf2.xlarge probe because
    # the string appeared in echoed context past character 400 -- past the end
    # of the preview stored below, so the receipt looked self-consistent.
    #
    # `pass` keeps the published convention. `pass_answer_only` judges just the
    # first block, before any echo begins, and `answer_full_chars` exposes the
    # truncation that hid the problem. Where the two disagree, the pass is echo.
    answer_full = j.get("answer") or ""
    answer_proper = answer_full.split("\n\n")[0]
    p_full = probe_pass(answer_full, probe["expect_any"])
    p_proper = probe_pass(answer_proper, probe["expect_any"])
    return {
        "q": probe["q"],
        "expect_any": probe["expect_any"],
        "pass": p_full,
        "pass_answer_only": p_proper,
        "pass_was_echo": bool(p_full and not p_proper),
        "answer_full_chars": len(answer_full),
        "answer_proper": answer_proper[:200] or None,
        "answer": (j.get("answer") or "")[:400] or None,
        "ids": j.get("ids"),
        "reranked": j.get("reranked"),
        "rerank_note": j.get("rerank_note"),
        "timings": {k: j.get(k) for k in
                    ("embed_ms", "retrieve_ms", "rerank_ms",
                     "ttft_ms", "e2e_ms")},
        "probe_wall_ms": round(wall_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="7 REPORT.md-grounded RAG probes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-llm", action="store_true",
                    help="judge on retrieved contexts (retrieval-hit pass)")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--dsn", default=os.environ.get(
        "PG_DSN", "postgresql://np_rag:np_rag@localhost:5432/np_rag"))
    ap.add_argument("--models-dir", default="/opt/np/models/neuron-compiled")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--rerank-k", type=int, default=3)
    args = ap.parse_args()
    if not args.no_llm and not args.llm_model:
        ap.error("--llm-model is required unless --no-llm")

    mode = "retrieval_only" if args.no_llm else "end_to_end"
    print(f"# probes: mode={mode} n={len(PROBES)}")
    results, errored = [], 0
    for i, probe in enumerate(PROBES):
        try:
            r = run_probe(probe, args)
        except Exception as exc:   # timeout etc: recorded, loop continues
            r = {"q": probe["q"], "expect_any": probe["expect_any"],
                 "pass": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        if r.get("error"):
            errored += 1
        results.append(r)
        mark = "PASS" if r["pass"] else ("ERR " if r.get("error") else "FAIL")
        print(f"  [{i + 1}/7] {mark}  expect~'{probe['expect_any'][0]}'  "
              f"{probe['q'][:60]}", flush=True)

    passed = sum(1 for r in results if r["pass"])
    payload = {
        "mode": mode,
        "llm_model": args.llm_model,
        "probes": results,
        "passed": passed,
        "total": len(PROBES),
        "pass_rate": round(passed / len(PROBES), 3),
        "errored": errored,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"{passed}/{len(PROBES)} passed ({mode}) -> {args.out}")
    return 1 if errored else 0


if __name__ == "__main__":
    sys.exit(main())
