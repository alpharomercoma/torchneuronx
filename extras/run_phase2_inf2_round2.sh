#!/usr/bin/env bash
# Phase-2 inf2 ROUND 2: only the lanes whose first attempt died on the two
# environment bugs (venv-bin PATH, missing HF auth) or the optimum-neuron
# class drift -- each now fixed at the source. Everything else from round 1
# (mistral sweep, knee, bisect 4096-8192, fp8-KV) stands and is skipped by
# the lanes' own resume guards.
#
#   F  rag (raw-trace embed/rerank artifacts)
#   C1 spec_decode (HF auth fixed)
#   C2 multitenant (--no-enable-prefix-caching)
#   B1 refinement: bisect at 3072 to bracket the serving cliff to <=1024
#      (2048 serves, 4096 crashes NCC_INLA001)
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
banner() { echo; echo "############ PHASE2-INF2-R2: $1 ($(date -u +%H:%M:%SZ)) ############"; echo; }

banner "waiting for any in-flight phase2 pass"
while pgrep -f "run_phase2_inf2.sh" >/dev/null 2>&1; do sleep 60; done

banner "F: rag"
bash extras/rag/run_rag.sh || echo "F FAILED (receipts in inf2/results/rag)"

banner "C1: spec_decode"
bash extras/spec_decode.sh || echo "C1 FAILED"

banner "C2: multitenant"
bash extras/multitenant.sh || echo "C2 FAILED"

banner "B1 refine: bisect 3072"
BISECT_LENS=3072 bash extras/bisect_ctx.sh || echo "B1 refine FAILED"

banner "final results push"
bash shared/bin/push_results.sh inf2 || echo "push FAILED"

banner "PHASE2 INF2 ROUND2 COMPLETE"
