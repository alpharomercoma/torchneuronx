#!/usr/bin/env bash
# Phase-2 MASTER orchestrator for inf2 -- chains every remaining inf2 track in
# the planned order, each sub-driver resumable and failure-tolerant, so ONE
# nohup of this script survives operator disconnects end-to-end:
#
#   A  (run_extras_inf2.sh: parity receipts + mistral serve)
#   D  (knee_search.sh: Poisson capacity knee)
#   B1 (bisect_ctx.sh: long-context cliff)
#   B2 (quant_lane.sh: fp8 KV / int8 weights)
#   F  (rag/run_rag.sh: pg+pgvector RAG appliance, overlay venv)
#   C1 (spec_decode.sh) -> C2 (multitenant.sh)
#   B3 gpt-oss-20b: NOT BUILT YET -- logged, not silently skipped.
#
# If a Track-A driver launched earlier is still running, we WAIT for it
# rather than double-book the NeuronCores.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"

banner() { echo; echo "############ PHASE2-INF2: $1 ($(date -u +%H:%M:%SZ)) ############"; echo; }

banner "waiting for any pre-existing run_extras_inf2 pass"
while pgrep -f "bash extras/run_extras_inf2.sh" >/dev/null 2>&1; do sleep 60; done

banner "A: run_extras_inf2 (resumable; done lanes skip)"
bash extras/run_extras_inf2.sh || echo "A driver FAILED (individual receipts recorded)"

banner "D: knee_search"
bash extras/knee_search.sh || echo "D FAILED"

banner "B1: bisect_ctx"
bash extras/bisect_ctx.sh || echo "B1 FAILED"

banner "B2: quant_lane"
bash extras/quant_lane.sh || echo "B2 FAILED"

banner "F: rag"
bash extras/rag/run_rag.sh || echo "F FAILED (receipts in inf2/results/rag)"

banner "C1: spec_decode"
bash extras/spec_decode.sh || echo "C1 FAILED"

banner "C2: multitenant"
bash extras/multitenant.sh || echo "C2 FAILED"

banner "B3: gpt-oss-20b MXFP4 -- driver not built yet; declared TODO, no receipt faked"

banner "final results push"
bash shared/bin/push_results.sh inf2 || echo "push FAILED (push manually)"

banner "PHASE2 INF2 ALL COMPLETE"
