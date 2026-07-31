#!/usr/bin/env bash
# Phase-2 inf2 ROUND 3: the two lanes still outstanding after round 2.
#   F  rag -- overlay v3 (underlay python + PYTHONPATH; v2's venv-from-venv
#      lost torch_neuronx). Old overlay dir is removed so v3 builds clean.
#   C1 spec_decode -- round 2 never re-ran it: its resume guard treats
#      failure receipts as recorded and the round-2 clear didn't take.
#      Receipts are deleted HERE, with verification, then the lane runs.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
banner() { echo; echo "############ PHASE2-INF2-R3: $1 ($(date -u +%H:%M:%SZ)) ############"; echo; }

banner "waiting for round 2"
while pgrep -f "run_phase2_inf2_round2.sh" >/dev/null 2>&1; do sleep 60; done

banner "clearing stale spec_decode receipts + v2 overlay"
rm -f "$BENCH_DIR"/inf2/results/extras/spec_decode/baseline.failure.json \
      "$BENCH_DIR"/inf2/results/extras/spec_decode/fused_spec.failure.json
ls "$BENCH_DIR"/inf2/results/extras/spec_decode/*.failure.json 2>/dev/null \
  && echo "WARN: spec receipts survived the delete -- lane will skip again"
rm -rf /opt/np/venvs/rag-overlay

banner "F: rag (overlay v3)"
bash extras/rag/run_rag.sh || echo "F FAILED (receipts in inf2/results/rag)"

banner "C1: spec_decode"
bash extras/spec_decode.sh || echo "C1 FAILED"

banner "final results push"
bash shared/bin/push_results.sh inf2 || echo "push FAILED"

banner "PHASE2 INF2 ROUND3 COMPLETE"
