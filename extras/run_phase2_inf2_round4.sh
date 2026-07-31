#!/usr/bin/env bash
# Phase-2 inf2 ROUND 4 (final retry round):
#   F  rag -- overlay verify fixed (overlay numpy stripped; check no longer
#      demands PeftAdapterMixin, which genuine transformers 4.57 moved)
#   C1 spec_decode -- local snapshot paths (inference_demo wants dirs, not
#      repo ids) + on-device-sampling probe for the fused path
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
banner() { echo; echo "############ PHASE2-INF2-R4: $1 ($(date -u +%H:%M:%SZ)) ############"; echo; }

banner "waiting for earlier rounds"
while pgrep -f "run_phase2_inf2_round[23].sh" >/dev/null 2>&1; do sleep 60; done

banner "clearing round-3 receipts for the retried lanes"
rm -f inf2/results/extras/spec_decode/baseline.failure.json \
      inf2/results/extras/spec_decode/fused_spec.failure.json \
      inf2/results/rag/rag_venv.failure.json \
      inf2/results/rag/compile_embed.json inf2/results/rag/compile_rerank.json
ls inf2/results/extras/spec_decode/*.failure.json 2>/dev/null && echo "WARN: spec receipts survived"

banner "F: rag"
bash extras/rag/run_rag.sh || echo "F FAILED (receipts in inf2/results/rag)"

banner "C1: spec_decode"
bash extras/spec_decode.sh || echo "C1 FAILED"

banner "final results push"
bash shared/bin/push_results.sh inf2 || echo "push FAILED"

banner "PHASE2 INF2 ROUND4 COMPLETE"
