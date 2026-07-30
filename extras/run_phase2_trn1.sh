#!/usr/bin/env bash
# Phase-2 MASTER orchestrator for trn1: wait for any in-flight extras pass,
# then run the extras driver ONCE MORE so lanes unlocked since that pass
# started (fixed NKI simulate -> device benchmark; parity lanes) execute.
# Completed lanes skip; one nohup survives operator disconnects.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"

echo "############ PHASE2-TRN1: waiting for any in-flight extras pass ############"
while pgrep -f "bash extras/run_extras_trn1.sh" >/dev/null 2>&1; do sleep 60; done

echo "############ PHASE2-TRN1: final extras pass ############"
bash extras/run_extras_trn1.sh || echo "extras pass FAILED (receipts recorded)"

echo "############ PHASE2-TRN1: results push ############"
bash shared/bin/push_results.sh trn1 || echo "push FAILED (push manually)"

echo "############ PHASE2 TRN1 ALL COMPLETE ############"
