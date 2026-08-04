#!/usr/bin/env bash
# Phase-3 MASTER orchestrator for the Trainium2 box. Self-contained and
# disconnect-proof -- launch it once with:
#
#   setsid nohup bash extras/run_phase3_trn2.sh >> /opt/np/phase3_trn2.log 2>&1 &
#
# and reattach from any machine with shared/bin/phase2_status.sh. Completion
# marker: "PHASE3 TRN2 ALL COMPLETE".
#
# ORDER IS THE METHOD, not convenience:
#   1. TP probe on TinyLlama -- cheapest thing that can invalidate everything
#      after it. If no rung of the ladder passes, nothing else may run.
#   2. Warm the NEFF cache from S3 (neuron-cache-v3 -- NEVER the v2 prefix).
#   3. The main suite: the SAME lanes trn1 ran, same hyperparameters. This is
#      the apples-to-apples comparison and it is not tuned.
#   4. Extras: context ladder, checkpoint timing, and only THEN the efficiency
#      levers, so each lever is separately attributable against a baseline that
#      already exists.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"

export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"

echo "############ PHASE3-TRN2: waiting for any in-flight pass ############"
while pgrep -f "bash extras/run_extras_trn2.sh" >/dev/null 2>&1; do sleep 60; done

echo "############ PHASE3-TRN2: HF login ############"
bash shared/bin/hf_login.sh || echo "hf_login FAILED (gated models will 401)"

echo "############ PHASE3-TRN2: TP probe (gates everything below) ############"
bash extras/tp_probe_trn2.sh || echo "tp probe pass returned nonzero (receipt recorded)"
if [ ! -s "trn2/results/extras/tp_probe.json" ]; then
  echo "############ PHASE3 TRN2 HALTED: no TP rung passed ############"
  bash shared/bin/push_results.sh trn2 || echo "push FAILED (push manually)"
  exit 3
fi

echo "############ PHASE3-TRN2: seed NEFF cache from S3 (v3 prefix) ############"
bash shared/bin/sync_neuron_cache.sh pull || echo "cache pull FAILED (cold compile ahead)"

echo "############ PHASE3-TRN2: main suite (same lanes as trn1) ############"
bash trn2/scripts/run_all.sh || echo "main suite FAILED (receipts recorded)"

echo "############ PHASE3-TRN2: extras (ctx ladder, ckpt, NKI, A-parity, levers) ############"
bash extras/run_extras_trn2.sh || echo "extras pass FAILED (receipts recorded)"

# Academic track LAST of the parity work: 6 small lanes (mnist/cifar x
# mlp/cnn/vit) that trn1 ran and trn2 would otherwise be missing. Deliberately
# after the 8B lanes -- these are minutes-scale and must never delay the
# measurement the block was bought for. BOX=trn2 keeps the results out of the
# trn1 set; the runner used to hardcode trn1 in both the path and the push.
echo "############ PHASE3-TRN2: academic track (trn1 parity, 6 lanes) ############"
BOX=trn2 RESULTS_DIR="$BENCH_DIR/trn2/results" \
  bash academic/run_academic.sh || echo "academic pass FAILED (receipts recorded)"

echo "############ PHASE3-TRN2: push cache + results ############"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh trn2 || echo "push FAILED (push manually)"

echo "############ PHASE3 TRN2 ALL COMPLETE ############"
