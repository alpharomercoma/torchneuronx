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

# ---------------------------------------------------------------------------
# Make the probe actually GATE the lanes it claims to gate.
#
# The extras/frontier/opportunistic drivers all read tp_probe.json. The MAIN
# SUITE did not: shared/run_all.sh called sft_lora.py with no overrides, so it
# fell back to the trn2 profile default (world 4 / TP 4) no matter what the
# ladder decided. If the probe had fallen to LNC=1 that is a compiler/runtime
# LNC mismatch (NCC_EARG001); if it fell to TP=2 the headline lane would run on
# half the chip while the report called it a one-chip comparison.
#
# NP_SFT_EXTRA_ARGS is unset on trn1, where it expands to nothing.
# ---------------------------------------------------------------------------
read -r LNC NPROC TP <<<"$("${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}/bin/python" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["lnc"], d["nproc"], d["tp"])' trn2/results/extras/tp_probe.json)"
export NEURON_LOGICAL_NC_CONFIG="$LNC"
export NP_TELEMETRY_CORES="$NPROC"
export NP_SFT_EXTRA_ARGS="--device-profile trn2 --nproc-per-node $NPROC --tensor-parallel-size $TP"
echo "main suite will run at LNC=$LNC world=$NPROC TP=$TP (from tp_probe.json)"
if [ "$NPROC" != "4" ] || [ "$TP" != "4" ]; then
  echo "WARNING: probe did NOT select the full chip (world=$NPROC TP=$TP)."
  echo "         Lanes still run and are still receipts, but the trn1-vs-trn2"
  echo "         headline must be labelled a partial-chip configuration."
fi

echo "############ PHASE3-TRN2: seed NEFF cache from S3 (v3 prefix) ############"
bash shared/bin/sync_neuron_cache.sh pull || echo "cache pull FAILED (cold compile ahead)"

echo "############ PHASE3-TRN2: main suite (same lanes as trn1) ############"
bash trn2/scripts/run_all.sh || echo "main suite FAILED (receipts recorded)"

# ---------------------------------------------------------------------------
# GATE: the primary lane must actually exist before anything optional runs.
#
# shared/run_all.sh isolates each lane with `|| echo FAILED` and exits 0, so a
# dead headline lane used to sail straight through here: extras would run, the
# ALL COMPLETE marker would print, and the opportunistic and frontier passes
# would spend the rest of a non-refundable window on bonus lanes while the one
# measurement the block was BOUGHT for was missing. The triplet rule is the
# gate: no telemetry, no number -- so check all three artifacts, not just JSON.
# ---------------------------------------------------------------------------
primary_ok() {
  local d="$BENCH_DIR/trn2/results/train"
  [ -s "$d/llama31_lora.json" ] && [ -s "$d/llama31_lora.log" ] \
    && [ -s "$d/llama31_lora.telemetry.csv" ]
}
if ! primary_ok; then
  echo "############ PHASE3 TRN2: PRIMARY LANE MISSING ############"
  echo "trn2/results/train/llama31_lora.{json,log,telemetry.csv} incomplete."
  echo "Extras, academic, opportunistic and frontier lanes are SKIPPED: they"
  echo "would consume the remaining paid window while the headline comparison"
  echo "does not exist. Retry the primary lane by hand, or accept the receipt."
  ls -la "$BENCH_DIR/trn2/results/train" 2>/dev/null || true
  bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
  bash shared/bin/push_results.sh trn2 || echo "push FAILED"
  echo "############ PHASE3 TRN2 HALTED (no primary result) ############"
  exit 4
fi
echo "primary lane verified: llama31_lora triplet present"

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

# ---------------------------------------------------------------------------
# Everything above is what the Capacity Block was bought for, and it is now on
# S3. Everything below is spending the remainder of a window that cannot be
# refunded: the trn1 suite measured 5.9 h against a 23.5 h block, so idle time
# is the only way left to waste the money.
#
# Deliberately AFTER the completion marker and after the push, so a failure
# down here can never cost a result from up there. run_opportunistic_trn2.sh
# re-checks the marker itself, hard-stops at 10:00Z (30 min before the deadline
# pusher, an hour before EC2 starts terminating), refuses to start any lane it
# cannot finish, and pushes after every single lane.
# ---------------------------------------------------------------------------
echo "############ PHASE3-TRN2: opportunistic lanes (paid-for remainder) ############"
bash extras/run_opportunistic_trn2.sh || echo "opportunistic pass ended early (by design or receipt)"

# Frontier AFTER opportunistic on purpose. Opportunistic leads with the seed
# repeats on lane 4, and variance on the HEADLINE number protects the report's
# central claim; the frontier lanes are additive capability findings. If the
# window runs short, losing "32B fits" hurts less than publishing n=1.
echo "############ PHASE3-TRN2: frontier lanes (what trn1 cannot do) ############"
bash extras/run_frontier_trn2.sh || echo "frontier pass ended early (by design or receipt)"

# MAXUTIL runs AFTER the efficiency sweeps because it selects its config FROM
# them. Lane 4 answers "same workload, different chip" (1.20x, only 34% of the
# 3.51x peak ratio -- the chip is starved at micro-batch 1). This answers "what
# can the chip actually do", at FULL length so it is comparable rather than
# extrapolated from a 100-step probe. Both numbers go in the report; neither
# replaces the other, and this never touches lane 4's artifacts.
echo "############ PHASE3-TRN2: max-utilisation lane (best-effort config) ############"
bash extras/run_maxutil_trn2.sh || echo "maxutil ended early (by design or receipt)"

echo "############ PHASE3-TRN2: roofline analysis (no device time) ############"
"${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}/bin/python" analysis/roofline.py \
  --roots trn2/results --out trn2/results/roofline.json \
  || echo "roofline analysis FAILED"
bash shared/bin/push_results.sh trn2 || echo "push FAILED"
