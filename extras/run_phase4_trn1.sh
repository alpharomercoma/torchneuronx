#!/usr/bin/env bash
# PHASE 4 on trn1: pretraining + post-training (SFT already banked, DPO, ORPO,
# GRPO/RLVR probe).
#
# Usage:  run_phase4_trn1.sh smoke   # ~1h, derisks every lane, decides budgets
#         run_phase4_trn1.sh full    # the long runs
#
# WHY TWO STAGES
# --------------
# Three of these four lanes are NEW code against an unproven combination
# (TRL trainers re-based onto optimum-neuron's NeuronTrainer). Committing ~25
# hours of paid Trainium to code that has never executed would be a bad trade.
# The smoke stage runs every lane at tiny scale first, so a broken collator or a
# missing generate() costs minutes instead of hours -- and the throughput it
# measures is what sizes the full pretraining budget.
#
# LANE ORDER IS DELIBERATE: cheapest and most certain first, so that if the
# budget runs out the bankable findings are already banked. The GRPO probe leads
# because a documented wall is a complete result in ~15 minutes.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
NP_TAG=PHASE4-TRN1
cd "$BENCH_DIR"

STAGE="${1:-smoke}"
BOX=trn1
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:/opt/aws/neuron/bin:$PATH"   # libneuronpjrt; cost seven lanes in Phase 2
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"

export NP_DEVICE=trn1
export NP_REGION="${NP_REGION:-us-west-2}"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --retry_failed_compilation"
# Stochastic rounding is the documented Neuron recipe for bf16 training without
# fp32 master weights. The pretrain lane's memory budget assumes it.
export NEURON_RT_STOCHASTIC_ROUNDING_EN="${NEURON_RT_STOCHASTIC_ROUNDING_EN:-1}"

OUT="$BENCH_DIR/trn1/results/phase4"
mkdir -p "$OUT"
DATA=/opt/np/data/pretrain/fineweb_edu_1B.uint16.bin

push() { bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || np_log "  push failed"; }

# Every lane runs under telemetry. METHODOLOGY invariant: no telemetry, no
# number -- a throughput figure with no power/utilisation trace beside it is not
# admissible in this study.
lane() {   # lane <tag> <script> [args...]
  local tag="$1"; shift
  if np_recorded "$OUT/$tag"; then np_log "skip $tag (recorded)"; return 0; fi
  np_step "$tag"
  np_wait_for_chip
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$@" > "$OUT/$tag.log" 2>&1 \
    || np_receipt "$OUT/$tag.failure.json" "$tag" "$BOX" "$OUT/$tag.log"
  tail -3 "$OUT/$tag.log" 2>/dev/null
  push
}

np_log "=== PHASE 4 stage=$STAGE on $BOX ==="
[ -s "$DATA" ] || np_log "WARN: $DATA missing -- pretrain lanes will fail fast"

if [ "$STAGE" = smoke ]; then
  # 1. GRPO/RLVR probe. Cheapest complete finding in the whole phase.
  lane grpo_rlvr_probe shared/train/grpo_probe.py \
       --out "$OUT/grpo_rlvr_probe.json" --tag grpo_rlvr_probe \
       --n-samples 64 --max-steps 2

  # 2. Pretraining throughput probe. 60 steps is enough for a steady-state
  #    median once the first-step compile is excluded, and it is what sizes the
  #    full run's token budget.
  lane pretrain_probe shared/train/pretrain_fineweb.py \
       --data "$DATA" --out "$OUT/pretrain_probe.json" --tag pretrain_probe \
       --max-steps 60 --target-tokens 1e12 --max-hours 1.0 --warmup-steps 10

  # 3+4. DPO and ORPO at tiny scale: proves the re-based trainer, the fixed-shape
  #      collator and the implicit reference all work before the real run.
  lane dpo_smoke shared/train/posttrain_align.py --algo dpo \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --out "$OUT/dpo_smoke.json" --tag dpo_smoke \
       --n-samples 32 --max-steps 4 --max-length 512 --max-prompt-length 256

  lane orpo_smoke shared/train/posttrain_align.py --algo orpo \
       --model meta-llama/Llama-3.1-8B \
       --out "$OUT/orpo_smoke.json" --tag orpo_smoke \
       --n-samples 32 --max-steps 4 --max-length 512 --max-prompt-length 256

  np_step "SMOKE COMPLETE -- read pretrain_probe.json tokens_per_s before sizing 'full'"

elif [ "$STAGE" = full ]; then
  # Pretraining is time-boxed rather than step-boxed: --max-hours is the spend
  # bound and the result records which limit actually bound the run.
  lane pretrain_smollm360m shared/train/pretrain_fineweb.py \
       --data "$DATA" --out "$OUT/pretrain_smollm360m.json" \
       --tag pretrain_smollm360m \
       --target-tokens "${NP_PRETRAIN_TOKENS:-1e9}" \
       --max-hours "${NP_PRETRAIN_HOURS:-8}"

  lane dpo_llama31_8b shared/train/posttrain_align.py --algo dpo \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --out "$OUT/dpo_llama31_8b.json" --tag dpo_llama31_8b \
       --n-samples "${NP_PREF_SAMPLES:-4000}" --max-length 1024

  lane orpo_llama31_8b shared/train/posttrain_align.py --algo orpo \
       --model meta-llama/Llama-3.1-8B \
       --out "$OUT/orpo_llama31_8b.json" --tag orpo_llama31_8b \
       --n-samples "${NP_PREF_SAMPLES:-4000}" --max-length 1024

  np_step "FULL COMPLETE"
else
  np_die "unknown stage '$STAGE' (want: smoke | full)" 2
fi

push
np_log "=== PHASE 4 stage=$STAGE done ==="
ls -la "$OUT"
