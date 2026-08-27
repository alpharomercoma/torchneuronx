#!/usr/bin/env bash
# The QUALITY GATE lane -- runs on BOTH chips, byte-identical.
#
#   BOX=trn1 bash extras/run_quality_gate.sh
#   BOX=trn2 bash extras/run_quality_gate.sh
#
# WHY THIS EXISTS
# ---------------
# Every number this study reports measures SPEED. None of them shows the model
# actually learned. We can say Trainium2 processed tokens 1.92x faster at seq
# 4096; we cannot say it trained the model correctly. A practitioner will ask
# exactly that, and until now the honest answer was "we don't know".
#
# Held-out loss, measured BEFORE and AFTER on rows never trained on, is the
# minimum credible answer. It is deliberately narrow: it supports "the
# fine-tune learned, and comparably on both chips". It does NOT support any
# claim about instruction quality, MMLU, or downstream benchmarks -- Dolly SFT
# is not designed to move those and evaluator noise would dominate.
#
# WHY THE SPLIT SEED IS FIXED AND SHARED
# --------------------------------------
# --holdout-split-seed defaults to 20260805 on both boxes, so both hold out the
# SAME rows. Scoring two chips on different examples would make the comparison
# meaningless. The split happens BEFORE packing, because packing concatenates
# examples and splitting afterwards would leak training content into eval
# sequences.
#
# TWO THINGS THIS RUN ALSO SETTLES
# --------------------------------
# 1. Whether --data-seed changes anything. Three trn1 seeds produced
#    BIT-IDENTICAL loss (tail-50 mean 1.102654, stdev 0.0), so --seed alone
#    does not perturb the trajectory on this stack. data_seed is the last lever
#    before concluding that packing pins the order.
# 2. Whether the trn1/trn2 final-loss gap (1.2063 vs 1.1489) is real. Since
#    runs are deterministic, it is NOT seed noise -- most likely TP=2 vs TP=4
#    changing collective accumulation order. Held-out loss says whether that
#    difference MATTERS or is cosmetic.
#
# SMOKE FIRST. NeuronSFTTrainer.evaluate() is unverified on this stack, and
# doubly so on NeuronCore-v3. A TinyLlama gate costs minutes; discovering it
# mid-8B costs an hour of a non-refundable window.
set -uo pipefail

BOX="${BOX:?set BOX=trn1 or BOX=trn2}"
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/$BOX/results/quality"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"      # libneuronpjrt-path
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"

export NP_DEVICE="$BOX"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
if [ "$BOX" = "trn2" ]; then
  export NP_REGION="${NP_REGION:-sa-east-1}"
  export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"
else
  export NP_REGION="${NP_REGION:-us-west-2}"
  export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache}"
fi

mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ QUALITY[$BOX]: $* ############"; echo; }

bash shared/bin/hf_login.sh >/dev/null 2>&1 || echo "WARN: hf_login failed"

# Parallelism: trn2 takes it from the probe so this lane matches every other
# trn2 lane. trn1 uses its profile default (world 2 / TP 2) exactly as
# published. Symmetry is in the RECIPE, not in the sharding -- the sharding is
# a property of each chip.
EXTRA=()
if [ "$BOX" = "trn2" ]; then
  P="$BENCH_DIR/trn2/results/extras/tp_probe.json"
  [ -s "$P" ] || { echo "FATAL: tp_probe.json missing"; exit 3; }
  read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1])); print(d["lnc"], d["nproc"], d["tp"])' "$P")"
  export NEURON_LOGICAL_NC_CONFIG="$LNC" NP_TELEMETRY_CORES="$NPROC"
  EXTRA=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP")
fi

lane() {   # lane <tag> <args...>
  local tag="$1"; shift
  { have "$OUT/$tag.json" || have "$OUT/$tag.failure.json"; } && { echo "skip $tag"; return 0; }
  step "$tag"
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" "${EXTRA[@]}" --out "$OUT/$tag.json" "$@" \
    > "$OUT/$tag.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}|NotImplementedError[^\"]{0,80}|AttributeError[^\"]{0,80}" "$OUT/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","box":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$BOX" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
      echo "  $tag FAILED (receipt)"; return 1; }
  bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || echo "  push FAILED"
  return 0
}

# ---- 1. Evaluator smoke: does trainer.evaluate() work here at all? ----------
lane quality_smoke --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --holdout-frac 0.05 --max-steps 20 || true

if ! have "$OUT/quality_smoke.json"; then
  echo "evaluator smoke did not produce a result -- refusing to spend an 8B lane on it"
  echo "the receipt stands as the finding: held-out evaluation is unavailable on this stack"
  bash shared/bin/push_results.sh "$BOX" || true
  exit 0
fi
if "$PY" -c "
import json,sys
d=json.load(open('$OUT/quality_smoke.json'))
h=d.get('holdout') or {}
sys.exit(0 if h.get('loss_before') is not None and h.get('loss_after') is not None else 1)" ; then
  echo "evaluator smoke OK -- held-out loss measured before and after"
else
  echo "smoke ran but produced no held-out loss; recording and stopping"
  bash shared/bin/push_results.sh "$BOX" || true
  exit 0
fi

# ---- 2. The paired quality lane, identical recipe on both chips -------------
# Same model, dataset, epochs, LoRA config and split as the published primary
# lane. The ONLY additions are the holdout and the evaluation, so this is
# directly interpretable against lane 4 on the same box.
lane llama31_lora_holdout --model meta-llama/Llama-3.1-8B-Instruct \
  --holdout-frac 0.05 || true

# ---- 3. Does data_seed move anything the plain seed did not? ---------------
# Cheap (20 steps). If the loss trace is STILL bit-identical to quality_smoke,
# packing has pinned the data order and that is a reportable property of the
# stack, not a knob we failed to find.
lane quality_smoke_dataseed --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --holdout-frac 0.05 --max-steps 20 --data-seed 777 || true

step "QUALITY GATE COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh "$BOX" || echo "push FAILED"
