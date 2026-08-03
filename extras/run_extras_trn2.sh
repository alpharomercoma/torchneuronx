#!/usr/bin/env bash
# trn2-side Phase-3 lanes: the context ladder (B4) that trn1 could not clear,
# and checkpoint save/restore timing (C4). Same resumable discipline as the
# trn1 driver it is cloned from.
#
# The ladder is the headline. On trn1 (16 GiB per NeuronCore-v2) seq 4096
# passed at 82.7% MFU and seq 8192 died with:
#   NCC_EOOM002] Maximum peak HBM usage of 18.12GB exceeds HBM limit of 16.00GB
# Trainium2 gives 24 GiB per logical core, so 8192 should now clear -- and 16384
# is added to find the NEW cliff rather than merely confirming the old one moved.
# Whichever way it lands, the outcome is a receipt.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$BENCH_DIR/trn2/results/extras"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"   # libneuronpjrt-path (Phase-1 gotcha #2)
PY="$NP_VENV/bin/python"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"
bash "$BENCH_DIR/shared/bin/hf_login.sh" >/dev/null 2>&1 \
  || echo "WARN: hf_login failed (gated models will 401)"
TELEM="$BENCH_DIR/shared/telemetry.py"
mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ $* ############"; echo; }

# ------------------------------------------- parallelism decided by the probe
# Reading it rather than hardcoding is the point: no lane may silently run at a
# different TP than the one the probe validated and the report declares.
if [ ! -s "$OUT/tp_probe.json" ]; then
  echo "FATAL: $OUT/tp_probe.json missing -- run extras/tp_probe_trn2.sh first" >&2
  exit 3
fi
read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["lnc"], d["nproc"], d["tp"])' "$OUT/tp_probe.json")"
export NEURON_LOGICAL_NC_CONFIG="$LNC"
export NP_TELEMETRY_CORES="$NPROC"
TP_ARGS=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP")
echo "trn2 parallelism (from tp_probe.json): LNC=$LNC world=$NPROC TP=$TP"

step "B4: training context ladder (20-step probes) -- 24 GiB/core vs trn1's 16"
for SL in 4096 8192 16384; do
  T="$OUT/ctx_${SL}.json"
  { have "$T" || have "$OUT/ctx_${SL}.failure.json"; } && { echo "skip ctx $SL (recorded)"; continue; }
  "$PY" "$TELEM" --out "$OUT/ctx_${SL}.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
      --model meta-llama/Llama-3.1-8B-Instruct --tag "ctx_${SL}" \
      "${TP_ARGS[@]}" --seq-len "$SL" --max-steps 20 --out "$T" \
    > "$OUT/ctx_${SL}.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/ctx_${SL}.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/ctx_${SL}.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"seq_len":%s,"status":"failed","reason":"%s","captured":"%s"}\n' \
        "$SL" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/ctx_${SL}.failure.json"
      echo "  ctx $SL FAILED (receipt)"; }
done

step "C4: checkpoint save/restore timing (TinyLlama, 30 steps, save every 10)"
have "$OUT/ckpt_timing.json" || "$PY" "$TELEM" --out "$OUT/ckpt_timing.telemetry.csv" -- \
  "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tag ckpt_timing \
    "${TP_ARGS[@]}" --max-steps 30 --save-steps 10 --out "$OUT/ckpt_timing.json" \
  > "$OUT/ckpt_timing.log" 2>&1 || echo "  ckpt timing FAILED"

# ------------------------------------------------------ efficiency levers (E1/E2)
# Deliberately AFTER the baseline lanes. Each is a separate declared lane with
# its own triplet, never an edit to the primary llama31_lora lane -- a tuned
# trn2 measured against an untuned trn1 would not be a comparison.
step "E1: seq 4096 as the efficient operating point (full 8B lane, 100 steps)"
have "$OUT/eff_seq4096.json" || "$PY" "$TELEM" --out "$OUT/eff_seq4096.telemetry.csv" -- \
  "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
    --model meta-llama/Llama-3.1-8B-Instruct --tag eff_seq4096 \
    "${TP_ARGS[@]}" --seq-len 4096 --max-steps 100 --out "$OUT/eff_seq4096.json" \
  > "$OUT/eff_seq4096.log" 2>&1 || echo "  eff_seq4096 FAILED"

step "E2: recompute OFF (24 GiB/core may make activation checkpointing unnecessary)"
have "$OUT/eff_norecompute.json" || "$PY" "$TELEM" --out "$OUT/eff_norecompute.telemetry.csv" -- \
  "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
    --model meta-llama/Llama-3.1-8B-Instruct --tag eff_norecompute \
    "${TP_ARGS[@]}" --max-steps 100 --no-gradient-checkpointing \
    --out "$OUT/eff_norecompute.json" \
  > "$OUT/eff_norecompute.log" 2>&1 || {
    REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/eff_norecompute.log" | head -1 | sed "s/\"/'/g")
    [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/eff_norecompute.log" | tr '\n' ' ' | sed "s/\"/'/g")
    printf '{"tag":"eff_norecompute","status":"failed","reason":"%s","captured":"%s"}\n' \
      "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/eff_norecompute.failure.json"
    echo "  eff_norecompute FAILED (receipt)"; }

step "TRN2 EXTRAS COMPLETE"
bash "$BENCH_DIR/shared/bin/push_results.sh" trn2 || echo "  push FAILED"
