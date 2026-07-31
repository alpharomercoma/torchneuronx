#!/usr/bin/env bash
# trn1-side Phase-2 lanes: NKI kernel (E), training ctx ladder (B4),
# checkpoint save/restore timing (C4). Same resumable discipline.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$BENCH_DIR/trn1/results/extras"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"  # libneuronpjrt-path must be findable (Phase-1 gotcha #2)
PY="$NP_VENV/bin/python"
# SSM shells run as root with a bare env: without these, gated-model lanes
# 401 on HF (ctx_8192 was misfiled as a compiler failure because of this)
# and every lane recompiles into a cold cache.
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
bash "$BENCH_DIR/shared/bin/hf_login.sh" >/dev/null 2>&1 || echo "WARN: hf_login failed (gated models will 401)"
TELEM="$BENCH_DIR/shared/telemetry.py"
mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ $* ############"; echo; }

step "E: NKI simulate (CPU correctness gate)"
have "$OUT/nki_sim.json" || "$PY" "$BENCH_DIR/extras/nki_softmax.py" \
  --mode simulate --out "$OUT/nki_sim.json" > "$OUT/nki_sim.log" 2>&1 \
  || echo "  nki simulate FAILED"

step "E: NKI device benchmark vs torch"
if have "$OUT/nki_sim.json" && grep -q '"correct": true' "$OUT/nki_sim.json"; then
  have "$OUT/nki_device.json" || "$PY" "$TELEM" --out "$OUT/nki_device.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/extras/nki_softmax.py" --mode device --out "$OUT/nki_device.json" \
    > "$OUT/nki_device.log" 2>&1 || echo "  nki device FAILED"
else
  echo "  skip device bench: simulate gate not green (receipt stands)"
fi

step "B4: training context ladder (20-step probes)"
for SL in 4096 8192; do
  T="$OUT/ctx_${SL}.json"
  { have "$T" || have "$OUT/ctx_${SL}.failure.json"; } && { echo "skip ctx $SL (recorded)"; continue; }
  "$PY" "$TELEM" --out "$OUT/ctx_${SL}.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
      --model meta-llama/Llama-3.1-8B-Instruct --tag "ctx_${SL}" \
      --seq-len "$SL" --max-steps 20 --out "$T" \
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
    --max-steps 30 --save-steps 10 --out "$OUT/ckpt_timing.json" \
  > "$OUT/ckpt_timing.log" 2>&1 || echo "  ckpt timing FAILED"

step "A-parity: whisper / clip / siglip (moved from inf2: the vLLM DLAMI ships a patched transformers under an upstream version number -- optimum-neuron cannot co-reside there; receipts in inf2/results/extras. Same NeuronCore-v2 silicon here.)"
for lane in whisper clip siglip; do
  have "$OUT/$lane.json" && { echo "skip $lane"; continue; }
  "$PY" "$TELEM" --out "$OUT/$lane.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/extras/${lane}_lane.py" --out "$OUT/$lane.json" \
    > "$OUT/$lane.log" 2>&1 || echo "  $lane FAILED (see extras/$lane.log)"
done

step "TRN1 EXTRAS COMPLETE"
bash "$BENCH_DIR/shared/bin/push_results.sh" trn1 || echo "  push FAILED"
