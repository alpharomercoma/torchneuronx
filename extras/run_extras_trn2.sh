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

# A 20-step TinyLlama rung proves the collective works at 1.1B. It does NOT
# prove Llama 3.1 8B compiles or fits at that TP -- different graph, different
# HBM profile, different collective volume. Record an explicit 8B preflight
# receipt so the report can say which claim the probe actually supports.
if [ ! -s "$OUT/tp_preflight_8b.json" ] && [ ! -s "$OUT/tp_preflight_8b.failure.json" ]; then
  step "TP preflight at 8B (the probe validated 1.1B, not this)"
  "$PY" "$TELEM" --out "$OUT/tp_preflight_8b.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
      --model meta-llama/Llama-3.1-8B-Instruct --tag tp_preflight_8b \
      "${TP_ARGS[@]}" --max-steps 20 --out "$OUT/tp_preflight_8b.json" \
    > "$OUT/tp_preflight_8b.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/tp_preflight_8b.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/tp_preflight_8b.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"tp_preflight_8b","lnc":%s,"nproc":%s,"tp":%s,"status":"failed","reason":"%s","captured":"%s"}\n' \
        "$LNC" "$NPROC" "$TP" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$OUT/tp_preflight_8b.failure.json"
      echo "  8B preflight FAILED at TP=$TP (receipt)"; }
fi

# If the ladder fell back to half the chip, the generational speed comparison
# is NOT publishable from these lanes -- it would be "a half-fed Trainium2".
if [ "$NPROC" -lt 4 ]; then
  echo "WARNING: world=$NPROC < 4 -- HALF THE CHIP IS IDLE."
  echo "         Lanes still run and are still receipts, but the trn1-vs-trn2"
  echo "         headline must be labelled an unsupported-configuration result,"
  echo "         not a one-chip generational comparison."
  printf '{"full_chip":false,"world":%s,"tp":%s,"note":"half-chip configuration; not the generational headline","captured":"%s"}\n' \
    "$NPROC" "$TP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/half_chip_warning.json"
fi

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

# ------------------------------------------------- trn1 PARITY: NKI (Track E)
# The kernel was written and benchmarked against NeuronCore-v2. Whether it even
# compiles for v3 is unknown, and that is worth measuring rather than skipping:
# "the custom kernel from the previous generation does not carry over" is a
# genuine, reportable portability finding for anyone planning a Trainium2
# migration. The simulate gate runs on CPU and is nearly free, and the device
# bench only runs if simulate is green -- same gating as trn1.
step "E: NKI simulate (CPU correctness gate)"
have "$OUT/nki_sim.json" || "$PY" "$BENCH_DIR/extras/nki_softmax.py" \
  --mode simulate --out "$OUT/nki_sim.json" > "$OUT/nki_sim.log" 2>&1 \
  || echo "  nki simulate FAILED"

step "E: NKI device benchmark vs torch (NeuronCore-v3 -- kernel was tuned for v2)"
if have "$OUT/nki_sim.json" && grep -q '"correct": true' "$OUT/nki_sim.json"; then
  have "$OUT/nki_device.json" || "$PY" "$TELEM" --out "$OUT/nki_device.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/extras/nki_softmax.py" --mode device --out "$OUT/nki_device.json" \
    > "$OUT/nki_device.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/nki_device.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/nki_device.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"nki_device","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/nki_device.failure.json"
      echo "  nki device FAILED (receipt -- v2 kernel may not target v3)"; }
else
  echo "  skip device bench: simulate gate not green (receipt stands)"
fi

# ------------------------------------ trn1 PARITY: whisper / clip / siglip (A)
# These ran on trn1 rather than inf2 because the vLLM DLAMI ships a patched
# transformers under an upstream version number and optimum-neuron cannot
# co-reside with it (receipts in inf2/results/extras). Running them here puts
# the same three lanes on v3 silicon. On trn1, clip and whisper produced failure
# receipts -- if either now passes on v3 that is itself a result.
step "A-parity: whisper / clip / siglip on NeuronCore-v3"
for lane in whisper clip siglip; do
  { have "$OUT/$lane.json" || have "$OUT/$lane.failure.json"; } && \
    { echo "skip $lane (recorded)"; continue; }
  "$PY" "$TELEM" --out "$OUT/$lane.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/extras/${lane}_lane.py" --out "$OUT/$lane.json" \
    > "$OUT/$lane.log" 2>&1 || {
      REASON=$(tail -c 300 "$OUT/$lane.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$lane" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$lane.failure.json"
      echo "  $lane FAILED (receipt)"; }
done

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

# --------------------------------------------- E3: micro-batch ladder (NEW)
# The most likely way this whole comparison MISLEADS.
#
# micro-batch 1 was chosen against trn1's 16 GiB per NeuronCore-v2, where seq
# 8192 already died at 18.12 GB. Trainium2 has 24 GiB per logical core and 3.18x
# the dense FLOPs, so the same micro-batch may simply UNDERFEED it -- and since
# MFU is measured against the whole chip, an underfed chip reports a low MFU
# that looks like a hardware verdict when it is really a configuration artifact.
#
# The fix is NOT to retune lane 4: that lane must stay identical to trn1's or
# there is no comparison left. This is a separate declared lane that finds the
# efficient operating point, so the report can state both "like-for-like" and
# "best-effort" numbers instead of conflating them.
step "E3: micro-batch ladder at seq 2048 (does micro-batch 1 underfeed Trainium2?)"
for MB in 2 4; do
  T="$OUT/eff_mb${MB}.json"
  { have "$T" || have "$OUT/eff_mb${MB}.failure.json"; } && { echo "skip mb $MB (recorded)"; continue; }
  "$PY" "$TELEM" --out "$OUT/eff_mb${MB}.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
      --model meta-llama/Llama-3.1-8B-Instruct --tag "eff_mb${MB}" \
      "${TP_ARGS[@]}" --micro-batch "$MB" --max-steps 100 --out "$T" \
    > "$OUT/eff_mb${MB}.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/eff_mb${MB}.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/eff_mb${MB}.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"eff_mb%s","micro_batch":%s,"status":"failed","reason":"%s","captured":"%s"}\n' \
        "$MB" "$MB" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/eff_mb${MB}.failure.json"
      echo "  eff_mb$MB FAILED (receipt)"; }
done

step "TRN2 EXTRAS COMPLETE"
bash "$BENCH_DIR/shared/bin/push_results.sh" trn2 || echo "  push FAILED"
