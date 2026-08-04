#!/usr/bin/env bash
# Opportunistic lanes: spend the paid-for remainder of the Capacity Block.
#
# WHY THIS EXISTS
# ---------------
# A Capacity Block is bought as a WINDOW, not as hours. It cannot be cancelled,
# unused time is not refunded, and finishing early returns nothing. The trn1
# suite measured 5.9 h; the window is 23.5 h. So once the science the block was
# bought for is done and pushed, idle time is the only remaining way to waste
# the money -- roughly 17 hours of it.
#
# THE RULES THIS FILE OBEYS
# -------------------------
# 1. NOTHING here may run before the mandatory work is complete and pushed.
#    The caller gates on the completion marker; this script re-checks.
# 2. NOTHING here may touch lane 4. The primary llama31_lora lane is the
#    trn1-vs-trn2 comparison and must stay byte-identical to trn1's config.
#    Every lane below writes its own tag.
# 3. HARD STOP at OPP_DEADLINE_UTC (default 10:00Z), a full 30 minutes before
#    the deadline pusher's final push at 10:30Z and a full hour before EC2
#    begins terminating at 11:00Z. A half-finished bonus lane must never
#    threaten the results that were actually paid for.
# 4. Cheapest and highest-value first, so a truncated tail still delivers the
#    most useful additions.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/trn2/results/opportunistic"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"      # libneuronpjrt-path
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
EXTRAS="$BENCH_DIR/trn2/results/extras"
OPP_DEADLINE_UTC="${OPP_DEADLINE_UTC:-10:00}"

export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"

mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ OPP: $* ############"; echo; }

# ---- Rule 1: the paid-for science must already be done -----------------------
if ! grep -q "PHASE3 TRN2 ALL COMPLETE" /opt/np/phase3_trn2.log 2>/dev/null; then
  echo "FATAL: main suite has not reported ALL COMPLETE -- refusing to start"
  echo "       opportunistic lanes must never compete with the mandatory ones"
  exit 3
fi

# ---- Rule 3: hard deadline --------------------------------------------------
seconds_left() {
  local now target
  now=$(date -u +%s)
  target=$(date -u -d "today ${OPP_DEADLINE_UTC}" +%s 2>/dev/null) || \
    target=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u +%Y-%m-%d) ${OPP_DEADLINE_UTC}" +%s)
  [ "$target" -le "$now" ] && target=$((target + 86400))
  echo $((target - now))
}

# Refuse to START a lane that cannot plausibly finish. Better to stop early with
# clean receipts than to be killed mid-write.
budget_ok() {
  local need_min="$1" left
  left=$(seconds_left)
  if [ "$left" -lt $((need_min * 60)) ]; then
    echo "  deadline guard: $((left / 60)) min left, lane needs ~${need_min} min -- stopping here"
    return 1
  fi
  return 0
}

# TP config exactly as the probe validated it -- never re-decide it here.
if [ ! -s "$EXTRAS/tp_probe.json" ]; then
  echo "FATAL: $EXTRAS/tp_probe.json missing"; exit 3
fi
read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["lnc"], d["nproc"], d["tp"])' "$EXTRAS/tp_probe.json")"
export NEURON_LOGICAL_NC_CONFIG="$LNC"
export NP_TELEMETRY_CORES="$NPROC"
TP_ARGS=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP")
echo "opportunistic lanes: LNC=$LNC world=$NPROC TP=$TP, hard stop ${OPP_DEADLINE_UTC}Z"

run_lane() {           # run_lane <tag> <need_min> <extra sft_lora args...>
  local tag="$1" need="$2"; shift 2
  { have "$OUT/$tag.json" || have "$OUT/$tag.failure.json"; } && { echo "skip $tag (recorded)"; return 0; }
  budget_ok "$need" || return 1
  step "$tag"
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" --tag "$tag" \
      "${TP_ARGS[@]}" --out "$OUT/$tag.json" "$@" \
    > "$OUT/$tag.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
      echo "  $tag FAILED (receipt)"; }
  # Push after EVERY lane: the instance is terminated, not stopped, so an
  # un-pushed result is a lost result.
  bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || echo "  push FAILED"
}

# ---- 1. Variance on the headline lane ---------------------------------------
# The single most valuable addition. Without repeats every trn1-vs-trn2 number
# in section 15 is n=1, and "3.0x faster" cannot be distinguished from
# "3.0x faster, +/- who knows". Same config as lane 4 in every respect EXCEPT
# the seed, written to its own tag so lane 4 itself is untouched.
for SEED in 43 44; do
  run_lane "llama31_lora_seed${SEED}" 150 \
    --model meta-llama/Llama-3.1-8B-Instruct --seed "$SEED" || break
done

# ---- 2. Push the context cliff further ---------------------------------------
# The headline is that 24 GiB/core moves the cliff trn1 hit at 8192. Finding
# where it ACTUALLY lands beats confirming it moved. Only attempted if 16384
# cleared -- if that already failed, the cliff is found and 32768 is wasted time.
if have "$EXTRAS/ctx_16384.json"; then
  run_lane "ctx_32768" 60 \
    --model meta-llama/Llama-3.1-8B-Instruct --seq-len 32768 --max-steps 20 || true
else
  echo "skip ctx_32768: 16384 did not pass, so the cliff is already located"
fi

# ---- 3. Finish the micro-batch ladder ---------------------------------------
# E3 probes 2 and 4. If 4 cleared, the efficient operating point is still above
# it and worth locating -- this is the lever that decides whether micro-batch 1
# was underfeeding the chip.
if have "$EXTRAS/eff_mb4.json"; then
  run_lane "eff_mb8" 90 \
    --model meta-llama/Llama-3.1-8B-Instruct --micro-batch 8 --max-steps 100 || true
else
  echo "skip eff_mb8: micro-batch 4 did not pass"
fi

# ---- 4. Combined best-effort configuration ----------------------------------
# Every lever that individually helped, applied together. This is the "what can
# this chip actually do" number, as distinct from the like-for-like comparison.
# Reported SEPARATELY and never substituted for lane 4.
run_lane "eff_combined" 120 \
  --model meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --micro-batch 2 \
  --no-gradient-checkpointing --max-steps 100 || true

# ---- 5. Qwen3 variance -------------------------------------------------------
run_lane "qwen3_lora_seed43" 200 --model Qwen/Qwen3-8B --seed 43 || true

echo
echo "############ OPPORTUNISTIC COMPLETE ($(seconds_left) s to ${OPP_DEADLINE_UTC}Z) ############"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh trn2 || echo "push FAILED"
