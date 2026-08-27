#!/usr/bin/env bash
# TRN2 LAST WINDOW -- the two open questions, in value order.
#
# The opportunistic seed lanes all self-skipped: their budget guards ask for
# 150-200 min and only ~139 remained. That is the guard working correctly, but
# it left a paid chip idle. These two lanes are worth more than the seed lanes
# were anyway, and both fit.
#
# 1. ISOLATION AT SEQ 8192. 25.1 found the maxutil lane spends only 36.6% of
#    its wall clock inside timed steps, against 89.1% at seq 2048 -- which is
#    why a 2.30x steady-state gain produced no end-to-end gain at all. That
#    section had to state: "We did not run the isolation lane at 8192 and
#    cannot attribute that gap to the dataloader." This closes exactly that
#    gap, using the same synthetic-vs-real pair as 21.
#
# 2. ctx_16384. The one question this study never answered. Two attempts, two
#    non-answers: a genuine host OOM on the original instance, and my own
#    stale-lock deadlock on this one. The box is otherwise idle, swap is
#    present, and either outcome is publishable.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
NP_VENV=/opt/aws_neuronx_venv_pytorch_2_9
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"
OUT="$BENCH_DIR/trn2/results/isolation"
EXTRAS="$BENCH_DIR/trn2/results/extras"
mkdir -p "$OUT" "$EXTRAS"

HARD_STOP="${HARD_STOP:-2026-08-06T09:35:00Z}"   # leaves 25 min before the 10:00 guard
stop_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] lastwindow: $*"; }
left() { echo $(( (stop_epoch - $(date -u +%s)) / 60 )); }

export NP_DEVICE=trn2 NP_REGION=sa-east-1 NP_CACHE_PREFIX=neuron-cache-v3
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --retry_failed_compilation"
read -r LNC NPROC TP <<<"$("$PY" -c '
import json;d=json.load(open("/opt/np/repo/trn2/results/extras/tp_probe.json"));print(d["lnc"],d["nproc"],d["tp"])')"
export NEURON_LOGICAL_NC_CONFIG="$LNC" NP_TELEMETRY_CORES="$NPROC"
EXTRA=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP")
bash shared/bin/hf_login.sh >/dev/null 2>&1 || log "hf_login failed"

lane() {  # lane <dir> <tag> <args...>
  local dir="$1" tag="$2"; shift 2
  { [ -s "$dir/$tag.json" ] || [ -s "$dir/$tag.failure.json" ]; } && { log "skip $tag (recorded)"; return 0; }
  log "$tag starting ($(left) min left)"
  find /opt/np/cache/neuron-compile-cache -name '*.lock' -delete 2>/dev/null
  "$PY" "$TELEM" --out "$dir/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" "${EXTRA[@]}" --out "$dir/$tag.json" "$@" \
    > "$dir/$tag.log" 2>&1 || {
      R=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,150}|ValueError: [^\"]{0,140}|\[F[0-9]+\][^\"]{0,120}" "$dir/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$R" ] || R=$(tail -c 300 "$dir/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      OOM=$(journalctl -k --since "-90 min" 2>/dev/null | grep -ci "out of memory" || echo 0)
      printf '{"tag":"%s","box":"trn2","status":"failed","reason":"%s","kernel_oom_events":%s,"captured":"%s"}\n' \
        "$tag" "$R" "$OOM" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dir/$tag.failure.json"
      log "$tag FAILED (receipt, kernel_oom=$OOM)"; }
  bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || log "push FAILED"
}

# ---- 1. isolation at seq 8192: closes 25.1's stated gap ---------------------
if [ "$(left)" -ge 45 ]; then
  lane "$OUT" real_ctl_8192  --model meta-llama/Llama-3.1-8B-Instruct --seq-len 8192 --max-steps 40
  lane "$OUT" synth_8192     --model meta-llama/Llama-3.1-8B-Instruct --seq-len 8192 --max-steps 40 --synthetic-data 160
else
  log "SKIP seq-8192 isolation ($(left) min)"
fi

# ---- 2. ctx_16384: the study's one unanswered question ----------------------
# Push first: this lane took the box down once already.
bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || true
if [ "$(left)" -ge 40 ]; then
  rm -f "$EXTRAS/ctx_16384.failure.json"   # prior receipts were a deferral and a deadlock, not results
  lane "$EXTRAS" ctx_16384 --model meta-llama/Llama-3.1-8B-Instruct --seq-len 16384 --max-steps 20
else
  log "SKIP ctx_16384 ($(left) min) -- not enough margin to attempt AND push"
fi

log "LAST WINDOW COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
