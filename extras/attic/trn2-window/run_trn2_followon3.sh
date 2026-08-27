#!/usr/bin/env bash
# TRN2 FOLLOW-ON, STAGE 3 -- the deliberate ctx_16384 retry.
#
#   systemd-run --unit=np-followon3 --property=TimeoutStartSec=infinity --collect \
#     /bin/bash -lc 'bash /opt/np/repo/extras/run_trn2_followon3.sh >> /opt/np/followon3_trn2.log 2>&1'
#
# WHY THIS RUNS LAST
# ------------------
# On the original Trainium2 instance, seq 16384 exhausted the 124 GiB host
# during compile. The OOM killer took down the whole suite and the instance was
# ultimately replaced, costing the better part of an hour of a non-refundable
# window. The automated ladder on the replacement box was therefore given a
# `deferred` receipt for that rung so it would skip it -- that receipt records a
# SCHEDULING DECISION and is explicitly not a measurement.
#
# The rung still deserves an attempt, because trn1 cannot reach 8192 at all and
# a passing 16384 on trn2 would be the strongest single line in the study. But
# it is the only lane here that can plausibly take the box down with it, so it
# goes last: after the quality gate (the one remaining symmetry item), after
# the dataloader isolation lane, and after the batch ladder. If it kills the
# box now, everything of value is already in S3.
#
# WHAT CHANGED SINCE THE FAILURE
# ------------------------------
# A 64 GiB swapfile is now mounted at /scratch/swapfile. trn1 has always had
# one; removing it from trn2's user-data is implicated in the original OOM.
# Swap does not make 16384 fit -- it makes an overrun SLOW instead of FATAL,
# which is the difference between a receipt and a lost instance.
#
# HONEST EXPECTATION: this probably still fails. Compile-time host memory is
# the binding constraint and 64 GiB of NVMe-backed swap against a 124 GiB
# shortfall is a cushion, not a fix. A clean receipt captured without losing the
# box is the realistic good outcome; a pass is the upside.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/trn2/results/extras"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"

HARD_STOP="${HARD_STOP:-2026-08-06T09:00:00Z}"   # an hour earlier than the others: see below
stop_epoch=$(date -u -d "$HARD_STOP" +%s 2>/dev/null || python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] followon3: $*"; }

log "waiting for stages 1 and 2"
while systemctl is-active --quiet np-followon 2>/dev/null \
   || systemctl is-active --quiet np-followon2 2>/dev/null \
   || pgrep -f 'run_phase3_trn2.sh|run_quality_gate.sh|run_dataloader_isolation.sh|run_batch_ladder.sh' >/dev/null 2>&1; do
  sleep 60
  if [ "$(date -u +%s)" -ge "$stop_epoch" ]; then
    log "hard stop reached while waiting -- 16384 not attempted, nothing lost"
    exit 0
  fi
done
sleep 20

# The stop is deliberately an hour earlier than the other stages. This lane can
# take the box down; the deadline pusher must still have a clear window to get
# everything into S3 afterwards. Buying that margin costs one optional rung.
now=$(date -u +%s); left=$(( (stop_epoch - now) / 60 ))
if [ "$left" -lt 45 ]; then
  log "SKIP ctx_16384 -- ${left} min to its (early) hard stop; not enough margin"
  log "     to both attempt it and still push results if it takes the box down"
  exit 0
fi
log "ctx_16384: ${left} min remain -- attempting"

# Push everything FIRST. If this lane kills the instance, the last-known-good
# state must already be off the box, not queued behind the thing that died.
bash shared/bin/push_results.sh trn2 || log "pre-push FAILED -- ABORTING, refusing"
bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || { log "pre-push failed twice; not attempting"; exit 1; }
log "pre-push done; everything of value is in S3"

swapon --show | grep -q swapfile && log "swap is on: $(swapon --show --noheadings | tr -s ' ')" \
  || log "WARNING: no swap; attempting anyway (receipt will note it)"

P="$OUT/tp_probe.json"
read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1])); print(d["lnc"], d["nproc"], d["tp"])' "$P")"
export NEURON_LOGICAL_NC_CONFIG="$LNC" NP_TELEMETRY_CORES="$NPROC"
export NP_DEVICE=trn2 NP_REGION=sa-east-1 NP_CACHE_PREFIX=neuron-cache-v3
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
bash shared/bin/hf_login.sh >/dev/null 2>&1 || log "hf_login failed"

# Overwrite the `deferred` receipt: this pass produces a real outcome, and a
# scheduling note must not survive alongside a measurement.
rm -f "$OUT/ctx_16384.failure.json"

"$PY" "$BENCH_DIR/shared/telemetry.py" --out "$OUT/ctx_16384.telemetry.csv" -- \
  "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
    --model meta-llama/Llama-3.1-8B-Instruct --tag ctx_16384 \
    --device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP" \
    --seq-len 16384 --max-steps 20 --out "$OUT/ctx_16384.json" \
  > "$OUT/ctx_16384.log" 2>&1 || {
    REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/ctx_16384.log" | head -1 | sed "s/\"/'/g")
    [ -n "$REASON" ] || REASON=$(tail -c 400 "$OUT/ctx_16384.log" | tr '\n' ' ' | sed "s/\"/'/g")
    OOM=$(journalctl -k --since "-90 min" 2>/dev/null | grep -ci "out of memory" || echo 0)
    printf '{"seq_len":16384,"status":"failed","reason":"%s","kernel_oom_events":%s,"swap_present":"%s","captured":"%s"}\n' \
      "$REASON" "$OOM" "$(swapon --show --noheadings | head -1 | tr -s ' ')" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/ctx_16384.failure.json"
    log "ctx_16384 FAILED (receipt captured, box survived)"; }

log "STAGE 3 COMPLETE"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
