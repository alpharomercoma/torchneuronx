#!/usr/bin/env bash
# Runs ON THE TRN2 BOX. Pushes everything to S3 before AWS reclaims the
# Capacity Block, and keeps pushing periodically until then.
#
#   setsid nohup bash extras/trn2_deadline_push.sh >> /opt/np/deadline_push.log 2>&1 &
#
# WHY
# ---
# A Capacity Block does not stop the instance at the end of its window, it
# TERMINATES it -- and termination begins at 11:00 UTC, thirty minutes before
# the nominal 11:30 UTC end. The EBS root goes with the instance, so unlike
# every other box in this project the warm NEFF cache does NOT survive. Anything
# not in S3 by 11:00Z is gone, including partial results from lanes that were
# still running.
#
# The main driver (run_phase3_trn2.sh) pushes when it FINISHES. That is exactly
# the wrong assumption to rely on here: if it is still mid-lane at the deadline,
# a finish-time push never happens and the whole window's work is lost. Hence a
# pusher on a clock rather than on completion.
set -uo pipefail
export AWS_PAGER=""

REPO="${REPO:-/opt/np/repo}"
DEADLINE_UTC="${DEADLINE_UTC:-10:30}"   # 30 min of margin before the 11:00Z kill
INTERVAL="${INTERVAL:-1800}"            # incremental push every 30 min

cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }
export NP_DEVICE=trn2 NP_REGION=sa-east-1 NP_CACHE_PREFIX=neuron-cache-v3

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] deadline-push: $*"; }

push() {
  # Never let one failing push abort the loop -- a partial save beats none.
  bash shared/bin/push_results.sh trn2 || log "push_results FAILED (continuing)"
  bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED (continuing)"
}

# Seconds until today's (or tomorrow's) DEADLINE_UTC.
seconds_to_deadline() {
  local now target
  now=$(date -u +%s)
  target=$(date -u -d "today ${DEADLINE_UTC}" +%s 2>/dev/null) || \
    target=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u +%Y-%m-%d) ${DEADLINE_UTC}" +%s)
  [ "$target" -le "$now" ] && target=$((target + 86400))
  echo $((target - now))
}

REMAIN=$(seconds_to_deadline)
log "armed: final push at ${DEADLINE_UTC}Z (in ${REMAIN}s), incremental every ${INTERVAL}s"

while :; do
  REMAIN=$(seconds_to_deadline)
  if [ "$REMAIN" -le "$INTERVAL" ]; then
    log "final window: sleeping ${REMAIN}s then pushing everything"
    sleep "$REMAIN"
    log "FINAL PUSH"
    push
    log "final push complete -- safe to lose the box"
    exit 0
  fi
  sleep "$INTERVAL"
  log "incremental push (${REMAIN}s to deadline)"
  push
done
