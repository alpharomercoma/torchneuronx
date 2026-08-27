#!/usr/bin/env bash
# TRN2 FOLLOW-ON QUEUE -- runs after the main Phase-3 suite exits.
#
#   systemd-run --unit=np-followon --property=TimeoutStartSec=infinity --collect \
#     /bin/bash -lc 'bash /opt/np/repo/extras/run_trn2_followon.sh >> /opt/np/followon_trn2.log 2>&1'
#
# WHY THIS EXISTS AS A BOX-SIDE SCRIPT
# ------------------------------------
# The Capacity Block is non-refundable and ends at a fixed wall-clock time. Any
# step that waits on a laptop to notice something finished is a step that can
# silently lose an hour of a window that cannot be extended. Everything that
# MUST happen inside the window therefore lives on the box and triggers itself.
#
# WHAT IT RUNS, AND WHY IN THIS ORDER
# -----------------------------------
# 1. QUALITY GATE. Highest priority, because it is the one place where the two
#    chips are currently ASYMMETRIC: trn1 has held-out loss before and after,
#    trn2 has none. Every other lane already exists on both sides. An
#    asymmetric quality claim is worse than no quality claim, so this runs
#    first and is the only item allowed to consume the remaining window.
# 2. DATALOADER ISOLATION. Second, because trn1 runs it for free and trn2 needs
#    it only for symmetry. Its finding (whether the synthetic uplift shrinks
#    with sequence length) is interesting on trn2 specifically -- trn2 is the
#    chip whose 1.20x headline the hypothesis is trying to explain -- but it is
#    a supporting result, not a load-bearing one.
#
# Neither may start after the hard stop. Being terminated mid-lane produces no
# artifact at all, so a lane that cannot finish is worse than a lane not run:
# it costs window that the other lane could have used.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"

# EC2 begins TERMINATING the block instance at 11:00Z and the deadline pusher
# makes its final push at 10:30Z. 10:00Z is the last moment it is honest to
# START anything; the quality gate's 8B lane alone runs the better part of an
# hour.
HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(date -u -d "$HARD_STOP" +%s 2>/dev/null || python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] followon: $*"; }

before_deadline() {   # before_deadline <needed_minutes> <label>
  local need="$1" label="$2" now left
  now=$(date -u +%s); left=$(( (stop_epoch - now) / 60 ))
  if [ "$left" -lt "$need" ]; then
    log "SKIP $label -- ${left} min to hard stop, needs ~${need}. Starting it would"
    log "     burn window and still be terminated mid-lane, producing no artifact."
    return 1
  fi
  log "$label: ${left} min remain, needs ~${need} -- proceeding"
  return 0
}

# ---- wait for the main suite to release the chip ---------------------------
# Both lanes below need all four logical NeuronCores. Running concurrently with
# the suite would not just be slow, it would corrupt BOTH measurements, since
# each would be timing a chip it does not exclusively own.
log "waiting for run_phase3_trn2.sh to exit"
waited=0
while pgrep -f run_phase3_trn2.sh >/dev/null 2>&1; do
  sleep 60
  waited=$((waited + 1))
  [ $((waited % 30)) -eq 0 ] && log "still waiting (${waited} min)"
  if [ "$(date -u +%s)" -ge "$stop_epoch" ]; then
    log "hard stop reached while waiting -- nothing started, nothing lost"
    exit 0
  fi
done
log "suite has exited after ${waited} min of waiting; chip is free"
sleep 20   # let the driver's final push_results.sh finish

# ---- 1. quality gate (the symmetry item) -----------------------------------
if before_deadline 75 "quality gate"; then
  BOX=trn2 bash extras/run_quality_gate.sh || log "quality gate returned nonzero (receipts on disk)"
  bash shared/bin/push_results.sh trn2 || log "push FAILED"
fi

# ---- 2. dataloader isolation (symmetry with trn1) --------------------------
if before_deadline 50 "dataloader isolation"; then
  BOX=trn2 bash extras/run_dataloader_isolation.sh || log "isolation returned nonzero (receipts on disk)"
  bash shared/bin/push_results.sh trn2 || log "push FAILED"
fi

log "FOLLOW-ON COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
