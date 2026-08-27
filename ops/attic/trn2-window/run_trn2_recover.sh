#!/usr/bin/env bash
# TRN2 RECOVERY QUEUE -- replaces follow-on stages 3 and 4.
#
#   systemd-run --unit=np-recover --property=TimeoutStartSec=infinity --collect \
#     /bin/bash -lc 'bash /opt/np/repo/extras/run_trn2_recover.sh >> /opt/np/recover_trn2.log 2>&1'
#
# WHY THIS EXISTS
# ---------------
# Stopping the optional passes to promote the quality gate left a torch
# distributed rendezvous process holding TCP port 29500. The quality gate and
# the isolation smoke both started seconds later and both died instantly with
# EADDRINUSE. Neither failure says anything about the stack: they are collateral
# from the kill, and their receipts are artifacts, not measurements.
#
# So this script does three things the previous chain could not:
#   1. WAITS FOR PORT 29500 before every lane, so a lingering rendezvous can
#      never again be recorded as a lane failure.
#   2. Deletes the two spurious receipts, with the reason logged, so the lanes
#      actually re-run instead of being skipped by have().
#   3. Runs everything that remains in ONE priority order, rather than in
#      separate units that each only knew about some of the others and could
#      race each other.
#
# ORDER, AND WHY
#   quality gate      -- the ONE remaining asymmetry between the chips
#   dataloader isolation -- symmetry with trn1, and trn2 is the chip whose
#                        headline the hypothesis was trying to explain
#   ctx_16384         -- the risky rung; it can take the box down, so it goes
#                        after everything that must be measured
#   optional passes   -- opportunistic / frontier / maxutil, nice-to-have
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"

HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
RISKY_STOP="${RISKY_STOP:-2026-08-06T09:00:00Z}"   # ctx_16384 stops an hour early
stop_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")
risky_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$RISKY_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] recover: $*"; }

mins_left() { echo $(( ($1 - $(date -u +%s)) / 60 )); }

wait_for_chip() {   # no other lane may hold the NeuronCores or the rendezvous port
  local waited=0
  while pgrep -f 'sft_lora.py|run_batch_ladder.sh|run_quality_gate.sh|run_dataloader_isolation.sh|run_frontier|run_maxutil|run_opportunistic|run_phase3_trn2.sh' >/dev/null 2>&1; do
    sleep 30; waited=$((waited+1))
    [ $((waited % 60)) -eq 0 ] && log "waiting for the chip ($((waited/2)) min)"
    [ "$(date -u +%s)" -ge "$stop_epoch" ] && { log "hard stop while waiting"; return 1; }
  done
  # THE PORT GUARD. This is the bug that cost the first quality-gate attempt:
  # the process was gone but its listening socket was not, and torchrun failed
  # instantly with EADDRINUSE.
  local p=0
  while ss -ltn 2>/dev/null | grep -q ':29500 '; do
    sleep 10; p=$((p+1))
    [ $((p % 6)) -eq 0 ] && log "port 29500 still held, waiting ($((p*10))s)"
    [ "$p" -gt 60 ] && { log "port 29500 held for 10 min -- proceeding anyway, receipt will show it"; break; }
  done
  sleep 10
  return 0
}

# ---- clear the receipts that are artifacts of the kill, not measurements ----
for f in trn2/results/quality/quality_smoke.failure.json \
         trn2/results/isolation/synth_smoke.failure.json; do
  if [ -s "$f" ] && grep -q EADDRINUSE "$f" 2>/dev/null; then
    mkdir -p "$(dirname "$f")/invalidated"
    mv "$f" "$(dirname "$f")/invalidated/" 2>/dev/null
    log "cleared $f -- EADDRINUSE from a killed rendezvous, not a stack failure"
  fi
done

# ---- 1. quality gate: the remaining asymmetry --------------------------------
if wait_for_chip; then
  left=$(mins_left "$stop_epoch")
  if [ "$left" -ge 75 ]; then
    log "quality gate starting (${left} min left)"
    BOX=trn2 bash extras/run_quality_gate.sh || log "quality gate nonzero (receipts on disk)"
    bash shared/bin/push_results.sh trn2 || log "push FAILED"
  else
    log "SKIP quality gate -- only ${left} min left, needs ~75"
  fi
fi

# ---- 2. dataloader isolation: symmetry with trn1 -----------------------------
if wait_for_chip; then
  left=$(mins_left "$stop_epoch")
  if [ "$left" -ge 50 ]; then
    log "dataloader isolation starting (${left} min left)"
    BOX=trn2 bash extras/run_dataloader_isolation.sh || log "isolation nonzero (receipts on disk)"
    bash shared/bin/push_results.sh trn2 || log "push FAILED"
  else
    log "SKIP isolation -- only ${left} min left"
  fi
fi

# ---- 3. ctx_16384: the rung that can take the box down -----------------------
if wait_for_chip; then
  left=$(mins_left "$risky_epoch")
  if [ "$left" -ge 45 ]; then
    log "ctx_16384 retry starting (${left} min to its early stop)"
    bash extras/run_trn2_followon3.sh || log "ctx_16384 stage nonzero"
  else
    log "SKIP ctx_16384 -- ${left} min to its early stop; not enough margin to"
    log "     attempt it AND still push results if it takes the box down"
  fi
fi

# ---- 4. optional passes ------------------------------------------------------
if wait_for_chip; then
  left=$(mins_left "$stop_epoch")
  if [ "$left" -ge 40 ]; then
    log "optional passes resuming (${left} min left)"
    bash extras/run_opportunistic_trn2.sh || log "opportunistic ended early"
    bash shared/bin/push_results.sh trn2 || true
    bash extras/run_frontier_trn2.sh || log "frontier ended early"
    bash shared/bin/push_results.sh trn2 || true
    bash extras/run_maxutil_trn2.sh || log "maxutil ended early"
  else
    log "SKIP optional passes -- only ${left} min left"
  fi
fi

log "RECOVERY QUEUE COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
