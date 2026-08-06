#!/usr/bin/env bash
# TRN2 FOLLOW-ON QUEUE, STAGE 2 -- the batch ladder.
#
#   systemd-run --unit=np-followon2 --property=TimeoutStartSec=infinity --collect \
#     /bin/bash -lc 'bash /opt/np/repo/extras/run_trn2_followon2.sh >> /opt/np/followon2_trn2.log 2>&1'
#
# WHY THIS IS A SECOND FILE INSTEAD OF THREE MORE LINES IN THE FIRST
# ------------------------------------------------------------------
# `run_trn2_followon.sh` is ALREADY EXECUTING on the box, sitting in its wait
# loop. bash reads a script lazily, by byte offset, as it runs -- appending to
# a running script makes the shell resume at a stale offset and execute
# garbage. That is a recorded gotcha in this project, so stage 2 is a separate
# file chained by a separate unit.
#
# WHAT IT RUNS
# ------------
# The micro-batch ladder at seq 4096 (see run_batch_ladder.sh for the
# hypothesis and the pre-registered prediction). It is stage 2 because stage 1
# holds the two symmetry items -- quality gate and dataloader isolation -- that
# trn1 has already completed. Symmetry with trn1 outranks a new experiment when
# the window is finite.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"

HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(date -u -d "$HARD_STOP" +%s 2>/dev/null || python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] followon2: $*"; }

log "waiting for np-followon (stage 1) to finish"
waited=0
while systemctl is-active --quiet np-followon 2>/dev/null; do
  sleep 60
  waited=$((waited + 1))
  [ $((waited % 30)) -eq 0 ] && log "still waiting (${waited} min)"
  if [ "$(date -u +%s)" -ge "$stop_epoch" ]; then
    log "hard stop reached while waiting -- nothing started, nothing lost"
    exit 0
  fi
done
log "stage 1 finished after ${waited} min"

# Belt and braces: stage 1 could have exited while the main suite was somehow
# still alive. Both contend for the whole chip.
while pgrep -f 'run_phase3_trn2.sh|run_quality_gate.sh|run_dataloader_isolation.sh' >/dev/null 2>&1; do
  log "another lane still holds the chip; waiting"
  sleep 60
  [ "$(date -u +%s)" -ge "$stop_epoch" ] && { log "hard stop reached"; exit 0; }
done
sleep 20

now=$(date -u +%s); left=$(( (stop_epoch - now) / 60 ))
# Four rungs at 30 steps, with the higher rungs slower per step; ~60 min is the
# honest estimate and a truncated ladder is still useful because each rung
# writes its own artifact as it completes.
if [ "$left" -lt 35 ]; then
  log "SKIP batch ladder -- ${left} min to hard stop, not enough for a useful ladder"
  exit 0
fi
log "batch ladder: ${left} min remain -- proceeding"

BOX=trn2 bash extras/run_batch_ladder.sh || log "batch ladder returned nonzero (receipts on disk)"

log "STAGE 2 COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
