#!/usr/bin/env bash
# TRN2 FOLLOW-ON, STAGE 4 -- resume the OPTIONAL passes, last.
#
#   systemd-run --unit=np-followon4 --property=TimeoutStartSec=infinity --collect \
#     /bin/bash -lc 'bash /opt/np/repo/extras/run_trn2_followon4.sh >> /opt/np/followon4_trn2.log 2>&1'
#
# WHY THE ORDER WAS CHANGED MID-RUN
# ---------------------------------
# The main suite finished, then queued three OPTIONAL passes back to back --
# opportunistic, frontier (which includes a 32B model), maxutil -- with the
# quality gate behind all of them. The quality gate is the ONE place the two
# chips are still asymmetric: trn1 has held-out loss before and after, trn2 has
# none. Every optional lane is a nice-to-have; the gate is the difference
# between "both chips were measured the same way" and an asymmetric claim.
#
# Worse, the lane in flight when this decision was made was llama31_lora_seed43.
# --seed is a NO-OP on this stack: three trn1 seeds produced BIT-IDENTICAL loss
# because packing pins the data order (§16, correction 6). That lane would have
# spent ~55 minutes of a non-refundable window reproducing a number we can
# predict exactly, while the gate waited behind it.
#
# So the suite was stopped and the queue re-ordered:
#   quality gate -> dataloader isolation -> batch ladder -> THESE.
#
# Nothing is lost that was not already recorded: the main suite had printed
# "PHASE3 TRN2 ALL COMPLETE" and all 27 primary artifacts were in S3 before the
# stop. What was given up is one redundant seed lane, deliberately.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"

HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(date -u -d "$HARD_STOP" +%s 2>/dev/null || python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] followon4: $*"; }

log "waiting for stages 1-3 and any running lane"
while systemctl is-active --quiet np-followon 2>/dev/null \
   || systemctl is-active --quiet np-followon2 2>/dev/null \
   || systemctl is-active --quiet np-followon3 2>/dev/null \
   || pgrep -f 'run_quality_gate.sh|run_dataloader_isolation.sh|run_batch_ladder.sh' >/dev/null 2>&1; do
  sleep 60
  if [ "$(date -u +%s)" -ge "$stop_epoch" ]; then
    log "hard stop reached while waiting -- optional passes not resumed, nothing lost"
    exit 0
  fi
done
sleep 20

# The optional passes carry their own deadline guards, so they can be handed
# whatever window is left and will stop themselves. 40 minutes is the floor
# below which starting one buys nothing.
now=$(date -u +%s); left=$(( (stop_epoch - now) / 60 ))
if [ "$left" -lt 40 ]; then
  log "SKIP optional passes -- only ${left} min to hard stop"
  exit 0
fi
log "resuming optional passes with ${left} min of window"

# Seed lanes are deliberately NOT resumed. --seed is a no-op here, so they
# re-measure timing noise only, and §20 already bounds that at 2.4% across two
# physical chips -- a stronger measurement than a second seed on one chip.
export NP_SKIP_SEED_LANES=1

bash extras/run_opportunistic_trn2.sh || log "opportunistic ended early (by design or receipt)"
bash shared/bin/push_results.sh trn2 || log "push FAILED"

bash extras/run_frontier_trn2.sh || log "frontier ended early (by design or receipt)"
bash shared/bin/push_results.sh trn2 || log "push FAILED"

bash extras/run_maxutil_trn2.sh || log "maxutil ended early (by design or receipt)"

log "STAGE 4 COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
