#!/usr/bin/env bash
# TRN2 CORRECTED BATCH LADDER -- runs after the recovery queue.
#
# The first trn2 ladder attempt was stopped mid-mb2 because it used the old
# design that varied grad_accum per rung. A control on trn1 showed steady-state
# throughput is not comparable across grad_accum values on this stack
# (MFU 146.7% at accum 4), so those rungs could not have been compared to mb1.
#
# mb1_seq4096.json from that attempt is KEPT: it ran at grad_accum 8, which is
# the published configuration and what the corrected ladder also uses, so it is
# a valid rung and have() will skip re-running it.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ladder_last: $*"; }

log "waiting for the recovery queue"
while systemctl is-active --quiet np-recover 2>/dev/null \
   || pgrep -f 'sft_lora.py|run_quality_gate.sh|run_dataloader_isolation.sh|run_frontier|run_maxutil|run_opportunistic' >/dev/null 2>&1; do
  sleep 60
  [ "$(date -u +%s)" -ge "$stop_epoch" ] && { log "hard stop while waiting"; exit 0; }
done
while ss -ltn 2>/dev/null | grep -q ':29500 '; do log "port 29500 held, waiting"; sleep 10; done
sleep 15

left=$(( (stop_epoch - $(date -u +%s)) / 60 ))
if [ "$left" -lt 30 ]; then log "SKIP -- only ${left} min left"; exit 0; fi
log "corrected ladder starting (${left} min left)"
BOX=trn2 bash extras/run_batch_ladder.sh || log "ladder nonzero (receipts on disk)"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
log "COMPLETE"
