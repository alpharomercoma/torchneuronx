#!/usr/bin/env bash
# TRN2 OPTIONAL PASSES -- dead last, after the corrected batch ladder.
#
# Re-ordered ahead of these because the ladder is a SYMMETRY item (trn1 has one,
# trn2 must too) while opportunistic/frontier/maxutil are nice-to-haves. With
# ~5 h left after a 3 h deadlock burned on a stale compile lock, symmetry wins.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] optional_last: $*"; }

log "waiting for the corrected ladder"
while systemctl is-active --quiet np-ladder-last 2>/dev/null \
   || pgrep -f 'run_batch_ladder.sh|sft_lora.py' >/dev/null 2>&1; do
  sleep 60
  [ "$(date -u +%s)" -ge "$stop_epoch" ] && { log "hard stop while waiting"; exit 0; }
done
while ss -ltn 2>/dev/null | grep -q ':29500 '; do sleep 10; done
sleep 15

left=$(( (stop_epoch - $(date -u +%s)) / 60 ))
if [ "$left" -lt 40 ]; then log "SKIP -- only ${left} min left"; exit 0; fi
log "optional passes starting (${left} min left)"
bash extras/run_opportunistic_trn2.sh || log "opportunistic ended early"
bash shared/bin/push_results.sh trn2 || true
bash extras/run_frontier_trn2.sh || log "frontier ended early"
bash shared/bin/push_results.sh trn2 || true
bash extras/run_maxutil_trn2.sh || log "maxutil ended early"
log "COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || true
bash shared/bin/push_results.sh trn2 || log "push FAILED"
