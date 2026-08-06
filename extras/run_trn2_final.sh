#!/usr/bin/env bash
# TRN2 FINAL QUEUE -- new information first, redundant lanes last.
#
# The optional work splits cleanly:
#   NEW INFORMATION  -- full fine-tuning vs LoRA, a second dataset, packing
#                       isolation, multi-model residency, max-utilisation run.
#                       No data on any of these.
#   REDUNDANT        -- llama31_lora_seed43/44 and qwen3_lora_seed43. `--seed`
#                       is a NO-OP on this stack (three trn1 seeds produced
#                       bit-identical loss), so these re-measure timing noise
#                       only -- and REPORT 20 already bounds that at 2.4% using
#                       two PHYSICAL chips, which is the stronger measurement.
#
# Running new information first forecloses nothing: the seed lanes still run if
# the window allows. It simply refuses to spend the last hours of a
# non-refundable block on numbers we can already predict.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] final: $*"; }
left() { echo $(( (stop_epoch - $(date -u +%s)) / 60 )); }

wait_chip() {
  while pgrep -f 'sft_lora.py|run_batch_ladder.sh|run_frontier|run_maxutil|run_opportunistic' >/dev/null 2>&1; do
    sleep 30; [ "$(date -u +%s)" -ge "$stop_epoch" ] && return 1
  done
  while ss -ltn 2>/dev/null | grep -q ':29500 '; do sleep 10; done
  # A killed compile leaves a lock and the next lane deadlocks on it. This cost
  # three hours on ctx_16384; never start a lane without clearing them.
  find /opt/np/cache/neuron-compile-cache -name '*.lock' -delete 2>/dev/null
  sleep 10; return 0
}

# ---- 1. FRONTIER: full-FT, datasets, residency -------------------------------
if wait_chip && [ "$(left)" -ge 40 ]; then
  log "frontier starting ($(left) min left)"
  bash extras/run_frontier_trn2.sh || log "frontier ended early (by design or receipt)"
  bash shared/bin/push_results.sh trn2 || log "push FAILED"
else
  log "SKIP frontier ($(left) min left)"
fi

# ---- 2. MAXUTIL: the best-configuration full run -----------------------------
if wait_chip && [ "$(left)" -ge 70 ]; then
  log "maxutil starting ($(left) min left)"
  bash extras/run_maxutil_trn2.sh || log "maxutil ended early"
  bash shared/bin/push_results.sh trn2 || log "push FAILED"
else
  log "SKIP maxutil ($(left) min left) -- it runs 3 full epochs and a truncated"
  log "     run is not comparable to lane 4, which is the whole point of it"
fi

# ---- 3. OPPORTUNISTIC: the redundant seed lanes, only if time is left --------
if wait_chip && [ "$(left)" -ge 70 ]; then
  log "opportunistic (seed lanes) starting with the remaining $(left) min"
  bash extras/run_opportunistic_trn2.sh || log "opportunistic ended early"
else
  log "SKIP opportunistic ($(left) min left). These are the seed lanes; --seed"
  log "     is a no-op here, so nothing is lost that REPORT 20 does not already"
  log "     bound better using two physical chips."
fi

log "FINAL QUEUE COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
