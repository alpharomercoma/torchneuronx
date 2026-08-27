#!/usr/bin/env bash
# TRN2 TAIL -- the two lanes the frontier pass missed, then the seed lanes.
#
# WHY THIS EXISTS
# ---------------
# fp8_probe and qwen3_32b_lora were SKIPPED during the frontier pass because
# they carried failure receipts -- and those receipts were EADDRINUSE artifacts
# of lanes I killed while re-ordering the queue, not real results. The receipts
# were cleared, but frontier had already walked past them and runs only once.
#
# They are worth more than the seed lanes that would otherwise fill the window:
#   fp8_probe      -- the study's ONLY FP8 datapoint. 16 explicitly names the
#                     FP8 gap as headroom an H100 can exploit and we cannot;
#                     this is the one measurement that speaks to it. ~15 min.
#   qwen3_32b_lora -- a 32B model on ONE chip, which Trainium1 could never
#                     attempt. ~45 min, and if it hits the same HBM wall the
#                     MoE and micro-batch lanes hit, that is itself publishable.
#
# The three seed lanes run afterwards if the window allows. They are last
# because --seed is a NO-OP on this stack (three trn1 seeds gave bit-identical
# loss) so they re-measure timing noise only -- and 20 already bounds that at
# 2.4% using two PHYSICAL chips, which is the stronger measurement.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
HARD_STOP="${HARD_STOP:-2026-08-06T10:00:00Z}"
stop_epoch=$(python3 -c "
import datetime;print(int(datetime.datetime.strptime('$HARD_STOP','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] tail: $*"; }
left() { echo $(( (stop_epoch - $(date -u +%s)) / 60 )); }

wait_chip() {
  while systemctl is-active --quiet np-final 2>/dev/null \
     || pgrep -f 'sft_lora.py|run_frontier|run_maxutil|run_opportunistic|run_batch_ladder' >/dev/null 2>&1; do
    sleep 60; [ "$(date -u +%s)" -ge "$stop_epoch" ] && return 1
  done
  while ss -ltn 2>/dev/null | grep -q ':29500 '; do sleep 10; done
  # A killed compile leaves a lock and the next lane deadlocks on it -- three
  # hours were lost to exactly this on ctx_16384.
  find /opt/np/cache/neuron-compile-cache -name '*.lock' -delete 2>/dev/null
  sleep 10; return 0
}

# ---- 0. cifar_vit: the one academic lane trn2 lacks -------------------------
# It was ATTEMPTED at 22:11Z and neuronx-cc was forcibly killed ([F137],
# insufficient system memory). No receipt was written because the academic
# driver only echoed on failure, so the lane looked "not run" rather than
# "tried and failed". Two things changed since: a 64 GiB swapfile now exists,
# and the driver writes a receipt either way. It needs ONE core and ~15 min, so
# it is cheap enough to go first.
if wait_chip && [ "$(left)" -ge 25 ]; then
  log "academic cifar_vit ($(left) min left) -- the other five lanes are recorded and will skip"
  BOX=trn2 bash academic/run_academic.sh || log "academic ended early (receipt written)"
  bash shared/bin/push_results.sh trn2 || log "push FAILED"
else
  log "SKIP cifar_vit ($(left) min left)"
fi

# ---- 1. second frontier pass: picks up ONLY the two cleared lanes -----------
# have() skips everything already recorded, so this re-runs fp8_probe and
# qwen3_32b_lora and nothing else.
if wait_chip && [ "$(left)" -ge 60 ]; then
  log "frontier pass 2 for fp8_probe + qwen3_32b_lora ($(left) min left)"
  bash extras/run_frontier_trn2.sh || log "frontier pass 2 ended early"
  bash shared/bin/push_results.sh trn2 || log "push FAILED"
else
  log "SKIP frontier pass 2 ($(left) min left)"
fi

# ---- 2. release the seed lanes if the window allows -------------------------
if wait_chip && [ "$(left)" -ge 70 ]; then
  log "releasing the deferred seed lanes with $(left) min left"
  for t in llama31_lora_seed43 llama31_lora_seed44 qwen3_lora_seed43; do
    f="$BENCH_DIR/trn2/results/opportunistic/$t.failure.json"
    if [ -s "$f" ] && grep -q '"status": *"deferred"' "$f"; then
      mkdir -p "$BENCH_DIR/trn2/results/opportunistic/deferred"
      mv "$f" "$BENCH_DIR/trn2/results/opportunistic/deferred/" && log "released $t"
    fi
  done
  bash extras/run_opportunistic_trn2.sh || log "opportunistic ended early"
else
  log "SKIP seed lanes ($(left) min left). Nothing is lost that 20 does not"
  log "     already bound better using two physical chips."
fi

log "TAIL COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || log "cache push FAILED"
bash shared/bin/push_results.sh trn2 || log "push FAILED"
