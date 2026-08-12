#!/usr/bin/env bash
# extras/lib/common.sh — THE single copy of the driver helpers.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
#
# WHY THIS EXISTS
# ---------------
# These helpers were copy-pasted into ~16 drivers, which is how the fleet
# acquired FIVE different definitions of "is this lane already done?" and three
# of "how long until the deadline?". Two real defects in this study traced
# directly to that drift:
#
#   * `have()` treated any non-empty receipt as a terminal outcome, so receipts
#     written during a forced shutdown silently skipped lanes that had never
#     actually run (three lanes, §29.4).
#   * deadline arithmetic was re-derived per driver; one variant compared a
#     LOCAL timestamp to a UTC one and reported -479 minutes, disabling its own
#     staleness alarm.
#
# One definition, one place to fix. Every function here is intentionally small
# and side-effect-free apart from the ones that obviously are not.
set -uo pipefail

# ---- paths ------------------------------------------------------------------
# Resolve BENCH_DIR from THIS file, so a driver works from any cwd and from
# systemd (which starts in /).
NP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${BENCH_DIR:-$(cd "$NP_LIB_DIR/../.." && pwd)}"
export BENCH_DIR

# ---- logging ----------------------------------------------------------------
np_log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${NP_TAG:-driver}: $*"; }
np_step() { echo; echo "############ ${NP_TAG:-LANE}: $* ############"; echo; }
np_die()  { np_log "FATAL: $*"; exit "${2:-3}"; }

# ---- resumability -----------------------------------------------------------
# A lane is "recorded" if it produced EITHER a result or a failure receipt.
# FORCE=1 re-runs regardless.
#
# CAVEAT, learned the hard way: a receipt is only trustworthy if it was written
# by the lane itself. Receipts produced while lanes were being killed are
# artifacts, not outcomes — always review receipts written during a forced
# shutdown before trusting them (§29.4).
np_have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
np_recorded() { np_have "$1.json" || np_have "$1.failure.json"; }

# ---- deadlines --------------------------------------------------------------
# ALL times UTC. Mixing local and UTC produced a negative age once and silently
# disabled a staleness alarm.
np_epoch_of() {   # np_epoch_of 2026-08-06T10:00:00Z -> epoch seconds
  python3 -c "
import datetime,sys
print(int(datetime.datetime.strptime(sys.argv[1],'%Y-%m-%dT%H:%M:%SZ')
      .replace(tzinfo=datetime.timezone.utc).timestamp()))" "$1"
}
np_mins_left() { echo $(( ($1 - $(date -u +%s)) / 60 )); }

# Refuse to START work that cannot FINISH. Being killed mid-lane produces no
# artifact at all, so a lane that cannot finish is worse than one not started:
# it burns window another lane could have used.
np_fits() {   # np_fits <stop_epoch> <needed_minutes> <label>
  local left; left=$(np_mins_left "$1")
  if [ "$left" -lt "$2" ]; then
    np_log "SKIP $3 — ${left} min left, needs ~${2}"
    return 1
  fi
  np_log "$3 — ${left} min left, needs ~${2}, proceeding"
  return 0
}

# ---- chip arbitration -------------------------------------------------------
# Every training lane needs the whole chip. Two lanes sharing it do not merely
# run slow, they corrupt BOTH measurements, since each times a chip it does not
# exclusively own.
#
# THE PATTERN DELIBERATELY DOES NOT MATCH DRIVER SCRIPTS.
# It used to also match `run_.*\.sh`, and since every driver IS a run_*.sh
# process, the first driver to call this helper waited forever for ITSELF. The
# signature is a process tree of exactly `bash(PID)---sleep(PID)` with a driver
# log that stops after the first banner and a lane log that was never created.
#
# Matching the training PROCESS rather than the driver is also more correct: a
# driver sitting between lanes holds nothing, while a driver running a lane
# always has an sft_lora.py/train_academic.py child that matches. np_ancestry
# is kept as a second line of defence for anything that re-broadens the pattern.
np_ancestry() {   # PIDs of this process and every ancestor, space separated
  local p="${BASHPID:-$$}" out=""
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    out="$out $p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d '[:space:]')
  done
  echo "$out"
}

np_chip_busy() {   # np_chip_busy "<space-padded PIDs to ignore>" -> true if SOMEONE ELSE has it
  local mine="${1:- }" pid
  for pid in $(pgrep -f 'sft_lora.py|train_academic.py' 2>/dev/null); do
    case "$mine" in *" $pid "*) continue;; esac
    return 0
  done
  return 1
}

np_wait_for_chip() {   # np_wait_for_chip [stop_epoch]
  local stop="${1:-0}" waited=0 mine
  mine=" $(np_ancestry) "
  while np_chip_busy "$mine"; do
    sleep 30; waited=$((waited+1))
    [ $((waited % 20)) -eq 0 ] && np_log "waiting for the chip ($((waited/2)) min)"
    [ "$stop" -gt 0 ] && [ "$(date -u +%s)" -ge "$stop" ] && { np_log "hard stop while waiting"; return 1; }
  done
  np_wait_for_port
  np_clear_compile_locks
  sleep 5
  return 0
}

# torch.distributed defaults to 29500. A process can be GONE while its listening
# socket is not, and the next lane then dies instantly with EADDRINUSE — which
# is what silently reduced the multi-tenancy experiment to a single job running
# alone (§27.4).
np_wait_for_port() {   # np_wait_for_port [port] [max_seconds]
  local port="${1:-29500}" max="${2:-600}" n=0
  while ss -ltn 2>/dev/null | grep -q ":${port} "; do
    sleep 10; n=$((n+10))
    [ $((n % 60)) -eq 0 ] && np_log "port $port still held (${n}s)"
    [ "$n" -ge "$max" ] && { np_log "port $port held for ${max}s — proceeding, receipt will show it"; return 1; }
  done
  return 0
}

# A killed compile leaves a lock behind and the NEXT lane deadlocks on it,
# advancing its log forever while no compiler runs. That cost three hours.
np_clear_compile_locks() {
  find "${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}" \
       -name '*.lock' -delete 2>/dev/null || true
}

# ---- receipts ---------------------------------------------------------------
# Capture the PROGRAM's error, never the wrapper's banner. A receipt recording
# torchrun's ChildFailedError hides the ValueError that actually explains the
# failure — that mistake cost a full re-read of the MoE lane (§24.5).
np_receipt() {   # np_receipt <out.failure.json> <tag> <box> <logfile>
  local out="$1" tag="$2" box="$3" log="$4" reason
  reason=$(grep -m1 -oE "ValueError: [^\"]{0,180}|NotImplementedError: [^\"]{0,160}|NCC_[A-Z0-9]+[^\"]{0,150}|\[F[0-9]+\][^\"]{0,140}|RuntimeError: [^\"]{0,140}" "$log" 2>/dev/null | head -1 | sed "s/\"/'/g")
  [ -n "$reason" ] || reason=$(tail -c 300 "$log" 2>/dev/null | tr '\n' ' ' | sed "s/\"/'/g")
  printf '{"tag":"%s","box":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
    "$tag" "$box" "$reason" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out"
  np_log "$tag FAILED — receipt written"
}
