#!/usr/bin/env bash
###############################################################################
#                                                                             #
#   PRECOMPILE NUMERICS ARE INVALID.                                          #
#                                                                             #
#   neuron_parallel_compile runs the training script with                     #
#   NEURON_EXTRACT_GRAPHS_ONLY=1. In that mode the Neuron runtime does NOT     #
#   execute the graphs -- it traces them, hands them to the compiler, and      #
#   feeds the script back zeros. Every loss, gradient and metric produced      #
#   during this phase is meaningless. That is not a bug and it is not a        #
#   degraded mode: it is the entire point, because tracing without executing   #
#   is what makes it cheap.                                                    #
#                                                                             #
#   So this lane records exactly one class of number: HOW LONG COMPILATION     #
#   TOOK and HOW MUCH CACHE IT PRODUCED. The training metrics JSON that        #
#   sft_lora.py writes here goes to /tmp and is deleted before this script     #
#   exits, so that a stray precompile loss curve can never be mistaken for a   #
#   result. This is pre-registered in shared/run_all.sh lane 3.                #
#                                                                             #
###############################################################################
#
# Usage:
#     bash shared/train/precompile.sh <model_id> <out_json>
#
#     bash shared/train/precompile.sh meta-llama/Llama-3.1-8B-Instruct \
#         trn1/results/compile/llama31_train.json
#     # -> the invalid-numerics banner, then compiler output for many minutes,
#     #    then:
#     #      PRECOMPILE COMPLETE  wall_s=1683.4  graphs_compiled=214
#     #        cache_dir_size_mb=1902
#     #    and trn1/results/compile/llama31_train.json:
#     #      {"model": "...", "wall_s": 1683.4, "graphs_compiled": 214, ...}
#
# WHY THIS LANE EXISTS AT ALL
# ---------------------------
# The first optimizer step of an XLA program pays for building and compiling
# the graph. On an 8B model that is tens of minutes. If it happened inside
# lane 4 it would sit inside first_step_ms, and every derived number --
# median step time, tokens/s, MFU -- would be measuring the compiler instead
# of the accelerator. Paying it here, into a persistent NEFF cache, is what
# lets lane 4 measure training.
#
# It also makes compile cost a first-class result rather than a hidden tax.
# "Time to first token of training" is a real operational number for anyone
# choosing Trainium, and it is invisible if you only ever report steady-state
# throughput.
#
# WHY graphs_compiled IS A DIFF, NOT A COUNT
# ------------------------------------------
# The NEFF cache is shared across lanes and is restored from S3 by
# shared/bin/sync_neuron_cache.sh, so it is rarely empty. Counting the *.neff
# files afterwards would report the whole cache. Counting the ones this run
# added is the number that answers "did the cache hit?" -- a warm cache gives
# graphs_compiled near 0 with a small wall_s, and that reading is itself the
# evidence that lane 4's fast first step was cache, not luck.
set -uo pipefail

MODEL_ID="${1:?usage: precompile.sh <model_id> <out_json>}"
OUT_JSON="${2:?usage: precompile.sh <model_id> <out_json>}"

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SFT_LORA="$BENCH_DIR/shared/train/sft_lora.py"

# Same venv resolution as shared/run_all.sh: the DLAMI ships one venv per
# framework and the training box's is the torch-neuronx one.
NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
PY="${NP_VENV:+$NP_VENV/bin/python}"
PY="${PY:-python3}"

# neuron_parallel_compile is a console script installed by libneuronxla into
# the same venv. Prefer the venv copy over PATH: several Neuron venvs coexist
# on a DLAMI and PATH order decides which one you would otherwise get.
NPC="${NP_VENV:+$NP_VENV/bin/neuron_parallel_compile}"
if [ -z "$NPC" ] || [ ! -x "$NPC" ]; then
  NPC="$(command -v neuron_parallel_compile || true)"
fi
if [ -z "$NPC" ]; then
  echo "FATAL: neuron_parallel_compile not found (looked in ${NP_VENV:-<no venv>}/bin and PATH)" >&2
  exit 3
fi

CACHE_DIR="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
mkdir -p "$CACHE_DIR"

# Discardable outputs. Both are removed on exit -- see the banner.
DISCARD_JSON="/tmp/precompile_discard.json"
DISCARD_ADAPTER="/tmp/precompile_discard_adapter"
cleanup() { rm -rf "$DISCARD_JSON" "$DISCARD_ADAPTER"; }
trap cleanup EXIT

count_neffs() { find "$CACHE_DIR" -name '*.neff' 2>/dev/null | wc -l | tr -d ' '; }

cat <<'BANNER'

###############################################################################
#  PRECOMPILE PHASE -- NUMERICS ARE INVALID                                   #
#  NEURON_EXTRACT_GRAPHS_ONLY=1: graphs are traced and compiled, never run.   #
#  Any loss printed below is a placeholder and is discarded. The only          #
#  results this lane produces are compile wall time and NEFF cache growth.    #
###############################################################################

BANNER

echo "model      : $MODEL_ID"
echo "cache dir  : $CACHE_DIR"
echo "python     : $PY"
echo "compiler   : $NPC"
echo "out json   : $OUT_JSON"
echo

NEFFS_BEFORE="$(count_neffs)"
CACHE_MB_BEFORE="$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1)"
CACHE_MB_BEFORE="${CACHE_MB_BEFORE:-0}"
echo "neff files before : $NEFFS_BEFORE  (cache ${CACHE_MB_BEFORE} MB)"
echo

# --max-steps 10 matches the graph coverage upstream uses: enough steps to
# trace the warmup graph, the steady-state training graph and the optimizer
# graph, without paying for a real epoch of tracing.
START_S=$(date +%s.%N)
"$NPC" "$PY" "$SFT_LORA" \
  --model "$MODEL_ID" \
  --tag precompile \
  --max-steps 10 \
  --out "$DISCARD_JSON" \
  --adapter-out "$DISCARD_ADAPTER"
RC=$?
END_S=$(date +%s.%N)

WALL_S=$(awk -v a="$START_S" -v b="$END_S" 'BEGIN{printf "%.2f", b-a}')

NEFFS_AFTER="$(count_neffs)"
CACHE_MB_AFTER="$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1)"
CACHE_MB_AFTER="${CACHE_MB_AFTER:-0}"
GRAPHS_COMPILED=$((NEFFS_AFTER - NEFFS_BEFORE))

if [ "$GRAPHS_COMPILED" -le 0 ]; then
  CACHE_HITS_NOTE="No new NEFFs were produced: every graph this model needs was already in the cache, so lane 4 should show a first step close to its median. wall_s here is cache-lookup and graph-tracing time only, not compilation."
else
  CACHE_HITS_NOTE="Compiled $GRAPHS_COMPILED new NEFFs into the cache (was $NEFFS_BEFORE, now $NEFFS_AFTER). wall_s is the compile cost lane 4 no longer has to pay. Push the cache with shared/bin/sync_neuron_cache.sh push so a replaced instance never pays it twice."
fi

mkdir -p "$(dirname "$OUT_JSON")"
TMP_JSON="${OUT_JSON}.tmp.$$"
cat > "$TMP_JSON" <<EOF
{
  "lane": "precompile",
  "model": "$MODEL_ID",
  "exit_code": $RC,
  "wall_s": $WALL_S,
  "cache_dir": "$CACHE_DIR",
  "cache_dir_size_mb": $CACHE_MB_AFTER,
  "cache_dir_size_mb_before": $CACHE_MB_BEFORE,
  "neff_files_before": $NEFFS_BEFORE,
  "neff_files_after": $NEFFS_AFTER,
  "graphs_compiled": $GRAPHS_COMPILED,
  "cache_hits_note": "$CACHE_HITS_NOTE",
  "numerics_note": "NEURON_EXTRACT_GRAPHS_ONLY=1 -- graphs were traced and compiled but never executed. No loss, gradient or throughput value from this phase is recorded anywhere; the training metrics JSON was written to $DISCARD_JSON and deleted.",
  "max_steps": 10,
  "captured": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
mv -f "$TMP_JSON" "$OUT_JSON"

echo
if [ "$RC" -ne 0 ]; then
  echo "PRECOMPILE FAILED (exit $RC) -- see the log; $OUT_JSON records the failure" >&2
else
  echo "PRECOMPILE COMPLETE  wall_s=$WALL_S  graphs_compiled=$GRAPHS_COMPILED  cache_dir_size_mb=$CACHE_MB_AFTER"
fi
echo "REMINDER: every numeric value produced by the training script during this phase was invalid and has been discarded."
echo "wrote $OUT_JSON"

exit "$RC"
