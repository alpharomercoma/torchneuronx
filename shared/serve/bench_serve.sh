#!/usr/bin/env bash
# Grid walker: boot one (model, config) server, walk its declared grid with
# the fallback client under telemetry, emit the triplet per point plus a
# self-declaring grid.json, stop the server. Resumable per point.
#
#   bench_serve.sh <model_key> <config>     # config: short | long | smoke
#
# The grid lives HERE, next to the code that executes it, and every sweep
# directory gets a grid.json declaring exactly what ran and why the grid is
# smaller than the GPU study's (METHODOLOGY rule 6). A reduced grid presented
# as a full one is the failure mode this file exists to prevent.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:?bench_serve.sh expects RESULTS_DIR from run_all.sh}"
MODEL_KEY="${1:?usage: bench_serve.sh <model_key> <config>}"
CONFIG="${2:?usage: bench_serve.sh <model_key> <config>}"

NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
PY="${NP_VENV:+$NP_VENV/bin/python}"
PY="${PY:-python3}"
TELEM="$BENCH_DIR/shared/telemetry.py"
CLIENT="$BENCH_DIR/shared/serve/fallback_client.py"
LAUNCH="$BENCH_DIR/shared/serve/launch_vllm.sh"

case "$CONFIG" in
  short) SHAPES="1024:1024";          CONC="1 4 8 16 32" ;;
  long)  SHAPES="1024:8192 8192:1024"; CONC="1 4 8" ;;
  smoke) SHAPES="128:128";            CONC="1" ;;
  *) echo "unknown config '$CONFIG'" >&2; exit 2 ;;
esac

OUT="$RESULTS_DIR/serve/${MODEL_KEY}_${CONFIG}"
mkdir -p "$OUT"
BOOT_JSON="$OUT/boot.json"

cleanup() { bash "$LAUNCH" stop "$BOOT_JSON" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ------------------------------------------------------------------- boot
if ! bash "$LAUNCH" "$MODEL_KEY" "$CONFIG" "$BOOT_JSON"; then
  # launch_vllm.sh already wrote load_failure.json next to boot.json.
  # A recorded failure is an outcome, not an error -- exit 0 so run_all's
  # lane bookkeeping shows "recorded", and the summarizer picks up the file.
  echo "--- boot failed; load_failure.json recorded in $OUT ---"
  exit 0
fi
MODEL_ID=$("$PY" -c "import json;print(json.load(open('$BOOT_JSON'))['model_id'])")

# ------------------------------------------------------------------- grid
for shape in $SHAPES; do
  ISL="${shape%%:*}"; OSL="${shape##*:}"
  for c in $CONC; do
    TAG="isl${ISL}_osl${OSL}_c${c}"
    if [ -s "$OUT/$TAG.json" ] && [ -s "$OUT/$TAG.telemetry.csv" ]; then
      echo "skip $TAG (triplet exists)"; continue
    fi
    if [ "$CONFIG" = "smoke" ]; then
      NP=4
    else
      NP=$((c * 4)); [ "$NP" -gt 128 ] && NP=128; [ "$NP" -lt 8 ] && NP=8
    fi
    echo "--- point $TAG (num_prompts=$NP) ---"
    "$PY" "$TELEM" --out "$OUT/$TAG.telemetry.csv" -- \
      "$PY" "$CLIENT" --model "$MODEL_ID" \
        --num-prompts "$NP" --max-concurrency "$c" \
        --input-len "$ISL" --output-len "$OSL" \
        --out "$OUT/$TAG.json" \
      > "$OUT/$TAG.log" 2>&1 || echo "  $TAG FAILED (see $TAG.log)"
  done
done

# ---------------------------------------------------------------- declare
# grid.json is written LAST: its presence is the "sweep complete" marker
# that run_all.sh keys resume on.
"$PY" - "$OUT/grid.json" <<PYEOF
import json, sys
from datetime import datetime, timezone
json.dump({
    "model_key": "$MODEL_KEY", "config": "$CONFIG", "model_id": "$MODEL_ID",
    "shapes": "$SHAPES".split(), "concurrency": [int(x) for x in "$CONC".split()],
    "num_prompts_rule": "min(c*4,128) floor 8 (smoke: 4)",
    "client": "fallback_client",
    "reduced": True,
    "reduction_note": ("single-device 32 GB HBM; KV 128 KiB/token bounds "
                       "resident sequences (~12 @ 9216 ctx, ~48 @ 2048 ctx); "
                       "higher concurrencies omitted rather than measured-as-"
                       "queue-depth; host is 4 vCPU (see cpu lane)"),
    "captured": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, open(sys.argv[1], "w"), indent=1)
PYEOF
echo "--- sweep complete: $OUT ---"
