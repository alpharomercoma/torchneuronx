#!/usr/bin/env bash
# Sustained lane: fixed load against one warm server until the wall clock
# says stop. Each iteration is its own result file; parsing happens later in
# parse_sustained.py -- deliberately decoupled, because a prior study lost a
# soak run to inline parsing (see that file's docstring).
#
#   sustained.sh <model_key> <minutes>      # config short, conc 8, 1024:1024
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:?sustained.sh expects RESULTS_DIR from run_all.sh}"
MODEL_KEY="${1:?usage: sustained.sh <model_key> <minutes>}"
MINUTES="${2:?usage: sustained.sh <model_key> <minutes>}"

NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
PY="${NP_VENV:+$NP_VENV/bin/python}"
PY="${PY:-python3}"
OUT="$RESULTS_DIR/sustained"
mkdir -p "$OUT"
BOOT_JSON="$OUT/boot.json"
LAUNCH="$BENCH_DIR/shared/serve/launch_vllm.sh"

cleanup() { bash "$LAUNCH" stop "$BOOT_JSON" >/dev/null 2>&1 || true; }
trap cleanup EXIT

bash "$LAUNCH" "$MODEL_KEY" short "$BOOT_JSON" || {
  echo "--- boot failed; see $OUT ---"; exit 0; }
MODEL_ID=$("$PY" -c "import json;print(json.load(open('$BOOT_JSON'))['model_id'])")

END=$(( $(date +%s) + MINUTES * 60 ))
i=1
while [ "$(date +%s)" -lt "$END" ]; do
  TAG=$(printf "iter_%03d" "$i")
  "$PY" "$BENCH_DIR/shared/telemetry.py" --out "$OUT/$TAG.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/serve/fallback_client.py" --model "$MODEL_ID" \
      --num-prompts 32 --max-concurrency 8 --input-len 1024 --output-len 1024 \
      --out "$OUT/$TAG.json" \
    > "$OUT/$TAG.log" 2>&1 || echo "  $TAG FAILED (recorded; loop continues)"
  i=$((i + 1))
done
echo "--- sustained complete: $((i - 1)) iterations over $MINUTES min ---"
