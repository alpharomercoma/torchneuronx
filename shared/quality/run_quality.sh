#!/usr/bin/env bash
# Quality lane: greedy determinism + logprobs over the 16 fixed prompts,
# against a freshly booted short-config server. Boot logic is launch_vllm.sh's
# job -- this file never duplicates it.
#
#   run_quality.sh <model_key>       # -> $RESULTS_DIR/quality/<model_key>.json
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:?run_quality.sh expects RESULTS_DIR from run_all.sh}"
MODEL_KEY="${1:?usage: run_quality.sh <model_key>}"

NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
PY="${NP_VENV:+$NP_VENV/bin/python}"
PY="${PY:-python3}"
OUT="$RESULTS_DIR/quality"
mkdir -p "$OUT"
BOOT_JSON="$OUT/${MODEL_KEY}_boot.json"
LAUNCH="$BENCH_DIR/shared/serve/launch_vllm.sh"

if [ -s "$OUT/$MODEL_KEY.json" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "skip quality $MODEL_KEY (exists)"; exit 0
fi

cleanup() { bash "$LAUNCH" stop "$BOOT_JSON" >/dev/null 2>&1 || true; }
trap cleanup EXIT

bash "$LAUNCH" "$MODEL_KEY" short "$BOOT_JSON" || {
  echo "--- boot failed; recorded ---"; exit 0; }
MODEL_ID=$("$PY" -c "import json;print(json.load(open('$BOOT_JSON'))['model_id'])")

"$PY" "$BENCH_DIR/shared/quality/eval.py" \
  --out "$OUT/$MODEL_KEY.json" --model "$MODEL_ID" \
  --base-url http://localhost:8000 --max-tokens 128
echo "--- quality complete: $OUT/$MODEL_KEY.json ---"
