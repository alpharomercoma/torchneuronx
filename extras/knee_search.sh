#!/usr/bin/env bash
# Track D: capacity-knee search on warm llama31_base with the Poisson
# open-loop client (MLPerf Server-style arrivals + goodput@SLO).
#
# One server boot, then a ladder of offered rates; each point is a triplet
# (result JSON + client log + telemetry CSV). The knee is read off the curve
# in the REPORT addendum: the last rate where SLO attainment holds, and the
# first where it collapses (declared SLOs: TTFT < 3 s, TPOT < 80 ms).
# Fixed-duration open-loop => offered load is rate-driven, not conc-driven;
# ISL/OSL 512/128 declared. Resumable per point.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras/knee"
LAUNCH="$BENCH_DIR/shared/serve/launch_vllm.sh"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
export PATH="$NP_VENV/bin:$PATH"  # libneuronpjrt-path must be findable (Phase-1 gotcha #2)
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
CLIENT="$BENCH_DIR/shared/serve/fallback_client.py"
RATES="${KNEE_RATES:-0.5 1 2 4 6 8}"
DUR="${KNEE_DURATION_S:-120}"
mkdir -p "$OUT"

# Skip the boot entirely if every point is already recorded.
ALL=1; for R in $RATES; do [ -s "$OUT/rate_${R}.json" ] || ALL=0; done
[ "$ALL" = "1" ] && { echo "knee: all points recorded, skipping"; exit 0; }

BOOT="$OUT/boot.json"
if ! bash "$LAUNCH" llama31_base short "$BOOT" > "$OUT/boot.log" 2>&1; then
  echo "knee: server boot FAILED (load_failure recorded by launch_vllm)"; exit 0
fi
MODEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_id'])" "$BOOT")

for R in $RATES; do
  RES="$OUT/rate_${R}.json"
  [ -s "$RES" ] && { echo "skip rate=$R (recorded)"; continue; }
  echo "############ knee: poisson rate=$R req/s x ${DUR}s ############"
  "$PY" "$TELEM" --out "$OUT/rate_${R}.telemetry.csv" -- \
    "$PY" "$CLIENT" --base-url http://127.0.0.1:8000 --model "$MODEL" \
      --arrival-mode poisson --request-rate "$R" --duration-s "$DUR" \
      --num-prompts 4096 --max-concurrency 64 \
      --input-len 512 --output-len 128 --seed 42 \
      --slo-ttft-ms 3000 --slo-tpot-ms 80 \
      --out "$RES" > "$OUT/rate_${R}.log" 2>&1 \
    || echo "  rate=$R FAILED (see rate_${R}.log)"
done
bash "$LAUNCH" stop "$BOOT" >/dev/null 2>&1 || true
echo "knee search complete"
