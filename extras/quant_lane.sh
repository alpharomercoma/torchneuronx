#!/usr/bin/env bash
# Track B2: quantization attempts on inf2, cheapest-first.
#   fp8 KV-cache: no checkpoint prep needed -> boot + one c8 point A/B.
#   int8 weights: needs a pre-quantized checkpoint; attempted, receipt if the
#   prep path isn't viable on this box (declared, not silent).
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras/quant"
LAUNCH="$BENCH_DIR/shared/serve/launch_vllm.sh"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
export PATH="$NP_VENV/bin:$PATH"  # libneuronpjrt-path must be findable (Phase-1 gotcha #2)
PY="$NP_VENV/bin/python"
CLIENT="$BENCH_DIR/shared/serve/fallback_client.py"
mkdir -p "$OUT"

point() {  # $1 tag  (server must be up)
  "$PY" "$BENCH_DIR/shared/telemetry.py" --out "$OUT/$1.telemetry.csv" -- \
    "$PY" "$CLIENT" --model meta-llama/Llama-3.1-8B-Instruct \
      --num-prompts 32 --max-concurrency 8 --input-len 1024 --output-len 1024 \
      --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1
}

if [ ! -s "$OUT/fp8kv_c8.json" ] && [ ! -s "$OUT/fp8kv_failure.json" ]; then
  echo "############ quant: fp8 KV-cache attempt ############"
  B="$OUT/fp8kv.boot.json"
  if OVERRIDE_NEURON_CONFIG='{"kv_cache_quant": true}' \
     bash "$LAUNCH" llama31_base short "$B" > "$OUT/fp8kv.launch.log" 2>&1; then
    point fp8kv_c8 || echo "  fp8kv point FAILED"
    bash "$LAUNCH" stop "$B" >/dev/null 2>&1
  else
    tail -1 "${B%.json}.server.tail.log" 2>/dev/null | cut -c1-200 \
      | sed 's/"/\x27/g' | xargs -I{} printf '{"status":"boot_failed","reason":"{}","captured":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/fp8kv_failure.json"
    echo "  fp8kv boot FAILED (receipt)"
  fi
fi
echo "int8 weights: requires quantized_checkpoints_path prep (NxDI); attempt via inference_demo quantize flow is manual-stage -- recorded as todo-receipt if absent"
[ -s "$OUT/int8_note.json" ] || printf '{"status":"not_attempted_this_pass","reason":"int8 needs offline checkpoint quantization stage; scheduled separately","captured":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/int8_note.json"
