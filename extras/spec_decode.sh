#!/usr/bin/env bash
# Track C1: NxDI fused speculative decoding A/B on inf2, no HTTP server --
# inference_demo's benchmark_sampling does the measuring. Draft must share
# the target's vocab: Llama-3.2-1B-Instruct (NOTE: separate HF gate from 3.1;
# a 401 here is a receipt naming the gate, not a bug).
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras/spec_decode"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
DEMO="$NP_VENV/bin/inference_demo"
mkdir -p "$OUT"
TARGET=meta-llama/Llama-3.1-8B-Instruct
DRAFT=meta-llama/Llama-3.2-1B-Instruct

run_case() {  # $1 tag  $2... extra args
  local tag="$1"; shift
  [ -s "$OUT/$tag.json" ] || [ -s "$OUT/$tag.failure.json" ] && { echo "skip $tag"; return; }
  echo "############ spec_decode: $tag ############"
  if "$DEMO" --model-type llama --task-type causal-lm run \
       --model-path "$TARGET" --compiled-model-path "/opt/np/models/neuron-compiled/spec_$tag" \
       --torch-dtype bfloat16 --tp-degree 2 --max-context-length 1024 --seq-len 1280 \
       --batch-size 1 --benchmark --prompt "The quick brown fox" "$@" \
       > "$OUT/$tag.log" 2>&1; then
    grep -E "latency|throughput|token" "$OUT/$tag.log" | tail -20 > "$OUT/$tag.extract.txt"
    printf '{"tag":"%s","status":"ran","metrics_extract":"%s.extract.txt","captured":"%s"}\n' \
      "$tag" "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.json"
  else
    tail -3 "$OUT/$tag.log" | tr '\n' ' ' | sed 's/"/\x27/g' | cut -c1-300 \
      | xargs -I{} printf '{"tag":"%s","status":"failed","reason":"{}","captured":"%s"}\n' \
        "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
    echo "  $tag FAILED (receipt)"
  fi
}

run_case baseline
run_case fused_spec --draft-model-path "$DRAFT" --enable-fused-speculation --speculation-length 5
echo "spec_decode lanes recorded"
