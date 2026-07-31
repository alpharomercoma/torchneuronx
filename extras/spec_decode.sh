#!/usr/bin/env bash
# Track C1: NxDI fused speculative decoding A/B on inf2, no HTTP server --
# inference_demo's benchmark_sampling does the measuring. Draft must share
# the target's vocab: Llama-3.2-1B-Instruct (NOTE: separate HF gate from 3.1;
# a 401 here is a receipt naming the gate, not a bug).
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras/spec_decode"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
export PATH="$NP_VENV/bin:$PATH"  # libneuronpjrt-path must be findable (Phase-1 gotcha #2)
# Gated Llama weights: bare SSM shells have neither HF_HOME nor a token
# (first pass 401'd here -- receipt in git history).
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
bash "$BENCH_DIR/shared/bin/hf_login.sh" >/dev/null 2>&1 || echo "WARN: hf_login failed (gated models will 401)"
DEMO="$NP_VENV/bin/inference_demo"
mkdir -p "$OUT"
# inference_demo's --model-path is a LOCAL directory (it appends '/' and the
# hub then rejects 'repo/id/' -- receipt in git history). Resolve HF ids to
# local snapshot dirs first; the 8B is already in HF_HOME from the serving
# lanes, the 1B draft is a ~2.5 GB one-time pull.
snap() { "$NP_VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))"; }
TARGET=$(snap meta-llama/Llama-3.1-8B-Instruct) || { echo "target snapshot failed"; exit 1; }
DRAFT=$(snap meta-llama/Llama-3.2-1B-Instruct) || { echo "draft snapshot failed"; exit 1; }
echo "target=$TARGET"; echo "draft=$DRAFT"

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
    # plain variable, not xargs: xargs choked on quote chars in tracebacks
    # and wrote a 0-byte receipt (baseline, round 3)
    REASON=$(tail -3 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g" | cut -c1-300)
    printf '{"tag":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
      "$tag" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
    echo "  $tag FAILED (receipt)"
  fi
}

run_case baseline
# Round-3 receipt: fused speculation crashed on
# draft_config.on_device_sampling_config=None -- NxDI's fused path assumes
# on-device sampling is configured. Probe the CLI for the flag spelling this
# build uses and pass it when available; the help dump itself is a receipt.
"$DEMO" --model-type llama --task-type causal-lm run --help \
  > "$OUT/run_help.txt" 2>&1 || true
EXTRA_FUSED=""
grep -q -- "--on-device-sampling" "$OUT/run_help.txt" && EXTRA_FUSED="--on-device-sampling"
run_case fused_spec --draft-model-path "$DRAFT" --enable-fused-speculation \
  --speculation-length 5 $EXTRA_FUSED
echo "spec_decode lanes recorded"
