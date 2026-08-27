#!/usr/bin/env bash
# Measure the DRAFT model alone, in byte-identical configuration to the 8B
# baseline, so r = draft_ms_per_token / target_ms_per_token is a MEASUREMENT
# rather than a back-solve.
#
# The published speculative-decoding speedups were derived: acceptance came
# from SpecDecode-Bench (39 prompts) and latency from the 8B baseline, joined
# through speedup(k) = E[accepted] / (1 + k*r) with r ASSUMED. The earlier
# draft_only attempt produced a 0-byte log, so r was never measured.
#
# Everything below is copied from extras/spec_decode.sh's run_case baseline --
# same tp-degree, max-context-length, seq-len, batch-size, dtype and prompt.
# Only --model-path changes. If any of those drift the ratio is meaningless.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
OUT="${RESULTS_DIR:-$BENCH_DIR/trn1/results}/specdec"
mkdir -p "$OUT"

# trn1 has no vLLM venv; inference_demo lives in the nxd_inference one.
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"
export PATH="$NP_VENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
DEMO="$NP_VENV/bin/inference_demo"
[ -x "$DEMO" ] || { echo "FATAL: no inference_demo at $DEMO"; exit 3; }

bash "$BENCH_DIR/shared/bin/hf_login.sh" >/dev/null 2>&1 || echo "WARN: hf_login failed"
snap() { "$NP_VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))"; }
DRAFT=$(snap meta-llama/Llama-3.2-1B-Instruct) || { echo "draft snapshot failed"; exit 3; }
echo "draft=$DRAFT"

tag=draft_baseline
if "$DEMO" --model-type llama --task-type causal-lm run \
     --model-path "$DRAFT" \
     --compiled-model-path "/opt/np/models/neuron-compiled/spec_${tag}" \
     --torch-dtype bfloat16 --tp-degree 2 --max-context-length 1024 --seq-len 1280 \
     --batch-size 1 --benchmark --prompt "The quick brown fox" \
     > "$OUT/$tag.log" 2>&1; then
  [ -s benchmark_report.json ] && cp benchmark_report.json "$OUT/$tag.report.json"
  grep -E "latency|throughput|token" "$OUT/$tag.log" | tail -20 > "$OUT/$tag.extract.txt"
  printf '{"tag":"%s","status":"ran","captured":"%s"}\n' "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.json"
  echo "DRAFT BASELINE RAN"
else
  REASON=$(tail -3 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g" | cut -c1-300)
  printf '{"tag":"%s","status":"failed","reason":"%s","captured":"%s"}\n' "$tag" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
  echo "DRAFT BASELINE FAILED: $REASON"
fi
