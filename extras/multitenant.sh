#!/usr/bin/env bash
# Track C2: two TP=1 servers on one Inferentia2 via NEURON_RT_VISIBLE_CORES.
# TinyLlama x2 (ungated; the mechanism is the demo, not the model). Measures
# each tenant under load simultaneously + notes isolation.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras/multitenant"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
export PATH="$NP_VENV/bin:$PATH"  # libneuronpjrt-path must be findable (Phase-1 gotcha #2)
PY="$NP_VENV/bin/python"; VLLM="$NP_VENV/bin/vllm"
CLIENT="$BENCH_DIR/shared/serve/fallback_client.py"
M=TinyLlama/TinyLlama-1.1B-Chat-v1.0
mkdir -p "$OUT"
[ -s "$OUT/tenant0_c4.json" ] && { echo "skip multitenant (recorded)"; exit 0; }
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference NEURON_SKIP_EFA_AFFINITY=1
start() { # $1 core  $2 port
  NEURON_RT_VISIBLE_CORES="$1" nohup "$VLLM" serve "$M" --tensor-parallel-size 1 \
    --max-model-len 2048 --max-num-seqs 8 --port "$2" \
    > "$OUT/server_core$1.log" 2>&1 & echo $!
}
P0=$(start 0 8000); P1=$(start 1 8001)
trap 'kill $P0 $P1 2>/dev/null' EXIT
for i in $(seq 1 360); do
  ok0=$(curl -sf localhost:8000/health >/dev/null 2>&1 && echo 1 || echo 0)
  ok1=$(curl -sf localhost:8001/health >/dev/null 2>&1 && echo 1 || echo 0)
  [ "$ok0$ok1" = "11" ] && break; sleep 5
done
if [ "${ok0:-0}${ok1:-0}" != "11" ]; then
  printf '{"status":"dual_boot_failed","core0_healthy":%s,"core1_healthy":%s,"captured":"%s"}\n' \
    "${ok0:-0}" "${ok1:-0}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/load_failure.json"
  echo "dual boot FAILED (receipt)"; exit 0
fi
echo "both tenants healthy; concurrent load"
"$PY" "$CLIENT" --base-url http://localhost:8000 --model "$M" --num-prompts 32 \
  --max-concurrency 4 --input-len 256 --output-len 256 --out "$OUT/tenant0_c4.json" > "$OUT/tenant0.log" 2>&1 &
C0=$!
"$PY" "$CLIENT" --base-url http://localhost:8001 --model "$M" --num-prompts 32 \
  --max-concurrency 4 --input-len 256 --output-len 256 --out "$OUT/tenant1_c4.json" > "$OUT/tenant1.log" 2>&1 &
wait $C0; wait
echo "multitenant complete"
