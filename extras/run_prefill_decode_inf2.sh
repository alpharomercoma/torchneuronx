#!/usr/bin/env bash
# Separate PREFILL from DECODE on inf2 -- the one real gap in the serving metrics.
#
# WHAT IS ALREADY MEASURED (and measured well)
# --------------------------------------------
# fallback_client.py already records, per request: TTFT, TPOT ((E2EL-TTFT)/
# (tokens-1), the same definition vllm bench uses), ITL distributions, E2EL,
# output and total token throughput, and goodput@SLO -- swept over concurrency
# with Poisson arrivals. That is more than most published serving numbers.
#
# WHAT IS MISSING
# ---------------
# Every existing grid point mixes the two phases. TTFT bundles queueing +
# prefill + the first decode step; TPOT is decode diluted by whatever prefill
# cost leaked into TTFT. So the existing numbers cannot answer the question the
# roofline analysis raises:
#
#   PREFILL is compute-bound   -- it processes the whole prompt in parallel,
#                                 arithmetic intensity scales with prompt length
#   DECODE  is memory-bound    -- one token at a time, so every weight is read
#                                 from HBM to produce a single token; intensity
#                                 is ~1 FLOP/byte regardless of model size
#
# That split is WHY an inference chip is specified the way it is, and it is the
# cleanest illustration of compute-bound vs memory-bound in the whole study.
# Without it, "Inferentia2 is good at X" has no mechanism attached.
#
# HOW THE ISOLATION WORKS
# -----------------------
#   prefill grid  ISL sweep at OSL=1   -> one output token, so E2EL is
#                 essentially the prefill pass. Throughput vs prompt length
#                 traces the compute-bound curve.
#   decode grid   ISL 128, OSL 2048    -> prefill is ~6% of the tokens, so TPOT
#                 is close to the pure per-token decode cost.
#
# Neither replaces the existing short/long sweeps; both are additive lanes with
# their own grid.json, so nothing published is restated.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/inf2/results}"
NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
export NP_VENV
export NP_DEVICE=inf2
export NP_REGION="${NP_REGION:-us-west-2}"

have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ INF2 PHASE-SPLIT: $* ############"; echo; }

# NEVER install optimum-neuron into the vLLM venv: duelling platform plugins
# brick all serving on this box. This driver only runs the existing server and
# client, and installs nothing.

for cfg in prefill decode; do
  D="$RESULTS_DIR/serve/llama31_base_${cfg}"
  if have "$D/grid.json" || have "$D/load_failure.json"; then
    echo "skip $cfg (outcome recorded)"; continue
  fi
  step "$cfg grid (Llama 3.1 8B base)"
  bash "$BENCH_DIR/shared/serve/bench_serve.sh" llama31_base "$cfg" \
    > "$RESULTS_DIR/serve/llama31_base_${cfg}.log" 2>&1 \
    || echo "  $cfg grid FAILED (see serve/llama31_base_${cfg}.log)"
done

step "INF2 PHASE-SPLIT COMPLETE"
bash shared/bin/push_results.sh inf2 || echo "push FAILED"
