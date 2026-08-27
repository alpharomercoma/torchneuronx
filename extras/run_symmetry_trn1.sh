#!/usr/bin/env bash
# TRN1 SYMMETRY PASS -- the lanes trn2 ran that trn1 had not.
#
#   bash extras/run_symmetry_trn1.sh
#
# WHY
# ---
# The study's rule is that what runs on one chip runs on the other. A lane
# audit found 33 lanes on both chips and a set that existed only on trn2 --
# the frontier lanes, the efficiency sweeps, and the FP8 probe. Without their
# trn1 counterparts, every one of those results is a single-chip observation
# that cannot be attributed to the GENERATION rather than to the workload.
#
# The MoE lane is the clearest case. On trn2 it failed with
#   "Model type qwen3_moe is not supported ... Supported types are:
#    ['llama','granite','qwen3']"
# which reads like a Neuron-wide capability limit -- but on ONE chip it is
# equally consistent with a v3-specific gap. Running it on trn1 costs about a
# minute and settles which. That asymmetry is exactly the kind that turns into
# an overclaim in a report.
#
# trn1 is on credits and idle, so this costs nothing but wall clock.
set -uo pipefail

BOX=trn1
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"
export NP_DEVICE=trn1 NP_REGION=us-west-2 NP_CACHE_PREFIX=neuron-cache
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --retry_failed_compilation"

FRONTIER="$BENCH_DIR/trn1/results/frontier"
EXTRAS="$BENCH_DIR/trn1/results/extras"
mkdir -p "$FRONTIER" "$EXTRAS" "$BENCH_DIR/trn1/results/academic"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ SYMMETRY[trn1]: $* ############"; echo; }

bash shared/bin/hf_login.sh >/dev/null 2>&1 || echo "WARN: hf_login failed"

lane() {  # lane <dir> <tag> <args...>
  local dir="$1" tag="$2"; shift 2
  { have "$dir/$tag.json" || have "$dir/$tag.failure.json"; } && { echo "skip $tag"; return 0; }
  step "$tag"
  find /opt/np/cache/neuron-compile-cache -name '*.lock' -delete 2>/dev/null
  "$PY" "$TELEM" --out "$dir/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" --out "$dir/$tag.json" "$@" \
    > "$dir/$tag.log" 2>&1 || {
      # Capture the PROGRAM's error, not torchrun's wrapper banner. A receipt
      # that records ChildFailedError hides the ValueError that actually
      # explains the failure -- that mistake cost a re-read of the MoE lane.
      REASON=$(grep -m1 -oE "ValueError: [^\"]{0,180}|NCC_[A-Z0-9]+[^\"]{0,140}|RuntimeError: [^\"]{0,140}" "$dir/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$dir/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","box":"trn1","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dir/$tag.failure.json"
      echo "  $tag FAILED (receipt)"; }
  bash shared/bin/push_results.sh trn1 >/dev/null 2>&1 || echo "  push FAILED"
}

# ---- cheapest and most decisive first ---------------------------------------
# Architecture rejection: seconds, and it settles whether the MoE limit is
# Neuron-wide or v3-specific.
lane "$FRONTIER" moe_qwen3_30b_a3b --model Qwen/Qwen3-30B-A3B --max-steps 50

# FP8: Trainium1 documents cFP8 support, so whether the SAME flag works on both
# generations is a real question, not a formality.
NEURON_CC_FLAGS="$NEURON_CC_FLAGS --auto-cast-type=fp8_e4m3" \
  lane "$FRONTIER" fp8_probe --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-steps 20

# Full fine-tuning vs LoRA -- the LoRA saving is only meaningful against it.
lane "$FRONTIER" fullft_tinyllama --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --no-lora --max-steps 50
lane "$FRONTIER" fullft_qwen3_1_7b --model Qwen/Qwen3-1.7B --no-lora --max-steps 50

# Dataset and packing sensitivity.
lane "$FRONTIER" data_alpaca --dataset tatsu-lab/alpaca \
  --dataset-fields "context=input,response=output" --max-steps 100
lane "$FRONTIER" data_dolly_nopack --no-packing --max-steps 100

# Efficiency levers, matched to trn2's.
lane "$EXTRAS" eff_norecompute --model meta-llama/Llama-3.1-8B-Instruct \
  --no-gradient-checkpointing --max-steps 100
lane "$EXTRAS" eff_mb2 --model meta-llama/Llama-3.1-8B-Instruct --micro-batch 2 --max-steps 100
lane "$EXTRAS" eff_mb4 --model meta-llama/Llama-3.1-8B-Instruct --micro-batch 4 --max-steps 100
lane "$EXTRAS" eff_seq4096 --model meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --max-steps 100

# 32B on a 32 GiB chip: expected to fail, and WHERE it fails is the datapoint.
lane "$FRONTIER" qwen3_32b_lora --model Qwen/Qwen3-32B --max-steps 50

step "TRN1 SYMMETRY PASS COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh trn1 || echo "push FAILED"
