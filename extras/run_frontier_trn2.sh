#!/usr/bin/env bash
# Frontier lanes: what a Trainium2 can do that a Trainium1 cannot.
#
# The rest of Phase 3 answers "how much FASTER is trn2". This file answers the
# more interesting question: "what is now POSSIBLE". A 3.5x speed ratio is a
# number; "this model does not fit on the previous generation at any batch
# size" is a capability boundary, and capability boundaries survive
# re-benchmarking.
#
# WHAT CHANGED IN THE TWO YEARS BETWEEN THE CHIPS (Neuron arch pages)
# -------------------------------------------------------------------
#            Trainium1 (NC-v2)      Trainium2 (NC-v3)
#   BF16     190 TFLOP/s            667          3.51x
#   FP8      no speedup             1299         2x BF16 ON THE SAME SILICON
#   sparse   none                   2563         new capability entirely
#   HBM      32 GiB @ 820 GB/s      96 @ 2.9 TB/s
#   SBUF     48 MiB                 224 MiB      4.7x
#
# Ridge point (peak FLOP/s over bandwidth -- where compute-bound begins):
#   trn1 BF16  190/0.82 = 232 FLOP/byte
#   trn2 BF16  667/2.9  = 230 FLOP/byte   <- essentially UNCHANGED
#   trn2 FP8   1299/2.9 = 448 FLOP/byte
# AWS scaled compute and bandwidth together, so a memory-bound workload stays
# memory-bound; you get 3.5x without changing which side of the roofline you
# are on. FP8 doubles the ridge point, so it only pays above ~230 FLOP/byte.
# That is the thesis these lanes test.
#
# PROBE-FIRST IS THE WHOLE DESIGN
# -------------------------------
# Every capability here is UNVERIFIED on NeuronCore-v3: optimum-neuron 0.4.3's
# training path was tuned against v2. So each group runs a minutes-scale
# TinyLlama probe before committing to anything expensive. A failed probe costs
# a receipt; a failed 32B run costs an hour of a window that cannot be refunded.
# "This did not work on v3" is a legitimate published finding either way.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/trn2/results/frontier"
EXTRAS="$BENCH_DIR/trn2/results/extras"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"
FRONTIER_DEADLINE_UTC="${FRONTIER_DEADLINE_UTC:-10:00}"

export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"

mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ FRONTIER: $* ############"; echo; }

[ -s "$EXTRAS/tp_probe.json" ] || { echo "FATAL: tp_probe.json missing"; exit 3; }
read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["lnc"], d["nproc"], d["tp"])' "$EXTRAS/tp_probe.json")"
export NEURON_LOGICAL_NC_CONFIG="$LNC"
export NP_TELEMETRY_CORES="$NPROC"
TP_ARGS=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP")

seconds_left() {
  local now target
  now=$(date -u +%s)
  target=$(date -u -d "today ${FRONTIER_DEADLINE_UTC}" +%s 2>/dev/null) || \
    target=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u +%Y-%m-%d) ${FRONTIER_DEADLINE_UTC}" +%s)
  [ "$target" -le "$now" ] && target=$((target + 86400))
  echo $((target - now))
}

# lane <tag> <need_min> <args...>   -- receipt on failure, push after success
lane() {
  local tag="$1" need="$2"; shift 2
  { have "$OUT/$tag.json" || have "$OUT/$tag.failure.json"; } && { echo "skip $tag"; return 0; }
  local left; left=$(seconds_left)
  if [ "$left" -lt $((need * 60)) ]; then
    echo "  deadline guard: $((left/60)) min left, $tag needs ~${need} -- skipping"
    return 1
  fi
  step "$tag"
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" "${TP_ARGS[@]}" --out "$OUT/$tag.json" "$@" \
    > "$OUT/$tag.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}|OutOfMemory|KeyError[^\"]{0,60}|NotImplementedError[^\"]{0,60}" "$OUT/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
      echo "  $tag FAILED (receipt)"; return 1; }
  bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || echo "  push FAILED"
  return 0
}

echo "frontier lanes: LNC=$LNC world=$NPROC TP=$TP, hard stop ${FRONTIER_DEADLINE_UTC}Z"

# ===========================================================================
# A1. FP8 -- the headline capability of the generation gap
# ===========================================================================
# NeuronCore-v3's TensorEngine does FP8 (E4M3/E5M2) matmul at DOUBLE the BF16
# rate. Trainium1 lists cFP8 at the same 190 TFLOP/s as BF16 -- i.e. no FP8
# speedup at all. This is the one lane where trn2 can beat its own 3.51x.
#
# THIS IS NOT AN FP8 TRAINING BENCHMARK, AND MUST NOT BE REPORTED AS ONE.
# AWS's NxD Training documentation states FP8 TRAINING is unsupported on
# Neuron, Trn2 included; the FP8 material AWS publishes for Trn2 is INFERENCE
# (quantized Llama 3.3 70B via NxD Inference). Separately, --auto-cast only
# rewrites selected FP32 operators, and these lanes already build a bf16=True
# graph -- so there may be no FP32 left to cast and this can be a complete
# no-op.
#
# A job that converges here therefore proves NOTHING about FP8. It would mean
# only "the compiler accepted the flag and produced an unverified mixed graph".
# Recorded as a compiler-cast COMPATIBILITY probe on TinyLlama only: no 8B
# lane, no throughput claim, and the FP8 peak (1299 TFLOP/s) and FP8 ridge
# (448 FLOP/byte) must NOT be used as denominators anywhere in the report.
step "A1: FP8 probe (TinyLlama, 20 steps) -- --auto-cast-type=fp8_e4m3"
# Subshell so the compiler flag cannot leak into any later lane -- an FP8 NEFF
# silently reused by a BF16 lane would corrupt the comparison AND the cache.
( export NEURON_CC_FLAGS="--auto-cast=all --auto-cast-type=fp8_e4m3"
  lane fp8_probe 25 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-steps 20 )
# Deliberately NO 8B FP8 lane: see above. Whatever this probe returns is a
# compatibility receipt, not a performance result.
echo "  fp8 cast probe recorded as COMPATIBILITY ONLY -- no FP8 performance claim"

# ===========================================================================
# A2. A model that does not fit on the previous generation
# ===========================================================================
# Qwen3-32B is ~64 GiB of BF16 weights. Trainium1 has 32 GiB per CHIP, so this
# does not fit at any batch size, sequence length or TP degree -- it is not a
# tuning question. Trainium2's 96 GiB holds it with room for activations.
# This is the strongest single claim in the whole study: not "faster", but
# "possible at all".
lane qwen3_32b_lora 180 --model Qwen/Qwen3-32B --max-steps 50 || true

# ===========================================================================
# A3. Mixture-of-Experts on ONE chip
# ===========================================================================
# NxD Training added Mixtral 8x7B support and MoE is where frontier models have
# gone.
#
# CORRECTION to an earlier framing here: "activates ~3B of 30B params" does NOT
# make this a capacity problem rather than a compute one. Active parameters
# determine COMPUTE per token; the full ~30B of BF16 weights (~60 GiB) must
# still be RESIDENT unless the runtime does expert paging or offload, which is
# not demonstrated here. So this is a weight-RESIDENCY and compatibility probe:
# can one chip hold and step a 30B MoE at all. Its MFU is NOT comparable to the
# dense lanes -- different FLOPs-per-resident-byte entirely -- and must not be
# put in the same table.
lane moe_qwen3_30b_a3b 150 --model Qwen/Qwen3-30B-A3B --max-steps 50 || true

# ===========================================================================
# B1. Full fine-tune vs LoRA on the same chip
# ===========================================================================
# LoRA freezes >99% of weights; a full FT updates all of them and carries
# fp32 master weights plus two Adam moments -- ~12 bytes/param of optimizer
# state on top of the weights. On trn1's 16 GiB/core that ceiling arrives
# almost immediately. Running both on ONE chip gives the honest LoRA-vs-full
# efficiency comparison the report currently cannot make, and the FLOPs
# accounting switches from 6-trainable/4-frozen to dense 6N automatically
# because frozen becomes zero.
lane fullft_tinyllama 40 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --no-lora --max-steps 50 || true
lane fullft_qwen3_1_7b 60 --model Qwen/Qwen3-1.7B --no-lora --max-steps 50 || true

# ===========================================================================
# B2. Dataset variation -- packing efficiency is the hidden MFU variable
# ===========================================================================
# Every published MFU here is on dolly-15k. Sequence-length DISTRIBUTION drives
# how well examples pack into a fixed 2048-token window, and packing efficiency
# moves MFU independently of the hardware. Comparing two chips on one dataset
# cannot separate those effects. Same prompt template throughout (the field
# remap exists so the formatter never changes), so the only variable is the
# data.
lane data_alpaca 60 --dataset tatsu-lab/alpaca \
  --dataset-fields "context=input,response=output" --max-steps 100 || true
lane data_dolly_nopack 60 --no-packing --max-steps 100 || true

# ===========================================================================
# C1. Two models resident on one chip
# ===========================================================================
# A production question, not a benchmark one: can one Trainium2 host two
# workloads without them fighting? 96 GiB and four logical cores make it
# plausible in a way trn1's 32 GiB/2 cores never was. Run two 2-core jobs
# concurrently and compare each against the same job run alone.
step "C1: multi-model residency (two 2-core jobs, concurrent)"
if ! have "$OUT/residency_pair_a.json" && [ "$NPROC" -ge 4 ]; then
  if [ "$(seconds_left)" -gt $((70 * 60)) ]; then
    NP_MASTER_PORT=29511 NEURON_RT_VISIBLE_CORES=0,1 "$PY" "$TELEM" --out "$OUT/residency_pair_a.telemetry.csv" -- \
      "$PY" "$SFT" --tag residency_pair_a --device-profile trn2 \
        --nproc-per-node 2 --tensor-parallel-size 2 \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-steps 50 \
        --out "$OUT/residency_pair_a.json" > "$OUT/residency_pair_a.log" 2>&1 &
    PID_A=$!
    NP_MASTER_PORT=29512 NEURON_RT_VISIBLE_CORES=2,3 "$PY" "$TELEM" --out "$OUT/residency_pair_b.telemetry.csv" -- \
      "$PY" "$SFT" --tag residency_pair_b --device-profile trn2 \
        --nproc-per-node 2 --tensor-parallel-size 2 \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-steps 50 \
        --out "$OUT/residency_pair_b.json" > "$OUT/residency_pair_b.log" 2>&1 &
    PID_B=$!
    wait $PID_A; wait $PID_B
    # SAME model on both halves (was TinyLlama vs Qwen-1.7B): with two
    # different models, a slowdown cannot be attributed to interference rather
    # than to the models differing. And a solo baseline is needed for BOTH
    # core pairs, not just one, or the b-side number has nothing to compare to.
    NP_MASTER_PORT=29513 NEURON_RT_VISIBLE_CORES=0,1 "$PY" "$TELEM" --out "$OUT/residency_solo_a.telemetry.csv" -- \
      "$PY" "$SFT" --tag residency_solo_a --device-profile trn2 \
        --nproc-per-node 2 --tensor-parallel-size 2 \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-steps 50 \
        --out "$OUT/residency_solo_a.json" > "$OUT/residency_solo_a.log" 2>&1 \
      || echo "  residency solo a FAILED"
    NP_MASTER_PORT=29514 NEURON_RT_VISIBLE_CORES=2,3 "$PY" "$TELEM" --out "$OUT/residency_solo_b.telemetry.csv" -- \
      "$PY" "$SFT" --tag residency_solo_b --device-profile trn2 \
        --nproc-per-node 2 --tensor-parallel-size 2 \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-steps 50 \
        --out "$OUT/residency_solo_b.json" > "$OUT/residency_solo_b.log" 2>&1 \
      || echo "  residency solo b FAILED"
    bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || true
  else
    echo "  deadline guard: skipping residency"
  fi
else
  echo "  skip residency (recorded, or fewer than 4 logical cores)"
fi

echo
echo "############ FRONTIER COMPLETE ($(($(seconds_left)/60)) min to ${FRONTIER_DEADLINE_UTC}Z) ############"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh trn2 || echo "push FAILED"
