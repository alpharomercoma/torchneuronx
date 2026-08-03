#!/usr/bin/env bash
# Trainium2 tensor-parallel probe. Decides the parallelism every later trn2
# lane runs at, and records HOW it decided.
#
# WHY THIS EXISTS
# ---------------
# trn2.3xlarge is one Trainium2: 8 physical NeuronCore-v3 grouped into 4
# logical cores at LNC=2, the Neuron default. World = TP, so the intended
# config is TP=4. But the Neuron runtime docs state that collectives are
# "limited to world size 1, 2, 8, 32" -- 4 is not in that list. That may be
# stale trn1-era guidance, or it may be a hard constraint. Assuming either way
# would put an unverified claim under every number in the Trainium2 section, so
# this probe descends a declared ladder on the cheapest possible model and
# writes the winner to $OUT/tp_probe.json for the drivers to read.
#
#   rung 1  LNC=2, world 4, TP=4   the whole chip, 24 GiB per logical core
#   rung 2  LNC=2, world 2, TP=2   HALF THE CHIP IDLE -- not a 1:1 comparison,
#                                  and the report must label it as such
#   rung 3  LNC=1, world 8, TP=8   8 physical v3 cores. The docs say "both
#                                  physical NeuronCores have access to the
#                                  entire 24GB HBM bank" -- they do NOT say the
#                                  bank is halved, so do not assume 12 GiB per
#                                  rank here; measure it.
#
# Doc status as of 2026-08-04, checked rather than assumed: there is NO AWS
# statement either supporting or forbidding world size 4 on Trainium2. The
# "1, 2, 8, 32" sentence is on a page tagged for Trn2 but is worded as a
# performance-placement heuristic with trn1-shaped examples, and AWS's own
# neuronx-distributed docs use tensor_parallel_size=4 freely. No AWS example
# runs a single Trainium2 chip at all. Hence: measure.
#
# LNC HARD RULE (docs, logical-neuroncore-config): the compiler -lnc flag and
# the runtime NEURON_LOGICAL_NC_CONFIG must MATCH; Neuron does not support
# compiling for one and running the other. Each rung therefore sets the runtime
# variable before any compile happens in that rung.
#
# Every rung's outcome is kept, not just the winner: a ladder that reports only
# its last step is indistinguishable from one that never ran.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$BENCH_DIR/trn2/results/extras"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"   # libneuronpjrt-path (Phase-1 gotcha #2)
PY="$NP_VENV/bin/python"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
bash "$BENCH_DIR/shared/bin/hf_login.sh" >/dev/null 2>&1 \
  || echo "WARN: hf_login failed (gated models will 401)"
TELEM="$BENCH_DIR/shared/telemetry.py"
mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }

if have "$OUT/tp_probe.json"; then
  echo "TP probe already decided:"; cat "$OUT/tp_probe.json"; exit 0
fi

# rung: <lnc> <nproc> <tp>
RUNGS=("2 4 4" "2 2 2" "1 8 8")
ATTEMPTS=""
WINNER=""

for rung in "${RUNGS[@]}"; do
  set -- $rung; LNC=$1; NPROC=$2; TP=$3
  TAG="tp_probe_lnc${LNC}_tp${TP}"
  echo; echo "############ TP probe: LNC=$LNC world=$NPROC TP=$TP ############"; echo

  # TinyLlama-1.1B, 20 steps: the cheapest thing that still exercises the full
  # dataset -> collator -> XLA compile -> collective -> step loop path. An 8B
  # probe would cost a cold compile per rung.
  NEURON_LOGICAL_NC_CONFIG="$LNC" NP_TELEMETRY_CORES="$NPROC" \
  "$PY" "$TELEM" --out "$OUT/$TAG.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tag "$TAG" \
      --device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP" \
      --max-steps 20 --out "$OUT/$TAG.json" \
    > "$OUT/$TAG.log" 2>&1

  if [ -s "$OUT/$TAG.json" ]; then
    ATTEMPTS="$ATTEMPTS{\"lnc\":$LNC,\"nproc\":$NPROC,\"tp\":$TP,\"status\":\"ok\"},"
    WINNER="$LNC $NPROC $TP"
    echo "  rung PASSED (lnc=$LNC tp=$TP)"
    break
  fi

  REASON=$(grep -m1 -oE "(NCC_[A-Z0-9]+|RuntimeError|nrt_[a-z_]+ failed)[^\"]{0,140}" \
           "$OUT/$TAG.log" | head -1 | sed "s/\"/'/g")
  [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$TAG.log" | tr '\n' ' ' | sed "s/\"/'/g")
  ATTEMPTS="$ATTEMPTS{\"lnc\":$LNC,\"nproc\":$NPROC,\"tp\":$TP,\"status\":\"failed\",\"reason\":\"$REASON\"},"
  echo "  rung FAILED (lnc=$LNC tp=$TP): $REASON"
done

CAPTURED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ -n "$WINNER" ]; then
  set -- $WINNER
  # full_chip=false means half the logical cores sat idle: the resulting MFU is
  # still measured against the WHOLE chip's 667 TFLOP/s, so it understates the
  # silicon rather than flattering it. The report must say so out loud.
  FULL=true; [ "$2" -eq 2 ] && FULL=false
  printf '{"status":"ok","lnc":%s,"nproc":%s,"tp":%s,"full_chip":%s,"attempts":[%s],"captured":"%s"}\n' \
    "$1" "$2" "$3" "$FULL" "${ATTEMPTS%,}" "$CAPTURED" > "$OUT/tp_probe.json"
else
  printf '{"status":"failed","note":"every rung of the declared TP ladder failed on trn2; no training lane can run","attempts":[%s],"captured":"%s"}\n' \
    "${ATTEMPTS%,}" "$CAPTURED" > "$OUT/tp_probe.failure.json"
fi

echo; echo "############ TP PROBE RESULT ############"
cat "$OUT/tp_probe.json" 2>/dev/null || cat "$OUT/tp_probe.failure.json"
