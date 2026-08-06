#!/usr/bin/env bash
# Academic track driver: 6 lanes (mnist/cifar x mlp/cnn/vit) on ONE NeuronCore,
# telemetry-wrapped, resumable like every other lane in this repo.
#
# BOX-parameterised so trn2 can run the identical track: `BOX=trn2` sends
# results to trn2/results/academic/ and pushes under that box. It used to
# hardcode trn1 in both places, which would have silently written Trainium2
# numbers into the Trainium1 result set.
set -uo pipefail

BOX="${BOX:-trn1}"
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/$BOX/results}/academic"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
FORCE="${FORCE:-0}"

# torch-neuronx shells out to libneuronpjrt-path on init; without the venv bin
# on PATH that is a FileNotFoundError. Phase 2 lost seven lanes to this.
export PATH="$NP_VENV/bin:$PATH"

# One core: these models are far too small to say anything about parallelism,
# so a per-core number is the honest unit. NOTE the unit differs by generation
# -- on trn1 this is one NeuronCore-v2 (16 GiB); on trn2 at the LNC=2 default it
# is one LOGICAL v3 core (24 GiB, two physical cores). Same "one core" label,
# different hardware underneath, and the report must say so rather than compare
# the two numbers as if the denominators matched.
export NEURON_RT_NUM_CORES=1

mkdir -p "$RESULTS_DIR"
have() { [ "$FORCE" != "1" ] && [ -s "$1" ]; }

# mnist first: minutes-scale, doubles as the plumbing smoke for the track.
for spec in mnist:mlp mnist:cnn mnist:vit cifar:mlp cifar:cnn cifar:vit; do
  ds="${spec%%:*}"; arch="${spec##*:}"
  tag="${ds}_${arch}"
  echo; echo "############ academic: $tag ############"; echo
  if have "$RESULTS_DIR/$tag.json" || have "$RESULTS_DIR/$tag.failure.json"; then
    echo "skip $tag (recorded)"; continue
  fi
  "$PY" "$TELEM" --out "$RESULTS_DIR/$tag.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/academic/train_academic.py" \
      --dataset "$ds" --arch "$arch" --out "$RESULTS_DIR/$tag.json" \
    > "$RESULTS_DIR/$tag.log" 2>&1 || {
      # A FAILURE MUST LEAVE A RECEIPT. This driver used to only echo, so when
      # cifar_vit's compiler was OOM-killed on trn2 the study had no record of
      # it at all -- the lane simply appeared absent, which reads as "not run"
      # rather than "tried and failed". Every other driver in this repo writes
      # a receipt; this one now does too.
      REASON=$(grep -m1 -oE "\[F[0-9]+\][^\"]{0,150}|NCC_[A-Z0-9]+[^\"]{0,140}|ValueError: [^\"]{0,140}" "$RESULTS_DIR/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$RESULTS_DIR/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","box":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$BOX" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RESULTS_DIR/$tag.failure.json"
      echo "  $tag FAILED (receipt written)"; }
done

echo; echo "############ ACADEMIC TRACK COMPLETE ############"
bash "$BENCH_DIR/shared/bin/push_results.sh" "$BOX" || echo "  push FAILED (push manually)"
