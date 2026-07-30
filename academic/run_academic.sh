#!/usr/bin/env bash
# Academic track driver: 6 lanes (mnist/cifar x mlp/cnn/vit) on ONE NeuronCore,
# telemetry-wrapped, resumable like every other lane in this repo. Runs on
# trn1; results land in trn1/results/academic/ and push at the end.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/trn1/results}/academic"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
FORCE="${FORCE:-0}"
export NEURON_RT_NUM_CORES=1   # small models; single-core is the honest unit

mkdir -p "$RESULTS_DIR"
have() { [ "$FORCE" != "1" ] && [ -s "$1" ]; }

# mnist first: minutes-scale, doubles as the plumbing smoke for the track.
for spec in mnist:mlp mnist:cnn mnist:vit cifar:mlp cifar:cnn cifar:vit; do
  ds="${spec%%:*}"; arch="${spec##*:}"
  tag="${ds}_${arch}"
  echo; echo "############ academic: $tag ############"; echo
  if have "$RESULTS_DIR/$tag.json"; then
    echo "skip $tag (exists)"; continue
  fi
  "$PY" "$TELEM" --out "$RESULTS_DIR/$tag.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/academic/train_academic.py" \
      --dataset "$ds" --arch "$arch" --out "$RESULTS_DIR/$tag.json" \
    > "$RESULTS_DIR/$tag.log" 2>&1 || echo "  $tag FAILED (see academic/$tag.log)"
done

echo; echo "############ ACADEMIC TRACK COMPLETE ############"
bash "$BENCH_DIR/shared/bin/push_results.sh" trn1 || echo "  push FAILED (push manually)"
