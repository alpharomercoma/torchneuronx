#!/usr/bin/env bash
# Extras track driver (Phase-2 Track A): mlx-parity inference lanes
# (whisper/clip/siglip) on ONE NeuronCore each, telemetry-wrapped, resumable
# like every other lane in this repo, then the Mistral-7B vLLM serve attempt
# on both cores. Runs on inf2; results land in inf2/results/extras/ and push
# at the end.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras"
# THE VENV BUG, fixed 2026-08-20. This line used to default to the vLLM venv
# (/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16), which exists to SERVE
# and ships no optimum-neuron. Result: all three parity lanes below died with
# "ModuleNotFoundError: No module named 'optimum'" and inf2 was written up as
# lacking multimodal support it in fact has. The lanes were never run.
#
# The banned-install comment further down is still correct and still applies:
# optimum-neuron must never be installed INTO the vLLM venv (it registers a
# second vLLM platform plugin and bricks every server boot). The fix is to
# point the parity lanes at the inference venv that already has optimum, and
# leave the serve lane on the vLLM venv -- so the two live side by side
# instead of one shadowing the other.
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
SERVE_VENV="${SERVE_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
PY="$NP_VENV/bin/python"
[ -x "$PY" ] || { echo "FATAL: no python at $PY"; exit 3; }
# Fail LOUDLY on a missing import rather than burning a lane on it. Three
# separate runs in this study were lost to a package missing from a venv
# (optimum here, accelerate in the gpt-oss serve lane, datasets in the
# perplexity lane); each looked like an unsupported model until the receipt
# was read.
"$PY" - <<'PREFLIGHT' || { echo "FATAL: $NP_VENV is missing lane dependencies"; exit 3; }
import importlib, sys
missing = [m for m in ("torch", "torch_neuronx", "transformers", "optimum.neuron")
           if not importlib.util.find_spec(m.split(".")[0])]
if missing:
    print("missing modules:", missing, file=sys.stderr); sys.exit(1)
import optimum.neuron  # noqa: F401  -- the import that actually failed before
PREFLIGHT
TELEM="$BENCH_DIR/shared/telemetry.py"
FORCE="${FORCE:-0}"
# Weights + neuronx-cc caches on the big EBS volume, same as the serve lanes.
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"

mkdir -p "$RESULTS_DIR"
have() { [ "$FORCE" != "1" ] && [ -s "$1" ]; }

# optimum-neuron must NEVER be installed into this venv. Measured 2026-07-31:
# (a) the venv's transformers is a patched fork under an upstream version
# string and lacks symbols optimum-neuron imports (PeftAdapterMixin,
# GenerationMixin) -- receipts in inf2/results/extras/*.failure.json; and
# (b) optimum-neuron registers a SECOND vLLM platform plugin, which bricks
# every server boot: "Only one platform plugin can be activated, but got:
# ['optimum_neuron', 'neuron']" (receipt: extras/serve/mistral7b_short).
# Parity lanes run on trn1's training venv instead (run_extras_trn1.sh).
echo; echo "############ extras: optimum-neuron install -- BANNED (see comment) ############"; echo

# whisper/clip/siglip are single-core traced models, so the core pin is
# per-lane (NOT exported: the mistral serve lane below needs both cores).
for lane in whisper clip siglip; do
  echo; echo "############ extras: $lane ############"; echo
  if have "$RESULTS_DIR/$lane.json"; then
    echo "skip $lane (exists)"; continue
  fi
  NEURON_RT_NUM_CORES=1 "$PY" "$TELEM" --out "$RESULTS_DIR/$lane.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/extras/${lane}_lane.py" --out "$RESULTS_DIR/$lane.json" \
    > "$RESULTS_DIR/$lane.log" 2>&1 || echo "  $lane FAILED (see extras/$lane.log)"
done

# Mistral-7B serve attempt (Track A4): bench_serve handles boot /
# load_failure / grid + per-point telemetry itself, and a recorded boot
# failure exits 0 there, so this lane cannot wedge the driver. grid.json is
# its own "sweep complete" marker (same resume key as run_all.sh).
echo; echo "############ extras: mistral7b serve ############"; echo
if have "$RESULTS_DIR/serve/mistral7b_short/grid.json"; then
  echo "skip mistral7b_short (grid.json exists)"
else
  RESULTS_DIR="$RESULTS_DIR" NP_VENV="$SERVE_VENV" bash "$BENCH_DIR/shared/serve/bench_serve.sh" mistral7b short \
    || echo "  mistral7b serve FAILED (see serve/mistral7b_short/)"
fi

echo; echo "############ EXTRAS TRACK COMPLETE ############"
bash "$BENCH_DIR/shared/bin/push_results.sh" inf2 || echo "  push FAILED (push manually)"
