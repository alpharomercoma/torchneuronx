#!/usr/bin/env bash
# Track F driver: the whole RAG appliance on one inf2, academic/run_academic.sh
# pattern (have()-resumable stages, results as JSON triplets, push at the end).
#
#   setup_pg -> compile_models -> embed_lane -> ingest
#     -> probes --no-llm (retrieval-only, BEFORE the LLM owns the cores)
#     -> boot LLM (shared/serve/launch_vllm.sh, RAG_LLM_KEY, short config)
#     -> probes with LLM -> stop server -> push_results inf2
#
# The --no-llm probe pass runs BEFORE the LLM boots: the TP=2 server owns
# both NeuronCores, and query-time embedding next to it is the DECLARED
# co-residency risk of the fallback plan (README.md "Co-residency"). Running
# the retrieval-only pass first guarantees clean retrieval timings exist even
# if the co-scheduled pass fails; a failed LLM pass is a recorded outcome.
#
# LLM ladder (validity table): RAG_LLM_KEY=qwen3_base to attempt Qwen3-8B
# (known §9 generation crash -- rerunning it here would just re-receipt),
# default llama31_base (proven). TODO-VERIFY: a qwen25_7b rung needs a
# catalogue entry in shared/serve/launch_vllm.sh before it can be exercised.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RAG_DIR="$BENCH_DIR/extras/rag"
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/rag"
MODELS_DIR="${MODELS_DIR:-/opt/np/models}"
NEURON_MODELS_DIR="${NEURON_MODELS_DIR:-$MODELS_DIR/neuron-compiled}"
RAG_LLM_KEY="${RAG_LLM_KEY:-llama31_base}"
FORCE="${FORCE:-0}"

# DLAMI venv resolution, same pattern as shared/serve/launch_vllm.sh.
NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
PY="${NP_VENV:+$NP_VENV/bin/python}"
PY="${PY:-python3}"

# Weights + NEFF cache on the big EBS volume (mirrors launch_vllm.sh so the
# embedding/reranker compiles land in the same synced cache).
export HF_HOME="${HF_HOME:-$MODELS_DIR/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"

mkdir -p "$RESULTS_DIR"
have() { [ "$FORCE" != "1" ] && [ -s "$1" ]; }

# ------------------------------------------------------- overlay venv guard
# optimum-neuron must NEVER be installed into the vLLM venv (duelling
# platform plugins brick every server boot -- measured 2026-07-31, receipts in
# inf2/results/extras/). All optimum-neuron stages (compile / embed / ingest /
# probes) run from a --system-site-packages OVERLAY venv built by
# setup_venv.sh; the LLM boot below stays on the untouched vLLM venv.
echo; echo "############ rag: overlay venv (setup_venv.sh) ############"; echo
RAG_VENV="${RAG_VENV:-/opt/np/venvs/rag-overlay}"
if ! bash "$RAG_DIR/setup_venv.sh" "$RESULTS_DIR/rag_venv.failure.json"; then
  echo "overlay venv FAILED -- receipt at $RESULTS_DIR/rag_venv.failure.json"
  exit 1
fi
# Execution contract (see setup_venv.sh v3): UNDERLAY python + overlay on
# PYTHONPATH. The overlay dir is not a runnable venv on its own.
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
PY="$NP_VENV/bin/python"
export PYTHONPATH="$RAG_VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$NP_VENV/bin:$PATH"  # libneuronpjrt-path (Phase-1 gotcha #2)

# ------------------------------------------------------------ F1: postgres
echo; echo "############ rag: setup_pg ############"; echo
if ! bash "$RAG_DIR/setup_pg.sh" 2>&1 | tee "$RESULTS_DIR/setup_pg.log"; then
  echo "setup_pg FAILED (see $RESULTS_DIR/setup_pg.log)"; exit 1
fi

# ------------------------------------------------------------ F2: compiles
# compile_models is compile-if-absent (OK markers), so it always runs and is
# cheap when cached; receipts land next to the other results either way.
echo; echo "############ rag: compile_models ############"; echo
"$PY" "$RAG_DIR/compile_models.py" \
    --models-dir "$NEURON_MODELS_DIR" --receipts-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/compile_models.log"
COMPILE_RC=${PIPESTATUS[0]}
if [ "$COMPILE_RC" != "0" ]; then
  echo "embedding compile FAILED -- RAG lanes cannot run (receipt recorded)"
  exit 1
fi

# ------------------------------------------------------------ F2: embed lane
echo; echo "############ rag: embed_lane ############"; echo
if have "$RESULTS_DIR/embed_lane.json"; then
  echo "skip embed_lane (exists)"
else
  "$PY" "$RAG_DIR/embed_lane.py" --models-dir "$NEURON_MODELS_DIR" \
      --out "$RESULTS_DIR/embed_lane.json" \
      > "$RESULTS_DIR/embed_lane.log" 2>&1 \
    || echo "  embed_lane FAILED (see embed_lane.log)"
fi

# ------------------------------------------------------------ F4: ingest
echo; echo "############ rag: ingest ############"; echo
if have "$RESULTS_DIR/ingest.json"; then
  echo "skip ingest (exists; upserts are rerun-safe, FORCE=1 to redo)"
else
  "$PY" "$RAG_DIR/ingest.py" --repo-root "$BENCH_DIR" \
      --models-dir "$NEURON_MODELS_DIR" --out "$RESULTS_DIR/ingest.json" \
      > "$RESULTS_DIR/ingest.log" 2>&1 \
    || { echo "  ingest FAILED (see ingest.log)"; exit 1; }
fi

# ---------------------------------------- F5a: retrieval-only probes (pre-boot)
echo; echo "############ rag: probes --no-llm (pre-boot) ############"; echo
if have "$RESULTS_DIR/probes_nollm.json"; then
  echo "skip probes_nollm (exists)"
else
  "$PY" "$RAG_DIR/probes.py" --no-llm --models-dir "$NEURON_MODELS_DIR" \
      --out "$RESULTS_DIR/probes_nollm.json" \
      2>&1 | tee "$RESULTS_DIR/probes_nollm.log" \
    || echo "  probes --no-llm had errors (see probes_nollm.log)"
fi

# ------------------------------------------------------------ F3: LLM boot
#
# CORE PARTITIONING (RAG_SPLIT_CORES=1, the co-residency fix)
# -----------------------------------------------------------
# A NeuronCore belongs to exactly one process. The original run booted an 8B
# model at TP=2, which owns BOTH cores of the inf2.xlarge, so the encoders had
# no device left and every end-to-end probe died with
#   RuntimeError: The PyTorch Neuron Runtime could not be initialized
# That was recorded as a co-residency failure, and read as "you cannot run the
# three models together on one Inferentia2". The real constraint is narrower:
# you cannot run an 8B LM together with anything, because 8B in bf16 is ~16 GB
# against a 16 GB per-core budget and must span both cores.
#
# With an LM that fits on one core, the appliance is possible: pin the server
# to nc0 at TP=1 and the encoders to nc1. The encoders are plain single-core
# torch_neuronx.trace artifacts (~1.2 GB each), so both fit on nc1 together.
echo; echo "############ rag: LLM boot ($RAG_LLM_KEY, short) ############"; echo
BOOT_JSON="$RESULTS_DIR/${RAG_LLM_KEY}_boot.json"
SERVER_UP=0
LLM_CORES="" ; ENC_CORES=""
if [ "${RAG_SPLIT_CORES:-0}" = "1" ]; then
  LLM_CORES="${RAG_LLM_CORES:-0}"
  ENC_CORES="${RAG_ENC_CORES:-1}"
  echo "--- core split: LLM on nc$LLM_CORES (tp=1), encoders on nc$ENC_CORES ---"
fi
# env -u PYTHONPATH: the overlay (optimum-neuron + its vllm platform plugin
# entry point) must NEVER be visible to a vLLM server process.
if env -u PYTHONPATH \
     ${LLM_CORES:+NP_VISIBLE_CORES="$LLM_CORES" TP_DEGREE_OVERRIDE=1} \
     bash "$BENCH_DIR/shared/serve/launch_vllm.sh" "$RAG_LLM_KEY" short "$BOOT_JSON"; then
  SERVER_UP=1
else
  echo "  LLM boot FAILED -- load_failure.json recorded next to $BOOT_JSON;"
  echo "  ladder: retry with RAG_LLM_KEY=<next rung> (see README.md)"
fi

# ------------------------------------------------- F5b: end-to-end probes
if [ "$SERVER_UP" = "1" ]; then
  echo; echo "############ rag: probes (with LLM) ############"; echo
  MODEL_NAME="$(python3 -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['model_id'])" \
    "$BOOT_JSON" 2>/dev/null || true)"
  if [ -z "$MODEL_NAME" ]; then
    echo "  could not read model_id from $BOOT_JSON; skipping LLM probes"
  elif have "$RESULTS_DIR/probes_llm.json"; then
    echo "skip probes_llm (exists)"
  else
    # The encoders take the core the server was NOT pinned to. Without this
    # the jit.load below hits a device already owned by the server process.
    env ${ENC_CORES:+NEURON_RT_VISIBLE_CORES="$ENC_CORES"} \
      "$PY" "$RAG_DIR/probes.py" --llm-model "$MODEL_NAME" \
        --models-dir "$NEURON_MODELS_DIR" \
        --out "$RESULTS_DIR/probes_llm.json" \
        2>&1 | tee "$RESULTS_DIR/probes_llm.log" \
      || echo "  probes with LLM had errors (co-residency? see probes_llm.log)"
  fi

  echo; echo "############ rag: stop LLM ############"; echo
  env -u PYTHONPATH bash "$BENCH_DIR/shared/serve/launch_vllm.sh" stop "$BOOT_JSON"
fi

# ------------------------------------------------------------------- push
echo; echo "############ RAG TRACK COMPLETE ############"; echo
bash "$BENCH_DIR/shared/bin/push_results.sh" inf2 || echo "  push FAILED (push manually)"
