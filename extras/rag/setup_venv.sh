#!/usr/bin/env bash
# Build the RAG overlay for inf2 -- the ONLY sanctioned way to get
# optimum-neuron next to this box's Neuron stack.
#
# History of this file is a chain of measured failures (receipts in
# inf2/results/ and git history):
#   v1: optimum-neuron pip-installed INTO the vLLM venv -> second vLLM
#       platform plugin -> every server boot bricked. NEVER again.
#   v2: `venv --system-site-packages` created FROM the vLLM venv python ->
#       venv-from-venv resolves "system site-packages" to the BASE system
#       python, not the vLLM venv, so torch_neuronx vanished and pip pulled
#       its own CPU torch 2.13 into the overlay.
#   v3 (this): the overlay venv exists ONLY as a pip-managed directory of
#       packages. Execution uses the UNDERLAY python with
#       PYTHONPATH=<overlay site-packages>: underlay supplies
#       torch/torch_neuronx/neuronx-cc; overlay shadows transformers (the
#       DLAMI ships a patched fork under an upstream version string) and
#       adds optimum-neuron. The overlay's own torch is uninstalled so the
#       underlay's Neuron-matched torch always wins. vLLM is never executed
#       with this PYTHONPATH -- callers MUST strip it when booting servers
#       (run_rag.sh uses `env -u PYTHONPATH`).
set -uo pipefail

NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
RAG_VENV="${RAG_VENV:-/opt/np/venvs/rag-overlay}"
OVERLAY_SITE="$RAG_VENV/lib/python3.12/site-packages"
RECEIPT="${1:-/tmp/rag_venv_receipt.json}"
export PATH="$NP_VENV/bin:$PATH"   # libneuronpjrt-path (Phase-1 gotcha #2)
CHECK="import optimum.neuron; import torch_neuronx"  # NOT PeftAdapterMixin: genuine transformers 4.57 moved it; optimum.neuron importing IS the gate

fail() {
  mkdir -p "$(dirname "$RECEIPT")"
  printf '{\n "status": "rag_venv_failed", "stage": "%s",\n "reason": %s,\n "captured": "%s"\n}\n' \
    "$1" "$(printf '%s' "$2" | head -c 400 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RECEIPT"
  echo "setup_venv FAILED at $1: $2" >&2
  exit 1
}

# Health check mirrors exactly how run_rag.sh executes (underlay python +
# overlay PYTHONPATH) -- a check weaker than the runtime contract is how v2
# slipped through.
if [ -d "$OVERLAY_SITE" ] \
   && PYTHONPATH="$OVERLAY_SITE" "$NP_VENV/bin/python" -c "$CHECK" >/dev/null 2>&1; then
  echo "rag overlay healthy: PYTHONPATH=$OVERLAY_SITE + $NP_VENV/bin/python"
  exit 0
fi

[ -x "$NP_VENV/bin/python" ] || fail probe "underlay venv missing: $NP_VENV"

mkdir -p "$(dirname "$RAG_VENV")"
"$NP_VENV/bin/python" -m venv "$RAG_VENV" || fail venv "python -m venv failed"
PIP="$RAG_VENV/bin/pip"

# No [neuronx] extra (backtracks pip into py3.12-incompatible numpy source
# builds -- REPORT.md corrections).
"$PIP" install --no-cache-dir "optimum-neuron==0.4.3" \
  || fail pip "optimum-neuron==0.4.3 install failed"

# The overlay resolved deps against a bare venv, so it holds its own torch;
# strip it (and companions) so the underlay's Neuron-matched torch wins
# under PYTHONPATH. Errors ignored: absent is the goal state.
"$PIP" uninstall -y torch torchgen torchvision torchaudio numpy >/dev/null 2>&1 || true

PYTHONPATH="$OVERLAY_SITE" "$NP_VENV/bin/python" -c "$CHECK" \
  || fail verify "combined underlay+overlay import check failed"

# Prove the underlay itself was never written to.
if "$NP_VENV/bin/python" -c "import optimum.neuron" >/dev/null 2>&1; then
  fail pollution "optimum.neuron importable from the bare vLLM venv -- DO NOT SERVE until fixed"
fi

echo "rag overlay ready: PYTHONPATH=$OVERLAY_SITE + $NP_VENV/bin/python (underlay clean)"
