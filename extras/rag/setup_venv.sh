#!/usr/bin/env bash
# Build the RAG overlay venv on inf2 -- the ONLY sanctioned way to get
# optimum-neuron onto this box.
#
# Why an overlay exists at all (measured 2026-07-31, receipts in
# inf2/results/extras/*.failure.json and REPORT addendum):
#   (a) installing optimum-neuron into the vLLM venv registers a SECOND vLLM
#       platform plugin -> "Only one platform plugin can be activated, but
#       got: ['optimum_neuron', 'neuron']" -> EVERY server boot bricked;
#   (b) the vLLM DLAMI's transformers is a patched fork under an upstream
#       version string, missing symbols optimum-neuron imports
#       (PeftAdapterMixin, GenerationMixin).
#
# The overlay: `python -m venv --system-site-packages` over the vLLM venv.
#   * torch / torch-neuronx / neuronx-cc come from the underlay (no reinstall);
#   * optimum-neuron + a GENUINE transformers land in the overlay, shadowing
#     the fork;
#   * vLLM is never executed from the overlay, so the duelling-plugin
#     condition (two plugins visible to a *running* vllm) can never trigger;
#   * the vLLM venv itself is never written to -- serving stays healthy.
set -uo pipefail

NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
RAG_VENV="${RAG_VENV:-/opt/np/venvs/rag-overlay}"
# import torch_neuronx shells out to `libneuronpjrt-path` (venv-bin script);
# without both bins on PATH the verify step false-fails under SSM shells.
export PATH="$RAG_VENV/bin:$NP_VENV/bin:$PATH"
RECEIPT="${1:-/tmp/rag_venv_receipt.json}"

fail() {
  mkdir -p "$(dirname "$RECEIPT")"
  printf '{\n "status": "rag_venv_failed", "stage": "%s",\n "reason": %s,\n "captured": "%s"\n}\n' \
    "$1" "$(printf '%s' "$2" | head -c 400 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RECEIPT"
  echo "setup_venv FAILED at $1: $2" >&2
  exit 1
}

if [ -x "$RAG_VENV/bin/python" ] \
   && "$RAG_VENV/bin/python" -c "import optimum.neuron; from transformers.integrations import PeftAdapterMixin" >/dev/null 2>&1; then
  echo "rag overlay venv already healthy: $RAG_VENV"
  exit 0
fi

[ -x "$NP_VENV/bin/python" ] || fail probe "underlay venv missing: $NP_VENV"

mkdir -p "$(dirname "$RAG_VENV")"
"$NP_VENV/bin/python" -m venv --system-site-packages "$RAG_VENV" \
  || fail venv "python -m venv failed"
PIP="$RAG_VENV/bin/pip"

# No [neuronx] extra (it backtracks pip into py3.12-incompatible numpy source
# builds -- REPORT.md corrections); neuron deps come from the underlay.
"$PIP" install --no-cache-dir "optimum-neuron==0.4.3" \
  || fail pip "optimum-neuron==0.4.3 install failed"

# If pip judged the underlay's patched-fork transformers "already satisfied",
# the fork leaks through. Detect by the missing symbol and shadow it with a
# genuine wheel of the SAME version pip resolved.
if ! "$RAG_VENV/bin/python" -c "from transformers.integrations import PeftAdapterMixin" >/dev/null 2>&1; then
  TV=$("$RAG_VENV/bin/python" -c "import transformers; print(transformers.__version__)") \
    || fail shadow "cannot read transformers version"
  echo "patched-fork transformers leaked through (v$TV); shadowing with genuine wheel"
  "$PIP" install --no-cache-dir --ignore-installed --no-deps "transformers==$TV" \
    || fail shadow "genuine transformers==$TV install failed"
fi

"$RAG_VENV/bin/python" -c "import optimum.neuron; from transformers.integrations import PeftAdapterMixin; import torch_neuronx" \
  || fail verify "post-install import check failed"

# Belt and braces: prove the underlay was not polluted.
if "$NP_VENV/bin/python" -c "import optimum.neuron" >/dev/null 2>&1; then
  fail pollution "optimum.neuron is importable from the vLLM venv -- underlay polluted, DO NOT SERVE until fixed"
fi

echo "rag overlay venv ready: $RAG_VENV (underlay clean)"
