# Overlay venv on inf2 so the ASR lane can import optimum.neuron WITHOUT
# installing it into the vLLM serving venv (which would register a second vLLM
# platform plugin and brick every server boot -- measured 2026-07-31).
#
# ATTEMPT 1 FAILED and the reason is worth keeping: `venv --system-site-packages`
# created FROM a venv's python inherits the SYSTEM site-packages, not the
# parent venv's. The overlay saw no torch/transformers at all, so --no-deps
# installs pulled a fresh transformers and numpy 2.5.2, shadowing the exact
# stack the traces must run against. A .pth is the mechanism that actually
# chains one venv onto another.
set -uo pipefail
BASE=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16
OV=/opt/np/venvs/optimum-overlay
BASE_SP="$(ls -d $BASE/lib/python3.*/site-packages | head -1)"

rm -rf "$OV"
python3 -m venv "$OV" || exit 3
OV_SP="$(ls -d $OV/lib/python3.*/site-packages | head -1)"
# `import site; site.addsitedir(...)`, NOT a bare path line. A bare path is
# appended to sys.path verbatim and the base venv's OWN .pth files never run --
# including distutils-precedence.pth, so torch_neuronx died on
# "No module named 'distutils'" (Python 3.12 removed it; setuptools reinstates
# it through exactly that .pth). addsitedir processes them.
# Appended, so anything installed INTO the overlay still wins; the base only
# fills in what the overlay lacks (torch, torch-neuronx, transformers, numpy).
echo "import site; site.addsitedir('$BASE_SP')" > "$OV_SP/_base_venv.pth"

# torch_neuronx shells out to `libneuronpjrt-path` at import time, and that
# executable lives in the BASE venv's bin. Symlinking the base console scripts
# into the overlay's bin means one PATH entry ($OV/bin) is enough -- which is
# exactly what the driver puts on PATH, so this cannot drift out of sync with
# how the lanes are actually launched.
for f in "$BASE/bin/"*; do
  b="$(basename "$f")"
  case "$b" in python*|pip*|activate*) continue ;; esac
  [ -e "$OV/bin/$b" ] || ln -s "$f" "$OV/bin/$b"
done

echo "=== inherited stack (must match the base venv exactly) ==="
PATH="$OV/bin:$PATH" "$OV/bin/python" -c "
import torch, transformers, numpy, torch_neuronx
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| numpy', numpy.__version__, '| torch-neuronx', torch_neuronx.__version__)
" || exit 3

"$OV/bin/pip" install -q --no-deps optimum-neuron==0.4.3 optimum 2>&1 | tail -2
"$OV/bin/pip" install -q soundfile sentencepiece protobuf 2>&1 | tail -2
for i in $(seq 1 12); do
  M=$(PATH="$OV/bin:$PATH" "$OV/bin/python" - <<'PY' 2>&1
try:
    from optimum.neuron import NeuronWhisperForConditionalGeneration  # noqa
    print("OK")
except ModuleNotFoundError as e:
    print("MISSING:" + (e.name or "?"))
except Exception as e:
    print("OTHER:" + type(e).__name__ + ":" + str(e)[:160])
PY
)
  echo "attempt $i -> $M"
  case "$M" in
    OK) break ;;
    # WITH deps here, not --no-deps: these are small pure-python packages
    # (httpcore, h11, ...) whose own deps are also missing, and a --no-deps
    # install of each would loop one module at a time. pip is constrained
    # below from touching the load-bearing four.
    MISSING:*) "$OV/bin/pip" install -q "${M#MISSING:}" 2>&1 | tail -1 ;;
    *) break ;;
  esac
done

echo "=== the stack MUST be unchanged after all that ==="
PATH="$OV/bin:$PATH" "$OV/bin/python" - <<'PY'
import sys, torch, transformers, numpy, torch_neuronx
want = {"transformers": "4.57.6", "torch": "2.9.1", "numpy": "2.2.6"}
got = {"transformers": transformers.__version__, "torch": torch.__version__,
       "numpy": numpy.__version__}
print("got", got, "| torch-neuronx", torch_neuronx.__version__)
bad = {k: (v, want[k]) for k, v in got.items() if not v.startswith(want[k])}
if bad:
    print("FATAL: overlay shadowed the base stack:", bad, file=sys.stderr)
    sys.exit(3)
from optimum.neuron import NeuronWhisperForConditionalGeneration  # noqa
import soundfile, sentencepiece  # noqa
print("overlay OK -- optimum.neuron + soundfile + sentencepiece importable")
PY
echo "=== base venv must still be free of optimum ==="
"$BASE/bin/python" -c "import optimum" 2>&1 | tail -1
