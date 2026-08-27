#!/usr/bin/env bash
# Accuracy-validity driver: industry-standard benchmarks run TWICE per model --
# once on the accelerator, once in float32 on the box's own vCPUs -- so the
# difference between them is attributable to the compile and nothing else.
#
#   ASR       LibriSpeech dev-all, WER          (MLPerf Inference v5.1 task)
#   zero-shot ImageNet-1k, top-1/top-5          (CLIP paper protocol)
#
# Runs unchanged on trn1 and inf2: the only per-box difference is which venv
# has optimum-neuron, resolved below.
#
#   bash extras/run_accuracy.sh                    # everything
#   LANES="asr" bash extras/run_accuracy.sh        # one family
#   PER_CLASS=2 N_PER_SPLIT=20 bash extras/run_accuracy.sh   # smoke pilot
#
# ORDER MATTERS AND IS DELIBERATE: the CPU (reference) arm runs FIRST. It is
# the slow one, it needs no accelerator, and if the box dies mid-run we would
# rather hold a reference without a candidate than a candidate with nothing to
# compare it to.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# np_pick_venv / np_box / np_log live here, in ONE copy.
source "$BENCH_DIR/extras/lib/common.sh"
BOX="${BOX:-$(np_box)}"
case "$BOX" in unknown*) echo "FATAL: cannot identify this box ($BOX);"\
  " refusing to write results into an arbitrary directory"; exit 3 ;; esac
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/${BOX}/results}/accuracy"
TELEM="$BENCH_DIR/shared/telemetry.py"
FORCE="${FORCE:-0}"
LANES="${LANES:-asr zeroshot}"
MODELS="${MODELS:-openai/clip-vit-base-patch32 google/siglip-base-patch16-224}"
PER_CLASS="${PER_CLASS:-10}"
N_PER_SPLIT="${N_PER_SPLIT:-250}"
IMAGE_BATCH="${IMAGE_BATCH:-16}"
MAX_TEMPLATES="${MAX_TEMPLATES:-0}"      # 0 = all 80, the published recipe

export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"

# TWO venvs, resolved by CAPABILITY rather than by a hardcoded path, because
# the AMIs differ: trn1 ships /opt/aws_neuronx_venv_pytorch_2_9, inf2 ships
# only the vLLM serving venv. Hardcoding one path is the exact bug that made
# inf2 look like it could not run these models at all.
#
#   zero-shot  needs torch + torch_neuronx + transformers  (raw trace)
#   ASR        additionally needs optimum.neuron           (NeuronWhisper*)
#
# optimum must NEVER be installed into the vLLM venv (it registers a second
# vLLM platform plugin and bricks every server boot), so on inf2 the ASR arm
# uses an overlay venv layered on top of it -- see setup_inf2_overlay.sh.
ZS_VENV="${ZS_VENV:-$(np_pick_venv torch torch_neuronx transformers PIL)}"
ASR_VENV="${ASR_VENV:-$(np_pick_venv torch torch_neuronx transformers optimum.neuron)}"
[ -n "$ZS_VENV" ] || { echo "FATAL: no venv has torch_neuronx + transformers"; exit 3; }
echo "zero-shot venv: ${ZS_VENV:-NONE}"
echo "ASR venv      : ${ASR_VENV:-NONE (optimum.neuron missing -- ASR lane will be skipped)}"
PY="$ZS_VENV/bin/python"
mkdir -p "$RESULTS_DIR"

# ---- preflight: name every missing package ONCE, up front ------------------
# A lane that dies on an import 40 minutes in has burned the window and taught
# nothing. Anything installable is installed here; anything that is not is a
# hard stop before a single accelerator-second is spent.
echo "############ accuracy: preflight ($ZS_VENV) ############"
PATH="$ZS_VENV/bin:$PATH" "$PY" - <<'PRE'
import importlib.util, subprocess, sys
REQUIRED = ["torch", "torch_neuronx", "transformers", "PIL", "numpy"]
# soundfile decodes LibriSpeech FLAC; datasets is NOT needed (we read the
# canonical openslr.org tarballs directly, no HF audio-decoding stack).
# sentencepiece/protobuf: SiglipTokenizer is sentencepiece-backed and raises
# at from_pretrained without them -- a missing tokenizer dep would read as
# "SigLIP unsupported on Neuron", which is exactly the false conclusion the
# vLLM-venv bug already produced once.
INSTALLABLE = {"soundfile": "soundfile", "sentencepiece": "sentencepiece",
               "google.protobuf": "protobuf"}
missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if missing:
    print("FATAL missing (not auto-installable):", missing, file=sys.stderr)
    sys.exit(3)
for mod, pkg in INSTALLABLE.items():
    if importlib.util.find_spec(mod) is None:
        print(f"installing {pkg} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
import torch, torch_neuronx, transformers, numpy  # noqa
print("preflight OK  torch", torch.__version__,
      "torch-neuronx", torch_neuronx.__version__,
      "transformers", transformers.__version__,
      "numpy", numpy.__version__)
PRE
[ $? -eq 0 ] || { echo "FATAL: preflight failed -- fix the venv before spending accelerator time"; exit 3; }

have() { [ "$FORCE" != "1" ] && [ -s "$1" ]; }

run_lane() {   # run_lane <venv> <tag> <script> <extra args...>
  local venv="$1"; shift
  local tag="$1"; shift
  local script="$1"; shift
  if [ -z "$venv" ]; then
    echo "SKIP $tag: no venv on this box satisfies its imports"
    "$PY" -c "
import json,sys,time
json.dump({'lane':'$tag','status':'lane_skipped',
           'reason':'no venv on this box provides the required modules',
           'captured':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},
          open('$RESULTS_DIR/$tag.failure.json','w'),indent=1)"
    return 0
  fi
  if have "$RESULTS_DIR/$tag.json"; then echo "skip $tag (exists)"; return 0; fi
  echo; echo "############ accuracy: $tag ($venv) ############"; echo
  # The venv bin MUST be on PATH, not merely used as an interpreter:
  # torch_neuronx shells out to libneuronpjrt-path and dies with a bare
  # FileNotFoundError if it is not there. Measured on inf2, 2026-08-20.
  PATH="$venv/bin:$PATH" NEURON_RT_NUM_CORES=1 \
    "$venv/bin/python" "$TELEM" --out "$RESULTS_DIR/$tag.telemetry.csv" -- \
    "$venv/bin/python" "$BENCH_DIR/extras/$script" \
    --out "$RESULTS_DIR/$tag.json" "$@" \
    > "$RESULTS_DIR/$tag.log" 2>&1 \
    || echo "  $tag FAILED (see accuracy/$tag.log)"
}

compare_lane() {   # compare_lane <tag> <script> <candidate> <reference>
  local tag="$1" script="$2" cand="$3" ref="$4"
  if [ ! -s "$RESULTS_DIR/$cand.json" ] || [ ! -s "$RESULTS_DIR/$ref.json" ]; then
    echo "skip $tag (need both $cand.json and $ref.json)"; return 0
  fi
  PATH="$ZS_VENV/bin:$PATH" "$PY" "$BENCH_DIR/extras/$script" \
    --out "$RESULTS_DIR/$tag.json" \
    --compare "$RESULTS_DIR/$cand.json" "$RESULTS_DIR/$ref.json" \
    2>&1 | tee "$RESULTS_DIR/$tag.log"
}

for lane in $LANES; do
case "$lane" in
  asr)
    for eng in cpu neuron; do
      run_lane "$ASR_VENV" "asr_wer_$eng" asr_wer_lane.py --engine "$eng" \
        --n-per-split "$N_PER_SPLIT"
    done
    compare_lane asr_wer_delta asr_wer_lane.py asr_wer_neuron asr_wer_cpu
    ;;
  zeroshot)
    for model in $MODELS; do
      short="$(basename "$model" | cut -d- -f1)"
      for eng in cpu neuron; do
        run_lane "$ZS_VENV" "zs_${short}_$eng" zeroshot_imagenet_lane.py \
          --engine "$eng" --model "$model" --per-class "$PER_CLASS" \
          --image-batch "$IMAGE_BATCH" --max-templates "$MAX_TEMPLATES"
      done
      compare_lane "zs_${short}_delta" zeroshot_imagenet_lane.py \
        "zs_${short}_neuron" "zs_${short}_cpu"
    done
    ;;
  bf16trap)
    # The rung that turns a known numerics failure into a scored number:
    # same model, same images, ONLY the compiler cast changes. Measured
    # 2026-07-31, CLIP traced with matmul->bf16 returned NaN logits. If that
    # still holds this lane reports top-1 at chance (0.1%) instead of an
    # anecdote; if it has been fixed upstream, this is how we find out.
    # CPU bf16 is the CONTROL, without which a bf16 NaN on Neuron cannot be
    # told apart from a bf16 NaN in the model itself.
    for a in "cpu --cpu-dtype bfloat16" "neuron --auto-cast matmult"; do
      set -- $a; eng="$1"; shift
      run_lane "$ZS_VENV" "zs_clip_${eng}_bf16" zeroshot_imagenet_lane.py \
        --engine "$eng" "$@" --model openai/clip-vit-base-patch32 \
        --per-class "$PER_CLASS" --image-batch "$IMAGE_BATCH" \
        --max-templates "$MAX_TEMPLATES"
    done
    compare_lane zs_clip_bf16_delta zeroshot_imagenet_lane.py \
      zs_clip_neuron_bf16 zs_clip_cpu_bf16
    ;;
  *) echo "unknown lane: $lane" ;;
esac
done

echo; echo "############ ACCURACY TRACK COMPLETE ($BOX) ############"
"$PY" - "$RESULTS_DIR" <<'SUM'
import json, os, sys
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if not f.endswith(".json") or f.endswith(".images.json") or f.endswith(".utterances.json"):
        continue
    p = json.load(open(os.path.join(d, f)))
    if "wer" in p:
        print(f"{f:32s} WER {p['wer']*100:7.3f}%  trunc {p['truncated']}")
    elif "top1" in p:
        print(f"{f:32s} top1 {p['top1']*100:6.2f}%  top5 {p['top5']*100:6.2f}%  "
              f"nonfinite {p['nonfinite_rows']}")
    elif "mlperf_gate" in p:
        g = p["mlperf_gate"]
        print(f"{f:32s} GATE passed={g.get('passed')} ratio={g.get('ratio')}")
    elif p.get("status") == "lane_failed":
        print(f"{f:32s} FAILED at {p.get('stage')}: {p.get('reason')}")
SUM
bash "$BENCH_DIR/shared/bin/push_results.sh" "$BOX" || echo "  push FAILED (push manually)"
