#!/usr/bin/env bash
# The full accuracy-validity run, one command per box. Sizes fixed here rather
# than passed ad hoc so both boxes provably run the SAME protocol -- the
# receipts' protocol guard would refuse a cross-box comparison otherwise.
#
#   nohup bash extras/run_accuracy_full.sh > /opt/np/accuracy_full.log 2>&1 &
#
# ~2.5-3 h per box, dominated by the CPU reference arm's text tower
# (1000 classes x 80 templates, twice, on 4-8 vCPU). That arm is the price of
# having a control at all.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
export HF_TOKEN="${HF_TOKEN:-$(aws ssm get-parameter --name /neuron-pipelines/hf-token \
  --with-decryption --query Parameter.Value --output text --region us-east-1 2>/dev/null)}"

# 250 per split = 500 utterances across MLPerf's dev-clean + dev-other.
# 10 per class = 10,000 ImageNet images, balanced, ~0.5pp standard error on an
# absolute top-1 and far tighter on the paired delta.
# MAX_TEMPLATES=0 keeps all 80 -- the published prompt ensemble, so the
# absolute number stays comparable to the 63.2% anchor.
export N_PER_SPLIT=250
export PER_CLASS=10
export MAX_TEMPLATES=0
export IMAGE_BATCH=16
export MODELS="openai/clip-vit-base-patch32 google/siglip-base-patch16-224"

# bf16trap runs last: it is the only rung expected to fail, and a failure there
# must not cost the lanes that answer the actual question.
export LANES="asr zeroshot bf16trap"

exec bash "$BENCH_DIR/extras/run_accuracy.sh"
