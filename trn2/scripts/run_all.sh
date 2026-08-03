#!/usr/bin/env bash
# Path wrapper only -- every lane lives in shared/run_all.sh, which runs the
# SAME training branch as trn1 so the two chips are compared on identical work.
# NP_DEVICE comes from /etc/profile.d/neuron-pipelines.sh (stack user-data); the
# fallback here keeps an SSM root shell, which sources no profile, honest.
cd "$(dirname "$0")/../.." || exit 1
BOX=trn2 BENCH_DIR="$PWD" RESULTS_DIR="$PWD/trn2/results" MODELS_DIR="${MODELS_DIR:-/opt/np/models}" \
  NP_DEVICE="${NP_DEVICE:-trn2}" NP_REGION="${NP_REGION:-sa-east-1}" \
  NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}" \
  exec bash shared/run_all.sh "$@"
