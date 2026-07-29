#!/usr/bin/env bash
# Path wrapper only -- every lane lives in shared/run_all.sh (byte-identical on both boxes).
cd "$(dirname "$0")/../.." || exit 1
BOX=trn1 BENCH_DIR="$PWD" RESULTS_DIR="$PWD/trn1/results" MODELS_DIR="${MODELS_DIR:-/opt/np/models}" \
  exec bash shared/run_all.sh "$@"
