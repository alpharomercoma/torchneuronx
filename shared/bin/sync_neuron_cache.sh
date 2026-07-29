#!/usr/bin/env bash
# Two-way sync of the Neuron compile cache (NEFFs) with S3.
#   sync_neuron_cache.sh push   # after a lane that compiled something new
#   sync_neuron_cache.sh pull   # after re-provisioning a box
# The EBS copy is primary; S3 is disaster recovery so a replaced instance
# never pays the tens-of-minutes 8B compile twice.
set -euo pipefail
CACHE="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
case "${1:?usage: sync_neuron_cache.sh push|pull}" in
  push) aws s3 sync "$CACHE/" "s3://neuron-pipelines-artifacts-600627330911/neuron-cache/" ;;
  pull) mkdir -p "$CACHE"; aws s3 sync "s3://neuron-pipelines-artifacts-600627330911/neuron-cache/" "$CACHE/" ;;
  *) echo "usage: sync_neuron_cache.sh push|pull" >&2; exit 2 ;;
esac
du -sh "$CACHE" 2>/dev/null || true
