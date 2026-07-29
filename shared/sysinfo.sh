#!/usr/bin/env bash
# Capture the full hardware/software provenance of whichever box this runs on.
# Runs on the HOST (not in a container) and needs no Python packages.
#
# Everything recorded here is quoted in README.md. Nothing gets benchmarked
# before this has run, so any surprise -- a partitioned GPU, a clock cap, a
# different H200 SKU than expected -- is caught before it silently skews a
# result rather than after.
set -u

OUT="${1:-specs.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "=== captured ==="
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo "hostname: $(hostname)"

  echo
  echo "=== os ==="
  . /etc/os-release && echo "$PRETTY_NAME"
  echo "kernel: $(uname -r)"
  echo "arch: $(uname -m)"

  echo
  echo "=== cpu ==="
  lscpu | grep -E "^Architecture|^Model name|^CPU\(s\)|^Thread\(s\) per core|^Core\(s\) per socket|^Socket\(s\)|^NUMA node\(s\)|^CPU max MHz|^CPU min MHz|^L3 cache|^Flags" | sed 's/  */ /g'

  echo
  echo "=== memory ==="
  free -g | head -2
  echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"

  echo
  echo "=== storage ==="
  lsblk -d -o NAME,SIZE,MODEL 2>/dev/null
  df -h / /mnt/scratch 2>/dev/null

  echo
  echo "=== accelerator ==="
  if command -v neuron-ls >/dev/null 2>&1; then
    echo "-- vendor: AWS Neuron"
    neuron-ls 2>/dev/null
    echo "-- devices (json):"
    neuron-ls --json-output 2>/dev/null | head -40
  else
    echo "neuron-ls not found -- not a Neuron box (or driver missing: check aws-neuronx-dkms)"
  fi
  echo "-- /dev/neuron* nodes:"
  ls -l /dev/neuron* 2>/dev/null || echo "NONE (aws-neuronx-dkms not loaded -- see PROVISIONING gotchas)"

  echo
  echo "=== accelerator stack ==="
  echo "-- driver/runtime packages:"
  dpkg -l 2>/dev/null | grep -E "aws-neuronx|neuron" | awk '{print $2, $3}' || echo n/a
  echo "-- neuronx-cc:"
  neuronx-cc --version 2>/dev/null || echo "not on PATH (lives in the DLAMI venvs)"
  echo "-- DLAMI venvs present:"
  ls -d /opt/aws_neuronx_venv* 2>/dev/null || echo none
  for v in /opt/aws_neuronx_venv*/bin/pip; do
    [ -x "$v" ] || continue
    echo "-- $(dirname "$(dirname "$v")") key packages:"
    "$v" list 2>/dev/null | grep -iE "^(torch|torch-neuronx|torch-xla|neuronx-cc|neuronx-distributed|optimum-neuron|vllm|transformers|libneuronxla) " || true
  done

  echo
  echo "=== neuron compile cache ==="
  echo "NEURON_COMPILE_CACHE_URL=${NEURON_COMPILE_CACHE_URL:-unset}"
  du -sh "${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}" 2>/dev/null || echo "cache dir absent"
} > "$OUT" 2>&1

echo "wrote $OUT"
