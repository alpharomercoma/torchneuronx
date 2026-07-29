#!/usr/bin/env bash
# inf2 box extras. No instance store on inf2.xlarge, so swap lives on the
# EBS root: the vLLM engine stages ~15-16 GB of 8B weights host-side before
# they move to device HBM, and a 16 GiB host gets OOM-killed mid-load
# (measured: kernel killed EngineCore at 14.5 GB RSS, 2026-07-29). The spike
# is transient -- after load, host RSS drops well under RAM -- so EBS swap
# (gp3, 250 MiB/s) costs boot time, not steady-state serving time. Boot wall
# time is a first-class metric either way.
# NOTE: there is no inf2.2xlarge; the family jumps xlarge -> 8xlarge (32
# vCPUs), which needs the pending quota increase. Swap is the quota-8 path.
set -euo pipefail
SWAPFILE=/swapfile
if [ ! -f "$SWAPFILE" ]; then
  fallocate -l 48G "$SWAPFILE"
  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE"
fi
swapon "$SWAPFILE" 2>/dev/null || true
grep -q "$SWAPFILE" /etc/fstab || echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
echo "inf2 user-data done: $(free -h | grep -i swap)"
