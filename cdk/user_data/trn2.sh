#!/bin/bash
# trn2.sh — Trainium2 training-lane setup.
#
# Two deliberate differences from trn1.sh:
#
#   1. NO SWAPFILE. trn1.2xlarge has 32 GiB of host RAM and needed 64 GiB of
#      swap so an 8B adapter merge would degrade instead of OOM. trn2.3xlarge
#      has 128 GiB; swap here would be dead weight on the scratch volume and
#      would mask a real regression if one ever appeared.
#   2. The device/region/cache environment is exported into the login profile,
#      because sft_lora.py, telemetry.py and sync_neuron_cache.sh all branch on
#      it. NP_TELEMETRY_CORES is deliberately ABSENT: the TP probe decides the
#      logical-core count at runtime (LNC=2 gives 4, LNC=1 gives 8) and sets it
#      per-lane. Pinning a guess here would mislabel every telemetry CSV.
#
# Instance store is still WIPED on stop/start, so the mount lives in a systemd
# oneshot that re-asserts on every boot — same as trn1.
set -euxo pipefail

cat >> /etc/profile.d/neuron-pipelines.sh <<'EOF'
# --- Trainium2 box (sa-east-1) ---
export NP_DEVICE=trn2
export NP_REGION=sa-east-1
# NEFFs are compiled per NeuronCore version: v3 artifacts must never land in
# the v2 prefix that trn1 and inf2 share.
export NP_CACHE_PREFIX=neuron-cache-v3
# LNC=2 is the Neuron default on Trainium2 (8 physical NeuronCore-v3 -> 4
# logical cores, 24 GiB HBM each). The TP probe may override it to 1.
export NEURON_LOGICAL_NC_CONFIG=2
EOF

cat > /usr/local/sbin/np-scratch.sh <<'EOF'
#!/bin/bash
set -euo pipefail

# Find the ephemeral NVMe device: the one whose model is EC2 instance storage
# (the EBS root reports "Amazon Elastic Block Store"). trn2.3xlarge ships one
# 470 GB NVMe.
dev=""
for d in /dev/nvme*n1; do
  [ -e "$d" ] || continue
  model="$(lsblk -ndo MODEL "$d" 2>/dev/null || true)"
  if echo "$model" | grep -qi "Instance Storage"; then
    dev="$d"
    break
  fi
done

if [ -z "$dev" ]; then
  echo "np-scratch: no instance-store NVMe present; nothing to do"
  exit 0
fi

# Format only if the device carries no filesystem (fresh or post stop/start).
if ! blkid "$dev" >/dev/null 2>&1; then
  mkfs.ext4 -F "$dev"
fi

mkdir -p /scratch
mountpoint -q /scratch || mount "$dev" /scratch
chown ubuntu:ubuntu /scratch
EOF
chmod +x /usr/local/sbin/np-scratch.sh

cat > /etc/systemd/system/np-scratch.service <<'EOF'
[Unit]
Description=neuron-pipelines: mount instance-store scratch
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/np-scratch.sh
RemainAfterExit=yes
# systemd's DefaultTimeoutStartSec is 90s. np-autorun waits up to 600s for S3
# to become reachable before pulling code, so on a slow-networking boot systemd
# would KILL it at 90s -- the box would come up, sit idle, and burn the entire
# non-refundable window with nothing running. "infinity" disables the timeout.
# NOTE TimeoutStartSec=0 does NOT mean infinite; it means terminate
# immediately. Learned that on the inf2 box an hour before this launch.
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now np-scratch.service

# ---- training packages the DLAMI does not ship -----------------------------
# These were installed BY HAND on the first box and died with its EBS volume
# when the ASG replaced the instance. The replacement then failed every TP rung
# in six seconds on ModuleNotFoundError: no module named 'datasets', halting the
# suite twice. Provisioning belongs in user-data so ANY replacement box brings
# itself up complete.
#
# Versions are pinned to trn1's exactly, which is also what makes the
# cross-generation comparison a hardware comparison rather than a software one.
#
# numpy is installed LAST and deliberately: optimum-neuron 0.4.3 declares
# numpy<=1.26.4, but neuronx-cc requires >=2.0 and trn1 has run every published
# lane on 2.5.1. Installing the pinned set first lets pip resolve, then numpy is
# forced forward. The resulting pip warning is expected and is not an error.
V=/opt/aws_neuronx_venv_pytorch_2_9
if [ -x "$V/bin/pip" ]; then
  "$V/bin/pip" install --quiet \
    optimum-neuron==0.4.3 trl==0.24.0 peft==0.17.0 datasets==5.0.1 \
    transformers==4.57.6 accelerate==1.8.1 || echo "np: pinned install returned nonzero"
  "$V/bin/pip" install --quiet "numpy>=2.0.0,<2.8" || echo "np: numpy bump returned nonzero"
  "$V/bin/python" -c "import datasets,trl,peft;from optimum.neuron import NeuronSFTTrainer" \
    && echo "np: training packages verified" \
    || echo "np: TRAINING PACKAGES BROKEN -- lanes will fail fast at the TP probe"
fi
