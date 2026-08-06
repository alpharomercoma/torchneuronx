#!/bin/bash
# trn1.sh — training-lane setup: instance-store scratch volume + swap.
#
# trn1.2xlarge ships one ~475 GB NVMe instance-store volume. Instance store is
# WIPED on every stop/start, so the mount+swap logic lives in a systemd oneshot
# (np-scratch.service) that re-asserts it on every boot, not just first boot.
set -euxo pipefail

cat > /usr/local/sbin/np-scratch.sh <<'EOF'
#!/bin/bash
set -euo pipefail

# Find the ephemeral NVMe device: the one whose model is EC2 instance storage
# (the EBS root reports "Amazon Elastic Block Store").
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

# 64 GiB swapfile: graceful degradation instead of OOM during big compiles.
if [ ! -f /scratch/swapfile ]; then
  fallocate -l 64G /scratch/swapfile
  chmod 600 /scratch/swapfile
  mkswap /scratch/swapfile
fi
swapon /scratch/swapfile 2>/dev/null || true
EOF
chmod +x /usr/local/sbin/np-scratch.sh

cat > /etc/systemd/system/np-scratch.service <<'EOF'
[Unit]
Description=neuron-pipelines: mount instance-store scratch and enable swap
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/np-scratch.sh
RemainAfterExit=yes

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

# PROVISIONING MARKER. common.sh writes /opt/np/.userdata-done, but that
# fires before this box-specific script has installed its packages and
# scratch disk -- so anything waiting on it can start too early. Each box
# now writes its OWN marker as the last action, which is the only reliable
# "this box is ready" signal for a launcher polling from outside.
date -u '+%Y-%m-%dT%H:%M:%SZ' > /opt/np/.userdata-trn1-done
