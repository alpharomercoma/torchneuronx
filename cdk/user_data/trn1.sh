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
