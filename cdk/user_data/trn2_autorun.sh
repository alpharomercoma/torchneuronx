#!/bin/bash
# trn2_autorun.sh — start the Phase-3 suite on first boot, with no operator.
#
# Included in user-data ONLY in scheduled (Capacity Block / ASG) mode. A manual
# `cdk deploy` still gives you a quiet box you can drive by hand.
#
# WHY THIS EXISTS
# ---------------
# Moving the LAUNCH into AWS (an ASG scheduled action) is only half the job. If
# the kickoff still comes from a laptop over SSM, the laptop is back on the
# critical path and the box boots into an expensive idle. A Capacity Block is
# paid for in full up front, so idle minutes are pure waste.
#
# Runs as a systemd oneshot rather than inline in user-data so it can wait for
# the network without blocking cloud-init, and re-asserts on reboot guarded by
# a marker file (the suite itself is resumable via have(), but starting two
# copies would race for the same result files).
set -euxo pipefail

cat > /usr/local/sbin/np-autorun.sh <<'EOF'
#!/bin/bash
exec >> /opt/np/autorun.log 2>&1
set -x
echo "np-autorun starting $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# One kickoff per instance. run_phase3_trn2.sh is resumable, but two concurrent
# copies would race on the same triplets.
if [ -f /opt/np/.autorun-done ]; then
  echo "already kicked off; nothing to do"
  exit 0
fi

export NP_DEVICE=trn2
export NP_REGION=sa-east-1
export NP_CACHE_PREFIX=neuron-cache-v3
export NEURON_LOGICAL_NC_CONFIG=2
export HF_HOME=/opt/np/models/hf
export NEURON_COMPILE_CACHE_URL=/opt/np/cache/neuron-compile-cache
export NP_BUCKET=neuron-pipelines-artifacts-600627330911
# torch-neuronx shells out to libneuronpjrt-path on init; without the venv bin
# on PATH that is a FileNotFoundError. Phase 2 lost seven lanes to this.
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH

# The bucket is in us-west-2 while this box is in sa-east-1; the CLI resolves
# the bucket's region itself (that is what the s3:GetBucketLocation grant in the
# instance role is for). Wait for it rather than assuming networking is up:
# cloud-init can reach here before the instance has working egress.
for _ in $(seq 1 60); do
  if aws s3 ls "s3://$NP_BUCKET/code/" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

aws s3 cp "s3://$NP_BUCKET/code/shared/bin/pull_code.sh" - | bash
cd /opt/np/repo

# Marker BEFORE launching: if the suite dies we want a record that the kickoff
# happened, not a reboot loop that starts it again.
date -u '+%Y-%m-%dT%H:%M:%SZ' > /opt/np/.autorun-done

setsid nohup bash extras/run_phase3_trn2.sh >> /opt/np/phase3_trn2.log 2>&1 &
setsid nohup bash extras/trn2_deadline_push.sh >> /opt/np/deadline_push.log 2>&1 &
echo "np-autorun kicked off suite + deadline pusher"
EOF
chmod +x /usr/local/sbin/np-autorun.sh

cat > /etc/systemd/system/np-autorun.service <<'SVC'
[Unit]
Description=neuron-pipelines: pull code and start the Phase-3 Trainium2 suite
After=network-online.target np-scratch.service cloud-final.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/np-autorun.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable --now np-autorun.service

date -u '+%Y-%m-%dT%H:%M:%SZ' > /opt/np/.userdata-trn2-autorun-done
