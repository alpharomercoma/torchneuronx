#!/bin/bash
# common.sh — shared first-boot setup for all neuron-pipelines instances.
# Idempotent: safe to re-run (cloud-init runs this once per instance, but a
# replaced instance re-runs it from scratch).
set -euxo pipefail

# Working tree: models/, results/, cache/ (Neuron compile cache), repo/.
mkdir -p /opt/np/{models,results,cache,repo}
chown -R ubuntu:ubuntu /opt/np

# Login-shell environment for the pipelines.
cat > /etc/profile.d/neuron-pipelines.sh <<'EOF'
export NEURON_COMPILE_CACHE_URL=/opt/np/cache/neuron-compile-cache
export HF_HOME=/opt/np/models/hf
export NP_BUCKET=neuron-pipelines-artifacts-600627330911
EOF

# uv for the ubuntu user (installs to /home/ubuntu/.local/bin), if absent.
if ! sudo -u ubuntu test -x /home/ubuntu/.local/bin/uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sudo -u ubuntu sh
fi

date -u '+%Y-%m-%dT%H:%M:%SZ' > /opt/np/.userdata-done
