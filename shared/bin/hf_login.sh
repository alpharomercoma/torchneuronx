#!/usr/bin/env bash
# Fetch the HF token from SSM Parameter Store (SecureString) and log in.
# The token never touches code, user-data, or the CFN template.
set -euo pipefail
TOKEN=$(aws ssm get-parameter --name /neuron-pipelines/hf-token \
        --with-decryption --query Parameter.Value --output text --region us-west-2)
mkdir -p ~/.cache/huggingface
printf '%s' "$TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
# The boxes export HF_HOME=/opt/np/models/hf, and huggingface_hub resolves the
# ambient token at $HF_HOME/token -- the default path above is only a fallback
# for shells that never sourced the profile. Both locations get the token.
HF_HOME="${HF_HOME:-/opt/np/models/hf}"
mkdir -p "$HF_HOME"
printf '%s' "$TOKEN" > "$HF_HOME/token"
chmod 600 "$HF_HOME/token"
echo "HF token installed for $(whoami) at ~/.cache/huggingface/token and $HF_HOME/token"
