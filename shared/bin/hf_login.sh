#!/usr/bin/env bash
# Fetch the HF token from SSM Parameter Store (SecureString) and log in.
# The token never touches code, user-data, or the CFN template.
set -euo pipefail
TOKEN=$(aws ssm get-parameter --name /neuron-pipelines/hf-token \
        --with-decryption --query Parameter.Value --output text --region us-west-2)
mkdir -p ~/.cache/huggingface
printf '%s' "$TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
echo "HF token installed for $(whoami) (stored_tokens not used; plain token file)"
