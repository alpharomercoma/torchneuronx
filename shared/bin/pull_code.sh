#!/usr/bin/env bash
# Sync the harness from S3 onto this box. Run over SSM after any local edit
# followed by 'make push-code' on the laptop.
set -euo pipefail
aws s3 sync "s3://neuron-pipelines-artifacts-600627330911/code/" /opt/np/repo/ --delete --exclude ".git/*"
chmod +x /opt/np/repo/shared/run_all.sh /opt/np/repo/*/scripts/run_all.sh \
         /opt/np/repo/shared/bin/*.sh /opt/np/repo/shared/*.sh 2>/dev/null || true
echo "code synced -> /opt/np/repo"
