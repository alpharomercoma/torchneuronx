#!/bin/bash
# inf2.sh — inference-lane setup placeholder.
#
# inf2 instances have no NVMe instance store (EBS only), so there is no
# scratch/swap dance here. Lane-specific setup (model server units, vLLM
# warm-up, etc.) lands in this file when needed.
set -euxo pipefail

date -u '+%Y-%m-%dT%H:%M:%SZ' > /opt/np/.userdata-inf2-done
