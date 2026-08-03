#!/usr/bin/env bash
# Push this box's raw result triplets to S3. $1 = box name (trn1|trn2|inf2).
set -euo pipefail
BOX="${1:?usage: push_results.sh trn1|trn2|inf2}"
aws s3 sync "/opt/np/repo/$BOX/results/" "s3://neuron-pipelines-artifacts-600627330911/results/$BOX/"
echo "results pushed -> s3://neuron-pipelines-artifacts-600627330911/results/$BOX/"
