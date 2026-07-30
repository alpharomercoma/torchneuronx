#!/usr/bin/env bash
# Reattach to Phase-2 from anywhere: one command, no session state needed.
# Polls both boxes over SSM and prints driver tails + recorded results.
#   bash shared/bin/phase2_status.sh
# Requires: aws CLI logged in (us-west-2). Instance IDs pinned below --
# update INF2 if the box is ever destroyed/redeployed again.
set -uo pipefail
export AWS_PAGER=""
REGION=us-west-2
TRN1=i-0cb9e758143a745d5
INF2=i-0936ae7615727251e

probe() { # $1 label, $2 instance, $3 log, $4 results-glob-dir
  local id="$2"
  local state
  state=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  echo "=================== $1 ($id: ${state:-unknown}) ==================="
  [ "$state" = "running" ] || return 0
  local cid
  cid=$(aws ssm send-command --region "$REGION" --instance-ids "$id" \
    --document-name AWS-RunShellScript \
    --parameters "commands=[\"echo '-- driver processes:'; pgrep -af 'run_phase2|run_extras|run_rag' || echo none; echo '-- log tail:'; tail -6 $3 2>/dev/null; echo '-- results:'; find $4 -name '*.json' 2>/dev/null | sed 's|.*/results/||' | sort | tr '\\n' ' '; echo\"]" \
    --query Command.CommandId --output text) || { echo "  (ssm send failed)"; return 0; }
  sleep 10
  for i in 1 2 3; do
    OUTP=$(aws ssm get-command-invocation --region "$REGION" --command-id "$cid" \
      --instance-id "$id" --query '[Status,StandardOutputContent]' --output text 2>/dev/null)
    case "$OUTP" in Success*|Failed*) break;; *) sleep 5;; esac
  done
  echo "$OUTP"
}

probe trn1 "$TRN1" /opt/np/phase2_trn1.log "/opt/np/repo/trn1/results/extras"
probe inf2 "$INF2" /opt/np/phase2_inf2.log "/opt/np/repo/inf2/results/extras /opt/np/repo/inf2/results/rag"
echo "Done markers to look for: 'PHASE2 TRN1 ALL COMPLETE' / 'PHASE2 INF2 ALL COMPLETE'"
