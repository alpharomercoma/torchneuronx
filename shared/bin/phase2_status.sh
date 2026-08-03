#!/usr/bin/env bash
# Reattach from anywhere: one command, no session state needed. Polls all three
# boxes over SSM and prints driver tails + recorded results.
#   bash shared/bin/phase2_status.sh
# Requires: aws CLI logged in.
#
# trn1/inf2 IDs stay pinned (those boxes are long-lived). The trn2 box lives in
# a different region and was created later, so it is discovered by its Name tag
# instead -- one less thing to hand-edit after a redeploy.
set -uo pipefail
export AWS_PAGER=""
REGION=us-west-2
TRN2_REGION=sa-east-1
TRN1=i-0cb9e758143a745d5
INF2=i-0936ae7615727251e
TRN2=$(aws ec2 describe-instances --region "$TRN2_REGION" \
  --filters Name=tag:Name,Values=neuron-pipelines-trn2 \
            Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)

probe() { # $1 label, $2 instance, $3 log, $4 results-glob-dir, $5 region
  local id="$2"
  local region="${5:-$REGION}"
  local state
  [ -n "$id" ] && [ "$id" != "None" ] || { echo "=================== $1 (not deployed) ==================="; return 0; }
  state=$(aws ec2 describe-instances --region "$region" --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  echo "=================== $1 ($id @ $region: ${state:-unknown}) ==================="
  [ "$state" = "running" ] || return 0
  local cid
  cid=$(aws ssm send-command --region "$region" --instance-ids "$id" \
    --document-name AWS-RunShellScript \
    --parameters "commands=[\"echo '-- driver processes:'; pgrep -af 'run_phase2|run_phase3|run_extras|run_rag|tp_probe' || echo none; echo '-- log tail:'; tail -6 $3 2>/dev/null; echo '-- results:'; find $4 -name '*.json' 2>/dev/null | sed 's|.*/results/||' | sort | tr '\\n' ' '; echo\"]" \
    --query Command.CommandId --output text) || { echo "  (ssm send failed)"; return 0; }
  sleep 10
  for i in 1 2 3; do
    OUTP=$(aws ssm get-command-invocation --region "$region" --command-id "$cid" \
      --instance-id "$id" --query '[Status,StandardOutputContent]' --output text 2>/dev/null)
    case "$OUTP" in Success*|Failed*) break;; *) sleep 5;; esac
  done
  echo "$OUTP"
}

probe trn1 "$TRN1" /opt/np/phase2_trn1.log "/opt/np/repo/trn1/results/extras"
probe inf2 "$INF2" /opt/np/phase2_inf2.log "/opt/np/repo/inf2/results/extras /opt/np/repo/inf2/results/rag"
probe trn2 "$TRN2" /opt/np/phase3_trn2.log "/opt/np/repo/trn2/results" "$TRN2_REGION"
echo "Done markers: 'PHASE2 TRN1 ALL COMPLETE' / 'PHASE2 INF2 ALL COMPLETE' / 'PHASE3 TRN2 ALL COMPLETE'"
