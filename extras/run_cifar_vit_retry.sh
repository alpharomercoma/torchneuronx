#!/usr/bin/env bash
# CIFAR-ViT ON TRAINIUM2 — capacity watcher + one-shot run + immediate teardown.
#
# WHY
# ---
# cifar_vit is the ONE lane that succeeds on Trainium1 (80.64% test accuracy)
# and fails on Trainium2. It failed twice with:
#   [F137] neuronx-cc was forcibly killed — insufficient system memory
# on a 124 GiB host, for a ~1-2M parameter ViT that compiles fine on trn1's
# 30 GiB host. That size mismatch says compiler pathology, not model size.
#
# BOTH previous attempts used DEFAULT compiler settings. This run adds
# `--optlevel=1`, which trades optimisation for substantially lower compile-time
# memory. It is the obvious lever and it has never been tried.
#
# COST DISCIPLINE
# ---------------
# ON-DEMAND HOURLY, never a capacity block: no 24-hour minimum. The instance is
# TERMINATED the moment the lane finishes, succeeds or fails. A hard attempt
# cap stops the watcher from grazing indefinitely.
set -uo pipefail
REGION=sa-east-1
AMI=ami-0b1b0d3aaa2171e1f
PROFILE=arn:aws:iam::600627330911:instance-profile/NeuronPipelinesTrainium2-Trn2LaunchTemplateProfileD51B68AC-kdBIYW7krhFw
SG=sg-049487e20a17cc2d4
SUBNETS="subnet-0489739583976c545 subnet-092833ea4d9c0210c subnet-088dd390f7aab1c53"
UD="${UD:?set UD=/path/to/userdata.b64}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-36}"      # 36 x 10 min = 6 hours, then give up
SLEEP="${SLEEP:-600}"
log(){ echo "[$(date -u +%H:%M:%SZ)] cifar_retry: $*"; }

IID=""
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  for SN in $SUBNETS; do
    IID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type trn2.3xlarge \
      --subnet-id "$SN" --security-group-ids $SG --iam-instance-profile Arn=$PROFILE \
      --metadata-options HttpTokens=required \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' \
      --user-data "file://$UD" \
      --capacity-reservation-specification CapacityReservationPreference=none \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=np-cifarvit-ondemand}]' \
      --count 1 --query 'Instances[0].InstanceId' --output text 2>/dev/null)
    case "$IID" in i-*) log "LAUNCHED $IID in $SN (attempt $i)"; break 2;; esac
    IID=""
  done
  log "no capacity in any AZ (attempt $i/$MAX_ATTEMPTS)"
  sleep "$SLEEP"
done
[ -n "$IID" ] || { log "gave up after $MAX_ATTEMPTS attempts — no on-demand capacity"; exit 1; }

trap 'log "terminating $IID"; aws ec2 terminate-instances --region $REGION --instance-ids "$IID" >/dev/null 2>&1' EXIT

log "waiting for SSM to come online (user-data provisioning takes ~10-15 min)"
for i in $(seq 1 60); do
  ok=$(aws ssm describe-instance-information --region $REGION \
        --filters "Key=InstanceIds,Values=$IID" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)
  [ "$ok" = "Online" ] && { log "SSM online"; break; }
  sleep 30
done

log "waiting for provisioning marker, then running cifar_vit with --optlevel=1"
CMD=$(aws ssm send-command --region $REGION --instance-ids "$IID" --document-name AWS-RunShellScript \
  --timeout-seconds 7200 --parameters 'commands=[
    "for i in $(seq 1 60); do [ -f /opt/np/.provisioned ] && break; sleep 30; done",
    "cd /opt/np/repo && bash shared/bin/pull_code.sh >/dev/null 2>&1",
    "export NEURON_CC_FLAGS=\"--optlevel=1\"",
    "export BOX=trn2 NP_DEVICE=trn2 NP_REGION=sa-east-1 NP_CACHE_PREFIX=neuron-cache-v3",
    "rm -f /opt/np/repo/trn2/results/academic/cifar_vit.failure.json",
    "cd /opt/np/repo && BOX=trn2 timeout 5400 bash academic/run_academic.sh 2>&1 | tail -40",
    "cat /opt/np/repo/trn2/results/academic/cifar_vit.json 2>/dev/null | head -c 600 || cat /opt/np/repo/trn2/results/academic/cifar_vit.failure.json 2>/dev/null",
    "bash /opt/np/repo/shared/bin/push_results.sh trn2 2>&1 | tail -1"
  ]' --query 'Command.CommandId' --output text 2>/dev/null)
log "command $CMD dispatched; polling"
for i in $(seq 1 240); do
  st=$(aws ssm get-command-invocation --region $REGION --command-id "$CMD" --instance-id "$IID" --query 'Status' --output text 2>/dev/null)
  case "$st" in Success|Failed|TimedOut|Cancelled) log "finished: $st"; break;; esac
  sleep 30
done
aws ssm get-command-invocation --region $REGION --command-id "$CMD" --instance-id "$IID" --query 'StandardOutputContent' --output text 2>/dev/null | tail -30
log "DONE — instance terminates on exit"
