#!/usr/bin/env bash
# Capture the Trainium2 box's state before AWS reclaims it.
#
# Runs FROM THE LAPTOP (or anywhere with credentials), not on the box:
#
#   nohup caffeinate -i bash ops/capacity/trn2_preserve_state.sh \
#     >> ~/trn2_preserve.log 2>&1 &
#   disown
#
# WHY
# ---
# A Capacity Block instance is TERMINATED, not stopped, starting 11:00Z on the
# final day, and the root volume is DeleteOnTermination=true. So unlike every
# other box in this project the disk does not survive -- and with it goes:
#
#   /opt/np/cache/neuron-compile-cache   the warm v3 NEFF cache
#   /opt/np/models/hf                    tens of GB of downloaded weights
#                                        (Llama 3.1 8B, Qwen3 8B, and if the
#                                        frontier lanes ran, Qwen3-32B and
#                                        Qwen3-30B-A3B)
#
# The NEFF cache is already pushed to S3 by the deadline pusher, which is the
# portable form and the one that matters most. The MODEL WEIGHTS are not, and
# re-downloading them costs a large slice of the next window before any
# measurement can start. An AMI captures both, so a future block boots warm:
# no re-download, no re-compile.
#
# You cannot keep the hardware -- that is AWS's to reclaim, and the only way to
# hold it longer is a paid Capacity Block extension. You CAN keep everything
# that took time to build on it, and that is nearly free.
#
# TIMING. Must complete before the ASG scales to 0 at 10:50Z, which is itself
# before EC2 starts terminating at 11:00Z. create-image returns as soon as the
# snapshot is REGISTERED (it continues in the background against the volume),
# but to be safe this runs at 10:35Z -- after the deadline pusher's 10:30Z
# final push, before the scale-in.
set -uo pipefail
export AWS_PAGER=""

REGION="${REGION:-sa-east-1}"
# ABSOLUTE timestamp, not a time-of-day. A "next 10:35Z" rule fires on the
# WRONG DAY: the box does not exist until 11:31Z on the first day, so the
# script would wake at 10:35Z today, find nothing, and exit as a no-op.
CAPTURE_AT="${CAPTURE_AT:-2026-08-06T10:35:00Z}"
NAME_TAG="${NAME_TAG:-neuron-pipelines-trn2}"
STATUS_FILE="${STATUS_FILE:-$HOME/trn2_preserve.status}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preserve: $*"; }
status() { echo "$*" > "$STATUS_FILE"; }

seconds_to() {
  local now target
  now=$(date -u +%s)
  target=$(date -u -d "$CAPTURE_AT" +%s 2>/dev/null) || \
    target=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$CAPTURE_AT" +%s)
  echo $((target - now))
}

WAIT=$(seconds_to)
if [ "$WAIT" -le 0 ]; then
  log "FATAL: CAPTURE_AT ($CAPTURE_AT) is already in the past"
  status "FAILED: capture time in the past"
  exit 6
fi
log "will capture AMI at $CAPTURE_AT (in $((WAIT/60)) min)"
status "WAITING to capture at $CAPTURE_AT"
# Long sleeps are fine here; this is a one-shot alarm, not a poll loop.
sleep "$WAIT"

IID=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=$NAME_TAG" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)

if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  log "no running $NAME_TAG instance -- nothing to capture"
  status "NOTHING TO CAPTURE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

STAMP=$(date -u +%Y%m%d-%H%M)
# --no-reboot is REQUIRED: a reboot here would kill any lane still running and
# could cost the very results this is meant to protect. The tradeoff is a
# crash-consistent image rather than a clean one, which is fine -- what we care
# about (NEFF cache, model weights) is static by this point in the window.
AMI=$(aws ec2 create-image --region "$REGION" --instance-id "$IID" \
  --name "neuron-pipelines-trn2-warm-$STAMP" \
  --description "Trainium2 box state: warm v3 NEFF cache + downloaded HF weights. Captured before Capacity Block reclaim." \
  --no-reboot \
  --tag-specifications "ResourceType=image,Tags=[{Key=Name,Value=neuron-pipelines-trn2-warm},{Key=Project,Value=neuron-pipelines}]" \
  --query 'ImageId' --output text 2>&1)

if [ -z "$AMI" ] || [ "${AMI#ami-}" = "$AMI" ]; then
  log "create-image FAILED: $AMI"
  status "FAILED at $(date -u +%Y-%m-%dT%H:%M:%SZ): $AMI"
  exit 5
fi

log "AMI $AMI registered from $IID (snapshot continues in background)"
log "next block: deploy with -c trn2AmiId=$AMI to skip re-download and re-compile"
status "CAPTURED $AMI at $(date -u +%Y-%m-%dT%H:%M:%SZ) from $IID"
