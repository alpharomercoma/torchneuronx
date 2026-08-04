#!/usr/bin/env bash
# Launch the Trainium2 box into a purchased EC2 Capacity Block the moment the
# window opens, then start the Phase-3 suite on it -- without a human present.
#
#   nohup caffeinate -i bash extras/trn2_block_launch.sh \
#     >> ~/trn2_block_launch.log 2>&1 &
#   disown
#
# WHY THIS EXISTS
# ---------------
# A Capacity Block bills for its ENTIRE window from its start time, occupied or
# not, and cancellations are not permitted. Unlike the on-demand watcher this
# replaces, there is no "keep polling and hope" mode: the money is already
# spent, so every minute between the window opening and the first lane starting
# is pure waste. Hence: no alarm clock, no manual deploy, no manual kickoff.
#
# THE TWO THINGS A BLOCK LAUNCH NEEDS THAT A NORMAL ONE DOES NOT
# --------------------------------------------------------------
# The block's InstanceMatchCriteria is "targeted" -- an ordinary launch does not
# fall into it, it fails on capacity exactly like every unreserved attempt. Both
# of these must be present, and both live on the launch template because
# AWS::EC2::Instance cannot express market options:
#
#   InstanceMarketOptions.MarketType          = capacity-block
#   CapacityReservationTarget.CapacityReservationId = cr-...
#
# The CDK stack wires them from -c trn2CapacityReservationId and REFUSES to
# synth without an explicit AZ, because a block lives in exactly one AZ and
# defaulting to sa-east-1a would miss a sa-east-1b block and burn the window.
#
# THE DEADLINE IS 11:00Z, NOT 11:30Z
# ----------------------------------
# AWS terminates -- not stops -- instances in a Capacity Block starting 30
# minutes before the end time. The warm v3 NEFF cache on EBS dies with it, so
# anything not pushed to S3 by then is gone. This script arms a separate
# deadline pusher on the box for exactly that reason.
set -uo pipefail
export AWS_PAGER=""

REGION="${REGION:-sa-east-1}"
CR_ID="${CR_ID:-cr-06bed0d84f5e2e6e2}"
AZ="${AZ:-sa-east-1b}"
SUBNET="${SUBNET:-subnet-092833ea4d9c0210c}"
REPO="${REPO:-/Users/alpha/neuron-pipelines}"
STATUS_FILE="${STATUS_FILE:-$HOME/trn2_block_launch.status}"
DEPLOY_LOG="${DEPLOY_LOG:-$HOME/trn2_block_deploy.log}"
LOCK="${LOCK:-$HOME/.trn2_block_launch.lock}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
status() { echo "$*" > "$STATUS_FILE"; }

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "another launcher is already running (pid $(cat "$LOCK")); exiting"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log "waiting for capacity block $CR_ID in $REGION ($AZ) to go active (pid $$)"
status "WAITING for $CR_ID since $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- 1. Wait for the block to open -----------------------------------------
# States: payment-pending -> scheduled -> active -> expired. Only "active" can
# accept a launch. Classify errors rather than discarding them: an expired
# session must be loud, because silently polling blind is how the previous
# watcher wasted nine hours.
while :; do
  ERR=$(mktemp)
  STATE=$(aws ec2 describe-capacity-reservations --region "$REGION" \
    --capacity-reservation-ids "$CR_ID" \
    --query 'CapacityReservations[0].State' --output text 2>"$ERR")
  ERRTXT=$(tr -d '\n' < "$ERR"); rm -f "$ERR"

  case "$ERRTXT" in
    "") : ;;
    *ExpiredToken*|*"security token"*|*InvalidClientTokenId*|*AuthFailure*)
      log "CREDENTIALS EXPIRED -- cannot launch. Re-auth and relaunch this script."
      status "STOPPED $(date -u +%Y-%m-%dT%H:%M:%SZ): credentials expired"
      exit 2 ;;
    *)
      log "error describing reservation (retrying): $ERRTXT" ;;
  esac

  case "$STATE" in
    active)   log "block is ACTIVE -- deploying"; break ;;
    payment-failed)
      # Learned the hard way: a purchased block whose charge is declined never
      # becomes active, and its EndDate is silently rewritten to ~now. Waiting
      # for "active" here would wait forever. Payment health is not something
      # this script can fix -- exit loudly so a human sees it.
      log "FATAL: block payment FAILED -- the charge was declined, no capacity"
      log "  fix the account payment method in the Billing console, then re-purchase"
      status "FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ): block payment-failed"
      exit 8 ;;
    expired|cancelled|failed|payment-pending-cancelled)
      log "FATAL: block state is '$STATE' -- nothing to launch into"
      status "FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ): block $STATE"
      exit 4 ;;
    *) : ;;   # payment-pending / scheduled -- keep waiting
  esac
  sleep "$POLL_SECONDS"
done

status "BLOCK ACTIVE at $(date -u +%Y-%m-%dT%H:%M:%SZ); deploying"

# ---- 2. Deploy into the block ----------------------------------------------
cd "$REPO/cdk" || { log "FATAL: $REPO/cdk missing"; exit 1; }
if ! npx --yes aws-cdk@latest deploy NeuronPipelinesTrainium2 \
     --require-approval never \
     -c "trn2CapacityReservationId=$CR_ID" \
     -c "trn2Az=$AZ" -c "trn2SubnetId=$SUBNET" \
     >> "$DEPLOY_LOG" 2>&1; then
  log "DEPLOY FAILED -- see $DEPLOY_LOG"
  tail -20 "$DEPLOY_LOG"
  # Deliberately NOT cancelling anything: a Capacity Block cannot be cancelled
  # and is already paid for. Leave it open so a manual retry can still use it.
  status "DEPLOY FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ); block still open, retry by hand"
  exit 5
fi

IID=$(aws ec2 describe-instances --region "$REGION" \
  --filters Name=tag:Name,Values=neuron-pipelines-trn2 \
            Name=instance-state-name,Values=pending,running \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
log "DEPLOY OK: $IID in $AZ"
status "DEPLOYED $IID at $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting for SSM"

# ---- 3. Wait for SSM, then start the suite ---------------------------------
# The box is useless until the lanes are running, and user-data finishing is not
# the same as SSM being ready. Poll for the managed instance rather than sleeping
# a guessed interval.
for _ in $(seq 1 40); do
  PING=$(aws ssm describe-instance-information --region "$REGION" \
    --filters "Key=InstanceIds,Values=$IID" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)
  [ "$PING" = "Online" ] && break
  sleep 30
done
if [ "$PING" != "Online" ]; then
  log "WARNING: $IID never registered with SSM. Deploy succeeded; start the run by hand."
  status "DEPLOYED $IID but SSM never came online -- start the run by hand"
  exit 6
fi
log "SSM online; starting Phase-3 suite"

# SSM shells run as root with a BARE environment and never source
# /etc/profile.d/. Export everything the drivers need, including the venv bin
# dir -- torch-neuronx shells out to libneuronpjrt-path on init, and its absence
# cost seven lanes in Phase 2.
CMD_ID=$(aws ssm send-command --region "$REGION" \
  --instance-ids "$IID" \
  --document-name AWS-RunShellScript \
  --comment "neuron-pipelines phase3 trn2 kickoff" \
  --parameters 'commands=[
    "set -eux",
    "export NP_DEVICE=trn2 NP_REGION=sa-east-1 NP_CACHE_PREFIX=neuron-cache-v3",
    "export HF_HOME=/opt/np/models/hf",
    "export NEURON_COMPILE_CACHE_URL=/opt/np/cache/neuron-compile-cache",
    "export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH",
    "aws s3 cp s3://neuron-pipelines-artifacts-600627330911/code/shared/bin/pull_code.sh - | bash",
    "cd /opt/np/repo",
    "setsid nohup bash extras/run_phase3_trn2.sh >> /opt/np/phase3_trn2.log 2>&1 &",
    "setsid nohup bash extras/trn2_deadline_push.sh >> /opt/np/deadline_push.log 2>&1 &",
    "sleep 5; echo kicked off"
  ]' \
  --query 'Command.CommandId' --output text 2>&1)

if [ -z "$CMD_ID" ] || [ "$CMD_ID" = "None" ]; then
  log "WARNING: kickoff send-command failed. Box is UP and BILLING -- start the run by hand."
  status "DEPLOYED $IID; kickoff failed -- start the run by hand"
  exit 7
fi

log "kickoff sent (command $CMD_ID)"
log "block ends 11:30Z; TERMINATION BEGINS 11:00Z -- push results before then"
log "follow with: bash shared/bin/phase2_status.sh"
status "RUNNING on $IID since $(date -u +%Y-%m-%dT%H:%M:%SZ); deadline 11:00Z next day"
