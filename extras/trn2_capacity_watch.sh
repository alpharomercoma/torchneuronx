#!/usr/bin/env bash
# Watch sa-east-1 for trn2.3xlarge capacity, grab it the moment it appears, and
# deploy the Trainium2 stack into whichever AZ opened.
#
# Runs UNATTENDED from the laptop, detached from any terminal:
#
#   nohup bash extras/trn2_capacity_watch.sh >> ~/trn2_capacity_watch.log 2>&1 &
#   disown
#
# (macOS has no setsid; nohup + disown is the equivalent. Wrap in `caffeinate -i`
# if the laptop might sleep -- a sleeping machine polls nothing.)
#
# WHY A CAPACITY RESERVATION IS THE PROBE
# ---------------------------------------
# `cdk deploy` is the wrong poll: each failed attempt burns ~5 minutes and
# leaves the stack in ROLLBACK_COMPLETE. `create-capacity-reservation` fails in
# seconds, costs nothing when capacity is absent, and when it SUCCEEDS it holds
# the slot -- so CloudFormation cannot lose the race to another account in the
# four minutes it needs to launch.
#
# Quota is not capacity. L-2C3B7624 was granted at 12 vCPU in sa-east-1 (exactly
# one trn2.3xlarge) on 2026-08-04, and all three AZs still returned
# InsufficientInstanceCapacity. Note the error text always names the OTHER two
# AZs as available -- that is a generic template, not availability information,
# and following it just walks you around the ring. Hence: rotate all three every
# round and believe none of them.
#
# BILLING: a reservation bills at the on-demand rate from creation, occupied or
# not. Exposure is bounded three ways -- RESERVE_HOURS self-expiry, deploying
# immediately on success, and cancelling the hold if that deploy fails.
set -uo pipefail
export AWS_PAGER=""

REGION=sa-east-1
REPO="${REPO:-/Users/alpha/neuron-pipelines}"
RESERVE_HOURS="${RESERVE_HOURS:-3}"
DEADLINE_HOURS="${DEADLINE_HOURS:-48}"
SLEEP_SECONDS="${SLEEP_SECONDS:-150}"
STATUS_FILE="${STATUS_FILE:-$HOME/trn2_capacity_watch.status}"
DEPLOY_LOG="${DEPLOY_LOG:-$HOME/trn2_capacity_deploy.log}"
LOCK="${LOCK:-$HOME/.trn2_capacity_watch.lock}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
status() { echo "$*" > "$STATUS_FILE"; }

# Single-instance guard: two watchers would race for the same reservation and
# could leave a paid-for hold with no instance in it.
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "another watcher is already running (pid $(cat "$LOCK")); exiting"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# macOS ships bash 3.2, which has no associative arrays.
subnet_for() {
  case "$1" in
    sa-east-1a) echo subnet-0489739583976c545 ;;
    sa-east-1b) echo subnet-092833ea4d9c0210c ;;
    sa-east-1c) echo subnet-088dd390f7aab1c53 ;;
  esac
}

DEADLINE=$(( $(date +%s) + DEADLINE_HOURS*3600 ))
attempt=0
log "watch started: trn2.3xlarge in $REGION, ${DEADLINE_HOURS}h deadline, ${SLEEP_SECONDS}s between rounds (pid $$)"
status "WATCHING since $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  attempt=$((attempt+1))
  for az in sa-east-1a sa-east-1b sa-east-1c; do
    END=$(date -u -v+${RESERVE_HOURS}H +%Y-%m-%dT%H:%M:%SZ)
    # Capture stderr rather than discarding it. Discarding it made an expired
    # session look identical to absent capacity, so the watcher polled blind
    # for hours and could never have succeeded even if a slot opened. An error
    # this loop cannot act on must be loud, not silent.
    ERR=$(mktemp)
    CRID=$(aws ec2 create-capacity-reservation \
      --instance-type trn2.3xlarge --instance-platform Linux/UNIX \
      --availability-zone "$az" --instance-count 1 \
      --end-date-type limited --end-date "$END" \
      --instance-match-criteria open \
      --region "$REGION" \
      --query 'CapacityReservation.CapacityReservationId' --output text 2>"$ERR")
    ERRTXT=$(tr -d '\n' < "$ERR"); rm -f "$ERR"

    if [ -z "$CRID" ] || [ "$CRID" = "None" ]; then
      case "$ERRTXT" in
        *InsufficientInstanceCapacity*|*"Insufficient capacity"*)
          : ;;  # the expected answer; keep polling quietly
        *ExpiredToken*|*"session has expired"*|*InvalidClientTokenId*|\
        *UnrecognizedClient*|*AuthFailure*|*"security token"*)
          log "CREDENTIALS EXPIRED -- cannot poll. Run 'aws login', then relaunch:"
          log "  nohup caffeinate -i bash extras/trn2_capacity_watch.sh >> ~/trn2_capacity_watch.log 2>&1 & disown"
          status "STOPPED $(date -u +%Y-%m-%dT%H:%M:%SZ): AWS credentials expired -- re-auth and relaunch"
          exit 2 ;;
        *UnauthorizedOperation*|*AccessDenied*)
          log "NOT AUTHORIZED to create capacity reservations: $ERRTXT"
          status "STOPPED $(date -u +%Y-%m-%dT%H:%M:%SZ): not authorized"
          exit 3 ;;
        ?*)
          log "unexpected error in $az (continuing): $ERRTXT" ;;
      esac
      continue
    fi

    log "CAPACITY SECURED: $CRID in $az (round $attempt)"
    status "CAPACITY SECURED $CRID in $az at $(date -u +%Y-%m-%dT%H:%M:%SZ); deploying"
    SUBNET=$(subnet_for "$az")
    log "deploying NeuronPipelinesTrainium2 into $az / $SUBNET"

    cd "$REPO/cdk" || { log "FATAL: $REPO/cdk missing"; exit 1; }
    if npx --yes aws-cdk@latest deploy NeuronPipelinesTrainium2 \
         --require-approval never \
         -c "trn2Az=$az" -c "trn2SubnetId=$SUBNET" \
         >> "$DEPLOY_LOG" 2>&1; then
      IID=$(aws ec2 describe-instances --region "$REGION" \
        --filters Name=tag:Name,Values=neuron-pipelines-trn2 \
                  Name=instance-state-name,Values=pending,running \
        --query 'Reservations[0].Instances[0].InstanceId' --output text)
      log "DEPLOY OK: instance $IID in $az"
      log "next: follow docs/runbook/12-trainium2.md -- pull_code, hf_login, then"
      log "      setsid nohup bash extras/run_phase3_trn2.sh >> /opt/np/phase3_trn2.log 2>&1 &"
      status "DEPLOYED $IID in $az at $(date -u +%Y-%m-%dT%H:%M:%SZ) -- box is RUNNING and BILLING"
    else
      log "DEPLOY FAILED after securing capacity; see $DEPLOY_LOG"
      tail -5 "$DEPLOY_LOG"
      # Release the hold rather than paying for an idle reservation.
      if aws ec2 cancel-capacity-reservation --capacity-reservation-id "$CRID" \
           --region "$REGION" >/dev/null 2>&1; then
        log "reservation $CRID cancelled (not paying for an empty hold)"
      else
        log "WARNING: could not cancel $CRID -- it bills until $END. Cancel it by hand."
      fi
      status "DEPLOY FAILED at $(date -u +%Y-%m-%dT%H:%M:%SZ); reservation $CRID released"
    fi
    exit 0
  done

  # One heartbeat every ~30 min, not every round.
  [ $((attempt % 12)) -eq 0 ] && log "still no capacity after $attempt rounds"
  sleep "$SLEEP_SECONDS"
done

log "GAVE UP: no trn2.3xlarge capacity in $REGION after ${DEADLINE_HOURS}h ($attempt rounds)"
status "GAVE UP at $(date -u +%Y-%m-%dT%H:%M:%SZ) after ${DEADLINE_HOURS}h"
