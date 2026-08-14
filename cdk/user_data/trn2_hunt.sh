#!/bin/bash
# trn2_hunt.sh — mark this box as one the on-demand CAPACITY HUNT caught.
#
# Included in user-data only when -c trn2CapacityHunt=true, and ordered BEFORE
# trn2_autorun.sh so the selector exists by the time np-autorun reads it.
#
# WHY THE DISTINCTION MATTERS
# ---------------------------
# A Capacity Block box and a hunt box look identical at boot and must behave
# completely differently.
#
#   Capacity Block: paid for up front, for a fixed window, whether used or not.
#     Idle minutes are pure waste, so it runs the FULL Phase-3 suite and is
#     reclaimed by EC2 on a clock it does not control.
#
#   Hunt box: on-demand, billed by the second, and it appears at an unpredictable
#     moment -- possibly while nobody is awake. Its EBS is fresh from the launch
#     template, so trn2/results/ is EMPTY and the suite's resumability guards all
#     miss. Running the full suite here would re-do everything already banked in
#     S3, at a rate of at least $6.41/hr (that is the sa-east-1 spot price, and
#     spot never exceeds on-demand). It must instead rehydrate from S3, run only
#     the genuine gaps, and TERMINATE ITSELF.
#
# The self-termination is the whole point of choosing auto-run over a shorter
# spend ceiling: the ceiling is a backstop hours away, not a plan.
set -euxo pipefail

mkdir -p /opt/np
echo "run_gapfill_trn2.sh" > /opt/np/.autorun-driver

# ---------------------------------------------------------------------------
# BOOT WATCHDOG -- the spend bound for an UNBOUNDED hunt.
# ---------------------------------------------------------------------------
# The hunt now runs with no trn2HuntStopAt, because a ceiling that expires
# disarms the hunt on a date that has nothing to do with when capacity appears.
# Removing it also removes the only thing that used to bound spend, so the
# bound has to move onto the box itself.
#
# The driver terminates from an EXIT trap, which covers every path where the
# driver RUNS. It does not cover the box booting and the driver never starting
# at all -- np-autorun exits non-zero if the S3 code pull fails, and then
# nothing on the box has any opinion about when to stop. That box would bill at
# >= $6.41/hr forever. This timer is the answer to exactly that case.
#
# Unconditional on purpose. A watchdog that defers while "work looks busy" is
# defeated by the hung compile it most needs to kill (three hours lost to a
# stale compile lock in Phase 2 is the precedent). 6h against ~1h of expected
# work is a wide margin; if it ever fires during real work, results are already
# in S3 because it pushes before terminating.
cat > /usr/local/sbin/np-hunt-watchdog.sh <<'EOF'
#!/bin/bash
exec >> /opt/np/hunt-watchdog.log 2>&1
set -x
echo "watchdog firing $(date -u '+%Y-%m-%dT%H:%M:%SZ') -- box has lived past its budget"
export PATH=/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin

# Bank first, kill second. A mid-lane kill still keeps every finished lane.
bash /opt/np/repo/shared/bin/push_results.sh trn2 || echo "push failed or nothing to push"

TOKEN=$(curl -s -m 5 -X PUT http://169.254.169.254/latest/api/token \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
IID=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/instance-id)

# --should-decrement-desired-capacity is load-bearing: without it the group
# just launches a replacement and the meter keeps running. Retried because a
# transient API error here is the difference between a dead box and a box that
# bills indefinitely -- and retrying the RIGHT call beats falling back to a
# `shutdown`, which halts the instance without decrementing anything and so
# invites the ASG to replace it.
for attempt in 1 2 3 4 5; do
  if [ -n "$IID" ] && aws autoscaling terminate-instance-in-auto-scaling-group \
       --region sa-east-1 --instance-id "$IID" --should-decrement-desired-capacity; then
    echo "terminated $IID via ASG with decrement (attempt $attempt)"
    exit 0
  fi
  echo "terminate attempt $attempt failed; retrying"
  sleep 30
done

echo "ASG terminate failed 5x -- halting instead; the group may replace this box"
shutdown -h now
EOF
chmod +x /usr/local/sbin/np-hunt-watchdog.sh

cat > /etc/systemd/system/np-hunt-watchdog.service <<'SVC'
[Unit]
Description=neuron-pipelines trn2 hunt: hard spend bound, terminate the box

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/np-hunt-watchdog.sh
SVC

cat > /etc/systemd/system/np-hunt-watchdog.timer <<'TMR'
[Unit]
Description=neuron-pipelines trn2 hunt: fire the spend bound 6h after boot

[Timer]
# Relative to boot, not to a wall-clock time: a hunt box appears at an
# unpredictable instant, so the only meaningful clock is its own uptime.
OnBootSec=6h
AccuracySec=1min

[Install]
WantedBy=timers.target
TMR

systemctl daemon-reload
systemctl enable --now np-hunt-watchdog.timer

date -u '+%Y-%m-%dT%H:%M:%SZ' > /opt/np/.userdata-trn2-hunt-done
