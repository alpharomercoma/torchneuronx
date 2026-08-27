# 12 — Trainium2 lane (trn2.3xlarge, sa-east-1)

Phase 3. Adds a third box so the study can answer the question REPORT.md §12
explicitly disclaims: **how much faster is one Trainium2 chip than one
Trainium1 chip on the same LoRA workload?**

Everything here is a different region from the rest of the project. That is not
a preference — `trn2.3xlarge` is offered **only in sa-east-1 on demand**
(verified 2026-08-03 with `describe-instance-type-offerings` across all 17
enabled regions). That API answers only for on-demand: the Capacity Blocks
supported-regions table also lists the type in ap-southeast-4 (Melbourne), so
availability is per *purchase mechanism*, not merely per region — see
REPORT-EXTENSIONS §16 correction 4. It is the only Trainium2 SKU
this account can run: quota `L-2C3B7624` was granted at **12 vCPU**, which is
exactly one instance.

## The two chips

| | trn1.2xlarge (us-west-2) | trn2.3xlarge (sa-east-1) |
|---|---|---|
| Chips | 1× Trainium1 | 1× Trainium2 |
| Physical NeuronCores | 2 × v2 | 8 × v3 |
| Logical cores (LNC=2, the default) | n/a | 4 (`NC_V3d`) |
| HBM per chip | 32 GiB | 96 GiB @ 2.9 TB/s |
| **HBM per logical core** | **16 GiB** | **24 GiB** |
| BF16 dense peak | 210 TFLOP/s | 667 TFLOP/s (3.18×) |
| vCPU / host RAM | 8 / 32 GiB | 12 / 128 GiB |
| Local NVMe | ~475 GB | 470 GB |
| Price | $1.34/hr | **$2.235/hr** — no pricing-API record; derived from a 24 h Capacity Block at $53.64 |

The per-core HBM row is the one that matters most: seq-len 8192 died on trn1
with `NCC_EOOM002] Maximum peak HBM usage of 18.12GB exceeds HBM limit of
16.00GB` (`trn1/results/extras/ctx_8192.failure.json`). 24 GiB should clear it.

## One-time prerequisites (already done 2026-08-03)

```bash
npx aws-cdk@latest bootstrap aws://600627330911/sa-east-1

# Replicate the HF token so the box never reaches across regions for a secret.
# Never echo the value.
TOK=$(aws ssm get-parameter --name /neuron-pipelines/hf-token --with-decryption \
      --query Parameter.Value --output text --region us-west-2)
aws ssm put-parameter --name /neuron-pipelines/hf-token --type SecureString \
  --value "$TOK" --region sa-east-1 --overwrite
unset TOK
```

`NeuronPipelinesBase` is **not** deployed here. The Trainium2 stack mints its
own VPC lookup, security group and role (CloudFormation cannot reference
another region's resources) but reads and writes the **us-west-2** artifacts
bucket by ARN, so all three boxes report into one `make_report.py`.

## Deploy

```bash
cd cdk && npx aws-cdk@latest deploy NeuronPipelinesTrainium2 --require-approval never
```

Outputs `InstanceId`, `AvailabilityZone` and a ready-to-paste `SsmConnect`.

### Quota is not capacity — expect to wait

All three sa-east-1 AZs *offer* trn2.3xlarge, but on 2026-08-04 **all three
returned `InsufficientInstanceCapacity`**. Note that the error text always
names the other two AZs as available; that is a generic template, not real
availability information, and following it just walks you around the ring.

```bash
# sa-east-1a subnet-0489739583976c545  (default)
# sa-east-1b subnet-092833ea4d9c0210c
# sa-east-1c subnet-088dd390f7aab1c53
npx aws-cdk@latest deploy NeuronPipelinesTrainium2 --require-approval never \
  -c trn2Az=sa-east-1b -c trn2SubnetId=subnet-092833ea4d9c0210c
```

**Do not poll with `cdk deploy`.** Each failed attempt costs ~5 minutes and
leaves the stack in `ROLLBACK_COMPLETE`. Poll with a capacity reservation
instead — it fails in seconds, costs nothing when capacity is absent, and when
it succeeds it *holds* the slot so CloudFormation cannot lose the race:

```bash
aws ec2 create-capacity-reservation --region sa-east-1 \
  --instance-type trn2.3xlarge --instance-platform Linux/UNIX \
  --availability-zone sa-east-1c --instance-count 1 \
  --instance-match-criteria open \
  --end-date-type limited --end-date "$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)"
```

A reservation **bills at the on-demand rate from the moment it is created**,
whether or not an instance occupies it — always set a limited `--end-date` so a
forgotten hold self-expires, and deploy immediately once one succeeds.

`ops/capacity/trn2_capacity_watch.sh` automates exactly this. It runs unattended from
the laptop, rotates all three AZs every 150 s, and on success deploys the stack
into whichever AZ opened — then exits:

```bash
nohup caffeinate -i bash ops/capacity/trn2_capacity_watch.sh \
  >> ~/trn2_capacity_watch.log 2>&1 &
disown          # macOS has no setsid; nohup + disown reparents to PID 1
```

`caffeinate -i` matters: a sleeping laptop polls nothing. Check on it with
`cat ~/trn2_capacity_watch.status` (one line: WATCHING / CAPACITY SECURED /
DEPLOYED / GAVE UP) or tail the log. A PID lockfile prevents two watchers
racing for the same reservation, and a failed deploy cancels the hold rather
than paying for an empty one.

### Capacity Blocks — the route that actually worked

On-demand polling never produced a slot. What did: **EC2 Capacity Blocks**,
once `L-64569A79` ("Concurrent TRN2 Capacity Blocks per account") was granted
on 2026-08-04. Before the grant, `describe-capacity-block-offerings` returned
`CapacityBlockDescribeLimitExceeded`; after it, real offerings appear. Note the
*per-organization* quota `L-24E8B4C0` is still 0 and does **not** block this —
the per-account grant is the one that counts.

```bash
aws ec2 describe-capacity-block-offerings --region sa-east-1 \
  --instance-type trn2.3xlarge --instance-count 1 --capacity-duration-hours 24
```

**24 h is the minimum** — 6 and 12 are rejected with `InvalidParameterValue`.
Offerings come with a fixed start time; you buy that window, not a duration.

```bash
aws ec2 purchase-capacity-block --region sa-east-1 \
  --capacity-block-offering-id cb-... --instance-platform Linux/UNIX
```

**This finally prices the instance.** $53.64 for 24 h = **$2.235/hr** for
trn2.3xlarge — 1.67× the trn1.2xlarge rate of $1.34/hr, for 3.18× the peak
FLOPs. The pricing API still has no on-demand record; this is the real number.

Two things about a block that an ODCR does not do:

1. **It bills for the entire window from its start time**, occupied or not. The
   clock does not wait for you. Automate the launch (see below) rather than
   planning to be awake for it.
2. **`InstanceMatchCriteria` is `targeted`.** An ordinary launch does *not*
   fall into the block — it fails on capacity exactly like every unreserved
   attempt. The launch must carry **both**:
   - `InstanceMarketOptions.MarketType = capacity-block`
   - `CapacityReservationTarget.CapacityReservationId = cr-...`

   `AWS::EC2::Instance` cannot express market options, so both live on the
   launch template. `cdk deploy` wires them when `trn2CapacityReservationId` is
   set, and **refuses to synth** unless `trn2Az`/`trn2SubnetId` are also given
   explicitly — a block lives in exactly one AZ, and silently defaulting to
   `sa-east-1a` would miss a `sa-east-1b` block and burn the paid window.

```bash
npx aws-cdk@latest deploy NeuronPipelinesTrainium2 --require-approval never \
  -c trn2CapacityReservationId=cr-08dc8b22d254cd3da \
  -c trn2Az=sa-east-1b -c trn2SubnetId=subnet-092833ea4d9c0210c
```

### Scheduled launch — keep the laptop off the critical path

`ops/capacity/trn2_block_launch.sh` waits for `active` and then deploys, but it runs
on a laptop, and that laptop must have its lid open and **unexpired** AWS
credentials at the instant the block opens. `aws login` issues temporary
credentials; ours expired mid-session more than once. A missed window costs the
entire upfront fee, so prefer the scheduled path and treat the laptop launcher
as a manual fallback only.

Pass `trn2ScheduleLaunchAt` and the stack builds an **Auto Scaling group** at
`desired=0` with two one-time scheduled actions instead of a bare instance.
AWS scales it to 1 when the block opens. This is the pattern AWS documents for
Capacity Blocks, and unlike a one-shot `RunInstances` it **retries transient
launch failures on its own**.

```bash
npx aws-cdk@latest deploy NeuronPipelinesTrainium2 --require-approval never \
  -c trn2CapacityReservationId=cr-08dc8b22d254cd3da \
  -c trn2Az=sa-east-1b -c trn2SubnetId=subnet-092833ea4d9c0210c \
  -c trn2ScheduleLaunchAt=2026-08-05T11:31:00Z \
  -c trn2ScheduleStopAt=2026-08-06T10:50:00Z
```

Both schedule flags are mandatory together, and the stack refuses to synth
without a reservation id. The **stop** action is not bookkeeping — per the Auto
Scaling guidance, *"Scale in your Auto Scaling group to zero more than 30
minutes before the Capacity Block reservation end time."* Left at `desired=1`,
EC2 reclaims the instance, the ASG reads that as a failed health check, and it
spends the rest of the window trying to replace an instance it can never get.

In scheduled mode the box also **starts the suite itself** from user-data
(`cdk/user_data/trn2_autorun.sh` → `np-autorun.service`): it waits for S3,
pulls the code, then launches `run_phase3_trn2.sh` and
`trn2_deadline_push.sh`. Scheduling the launch but not the kickoff would put
the laptop right back on the critical path and pay for an idle box. A plain
`cdk deploy` without the schedule flags still gives a quiet box to drive by
hand.

Verify AWS actually registered it — the deploy output is not proof:

```bash
ASG=$(aws cloudformation describe-stacks --stack-name NeuronPipelinesTrainium2 \
  --region sa-east-1 --query \
  'Stacks[0].Outputs[?OutputKey==`AutoScalingGroupName`].OutputValue' --output text)
aws autoscaling describe-scheduled-actions --auto-scaling-group-name "$ASG" \
  --region sa-east-1 --query \
  'ScheduledUpdateGroupActions[*].[StartTime,DesiredCapacity,Recurrence]' --output text
```

`Recurrence` must be `None` on both — a cron would relaunch daily into a block
that no longer exists.

**Do not leave the laptop launcher running alongside this.** It deploys
*without* the schedule flags, which would replace the ASG with a plain
instance. Likewise, stop the ODCR watcher before buying a block — if it ever
succeeded it would hold a second, separately-billing reservation.

No AMI pin is needed. The same SSM parameter the trn1 stack uses
(`/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id`) resolves
in-region to `ami-0b1b0d3aaa2171e1f`. The load-bearing `ami-035c945d557065665`
pin in runbook 06 is an **Inferentia** concern (vLLM 0.16 vs the Trn2-only 0.21
kernels) and does not apply to this training box.

## First connect

```bash
make push-code
aws ssm start-session --region sa-east-1 --target i-...
sudo -i
aws s3 cp s3://neuron-pipelines-artifacts-600627330911/code/shared/bin/pull_code.sh - | bash
cd /opt/np/repo
```

**SSM shells run as root with a bare environment and never source
`/etc/profile.d/`.** The drivers re-export what they need, but if you run
anything by hand, export these first — this is the same class of mistake that
misfiled a gated-model 401 as a compiler failure in Phase 2:

```bash
export NP_DEVICE=trn2 NP_REGION=sa-east-1 NP_CACHE_PREFIX=neuron-cache-v3
export HF_HOME=/opt/np/models/hf
export NEURON_COMPILE_CACHE_URL=/opt/np/cache/neuron-compile-cache
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH   # libneuronpjrt-path
bash shared/bin/hf_login.sh
```

Confirm the hardware before trusting any number — this also settles a real
discrepancy: the EC2 API reports `TotalNeuronDeviceMemoryInMiB = 524288`
(512 GiB) for this instance, while the Neuron docs say 96 GiB per chip.

```bash
neuron-ls
```

## Run

One command, disconnect-proof, resumable:

```bash
setsid nohup bash extras/run_phase3_trn2.sh >> /opt/np/phase3_trn2.log 2>&1 &
```

Completion marker: `PHASE3 TRN2 ALL COMPLETE`. Reattach from any machine with
`bash shared/bin/phase2_status.sh`.

The master runs, in this order and for these reasons:

1. **TP probe** (`extras/tp_probe_trn2.sh`) — the cheapest thing that can
   invalidate everything after it. World = TP, so the intended config is TP=4.

   **The docs are genuinely silent on whether world size 4 is valid on
   Trainium2**, which is the whole reason this is measured. The often-quoted
   "world size limited to 1, 2, 8, 32" sentence lives on a page tagged for Trn2,
   so it cannot be dismissed as stale — but it is worded as a *performance
   placement* heuristic and its examples (0/8/16/24) are trn1.32xlarge-shaped.
   Meanwhile AWS's own `neuronx-distributed` docs use `tensor_parallel_size=4`
   freely, no doc states a power-of-2 or divisibility rule, and **no AWS example
   anywhere runs a single Trainium2 chip at any TP**. So: ladder, not assumption.

   | rung | LNC | world | TP | note |
   |---|---|---|---|---|
   | 1 | 2 | 4 | 4 | the whole chip, 24 GiB per logical core |
   | 2 | 2 | 2 | 2 | **half the chip idle — not a 1:1 comparison** |
   | 3 | 1 | 8 | 8 | 8 physical v3 cores. Do **not** assume 12 GiB each: the docs say "both physical NeuronCores have access to the entire 24GB HBM bank" |

   If rung 2 wins, `full_chip: false` is recorded and **the report must say the
   comparison is against half a Trainium2**. If no rung passes, the master
   halts — there is no training lane without a working collective.

   Hard rule from the LNC docs: the compiler's `-lnc` flag and the runtime's
   `NEURON_LOGICAL_NC_CONFIG` **must match**. Neuron does not support compiling
   for one and running the other, so each rung sets the runtime variable before
   anything in that rung compiles.

2. **NEFF cache seed** from `s3://.../neuron-cache-v3/`. NEFFs are compiled per
   NeuronCore version: v3 artifacts must **never** land in the `neuron-cache/`
   prefix that trn1 and inf2 share. The first run is cold, so the full 8B
   compile is paid — and per METHODOLOGY rule 3, that is a result, not overhead.

3. **The main suite** (`trn2/scripts/run_all.sh`) — the same six lanes trn1 ran,
   through the same `shared/run_all.sh` branch, with the same hyperparameters
   (LoRA r16/α32, micro-batch 1, grad-accum 8, dolly-15k, 3 epochs, seed 42).
   **Do not tune this.** A tuned trn2 measured against an untuned trn1 is not a
   comparison.

4. **Extras** (`extras/run_extras_trn2.sh`) — context ladder 4096/8192/**16384**
   (16384 is new: it finds the *new* cliff rather than confirming the old one
   moved), checkpoint timing, and only then the efficiency levers E1 (seq 4096)
   and E2 (recompute off). Each lever is its own declared lane with its own
   triplet, so it is separately attributable against a baseline that already
   exists.

## Cost control

The rate is **$2.235/hr**, derived from the Capacity Block price rather than
published anywhere. A 24 h block is $53.64 against the **$100** ceiling, and it
is spent the moment the window opens — the remaining risk is not overspend but
*waste*, so the lanes must be running by 11:30Z. Confirm against the real bill
once a line item appears:

```bash
aws ce get-cost-and-usage --time-period Start=$(date -u +%Y-%m-%d),End=$(date -u -v+1d +%Y-%m-%d) \
  --granularity DAILY --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"REGION","Values":["sa-east-1"]}}' --region us-east-1
```

Ceiling for this phase: **$100**.

## Stop

```bash
bash shared/bin/sync_neuron_cache.sh push     # v3 NEFFs -> S3
bash shared/bin/push_results.sh trn2
aws ec2 stop-instances --region sa-east-1 --instance-ids i-...
```

**Stop, never terminate.** The warm v3 compile cache on EBS is the asset, and
`cdk destroy` would take the instance store and the cache with it.

**Push results before the block ends — the deadline is 11:00Z, not 11:30Z.**
AWS docs: *"Capacity Blocks end at 11:30AM Coordinated Universal Time (UTC). The
termination process for instances running in a Capacity Block begins at 11:00AM
Coordinated Universal Time (UTC) on the final day of the reservation."*

So the instance is **terminated**, not stopped — the one case where the
stop-never-terminate rule cannot save you. Anything not in S3 by 11:00Z on the
final day is gone, including the warm v3 NEFF cache on EBS. For the block
purchased here that means:

| | UTC |
|---|---|
| Window opens, billing starts | 2026-08-04 **11:30** |
| **Hard deadline — push everything** | 2026-08-05 **11:00** |
| Nominal end | 2026-08-05 11:30 |

Usable time is therefore **23.5 h**, not 24. Cancellations are not permitted,
so the money is spent either way.
