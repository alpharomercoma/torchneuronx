# cdk — infrastructure (AWS CDK, Python)

Four independently deployable CloudFormation stacks across two regions, for
account `600627330911`:

| Stack | Region | What it provisions |
|---|---|---|
| `NeuronPipelinesBase` | us-west-2 | S3 artifacts bucket (RETAINed), EC2 instance role + profile, egress-only SG on the default VPC, $200/mo cost budget with email alerts |
| `NeuronPipelinesTrainium` | us-west-2 | one `trn1.2xlarge` on the Neuron PyTorch-2.9 DLAMI (AMI via SSM parameter, resolved at deploy) |
| `NeuronPipelinesInferentia` | us-west-2 | one `inf2.xlarge` on the Neuron PyTorch Inference vLLM DLAMI (AMI via lookup, cached in `cdk.context.json`) |
| `NeuronPipelinesTrainium2` | sa-east-1 | one `trn2.3xlarge`. By default a plain `AWS::EC2::Instance`; it becomes an AutoScalingGroup only when `trn2ScheduleLaunchAt` (Capacity Block) or capacity-hunt context is set. Separate region because sa-east-1 is the only region offering the small Trainium2 SKU **on demand** — Capacity Blocks also list it in ap-southeast-4 (Melbourne), see REPORT-EXTENSIONS §16 — and a separate stack because CloudFormation cannot reference the Base stack's VPC/SG/role across regions. It still writes to the us-west-2 bucket. |

> **Deployment state, 2026-08-27.** `NeuronPipelinesTrainium`,
> `NeuronPipelinesInferentia` and `Ec2AutoshutdownStack` were **deleted** on
> 2026-08-26 with their instances; `NeuronPipelinesBase` and
> `NeuronPipelinesTrainium2` remain. Redeploying the two lane stacks recreates
> working boxes and restarts billing. See
> [ops/preservation/](../ops/preservation/2026-08-26-RECOVERY.md).

Access is SSM Session Manager only: no SSH keys, zero ingress rules, IMDSv2
required.

## Setup

First-run side effects, so nothing surprises you:

- `uv sync` creates `cdk/.venv` (~247 MB — aws-cdk-lib ships the whole CFN
  spec) with 18 packages; first run takes ~30 s, cached after.
- `cdk bootstrap` is **one-time per account/region**: it creates the
  `CDKToolkit` CloudFormation stack (staging S3 bucket + IAM roles). Skip if
  this account/region was ever bootstrapped.
- `cdk synth` shells out to `uv run python app.py` (see `cdk.json`) and
  performs **read-only** AWS calls (`DescribeVpcs`, `DescribeSubnets`,
  `DescribeImages`) to resolve the default VPC and the inf2 AMI. Results are
  cached into `cdk.context.json`.
- Unit tests make **no** AWS calls — lookup context is pre-seeded in
  `tests/conftest.py`.

> **Gotcha:** `Vpc.from_lookup()` / `MachineImage.lookup()` need live AWS
> credentials on first synth, and they cache into `cdk.context.json` — that
> file **MUST be committed** so later synths (and CI) are deterministic and
> credential-free. The inf2 AMI stays pinned there until you
> `npx aws-cdk@2 context --reset <key>`.
>
> The flip side, and it bites silently: the cached VPC, subnet and AMI ids
> belong to account `600627330911`. Synthesising against a different account
> with this file in place **succeeds** and targets the wrong network. Delete
> `cdk.context.json` first and let it re-resolve. The README's
> ["Forking this to your own AWS account"](../README.md#forking-this-to-your-own-aws-account)
> section lists every other account-pinned value.

## Files

| File | What it does |
|---|---|
| `app.py` | entrypoint: wires the four stacks across two regions, pins account/region |
| `cdk.json` | `app` command (`uv run python app.py`), context defaults (`az`, `subnetId`, `volumeGb`, `budgetUsd`, `alertEmail`, `trn1InstanceType`, `inf2InstanceType`), feature flags |
| `stacks/base_stack.py` | bucket, scoped IAM role (no `Resource:"*"`), egress-only SG, CfnBudget |
| `stacks/trainium_stack.py` | trn1 instance; AMI from `/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id`; launch template carries IMDSv2 + 500 GiB gp3 (3000 IOPS / 250 MiB/s) root |
| `stacks/trainium2_stack.py` | trn2 box in sa-east-1; same DLAMI SSM parameter (no AMI pin needed on v3); plain instance by default, ASG under `trn2ScheduleLaunchAt` / capacity-hunt context; cross-region IAM scoped to the us-west-2 bucket ARN and the sa-east-1 hf-token ARN |
| `stacks/inferentia_stack.py` | inf2 instance; AMI lookup `Deep Learning AMI Neuron PyTorch Inference vLLM*Ubuntu 24.04*`, escape hatch `-c inf2AmiId=ami-...`; same posture |
| `user_data/common.sh` | `/opt/np` tree, `/etc/profile.d/neuron-pipelines.sh` env, uv install, done-marker |
| `user_data/trn1.sh` | `np-scratch.service`: instance-store NVMe → `/scratch` + 64 GiB swap, re-asserted every boot |
| `user_data/inf2.sh` | inf2 has no instance store, so swap goes on the EBS root: a 48 GiB swapfile. vLLM stages ~15-16 GB of 8B weights host-side before they reach HBM, and a 16 GiB host gets OOM-killed mid-load (measured: EngineCore killed at 14.5 GB RSS) |
| `user_data/trn2.sh` | trn2 equivalent of `trn1.sh`: NVMe `/scratch`, no swapfile (128 GiB host RAM), `NEURON_LOGICAL_NC_CONFIG=2`, `NP_CACHE_PREFIX=neuron-cache-v3` |
| `user_data/trn2_autorun.sh`, `trn2_hunt.sh` | Capacity-Block and capacity-hunt boot paths: start the driver unattended, then self-terminate at the window's end |
| `tests/` | `aws_cdk.assertions.Template` tests, no AWS calls |
| `cdk.context.json` | cached lookup results (default VPC, subnets, inf2 AMI) — committed. **Account-specific: delete it before synthesising against a different account** (see below) |
| `uv.lock` | pinned Python deps — committed |

## Commands

```bash
cd cdk
uv sync                                          # 18 packages into .venv (~247 MB)
uv run pytest                                    # 55 passed
npx --yes aws-cdk@2 synth --all --quiet          # 4 stacks -> cdk.out/, caches lookups into cdk.context.json

# one-time per account/region (already done if CDKToolkit stack exists):
npx --yes aws-cdk@2 bootstrap aws://600627330911/us-west-2   # creates CDKToolkit stack

# deploy in order (base first — the lanes import its role/SG):
npx --yes aws-cdk@2 deploy NeuronPipelinesBase                # free-tier-ish: bucket + IAM + budget
npx --yes aws-cdk@2 deploy NeuronPipelinesTrainium            # COSTS $1.34/hr while running
npx --yes aws-cdk@2 deploy NeuronPipelinesInferentia          # COSTS ~$0.758/hr (inf2.xlarge)

# quota fallback / upgrades — instance type is one context flag:
npx --yes aws-cdk@2 deploy NeuronPipelinesInferentia -c inf2InstanceType=inf2.8xlarge   # $1.97/hr, needs 32-vCPU Inf quota

# stop-not-terminate lifecycle (EBS persists, compute billing stops):
aws ec2 stop-instances  --region us-west-2 --instance-ids <InstanceId>   # ~$0.11/hr keeps accruing for the 500 GiB gp3 volume
aws ec2 start-instances --region us-west-2 --instance-ids <InstanceId>   # same instance id, new instance-store contents on trn1

# tear down (reverse order; base last):
npx --yes aws-cdk@2 destroy NeuronPipelinesInferentia         # terminates the inf2 box
npx --yes aws-cdk@2 destroy NeuronPipelinesTrainium           # terminates the trn1 box
npx --yes aws-cdk@2 destroy NeuronPipelinesTrainium2          # sa-east-1; set DesiredCapacity 0 first
npx --yes aws-cdk@2 destroy NeuronPipelinesBase               # bucket is RETAINed (artifacts survive)
```

`<InstanceId>` is in each lane stack's outputs, as is the ready-made
`aws ssm start-session` connect command.

> **Gotcha:** changing anything in `user_data/*.sh` changes the instance's
> user data, and CloudFormation **replaces the instance** — the Neuron compile
> cache and anything not synced to S3 dies with it. Sync `/opt/np` to
> `s3://neuron-pipelines-artifacts-600627330911` before deploying user-data
> changes.

> **Gotcha:** `inf2.xlarge` has only **16 GiB host RAM**. Keep heavyweight
> client tooling (fat conda stacks, local model conversion, big notebooks) off
> this box — compile on the trn1 or locally, ship artifacts via the bucket.
> **There is no `inf2.2xlarge`** — the family jumps `xlarge` → `8xlarge` (32
> vCPU), which needs Inf quota for 32 vCPU. On an 8-vCPU quota the swapfile in
> `user_data/inf2.sh` is the path, not a bigger instance.

> **Gotcha:** trn1 instance store is **wiped on every stop/start**.
> `np-scratch.service` re-formats/re-mounts `/scratch` and re-creates the
> swapfile automatically at boot — don't keep the only copy of anything in
> `/scratch`.

> **Gotcha:** the root-volume gp3 `Throughput: 250` lives on the **launch
> template**, not the instance resource — CloudFormation's
> `AWS::EC2::Instance` Ebs mapping simply has no Throughput property.
