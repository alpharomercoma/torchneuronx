# cdk — infrastructure (AWS CDK, Python)

Three independently deployable CloudFormation stacks for account
`600627330911` in `us-west-2`:

| Stack | What it provisions |
|---|---|
| `NeuronPipelinesBase` | S3 artifacts bucket (RETAINed), EC2 instance role + profile, egress-only SG on the default VPC, $200/mo cost budget with email alerts |
| `NeuronPipelinesTrainium` | one `trn1.2xlarge` on the Neuron PyTorch-2.9 DLAMI (AMI via SSM parameter, resolved at deploy) |
| `NeuronPipelinesInferentia` | one `inf2.xlarge` on the Neuron PyTorch Inference vLLM DLAMI (AMI via lookup, cached in `cdk.context.json`) |

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

## Files

| File | What it does |
|---|---|
| `app.py` | entrypoint: wires the three stacks, pins account/region |
| `cdk.json` | `app` command (`uv run python app.py`), context defaults (`az`, `subnetId`, `volumeGb`, `budgetUsd`, `alertEmail`, `trn1InstanceType`, `inf2InstanceType`), feature flags |
| `stacks/base_stack.py` | bucket, scoped IAM role (no `Resource:"*"`), egress-only SG, CfnBudget |
| `stacks/trainium_stack.py` | trn1 instance; AMI from `/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id`; launch template carries IMDSv2 + 500 GiB gp3 (3000 IOPS / 250 MiB/s) root |
| `stacks/inferentia_stack.py` | inf2 instance; AMI lookup `Deep Learning AMI Neuron PyTorch Inference vLLM*Ubuntu 24.04*`, escape hatch `-c inf2AmiId=ami-...`; same posture |
| `user_data/common.sh` | `/opt/np` tree, `/etc/profile.d/neuron-pipelines.sh` env, uv install, done-marker |
| `user_data/trn1.sh` | `np-scratch.service`: instance-store NVMe → `/scratch` + 64 GiB swap, re-asserted every boot |
| `user_data/inf2.sh` | placeholder (inf2 has no instance store) |
| `tests/` | `aws_cdk.assertions.Template` tests, no AWS calls |
| `cdk.context.json` | cached lookup results (default VPC, inf2 AMI) — committed |
| `uv.lock` | pinned Python deps — committed |

## Commands

```bash
cd cdk
uv sync                                          # 18 packages into .venv (~247 MB)
uv run pytest                                    # 20 passed
npx --yes aws-cdk@2 synth --all --quiet          # 3 stacks -> cdk.out/, caches lookups into cdk.context.json

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
> `inf2.2xlarge`/`inf2.8xlarge` are one-flag upgrades via
> `-c inf2InstanceType=...` once the pending quota increase lands.

> **Gotcha:** trn1 instance store is **wiped on every stop/start**.
> `np-scratch.service` re-formats/re-mounts `/scratch` and re-creates the
> swapfile automatically at boot — don't keep the only copy of anything in
> `/scratch`.

> **Gotcha:** the root-volume gp3 `Throughput: 250` lives on the **launch
> template**, not the instance resource — CloudFormation's
> `AWS::EC2::Instance` Ebs mapping simply has no Throughput property.
