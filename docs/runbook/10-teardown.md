# 10 — teardown

```bash
# final artifact + cache sync from each box before destroying anything
bash shared/bin/push_results.sh trn1 ; bash shared/bin/push_results.sh inf2
bash shared/bin/sync_neuron_cache.sh push

cd ~/neuron-pipelines/cdk
uv run cdk destroy NeuronPipelinesInferentia
uv run cdk destroy NeuronPipelinesTrainium
uv run cdk destroy NeuronPipelinesTrainium2   # sa-east-1; set DesiredCapacity 0 first
# BaseStack: keep until the final bill clears (budget alerts live there)

# Terminating a CFN-managed instance by hand drifts its stack, and a later
# deploy may recreate it. Destroy the stack, or accept the drift knowingly.

aws ec2 describe-instances --region us-west-2 \
  --query 'Reservations[].Instances[].State.Name' --output text  # (empty/terminated)
aws ec2 describe-volumes --region us-west-2 \
  --query 'Volumes[].VolumeId' --output text                     # (empty)
aws ssm delete-parameter --region us-west-2 --name /neuron-pipelines/hf-token
aws ssm delete-parameter --region sa-east-1 --name /neuron-pipelines/hf-token
```

## Before you destroy anything: snapshot

A root volume here has `DeleteOnTermination: true`, so terminating destroys the
disk and everything on it — including the Neuron compile cache, which is the
expensive part. Snapshot first; an EBS snapshot bills only written blocks, is
independent of the source volume, and needs no running instance.

```bash
aws ec2 create-snapshot --region us-west-2 --volume-id vol-... \
  --description 'np trn1 pre-teardown' \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=DoNotDelete,Value=true}]'
```

Two traps this study hit, both worth knowing before you start:

- **Persistent spot requests relaunch.** Terminating a spot-backed instance
  spawned a replacement within seconds. Cancel the spot request *first*, then
  terminate.
- **Not everything is on a disk.** The trn2 box had no snapshot, and 35.8 GiB
  of its results existed only in S3. Check for S3-only data before deleting a
  bucket.

## What was actually torn down

On **2026-08-26**, all seven instances across seven regions were terminated,
2,080 GiB of EBS deleted, and the account taken from ~$176/mo to $66.65/mo.
Seven snapshots totalling 1,333.0 GB were taken first, and the trn2 box's
S3-only data (51,280 objects, 54.55 GB) was archived to Glacier Deep Archive at
$0.054/month. Full record, with snapshot ids, AMI pins, KMS keys and the
restore steps: [ops/preservation/](../../ops/preservation/2026-08-26-RECOVERY.md).

Root access keys were **not** retired — [runbook 01](01-security-hardening.md)
is still open on this account.
