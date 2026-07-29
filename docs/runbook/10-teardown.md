# 10 — teardown

```bash
# final artifact + cache sync from each box before destroying anything
bash shared/bin/push_results.sh trn1 ; bash shared/bin/push_results.sh inf2
bash shared/bin/sync_neuron_cache.sh push

cd ~/neuron-pipelines/cdk
uv run cdk destroy NeuronPipelinesInferentia
uv run cdk destroy NeuronPipelinesTrainium
# BaseStack: keep until the final bill clears (budget alerts live there)

aws ec2 describe-instances --region us-west-2 \
  --query 'Reservations[].Instances[].State.Name' --output text  # (empty/terminated)
aws ec2 describe-volumes --region us-west-2 \
  --query 'Volumes[].VolumeId' --output text                     # (empty)
aws ssm delete-parameter --region us-west-2 --name /neuron-pipelines/hf-token
aws iam list-access-keys   # still empty for root (01 stays done)
```
