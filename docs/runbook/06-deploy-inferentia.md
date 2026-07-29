# 06 — deploy Inferentia

No quota wait: inf2.xlarge fits the existing 8 vCPU quota.

```bash
cd ~/neuron-pipelines/cdk
uv run cdk deploy NeuronPipelinesInferentia         # $0.76/hr STARTS HERE
aws ssm start-session --region us-west-2 --target i-...
# on box: pull code, hf_login, PROVISIONING verified-state block, then:
bash /opt/np/repo/shared/bin/sync_neuron_cache.sh pull   # seed NEFF cache if any transfers
cd /opt/np/repo && inf2/scripts/run_all.sh               # lane 2 smoke first
```
