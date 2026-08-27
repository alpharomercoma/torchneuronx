# 06 — deploy Inferentia

No quota wait: inf2.xlarge fits the existing 8 vCPU quota.

> **AMI pin is load-bearing:** the latest "vLLM" Neuron DLAMI (0.21, 20260721)
> ships Trn2-only kernels and cannot boot ANY model on inf2 (see PROVISIONING
> gotchas). Deploy the vLLM 0.16 line explicitly:

```bash
cd ~/neuron-pipelines/cdk
npx -y aws-cdk@2 deploy NeuronPipelinesInferentia \
  -c inf2AmiId=ami-035c945d557065665              # vLLM 0.16 DLAMI; $0.76/hr STARTS HERE
aws ssm start-session --region us-west-2 --target i-...
# on box: pull code, hf_login, PROVISIONING verified-state block, then:
bash /opt/np/repo/shared/bin/sync_neuron_cache.sh pull   # seed NEFF cache if any transfers
cd /opt/np/repo && inf2/scripts/run_all.sh               # lane 2 smoke first
```
