# 05 — training lanes (trn1)

```bash
# on the box, inside a tmux over SSM (lanes survive session drops):
cd /opt/np/repo && trn1/scripts/run_all.sh
# lane 3 precompile: expect tens of minutes wall, TODO-VERIFY actual
# lane 4 llama31 lora: hours-scale, TODO-VERIFY actual
# watch from a second session:
tail -f trn1/results/train/llama31_lora.log
watch -n 30 neuron-ls
```

After each lane lands: `bash shared/bin/push_results.sh trn1` and
`bash shared/bin/sync_neuron_cache.sh push`.

When lane 6 (merge) finishes, confirm the artifact round-trip hash matches
`train/merge_llama31.json`, then **stop the instance** (EBS + cache persist):

```bash
aws ec2 stop-instances --region us-west-2 --instance-ids i-...  # stopping
```
