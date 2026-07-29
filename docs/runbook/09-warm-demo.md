# 09 — warm demo path (rehearse this twice)

```bash
aws ec2 start-instances --region us-west-2 --instance-ids i-INF2  # ~40s to running
aws ssm start-session --region us-west-2 --target i-INF2
# on box:
cd /opt/np/repo && bash shared/serve/launch_vllm.sh llama31_dolly short /tmp/boot.json
cat /tmp/boot.json   # warm=true, boot_wall_s=TODO-VERIFY (target: minutes, not tens)
```

From a second session (or port-forwarded laptop):

```bash
python3 demo/live_ttft.py --model /opt/np/models/llama31-8b-dolly-merged
# [first token after ~XXX ms] ... TTFT=XXXms TPOT=XX.Xms  TODO-VERIFY at rehearsal
python3 demo/headline.py --serve
```

Measure minutes start→first-token at rehearsal; write them HERE. Stop the
instance after.
