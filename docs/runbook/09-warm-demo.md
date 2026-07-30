# 09 — warm demo path (rehearse this twice)

```bash
aws ec2 start-instances --region us-west-2 --instance-ids i-INF2  # ~40s to running
aws ssm start-session --region us-west-2 --target i-INF2
# on box:
cd /opt/np/repo && bash shared/serve/launch_vllm.sh llama31_dolly short /tmp/boot.json
cat /tmp/boot.json   # measured: fine-tune boots in 548 s against the 616 MB NEFF cache, 0 new compiles
```

From a second session (or port-forwarded laptop):

```bash
python3 demo/live_ttft.py --model /opt/np/models/llama31-8b-dolly-merged
# sweep-measured c1 reference: TTFT p50 ~397 ms, TPOT ~63 ms (rehearse to confirm live)
python3 demo/headline.py --serve
```

Measure minutes start→first-token at rehearsal; write them HERE. Stop the
instance after.
