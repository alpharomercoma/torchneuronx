# 09 — warm demo path (rehearse this twice)

> **This page cannot be executed against this account any more.** The inf2 box
> was terminated on 2026-08-26 and its EBS volume — which held the 616 MB warm
> NEFF cache — went with it. The cache survives inside snapshot
> `snap-0a25db44c14e44eb0` (517.7 GB) recorded in
> [ops/preservation/](../../ops/preservation/2026-08-26-RECOVERY.md); restoring
> it is the fast path back to a warm demo. On a fresh account, deploy per
> runbook 06, run the lanes, and the numbers below are what to expect.

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

Measured at rehearsal: **548 s** from `start-instances` to first token against
the 616 MB warm cache, 0 new compiles. Cold — no cache — the same path is a
2,372 s compile before the server will answer at all. That gap is the demo.

Stop the instance after. (This one was later terminated; see the banner above.)
