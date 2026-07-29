# 02 — quotas

State as of 2026-07-29, us-west-2 (all other regions are zero for both):

| Quota | Code | Value | Enough for |
|---|---|---|---|
| Running On-Demand Trn instances | L-2C3B7624 | 64 vCPU | trn1.2xlarge (needs 8) ✔ |
| Running On-Demand Inf instances | L-1945791B | 8 vCPU | **inf2.xlarge (needs 4) ✔** |

The suite as designed needs **no quota approval** — that is why the
inference box is inf2.xlarge. An increase to 64 was filed anyway as an
upgrade path (8xlarge host headroom, or parallel boxes):

```bash
aws service-quotas get-requested-service-quota-change --region us-west-2 \
  --request-id 8c51c46cc7564c66ae9277e5bea96577S5k1yKWp \
  --query 'RequestedQuota.[Status,DesiredValue]' --output text
# PENDING 64.0        (check daily; APPROVED unlocks -c inf2InstanceType=inf2.8xlarge)
```
