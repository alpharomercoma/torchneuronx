# 02 — quotas

Two separate gates, and confusing them costs days: a **quota** is permission to
run vCPU of a family in a region; **capacity** is whether AWS actually has a
machine to give you. This study hit both walls.

## What each box needs

| Instance | Quota code | vCPU needed | Region used |
|---|---|---|---|
| `trn1.2xlarge` | `L-2C3B7624` (On-Demand Trn) | 8 | us-west-2 |
| `inf2.xlarge` | `L-1945791B` (On-Demand Inf) | 4 | us-west-2 |
| `trn2.3xlarge` | `L-2C3B7624` (On-Demand Trn) | 12 | **sa-east-1 only** |

The core suite was designed to need **no quota approval**: that is precisely
why the inference box is an `inf2.xlarge`. Everything past it did need
approvals, and they were slow.

```bash
# what you have, in the region you care about
aws service-quotas get-service-quota --region us-west-2 \
  --service-code ec2 --quota-code L-2C3B7624 --query 'Quota.Value' --output text
aws service-quotas get-service-quota --region sa-east-1 \
  --service-code ec2 --quota-code L-2C3B7624 --query 'Quota.Value' --output text

# a request you have already filed
aws service-quotas get-requested-service-quota-change --region us-west-2 \
  --request-id <id> --query 'RequestedQuota.[Status,DesiredValue]' --output text
```

## What this account actually learned

- **Inf increases took 11–13 days each**, across two requests. Anything past
  the default needs to be filed on day 0, not when you need it.
- **A larger request needs a reply from you.** Support asks what the capacity
  is for; the clock does not start until you answer.
- **An approved quota can still be unusable.** Melbourne granted 8 vCPU of
  Trn — but the smallest Trainium instance offered there needs 12. Enabled and
  unusable.
- **Trainium2 was only reachable through a Capacity Block reservation**, which
  itself needs quota, has a 24-hour minimum, and is paid upfront ($53.64 for
  this study's window). It is non-refundable, so the run has to be ready before
  the window opens — see [12-trainium2.md](12-trainium2.md).
- **Quota granted, capacity empty.** Even with sa-east-1 quota in hand,
  on-demand `trn2.3xlarge` returned `InsufficientInstanceCapacity` across all
  three AZs for the entire study. Every trn2 number in the report came from the
  reserved block.
- **Trainium3 is not generally available.**

Budget the calendar, not just the money: the two slow clocks in
[00-prerequisites.md](00-prerequisites.md) are the HF license and these quotas.
