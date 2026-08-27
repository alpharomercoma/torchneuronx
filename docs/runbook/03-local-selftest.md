# 03 — local self-test (gate 1: $0, no hardware)

Nothing deploys until all of this is green.

```bash
cd ~/neuron-pipelines
make test                     # 181 passed, 11 skipped (root) + 55 passed (cdk)
python3 shared/train/test_model.py                  # CPU self-test OK
cd cdk && uv sync && uv run cdk synth > /dev/null && echo SYNTH OK
                                                    # SYNTH OK (4 stacks, 2 regions)
```

Counts as of 2026-08-27. `make test` runs both suites; the 11 skips are lanes
that need torch or Neuron hardware and are skipped by design on a laptop.

> `cdk synth` performs read-only AWS lookups (default VPC, inf2 DLAMI) and
> caches them into `cdk/cdk.context.json`. Commit that file — it is what pins
> the deploy to reviewed values. **If you are reproducing this on a different
> AWS account, delete it first**: the cached ids belong to account
> 600627330911, and a stale cache makes synth succeed while targeting the
> wrong network. See the "Forking this to your own AWS account" section of the
> README for the full rename list.
