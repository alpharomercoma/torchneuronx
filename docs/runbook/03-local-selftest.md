# 03 — local self-test (gate 1: $0, no hardware)

Nothing deploys until all of this is green.

```bash
cd ~/neuron-pipelines
make test                     # 181 passed, 11 skipped (root) + 55 passed (cdk)
python3 shared/train/test_model.py                  # CPU self-test OK
cd cdk && uv sync && npx -y aws-cdk@2 synth > /dev/null && echo SYNTH OK
                                                    # SYNTH OK (4 stacks, 2 regions)
```

Counts as of 2026-08-27. `make test` runs both suites; the 11 skips are lanes
that need torch or Neuron hardware and are skipped by design on a laptop.

`uv sync` installs the Python side (`aws-cdk-lib`, the constructs the stacks
import); the CDK **CLI** is a separate npm package that this project
deliberately does not vendor, so every `cdk` invocation in these runbooks goes
through `npx -y aws-cdk@2`. `uv run cdk` fails with `Failed to spawn: cdk` —
there is no such console script in the venv.

> `cdk synth` performs read-only AWS lookups (default VPC, inf2 DLAMI) and
> caches them into `cdk/cdk.context.json`. Commit that file — it is what pins
> the deploy to reviewed values. **If you are reproducing this on a different
> AWS account, delete it first**: the cached ids belong to account
> 600627330911, and a stale cache makes synth succeed while targeting the
> wrong network. See the "Forking this to your own AWS account" section of the
> README for the full rename list.
