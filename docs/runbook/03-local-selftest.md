# 03 — local self-test (gate 1: $0, no hardware)

Nothing deploys until all of this is green.

```bash
cd ~/neuron-pipelines
uv run --with pytest python -m pytest tests/ -q     # all passed
python3 shared/train/test_model.py                  # CPU self-test OK  TODO-VERIFY
cd cdk && uv sync && uv run pytest -q               # all passed        TODO-VERIFY
uv run cdk synth > /dev/null && echo SYNTH OK       # SYNTH OK (3 stacks) TODO-VERIFY
```

> `cdk synth` performs read-only AWS lookups (default VPC, inf2 DLAMI) and
> caches them into `cdk/cdk.context.json`. Commit that file — it is what pins
> the deploy to reviewed values.
