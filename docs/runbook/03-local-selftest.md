# 03 — local self-test (gate 1: $0, no hardware)

Nothing deploys until all of this is green.

```bash
cd ~/neuron-pipelines
uv run --with pytest python -m pytest tests/ -q     # 21 passed
python3 shared/train/test_model.py                  # CPU self-test OK
cd cdk && uv sync && uv run pytest -q               # 20 passed
uv run cdk synth > /dev/null && echo SYNTH OK       # SYNTH OK (3 stacks; measured 2026-07-29)
```

> `cdk synth` performs read-only AWS lookups (default VPC, inf2 DLAMI) and
> caches them into `cdk/cdk.context.json`. Commit that file — it is what pins
> the deploy to reviewed values.
