# 00 — prerequisites (day 0: start the two slow clocks)

Two approvals have latency measured in days. Start both before touching code.

## 1. Hugging Face: Llama 3.1 license + token

1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   (approval can take hours–days; Qwen3-8B is ungated and is the fallback primary).
2. Create a **read** token: https://huggingface.co/settings/tokens
3. Store it in SSM (never in code, user-data, or git):

```bash
aws ssm put-parameter --region us-west-2 \
  --name /neuron-pipelines/hf-token --type SecureString \
  --value 'hf_...'                          # -> {"Version": 1}
aws ssm get-parameter --region us-west-2 --name /neuron-pipelines/hf-token \
  --query Parameter.Type --output text      # SecureString
```

If you will also run the Trainium2 lane (runbook 12), replicate the same
parameter into `sa-east-1` — SSM parameters are regional, and the trn2 box
cannot read a us-west-2 SecureString:

```bash
aws ssm put-parameter --region sa-east-1 \
  --name /neuron-pipelines/hf-token --type SecureString --value 'hf_...'
```

## 2. Quotas

See [02-quotas.md](02-quotas.md) for what each family needs — and for why an
approved quota still does not guarantee you can launch anything.

## 3. Local tooling

```bash
uv --version          # >= 0.4
node --version        # >= 18 (cdk CLI)
aws sts get-caller-identity --query Account --output text   # 600627330911 (ours)
python3 -m pytest --version 2>/dev/null || uv tool install pytest
```
