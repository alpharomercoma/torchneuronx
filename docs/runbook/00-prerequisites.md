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

## 2. Quotas

Already filed — see [02-quotas.md](02-quotas.md). Verify status today.

## 3. Local tooling

```bash
uv --version          # >= 0.4
node --version        # >= 18 (cdk CLI)
aws sts get-caller-identity --query Account --output text   # 600627330911
python3 -m pytest --version 2>/dev/null || uv tool install pytest
```
