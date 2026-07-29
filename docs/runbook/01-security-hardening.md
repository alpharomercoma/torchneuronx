# 01 — security hardening (before any deploy)

> **Why this is step 01:** the account currently authenticates with ROOT
> access keys, and those keys were pasted into a chat session. Root keys
> cannot be permission-scoped; rotating them is not optional hygiene, it is
> incident response. Automation intentionally does not run these commands —
> IAM mutations are done by a human, once.

```bash
# 1. create the working IAM user
aws iam create-user --user-name neuron-pipelines-admin
aws iam attach-user-policy --user-name neuron-pipelines-admin \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# 2. new keys -> new profile (also make it the default)
aws iam create-access-key --user-name neuron-pipelines-admin \
  --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text
aws configure --profile neuron-pipelines   # paste the pair; region us-west-2
aws configure                              # paste the SAME pair as default
aws sts get-caller-identity --profile neuron-pipelines \
  --query Arn --output text                # ...user/neuron-pipelines-admin

# 3. retire the root keys (list, deactivate, verify nothing broke, delete)
aws iam list-access-keys                   # shows the root key id(s)
aws iam update-access-key --access-key-id AKIA... --status Inactive
aws sts get-caller-identity                # still works (now via IAM user)
aws iam delete-access-key --access-key-id AKIA...

# 4. MFA on both root and the new user (console, not CLI)
```

Done when: `aws iam list-access-keys` under root shows none, and every
later runbook command works with the default (IAM-user) profile.
