# 04 — deploy Base + Trainium, verify, smoke

> The account id and bucket name below are this study's. On another account,
> change them first — [Forking this to your own AWS
> account](../../README.md#forking-this-to-your-own-aws-account) lists every
> site, and `cdk/cdk.context.json` must be **deleted** before the first synth
> or it will resolve to this account's VPC.

```bash
cd ~/neuron-pipelines/cdk
npx -y aws-cdk@2 bootstrap aws://600627330911/us-west-2   # one-time
npx -y aws-cdk@2 deploy NeuronPipelinesBase               # bucket, role, SG, budget
npx -y aws-cdk@2 deploy NeuronPipelinesTrainium           # $1.34/hr STARTS HERE
# Outputs: InstanceId=i-...  SsmConnect="aws ssm start-session ..."
```

Push the harness, then verify on-box (SSM session):

```bash
cd ~/neuron-pipelines && make push-code
aws ssm start-session --region us-west-2 --target i-...   # from stack output
# on the box:
aws s3 cp s3://neuron-pipelines-artifacts-600627330911/code/shared/bin/pull_code.sh - | bash
bash /opt/np/repo/shared/bin/hf_login.sh        # HF token installed
# work through trn1/docs/PROVISIONING.md "Verified state" -- paste real output there
```

Smoke (gate 2, ~$0.50):

```bash
cd /opt/np/repo && FORCE=0 trn1/scripts/run_all.sh   # lanes 0-2 only matter here
ls trn1/results/train/smoke_tinyllama.{json,log,telemetry.csv}   # triplet exists
bash shared/bin/push_results.sh trn1
```

Laptop: `make pull-results` → commit the smoke triplet → replace
`tests/fixtures/neuron_monitor_inf2.json` with a real captured line
(the fixture-becomes-real rule) → 8B lanes are now allowed.
