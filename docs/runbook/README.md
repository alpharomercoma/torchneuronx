# runbook — the study in execution order

Fourteen steps, `00` through `13`, each one a short page of commands with the
output they actually produced. Read them in order the first time; after that
they are reference.

> **State, 2026-08-27.** The study is complete through Phase 5 and **all three
> instances were terminated on 2026-08-26**. These pages are written as a
> procedure a reproducer can follow on their own account, not as a live
> operations guide for this one. Anything account-specific — the account id,
> the artifacts bucket name, `cdk/cdk.context.json` — must be changed first:
> see [Forking this to your own AWS account](../../README.md#forking-this-to-your-own-aws-account).
> The teardown record is in [ops/preservation/](../../ops/preservation/2026-08-26-RECOVERY.md).

| # | page | what it covers | costs money |
|---|---|---|---|
| 00 | [prerequisites](00-prerequisites.md) | HF license + token into SSM, local tooling | no |
| 01 | [security hardening](01-security-hardening.md) | retire root access keys, IAM user, MFA | no |
| 02 | [quotas](02-quotas.md) | what each instance family needs, and why quota ≠ capacity | no |
| 03 | [local self-test](03-local-selftest.md) | gate 1: 236 tests + synth, no AWS, no hardware | no |
| 04 | [deploy Trainium](04-deploy-trainium.md) | Base + trn1 stacks, push harness, smoke lane | **yes** — $1.34/hr |
| 05 | [training lanes](05-training-lanes.md) | trn1 lanes 3-6: precompile, two LoRA runs, merge | **yes** |
| 06 | [deploy Inferentia](06-deploy-inferentia.md) | inf2 stack — the AMI pin is load-bearing | **yes** — $0.76/hr |
| 07 | [inference lanes](07-inference-lanes.md) | inf2 serving sweeps, sustained, quality, fine-tune | **yes** |
| 08 | [analysis and report](08-analysis-report.md) | pull results, regenerate comparison.json | no |
| 09 | [warm demo](09-warm-demo.md) | the live demo path, rehearsed against a warm cache | **yes** |
| 10 | [teardown](10-teardown.md) | stop, destroy, verify nothing is left billing | no |
| 11 | [Phase-2 extensions](11-extensions.md) | the extension lanes on both boxes | **yes** |
| 12 | [Trainium2](12-trainium2.md) | sa-east-1, Capacity Blocks, LNC, the trn1↔trn2 comparison | **yes** |
| 13 | [Phase-4 post-training](13-phase4-posttraining.md) | ORPO, DPO, GRPO — what worked and what is architecturally blocked | **yes** |

## The three environment rules

Violating any one of these wastes a full pass. They are repeated in 11 because
that is where they were first paid for:

1. **PATH must contain the venv bin** before any Neuron process starts
   (`libneuronpjrt-path`).
2. **Never install optimum-neuron into the inf2 vLLM venv** — duelling platform
   plugins brick all serving.
3. **Export `HF_HOME` and run `shared/bin/hf_login.sh`** in any shell touching
   gated models. A bare SSM shell has neither.
