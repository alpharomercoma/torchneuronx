# neuron-pipelines — Trainium + Inferentia, end to end

Train a modern 8B model on AWS Trainium1, serve it on Inferentia2, measure
everything. Read [METHODOLOGY.md](METHODOLOGY.md) before quoting anything.

**The claim under test:** AWS's own AI silicon, on its smallest rentable
instances (~$1/hr each), can run a *production-shaped* LLM pipeline — LoRA
fine-tune of Llama 3.1 8B Instruct on a trn1.2xlarge, adapter merged and
shipped through S3, compiled and served by vLLM on an inf2.xlarge — with the
serving and training metrics reported the way the GPU world reports them
(TTFT/TPOT/ITL/E2EL percentiles, tokens/s, MFU, loss traces, sustained
retention), compile costs included rather than hidden.

Companion repo: [MI300X-vs-H200](https://github.com/alpharomercoma/MI300X-vs-H200)
— same harness lineage, same metrics schema, so numbers are directly
comparable across repos.

## Machines

```
trn1.2xlarge  (us-west-2)                 inf2.xlarge  (us-west-2)
  1x Trainium1: 2 NeuronCores v2            1x Inferentia2: 2 NeuronCores v2
  32 GB HBM, ~820 GB/s                      32 GB HBM
  ~210 TFLOP/s dense BF16 (paper)           ~190 TFLOP/s dense BF16 (paper)
  8 vCPU, 32 GiB host RAM                   4 vCPU, 16 GiB host RAM
  Neuron DLAMI (PyTorch 2.9, Ubuntu 24.04)  Neuron vLLM 0.16 DLAMI (Ubuntu 24.04)
  torch-neuronx 2.9.0 | neuronx-cc 2.26     vllm 0.16.0 | NxDI 0.10 | neuronx-cc 2.26
  optimum-neuron 0.4.3 (pins: trl 0.24.0)   ami-035c945d557065665 (pin is load-bearing)
```

Both boxes are provisioned by the CDK app in [cdk/](cdk/), accessed only via
SSM Session Manager (no SSH, zero ingress rules), and stopped — not
terminated — between sessions so the Neuron compile cache on EBS survives.

## Models

| Model | Role | Regime it probes |
|---|---|---|
| Llama 3.1 8B Instruct | train (LoRA) + serve | the headline modern-model claim |
| Qwen3 8B | train (LoRA); serve *attempt* | a second architecture; serve outcome recorded either way |
| trn1 fine-tune (Llama 3.1 8B + dolly LoRA, merged) | serve | the end-to-end story |
| TinyLlama-1.1B | smoke only | $1 plumbing gate, never a headline |

## What is measured

See the lane table in [METHODOLOGY.md](METHODOLOGY.md#what-is-measured).
Short version: provenance → host CPU floor → smoke → (trn1) precompile,
LoRA SFT ×2, merge → (inf2) two-config serving sweeps, 30-min sustained,
quality, fine-tune serve, Qwen3 attempt.

## Repository layout

```
cdk/          AWS CDK app (Python): BaseStack, TrainiumStack, InferentiaStack
shared/       the harness -- synced byte-identical to both boxes
trn1/ inf2/   per-box: PROVISIONING docs, 4-line run wrapper, raw results
analysis/     make_report.py -> comparison.json -> every table in REPORT.md
demo/         live TTFT streamer + headline tables against a warm endpoint
docs/runbook/ 00..10, in execution order -- every command with expected output
tests/        local gate: fixtures, no AWS or Neuron hardware needed
```

## Reproducing

```bash
# 0. read docs/runbook/00-prerequisites.md (HF license, token, quotas)
python3 -m pytest tests/ -q          # local gate, no hardware
cd cdk && uv run pytest -q && cd ..  # infra assertions
cd cdk && uv run cdk deploy NeuronPipelinesBase NeuronPipelinesTrainium
# ... then follow docs/runbook/04..07 lane by lane; each box runs:
#   <box>/scripts/run_all.sh          # resumable; FORCE=1 to redo a lane
make pull-results && make report     # regenerate REPORT.md numbers
```

## Caveats worth knowing before you read the numbers

- Compile time is real and reported (rule 3) — the cold path costs tens of
  minutes per server config; the warm path is the production path.
- The serving concurrency grid tops out where KV memory says it must
  (rule 6); every sweep declares its grid in `grid.json`.
- Board power is not exposed on these instances: no energy numbers, by rule 7.
- One seed, one box per role, demonstration-scale SFT (known limits section).
- Trainium1/Inferentia2 are the previous Neuron generation — deliberately:
  that is what a newcomer can actually rent at these prices.

## Status

Complete. Measured results in [REPORT.md](REPORT.md); every number regenerates from `analysis/comparison.json` via `make report`.
