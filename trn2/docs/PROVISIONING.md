# trn2.3xlarge provisioning notes (sa-east-1)

> **This box no longer exists, and it was never snapshotted.** The trn2 ran
> inside a paid, non-refundable Capacity Block window and was gone when the
> window closed. Its results survived only in S3 and are now archived to
> Glacier Deep Archive — 51,280 objects, 54.55 GB — and committed to this repo
> under `trn2/results/`. That near-miss is written up in REPORT-EXTENSIONS §39.
>
> The `NeuronPipelinesTrainium2` stack still exists as a scale-to-zero ASG. It
> will launch and bill the moment sa-east-1 has trn2 capacity.

Companion to `trn1/docs/PROVISIONING.md`. What is different about the Trainium2
box, and what bit us getting there.

## The instance

| | value | source |
|---|---|---|
| Instance type | `trn2.3xlarge` | the only small Trainium2 SKU; sa-east-1 only |
| vCPU / host RAM | 12 / 128 GiB | `describe-instance-types` |
| Local NVMe | 470 GB, encryption required | `describe-instance-types` |
| Accelerator | 1× Trainium2 | |
| Physical NeuronCores | 8 × v3 | |
| Logical NeuronCores | **4** at LNC=2 (default) | Neuron LNC docs |
| HBM | **96 GiB** @ 2.9 TB/s → 24 GiB per logical core | Neuron trn2-arch + NKI arch guide |
| BF16 dense peak | 667 TFLOP/s per chip | Neuron trn2-arch |
| AMI | `ami-0b1b0d3aaa2171e1f` (Neuron PyTorch 2.9 DLAMI, Ubuntu 24.04) | SSM param, resolved at deploy |
| Price | **$2.235/hr** | derived: $53.64 per 24 h Capacity Block; no pricing-API record exists |

### The EC2 API reports the wrong HBM

`describe-instance-types` gives `TotalNeuronDeviceMemoryInMiB = 524288` — 512
GiB — for a single-chip instance. That contradicts the documented 96 GiB per
chip, and the documented figure is the consistent one: trn2.48xlarge advertises
1,536 GiB across 16 chips, which is 96 each. **Trust the docs over the API
here**, and capture `neuron-ls` in `specs.txt` (lane 0) to settle it on the box.
This is recorded as a correction in REPORT-EXTENSIONS.

## What is different from trn1

| | trn1.2xlarge | trn2.3xlarge |
|---|---|---|
| Swapfile | 64 GiB on `/scratch` | **none** — 128 GiB of host RAM makes it dead weight, and it would mask a real regression |
| `MALLOC_ARENA_MAX=64` | needed (32 GiB host OOM mitigation) | harmless, kept for uniformity |
| Adapter merge host RAM | tight, ~high-20s GiB peak on a 32 GiB box | comfortable |
| S3 NEFF prefix | `neuron-cache/` | **`neuron-cache-v3/`** — NEFFs are compiled per NeuronCore version and must never mix |
| LNC | n/a | `NEURON_LOGICAL_NC_CONFIG=2` |
| Region | us-west-2 | sa-east-1 (bucket stays in us-west-2) |

## Environment the box exports

`cdk/user_data/trn2.sh` appends to `/etc/profile.d/neuron-pipelines.sh`:

```bash
export NP_DEVICE=trn2
export NP_REGION=sa-east-1
export NP_CACHE_PREFIX=neuron-cache-v3
export NEURON_LOGICAL_NC_CONFIG=2
```

`NP_TELEMETRY_CORES` is deliberately **not** set here. The TP probe decides the
logical-core count at runtime (LNC=2 → 4, LNC=1 → 8) and exports it per lane;
pinning a guess in the profile would mislabel every telemetry CSV.

**SSM shells run as root with a bare environment and never source
`/etc/profile.d/`.** Every driver re-exports what it needs. If you run anything
by hand, export the block in runbook 12 first — the same omission misfiled a
gated-model 401 as a compiler failure in Phase 2.

## Gotchas

1. **`libneuronpjrt-path`.** torch-neuronx shells out to this venv-bin helper on
   init. Running `$NP_VENV/bin/python` without `$NP_VENV/bin` on PATH gives
   `FileNotFoundError`. Phase 2 lost seven lanes to this. `shared/run_all.sh`
   now exports it for all boxes, not just the extras drivers.

2. **LNC must match between compiler and runtime.** The docs are explicit:
   "AWS Neuron currently doesn't support setting the compiler flag to a
   different LNC configuration than the Neuron Runtime environment variable."
   `NCC_EARG001` is the error when the target arch rejects the requested lnc.
   The TP probe sets `NEURON_LOGICAL_NC_CONFIG` before anything in a rung
   compiles, for exactly this reason.

3. **Cold NEFF cache.** No v3 NEFFs existed anywhere at the start of Phase 3.
   The first 8B compile is paid in full. Per METHODOLOGY rule 3 that is a
   first-class result, not overhead to hide.

4. **Quota is not capacity, and on-demand never arrived.** All three sa-east-1
   AZs refused `trn2.3xlarge` on 2026-08-04 despite a granted 12-vCPU quota,
   through ~10 h of ODCR polling. The box was ultimately obtained with a
   purchased **Capacity Block** — see runbook 12. Consequences for this box:

   - It launches only via `-c trn2CapacityReservationId=...` plus an explicit
     `trn2Az`/`trn2SubnetId`; the stack refuses to synth otherwise, because a
     block lives in one AZ and a silent default would miss it.
   - The real price is **$2.235/hr** ($53.64 per 24 h block), a figure the AWS
     pricing API does not carry.
   - **The instance is terminated, not stopped, at the end of the window** —
     termination begins 11:00 UTC, 30 min before the 11:30 UTC end. `/scratch`
     *and the EBS root* both go, so unlike a normal stop the warm v3 NEFF cache
     does not survive. Push to `neuron-cache-v3/` before the deadline or the
     next block starts cold again.

## Extra packages

Same as trn1 — the DLAMI does not ship these, and the `[neuronx]` extra must be
avoided:

```bash
pip install optimum-neuron==0.4.3 trl==0.24.0 peft==0.17.0 datasets
```

Whether optimum-neuron 0.4.3's training path works on NeuronCore-v3 is
**unverified**: the `patch_optimum_modeling_checkpoint()` shim and the four-way
`from_pretrained` kwarg fallback in `sft_lora.py` were both tuned against v2. A
hard failure there is a legitimate reportable finding, and lane 2 (the TinyLlama
smoke / TP probe) is deliberately the cheapest place to discover it.
