# inf2.xlarge — provisioning

Deployed by `cdk deploy NeuronPipelinesInferentia`. Same Neuron driver story
as [trn1](../../trn1/docs/PROVISIONING.md) — the Gotchas table there applies
here too. This doc adds only what is different on the inference box.

## What is different here

- **DLAMI**: the *vLLM inference* Neuron DLAMI (server stack preinstalled),
  not the PyTorch training one. Lane 0 records the exact vLLM/Neuron versions.
- **Host is small on purpose**: 4 vCPU / 16 GiB RAM. Nothing heavyweight gets
  installed client-side; the bench client is stdlib-only
  (`shared/serve/fallback_client.py`). If host RAM ever becomes the binding
  constraint, `-c inf2InstanceType=inf2.2xlarge` is a one-flag redeploy — but
  that is a *decision to record*, not a silent upgrade.
- **No instance store** on inf2.xlarge: no `/scratch`, no swapfile. The 8B
  merge never runs here (16 GiB host); the merged model arrives via S3.
- **Compile cache is the demo**: cold boot of an 8B server config costs tens
  of minutes of neuronx-cc; warm boot from the EBS cache should be minutes.
  Both numbers are recorded in `serve/*/boot.json` — treat a surprise
  recompile as a bug (`echo $NEURON_COMPILE_CACHE_URL` first).

## Gotchas (inference-specific)

| Symptom | Cause | Fix |
|---|---|---|
| every boot dies: `Engine core initialization failed` → worker traceback ends `ValueError: Unsupported instance family: inf2. Supported: trn2, trn3pd, trn3pds` | vllm-neuron 0.21 unconditionally maps EFA NICs; inf2 has no EFA and the mapper only knows Trn2/Trn3 | `export NEURON_SKIP_EFA_AFFINITY=1` (plugin's own flag; launch_vllm.sh sets it). The real error is in the WORKER lines of boot.server.log — the APIServer's "Failed core proc(s): {}" line is a red herring |
| past the EFA skip, compile dies: `nki/isa/_copy.py: tensor_copy does not support engine.scalar on NeuronCore-v2 (Trn1/Inf2)` | **the vLLM 0.21 DLAMI's kernels target NeuronCore-v3 (Trn2+) and assert on Inf2 silicon** — this DLAMI generation cannot serve inf2 at all | deploy the previous line: `cdk deploy NeuronPipelinesInferentia -c inf2AmiId=ami-035c945d557065665` (vLLM 0.16, Ubuntu 24.04, 20260715 — last generation before the Trn2-only switch on 20260721). Measured 2026-07-30 |
| server boots then first request stalls minutes | warmup/compile on first shapes | the warm lane exists for this; never benchmark the first request |
| boot OOMs at KV allocation | max_num_seqs × max_model_len over HBM budget | use the declared configs only (short: 2048×32, long: 10240×8) — METHODOLOGY rule 6 |
| client-side ceiling at concurrency 32 | 4 vCPU host tokenization | cpu lane bounds this; report it as host limit, not chip limit |
| Qwen3 fails to boot | backend support gap | that lane is attempt-only; `load_failure.json` IS the result |

## Verified state

```
TODO-VERIFY: paste after first boot
$ neuron-ls                        # 1 device, 2 NeuronCores, 32 GB
$ echo $NEURON_COMPILE_CACHE_URL   # /opt/np/cache/neuron-compile-cache
$ free -h                          # ~15Gi total, no swap
$ ls /opt/aws_neuronx_venv*        # vLLM inference venv present
$ /opt/aws_neuronx_venv*/bin/pip list | grep -E "vllm|neuronx"
```
