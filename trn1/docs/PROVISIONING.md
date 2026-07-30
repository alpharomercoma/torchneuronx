# trn1.2xlarge — provisioning

Deployed by `cdk deploy NeuronPipelinesTrainium` (see [cdk/README.md](../../cdk/README.md)).
This doc is the bring-up verification: what a healthy box looks like, and the
failure modes already paid for once in a previous project so they never cost
an afternoon again.

## The one that will cost you an afternoon

If `neuron-ls` prints `no neuron device found` on a *non-DLAMI* image, the
kernel driver is missing — the Neuron runtime lives in userspace but needs
`aws-neuronx-dkms` loaded. The Neuron DLAMI this stack pins ships it, so on
this box the check should just pass; if it ever doesn't:

```bash
lsmod | grep neuron               # expect: neuron  <size>  0
ls /dev/neuron*                   # expect: /dev/neuron0
# only if both are empty (wrong AMI?):
sudo apt-get install -y linux-headers-$(uname -r) aws-neuronx-dkms=2.*
```

## Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `neuron-ls`: no neuron device found | `aws-neuronx-dkms` not loaded (non-DLAMI image) | install dkms + headers, re-check `/dev/neuron0` |
| `ImportError: WRONG PACKAGE ... pip.repos.neuron.amazonaws.com` | placeholder `torch-neuronx` from PyPI shadowing the real one | never `pip install torch-neuronx` outside the DLAMI venv; the venv already has the real one |
| `torch.accelerator.is_available()` is False | that API cannot see XLA backends | use `torch_xla.device()`; device type is `xla` |
| loss never changes / graph "runs" instantly | XLA laziness: nothing executes until sync | trainer handles it; in hand-rolled loops call `torch_xla.sync()` per step |
| second run recompiles everything | default cache is `/var/tmp/neuron-compile-cache` (lost on replace) | `NEURON_COMPILE_CACHE_URL=/opt/np/cache/...` is exported by `/etc/profile.d/neuron-pipelines.sh`; verify it before any lane |
| lane hangs, `neuron-ls` shows cores busy | dead process still holds NeuronCores | `pkill -f sft_lora; sleep 5; neuron-ls` until cores free |
| host OOM-killer during 8B load/merge | 32 GiB host RAM | 64 GiB swapfile on NVMe `/scratch` (user-data asserts it each boot); check `free -h` before lanes 4–6 |
| precompile "loss" looks plausible | `neuron_parallel_compile` runs garbage numerics by design | METHODOLOGY rule 4: never record it |
| `NeuronSFTTrainer` missing | DLAMI training venv ships torch-neuronx but NOT optimum-neuron/peft/trl | `pip install optimum-neuron==0.4.3 trl==0.24.0 peft==0.17.0 datasets` into the venv (no `[neuronx]` extra — see below) |
| `ImportError: clone_chat_template from trl.models` at trainer import | unpinned `pip install trl` grabs trl 1.x; optimum-neuron 0.4.3 needs its declared training pins | exactly `trl==0.24.0 peft==0.17.0` (read them from the wheel's METADATA `extra == "training"` lines, not from memory) |
| pip backtracks to ancient optimum-neuron, numpy 1.25 source-build explodes on py3.12 | `optimum-neuron[neuronx]` extra pins Neuron packages that conflict with the DLAMI's newer stack | install **without** the extra; the DLAMI already provides torch-neuronx/neuronx-cc |
| `neuronx-cc requires numpy>=2` vs `optimum-neuron requires numpy<=1.26` | genuine pin conflict in 0.4.3 | keep **numpy>=2** (the compiler wins); optimum-neuron 0.4.3 imports and runs fine under numpy 2.5, pip's warning notwithstanding — measured, not assumed |
| `FileNotFoundError: 'libneuronpjrt-path'` at import | XLA runtime shells out to a venv-bin helper; bare `$VENV/bin/python` misses it | `export PATH=$VENV/bin:$PATH` (or activate the venv) before anything imports torch-neuronx |

## Verified state

Fill during first bring-up (runbook 04); every later session diffs against it.

Captured 2026-07-29/30 on i-0cb9e758143a745d5 (verbatim from bring-up):

```
$ /opt/aws/neuron/bin/neuron-ls
| 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 0-7      | -1   |
$ echo $NEURON_COMPILE_CACHE_URL
/opt/np/cache/neuron-compile-cache        # grew to 526 MB across the study
$ free -h | grep -i swap
Swap:           63Gi          0B        63Gi   # /scratch/swapfile (NVMe)
$ df -h / /scratch
/dev/root       484G  ...  /            # 500G gp3 EBS
/dev/nvme1n1    434G  ...  /scratch     # instance store
$ ls -d /opt/aws_neuronx_venv*
/opt/aws_neuronx_venv_pytorch_2_9  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
$ pip list | grep -E "torch-neuronx|optimum-neuron|neuronx-cc|trl|peft"
neuronx-cc 2.26.6360.0  torch-neuronx 2.9.0.2.15  torch 2.9.1
optimum-neuron 0.4.3  trl 0.24.0  peft 0.17.0  (installed per Gotchas)
```
