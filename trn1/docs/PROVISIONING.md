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

## Verified state

Fill during first bring-up (runbook 04); every later session diffs against it.

```
TODO-VERIFY: paste after first boot
$ neuron-ls                        # 1 device, 2 NeuronCores, 32 GB
$ echo $NEURON_COMPILE_CACHE_URL   # /opt/np/cache/neuron-compile-cache
$ free -h | grep -i swap           # 64Gi swap on /scratch/swapfile
$ df -h / /scratch                 # 500G EBS root, ~440G usable NVMe
$ ls /opt/aws_neuronx_venv*        # pytorch training venv present
$ /opt/aws_neuronx_venv*/bin/pip list | grep -E "torch-neuronx|optimum-neuron|neuronx-cc"
```
