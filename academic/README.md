# academic — MNIST + CIFAR-10 on one NeuronCore (mlx-models parity)

Faithful torch-neuronx ports of the from-scratch training track in
[mlx-models](https://github.com/alpharomercoma/mlx-models): same
architectures (MLP / CNN / ViT), same hyperparameters, same seed, same
datasets — measured there on an Apple M5, measured here on **one Trainium1
NeuronCore**. Two consumer-priced chips, identical academic workloads.

## Setup

Runs inside the trn1 DLAMI training venv; no extra installs (torchvision
ships in it). First run downloads MNIST (~11 MB) and CIFAR-10 (~170 MB via
torchvision) to `/opt/np/models/academic-data` and caches them. First step
of each lane compiles the XLA graph — reported as `compile_s_estimate`,
never averaged into step times.

## Files

| file | what it does |
|---|---|
| `models.py` | MLP/CNN/ViT for both datasets, dimension-for-dimension MLX ports |
| `train_academic.py` | training loop: XLA-aware, fixed shapes, per-epoch acc, metrics JSON |
| `run_academic.sh` | 6-lane resumable driver, telemetry-wrapped, pushes results |

## Run

```bash
# on the trn1 box:
bash academic/run_academic.sh          # all 6 lanes, resumable
# single lane:
python3 academic/train_academic.py --dataset cifar --arch cnn \
  --out trn1/results/academic/cifar_cnn.json    # ~88-90% (M5 reference)
```

## Results (Trainium1, 1 NeuronCore, fp32, seed 0, defaults)

M5 reference ranges are from the mlx-models READMEs (same recipes).

| dataset | arch | test acc (M5) | test acc (trn1) | steady step | samples/s | compile |
|---|---|---|---|---|---|---|
| mnist | mlp | see mlx repo | _running_ | | | |
| mnist | cnn | ~99% | _running_ | | | |
| mnist | vit | see mlx repo | _running_ | | | |
| cifar | mlp | ~55–60% | _running_ | | | |
| cifar | cnn | ~88–90% | _running_ | | | |
| cifar | vit | ~78–82% (from scratch) | _running_ | | | |

> Accuracy is compared as RANGES: frameworks differ in default weight init
> and data-order RNG, so point-value equality across MLX and torch would be
> luck, not signal. Matching the range is the parity claim.

## Port notes (declared, not hidden)

- torch is NCHW, MLX is NHWC; flatten/reshape points mirrored exactly.
- MLX `MultiHeadAttention` defaults to bias-free qkv; torch adds biases
  (~0.1% of ViT params).
- MNIST normalization is `/255` only, CIFAR adds the standard per-channel
  mean/std — both exactly as in mlx-models `data.py`.
- Single NeuronCore on purpose: these models are far too small for tensor
  parallelism; per-core numbers are the honest unit (`NEURON_RT_NUM_CORES=1`).
