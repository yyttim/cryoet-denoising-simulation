# Cryo-ET Denoising and Simulation

This project is an unsupervised 3D domain-translation framework for cryo-electron tomography (cryo-ET).

The training code uses unpaired samples. A clean/reference subvolume and a noisy/target subvolume in the same mini-batch do not need to describe the same physical region.

## Repository scope

This repository contains compact PyTorch training code for prepared 3D subvolumes. It includes:

- an unpaired two-domain HDF5 dataset loader;
- 3D U-Net translators for the two translation directions;
- spatial-domain discriminators;
- Fourier-domain discriminators;
- fixed Fourier masks for missing-wedge-aware training;
- cyclic reconstruction losses;
- training visualizations, loss logging, and checkpoint saving.

The repository is focused on model training. It does not include raw tilt-series processing, tomogram reconstruction, subvolume extraction from raw tomograms, full-volume inference scripts, downstream segmentation scripts, particle-picking evaluation, or subtomogram-averaging workflows.

## Repository layout

```text
.
├── .gitignore
├── train.py
├── requirements.txt
├── datasets/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── model.py
│   └── NEM.py
└── networks/
    ├── __init__.py
    ├── unet.py
    ├── discriminator.py
    ├── encoder.py
    ├── generators.py
    ├── denoiser.py
    └── blocks.py
```

## Data interface

The default loader expects two HDF5 files under `--dataroot`:

```text
data_noisy_missing.h5
data_noisy.h5
```

Both files must contain a dataset named:

```text
dataset_1
```

The filenames are fixed by the code. The active model treats them as two unpaired domains:

```text
data_noisy_missing.h5 -> Domain A
data_noisy.h5         -> Domain B
```

The active translation directions are:

```text
G2  : Domain A -> Domain B
dns : Domain B -> Domain A
```

For a denoising/simulation experiment, Domain A is typically the clean-like or reference structural domain, and Domain B is the noisy or target tomogram domain.

The training script is configured for one-channel `64 × 64 × 64` subvolumes.

## Quick start

Create output folders before training:

```bash
mkdir -p logs/imgs logs/models
```

Run training:

```bash
python train.py \
  --dataroot /path/to/data \
  --batch_size 8 \
  --num_workers 8 \
  --lr 1e-4
```

## Pretrained model

A pretrained checkpoint is available from [Google Drive](https://drive.google.com/drive/folders/1EiPmvJn89Yb_1129EjdCijTi-KK--fce?usp=drive_link). Download the checkpoint and replace `/path/to/checkpoint.pth` in the example below with its local path.

The checkpoint follows the standard format saved by `models/model.py::save_model()`. For denoising, load the `dns` state dictionary. For Domain-A-to-Domain-B translation, load `G2`.

Minimal patch-level denoising example:

```python
import torch
from networks.unet import UNet_generator

checkpoint = torch.load("/path/to/checkpoint.pth", map_location="cuda")

model = torch.nn.DataParallel(
    UNet_generator(in_channels=1, out_channels=1, init_features=32)
).cuda()

model.load_state_dict(checkpoint["dns"], strict=True)
model.eval()

# x should be a preprocessed and standardized 64 x 64 x 64 tile.
x = torch.as_tensor(x).float()[None, None].cuda()

with torch.no_grad():
    y = model(x).squeeze().cpu().numpy()
```

For full tomograms, apply the same preprocessing used during training, run the network on overlapping `64^3` tiles, blend the tile predictions, and reassemble the output volume.

## Outputs

```text
logs/imgs/       # training visualizations
logs/models/     # checkpoints
logs/loss.csv    # scalar loss log
```

After training, the two deployment networks are:

```text
dns -> Domain-B-to-Domain-A translation, typically denoising
G2  -> Domain-A-to-Domain-B translation, typically simulation or target-style transfer
```
