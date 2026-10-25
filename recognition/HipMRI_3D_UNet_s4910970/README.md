# HipMRI 3D U-Net: prostate contour drafting

This directory contains my COMP3710 Pattern Recognition and Analysis project.
It will implement and evaluate 3D U-Net models for multi-class semantic
segmentation of downsampled pelvic MRI volumes. The project treats the
prostate as the primary reporting structure while retaining the associated
pelvic labels needed for multi-class learning.

The project currently implements data preparation, models, training and prediction. It makes no claim
about training completion, Dice performance, clinical suitability, or final
assessment status.

## Dataset

The project uses the CSIRO collection *Labelled weekly MR images of the male
pelvis* (v2, DOI: [10.25919/45T8-P065](https://doi.org/10.25919/45T8-P065)).
The local release contains paired MRI volumes and semantic masks. Its supplied
label description identifies the following integer labels:

| Value | Structure |
| ---: | --- |
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

Raw course data stay under `data/` and are excluded from version control.
The repository must never include patient volumes, trained weights, prediction
tensors, or generated experiment outputs.

## Planned model comparison

The baseline is a volumetric 3D U-Net. A separate Improved model will be
evaluated against that baseline using the same dataset protocol. Later commits
will document the implemented architecture, preprocessing, patient-level data
split, loss function, checkpoint-selection rule, and held-out evaluation.

The word “Improved” will describe an implemented comparison model, not a
guarantee that it outperforms the baseline.

## Initial file layout

| File | Intended responsibility |
| --- | --- |
| `dataset.py` | Discover paired NIfTI images and masks, validate them, and expose datasets. |
| `modules.py` | Define reusable neural-network modules and 3D U-Net architectures. |
| `train.py` | Train and validate a declared model configuration. |
| `predict.py` | Restore a selected checkpoint and generate prediction outputs and metrics. |
| `utils.py` | Plot saved training histories and validation-selected epochs. |

`dataset.py` now implements NIfTI pairing, geometry validation, deterministic
patient-level train/validation/test splits, z-score normalization, optional
3D resizing and foreground-aware patch sampling. MRI resizing uses trilinear
interpolation; masks use nearest-neighbour interpolation. Multiple weekly scans
from the same patient remain in the same partition.

`modules.py` implements the baseline and Improved architectures.
`train.py` implements training and validation. `predict.py` restores a selected
checkpoint and evaluates reconstructed volumes from the test partition.

## Model architecture

Both models use three encoder levels, max-pooling, a bottleneck and three
decoder levels with transposed convolutions and concatenated encoder skips.
Convolution blocks use two 3D convolutions, instance normalization and
LeakyReLU. `BaselineUNet3D` uses these plain blocks; `ImprovedUNet3D` adds
residual blocks, with a 1x1x1 projection when channel counts differ.

The Improved model has two auxiliary decoder heads for training-time deep
supervision. In training mode it returns final logits and the two lower-resolution
auxiliary outputs. In evaluation mode both models return only final logits.
The training loss combines cross-entropy and soft Dice, adding each auxiliary
loss with weight 0.25 after nearest-neighbour resizing of its mask target.

Inputs have shape `(N, 1, D, H, W)`, with spatial dimensions divisible by eight.
Use `num_classes=6` for the declared pelvic label set; the constructor default
is two classes. Final logits have shape `(N, 6, D, H, W)` for that configuration.
Architecture implementation does not establish a performance advantage.

```python
import torch
from modules import BaselineUNet3D, ImprovedUNet3D

model = ImprovedUNet3D(in_channels=1, num_classes=6, base_channels=16)
model.eval()
with torch.no_grad():
    logits = model(torch.zeros(1, 1, 16, 16, 16))
```

## Usage

Run these commands from this project directory:

```powershell
python -m pip install torch numpy nibabel matplotlib
```

The data root must contain `semantic_MRs/` and `semantic_labels_only/`.
`discover_volume_pairs(data_root)` validates matched volumes and returns their
relative paths and metadata. `build_patient_split_manifest(pairs, seed=42)`
creates patient-disjoint partitions. `HipMRI3DDataset` loads a selected partition
as image and mask tensors for the later training pipeline.


## Training

Training consumes a fixed JSON manifest with `summary.labels` equal to
`[0, 1, 2, 3, 4, 5]`, `splits` containing patient lists for `train`, `validation`
and `test`, and a `cases` list. Each case records `case_id`, `patient_id`,
relative `image_path` and `mask_path`, `shape`, `spacing`, `orientation`,
`labels` and its `split`. The following example creates this input from separately obtained data:

```python
import json
from dataclasses import asdict
from pathlib import Path
from dataset import discover_volume_pairs, build_patient_split_manifest

pairs = discover_volume_pairs(Path('data/HipMRI_Study_open'))
splits = build_patient_split_manifest(pairs, seed=42)
partition = {patient: split for split in ('train', 'validation', 'test')
             for patient in splits[split]}
cases = []
for pair in pairs:
    record = asdict(pair)
    record['image_path'] = pair.image_path.as_posix()
    record['mask_path'] = pair.mask_path.as_posix()
    record['split'] = partition[pair.patient_id]
    cases.append(record)
manifest = {
    'summary': {'labels': sorted({v for p in pairs for v in p.labels})},
    'splits': splits,
    'cases': cases,
}
output = Path('artifacts/dataset_manifest.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
```

```powershell
python train.py --manifest artifacts/dataset_manifest.json --data-root data/HipMRI_Study_open --output-dir artifacts/baseline_run --model baseline --epochs 20 --prostate-label 5 --prostate-label-source "CSIRO v2 DOI 10.25919/45T8-P065, Labels: 5=Prostate" --selection-metric prostate_dice
```

Use `--model improved` with a separate empty output directory for the comparison
arm. Preserve the same manifest and all shared settings. The training process
uses only train and validation partitions, saves a validation-selected `best.pt`,
configuration, history, resource summary and training curves under the ignored
output directory. Prostate-Dice selection uses patient-level validation scores;
held-out test evaluation belongs to the later prediction milestone.


## References

1. J. Dowling and P. Greer, *Labelled weekly MR images of the male pelvis*,
   CSIRO Data Collection, v2, DOI: 10.25919/45T8-P065.
2. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
   “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
   MICCAI, 2016. https://arxiv.org/abs/1606.06650
3. P. Isensee et al., “Brain Tumor Segmentation and Radiomics Survival
   Prediction: Contribution to the BRATS 2017 Challenge,” 2018.
   https://arxiv.org/abs/1802.10508
