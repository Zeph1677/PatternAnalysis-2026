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

## File layout

| File | Intended responsibility |
| --- | --- |
| `audit.py` | Validate paired volumes and write a deterministic patient-level manifest. |
| `dataset.py` | Discover paired NIfTI images and masks, validate them, and expose datasets. |
| `modules.py` | Define reusable neural-network modules and 3D U-Net architectures. |
| `train.py` | Train and validate a declared model configuration. |
| `predict.py` | Restore a selected checkpoint and generate prediction outputs and metrics. |
| `plot_training.py` | Plot saved training histories, validation-selected epochs, and recorded learning rates. |

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
Create the audited, patient-disjoint manifest before training:

```powershell
python audit.py data/HipMRI_Study_open --output artifacts/dataset_manifest.json --seed 42
```

The audit verifies image-mask pairing and spatial metadata, records each case's
split, and keeps all scans from one patient in the same partition. Both model
arms must use the same unchanged manifest.


## Training

Training consumes the fixed JSON manifest produced by `audit.py`, with `summary.labels` equal to
`[0, 1, 2, 3, 4, 5]`, `splits` containing patient lists for `train`, `validation`
and `test`, and a `cases` list. Each case records `case_id`, `patient_id`,
relative `image_path` and `mask_path`, `shape`, `spacing`, `orientation`,
`labels` and its `split`.

```powershell
python train.py --manifest artifacts/dataset_manifest.json --data-root data/HipMRI_Study_open --output-dir artifacts/baseline_run --model baseline --epochs 20 --prostate-label 5 --prostate-label-source "CSIRO v2 DOI 10.25919/45T8-P065, Labels: 5=Prostate" --selection-metric prostate_dice
```

Use `--model improved --architecture-version improved_v1` with a separate empty
output directory for the comparison arm. Preserve the same manifest and all shared settings. The training process
uses only train and validation partitions, saves a validation-selected `best.pt`,
configuration, history, resource summary and training curves under the ignored
output directory. Prostate-Dice selection uses patient-level validation scores;
held-out test evaluation is performed separately by `predict.py`.

## Prediction and evaluation

After training has selected a checkpoint using validation data, run:

```powershell
python predict.py --checkpoint artifacts/baseline_run/best.pt --manifest artifacts/dataset_manifest.json --data-root data/HipMRI_Study_open --output-dir artifacts/baseline_test --patch-size 128 128 128 --stride 64 64 64
```

The checkpoint strictly restores its declared architecture version, model type,
channel width, class count and target geometry. The supplied manifest must have the same SHA-256 as the training
manifest. Keep the manifest bytes unchanged between training and prediction.
Prediction reads only its test patients, averages overlapping patch logits,
then applies argmax after full-volume reconstruction. Patch dimensions must
fit within the target volume and be divisible by eight; stride must be positive
and no greater than the corresponding patch size.

Each patient's score is the mean prostate Dice across that patient's scans.
The headline metric is the arithmetic mean across patients. Supplementary Dice
and IoU are averaged across scans for each non-background class. The output
directory contains `test_metrics.json`, prediction tensors in `predictions/`
and three-plane images in `failure_panels/`. For each patient's lowest-Dice
scan, the axial, coronal and sagittal views independently select the slice with
the most prostate false-positive plus false-negative voxels. The selected case,
indices and fallback rule are recorded in `failure_panel_selection`; these are
error-focused views, not anatomically verified prostate base or apex slices.

Use a separate empty output directory for every run. Training and prediction
refuse to overwrite a non-empty output directory. Evaluate held-out data after
freezing the model and shared settings; use validation data for configuration
selection. Neither command supplies the raw data or pretrained weights.

## Local execution check

Both architectures completed a one-epoch CPU smoke run on temporary synthetic
six-label NIfTI volumes with target geometry `16x16x16` and base channels 2.
The check exercised data preparation, training, validation, checkpoint saving
and loading, full-volume prediction, metric JSON and PNG output. It used this
directory's modules and commands without private-repository imports. Temporary
data and outputs were removed after verification; no test scripts are included
in the submission directory.

For a small execution check on separately obtained data, use `--epochs 1
--base-channels 2 --target-shape 16 16 16 --device cpu` for training and
`--patch-size 16 16 16 --stride 8 8 8 --device cpu` for prediction. These settings
check execution only and are unsuitable for reporting segmentation quality.
Real-data performance, GPU resource use and the final experiment settings
remain to be documented in subsequent milestones.


## References

1. J. Dowling and P. Greer, *Labelled weekly MR images of the male pelvis*,
   CSIRO Data Collection, v2, DOI: 10.25919/45T8-P065.
2. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
   “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
   MICCAI, 2016. https://arxiv.org/abs/1606.06650
3. P. Isensee et al., “Brain Tumor Segmentation and Radiomics Survival
   Prediction: Contribution to the BRATS 2017 Challenge,” 2018.
   https://arxiv.org/abs/1802.10508
