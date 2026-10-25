# HipMRI 3D U-Net: prostate contour drafting

This directory contains my COMP3710 Pattern Recognition and Analysis project.
It will implement and evaluate 3D U-Net models for multi-class semantic
segmentation of downsampled pelvic MRI volumes. The project treats the
prostate as the primary reporting structure while retaining the associated
pelvic labels needed for multi-class learning.

The project currently implements the data preparation layer. It makes no claim
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

The baseline will be a volumetric 3D U-Net. A separate Improved model will be
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

`dataset.py` now implements NIfTI pairing, geometry validation, deterministic
patient-level train/validation/test splits, z-score normalization, optional
3D resizing and foreground-aware patch sampling. MRI resizing uses trilinear
interpolation; masks use nearest-neighbour interpolation. Multiple weekly scans
from the same patient remain in the same partition.

`modules.py`, `train.py` and `predict.py` remain placeholders for later milestones.

## Usage

Run these commands from this project directory:

```powershell
python -m pip install torch numpy nibabel
```

The data root must contain `semantic_MRs/` and `semantic_labels_only/`.
`discover_volume_pairs(data_root)` validates matched volumes and returns their
relative paths and metadata. `build_patient_split_manifest(pairs, seed=42)`
creates patient-disjoint partitions. `HipMRI3DDataset` loads a selected partition
as image and mask tensors for the later training pipeline.

The data layer is migrated from private development commit `4392d92` of
`COMP3710-A1`. These commits record the staged migration of the project.

## References

1. J. Dowling and P. Greer, *Labelled weekly MR images of the male pelvis*,
   CSIRO Data Collection, v2, DOI: 10.25919/45T8-P065.
2. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
   “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
   MICCAI, 2016. https://arxiv.org/abs/1606.06650
3. P. Isensee et al., “Brain Tumor Segmentation and Radiomics Survival
   Prediction: Contribution to the BRATS 2017 Challenge,” 2018.
   https://arxiv.org/abs/1802.10508
