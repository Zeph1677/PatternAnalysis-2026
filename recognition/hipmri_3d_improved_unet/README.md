# HipMRI 3D Improved U Net for Prostate Segmentation

## Scope

This is the COMP3710 Hard segmentation project. It uses downsampled HipMRI 3D NIfTI volumes from `/home/groups/comp3710/HipMRI_Study_open` to test whether volumetric residual modelling and deep supervision reduce clinically important prostate boundary and cross-slice errors enough to justify their compute cost.

The implementation compares a plain 3D U-Net with a 3D Improved U-Net. The improved model contains volumetric residual units and two decoder deep-supervision heads. This is distinct from the repository's OASIS 2D work and must not reuse its reported values.

## Data and validity controls

`dataset.py` pairs `semantic_MRs` and `semantic_labels_only` by their relative NIfTI paths. It validates 3D shape, affine agreement, finite intensity values and integral labels. Splits are by patient, not slice or patch, and the exact seed-42 manifest must be reused by both models.

No HipMRI data, checkpoints or final results are committed. The code has been checked only against synthetic NIfTI volumes locally; real performance, class definitions, memory use and image orientation must be audited on Rangpur before reporting results.

## Feasibility review

**Need and scope.** The intended user is a radiotherapy contouring specialist. The prototype drafts a 3D segmentation for review; it is not an autonomous clinical decision system.

**Acceptance criteria.** Zero patient overlap; identical test patients for both models; held-out prostate Dice at least 0.70; per-class Dice/IoU, peak VRAM and per-case inference time; 3-5 real base, apex, small-target or boundary-leakage failure cases.

**Preliminary evidence.** Local synthetic-NIfTI tests validate recursive pairing, missing-pair rejection, shape/affine checks, patient splitting, 3D forward/backward passes, deep supervision and held-out metric fields. This is engineering evidence, not a trained-model result.

**Risks, budget and fallback.** 3D memory demand, small foreground regions, NIfTI orientation, slow I/O and failed cluster jobs require a short GPU smoke run before full Slurm training. If the improved model does not improve real held-out boundary errors at acceptable cost, report the plain 3D U-Net as the recommended fallback.

## Current verification

```powershell
cd E:\vscode\PatternAnalysis-2026\recognition\hipmri_3d_improved_unet
python -m pip install -r requirements.txt
python -m pytest -v
```

Before a final claim, run the data audit and full baseline/improved training on Rangpur, retain configuration/log/checkpoint/metric artifacts, and inspect 3-5 held-out cases. The tutor check-off, PR, Turnitin receipt and teaching-team review are separate external requirements.
