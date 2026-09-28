# OASIS 2D Improved U-Net Design

## Scope

This project implements the PDF's Easy-difficulty OASIS Brain 2D segmentation option. It compares a small 2D U-Net baseline with an improved residual, multi-scale 2D U-Net on the same subject-disjoint test set. It does not claim a final metric until a reproducible training command has produced the tracked, small result files.

## Data and leakage control

`dataset.py` discovers image/mask pairs, extracts a subject identifier from a configurable regular expression, and splits unique subjects deterministically into train, validation, and test partitions. A subject cannot appear in more than one partition. Images are converted to one channel, normalized per image, and masks are validated as integer class maps before one-hot conversion in the loss.

## Components

`modules.py` provides a plain U-Net baseline and an ImprovedUNet using residual encoder blocks, bottleneck dropout, attention gates, and deep-supervision heads. `train.py` owns configuration, optimization, checkpointing, metrics, plots and CSV/JSON evidence. `predict.py` loads a checkpoint and writes per-case PNG panels plus a failure ranking. `utils.py` owns class-wise Dice/IoU, reproducibility and atomic JSON/CSV-friendly serialisation.

## Evidence and safety

All model comparison values are computed over the same held-out test subjects. The report distinguishes executed evidence from planned experiments, records package versions and hardware, and requires three to five case-level error panels before a deployment recommendation. Datasets, checkpoints and generated large artifacts are ignored by Git.

## Acceptance criteria

1. The pair index contains no duplicate slice or cross-split subject.
2. The baseline and improved model accept `(N, 1, H, W)` and emit `(N, 4, H, W)` logits.
3. Held-out evaluation exports class-wise Dice and IoU, mean values, inference time and peak VRAM.
4. Training is seedable, resumes only from an explicit checkpoint, and writes plots/checkpoints outside Git.
5. Inference writes input, ground-truth, prediction, and false-positive/false-negative overlays for the ranked failure cases.
