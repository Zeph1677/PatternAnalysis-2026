# HipMRI Hard 3D Improved U Net Design

## Scope and decision

This is the COMP3710 Hard segmentation route: HipMRI 3D prostate and organ-at-risk segmentation from downsampled NIfTI volumes. The assessed implementation is a 3D Improved U-Net with volumetric residual units and deep supervision. A plain 3D U-Net is the matched architectural baseline. A 2D baseline is optional diagnostic evidence for the 2D-versus-3D engineering question, not a substitute for the matched 3D baseline.

The existing OASIS 2D project is retained unchanged as historical Easy-tier work. It is not cited as evidence for this Hard project.

## Data contract

The loader discovers NIfTI files recursively below a supplied HipMRI root, pairing `semantic_MRs` volumes with corresponding volumes in `semantic_labels_only`. It derives a stable patient identifier from the matched relative path, reads a three-dimensional array (or removes one singleton/channel dimension from a four-dimensional NIfTI), and rejects missing pairs, non-finite images, shape mismatches, non-integral labels, and labels outside the observed/declared class set.

The audit records paths relative to the selected data root, patient ID, image/mask shapes, spacing, orientation codes, affine equality, labels and split allocation. The split is deterministic and patient-disjoint. No training sample, patch, model selection step or final test metric may cross patient partitions.

## Modelling and training

`BaselineUNet3D` uses 3D convolution, 3D pooling, 3D transposed convolution and direct skips. `ImprovedUNet3D` replaces convolution blocks with volumetric residual blocks and emits a final segmentation head plus decoder-side deep-supervision heads during training. The final-resolution logits are always the inference output. Cross-entropy plus soft Dice is computed at full resolution; auxiliary losses are resized to their target scale and weighted explicitly.

Training selects a checkpoint only by validation macro Dice and evaluates the selected checkpoint once on held-out test patients. Both models use one saved manifest, matching preprocessing, deterministic seed and declared patch/full-volume geometry. Results include per-class Dice/IoU, parameter count, peak GPU VRAM, training time and per-patient inference time.

## Full-volume inference and failure evidence

Inference must reconstruct a full patient volume from overlapping 3D patches by accumulating logits and normalising by the number of predictions per voxel. It writes a machine-readable metric record and figure panels for axial, coronal and sagittal views. Failure selection ranks held-out patients by a declared error statistic. The final report must inspect 3-5 actual cases, including base, apex, small target or boundary leakage when they occur, and must distinguish false positives, false negatives, discontinuity and boundary displacement.

## Feasibility review and completion boundary

The one-page review names HipMRI 3D, the Hard tier, plain 3D U-Net baseline, 3D Improved U-Net, patient-level split, true volumetric inference, resource budget and a fallback. It must not claim metrics before a real Rangpur run. The consultation, pull request, Turnitin receipt and final GPU results remain external completion conditions.
