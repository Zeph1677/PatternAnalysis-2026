# HipMRI Hard 3D Improved U Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a testable, evidence-safe COMP3710 Hard HipMRI 3D segmentation project and a matching one-page feasibility review.

**Architecture:** A NIfTI-aware data layer emits patient-disjoint 3D volumes and explicit audit manifests. Plain and improved 3D U-Nets share the training/evaluation protocol; the improved model adds residual blocks and deep supervision. Full-volume patch aggregation supports reproducible held-out evidence.

**Tech Stack:** Python 3.11, PyTorch, NumPy, NiBabel, Matplotlib, pytest, python-docx.

**Spec:** `recognition/hipmri_3d_improved_unet/docs/superpowers/specs/2026-10-15-hipmri-hard-3d-design.md`

## Global Constraints

- Use HipMRI NIfTI volumes from `/home/groups/comp3710/HipMRI_Study_open` only for the final experiment.
- The Hard model is a 3D Improved U-Net with residual units and deep supervision.
- Keep all patient partitions disjoint; retain one shared saved manifest for both models.
- Do not commit data, checkpoints, generated large files or credentials.
- Report no Dice, VRAM or clinical-error claims without corresponding saved real-run artifacts.

---

### Task 1: Project scaffold and NIfTI data contract

**Files:** Create `requirements.txt`, `dataset.py`, `audit.py`, `tests/test_dataset.py`, `tests/test_audit.py`.

**Interfaces:** `discover_volume_pairs(data_root) -> list[VolumePair]`; `build_patient_split_manifest(pairs, seed) -> dict`; `HipMRI3DDataset(records, data_root) -> Dataset`.

- [ ] Write synthetic-NIfTI tests for recursive paired discovery, shape/affine validation, integer labels and zero patient overlap.
- [ ] Run the tests and confirm they fail because 3D loader interfaces do not exist.
- [ ] Implement only the data contract required by those tests.
- [ ] Re-run focused tests and commit the data layer.

### Task 2: 3D baseline and Hard model

**Files:** Create `modules.py`, `tests/test_modules.py`.

**Interfaces:** `BaselineUNet3D(in_channels, num_classes, base_channels)` and `ImprovedUNet3D(...)`; training-mode improved output is `(final_logits, auxiliary_logits)` and evaluation-mode output is final logits.

- [ ] Write failing tests for spatial-shape preservation, finite gradients, residual model deep-supervision outputs and inference-only final logits.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement plain 3D U-Net, residual blocks and deep-supervision heads.
- [ ] Re-run model tests and commit the architectures.

### Task 3: Training, held-out evaluation and resource records

**Files:** Create `train.py`, `utils.py`, `tests/test_train.py`.

**Interfaces:** `composite_3d_loss`, `evaluate_3d_batches`, `train.main()`; output JSON contains class-wise metrics, parameter count, peak VRAM and elapsed time.

- [ ] Write failing tests for final-and-auxiliary loss, held-out metric fields and validation-only checkpoint selection.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement deterministic training and the resource-safe metrics writer.
- [ ] Re-run training tests and commit the command path.

### Task 4: Full-volume patch aggregation and failure panels

**Files:** Create `predict.py`, `tests/test_predict.py`.

**Interfaces:** `sliding_window_logits(model, volume, patch_size, stride)` returns full-volume logits; `render_case_views(...)` writes axial/coronal/sagittal panels.

- [ ] Write failing tests proving overlap averaging reconstructs constant logits and held-out selection respects the manifest.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement aggregation, metric ranking and multi-plane panel generation.
- [ ] Re-run prediction tests and commit inference evidence support.

### Task 5: Hard project documentation and feasibility review

**Files:** Create `README.md`, `build_feasibility_review.py`, `evidence/COMP3710_HipMRI_Hard_Feasibility_Review.docx`, `evidence/COMP3710_HipMRI_Hard_Feasibility_Review.pdf`.

- [ ] Draft source-backed Hard scope without fabricated experiment results.
- [ ] Build the one-page review from declared project facts and only verified local audit fields.
- [ ] Render every DOCX page to PNG and inspect page count, clipping, table wrapping and footer placement.
- [ ] Run `pytest -v`, CLI help, `git diff --check`, and commit documentation.
