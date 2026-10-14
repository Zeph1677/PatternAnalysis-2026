# OASIS Improved U-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible, subject-disjoint OASIS 2D segmentation prototype with a fair baseline comparison and an evidence-safe assessed README.

**Architecture:** A pair-indexing data layer prevents subject leakage; PyTorch model definitions remain independent from file I/O; training and inference consume the same explicit split manifest. Results are generated into ignored artifact directories and the README only reports values produced there.

**Tech Stack:** Python 3.10+, PyTorch, Pillow, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-oasis-improved-unet-design.md`

## Global Constraints

- Use `/home/groups/comp3710/OASIS` only as a documented Rangpur default; never commit data or weights.
- Use deterministic subject-level train/validation/test splits.
- Evaluate baseline and ImprovedUNet on the same test manifest.
- Do not state unexecuted metrics as results.
- Keep all modules CPU-testable on synthetic tensors.

---

### Task 1: Project safety and metrics

**Files:** Create `.gitignore`, `requirements.txt`, `utils.py`, `tests/test_utils.py`.

- [ ] Write tests proving a perfect hard prediction has Dice/IoU 1 and an absent class is excluded from a macro average.
- [ ] Run `pytest tests/test_utils.py -v` and confirm the expected import failure.
- [ ] Implement deterministic seeding, class-wise hard Dice/IoU, and JSON-safe conversion.
- [ ] Re-run the focused tests and commit `feat: add reproducible segmentation metrics`.

### Task 2: Leakage-free data layer

**Files:** Create `dataset.py`, `tests/test_dataset.py`.

- [ ] Write tests with temporary PNG pairs proving deterministic subject splits and rejecting a subject shared by two requested partitions.
- [ ] Run the focused test and confirm the expected module failure.
- [ ] Implement pairing, subject extraction, manifest generation/loading, normalisation and `SegmentationDataset`.
- [ ] Re-run tests and commit `feat: add subject-disjoint OASIS dataset pipeline`.

### Task 3: Baseline and improved models

**Files:** Create `modules.py`, `tests/test_modules.py`.

- [ ] Write tensor-shape tests for `BaselineUNet` and `ImprovedUNet`.
- [ ] Run tests and confirm missing-class failure.
- [ ] Implement convolutional blocks, residual blocks, attention gates, both models, and deep-supervision-compatible logits.
- [ ] Re-run tests and commit `feat: add baseline and improved unet models`.

### Task 4: Training and evaluation entry point

**Files:** Create `train.py`, `tests/test_train.py`.

- [ ] Write tests for composite Dice/cross-entropy loss and a one-batch evaluator result schema.
- [ ] Run tests and confirm the expected import failure.
- [ ] Implement CLI configuration, fair model selection, epoch loop, test evaluation, timing/VRAM profile, checkpointing, result JSON/CSV and learning curves.
- [ ] Re-run tests and commit `feat: add reproducible training and evaluation entry point`.

### Task 5: Inference and assessed documentation

**Files:** Create `predict.py`, `tests/test_predict.py`; modify `README.md`.

- [ ] Write a test for categorical error-overlay construction.
- [ ] Run it and confirm missing-function failure.
- [ ] Implement checkpoint inference, panels and failure ranking; write installation, feasibility review, report template, AI disclosure and exact run commands in README.
- [ ] Run all tests, syntax checks and CLI help, then commit `docs: add assessed project report and inference workflow`.
