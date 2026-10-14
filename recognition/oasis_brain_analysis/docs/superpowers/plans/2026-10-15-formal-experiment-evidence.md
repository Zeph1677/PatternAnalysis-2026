# OASIS Formal Experiment Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OASIS baseline-versus-improved comparison reproducible, resource-profiled, Rangpur-runnable, and evidence-safe.

**Architecture:** `dataset.py` validates a reusable manifest; `train.py` writes complete evidence; `predict.py` reads the same manifest. One Slurm script runs both models and the README reports only retrieved evidence.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pillow, Matplotlib, pytest, Slurm/bash.

**Spec:** `docs/superpowers/specs/2026-10-15-formal-experiment-evidence-design.md`

## Global Constraints

- Keep OASIS images, masks, checkpoints, and large generated artifacts outside Git.
- Use one subject-disjoint shared manifest and seed 42 unless a recorded configuration states otherwise.
- Select checkpoints by validation macro Dice, then evaluate each selected checkpoint once on held-out test records.
- Do not replace README pending fields with unverified values.
- Preserve `docs/superpowers/plans/COMP3710_A1_Assignment_Execution_Guide.md` unchanged.
- Use `/home/groups/comp3710/OASIS` as Rangpur default only; permit explicit alternate data roots.

---

### Task 1: Validate and reuse a subject manifest

**Files:** Modify `dataset.py` and `tests/test_dataset.py`.

**Interfaces:** `validate_subject_split_manifest(manifest: dict[str, list[dict[str, str]]], data_root: str | Path) -> dict[str, list[dict[str, str]]]` consumes records with `image`, `mask`, `subject_id`, and `case_id`.

- [ ] **Step 1: Write failing tests**

```python
def test_manifest_validator_rejects_subject_in_two_partitions(tmp_path: Path):
    manifest = {"train": [_record(tmp_path, "001")], "validation": [_record(tmp_path, "001")], "test": [_record(tmp_path, "002")]}
    with pytest.raises(ValueError, match="multiple partitions"):
        validate_subject_split_manifest(manifest, tmp_path)

def test_manifest_validator_returns_valid_records(tmp_path: Path):
    manifest = _three_partition_manifest(tmp_path)
    assert validate_subject_split_manifest(manifest, tmp_path) == manifest
```

- [ ] **Step 2: Run:** `python -m pytest tests/test_dataset.py -k manifest_validator -v`  
Expected: an import failure for the undefined validator.

- [ ] **Step 3: Implement** a validator that requires non-empty `train`, `validation`, and `test`; requires all four record keys; rejects a subject appearing in two partitions; and rejects image/mask paths that are missing or outside the resolved data root. Correct the OASIS display-label comment without changing mapping behavior.

- [ ] **Step 4: Run:** `python -m pytest tests/test_dataset.py -v; python -m pytest -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add recognition/oasis_brain_analysis/dataset.py recognition/oasis_brain_analysis/tests/test_dataset.py
git commit -m "feat: validate shared OASIS split manifests"
```

### Task 2: Persist complete run evidence

**Files:** Modify `utils.py`, `train.py`, and `tests/test_train.py`.

**Interfaces:** `collect_environment(device: torch.device) -> dict[str, object]`; `load_or_create_manifest(data_root: str | Path, seed: int, manifest_path: str | Path | None) -> dict[str, list[dict[str, str]]]`; `write_learning_curves(history: list[dict], path: str | Path) -> None`.

- [ ] **Step 1: Write failing tests**

```python
def test_environment_record_identifies_python_and_torch():
    record = collect_environment(torch.device("cpu"))
    assert {"python_version", "torch_version", "device", "cuda_available"} <= record.keys()

def test_existing_manifest_is_loaded_and_validated(tmp_path: Path):
    path = tmp_path / "split_manifest.json"
    write_json(path, _valid_manifest(tmp_path))
    assert load_or_create_manifest(tmp_path, 42, path) == _valid_manifest(tmp_path)
```

- [ ] **Step 2: Run:** `python -m pytest tests/test_train.py -k "environment_record or existing_manifest" -v`  
Expected: import failure for undefined functions.

- [ ] **Step 3: Implement** `--split-manifest` and `--learning-curves`. Save `config.json` and `environment.json` before training. Save `resource_profile.json` after the one test evaluation with `train_seconds`, `trainable_parameters`, `test_inference_seconds`, `test_cases`, `seconds_per_case`, and `peak_vram_mb`. Save a PNG of train loss, validation Dice, and validation IoU. Synchronize CUDA before and after timing.

- [ ] **Step 4: Run:** `python -m pytest tests/test_train.py -v; python -m pytest -v; python -m py_compile train.py utils.py; python train.py --help`  
Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```powershell
git add recognition/oasis_brain_analysis/utils.py recognition/oasis_brain_analysis/train.py recognition/oasis_brain_analysis/tests/test_train.py
git commit -m "feat: record reproducible training evidence"
```

### Task 3: Generate manifest-driven comparison panels

**Files:** Modify `predict.py` and `tests/test_predict.py`.

**Interfaces:** `load_prediction_records(data_root: str | Path, manifest_path: str | Path, maximum: int) -> list[dict[str, str]]` consumes one shared manifest and one or two checkpoints.

- [ ] **Step 1: Write the failing test**

```python
def test_prediction_records_use_test_partition_from_saved_manifest(tmp_path: Path):
    path = tmp_path / "split.json"
    write_json(path, _valid_manifest(tmp_path))
    assert [r["case_id"] for r in load_prediction_records(tmp_path, path, 0)] == ["003_slice_0"]
```

- [ ] **Step 2: Run:** `python -m pytest tests/test_predict.py -k saved_manifest -v`  
Expected: import failure for `load_prediction_records`.

- [ ] **Step 3: Implement** `--split-manifest`, manifest validation, and optional `--comparison-checkpoint`. The comparison renders input, truth, two predictions, and two categorical overlays. Retain four-panel single-model smoke mode. Save `failure_ranking.json` containing case IDs and per-model pixel error.

- [ ] **Step 4: Run:** `python -m pytest tests/test_predict.py -v; python -m pytest -v; python -m py_compile predict.py; python predict.py --help`  
Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```powershell
git add recognition/oasis_brain_analysis/predict.py recognition/oasis_brain_analysis/tests/test_predict.py
git commit -m "feat: make failure panels manifest-driven"
```

### Task 4: Add a Rangpur formal-experiment entry point

**Files:** Create `scripts/run_formal_experiment.slurm`; modify repository `.gitignore` and project `README.md`.

**Interfaces:** consumes `train.py --split-manifest`, `predict.py --split-manifest`, and `FORMAL_OUTPUT_ROOT`; produces one job directory with both model artifact sets, logs, hardware evidence, and comparison panels.

- [ ] **Step 1: Red check:** `rg --files scripts`  
Expected: no formal script exists.

- [ ] **Step 2: Implement** a script using `set -euo pipefail`. It resolves the repository directory, creates `${FORMAL_OUTPUT_ROOT}/formal_${SLURM_JOB_ID}`, records `nvidia-smi`, `python --version`, `python -m pip freeze`, and exact command lines. It generates baseline manifest, runs improved using it, and makes comparison panels from both checkpoints. Defaults: `DATA_ROOT=/home/groups/comp3710/OASIS`, `EPOCHS=30`, `BATCH_SIZE=8`; environment variables override them. Ignore `artifacts/`, `formal_results/`, `*.pt`. README starts commands in the project directory and names retrieve/external-completion gates.

- [ ] **Step 3: Run:** `bash -n scripts/run_formal_experiment.slurm; rg -n "set -euo pipefail|--split-manifest|nvidia-smi" scripts/run_formal_experiment.slurm; rg -n "cd recognition/oasis_brain_analysis|sbatch scripts/run_formal_experiment.slurm|Turnitin|Tutor" README.md`  
Expected: all checks exit 0.

- [ ] **Step 4: Commit**

```powershell
git add recognition/oasis_brain_analysis/scripts/run_formal_experiment.slurm .gitignore recognition/oasis_brain_analysis/README.md
git commit -m "feat: add Rangpur formal experiment workflow"
```

### Task 5: Execute only after code preflight passes

**Files:** Modify `README.md` only after artifact retrieval; create results outside Git.

- [ ] **Step 1: Run:** `cd recognition/oasis_brain_analysis; python -m pytest -v; python train.py --help; python predict.py --help; bash -n scripts/run_formal_experiment.slurm`  
Expected: all commands exit 0.

- [ ] **Step 2: Submit on Rangpur:** `sbatch scripts/run_formal_experiment.slurm`  
Expected: record a job ID; do not claim completion.

- [ ] **Step 3: Verify:** `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS`  
Expected: `COMPLETED` and `0:0`, plus both metrics JSON files, resource profiles, histories, logs, checkpoints, and panels.

- [ ] **Step 4: Retrieve:** `scp -r <host>:<formal-output-directory> ./formal_results/<job-id>`; confirm the two stored manifests are identical before reporting values.

- [ ] **Step 5: Finalize README** from JSON values only, with job/hardware/configuration/case IDs and a conservative recommendation. Keep tutor, PR, and Turnitin items open until distinct receipts exist.

- [ ] **Step 6: Run:** `git diff --check; cd recognition/oasis_brain_analysis; python -m pytest -v; python train.py --help; python predict.py --help`  
Expected: all commands exit 0; then commit `docs: report verified OASIS formal experiment results`.
