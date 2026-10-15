# OASIS Brain Analysis Project

## Current implemented module

The current repository module is **2D brain tissue segmentation**: a baseline 2D U-Net is compared with an improved 2D U-Net on the OASIS PNG data. The directory name intentionally leaves room for later OASIS analysis modules; it does not claim that 3D segmentation, VAE, GAN, or any other extension is already implemented.

## Project and engineering question

This COMP3710 Pattern Recognition project implements the supplied specification's Easy-difficulty OASIS Brain 2D segmentation option. The prototype labels a grayscale MRI slice with four tissue classes for clinician or imaging-researcher review; it is not an autonomous diagnostic system.

The investigation asks whether residual encoding, attention-gated skip connections and bottleneck dropout improve tissue-boundary utility over a plain 2D U-Net sufficiently to justify extra parameters, GPU memory and latency. Baseline and advanced models use the same deterministic subject-disjoint held-out set.

## Working principles

`BaselineUNet` has three 2D encoder/decoder levels, convolution blocks, max pooling, transposed convolutions and direct skips. `ImprovedUNet` keeps comparable depth but uses residual blocks, attention gates on skip features and bottleneck dropout. Both emit four logits per pixel; training combines cross-entropy with soft Dice loss.

```text
MRI slice -> normalisation -> encoder -> bottleneck -> decoder -> 4-class logits
                              \-> baseline direct skips
                              \-> improved residual blocks + attention gates
```

## Feasibility review - compulsory midpoint check-off

**Need and scope.** The prototype produces reviewable tissue maps from preprocessed OASIS PNG slices. `dataset.py` pairs images and masks then allocates complete subjects, rather than slices, into train, validation and test partitions.

**Acceptance criteria.** (1) no subject overlaps across partitions; (2) both models emit `(N, 4, H, W)` logits; (3) class-wise held-out Dice and IoU are exported, with the course target Dice >= 0.90 across tissue labels; (4) peak VRAM and inference time are recorded; (5) three to five high-error cases have input, target, prediction and FP/FN panels.

**Initial feasibility evidence.** CPU tests verify deterministic subject allocation, pair validation, output geometry, multiclass loss backpropagation, Dice/IoU, evaluation-result fields and overlay semantics. This is software evidence, not a trained-model result.

**Risk, budget and fallback.** Pair naming, class imbalance, memory demand and high mean Dice masking thin-boundary errors are the main risks. The loader rejects unmatched pairs, records the exact split and defaults to small 2D batches. If the improved model does not improve per-class scores and qualitative boundary cases, retain the simpler baseline with human review rather than claim an unsupported gain.

**Tutor consultation.** Not yet recorded in this private planning repository. This review must be presented at the compulsory practical check-off; add date and feedback before submission.

## Data, installation and execution

The specification lists OASIS at Rangpur path `/home/groups/comp3710/OASIS`. Never commit images, masks, checkpoints or credentials. The loader expects `case_<subject>_<slice>.png` paired with `seg_<subject>_<slice>.png`.

```powershell
cd E:\vscode\PatternAnalysis-2026\recognition\oasis_brain_analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -v
python train.py --data-root /home/groups/comp3710/OASIS --model baseline --seed 42 --output-dir artifacts
python train.py --data-root /home/groups/comp3710/OASIS --model improved --seed 42 --split-manifest artifacts/baseline/split_manifest.json --output-dir artifacts
python predict.py --data-root /home/groups/comp3710/OASIS --split-manifest artifacts/baseline/split_manifest.json --checkpoint artifacts/baseline/best.pt --comparison-checkpoint artifacts/improved/best.pt --output-dir artifacts/predictions --max-cases 5
```

Replace the Rangpur path with the local PNG directory when running locally. Training writes `config.json`, `environment.json`, `split_manifest.json`, `history.csv`, `learning_curves.png`, `test_metrics.json`, `resource_profile.json` and a checkpoint. Formal prediction writes six-panel baseline/improved comparisons plus `failure_ranking.json`.

### Rangpur formal experiment

Upload the current local project files to `~/oasis_brain_analysis` on Rangpur without using GitHub, then submit from that exact directory:

```bash
cd ~/oasis_brain_analysis
chmod +x scripts/run_formal_experiment.slurm
sbatch scripts/run_formal_experiment.slurm
```

The script uses `/home/groups/comp3710/OASIS`, 30 epochs, batch size 8 and seed 42 by default. Override only through recorded environment variables, for example `EPOCHS=40 BATCH_SIZE=4 sbatch scripts/run_formal_experiment.slurm`. It writes the job output beneath `~/oasis_formal_results/formal_<job-id>/`; retain `stdout.log`, `stderr.log`, `nvidia-smi.txt`, `pip_freeze.txt`, both model folders and `comparison_panels/`.

After `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS` reports `COMPLETED` and `0:0`, retrieve the exact folder before documenting any metric:

```powershell
scp -r <uq-username>@<rangpur-host>:~/oasis_formal_results/formal_<job-id> .\formal_results\<job-id>
```

## Experimental protocol, results and failures

Train both models with the same seed and saved subject-level manifest. Select by validation mean Dice, load the best validation checkpoint and evaluate once on the held-out test partition. Report class-wise Dice and IoU, macro values, parameter count, peak VRAM, total inference time and per-case time. Review the highest-error cases, not only favourable images.

| Measure | Baseline | Improved U-Net | Evidence |
|---|---:|---:|---|
| Held-out class-wise Dice / IoU | Not executed here | Not executed here | `test_metrics.json` |
| Peak GPU VRAM | Not executed here | Not executed here | runtime record |
| Inference time | Not executed here | Not executed here | runtime record |

No final deployment recommendation is warranted before real result files and 3-5 panels exist. Red overlay pixels are false positives; blue pixels are false negatives. Record the case ID, tissue class, local trigger, FP/FN pattern and recommended human-review action for each analysed failure.

### Completion gates outside this repository

- Present the one-page feasibility review at the tutor consultation and record the date plus feedback.
- Confirm Blackboard's 2026 due date, submission target and review-response deadline.
- Verify each retrieved result number against its `test_metrics.json` and `resource_profile.json`, then replace the pending table cells and add the five case IDs.
- Create the required upstream pull request only when you choose to publish the verified local commits.
- Export the matching README revision to PDF, submit it to Turnitin and retain the receipt.
- Respond to teaching-team GitHub review with evidence-backed commits and replies after the PR exists.

### Real-data smoke-test evidence (not a final experiment)

On 9 October 2026, the local OASIS PNG source was audited successfully: 11,328 aligned image/mask pairs, 256x256 slices and four remapped label classes. The fixed seed created 247/53/54 train/validation/test subjects. A GPU smoke test ran one baseline epoch with eight slices per split on an NVIDIA GeForce RTX 3060 Laptop GPU (PyTorch 2.11.0+cu128). It wrote a checkpoint, manifest, CSV, held-out JSON and eight prediction panels. Its held-out mean Dice was 0.0182 and mean IoU was 0.0094; the eight visualised cases had pixel error rates of 96.1-96.5%. These deliberately small-run values demonstrate the pipeline and failure reporting path only. They are not a model-quality claim and do not approach the course Dice target.

The smoke result establishes that a complete experiment must train both models on the full fixed manifest, retain learning curves, record the exact hardware/configuration, compare class-wise metrics and inspect at least three to five ranked failure panels. The compulsory tutor feasibility consultation and the required course-fork Pull Request remain external submission requirements.

## Structure and AI disclosure

`modules.py` contains models; `dataset.py` handles pairing/splitting/loading; `train.py` trains and evaluates; `predict.py` creates error panels; `utils.py` provides metrics and reproducibility; `tests/` contains focused behavioural checks.

An AI coding assistant assisted with implementation planning, test-first code drafts, documentation language and command construction. The author must audit every changed line, run and inspect the local tests, review Rangpur logs and saved artifacts, verify every reported metric against the relevant JSON/CSV output, select no result solely because it appears favourable, and revise any conclusion unsupported by evidence.

## References

1. COMP3710 / COMP7710, *Pattern Recognition Working Prototype Project*, supplied Version 2.2 draft.
2. OASIS Brains, Open Access Series of Imaging Studies; course-preprocessed copy at the stated Rangpur path.
3. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015.
4. Z. Oktay et al., “Attention U-Net: Learning Where to Look for the Pancreas,” arXiv:1804.03999, 2018.
