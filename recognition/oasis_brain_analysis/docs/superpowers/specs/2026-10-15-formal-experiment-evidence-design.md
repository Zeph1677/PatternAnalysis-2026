# OASIS Formal Experiment and Evidence Design

## Purpose

Produce a course-assessable, reproducible comparison of `BaselineUNet` and
`ImprovedUNet` on the OASIS 2D four-class segmentation task. The experiment
must use the same subject-disjoint allocation, select checkpoints only with
validation data, and evaluate each selected checkpoint once on the held-out
test subjects.

This design distinguishes executable local or Rangpur work from actions that
must occur in Blackboard, Turnitin, a tutor consultation, or GitHub.

## Experiment contract

Each model run receives an explicit experiment configuration containing the
data root, seed, epochs, batch size, optimizer settings, model name, and a
shared split-manifest path. The first run generates the split manifest from
the audited data; the second run validates and reuses exactly that manifest.
The manifest stores relative paths, subject IDs, case IDs, allocation name,
and dataset identity information so a changed dataset cannot silently reuse
the allocation.

For each run, the output directory contains:

- `config.json` with command-line configuration and immutable protocol fields;
- `environment.json` with Python, PyTorch, CUDA, GPU, device, and platform
  details;
- `split_manifest.json` or a verified copy of the shared manifest;
- `history.csv` and `learning_curves.png` for epoch-level training and
  validation evidence;
- `best.pt`, selected only by validation macro Dice;
- `test_metrics.json`, written after the single held-out evaluation;
- `resource_profile.json` with train duration, parameter count, peak VRAM,
  full-test latency, and per-case latency;
- stdout/stderr and Slurm job metadata when run on Rangpur.

Checkpoints and large images remain ignored by Git. Small, derived result
summaries may be committed only after they have been reviewed against the raw
artifacts.

## Training, evaluation, and integrity controls

`train.py` will accept a shared manifest rather than independently rebuilding
a split for every invocation. It validates that all paths exist, every record
belongs to its named partition, subject sets are mutually disjoint, and each
record remains within the requested data root. It saves one configuration and
environment snapshot before training starts.

Training records train loss and validation metrics per epoch. The test set is
not iterated during model selection. After training, `best.pt` is loaded and
the test set is evaluated once. Resource measurements are separated from
validation measurements. CUDA synchronization surrounds measured inference
where CUDA is available, preventing asynchronous execution from understating
latency.

## Qualitative failure evidence

`predict.py` will consume the saved experiment manifest and checkpoint, not
recreate a partition from a seed. It will calculate an error score per
held-out case, write a machine-readable ranking, and render the selected
three to five highest-error cases. Each panel will show MRI input, ground
truth, baseline prediction, improved prediction, and category-aware error
information where both model artifacts are supplied. The report will state
the exact case IDs and describe observed false positives, false negatives,
and tissue confusions without claiming a clinical interpretation unsupported
by the images.

## Rangpur execution

A committed Slurm script will run the baseline and improved configurations
serially against `/home/groups/comp3710/OASIS` (or an explicitly supplied
course path). It will write a job-specific log directory, preserve the shell
command, record `nvidia-smi` output and package versions, and stop on command
failure. It will not run an extra test pass merely to populate a report.

The script will use an explicit output root outside version control. After a
job reports successful completion, the corresponding `test_metrics.json`,
resource profile, history, checkpoint metadata, and failure panels must be
retrieved and checked before README values are updated.

## Documentation and external completion boundary

The README will use directory-correct setup and test commands, identify the
OASIS label mapping without calling it HipMRI, explain the protocol, and give
an evidence table whose fields remain marked pending until artifact files
exist. It will include a precise AI-use disclosure describing assistance,
human review, and verification responsibilities.

The following cannot be created from repository code: the tutor's recorded
midpoint check-off, Blackboard due-date confirmation, Turnitin submission
receipt, an upstream pull request, and replies to teaching-team review. The
README will list these as completion gates rather than treating local files as
substitutes.

## Test strategy

Before each behavior change, add a focused pytest that fails for the missing
behavior. Tests cover shared-manifest validation, environment/config evidence
schema, metric and timing fields, and manifest-driven prediction selection.
Run focused tests during changes, then run the project suite, syntax checks,
CLI help, and a small CPU smoke run that writes the new artifact schema.
