#!/usr/bin/env bash
#SBATCH --job-name=oasis-formal
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Run both OASIS models on one saved subject-disjoint manifest.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/groups/comp3710/OASIS}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${HOME}/oasis_formal_results}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
RUN_DIR="${FORMAL_OUTPUT_ROOT}/formal_${SLURM_JOB_ID:-manual_$(date +%Y%m%dT%H%M%S)}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/stdout.log") 2> >(tee -a "${RUN_DIR}/stderr.log" >&2)

cd "${PROJECT_DIR}"
source "${HOME}/miniconda3/bin/activate"
conda activate torch
printf 'project_dir=%q\ndata_root=%q\nepochs=%q\nbatch_size=%q\nseed=%q\n' \
  "${PROJECT_DIR}" "${DATA_ROOT}" "${EPOCHS}" "${BATCH_SIZE}" "${SEED}" > "${RUN_DIR}/run_config.env"
printf '%q ' "$@" > "${RUN_DIR}/slurm_arguments.txt"
printf '\n' >> "${RUN_DIR}/slurm_arguments.txt"
nvidia-smi > "${RUN_DIR}/nvidia-smi.txt"
python --version > "${RUN_DIR}/python_version.txt"
python -m pip freeze > "${RUN_DIR}/pip_freeze.txt"

echo "[$(date --iso-8601=seconds)] baseline training starts"
python train.py --data-root "${DATA_ROOT}" --model baseline --seed "${SEED}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --output-dir "${RUN_DIR}"

MANIFEST="${RUN_DIR}/baseline/split_manifest.json"
echo "[$(date --iso-8601=seconds)] improved training starts"
python train.py --data-root "${DATA_ROOT}" --model improved --seed "${SEED}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --split-manifest "${MANIFEST}" --output-dir "${RUN_DIR}"

echo "[$(date --iso-8601=seconds)] held-out comparison panels start"
python predict.py --data-root "${DATA_ROOT}" --split-manifest "${MANIFEST}" \
  --checkpoint "${RUN_DIR}/baseline/best.pt" --comparison-checkpoint "${RUN_DIR}/improved/best.pt" \
  --output-dir "${RUN_DIR}/comparison_panels" --max-cases 5

echo "[$(date --iso-8601=seconds)] formal experiment completed"
