#!/bin/bash
# Usage: sbatch scripts/check_dynamic.sbatch <model-checkpoint>
# Example:
#   sbatch scripts/check_dynamic.sbatch Qwen3-4B-GRPO-rubric_plus_accuracy_20260415_233315/checkpoint-250

#SBATCH --job-name=check
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euxo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch scripts/check_dynamic.sbatch <model-checkpoint>" >&2
    exit 1
fi

MODEL_CHECKPOINT="$1"

echo "STEP 1: starting job"
echo "Checkpoint: ${MODEL_CHECKPOINT}"

module load Miniforge3
echo "STEP 2: module loaded"

source "$(conda info --base)/etc/profile.d/conda.sh"
echo "STEP 3: conda.sh sourced"
conda activate ~/envs/tdnr

echo "STEP 4: conda env activated"
cd "$HOME/code/Teaching-DNR"
echo "STEP 5: cd successful"
conda run -p ~/envs/tdnr python -u eval/check.py "${MODEL_CHECKPOINT}"
echo "STEP 6: python finished"
