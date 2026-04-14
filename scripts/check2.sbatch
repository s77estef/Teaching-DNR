#!/bin/bash
#SBATCH --job-name=check
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euxo pipefail

echo "STEP 1: starting job"

module load Miniforge3
echo "STEP 2: module loaded"

source "$(conda info --base)/etc/profile.d/conda.sh"
echo "STEP 3: conda.sh sourced"
conda activate ~/envs/tdnr

echo "STEP 4: conda env activated"
cd "$HOME/code/Teaching-DNR"
echo "STEP 5: cd successful"
conda run -p ~/envs/tdnr python -u eval/check.py Qwen3-4B-GRPO-rubric_plus_accuracy_20260413_185748/checkpoint-250
echo "STEP 6: python finished"
