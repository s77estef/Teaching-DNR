#!/bin/bash
#SBATCH --job-name=tdnr
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load Anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/tdnr
cd "$HOME/code/Teaching-DNR"
python fine_tune/fine_tune_grpo_label.py