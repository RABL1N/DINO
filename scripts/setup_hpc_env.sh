#!/bin/bash
# Run once on the HPC login node to set up the Python environment.
# Usage: bash scripts/setup_hpc_env.sh

set -e

PROJECT_DIR=$(pwd)
cd $PROJECT_DIR

module load cuda/12.4

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# DINO dependencies (includes wandb)
pip install -r requirements.txt

# NOTE: The CUDA ops (MultiScaleDeformableAttention) are compiled automatically
# by submit_dino_fungi.sh on the GPU node — no need to do it here.

echo "Environment ready. Submit the job with: bsub < scripts/submit_dino_fungi.sh"
