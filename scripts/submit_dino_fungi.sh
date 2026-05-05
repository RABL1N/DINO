#!/bin/bash
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -J dino_r50_48img_lr1e6
#BSUB -o logs/dino_r50_48img_lr1e6_%J.out
#BSUB -e logs/dino_r50_48img_lr1e6_%J.err
#BSUB -u s234806@dtu.dk
#BSUB -B
#BSUB -N

set -e

# Update this path if your DINO folder is in a different location on the HPC
PROJECT_DIR=$(pwd)
cd $PROJECT_DIR

module load cuda/12.4
source .venv/bin/activate

# Compile CUDA ops if not already importable
if ! python -c "import MultiScaleDeformableAttention" 2>/dev/null; then
    echo "Compiling DINO CUDA ops..."
    cd models/dino/ops
    python setup.py build install
    cd $PROJECT_DIR
else
    echo "DINO CUDA ops already compiled, skipping."
fi

mkdir -p logs/dino_r50_48img_lr1e6

# Run the training script we created earlier
bash scripts/DINO_train_fungi.sh checkpoints/checkpoint0033_4scale.pth
