#!/bin/bash
# Usage: bash scripts/DINO_train_fungi.sh checkpoints/checkpoint0033_4scale.pth

CHECKPOINT=$1

if [ -z "$CHECKPOINT" ]; then
    echo "Error: Please provide the path to the pre-trained checkpoint."
    echo "Usage: bash scripts/DINO_train_fungi.sh checkpoints/checkpoint0033_4scale.pth"
    exit 1
fi

python main.py \
  --output_dir logs/dino_r50_75img_lr1e6_flip \
  -c config/DINO/DINO_4scale_fungi.py \
  --coco_path datasets/fungi_31_05_26 \
  --pretrain_model_path $CHECKPOINT \
  --finetune_ignore label_enc.weight class_embed \
  --options dn_box_noise_scale=1.0 \
  --save_log
