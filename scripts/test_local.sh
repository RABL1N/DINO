#!/bin/bash
# Local test run for 1 epoch with Batch Size 1 to fit on 10GB RTX 3080

source .venv/bin/activate
python main.py \
  --output_dir logs/dino_r50_75img_lr1e6_flip_test \
  -c config/DINO/DINO_4scale_fungi.py \
  --coco_path datasets/fungi_31_05_26 \
  --pretrain_model_path checkpoints/checkpoint0033_4scale.pth \
  --finetune_ignore label_enc.weight class_embed \
  --options batch_size=1 epochs=3 \
  --save_log
