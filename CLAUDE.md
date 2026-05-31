# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Environment Setup

**Python environment**: Use the `.venv` virtualenv in the project root.
```bash
source .venv/bin/activate
```

**CUDA ops** (must be compiled before first use):
```bash
source .venv/bin/activate
cd models/dino/ops
python setup.py build install
cd -
```

**Key dependencies**: PyTorch 2.x, pycocotools, wandb.

## Common Commands

**Local test run (1 epoch, batch size 1 for RTX 3080)**:
```bash
bash scripts/local_test.sh
```

**Training on HPC (LSF/bsub)**:
```bash
bsub < scripts/dino_fungi_hpc.sh
```

**Training locally**:
```bash
bash scripts/DINO_train_fungi.sh checkpoints/checkpoint0033_4scale.pth
```

**Evaluation only**:
```bash
python main.py \
  --output_dir logs/eval \
  -c config/DINO/DINO_4scale_fungi.py \
  --coco_path datasets/fungi_31_05_26 \
  --eval --resume logs/dino_r50_75img_lr1e6_flip/checkpoint_best_regular.pth \
  --options dn_box_noise_scale=1.0
```

## Project Context

This is a Bachelor's project comparing DINO (detection-only) against MaskDINO (instance segmentation) on a small custom fungi dataset. The goal is an apples-to-apples comparison of whether using coarse masks (MaskDINO) is better than no masks (DINO) for colony detection.

- **Dataset**: `datasets/fungi_31_05_26/` — COCO-format, single class (`mold`), 75 train / 19 val fungi-colony petri-dish images (replaces the older `fungi_01_05_26`)
- **Pretrained checkpoint**: `checkpoints/checkpoint0033_4scale.pth` — DINO R50 4-scale epoch 32 from the official repo
- **Fungi config**: `config/DINO/DINO_4scale_fungi.py` — overrides base with `num_classes=2` (one foreground class + background, DINO's max-id+1 convention), 1000 epochs, lr=1e-6/1e-7, batch=4, 1024px resolution (matches MaskDINO setup)
- **Training logs**: `logs/dino_r50_75img_lr1e6_flip/`
- **Augmentation** (`datasets/coco.py` → `make_coco_transforms`, train branch): random **horizontal + vertical flips** then resize to 1024px. No rotation — DINO is detection-only, so it can't recompute exact rotated boxes the way MaskDINO does (which uses masks); flips are label-exact for axis-aligned boxes. See `RandomVerticalFlip` in `datasets/transforms.py`.

## W&B Logging

Wandb logging is integrated in `main.py`. It runs only during training (not eval/test mode), only on rank 0.

- **Project**: `dtu_bachelor`
- **Entity**: `rasmuslinnemann-danmarks-tekniske-universitet-dtu`
- **Run name**: derived from `--output_dir` basename (e.g. `DINO_fungi_finetune`)
- **Logged metrics**:
  - `train/loss`, `train/loss_ce`, `train/loss_bbox`, `train/loss_giou`, `train/lr`, etc. (per epoch, unscaled variants excluded)
  - `eval/AP`, `eval/AP50`, `eval/AP75`, `eval/APs/APm/APl`, `eval/AR*` (from COCO bbox eval)
  - `eval/loss`, `eval/class_error` (validation set losses)

## Architecture Overview

DINO is a Transformer-based object detector using DINO-style self-supervised pretraining ideas applied to detection queries. No mask output — bounding boxes only.

**Key files**:
- `main.py` — training entry point, argument parsing, epoch loop, W&B integration
- `engine.py` — `train_one_epoch()`, `evaluate()`, `test()`
- `config/DINO/DINO_4scale_fungi.py` — fungi fine-tuning config
- `models/` — DINO model, criterion, postprocessors
- `util/misc.py` — `MetricLogger`, `SmoothedValue`, distributed utils

## Known Fixes (vs upstream)

- `torch.load(..., weights_only=False)` added everywhere for PyTorch 2.x compatibility
- Dataset paths updated for custom COCO-format fungi dataset
