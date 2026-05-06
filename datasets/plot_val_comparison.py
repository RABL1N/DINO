"""
Plot input | prediction | GT boxes side-by-side for every val image.

Usage:
    python datasets/plot_val_comparison.py \
        --dataset datasets/fungi_01_05_26 \
        --preds pred/dino_r50_48img_lr1e6/val \
        --output val_comparison.jpg
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mplfig
import matplotlib.backends.backend_agg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_boxes(img_rgb: np.ndarray, boxes_xyxy, color: str) -> np.ndarray:
    """Draw boxes on img_rgb using the Detectron2 VisImage approach; return RGB array."""
    h, w = img_rgb.shape[:2]
    dpi = 72
    fig = mplfig.Figure(frameon=False,
                        figsize=((w + 1e-2) / dpi, (h + 1e-2) / dpi),
                        dpi=dpi)
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.imshow(img_rgb, extent=(0, w, h, 0), interpolation="nearest")
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])

    font_size = max(np.sqrt(h * w) / 90, 10)
    linewidth  = max(font_size / 4, 1)

    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8,
        ))

    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    out = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    mplfig.Figure.clf(fig)
    return out


def load_gt_boxes(ann_path: Path) -> dict:
    """Return {filename: [[x1,y1,x2,y2], ...]} from a COCO JSON."""
    with open(ann_path) as f:
        coco = json.load(f)
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    gt = {}
    for ann in coco["annotations"]:
        fname = Path(id2file[ann["image_id"]]).name
        x, y, bw, bh = ann["bbox"]
        gt.setdefault(fname, []).append([x, y, x + bw, y + bh])
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/fungi_01_05_26")
    parser.add_argument("--preds",   default="pred/dino_r50_48img_lr1e6/val")
    parser.add_argument("--output",  default="val_comparison.jpg")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    pred_dir    = Path(args.preds)
    val_dir     = dataset_dir / "val"
    ann_path    = dataset_dir / "annotations" / "instances_val.json"

    gt_boxes = load_gt_boxes(ann_path)
    names    = sorted(p.name for p in val_dir.glob("*.jpg"))

    rows_per_grid = 3
    chunks   = [names[i:i + rows_per_grid] for i in range(0, len(names), rows_per_grid)]
    out_stem = Path(args.output).stem
    out_suffix = Path(args.output).suffix
    out_dir  = Path(args.output).parent

    for grid_idx, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), 3, figsize=(18, 6 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[None, :]

        for col, title in enumerate(["Input", "Prediction", "GT boxes"]):
            axes[0, col].set_title(title, fontsize=16, fontweight="bold", pad=8)

        for row, name in enumerate(chunk):
            stem = Path(name).stem
            img_rgb = np.array(Image.open(val_dir / name).convert("RGB"))
            pred_rgb = np.array(Image.open(pred_dir / (stem + ".jpg")).convert("RGB"))
            gt_rgb  = render_boxes(img_rgb, gt_boxes.get(name, []), color="#00FF00")

            for col, img in enumerate([img_rgb, pred_rgb, gt_rgb]):
                axes[row, col].imshow(img)
                axes[row, col].axis("off")

            axes[row, 0].set_ylabel(stem, fontsize=9, rotation=0, labelpad=160, va="center")

        plt.tight_layout()
        out_path = out_dir / f"{out_stem}_{grid_idx}{out_suffix}"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
