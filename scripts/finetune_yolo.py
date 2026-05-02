"""Fine-tune YOLOv8 on a Roboflow-exported volleyball dataset.

Workflow:
  1. Label frames in Roboflow (one class: "volleyball")
  2. Export as "YOLOv8" format — downloads a zip with data.yaml + images/labels/
  3. Unzip into e.g. data/roboflow_dataset/
  4. Run this script

Usage:
  uv run python scripts/finetune_yolo.py \
      --data data/roboflow_dataset/data.yaml \
      --model yolov8n.pt \
      --epochs 50 \
      --device mps

  # After training, the best weights are at:
  #   runs/detect/volleybot_*/weights/best.pt
  # Drop that path into cut_rallies.py --model to use it.

Model size tradeoffs (inference speed on MPS @ 640px, approx):
  yolov8n  ~25 fps   smallest, fastest — good starting point
  yolov8s  ~15 fps   moderate accuracy gain
  yolov8m  ~10 fps   diminishing returns unless recall is still low
  yolov8x   ~8 fps   current baseline (pre-fine-tune)

Start with yolov8n — a fine-tuned nano often beats a pretrained xlarge on
domain-specific footage because COCO generalisation hurts more than model size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on a Roboflow volleyball dataset."
    )
    p.add_argument("--data", type=Path, required=True,
                   help="path to data.yaml from Roboflow export")
    p.add_argument("--model", default="yolov8n.pt",
                   help="base model: yolov8n.pt / yolov8s.pt / yolov8m.pt (default yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=50,
                   help="training epochs (default 50; use 30 for a quick first run)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="training image size (default 640)")
    p.add_argument("--batch", type=int, default=16,
                   help="batch size (default 16; reduce to 8 if MPS OOM)")
    p.add_argument("--device", default="mps",
                   help="training device: mps | cpu | cuda (default mps)")
    p.add_argument("--name", default="volleybot",
                   help="run name under runs/detect/ (default volleybot)")
    p.add_argument("--patience", type=int, default=15,
                   help="early-stop patience in epochs (default 15)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        print(f"data.yaml not found: {args.data}")
        sys.exit(1)

    print(f"base model : {args.model}")
    print(f"dataset    : {args.data}")
    print(f"device     : {args.device}")
    print(f"epochs     : {args.epochs}  (patience={args.patience})")
    print(f"imgsz      : {args.imgsz}   batch={args.batch}")
    print()

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # Augmentations well-suited for fixed-camera footage:
        hsv_h=0.01,     # slight hue shift (gym lighting varies a little)
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,     # left-right flip (ball can be anywhere)
        flipud=0.0,     # never flip vertically (gravity is fixed)
        degrees=3.0,    # tiny rotation (camera is level)
        translate=0.1,
        scale=0.4,      # ball appears at many distances
        mosaic=0.5,
        mixup=0.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best}")
    print(f"\nTo use in the pipeline:")
    print(f"  uv run python scripts/cut_rallies.py --input data/20220805g1.mp4 --model {best}")


if __name__ == "__main__":
    main()
