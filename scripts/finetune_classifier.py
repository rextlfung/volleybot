"""Fine-tune YOLOv8-cls for live/dead play-state classification.

The model takes full 640×640 frames as input and outputs two class probabilities:
  live  — rally is in progress
  dead  — between points (serve setup, rotation, ball retrieval, etc.)

Workflow:
  1. Sample frames:
       uv run python scripts/sample_frames_for_classification.py \
           --inputs data/20220805g1.mp4 data/20230110.mp4 data/20250508g1.mp4 \
           --n-frames 200

  2. Create a Roboflow Classification project, upload the frames, and label each
     image as "live" or "dead". Export as "Folder Structure / YOLOv8" format.
     Unzip to data/roboflow_classification/.

  3. Run this script:
       uv run python scripts/finetune_classifier.py \
           --data data/roboflow_classification \
           --epochs 30 --device cpu

  Best weights: runs/classify/volleybot_cls/weights/best.pt

Labeling guidelines — label as LIVE from the moment the server tosses the ball
until the point ends (ball hits floor / goes out / is caught). Everything else
is DEAD: players rotating, server bouncing the ball, ball being retrieved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from volleybot.device import best_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n-cls for live/dead play-state classification."
    )
    p.add_argument("--data", type=Path, required=True,
                   help="path to Roboflow classification export directory "
                        "(contains train/, valid/, test/ subfolders)")
    p.add_argument("--model", default="yolov8n-cls.pt",
                   help="base model: yolov8n-cls.pt / yolov8s-cls.pt (default yolov8n-cls.pt)")
    p.add_argument("--epochs", type=int, default=30,
                   help="training epochs (default 30 — classification converges faster)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="training image size (default 640)")
    p.add_argument("--batch", type=int, default=32,
                   help="batch size (default 32)")
    p.add_argument("--device", default=best_device(),
                   help="training device: cpu | cuda | mps")
    p.add_argument("--name", default="volleybot_cls",
                   help="run name under runs/classify/ (default volleybot_cls)")
    p.add_argument("--patience", type=int, default=10,
                   help="early-stop patience (default 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        print(f"dataset directory not found: {args.data}")
        sys.exit(1)

    # Validate expected folder structure
    for split in ["train", "valid"]:
        if not (args.data / split).exists():
            print(f"missing {split}/ folder in {args.data}")
            print("expected structure: train/live/, train/dead/, valid/live/, valid/dead/")
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
        project="runs/classify",
        # Augmentations appropriate for fixed-angle footage:
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        fliplr=0.5,
        flipud=0.0,
        degrees=3.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best}")
    print(f"\nTo classify a video:")
    print(f"  uv run python scripts/classify_frames.py \\")
    print(f"      --input data/20220805g1.mp4 \\")
    print(f"      --model {best} --device cpu")
    print(f"\nTo run the full pipeline with classifier-driven segmentation:")
    print(f"  uv run python scripts/cut_rallies.py \\")
    print(f"      --input data/20220805g1.mp4 \\")
    print(f"      --classifier-model {best} --concat")


if __name__ == "__main__":
    main()
