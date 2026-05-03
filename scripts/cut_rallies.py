"""End-to-end pipeline: run YOLO detection → segment → cut rally clips.

Segmentation can be driven by either:
  - Ball detection (default): uses the smoothed ball detection mask
  - Frame classifier (--classifier-*): uses a fine-tuned YOLOv8-cls live/dead model

Ball detection always runs (for tracking CSV used by future highlight/stats features).

Usage:
  # Detection-based segmentation (default):
  uv run python scripts/cut_rallies.py --input data/20220805g1.mp4 --concat

  # Classifier-based segmentation (fine-tuned live/dead model):
  uv run python scripts/cut_rallies.py --input data/20220805g1.mp4 \
      --classifier-model runs/classify/volleybot_cls/weights/best.pt --concat

  # Skip detection + classification if CSVs already exist:
  uv run python scripts/cut_rallies.py --input data/20220805g1.mp4 \
      --csv outputs/20220805g1/detections.csv \
      --classifier-csv outputs/20220805g1/classification.csv --concat
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2

from volleybot.detection import load_csv, filter_detections, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping, Segment
from volleybot.cutter import cut_segments, concat_segments
from volleybot.classification import load_classification_csv, classification_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="source video")
    p.add_argument("--csv", type=Path, default=None,
                   help="existing detections CSV (skips YOLO if provided)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory for cut clips (default: outputs/<stem>/rallies)")
    p.add_argument("--device", default="mps", help="mps | cpu | cuda")
    p.add_argument("--model", default="yolov8x.pt")
    p.add_argument("--conf", type=float, default=0.15,
                   help="YOLO detection confidence threshold")
    p.add_argument("--ball-class", type=int, default=None,
                   help="YOLO class id for the ball; auto-detected if omitted")
    # Detection filtering (applied post-YOLO to reduce false positives)
    p.add_argument("--filter-conf", type=float, default=0.25,
                   help="minimum confidence to keep after detection (default 0.25)")
    p.add_argument("--filter-max-px", type=int, default=80,
                   help="maximum ball width in pixels (default 80)")
    p.add_argument("--filter-max-y", type=float, default=0.72,
                   help="maximum ball center y as fraction of frame height (default 0.72)")
    p.add_argument("--no-filter", action="store_true",
                   help="disable post-detection filtering")
    # Segmentation
    p.add_argument("--dead-gap", type=float, default=3.0,
                   help="seconds of no detection before ending a rally")
    p.add_argument("--pre-roll", type=float, default=1.5,
                   help="seconds to include before first ball detection")
    p.add_argument("--post-roll", type=float, default=2.0,
                   help="seconds to include after last ball detection")
    # Classifier-based segmentation (overrides detection-based when provided)
    p.add_argument("--classifier-csv", type=Path, default=None,
                   help="existing classification CSV (skips classify_frames.py if provided)")
    p.add_argument("--classifier-model", type=Path, default=None,
                   help="YOLOv8-cls weights for live/dead classification; "
                        "if provided, classifier-driven segmentation is used")
    p.add_argument("--cls-fill-gap", type=float, default=0.5,
                   help="fill dead gaps ≤ this many seconds within live windows (default 0.5)")
    p.add_argument("--cls-min-conf", type=float, default=0.0,
                   help="minimum live confidence to count a frame as live (default 0.0)")
    p.add_argument("--concat", action="store_true",
                   help="concatenate all rally clips into a single output file")
    p.add_argument("--reencode", action="store_true",
                   help="re-encode cuts for frame accuracy (slower)")
    return p.parse_args()


def run_detection(args) -> Path:
    csv_path = Path("outputs") / args.input.stem / "detections.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"running YOLO detection on {args.input} ...")
    cmd = [
        "uv", "run", "python", "scripts/yolo_ball_detect_mps.py",
        "--input", str(args.input),
        "--out-video", str(csv_path.parent / "annotated.mp4"),
        "--out-csv", str(csv_path),
        "--model", args.model,
        "--device", args.device,
        "--conf", str(args.conf),
    ]
    if args.ball_class is not None:
        cmd += ["--ball-class", str(args.ball_class)]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"detection done in {time.time()-t0:.0f}s")
    return csv_path


def run_classification(args) -> Path:
    cls_csv = Path("outputs") / args.input.stem / "classification.csv"
    cls_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"running classifier on {args.input} ...")
    cmd = [
        "uv", "run", "python", "scripts/classify_frames.py",
        "--input", str(args.input),
        "--model", str(args.classifier_model),
        "--out-csv", str(cls_csv),
        "--device", args.device,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"classification done in {time.time()-t0:.0f}s")
    return cls_csv


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}")
        sys.exit(1)

    # Ball detection always runs (tracking CSV needed for future highlights/stats).
    csv_path = args.csv if args.csv else run_detection(args)

    if not csv_path.exists():
        print(f"detections CSV not found: {csv_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_s = total_frames / fps

    # Decide segmentation source: classifier (if provided) or ball detections (default).
    use_classifier = args.classifier_csv or args.classifier_model
    if use_classifier:
        cls_csv = args.classifier_csv if args.classifier_csv else run_classification(args)
        if not cls_csv.exists():
            print(f"classification CSV not found: {cls_csv}")
            sys.exit(1)
        print(f"\nloading classification from {cls_csv} ...")
        cls_results = load_classification_csv(cls_csv)
        mask = classification_mask(
            cls_results, fps,
            fill_gap_s=args.cls_fill_gap,
            min_conf=args.cls_min_conf,
        )
        print(f"classifier mask: {mask.sum()} live frames "
              f"({mask.mean()*100:.0f}% of {len(mask)})")
    else:
        print(f"\nloading detections from {csv_path} ...")
        detections = load_csv(csv_path)

        if not args.no_filter:
            n_before = sum(d.detected for d in detections)
            detections = filter_detections(
                detections,
                min_conf=args.filter_conf,
                min_ball_px=15,
                max_ball_px=args.filter_max_px,
                max_y_frac=args.filter_max_y,
                frame_height=H,
            )
            n_after = sum(d.detected for d in detections)
            print(f"filtered detections: {n_before} → {n_after} "
                  f"(conf≥{args.filter_conf}, width≤{args.filter_max_px}px, cy<{args.filter_max_y:.0%})")

        mask = smoothed_mask(detections, fps, fill_gap_s=0.5)

    segments = detect_rallies(
        mask, fps,
        dead_gap_s=args.dead_gap,
        pre_roll_s=args.pre_roll,
        post_roll_s=args.post_roll,
        total_duration_s=duration_s,
    )
    segments = merge_overlapping(segments)

    total_play = sum(s.duration_s for s in segments)
    print(f"\nfound {len(segments)} rally segments")
    print(f"total play time: {total_play:.0f}s / {duration_s:.0f}s "
          f"({total_play/duration_s*100:.0f}% of video)")

    out_dir = args.out_dir or (Path("outputs") / args.input.stem / "rallies")
    print(f"\ncutting segments to {out_dir} ...")
    clip_paths = cut_segments(
        args.input, segments, out_dir,
        prefix=args.input.stem,
        reencode=args.reencode,
    )

    if args.concat and clip_paths:
        concat_out = out_dir.parent / f"{args.input.stem}_all_rallies.mp4"
        print(f"\nconcatenating {len(clip_paths)} clips → {concat_out} ...")
        concat_segments(clip_paths, concat_out)
        print(f"highlight reel: {concat_out}")

    print(f"\ndone. {len(clip_paths)} clips in {out_dir}/")


if __name__ == "__main__":
    main()
