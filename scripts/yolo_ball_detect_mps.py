"""YOLOv8 ball detection using ultralytics video source — properly uses MPS.

Processes the video in one pass via model.predict(), which streams frames
efficiently and respects the device setting. Produces an annotated video
and a detections CSV.

Usage: uv run python scripts/yolo_ball_detect_mps.py [--input PATH] [--model yolov8x.pt]
"""

from pathlib import Path
import argparse
import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO

COCO_BALL_CLASS = 32     # sports ball in COCO 80-class models
CONF_THRESHOLD = 0.15    # low threshold — we'll filter in post
DEFAULT_MODEL = "yolov8x.pt"
DEFAULT_INPUT = Path("outputs/clips/test_2min.mp4")
DEFAULT_OUT_VIDEO = Path("outputs/clips/test_2min_annotated_mps.mp4")
DEFAULT_OUT_CSV = Path("outputs/clips/detections_mps.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-video", type=Path, default=DEFAULT_OUT_VIDEO)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="mps", help="mps | cpu | cuda")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--ball-class", type=int, default=None,
                   help="YOLO class id for the ball; auto-detected if omitted "
                        "(32 for COCO models, 0 for fine-tuned single-class models)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total_frames / fps
    print(f"{args.input}: {W}x{H} @ {fps:.2f}fps, {total_frames} frames ({duration_s:.1f}s)")

    model = YOLO(args.model)

    # Auto-detect ball class: COCO models have 80 classes (32=sports ball),
    # fine-tuned single-class models have only class 0.
    if args.ball_class is not None:
        ball_class = args.ball_class
    elif COCO_BALL_CLASS in model.names:
        ball_class = COCO_BALL_CLASS
    else:
        ball_class = 0
    print(f"ball class: {ball_class} ({model.names.get(ball_class, '?')})")

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (W, H))

    rows: list[dict] = []
    detected = 0
    t0 = time.time()

    # model.predict() with a video source streams frames and properly uses device
    stream = model.predict(
        source=str(args.input),
        classes=[ball_class],
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        frame = result.orig_img.copy()
        time_s = frame_idx / fps

        row = {
            "frame": frame_idx,
            "time_s": f"{time_s:.3f}",
            "detected": 0,
            "conf": "",
            "x1": "", "y1": "", "x2": "", "y2": "",
            "ball_w": "", "ball_h": "",
        }

        if len(result.boxes) > 0:
            best = result.boxes[result.boxes.conf.argmax()]
            conf = float(best.conf[0])
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max(bw, bh) // 2

            cv2.circle(frame, (cx, cy), radius + 4, (0, 255, 0), 3)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            row.update({
                "detected": 1, "conf": f"{conf:.3f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "ball_w": bw, "ball_h": bh,
            })
            detected += 1

        label_color = (0, 255, 0) if row["detected"] else (0, 0, 255)
        cv2.putText(frame, f"t={time_s:.1f}s  ball={'YES' if row['detected'] else 'no'}",
                    (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2)

        rows.append(row)
        writer.write(frame)

        if frame_idx % 300 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
            pct = detected / (frame_idx + 1) * 100
            print(f"  frame {frame_idx}/{total_frames}  {fps_proc:.1f}fps  "
                  f"detected so far: {detected} ({pct:.0f}%)  ETA: {eta:.0f}s")

    writer.release()

    with open(args.out_csv, "w", newline="") as f:
        fieldnames = ["frame", "time_s", "detected", "conf", "x1", "y1", "x2", "y2", "ball_w", "ball_h"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    recall = detected / total_frames * 100
    elapsed = time.time() - t0
    print(f"\n=== results ===")
    print(f"frames processed : {total_frames}")
    print(f"ball detected    : {detected} ({recall:.1f}%)")
    print(f"total time       : {elapsed:.0f}s ({total_frames/elapsed:.1f} fps processing)")
    print(f"annotated video  : {args.out_video}")
    print(f"detections csv   : {args.out_csv}")


if __name__ == "__main__":
    main()
