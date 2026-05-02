"""Run YOLOv8 ball detection on a test clip and produce an annotated video + stats.

Usage: uv run python scripts/yolo_ball_detect.py
"""

from pathlib import Path

import csv
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

INPUT = Path("outputs/clips/test_2min.mp4")
OUT_VIDEO = Path("outputs/clips/test_2min_annotated.mp4")
OUT_CSV = Path("outputs/clips/detections.csv")

# COCO class 32 = sports ball (covers volleyballs)
BALL_CLASS = 32
CONF_THRESHOLD = 0.15   # low threshold to catch small/blurry balls
MODEL_NAME = "yolov8x.pt"


def main() -> None:
    model = YOLO(MODEL_NAME)
    print(f"loaded {MODEL_NAME}, running on: {next(model.model.parameters()).device}")

    cap = cv2.VideoCapture(str(INPUT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{INPUT}: {W}x{H} @ {fps:.2f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (W, H))

    rows: list[dict] = []
    detected = 0

    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = ["frame", "time_s", "detected", "conf", "x1", "y1", "x2", "y2"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()

        for frame_idx in tqdm(range(total_frames), desc="detecting"):
            ret, frame = cap.read()
            if not ret:
                break

            time_s = frame_idx / fps
            results = model(frame, classes=[BALL_CLASS], conf=CONF_THRESHOLD,
                            device="mps", verbose=False)[0]

            row = {"frame": frame_idx, "time_s": f"{time_s:.3f}",
                   "detected": 0, "conf": "", "x1": "", "y1": "", "x2": "", "y2": ""}

            if len(results.boxes) > 0:
                # take highest-confidence detection
                best = results.boxes[results.boxes.conf.argmax()]
                conf = float(best.conf[0])
                x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # draw circle + confidence
                radius = max((x2 - x1), (y2 - y1)) // 2
                cv2.circle(frame, (cx, cy), radius + 4, (0, 255, 0), 3)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                row.update({"detected": 1, "conf": f"{conf:.3f}",
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                detected += 1

            # timestamp overlay
            cv2.putText(frame, f"t={time_s:.1f}s  ball={'YES' if row['detected'] else 'no'}",
                        (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if row["detected"] else (0, 0, 255), 2)

            writer_csv.writerow(row)
            writer.write(frame)

    cap.release()
    writer.release()

    recall = detected / total_frames * 100
    print(f"\n=== results ===")
    print(f"frames processed : {total_frames}")
    print(f"ball detected    : {detected} ({recall:.1f}%)")
    print(f"annotated video  : {OUT_VIDEO}")
    print(f"detections csv   : {OUT_CSV}")


if __name__ == "__main__":
    main()
