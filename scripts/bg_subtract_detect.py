"""Background-subtraction ball detector — fast alternative to YOLO.

Exploits the fixed camera + green curtain backdrop. Works by:
1. Building a background model from the first N seconds of video
2. For each frame, computing the foreground mask (moving pixels)
3. In the upper 2/3 of frame (above net), finding circular blobs ~ball-sized
4. Filtering by size, circularity, and position (avoid player bodies)

This runs 60-100x faster than YOLOv8 since there's no deep network inference.
It will produce false positives from player arms/jumps but combined with the
YOLO detections it can fill gaps.

Usage: uv run python scripts/bg_subtract_detect.py [--input PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_INPUT = Path("outputs/clips/test_2min.mp4")
DEFAULT_OUT_VIDEO = Path("outputs/clips/test_2min_bg_annotated.mp4")
DEFAULT_OUT_CSV = Path("outputs/clips/detections_bg.csv")

# Court region: top half of frame is where the ball spends most time in flight.
# Bottom half is dominated by players — we keep it but weight detections higher
# in the upper region.
SEARCH_TOP_FRAC = 0.1     # ignore top 10% (ceiling/lights)
SEARCH_BOT_FRAC = 0.75    # ignore bottom 25% (floor where players' feet are)

# Ball size at 1080p back-of-court: ~25-60px diameter
BALL_MIN_PX = 15
BALL_MAX_PX = 80

# Minimum circularity (0=line, 1=perfect circle)
MIN_CIRCULARITY = 0.55


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-video", type=Path, default=DEFAULT_OUT_VIDEO)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--bg-frames", type=int, default=300,
                   help="frames to use for background model initialisation")
    p.add_argument("--no-video", action="store_true",
                   help="skip writing annotated video (faster)")
    return p.parse_args()


def build_background(cap: cv2.VideoCapture, n_frames: int) -> np.ndarray:
    """Compute median background from the first n_frames."""
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def find_ball(fg_mask: np.ndarray, H: int, W: int) -> tuple[int, int, int] | None:
    """Find the most ball-like contour in the foreground mask.

    Returns (cx, cy, radius) or None.
    """
    # restrict search region
    y_top = int(H * SEARCH_TOP_FRAC)
    y_bot = int(H * SEARCH_BOT_FRAC)
    roi = fg_mask[y_top:y_bot, :]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, tuple[int, int, int]] | None = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        diameter = radius * 2

        if diameter < BALL_MIN_PX or diameter > BALL_MAX_PX:
            continue
        if circularity < MIN_CIRCULARITY:
            continue

        score = circularity * min(area, np.pi * radius ** 2) / max(area, 1)
        if best is None or score > best[0]:
            best = (score, (int(cx), int(cy + y_top), int(radius)))

    return best[1] if best else None


def main() -> None:
    args = parse_args()
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.input}: {W}x{H} @ {fps:.2f}fps, {total_frames} frames")

    print(f"building background model from first {args.bg_frames} frames ...")
    bg = build_background(cap, args.bg_frames)
    print("done")

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=False)

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (W, H))

    rows: list[dict] = []
    detected = 0
    t0 = time.time()

    with open(args.out_csv, "w", newline="") as f:
        # YOLO-compatible format so analyze_detections.py can consume this CSV
        fieldnames = ["frame", "time_s", "detected", "conf", "x1", "y1", "x2", "y2"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for frame_idx in tqdm(range(total_frames), desc="bg-subtract"):
            ret, frame = cap.read()
            if not ret:
                break

            time_s = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # combine adaptive MOG2 with static background diff
            fg_mog = mog2.apply(gray)
            fg_static = cv2.absdiff(gray, bg)
            _, fg_thresh = cv2.threshold(fg_static, 30, 255, cv2.THRESH_BINARY)

            fg = cv2.bitwise_and(fg_mog, fg_thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

            result = find_ball(fg, H, W)
            row: dict = {"frame": frame_idx, "time_s": f"{time_s:.3f}",
                         "detected": 0, "conf": "",
                         "x1": "", "y1": "", "x2": "", "y2": ""}

            if result is not None:
                cx, cy, radius = result
                detected += 1
                x1, y1 = cx - radius, cy - radius
                x2, y2 = cx + radius, cy + radius
                row.update({"detected": 1, "conf": "1.000",
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})

                if writer is not None:
                    cv2.circle(frame, (cx, cy), radius + 4, (255, 128, 0), 3)
                    cv2.putText(frame, "BG", (cx - 10, cy - radius - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

            if writer is not None:
                label_color = (255, 128, 0) if row["detected"] else (80, 80, 80)
                cv2.putText(frame, f"t={time_s:.1f}s  BG={'YES' if row['detected'] else 'no'}",
                            (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
                writer.write(frame)

            csv_writer.writerow(row)

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    recall = detected / total_frames * 100
    print(f"\n=== bg-subtract results ===")
    print(f"frames processed : {total_frames}")
    print(f"ball detected    : {detected} ({recall:.1f}%)")
    print(f"total time       : {elapsed:.0f}s ({total_frames/elapsed:.1f} fps processing)")
    if not args.no_video:
        print(f"annotated video  : {args.out_video}")
    print(f"detections csv   : {args.out_csv}")


if __name__ == "__main__":
    main()
