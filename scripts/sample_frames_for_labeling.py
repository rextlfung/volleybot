"""Sample frames from one or more videos for Roboflow labeling.

Samples N frames spread across each video. When a detections CSV is supplied,
sampling is biased: half the frames come from windows around existing YOLO
detections (good candidates for both positives and hard negatives), and half
are drawn uniformly so underrepresented game states are included.

Usage:
  # Uniform sampling from multiple videos:
  uv run python scripts/sample_frames_for_labeling.py \
      --inputs data/20220805g1.mp4 data/gym2.mp4 data/gym3.mp4 \
      --n-frames 150

  # Biased sampling using existing detections for the first video:
  uv run python scripts/sample_frames_for_labeling.py \
      --inputs data/20220805g1.mp4 \
      --n-frames 200 \
      --csv outputs/clips/detections.csv

Output: outputs/labeling/<video_stem>/frame_XXXXXX.jpg
Upload the labeling/ directory to Roboflow.
Roboflow tip: use "Auto Label" → COCO sports-ball to pre-label, then correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _uniform_indices(total_frames: int, n: int) -> list[int]:
    if n >= total_frames:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]


def _biased_indices(
    csv_path: Path,
    total_frames: int,
    n: int,
    frame_height: int = 1080,
) -> list[int]:
    """Half from near-detection windows, half uniform."""
    from volleybot.detection import load_csv, filter_detections

    detections = filter_detections(
        load_csv(csv_path),
        min_conf=0.25,
        max_ball_px=80,
        max_y_frac=0.72,
        frame_height=frame_height,
    )
    detected = [d.frame for d in detections if d.detected]

    rng = np.random.default_rng(42)
    half = n // 2

    # Sample from detected frames (positive windows)
    if len(detected) >= half:
        pos = sorted(rng.choice(detected, half, replace=False).tolist())
    else:
        pos = detected[:]

    # Pad with uniform to reach n
    uniform = _uniform_indices(total_frames, n - len(pos))
    combined = sorted(set(pos) | set(uniform))
    return combined[:n]


def _extract(video: Path, indices: list[int], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    want = set(indices)
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in want:
            cv2.imwrite(
                str(out_dir / f"frame_{idx:06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            saved += 1
        idx += 1
    cap.release()
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample frames from volleyball videos for Roboflow labeling."
    )
    p.add_argument("--inputs", nargs="+", type=Path, required=True,
                   help="source video files")
    p.add_argument("--n-frames", type=int, default=150,
                   help="frames to sample per video (default 150)")
    p.add_argument("--csv", type=Path, default=None,
                   help="detections CSV for the first input — enables biased sampling")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/labeling"),
                   help="root output directory (default outputs/labeling/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for i, video in enumerate(args.inputs):
        if not video.exists():
            print(f"[skip] {video} not found")
            continue

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        use_csv = (i == 0) and args.csv and args.csv.exists()
        if use_csv:
            indices = _biased_indices(args.csv, total, args.n_frames, frame_height=h)
            mode = f"biased (csv={args.csv.name})"
        else:
            indices = _uniform_indices(total, args.n_frames)
            mode = "uniform"

        out_dir = args.out_dir / video.stem
        print(f"{video.name}  [{total} frames @ {fps:.0f}fps, {h}p]")
        print(f"  sampling {len(indices)} frames ({mode}) → {out_dir}/")
        saved = _extract(video, indices, out_dir)
        print(f"  saved {saved} JPEGs")

    print(f"\nAll done. Upload {args.out_dir}/ to Roboflow.")
    print("In Roboflow: Auto Label → COCO 'sports ball' → correct mistakes manually.")


if __name__ == "__main__":
    main()
