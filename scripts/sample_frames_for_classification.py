"""Sample frames from videos for live/dead play-state classification labeling.

Produces frames for upload to a Roboflow Classification project. Uses three
sampling strategies to ensure good coverage of all game states:

  1. Uniform  — frames spread evenly across the full video (baseline coverage)
  2. Live-biased — frames near ball detections (likely in-play)
  3. Transition-biased — frames near rally boundaries (the hard edge cases)

Transition frames are the most valuable: the moment a serve starts and the
moment a point ends. The model needs to learn these boundaries precisely.

Usage:
  # Uniform only (no detection CSV required):
  uv run python scripts/sample_frames_for_classification.py \
      --inputs data/20220805g1.mp4 data/20230110.mp4 data/20250508g1.mp4 \
      --n-frames 200

  # With detection CSV for transition-biased sampling:
  uv run python scripts/sample_frames_for_classification.py \
      --inputs data/20220805g1.mp4 \
      --n-frames 200 \
      --csv outputs/20220805g1/detections.csv

Output: outputs/labeling_cls/<video_stem>/frame_XXXXXX.jpg
Upload to a Roboflow Classification project (not detection).
Label each image as "live" or "dead".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _uniform(total: int, n: int) -> list[int]:
    if n >= total:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def _transition_biased(
    csv_path: Path,
    total: int,
    n: int,
    fps: float,
    frame_height: int = 1080,
) -> list[int]:
    """Return frames clustered around rally start/end transitions.

    Transitions are the hardest cases for the classifier and give the most
    labeling value — the serve toss (dead→live) and point end (live→dead).
    """
    from volleybot.detection import load_csv, filter_detections, smoothed_mask
    from volleybot.segmentation import detect_rallies, merge_overlapping

    dets = filter_detections(
        load_csv(csv_path),
        min_conf=0.25, max_ball_px=80, max_y_frac=0.72,
        frame_height=frame_height,
    )
    mask = smoothed_mask(dets, fps, fill_gap_s=0.5)
    total_s = total / fps
    segs = merge_overlapping(detect_rallies(
        mask, fps, dead_gap_s=3.0, pre_roll_s=0.0, post_roll_s=0.0,
        total_duration_s=total_s,
    ))

    # Collect frames within ±3s of each segment boundary
    window_frames = int(3.0 * fps)
    candidates: set[int] = set()
    for seg in segs:
        start_f = int(seg.raw_start_s * fps)
        end_f = int(seg.raw_end_s * fps)
        for f in range(max(0, start_f - window_frames),
                       min(total, start_f + window_frames)):
            candidates.add(f)
        for f in range(max(0, end_f - window_frames),
                       min(total, end_f + window_frames)):
            candidates.add(f)

    rng = np.random.default_rng(42)
    if len(candidates) > n:
        return sorted(rng.choice(list(candidates), n, replace=False).tolist())
    return sorted(candidates)


def _extract(video: Path, indices: list[int], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    saved = 0
    for idx in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(
                str(out_dir / f"frame_{idx:06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            saved += 1
    cap.release()
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample frames for live/dead classification labeling."
    )
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=200,
                   help="frames per video (default 200); split ~50/50 "
                        "uniform + transition-biased when CSV is provided")
    p.add_argument("--csv", type=Path, default=None,
                   help="detection CSV for the first input — enables transition sampling")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/labeling_cls"))
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
            half = args.n_frames // 2
            uniform = _uniform(total, half)
            transitions = _transition_biased(args.csv, total, half, fps, h)
            indices = sorted(set(uniform) | set(transitions))[:args.n_frames]
            mode = f"uniform + transition-biased (csv={args.csv.name})"
        else:
            indices = _uniform(total, args.n_frames)
            mode = "uniform"

        out_dir = args.out_dir / video.stem
        print(f"{video.name}  [{total} frames @ {fps:.0f}fps]")
        print(f"  sampling {len(indices)} frames ({mode}) → {out_dir}/")
        saved = _extract(video, indices, out_dir)
        print(f"  saved {saved} JPEGs")

    print(f"\nDone. Upload {args.out_dir}/ to a Roboflow Classification project.")
    print("Label each image as 'live' or 'dead'.")
    print("LIVE = from serve toss until point ends. DEAD = everything else.")


if __name__ == "__main__":
    main()
