"""Per-frame live/dead play-state classification using a fine-tuned YOLOv8-cls model.

Produces a CSV of live/dead predictions for every frame, used by cut_rallies.py
for rally segmentation. The ball detector runs separately and its CSV is preserved
for future use (highlight reels, statistics).

Usage:
  uv run python scripts/classify_frames.py \
      --input data/20220805g1.mp4 \
      --model runs/classify/volleybot_cls/weights/best.pt \
      --device cpu

  # With annotated output video (optional, slower):
  uv run python scripts/classify_frames.py \
      --input data/20220805g1.mp4 \
      --model runs/classify/volleybot_cls/weights/best.pt \
      --out-video outputs/20220805g1/classified.mp4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from volleybot.device import best_device

_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify each video frame as live or dead play."
    )
    p.add_argument("--input", type=Path, required=True, help="source video")
    p.add_argument("--model", type=Path, required=True,
                   help="YOLOv8-cls weights (e.g. runs/classify/volleybot_cls/weights/best.pt)")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="output CSV path (default: outputs/<stem>/classification.csv)")
    p.add_argument("--out-video", type=Path, default=None,
                   help="optional annotated output video")
    p.add_argument("--device", default=best_device(), help="cpu | cuda | mps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}")
        return

    out_csv = args.out_csv or (
        Path("outputs") / args.input.stem / "classification.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"{args.input.name}: {W}x{H} @ {fps:.2f}fps, {total_frames} frames")

    model = YOLO(str(args.model))

    # Resolve which class index corresponds to "live".
    # Roboflow sorts alphabetically: dead=0, live=1.
    name_to_idx = {v: k for k, v in model.names.items()}
    live_idx = name_to_idx.get("live", 1)
    dead_idx = name_to_idx.get("dead", 0)
    print(f"class mapping: live={live_idx}, dead={dead_idx}  (model.names={model.names})")

    writer = None
    if args.out_video:
        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (W, H))

    rows: list[dict] = []
    n_live = 0
    t0 = time.time()

    for result in model.predict(
        source=str(args.input),
        device=args.device,
        stream=True,
        verbose=False,
    ):
        frame_idx = len(rows)
        time_s = frame_idx / fps
        probs = result.probs

        conf_live = float(probs.data[live_idx])
        conf_dead = float(probs.data[dead_idx])
        live = conf_live >= conf_dead

        rows.append({
            "frame": frame_idx,
            "time_s": f"{time_s:.3f}",
            "live": int(live),
            "conf_live": f"{conf_live:.4f}",
            "conf_dead": f"{conf_dead:.4f}",
        })
        if live:
            n_live += 1

        if writer is not None:
            frame = result.orig_img.copy()
            label = f"LIVE {conf_live:.2f}" if live else f"dead {conf_dead:.2f}"
            color = _GREEN if live else _RED
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), color, 6)
            cv2.putText(frame, label, (16, 44), _FONT, 1.2, color, 3)
            writer.write(frame)

        if frame_idx % 600 == 0 and frame_idx > 0:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_proc
            print(f"  frame {frame_idx}/{total_frames}  {fps_proc:.1f}fps  "
                  f"live so far: {n_live/frame_idx*100:.0f}%  ETA: {eta:.0f}s")

    if writer:
        writer.release()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "live", "conf_live", "conf_dead"])
        w.writeheader()
        w.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n=== results ===")
    print(f"frames processed : {len(rows)}")
    print(f"live             : {n_live} ({n_live/len(rows)*100:.1f}%)")
    print(f"dead             : {len(rows)-n_live} ({(len(rows)-n_live)/len(rows)*100:.1f}%)")
    print(f"total time       : {elapsed:.0f}s ({len(rows)/elapsed:.1f} fps)")
    print(f"output csv       : {out_csv}")


if __name__ == "__main__":
    main()
