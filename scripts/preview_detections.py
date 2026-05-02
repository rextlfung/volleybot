"""Overlay YOLO detections on sampled frames for visual QA before Roboflow upload.

Reads the frames already extracted to outputs/labeling/<stem>/ and annotates
each one with the corresponding YOLO bounding box (if any), confidence score,
and a colour-coded border (green = detected, red = missed).

Usage:
  uv run python scripts/preview_detections.py \
      --frames-dir outputs/labeling/20220805g1 \
      --csv outputs/clips/detections.csv \
      --out-dir outputs/labeling/20220805g1_preview

Output: one annotated JPEG per frame in <out-dir>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from volleybot.detection import load_csv, filter_detections, Detection


_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 200, 220)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _frame_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _annotate(
    img,
    raw: Detection | None,
    filtered: Detection | None,
) -> None:
    h, w = img.shape[:2]
    border = 6

    if filtered and filtered.detected:
        # Confirmed detection — green border + box
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), _GREEN, border)
        x1, y1, x2, y2 = filtered.x1, filtered.y1, filtered.x2, filtered.y2
        cv2.rectangle(img, (x1, y1), (x2, y2), _GREEN, 2)
        label = f"ball {filtered.conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 6), _FONT, 0.7, _GREEN, 2)
    elif raw and raw.detected:
        # Filtered out — yellow border + dashed box annotation
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), _YELLOW, border)
        x1, y1, x2, y2 = raw.x1, raw.y1, raw.x2, raw.y2
        cv2.rectangle(img, (x1, y1), (x2, y2), _YELLOW, 2)
        label = f"FILTERED {raw.conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 6), _FONT, 0.7, _YELLOW, 2)
    else:
        # No detection — red border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), _RED, border)

    # Frame index in top-left
    fi = raw.frame if raw else "?"
    cv2.putText(img, f"frame {fi}", (10, 28), _FONT, 0.7, (255, 255, 255), 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate sampled frames with YOLO detections for QA."
    )
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="directory of frame_XXXXXX.jpg files")
    p.add_argument("--csv", type=Path, required=True,
                   help="YOLO detections CSV")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: <frames-dir>_preview)")
    p.add_argument("--filter-conf", type=float, default=0.25)
    p.add_argument("--filter-max-px", type=int, default=80)
    p.add_argument("--filter-max-y", type=float, default=0.72)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path(str(args.frames_dir) + "_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup by frame index
    raw_dets = {d.frame: d for d in load_csv(args.csv)}

    first_img = next(args.frames_dir.glob("frame_*.jpg"), None)
    if first_img is None:
        print(f"no frames found in {args.frames_dir}")
        return
    h = cv2.imread(str(first_img)).shape[0]

    filtered_list = filter_detections(
        list(raw_dets.values()),
        min_conf=args.filter_conf,
        max_ball_px=args.filter_max_px,
        max_y_frac=args.filter_max_y,
        frame_height=h,
    )
    filtered_dets = {d.frame: d for d in filtered_list}

    frame_paths = sorted(args.frames_dir.glob("frame_*.jpg"))
    n_detected = n_filtered = n_missed = 0

    for path in frame_paths:
        fi = _frame_index(path)
        raw = raw_dets.get(fi)
        filt = filtered_dets.get(fi)

        img = cv2.imread(str(path))
        _annotate(img, raw, filt)
        cv2.imwrite(str(out_dir / path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        if filt and filt.detected:
            n_detected += 1
        elif raw and raw.detected:
            n_filtered += 1
        else:
            n_missed += 1

    total = len(frame_paths)
    print(f"annotated {total} frames → {out_dir}/")
    print(f"  green  (kept):     {n_detected:3d} ({n_detected/total*100:.0f}%)")
    print(f"  yellow (filtered): {n_filtered:3d} ({n_filtered/total*100:.0f}%)")
    print(f"  red    (missed):   {n_missed:3d}  ({n_missed/total*100:.0f}%)")


if __name__ == "__main__":
    main()
