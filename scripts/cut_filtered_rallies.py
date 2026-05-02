"""Cut rallies using filtered YOLO detections (conf/size/position filtered).

Applies filter_detections() before segmenting to reduce false positives.
Produces outputs/clips/rallies_filtered/ and a concatenated highlight reel.

Usage: uv run python scripts/cut_filtered_rallies.py
"""

from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from volleybot.detection import load_csv, filter_detections, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping
from volleybot.cutter import cut_segments, concat_segments

YOLO_CSV = Path("outputs/clips/detections.csv")
VIDEO = Path("outputs/clips/test_2min.mp4")
OUT_DIR = Path("outputs/clips/rallies_filtered")
CONCAT_OUT = Path("outputs/clips/test_2min_filtered_rallies.mp4")


def main() -> None:
    dets = load_csv(YOLO_CSV)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total_frames / fps

    # Apply position + size + confidence filter
    filtered = filter_detections(
        dets,
        min_conf=0.25,
        min_ball_px=15,
        max_ball_px=80,
        max_y_frac=0.72,
        frame_height=1080,
    )

    n_raw = sum(d.detected for d in dets)
    n_filt = sum(d.detected for d in filtered)
    print(f"raw detections    : {n_raw} ({n_raw/len(dets)*100:.1f}%)")
    print(f"after filter      : {n_filt} ({n_filt/len(filtered)*100:.1f}%)")

    mask = smoothed_mask(filtered, fps, fill_gap_s=0.5)
    segments = merge_overlapping(
        detect_rallies(mask, fps, dead_gap_s=3.0, min_rally_s=1.5,
                       pre_roll_s=1.5, post_roll_s=2.0,
                       total_duration_s=duration_s))

    total_play = sum(s.duration_s for s in segments)
    print(f"\nrally segments    : {len(segments)}")
    print(f"total play time   : {total_play:.0f}s / {duration_s:.0f}s "
          f"({total_play/duration_s*100:.0f}%)")
    for i, s in enumerate(segments, 1):
        print(f"  {i}. {s.start_s:.1f}–{s.end_s:.1f}s ({s.duration_s:.0f}s)")

    print(f"\ncutting to {OUT_DIR} ...")
    clips = cut_segments(VIDEO, segments, OUT_DIR, prefix="test_filtered")

    if clips:
        print(f"\nconcatenating → {CONCAT_OUT} ...")
        concat_segments(clips, CONCAT_OUT)
        print(f"done: {CONCAT_OUT}")


if __name__ == "__main__":
    main()
