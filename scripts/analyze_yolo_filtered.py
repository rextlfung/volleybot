"""Apply position/size/confidence filters to YOLO detections and compare results.

Usage: uv run python scripts/analyze_yolo_filtered.py
"""

from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from volleybot.detection import load_csv, filter_detections, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping

YOLO_CSV = Path("outputs/clips/detections.csv")
VIDEO = Path("outputs/clips/test_2min.mp4")


def report(label: str, dets, fps: float, duration: float) -> None:
    n_det = sum(d.detected for d in dets)
    mask = smoothed_mask(dets, fps, fill_gap_s=0.5)
    segs = merge_overlapping(
        detect_rallies(mask, fps, dead_gap_s=3.0, min_rally_s=1.5,
                       pre_roll_s=1.5, post_roll_s=2.0,
                       total_duration_s=duration))
    total_play = sum(s.duration_s for s in segs)
    print(f"\n{label}")
    print(f"  detected frames : {n_det}/{len(dets)} ({n_det/len(dets)*100:.1f}%)")
    print(f"  (smoothed)      : {mask.sum()} ({mask.mean()*100:.1f}%)")
    print(f"  rally segments  : {len(segs)}, total play = {total_play:.0f}s / {duration:.0f}s")
    for i, s in enumerate(segs, 1):
        print(f"    {i}. {s.start_s:.1f}–{s.end_s:.1f}s ({s.duration_s:.0f}s)")


def main() -> None:
    dets = load_csv(YOLO_CSV)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = dets[-1].time_s

    report("Raw YOLO (conf≥0.15, no size/pos filter)", dets, fps, duration)

    f1 = filter_detections(dets, min_conf=0.30, min_ball_px=12,
                           max_ball_px=100, max_y_frac=0.80)
    report("conf≥0.30, width 12-100px, cy<80% frame", f1, fps, duration)

    f2 = filter_detections(dets, min_conf=0.40, min_ball_px=15,
                           max_ball_px=80, max_y_frac=0.72)
    report("conf≥0.40, width 15-80px, cy<72% frame", f2, fps, duration)

    f3 = filter_detections(dets, min_conf=0.25, min_ball_px=15,
                           max_ball_px=80, max_y_frac=0.72)
    report("conf≥0.25, width 15-80px, cy<72% frame", f3, fps, duration)


if __name__ == "__main__":
    main()
