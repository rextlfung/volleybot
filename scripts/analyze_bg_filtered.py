"""Quick script: apply velocity filter to bg-subtract detections and re-segment.

Compares raw vs. filtered rally detection on the test clip.

Usage: uv run python scripts/analyze_bg_filtered.py
"""

from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from volleybot.detection import load_csv, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping
from volleybot.tracking import apply_velocity_filter, apply_arc_filter

CSV = Path("outputs/clips/detections_bg.csv")
VIDEO = Path("outputs/clips/test_2min.mp4")


def main() -> None:
    detections = load_csv(CSV)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration_s = detections[-1].time_s

    def report(label: str, dets) -> None:
        mask = smoothed_mask(dets, fps, fill_gap_s=0.5)
        segs = merge_overlapping(
            detect_rallies(mask, fps, dead_gap_s=3.0, min_rally_s=1.5,
                           pre_roll_s=0.0, post_roll_s=0.0,
                           total_duration_s=duration_s))
        n_detected = sum(d.detected for d in dets)
        total_play = sum(s.duration_s for s in segs)
        print(f"\n{label}")
        print(f"  detected frames  : {n_detected}/{len(dets)} ({n_detected/len(dets)*100:.1f}%)")
        print(f"  rally segments   : {len(segs)}")
        print(f"  total rally time : {total_play:.0f}s / {duration_s:.0f}s")
        if segs:
            for i, s in enumerate(segs, 1):
                print(f"    {i}. {s.start_s:.1f}s – {s.end_s:.1f}s ({s.duration_s:.0f}s)")

    report("raw bg-subtract", detections)

    vel_filtered = apply_velocity_filter(detections, max_jump_px=90)
    report("after velocity filter", vel_filtered)

    arc_filtered = apply_arc_filter(vel_filtered, window=30, outlier_px=80)
    report("after arc filter", arc_filtered)


if __name__ == "__main__":
    main()
