"""Compare YOLO and bg-subtract detectors side by side.

Produces a single summary plot with:
- YOLO raw detection timeline
- BG-subtract velocity-filtered timeline
- Hybrid (union of both) timeline
- Rally segments from each

Usage: uv run python scripts/compare_detectors.py
"""

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from volleybot.detection import load_csv, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping
from volleybot.tracking import apply_velocity_filter, apply_arc_filter

YOLO_CSV = Path("outputs/clips/detections.csv")
BG_CSV = Path("outputs/clips/detections_bg.csv")
VIDEO = Path("outputs/clips/test_2min.mp4")
OUT_DIR = Path("outputs/analysis")


def get_mask_and_segments(dets, fps: float, duration: float, label: str):
    mask = smoothed_mask(dets, fps, fill_gap_s=0.5)
    segs = merge_overlapping(
        detect_rallies(mask, fps, dead_gap_s=3.0, min_rally_s=1.5,
                       pre_roll_s=1.5, post_roll_s=2.0,
                       total_duration_s=duration))
    total_play = sum(s.duration_s for s in segs)
    print(f"\n{label}: {sum(d.detected for d in dets)/len(dets)*100:.1f}% detection → "
          f"{len(segs)} segments, {total_play:.0f}s play time")
    for i, s in enumerate(segs, 1):
        print(f"  {i}. {s.start_s:.1f}–{s.end_s:.1f}s ({s.duration_s:.0f}s)")
    return mask, segs


def plot_comparison(masks_segs: list, labels: list, colors: list,
                    fps: float, duration: float) -> None:
    n = len(masks_segs)
    fig, axes = plt.subplots(n * 2, 1, figsize=(16, n * 3), sharex=True)

    for i, ((mask, segs), label, color) in enumerate(zip(masks_segs, labels, colors)):
        times = np.linspace(0, duration, len(mask))
        ax_strip = axes[i * 2]
        ax_seg = axes[i * 2 + 1]

        ax_strip.imshow(mask[np.newaxis, :], aspect="auto",
                        extent=[0, duration, 0, 1],
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax_strip.set_yticks([])
        ax_strip.set_title(f"{label} — raw detection timeline", fontsize=10)

        for seg in segs:
            ax_seg.axvspan(seg.start_s, seg.end_s, alpha=0.4, color=color)
            ax_seg.text((seg.start_s + seg.end_s) / 2, 0.5,
                        f"{seg.duration_s:.0f}s", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        ax_seg.set_ylim(0, 1)
        ax_seg.set_yticks([])
        ax_seg.set_title(f"{label} — segments ({len(segs)} rallies)", fontsize=10)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out = OUT_DIR / "detector_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nsaved {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = 120.0

    masks_segs = []
    labels = []
    colors = []

    if YOLO_CSV.exists():
        yolo_dets = load_csv(YOLO_CSV)
        mask_yolo, segs_yolo = get_mask_and_segments(
            yolo_dets, fps, duration, "YOLO (raw)")
        masks_segs.append((mask_yolo, segs_yolo))
        labels.append("YOLOv8x")
        colors.append("steelblue")
    else:
        print(f"YOLO CSV not found: {YOLO_CSV}")

    if BG_CSV.exists():
        bg_dets = load_csv(BG_CSV)
        vel_dets = apply_velocity_filter(bg_dets, max_jump_px=90)
        arc_dets = apply_arc_filter(vel_dets, window=30, outlier_px=80)
        mask_bg, segs_bg = get_mask_and_segments(
            arc_dets, fps, duration, "BG-subtract (velocity+arc filtered)")
        masks_segs.append((mask_bg, segs_bg))
        labels.append("BG-subtract (filtered)")
        colors.append("darkorange")
    else:
        print(f"BG CSV not found: {BG_CSV}")

    if len(masks_segs) == 2:
        # hybrid: union of both smoothed masks, then re-segment
        hybrid_mask = masks_segs[0][0] | masks_segs[1][0]
        segs_hybrid = merge_overlapping(
            detect_rallies(hybrid_mask, fps, dead_gap_s=3.0, min_rally_s=1.5,
                           pre_roll_s=1.5, post_roll_s=2.0, total_duration_s=duration))
        total_play = sum(s.duration_s for s in segs_hybrid)
        print(f"\nHybrid (union): {len(segs_hybrid)} segments, {total_play:.0f}s play time")
        for i, s in enumerate(segs_hybrid, 1):
            print(f"  {i}. {s.start_s:.1f}–{s.end_s:.1f}s ({s.duration_s:.0f}s)")
        masks_segs.append((hybrid_mask, segs_hybrid))
        labels.append("Hybrid (YOLO ∪ BG-filtered)")
        colors.append("seagreen")

    plot_comparison(masks_segs, labels, colors, fps, duration)


if __name__ == "__main__":
    main()
