"""Generate comparison figures between two detection runs (e.g. phase 0 vs phase 1).

Produces:
  1. Detection timeline — side-by-side detected/missed bars over time
  2. Confidence distribution — histogram overlay
  3. Ball size distribution — width in pixels
  4. Rally segmentation comparison — segments found by each detector
  5. Summary stats table saved as PNG

Usage:
  uv run python scripts/compare_phases.py \
      --csv-a outputs/clips/detections.csv \
      --csv-b outputs/clips/detections_finetuned.csv \
      --label-a "Phase 0 (YOLOv8x COCO)" \
      --label-b "Phase 1 (YOLOv8n fine-tuned)" \
      --video outputs/clips/test_2min.mp4 \
      --out-dir outputs/analysis/phase_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from volleybot.detection import load_csv, filter_detections, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping

FILTER_CONF = 0.25
FILTER_MAX_PX = 80
FILTER_MAX_Y = 0.72
DEAD_GAP_S = 3.0
PRE_ROLL_S = 1.5
POST_ROLL_S = 2.0

_A_COLOR = "#2196F3"   # blue
_B_COLOR = "#FF5722"   # orange-red


def _load_and_filter(csv: Path, fps: float, frame_height: int):
    raw = load_csv(csv)
    filtered = filter_detections(
        raw, min_conf=FILTER_CONF, max_ball_px=FILTER_MAX_PX,
        max_y_frac=FILTER_MAX_Y, frame_height=frame_height,
    )
    mask = smoothed_mask(filtered, fps, fill_gap_s=0.5)
    return raw, filtered, mask


def plot_timeline(ax, mask_a, mask_b, fps, label_a, label_b):
    times = np.arange(len(mask_a)) / fps
    ax.fill_between(times, 1.15, 2.0, where=mask_a, color=_A_COLOR, alpha=0.85, linewidth=0)
    ax.fill_between(times, 0.15, 1.0, where=mask_b, color=_B_COLOR, alpha=0.85, linewidth=0)
    ax.set_yticks([0.575, 1.575])
    ax.set_yticklabels([label_b, label_a], fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_title("Detection timeline (smoothed, filtered)")
    ax.set_xlim(0, times[-1])
    ax.set_ylim(0, 2.15)
    ax.grid(axis="x", alpha=0.3)


def plot_confidence(ax, raw_a, raw_b, label_a, label_b):
    confs_a = [d.conf for d in raw_a if d.detected and d.conf is not None]
    confs_b = [d.conf for d in raw_b if d.detected and d.conf is not None]
    bins = np.linspace(0, 1, 30)
    ax.hist(confs_a, bins=bins, alpha=0.6, color=_A_COLOR, label=label_a, density=True)
    ax.hist(confs_b, bins=bins, alpha=0.6, color=_B_COLOR, label=label_b, density=True)
    ax.axvline(FILTER_CONF, color="gray", linestyle="--", linewidth=1, label=f"filter threshold ({FILTER_CONF})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distribution (raw detections)")
    ax.legend(fontsize=8)


def plot_ball_size(ax, raw_a, raw_b, label_a, label_b):
    def widths(dets):
        return [d.x2 - d.x1 for d in dets
                if d.detected and d.x1 is not None and (d.x2 - d.x1) < 200]
    wa, wb = widths(raw_a), widths(raw_b)
    bins = np.linspace(0, 150, 40)
    ax.hist(wa, bins=bins, alpha=0.6, color=_A_COLOR, label=label_a, density=True)
    ax.hist(wb, bins=bins, alpha=0.6, color=_B_COLOR, label=label_b, density=True)
    ax.axvline(FILTER_MAX_PX, color="gray", linestyle="--", linewidth=1,
               label=f"max filter ({FILTER_MAX_PX}px)")
    ax.set_xlabel("Detected width (px)")
    ax.set_ylabel("Density")
    ax.set_title("Ball bounding-box width distribution")
    ax.legend(fontsize=8)


def plot_segments(ax, mask_a, mask_b, fps, duration_s, label_a, label_b):
    def segs(mask):
        s = detect_rallies(mask, fps, dead_gap_s=DEAD_GAP_S,
                           pre_roll_s=PRE_ROLL_S, post_roll_s=POST_ROLL_S,
                           total_duration_s=duration_s)
        return merge_overlapping(s)

    segs_a = segs(mask_a)
    segs_b = segs(mask_b)

    for seg in segs_a:
        ax.barh(1, seg.end_s - seg.start_s, left=seg.start_s,
                height=0.6, color=_A_COLOR, alpha=0.85)
    for seg in segs_b:
        ax.barh(0, seg.end_s - seg.start_s, left=seg.start_s,
                height=0.6, color=_B_COLOR, alpha=0.85)

    play_a = sum(s.duration_s for s in segs_a)
    play_b = sum(s.duration_s for s in segs_b)

    ax.set_yticks([0, 1])
    ax.set_yticklabels([
        f"{label_b}\n{len(segs_b)} segs, {play_b:.0f}s",
        f"{label_a}\n{len(segs_a)} segs, {play_a:.0f}s",
    ], fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, duration_s)
    ax.set_title("Rally segments detected")
    ax.grid(axis="x", alpha=0.3)


def plot_summary_table(ax, raw_a, filtered_a, mask_a,
                       raw_b, filtered_b, mask_b,
                       fps, duration_s, label_a, label_b):
    def stats(raw, filtered, mask):
        total = len(raw)
        raw_det = sum(d.detected for d in raw)
        filt_det = sum(d.detected for d in filtered)
        smooth_det = mask.sum()
        segs = merge_overlapping(detect_rallies(
            mask, fps, dead_gap_s=DEAD_GAP_S,
            pre_roll_s=PRE_ROLL_S, post_roll_s=POST_ROLL_S,
            total_duration_s=duration_s,
        ))
        play_s = sum(s.duration_s for s in segs)
        return [
            f"{raw_det}/{total} ({raw_det/total*100:.1f}%)",
            f"{filt_det}/{total} ({filt_det/total*100:.1f}%)",
            f"{smooth_det}/{total} ({smooth_det/total*100:.1f}%)",
            str(len(segs)),
            f"{play_s:.0f}s / {duration_s:.0f}s ({play_s/duration_s*100:.0f}%)",
        ]

    rows = ["Raw detections", "After filtering", "After gap-fill", "Rally segments", "Play time"]
    col_a = stats(raw_a, filtered_a, mask_a)
    col_b = stats(raw_b, filtered_b, mask_b)

    ax.axis("off")
    table = ax.table(
        cellText=[[r, a, b] for r, a, b in zip(rows, col_a, col_b)],
        colLabels=["Metric", label_a, label_b],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDDDDD")
        elif col == 1:
            cell.set_facecolor("#E3F2FD")
        elif col == 2:
            cell.set_facecolor("#FBE9E7")
    ax.set_title("Summary comparison", fontweight="bold", pad=12)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-a", type=Path, required=True)
    p.add_argument("--csv-b", type=Path, required=True)
    p.add_argument("--label-a", default="Phase 0 (YOLOv8x COCO)")
    p.add_argument("--label-b", default="Phase 1 (YOLOv8n fine-tuned)")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/phase_comparison"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_s = total / fps

    print(f"video: {args.video.name}  {total} frames @ {fps:.0f}fps  ({duration_s:.1f}s)")

    raw_a, filtered_a, mask_a = _load_and_filter(args.csv_a, fps, H)
    raw_b, filtered_b, mask_b = _load_and_filter(args.csv_b, fps, H)

    # --- Figure 1: timeline + confidence + size + segments ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"{args.label_a}  vs  {args.label_b}", fontsize=12, fontweight="bold")
    plt.subplots_adjust(hspace=0.55)

    plot_timeline(axes[0], mask_a, mask_b, fps, args.label_a, args.label_b)
    plot_confidence(axes[1], raw_a, raw_b, args.label_a, args.label_b)
    plot_ball_size(axes[2], raw_a, raw_b, args.label_a, args.label_b)
    plot_segments(axes[3], mask_a, mask_b, fps, duration_s, args.label_a, args.label_b)

    out1 = args.out_dir / "detection_comparison.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out1}")

    # --- Figure 2: summary table ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_summary_table(ax2, raw_a, filtered_a, mask_a,
                       raw_b, filtered_b, mask_b,
                       fps, duration_s, args.label_a, args.label_b)
    out2 = args.out_dir / "summary_table.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved: {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
