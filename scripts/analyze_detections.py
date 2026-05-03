"""Analyze ball detection results and produce visualizations + rally segments.

Reads the CSV produced by yolo_ball_detect*.py and outputs:
  1. A detection timeline plot (heatmap + rolling recall)
  2. Sample frames at detected / missed moments
  3. Rally segments using the state machine
  4. A summary stats printout

Usage: uv run python scripts/analyze_detections.py [--csv PATH] [--video PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from volleybot.detection import load_csv, detection_mask, smoothed_mask
from volleybot.segmentation import detect_rallies, merge_overlapping

DEFAULT_CSV = Path("outputs/clips/detections.csv")
DEFAULT_VIDEO = Path("outputs/clips/test_2min.mp4")
OUT_DIR = Path("outputs/analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--samples", type=int, default=12,
                   help="number of sample frames to extract (detected + missed)")
    return p.parse_args()


def plot_timeline(detections, fps: float, out_dir: Path) -> None:
    mask = detection_mask(detections)
    times = np.array([d.time_s for d in detections])
    duration = times[-1]

    # rolling recall over 5-second windows
    window = int(5 * fps)
    rolling = np.convolve(mask.astype(float),
                          np.ones(window) / window, mode="same")

    smoothed = smoothed_mask(detections, fps, fill_gap_s=0.5)

    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)

    # raw detection strip
    axes[0].imshow(mask[np.newaxis, :], aspect="auto",
                   extent=[0, duration, 0, 1], cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_yticks([])
    axes[0].set_title("raw ball detection (green=detected, red=missed)")

    # smoothed strip
    axes[1].imshow(smoothed[np.newaxis, :], aspect="auto",
                   extent=[0, duration, 0, 1], cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_yticks([])
    axes[1].set_title("smoothed detection (0.5s gap fill)")

    # rolling recall
    axes[2].fill_between(times, rolling, alpha=0.7, color="steelblue")
    axes[2].axhline(0.5, color="orange", linestyle="--", linewidth=1, label="50%")
    axes[2].set_ylabel("recall (5s window)")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("5-second rolling detection rate")

    fig.tight_layout()
    out_path = out_dir / "detection_timeline.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_rally_segments(detections, fps: float, out_dir: Path) -> list:
    smoothed = smoothed_mask(detections, fps, fill_gap_s=0.5)
    duration = detections[-1].time_s

    segments = detect_rallies(smoothed, fps, total_duration_s=duration)
    segments = merge_overlapping(segments)

    times = np.array([d.time_s for d in detections])

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.imshow(smoothed[np.newaxis, :], aspect="auto",
              extent=[0, duration, 0, 1], cmap="Greys", vmin=0, vmax=1, alpha=0.3)

    for seg in segments:
        ax.axvspan(seg.start_s, seg.end_s, alpha=0.35, color="steelblue")
        ax.text((seg.start_s + seg.end_s) / 2, 0.6,
                f"{seg.duration_s:.0f}s", ha="center", va="center",
                fontsize=8, color="navy")

    rally_patch = mpatches.Patch(color="steelblue", alpha=0.5, label="rally segment (padded)")
    ax.legend(handles=[rally_patch], loc="upper right")
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"detected rally segments ({len(segments)} rallies, "
                 f"total play time = {sum(s.duration_s for s in segments):.0f}s / {duration:.0f}s)")

    out_path = out_dir / "rally_segments.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")
    return segments


def extract_sample_frames(detections, video_path: Path, n: int, out_dir: Path) -> None:
    detected = [d for d in detections if d.detected]
    missed = [d for d in detections if not d.detected]

    half = n // 2
    sampled_det = detected[::max(1, len(detected) // half)][:half]
    sampled_miss = missed[::max(1, len(missed) // half)][:half]
    samples = sorted(sampled_det + sampled_miss, key=lambda d: d.frame)

    cap = cv2.VideoCapture(str(video_path))
    frames_dir = out_dir / "sample_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for det in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame)
        ret, frame = cap.read()
        if not ret:
            continue

        label = "DETECTED" if det.detected else "MISSED"
        color = (0, 200, 0) if det.detected else (0, 0, 220)
        cv2.putText(frame, f"t={det.time_s:.1f}s  {label}",
                    (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        if det.detected and det.x1 is not None:
            cx = (det.x1 + det.x2) // 2
            cy = (det.y1 + det.y2) // 2
            r = max(det.x2 - det.x1, det.y2 - det.y1) // 2
            cv2.circle(frame, (cx, cy), r + 6, (0, 255, 0), 3)
            cv2.putText(frame, f"conf={det.conf:.2f}", (det.x1, max(det.y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        fname = frames_dir / f"frame_{det.frame:05d}_{label.lower()}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cap.release()
    print(f"saved {len(samples)} sample frames to {frames_dir}")


def print_stats(detections, fps: float) -> None:
    mask = detection_mask(detections)
    smoothed = smoothed_mask(detections, fps, fill_gap_s=0.5)
    duration = detections[-1].time_s

    confs = [d.conf for d in detections if d.detected and d.conf]
    sizes = [(d.x2 - d.x1) for d in detections
             if d.detected and d.x1 is not None]

    segments = detect_rallies(smoothed, fps, total_duration_s=duration)
    segments = merge_overlapping(segments)
    total_play = sum(s.duration_s for s in segments)

    print("\n=== detection analysis ===")
    print(f"clip duration        : {duration:.1f}s ({duration/60:.1f} min)")
    print(f"total frames         : {len(detections)}")
    print(f"frames detected      : {mask.sum()} ({mask.mean()*100:.1f}%)")
    print(f"frames detected (sm) : {smoothed.sum()} ({smoothed.mean()*100:.1f}%)")
    if confs:
        print(f"confidence           : mean={np.mean(confs):.3f} "
              f"min={np.min(confs):.3f} max={np.max(confs):.3f}")
    if sizes:
        print(f"detected ball width  : mean={np.mean(sizes):.0f}px "
              f"min={np.min(sizes)}px max={np.max(sizes)}px")
    print(f"\nrally segments found : {len(segments)}")
    print(f"total rally time     : {total_play:.0f}s ({total_play/duration*100:.0f}% of clip)")
    if segments:
        durs = [s.duration_s for s in segments]
        print(f"rally duration       : mean={np.mean(durs):.0f}s "
              f"min={np.min(durs):.0f}s max={np.max(durs):.0f}s")
        print("\nindividual rally segments:")
        for i, s in enumerate(segments, 1):
            print(f"  {i:2d}. {s.start_s:6.1f}s – {s.end_s:6.1f}s  "
                  f"({s.duration_s:.0f}s)  raw: {s.raw_start_s:.1f}–{s.raw_end_s:.1f}s")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    detections = load_csv(args.csv)
    if not detections:
        print("no detections loaded")
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"loaded {len(detections)} frames from {args.csv}, fps={fps:.2f}")
    print_stats(detections, fps)
    plot_timeline(detections, fps, OUT_DIR)
    segments = plot_rally_segments(detections, fps, OUT_DIR)
    extract_sample_frames(detections, args.video, args.samples, OUT_DIR)
    print(f"\nall outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
