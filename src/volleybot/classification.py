"""Per-frame live/dead play-state classification results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from volleybot.detection import fill_short_gaps


@dataclass
class FrameClassification:
    frame: int
    time_s: float
    live: bool
    conf_live: float
    conf_dead: float


def load_classification_csv(path: Path) -> list[FrameClassification]:
    """Load a classification CSV produced by classify_frames.py."""
    results: list[FrameClassification] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            results.append(FrameClassification(
                frame=int(row["frame"]),
                time_s=float(row["time_s"]),
                live=bool(int(row["live"])),
                conf_live=float(row["conf_live"]),
                conf_dead=float(row["conf_dead"]),
            ))
    return results


def classification_mask(
    results: list[FrameClassification],
    fps: float,
    fill_gap_s: float = 0.5,
    min_conf: float = 0.0,
) -> np.ndarray:
    """Convert per-frame classifications to a boolean live/dead mask.

    Short dead gaps (≤ fill_gap_s) within a live window are filled to avoid
    single-frame classifier noise breaking a rally into two segments.

    Args:
        results: per-frame classification results.
        fps: video frame rate (used to convert fill_gap_s to frames).
        fill_gap_s: fill dead gaps shorter than this within a live window.
        min_conf: only count a frame as live if conf_live ≥ this value.
    """
    mask = np.array(
        [r.live and r.conf_live >= min_conf for r in results], dtype=bool
    )
    return fill_short_gaps(mask, int(fill_gap_s * fps))
