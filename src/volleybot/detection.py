"""Load and work with per-frame ball detection results from YOLO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import csv
import numpy as np


@dataclass
class Detection:
    frame: int
    time_s: float
    detected: bool
    conf: float | None
    x1: int | None
    y1: int | None
    x2: int | None
    y2: int | None

    @property
    def bbox_area(self) -> int | None:
        if self.x1 is None:
            return None
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int] | None:
        if self.x1 is None:
            return None
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


def load_csv(path: Path) -> list[Detection]:
    """Load a detections CSV produced by yolo_ball_detect*.py."""
    detections: list[Detection] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            det = Detection(
                frame=int(row["frame"]),
                time_s=float(row["time_s"]),
                detected=bool(int(row["detected"])),
                conf=float(row["conf"]) if row["conf"] else None,
                x1=int(row["x1"]) if row["x1"] else None,
                y1=int(row["y1"]) if row["y1"] else None,
                x2=int(row["x2"]) if row["x2"] else None,
                y2=int(row["y2"]) if row["y2"] else None,
            )
            detections.append(det)
    return detections


def filter_detections(
    detections: list[Detection],
    *,
    min_conf: float = 0.30,
    min_ball_px: int = 12,
    max_ball_px: int = 100,
    max_y_frac: float = 0.80,
    frame_height: int = 1080,
) -> list[Detection]:
    """Clear detections that are implausible for a volleyball.

    Filters applied:
    - conf < min_conf: too uncertain
    - ball wider than max_ball_px or smaller than min_ball_px: wrong size
    - ball center below max_y_frac * frame_height: on the floor (false positive)

    Cleared detections are returned as detected=False with no bbox.

    Args:
        min_conf: minimum YOLO confidence to keep.
        min_ball_px: minimum ball bounding-box width in pixels.
        max_ball_px: maximum ball bounding-box width in pixels.
        max_y_frac: ball center must be above this fraction of frame height.
        frame_height: full frame height in pixels (default 1080).
    """
    result: list[Detection] = []
    max_cy = int(max_y_frac * frame_height)

    for det in detections:
        if not det.detected:
            result.append(det)
            continue

        w = (det.x2 - det.x1) if det.x1 is not None else 0
        cy = det.center[1] if det.center else 0
        conf = det.conf or 0.0

        if conf < min_conf or w < min_ball_px or w > max_ball_px or cy > max_cy:
            result.append(Detection(
                frame=det.frame, time_s=det.time_s,
                detected=False, conf=None,
                x1=None, y1=None, x2=None, y2=None,
            ))
        else:
            result.append(det)

    return result


def detection_mask(detections: list[Detection]) -> np.ndarray:
    """Return a boolean array (one entry per frame) of whether ball was detected."""
    return np.array([d.detected for d in detections], dtype=bool)


def smoothed_mask(
    detections: list[Detection],
    fps: float,
    fill_gap_s: float = 0.5,
) -> np.ndarray:
    """Fill short detection gaps (≤ fill_gap_s) to smooth over missed frames.

    A ball moving at 20 m/s can cross a frame in <2 frames at 60fps, so
    isolated single-frame misses are almost always false negatives.
    """
    mask = detection_mask(detections).copy()
    gap_frames = int(fill_gap_s * fps)
    seen_detection = False
    i = 0
    while i < len(mask):
        if not mask[i]:
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            # Only fill gaps between two detected windows, never leading gaps.
            if seen_detection and j < len(mask) and (j - i) <= gap_frames:
                mask[i:j] = True
            i = j
        else:
            seen_detection = True
            i += 1
    return mask
