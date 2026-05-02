"""Ball trajectory consistency filtering.

The bg subtract detector has high recall but also high false positive rate
(player arms, heads). This module filters detections using physics-based
trajectory plausibility:

1. Max velocity filter: real ball can't teleport >MAX_JUMP_PX pixels between
   consecutive frames at 60fps. Detections that jump too far are suppressed.

2. Trajectory smoothing: fit a short parabolic arc through recent detections
   and suppress outliers (>OUTLIER_THRESHOLD px from fitted arc).

Both filters can be applied to any detector's output (YOLO or bg-subtract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from volleybot.detection import Detection


# Maximum plausible displacement per frame at 60fps.
# Volleyball spike at ~30 m/s, field ~12m wide → ~90px/m.
# 30 m/s / 60 fps * 90 px/m ≈ 45 px/frame. Use 2× margin.
MAX_JUMP_PX = 90
OUTLIER_THRESHOLD_PX = 80


def apply_velocity_filter(
    detections: list[Detection],
    max_jump_px: float = MAX_JUMP_PX,
) -> list[Detection]:
    """Return a new list where impossible-jump detections are cleared.

    Any detection whose center is more than max_jump_px pixels from the
    previous detection is cleared (detected=False, bbox cleared).
    Gaps (non-detected frames) are skipped when computing jumps.
    """
    result = list(detections)
    last_detected_idx: int | None = None

    for i, det in enumerate(result):
        if not det.detected:
            continue

        if last_detected_idx is not None:
            prev = result[last_detected_idx]
            gap_s = det.time_s - prev.time_s
            if gap_s <= 0:
                continue

            # Scale allowed displacement by elapsed time, normalised to 60fps.
            # Cap at 3 frames worth so long gaps (ball off-screen) don't allow
            # arbitrarily large jumps.
            allowed = max_jump_px * min(gap_s * 60.0, 3.0)
            dist = _center_dist(prev, det)

            if dist > allowed:
                result[i] = _clear(det)
                continue

        last_detected_idx = i

    return result


def apply_arc_filter(
    detections: list[Detection],
    window: int = 30,
    outlier_px: float = OUTLIER_THRESHOLD_PX,
) -> list[Detection]:
    """Suppress detections that don't fit a local parabolic arc.

    A volleyball follows a parabolic trajectory. Within any 30-frame window,
    detections that deviate more than outlier_px from a fitted parabola are
    suppressed.

    Note: this is a soft heuristic — parabola fitting over short windows is
    noisy. It's most useful for suppressing large-displacement false positives.
    """
    result = list(detections)
    # recompute from result since a prior filter may have cleared some entries
    detected_idx = [i for i, d in enumerate(result) if d.detected and d.center is not None]

    for center_i, idx in enumerate(detected_idx):
        half = window // 2
        lo = max(0, center_i - half)
        hi = min(len(detected_idx), center_i + half + 1)
        nearby = [j for j in detected_idx[lo:hi] if result[j].center is not None]

        if len(nearby) < 4:
            continue

        xs = np.array([result[j].time_s for j in nearby])
        ys = np.array([result[j].center[1] for j in nearby])  # vertical position

        try:
            coeffs = np.polyfit(xs, ys, 2)
            predicted_y = np.polyval(coeffs, result[idx].time_s)
            if result[idx].center is None:
                continue
            residual = abs(result[idx].center[1] - predicted_y)
            if residual > outlier_px:
                result[idx] = _clear(result[idx])
        except (np.linalg.LinAlgError, TypeError):
            pass

    return result


def _center_dist(a: Detection, b: Detection) -> float:
    if a.center is None or b.center is None:
        return float("inf")
    return float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))


def _clear(det: Detection) -> Detection:
    return Detection(
        frame=det.frame, time_s=det.time_s,
        detected=False, conf=None,
        x1=None, y1=None, x2=None, y2=None,
    )
