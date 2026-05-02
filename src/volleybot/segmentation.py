"""Rally segmentation: convert per-frame ball detection signals into rally segments.

A rally is "in progress" when the ball is active. Between rallies there is dead
time: serving setup, point celebration, rotation, timeouts, etc.

State machine
─────────────
DEAD  ──[ball detected]──► RALLY
RALLY ──[no ball for dead_gap_s]──► DEAD

Each rally segment is padded by pre_roll_s before and post_roll_s after so that
the serve startup and the kill/out moment are included in the cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    start_s: float
    end_s: float
    raw_start_s: float   # before padding
    raw_end_s: float     # before padding

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def detect_rallies(
    mask: np.ndarray,
    fps: float,
    dead_gap_s: float = 3.0,
    min_rally_s: float = 1.5,
    pre_roll_s: float = 1.5,
    post_roll_s: float = 2.0,
    total_duration_s: float | None = None,
) -> list[Segment]:
    """Detect rally segments from a boolean detection mask.

    Args:
        mask: per-frame boolean array, True = ball detected.
        fps: video frame rate.
        dead_gap_s: seconds of no detection before ending a rally.
        min_rally_s: discard rallies shorter than this (likely false positives).
        pre_roll_s: seconds to include before the first detection.
        post_roll_s: seconds to include after the last detection.
        total_duration_s: clip duration (used to clamp segment boundaries).

    Returns:
        List of Segment objects sorted by start time.
    """
    dead_gap_frames = int(dead_gap_s * fps)
    min_rally_frames = int(min_rally_s * fps)
    max_frame = len(mask) - 1

    segments: list[Segment] = []
    in_rally = False
    rally_start = 0
    last_detect = -dead_gap_frames - 1

    for i, detected in enumerate(mask):
        if detected:
            if not in_rally:
                in_rally = True
                rally_start = i
            last_detect = i
        else:
            if in_rally and (i - last_detect) > dead_gap_frames:
                # rally ended dead_gap_frames ago
                raw_end = last_detect
                if (raw_end - rally_start) >= min_rally_frames:
                    segments.append(_make_segment(rally_start, raw_end, fps,
                                                  pre_roll_s, post_roll_s,
                                                  max_frame, total_duration_s))
                in_rally = False

    # handle rally that extends to end of clip
    if in_rally:
        raw_end = last_detect
        if (raw_end - rally_start) >= min_rally_frames:
            segments.append(_make_segment(rally_start, raw_end, fps,
                                          pre_roll_s, post_roll_s,
                                          max_frame, total_duration_s))

    return segments


def _make_segment(
    raw_start_frame: int,
    raw_end_frame: int,
    fps: float,
    pre_roll_s: float,
    post_roll_s: float,
    max_frame: int,
    total_duration_s: float | None,
) -> Segment:
    raw_start_s = raw_start_frame / fps
    raw_end_s = raw_end_frame / fps
    start_s = max(0.0, raw_start_s - pre_roll_s)
    end_s = raw_end_s + post_roll_s
    if total_duration_s is not None:
        end_s = min(end_s, total_duration_s)
    return Segment(start_s=start_s, end_s=end_s,
                   raw_start_s=raw_start_s, raw_end_s=raw_end_s)


def merge_overlapping(segments: list[Segment]) -> list[Segment]:
    """Merge segments whose padded ranges overlap."""
    if not segments:
        return []
    merged: list[Segment] = []
    cur = segments[0]
    for nxt in segments[1:]:
        if nxt.start_s <= cur.end_s:
            cur = Segment(
                start_s=cur.start_s,
                end_s=max(cur.end_s, nxt.end_s),
                raw_start_s=cur.raw_start_s,
                raw_end_s=max(cur.raw_end_s, nxt.raw_end_s),
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged
