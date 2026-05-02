"""Tests for volleybot.segmentation."""

import numpy as np
import pytest

from volleybot.segmentation import detect_rallies, merge_overlapping, Segment


FPS = 30.0


def bool_mask(total: int, on_ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(total, dtype=bool)
    for s, e in on_ranges:
        mask[s:e] = True
    return mask


class TestDetectRallies:
    def test_single_rally(self):
        # ball detected frames 30-120 (1–4s at 30fps)
        mask = bool_mask(300, [(30, 120)])
        segs = detect_rallies(mask, FPS, dead_gap_s=1.0, min_rally_s=0.5,
                              pre_roll_s=0.0, post_roll_s=0.0, total_duration_s=10.0)
        assert len(segs) == 1
        assert pytest.approx(segs[0].raw_start_s, abs=0.1) == 1.0
        assert pytest.approx(segs[0].raw_end_s, abs=0.1) == 4.0

    def test_two_rallies_separated_by_gap(self):
        # rally 1: frames 0-60; gap 61-150; rally 2: frames 151-240
        mask = bool_mask(300, [(0, 60), (151, 240)])
        segs = detect_rallies(mask, FPS, dead_gap_s=2.0, min_rally_s=0.5,
                              pre_roll_s=0.0, post_roll_s=0.0, total_duration_s=10.0)
        assert len(segs) == 2

    def test_short_rally_filtered(self):
        # only 10 frames (0.33s), below min_rally_s=1.0
        mask = bool_mask(300, [(50, 60)])
        segs = detect_rallies(mask, FPS, dead_gap_s=1.0, min_rally_s=1.0,
                              pre_roll_s=0.0, post_roll_s=0.0)
        assert len(segs) == 0

    def test_pre_post_roll_applied(self):
        mask = bool_mask(600, [(120, 240)])  # frames 4–8s
        segs = detect_rallies(mask, FPS, dead_gap_s=1.0, min_rally_s=0.5,
                              pre_roll_s=1.5, post_roll_s=2.0, total_duration_s=20.0)
        assert len(segs) == 1
        assert pytest.approx(segs[0].start_s, abs=0.1) == 4.0 - 1.5
        assert pytest.approx(segs[0].end_s, abs=0.1) == 8.0 + 2.0

    def test_no_detections(self):
        mask = np.zeros(300, dtype=bool)
        segs = detect_rallies(mask, FPS)
        assert len(segs) == 0

    def test_all_detections(self):
        mask = np.ones(300, dtype=bool)
        segs = detect_rallies(mask, FPS, dead_gap_s=1.0, min_rally_s=0.5,
                              pre_roll_s=0.0, post_roll_s=0.0, total_duration_s=10.0)
        assert len(segs) == 1

    def test_end_clamped_by_total_duration(self):
        mask = bool_mask(180, [(120, 179)])  # rally ends at end of clip
        segs = detect_rallies(mask, FPS, dead_gap_s=1.0, min_rally_s=0.5,
                              pre_roll_s=0.0, post_roll_s=5.0, total_duration_s=6.0)
        assert len(segs) == 1
        assert segs[0].end_s <= 6.0


class TestMergeOverlapping:
    def test_non_overlapping(self):
        a = Segment(0, 5, 1, 4)
        b = Segment(10, 15, 11, 14)
        result = merge_overlapping([a, b])
        assert len(result) == 2

    def test_overlapping(self):
        a = Segment(0, 10, 1, 9)
        b = Segment(8, 15, 9, 14)
        result = merge_overlapping([a, b])
        assert len(result) == 1
        assert result[0].start_s == 0
        assert result[0].end_s == 15

    def test_adjacent_merged(self):
        a = Segment(0, 5, 1, 4)
        b = Segment(5, 10, 6, 9)  # starts exactly where a ends
        result = merge_overlapping([a, b])
        assert len(result) == 1

    def test_empty(self):
        assert merge_overlapping([]) == []

    def test_single(self):
        s = Segment(1, 5, 2, 4)
        assert merge_overlapping([s]) == [s]
