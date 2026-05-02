"""Tests for volleybot.tracking."""

import pytest

from volleybot.detection import Detection
from volleybot.tracking import apply_velocity_filter


def make_det(frame: int, cx: int, cy: int, fps: float = 60.0) -> Detection:
    return Detection(
        frame=frame, time_s=frame / fps, detected=True, conf=0.9,
        x1=cx - 15, y1=cy - 15, x2=cx + 15, y2=cy + 15,
    )


def make_miss(frame: int, fps: float = 60.0) -> Detection:
    return Detection(
        frame=frame, time_s=frame / fps, detected=False, conf=None,
        x1=None, y1=None, x2=None, y2=None,
    )


class TestVelocityFilter:
    def test_nearby_detections_kept(self):
        dets = [make_det(0, 500, 300), make_det(1, 510, 295)]
        result = apply_velocity_filter(dets, max_jump_px=90)
        assert result[0].detected
        assert result[1].detected

    def test_impossible_jump_cleared(self):
        # jump of 500px in 1 frame — impossible
        dets = [make_det(0, 100, 300), make_det(1, 600, 300)]
        result = apply_velocity_filter(dets, max_jump_px=90)
        assert result[0].detected
        assert not result[1].detected

    def test_gap_allows_larger_jump(self):
        # 3-frame gap: allowed jump = 90 * 3 = 270px
        dets = [make_det(0, 100, 300), make_miss(1), make_miss(2), make_det(3, 350, 290)]
        result = apply_velocity_filter(dets, max_jump_px=90)
        assert result[0].detected
        assert result[3].detected

    def test_first_detection_always_kept(self):
        dets = [make_det(0, 960, 500)]
        result = apply_velocity_filter(dets)
        assert result[0].detected

    def test_non_detected_untouched(self):
        dets = [make_miss(0), make_miss(1)]
        result = apply_velocity_filter(dets)
        assert not result[0].detected
        assert not result[1].detected

    def test_fps_aware_allows_more_at_30fps(self):
        # At 30fps a 1-frame gap = 33ms; ball can travel farther than at 60fps.
        # 150px jump: too far for 60fps (>90px/frame) but fine at 30fps
        # (equivalent to 75px/frame at 60fps).
        dets_30 = [make_det(0, 100, 300, fps=30.0), make_det(1, 250, 300, fps=30.0)]
        dets_60 = [make_det(0, 100, 300, fps=60.0), make_det(1, 250, 300, fps=60.0)]
        assert apply_velocity_filter(dets_30, max_jump_px=90)[1].detected
        assert not apply_velocity_filter(dets_60, max_jump_px=90)[1].detected
