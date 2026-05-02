"""Tests for filter_detections()."""

import pytest

from volleybot.detection import Detection, filter_detections


def det(frame: int, conf: float, cx: int, cy: int, w: int = 30) -> Detection:
    return Detection(
        frame=frame, time_s=frame / 60.0, detected=True, conf=conf,
        x1=cx - w // 2, y1=cy - w // 2,
        x2=cx + w // 2, y2=cy + w // 2,
    )


def miss(frame: int) -> Detection:
    return Detection(frame=frame, time_s=frame / 60.0, detected=False,
                     conf=None, x1=None, y1=None, x2=None, y2=None)


class TestFilterDetections:
    FH = 1080

    def test_low_conf_cleared(self):
        dets = [det(0, conf=0.20, cx=500, cy=400)]
        result = filter_detections(dets, min_conf=0.25, max_y_frac=1.0, frame_height=self.FH)
        assert not result[0].detected

    def test_sufficient_conf_kept(self):
        dets = [det(0, conf=0.40, cx=500, cy=400)]
        result = filter_detections(dets, min_conf=0.25, max_y_frac=1.0, frame_height=self.FH)
        assert result[0].detected

    def test_too_wide_cleared(self):
        # ball width = 150px > max_ball_px=80
        dets = [det(0, conf=0.8, cx=500, cy=400, w=150)]
        result = filter_detections(dets, min_conf=0.1, max_ball_px=80,
                                   max_y_frac=1.0, frame_height=self.FH)
        assert not result[0].detected

    def test_floor_detection_cleared(self):
        # cy = 900 = 83% of 1080 > max_y_frac=0.72
        dets = [det(0, conf=0.9, cx=500, cy=900)]
        result = filter_detections(dets, min_conf=0.1, max_y_frac=0.72,
                                   frame_height=self.FH)
        assert not result[0].detected

    def test_in_air_detection_kept(self):
        # cy = 400 = 37% of 1080 — well above the floor threshold
        dets = [det(0, conf=0.5, cx=500, cy=400)]
        result = filter_detections(dets, min_conf=0.25, max_ball_px=80,
                                   max_y_frac=0.72, frame_height=self.FH)
        assert result[0].detected

    def test_missed_frames_untouched(self):
        dets = [miss(0), miss(1)]
        result = filter_detections(dets)
        assert not result[0].detected
        assert not result[1].detected
