"""Tests for volleybot.detection."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from volleybot.detection import Detection, load_csv, detection_mask, smoothed_mask


def write_temp_csv(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    fieldnames = ["frame", "time_s", "detected", "conf", "x1", "y1", "x2", "y2"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return Path(f.name)


def make_row(frame: int, detected: bool, conf: float = 0.9,
             x1=100, y1=200, x2=130, y2=230) -> dict:
    if detected:
        return {"frame": frame, "time_s": frame / 30,
                "detected": 1, "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return {"frame": frame, "time_s": frame / 30,
            "detected": 0, "conf": "", "x1": "", "y1": "", "x2": "", "y2": ""}


class TestLoadCsv:
    def test_loads_detected(self):
        path = write_temp_csv([make_row(0, True), make_row(1, False)])
        dets = load_csv(path)
        assert len(dets) == 2
        assert dets[0].detected is True
        assert dets[0].conf == pytest.approx(0.9)
        assert dets[1].detected is False
        assert dets[1].conf is None

    def test_bbox_properties(self):
        path = write_temp_csv([make_row(0, True, x1=100, y1=200, x2=130, y2=230)])
        det = load_csv(path)[0]
        assert det.bbox_area == 30 * 30
        assert det.center == (115, 215)

    def test_no_bbox_when_not_detected(self):
        path = write_temp_csv([make_row(0, False)])
        det = load_csv(path)[0]
        assert det.bbox_area is None
        assert det.center is None


class TestSmoothedMask:
    def test_fills_short_gap(self):
        # gap of 5 frames in a 30fps signal — fill_gap_s=0.5 means fill up to 15 frames
        rows = ([make_row(i, True) for i in range(10)] +
                [make_row(i, False) for i in range(10, 15)] +
                [make_row(i, True) for i in range(15, 25)])
        path = write_temp_csv(rows)
        dets = load_csv(path)
        mask = smoothed_mask(dets, fps=30.0, fill_gap_s=0.5)
        assert mask[10:15].all()

    def test_does_not_fill_long_gap(self):
        # gap of 60 frames (2s) > fill_gap_s=0.5
        rows = ([make_row(i, True) for i in range(10)] +
                [make_row(i, False) for i in range(10, 70)] +
                [make_row(i, True) for i in range(70, 80)])
        path = write_temp_csv(rows)
        dets = load_csv(path)
        mask = smoothed_mask(dets, fps=30.0, fill_gap_s=0.5)
        assert not mask[10:70].any()

    def test_does_not_fill_leading_gap(self):
        # frames 0-4 are undetected, first detection at frame 5 — short enough
        # to fill if the gap-fill were naive, but should NOT be filled because
        # there is no prior detection window on the left.
        rows = ([make_row(i, False) for i in range(5)] +
                [make_row(i, True) for i in range(5, 15)])
        path = write_temp_csv(rows)
        dets = load_csv(path)
        mask = smoothed_mask(dets, fps=30.0, fill_gap_s=1.0)
        assert not mask[0:5].any()
