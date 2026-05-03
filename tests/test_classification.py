"""Tests for volleybot.classification."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from volleybot.classification import (
    FrameClassification,
    load_classification_csv,
    classification_mask,
)


def write_cls_csv(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    fieldnames = ["frame", "time_s", "live", "conf_live", "conf_dead"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return Path(f.name)


def make_row(frame: int, live: bool, conf_live: float = 0.9) -> dict:
    return {
        "frame": frame,
        "time_s": f"{frame / 30:.3f}",
        "live": int(live),
        "conf_live": f"{conf_live:.4f}",
        "conf_dead": f"{1 - conf_live:.4f}",
    }


# --- load_classification_csv ---

def test_load_returns_correct_types():
    rows = [make_row(0, False, 0.1), make_row(1, True, 0.9)]
    path = write_cls_csv(rows)
    results = load_classification_csv(path)
    assert len(results) == 2
    assert isinstance(results[0], FrameClassification)
    assert results[0].frame == 0
    assert results[0].live is False
    assert abs(results[0].conf_live - 0.1) < 1e-4
    assert results[1].live is True
    assert results[1].frame == 1


def test_load_empty_csv():
    path = write_cls_csv([])
    results = load_classification_csv(path)
    assert results == []


def test_load_preserves_order():
    rows = [make_row(i, i % 2 == 0) for i in range(10)]
    path = write_cls_csv(rows)
    results = load_classification_csv(path)
    assert [r.frame for r in results] == list(range(10))


# --- classification_mask ---

def _make_results(live_flags: list[bool], fps: float = 30.0) -> list[FrameClassification]:
    return [
        FrameClassification(
            frame=i,
            time_s=i / fps,
            live=v,
            conf_live=0.9 if v else 0.1,
            conf_dead=0.1 if v else 0.9,
        )
        for i, v in enumerate(live_flags)
    ]


def test_mask_all_dead():
    results = _make_results([False] * 10)
    mask = classification_mask(results, fps=30.0)
    assert not mask.any()


def test_mask_all_live():
    results = _make_results([True] * 10)
    mask = classification_mask(results, fps=30.0)
    assert mask.all()


def test_mask_length_matches_input():
    results = _make_results([True, False, True] * 5)
    mask = classification_mask(results, fps=30.0)
    assert len(mask) == 15


def test_mask_fills_short_dead_gap():
    # dead gap of 2 frames (0.067s at 30fps) between two live windows
    flags = [True] * 5 + [False, False] + [True] * 5
    results = _make_results(flags)
    mask = classification_mask(results, fps=30.0, fill_gap_s=0.5)
    assert mask.all(), "short gap inside live window should be filled"


def test_mask_does_not_fill_long_dead_gap():
    # dead gap of 60 frames (2s at 30fps) — longer than fill_gap_s=0.5
    flags = [True] * 5 + [False] * 60 + [True] * 5
    results = _make_results(flags)
    mask = classification_mask(results, fps=30.0, fill_gap_s=0.5)
    assert not mask[5:65].any(), "long gap should not be filled"


def test_mask_does_not_fill_leading_gap():
    # leading dead frames before first live — must NOT be filled
    flags = [False] * 10 + [True] * 5 + [False, False] + [True] * 5
    results = _make_results(flags)
    mask = classification_mask(results, fps=30.0, fill_gap_s=0.5)
    assert not mask[:10].any(), "leading gap before first live frame must not be filled"


def test_mask_does_not_fill_trailing_gap():
    # trailing dead frames after last live window — must NOT be filled
    flags = [True] * 5 + [False] * 10
    results = _make_results(flags)
    mask = classification_mask(results, fps=30.0, fill_gap_s=0.5)
    assert not mask[5:].any(), "trailing gap after last live window must not be filled"


def test_mask_min_conf_filters_low_confidence():
    # frame is labeled live=True but with only 0.6 confidence — filtered by min_conf=0.7
    results = _make_results([True] * 5, fps=30.0)
    for r in results:
        r.conf_live = 0.6
    mask = classification_mask(results, fps=30.0, min_conf=0.7)
    assert not mask.any(), "frames below min_conf should not count as live"


def test_mask_min_conf_keeps_high_confidence():
    results = _make_results([True] * 5, fps=30.0)
    mask = classification_mask(results, fps=30.0, min_conf=0.5)
    assert mask.all()
