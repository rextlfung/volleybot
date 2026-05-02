"""Cut rally segments from a video file using ffmpeg (stream-copy — lossless, fast)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from volleybot.segmentation import Segment


def cut_segments(
    input_path: Path,
    segments: list[Segment],
    output_dir: Path,
    prefix: str = "rally",
    *,
    reencode: bool = False,
) -> list[Path]:
    """Cut each segment from input_path and write to output_dir.

    Uses stream-copy (-c copy) by default for fast lossless cutting.
    Set reencode=True to get frame-accurate cuts (slower, re-encodes video).

    Returns list of output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for i, seg in enumerate(segments, 1):
        out = output_dir / f"{prefix}_{i:03d}_{seg.start_s:.1f}-{seg.end_s:.1f}s.mp4"
        _ffmpeg_cut(input_path, seg.start_s, seg.end_s, out, reencode=reencode)
        outputs.append(out)
        print(f"  [{i}/{len(segments)}] {out.name}  ({seg.duration_s:.1f}s)")

    return outputs


def concat_segments(
    segment_paths: list[Path],
    output_path: Path,
) -> Path:
    """Concatenate segment files into a single output using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=output_path.parent
    ) as tmp:
        for p in segment_paths:
            tmp.write(f"file '{p.resolve()}'\n")
        list_file = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y", str(output_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        list_file.unlink(missing_ok=True)
    return output_path


def _ffmpeg_cut(
    src: Path,
    start_s: float,
    end_s: float,
    dst: Path,
    *,
    reencode: bool,
) -> None:
    duration = end_s - start_s
    if reencode:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-y", str(dst),
        ]
    else:
        # stream-copy: fast but cut points align to nearest keyframe
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-y", str(dst),
        ]
    subprocess.run(cmd, check=True)
