"""Run full analysis pipeline once YOLO detection CSV is ready.

Produces:
- Detection timeline + sample frame plots (outputs/analysis/)
- Detector comparison plot (outputs/analysis/detector_comparison.png)
- Rally-cut clips from the test video (outputs/clips/rallies/)
- A concatenated "all rallies" video (outputs/clips/test_2min_rallies.mp4)

Usage: uv run python scripts/post_yolo_analysis.py
"""

from pathlib import Path
import subprocess
import sys

YOLO_CSV = Path("outputs/clips/detections.csv")
YOLO_VIDEO = Path("outputs/clips/test_2min.mp4")
ANNOTATED_VIDEO = Path("outputs/clips/test_2min_annotated.mp4")


def main() -> None:
    if not YOLO_CSV.exists():
        print(f"YOLO CSV not found: {YOLO_CSV}")
        sys.exit(1)

    n_frames = sum(1 for _ in open(YOLO_CSV)) - 1  # subtract header
    print(f"YOLO CSV found: {n_frames} frames")

    # 1. Analyze detections + sample frames
    print("\n--- detection analysis ---")
    subprocess.run(
        ["uv", "run", "python", "scripts/analyze_detections.py",
         "--csv", str(YOLO_CSV),
         "--video", str(YOLO_VIDEO),
         "--samples", "16"],
        check=True,
    )

    # 2. Detector comparison
    print("\n--- detector comparison ---")
    subprocess.run(
        ["uv", "run", "python", "scripts/compare_detectors.py"],
        check=True,
    )

    # 3. Cut rally segments from test clip
    print("\n--- cutting rally segments ---")
    subprocess.run(
        ["uv", "run", "python", "scripts/cut_rallies.py",
         "--input", str(YOLO_VIDEO),
         "--csv", str(YOLO_CSV),
         "--out-dir", "outputs/clips/rallies",
         "--pre-roll", "1.5",
         "--post-roll", "2.0",
         "--concat"],
        check=True,
    )

    print("\n=== done ===")
    print("outputs/analysis/   — timeline plots + sample frames")
    print("outputs/analysis/detector_comparison.png — YOLO vs bg-subtract")
    print("outputs/clips/rallies/ — individual rally clips")
    print("outputs/clips/test_2min_rallies.mp4 — concatenated highlight reel")


if __name__ == "__main__":
    main()
