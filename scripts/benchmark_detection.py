"""Benchmark detection speed: pure YOLO inference, no video writing.

Tests MPS vs CPU and with/without 1080p→640 resize.

Usage: uv run python scripts/benchmark_detection.py
"""

from pathlib import Path
import time
import sys

from ultralytics import YOLO

CLIP = Path("outputs/clips/test_10s.mp4")
MODEL = "yolov8x.pt"
BALL_CLASS = 32
CONF = 0.25


def bench(device: str, resize: int | None = None) -> None:
    model = YOLO(MODEL)
    label = f"device={device}" + (f" resize={resize}" if resize else " fullres")

    kwargs = dict(
        source=str(CLIP),
        classes=[BALL_CLASS],
        conf=CONF,
        device=device,
        stream=True,
        verbose=False,
        imgsz=resize if resize else 1920,
        save=False,
    )

    # warmup
    for _ in model.predict(**kwargs):
        break

    t0 = time.time()
    n = 0
    for _ in model.predict(**kwargs):
        n += 1
    elapsed = time.time() - t0
    print(f"  {label}: {n/elapsed:.1f} fps ({elapsed:.1f}s for {n} frames)")


def main() -> None:
    print(f"benchmarking {MODEL} on {CLIP}")
    print("(each run processes all frames with no disk writes)\n")
    bench("cpu", resize=640)
    bench("mps", resize=640)
    bench("mps", resize=1280)
    bench("mps", resize=None)


if __name__ == "__main__":
    main()
