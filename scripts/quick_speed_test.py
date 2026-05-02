"""Quick speed test: process 100 frames at 640px on CPU vs MPS.

Usage: uv run python scripts/quick_speed_test.py
"""

from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "yolov8x.pt"
BALL_CLASS = 32
CONF = 0.25
N_FRAMES = 100
RESIZE = 640


def bench_device(device: str, frames: list[np.ndarray]) -> float:
    model = YOLO(MODEL)
    # warmup (1 frame)
    model.predict(frames[:1], classes=[BALL_CLASS], conf=CONF,
                  device=device, imgsz=RESIZE, verbose=False)
    t0 = time.time()
    for frame in frames:
        model.predict([frame], classes=[BALL_CLASS], conf=CONF,
                      device=device, imgsz=RESIZE, verbose=False)
    return N_FRAMES / (time.time() - t0)


def main() -> None:
    clip = Path("outputs/clips/test_10s.mp4")
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while len(frames) < N_FRAMES:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    print(f"loaded {len(frames)} frames from {clip}")

    cpu_fps = bench_device("cpu", frames)
    print(f"CPU: {cpu_fps:.1f} fps at {RESIZE}px")

    mps_fps = bench_device("mps", frames)
    print(f"MPS: {mps_fps:.1f} fps at {RESIZE}px  ({mps_fps/cpu_fps:.1f}× speedup)")


if __name__ == "__main__":
    main()
