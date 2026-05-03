"""Compare two YOLO models on the same video: detection rate, speed, confidence.

Useful for measuring improvement after fine-tuning. Runs both models on the
same clip and reports detection rate, FPS, and confidence distribution.
Optionally saves a side-by-side annotated video.

Usage:
  # Quick stats only:
  uv run python scripts/compare_models.py \
      --video outputs/clips/test_10s.mp4 \
      --model-a yolov8x.pt \
      --model-b runs/detect/volleybot/weights/best.pt

  # Also write side-by-side annotated video:
  uv run python scripts/compare_models.py \
      --video outputs/clips/test_10s.mp4 \
      --model-a yolov8x.pt \
      --model-b runs/detect/volleybot/weights/best.pt \
      --out-video outputs/analysis/model_comparison.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from volleybot.device import best_device

COCO_BALL_CLASS = 32
_GREEN = (0, 200, 0)
_BLUE = (220, 100, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ModelResult:
    name: str
    detections: int = 0
    total: int = 0
    confs: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def detection_rate(self) -> float:
        return self.detections / self.total if self.total else 0.0

    @property
    def fps(self) -> float:
        return self.total / self.elapsed_s if self.elapsed_s else 0.0

    def summary(self) -> str:
        cr = np.mean(self.confs) if self.confs else 0.0
        return (
            f"  {self.name}\n"
            f"    detected : {self.detections}/{self.total} "
            f"({self.detection_rate*100:.1f}%)\n"
            f"    mean conf: {cr:.3f}\n"
            f"    speed    : {self.fps:.1f} fps"
        )


def _ball_class(model: YOLO) -> int:
    return COCO_BALL_CLASS if COCO_BALL_CLASS in model.names else 0


def _run_model(model: YOLO, video: Path, device: str, conf: float) -> tuple[ModelResult, list]:
    bc = _ball_class(model)
    result = ModelResult(name=Path(model.ckpt_path).name if hasattr(model, "ckpt_path") else str(model.model))
    per_frame: list[dict | None] = []

    t0 = time.time()
    for r in model.predict(
        source=str(video), classes=[bc], conf=conf,
        device=device, stream=True, verbose=False,
    ):
        result.total += 1
        if len(r.boxes) > 0:
            best = r.boxes[r.boxes.conf.argmax()]
            c = float(best.conf[0])
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            result.detections += 1
            result.confs.append(c)
            per_frame.append({"conf": c, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        else:
            per_frame.append(None)

    result.elapsed_s = time.time() - t0
    return result, per_frame


def _draw_box(frame, det: dict | None, color, label_prefix: str) -> None:
    if det is None:
        return
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max(x2 - x1, y2 - y1) // 2
    cv2.circle(frame, (cx, cy), r + 4, color, 2)
    cv2.putText(frame, f"{label_prefix} {det['conf']:.2f}",
                (x1, max(y1 - 8, 20)), _FONT, 0.6, color, 2)


def _write_comparison_video(
    video: Path,
    frames_a: list[dict | None],
    frames_b: list[dict | None],
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # side-by-side: two frames at half width
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W * 2, H))

    for i, (da, db) in enumerate(zip(frames_a, frames_b)):
        ret, frame = cap.read()
        if not ret:
            break
        fa = frame.copy()
        fb = frame.copy()
        _draw_box(fa, da, _GREEN, "A")
        _draw_box(fb, db, _BLUE, "B")
        cv2.putText(fa, label_a, (10, H - 14), _FONT, 0.7, _GREEN, 2)
        cv2.putText(fb, label_b, (10, H - 14), _FONT, 0.7, _BLUE, 2)
        writer.write(np.hstack([fa, fb]))

    cap.release()
    writer.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two YOLO models on the same video."
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model-a", type=Path, required=True,
                   help="baseline model (e.g. yolov8x.pt)")
    p.add_argument("--model-b", type=Path, required=True,
                   help="comparison model (e.g. fine-tuned best.pt)")
    p.add_argument("--device", default=best_device())
    p.add_argument("--conf", type=float, default=0.15,
                   help="confidence threshold (default 0.15 — same as pipeline)")
    p.add_argument("--out-video", type=Path, default=None,
                   help="write side-by-side annotated video here (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for p in [args.video, args.model_a, args.model_b]:
        if not Path(p).exists():
            print(f"not found: {p}")
            sys.exit(1)

    print(f"video   : {args.video}")
    print(f"model A : {args.model_a}")
    print(f"model B : {args.model_b}")
    print(f"device  : {args.device}  conf≥{args.conf}\n")

    model_a = YOLO(str(args.model_a))
    model_b = YOLO(str(args.model_b))

    print("running model A ...")
    res_a, frames_a = _run_model(model_a, args.video, args.device, args.conf)
    res_a.name = args.model_a.name

    print("running model B ...")
    res_b, frames_b = _run_model(model_b, args.video, args.device, args.conf)
    res_b.name = args.model_b.name

    print("\n=== results ===")
    print(res_a.summary())
    print(res_b.summary())

    delta = res_b.detection_rate - res_a.detection_rate
    sign = "+" if delta >= 0 else ""
    print(f"\n  detection rate delta (B - A): {sign}{delta*100:.1f}pp")

    if args.out_video:
        print(f"\nwriting side-by-side video → {args.out_video} ...")
        _write_comparison_video(
            args.video, frames_a, frames_b,
            res_a.name, res_b.name, args.out_video,
        )
        print("done.")


if __name__ == "__main__":
    main()
