# volleybot

Automatic volleyball video editing toolkit. Detects rallies in fixed-angle amateur footage and cuts out dead time between points.

## What it does

1. **Detects** the volleyball in each frame using YOLOv8
2. **Segments** the video into rally windows using a detection state machine
3. **Cuts** each rally to a clip with pre/post roll padding, removing dead time

The result is a highlight reel with serves, rallies, and kills — no standing around.

## Results

Fine-tuned YOLOv8n on ~500 labeled frames from 3 different gyms:

| | YOLOv8x COCO (baseline) | YOLOv8n fine-tuned |
|---|---|---|
| Detection rate | 32–66% | **47–84%** |
| Inference speed (Apple MPS) | 6.3 fps | **38.6 fps** |
| Best val mAP50 | — | 0.523 |

A nano model fine-tuned on domain footage beats a pretrained xlarge by +17pp and runs 6× faster.

## Setup

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), ffmpeg, and yt-dlp.

```bash
brew install ffmpeg yt-dlp
git clone https://github.com/rextlfung/volleybot
cd volleybot
uv sync --all-groups
uv run pytest          # 30 tests
```

## Quick start

```bash
# Download a video
yt-dlp -f "bestvideo[height<=1080]+bestaudio/best" --merge-output-format mp4 \
    -o "data/mygame.mp4" <youtube-url>

# Detect + segment + cut (uses fine-tuned weights if present, else yolov8x.pt)
uv run python scripts/cut_rallies.py --input data/mygame.mp4 --concat

# Output: outputs/mygame/rallies/*.mp4  +  outputs/mygame/mygame_all_rallies.mp4
```

## Project structure

```
src/volleybot/
  detection.py      # load CSV, filter_detections(), smoothed_mask()
  segmentation.py   # rally state machine (detection mask → Segment list)
  cutter.py         # ffmpeg video cutting and concat
  tracking.py       # velocity + parabolic arc trajectory filters

scripts/
  # Detection
  yolo_ball_detect_mps.py       # YOLOv8 stream detection with MPS support
  bg_subtract_detect.py         # background subtraction (fast, noisier)

  # Analysis
  analyze_detections.py         # timeline plots + sample frames
  compare_models.py             # detection rate / speed / side-by-side video
  compare_phases.py             # comparison figures between two detection CSVs

  # Labeling & fine-tuning
  sample_frames_for_labeling.py # sample frames for Roboflow upload
  preview_detections.py         # annotate sampled frames with YOLO boxes for QA
  finetune_yolo.py              # fine-tune YOLOv8 on a Roboflow-exported dataset

  # Pipeline
  cut_rallies.py                # end-to-end entrypoint: detect → filter → segment → cut
```

## Fine-tuning your own model

1. **Sample frames** from your videos:
   ```bash
   uv run python scripts/sample_frames_for_labeling.py \
       --inputs data/game1.mp4 data/game2.mp4 --n-frames 150
   ```

2. **Label in [Roboflow](https://roboflow.com)** — one class: `volleyball`. Use Auto Label → COCO sports ball to pre-label, then correct. Target 300–500 frames. Export as YOLOv8 format → unzip to `data/roboflow_dataset/`.

3. **Fine-tune:**
   ```bash
   uv run python scripts/finetune_yolo.py \
       --data data/roboflow_dataset/data.yaml \
       --model yolov8n.pt --epochs 50 --device mps
   ```

4. **Compare against baseline:**
   ```bash
   uv run python scripts/compare_models.py \
       --video outputs/clips/test_10s.mp4 \
       --model-a yolov8x.pt \
       --model-b runs/detect/volleybot/weights/best.pt \
       --out-video outputs/analysis/comparison.mp4
   ```

5. **Run pipeline with fine-tuned model:**
   ```bash
   uv run python scripts/cut_rallies.py \
       --input data/mygame.mp4 \
       --model runs/detect/volleybot/weights/best.pt \
       --concat
   ```

## Sample footage

Videos used for development and fine-tuning:

| File | YouTube | Notes |
|------|---------|-------|
| `20220805g1.mp4` | https://www.youtube.com/watch?v=KlhkMhyrC-g | 1080p60, Old CCRB |
| `20250508g1.mp4` | https://www.youtube.com/watch?v=vF-y5HbF5_c | 1080p30, Clauge middle school (Ann Arbor Rec League) |
| `20230110.mp4`   | https://www.youtube.com/watch?v=qUhEQGQVcVU | 10800p60, NCRB |

## Design notes

- **Audio is not used.** Recreational play has continuous chatter — no whistle or silence signals.
- **Ball class is auto-detected:** class 32 for COCO models, class 0 for fine-tuned single-class models.
- **Stream-copy cutting** (`-c copy`) is used by default for fast lossless cuts. Pass `--reencode` for frame-accurate cuts.
- **Velocity filter** is fps-aware: displacement is scaled by elapsed time, so 30fps and 60fps footage get equivalent thresholds.
