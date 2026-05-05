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
| Inference speed — Apple Silicon (MPS) | 6.3 fps | **38.6 fps** |
| Inference speed — CPU only | 3.1 fps | faster, not yet benchmarked |
| Best val mAP50 | — | 0.523 |

A nano model fine-tuned on domain footage beats a pretrained xlarge by +17pp and runs 6× faster.

**Side-by-side: YOLOv8x COCO (left) vs YOLOv8n fine-tuned (right)**

https://github.com/rextlfung/volleybot/releases/download/v0.1.0/model_comparison.mp4

## Setup

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), ffmpeg, and yt-dlp.

```bash
# macOS
brew install ffmpeg yt-dlp

# Windows
winget install Gyan.FFmpeg
winget install yt-dlp.yt-dlp
```

```bash
git clone https://github.com/rextlfung/volleybot
cd volleybot
uv sync --all-groups
uv run pytest          # 36 tests
```

### GPU / device support

Scripts auto-detect the best available device (`cuda > mps > cpu`). No `--device` flag is needed in most cases, but see the platform notes below.

| Platform | Device used | Notes |
|----------|-------------|-------|
| macOS — Apple Silicon | MPS | ~38.6 fps with fine-tuned YOLOv8n |
| Linux — NVIDIA GPU | CUDA | fastest; set `--device cuda` if auto-detect misses it |
| Linux — AMD GPU (RDNA2+) | CUDA\* | requires ROCm — see AMD GPU note in CLAUDE.md |
| Windows — NVIDIA GPU | CUDA | |
| **Windows — AMD GPU** | **CPU** | **PyTorch has no ROCm support on Windows** |
| Any machine, no GPU | CPU | significantly slower than real-time |

**CPU-only performance:** The COCO YOLOv8x baseline runs at ~3.1 fps on CPU. The fine-tuned
YOLOv8n is lighter but the pipeline still takes several hours per 2-hour 60fps game video (detection +
classification both run frame-by-frame). **Test on a short clip before committing to a full run:**

```bash
# Cut a 60-second test clip from minute 9
ffmpeg -ss 540 -i data/mygame.mp4 -t 60 -c copy outputs/test_60s.mp4 -y
uv run python scripts/cut_rallies.py --input outputs/test_60s.mp4 --concat
```

For full-length games on CPU, use `run_pipeline.bat` (Windows) or `nohup` / `screen` (Linux/macOS) to run overnight.

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
  detection.py        # load CSV, filter_detections(), smoothed_mask()
  segmentation.py     # rally state machine (detection mask → Segment list)
  cutter.py           # ffmpeg video cutting and concat
  classification.py   # load classification CSV, classification_mask()
  device.py           # best_device(): auto-detect cuda > mps > cpu

scripts/
  # Detection
  yolo_ball_detect_mps.py             # YOLOv8 stream detection with MPS support

  # Analysis
  analyze_detections.py               # timeline plots + sample frames
  compare_models.py                   # detection rate / speed / side-by-side video
  compare_phases.py                   # comparison figures between two detection CSVs

  # Labeling & fine-tuning (ball detection)
  sample_frames_for_labeling.py       # sample frames for Roboflow upload
  preview_detections.py               # annotate sampled frames with YOLO boxes for QA
  finetune_yolo.py                    # fine-tune YOLOv8 on a Roboflow-exported dataset

  # Labeling & fine-tuning (play-state classifier)
  sample_frames_for_classification.py # sample frames for live/dead labeling
  classify_frames.py                  # run YOLOv8-cls → per-frame live/dead CSV
  finetune_classifier.py              # fine-tune YOLOv8n-cls on Roboflow Classification export

  # Pipeline
  cut_rallies.py                      # end-to-end entrypoint: detect → segment → cut
                                      # --classifier-model / --classifier-csv for classifier-driven segmentation
  run_pipeline_all.py                 # run cut_rallies.py on all 3 game videos sequentially

run_pipeline.bat                      # Windows: run run_pipeline_all.py via venv Python, log to outputs/logs/
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
       --model yolov8n.pt --epochs 50
   # Logs to wandb project "volleybot" by default; pass --wandb-project "" to disable
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

## Classifier-based segmentation (Phase 3)

Ball detection alone gives imprecise rally boundaries when the detector is sensitive enough to also pick up the ball during dead time (serve setup, ball retrieval). A frame-level live/dead classifier solves this by using full-frame context.

1. **Sample frames** for labeling (transition-biased — captures serve toss and point end):
   ```bash
   uv run python scripts/sample_frames_for_classification.py \
       --inputs data/game1.mp4 data/game2.mp4 \
       --n-frames 200 --csv outputs/game1/detections.csv
   ```

2. **Label in Roboflow** (Classification project, not Detection). Two classes: `live` / `dead`.
   - `live` = from serve toss until the point ends
   - `dead` = everything else (rotation, server bouncing ball, ball retrieval)
   Export as Folder Structure / YOLOv8 → unzip to `data/roboflow_classification/`.

3. **Fine-tune:**
   ```bash
   uv run python scripts/finetune_classifier.py \
       --data data/roboflow_classification --epochs 30
   # Logs to wandb project "volleybot" by default; pass --wandb-project "" to disable
   ```

4. **Run the pipeline with classifier-driven segmentation:**
   ```bash
   uv run python scripts/cut_rallies.py \
       --input data/mygame.mp4 \
       --classifier-model runs/classify/volleybot_cls/weights/best.pt \
       --concat
   ```
   Ball detection still runs for every video (tracking CSV for future highlight reels and statistics).

## Batch processing

To run the full pipeline on all game videos sequentially:

```bash
uv run python scripts/run_pipeline_all.py
```

On Windows, `run_pipeline.bat` runs the same script using the venv Python directly and appends stdout/stderr to `outputs/logs/all_pipeline.log` / `all_pipeline.err`. Useful for running overnight as a scheduled task without a terminal window.

## Sample footage

Videos used for development and fine-tuning:

| File | YouTube | Notes |
|------|---------|-------|
| `20220805g1.mp4` | https://www.youtube.com/watch?v=KlhkMhyrC-g | 1080p60, Old CCRB |
| `20230110.mp4`   | https://www.youtube.com/watch?v=qUhEQGQVcVU | 1080p60, NCRB |
| `20250508g1.mp4` | https://www.youtube.com/watch?v=vF-y5HbF5_c | 1080p30, Clague Middle School (Ann Arbor Rec League) |

## Design notes

- **Audio is not used.** Recreational play has continuous chatter — no whistle or silence signals.
- **Ball class is auto-detected:** class 32 for COCO models, class 0 for fine-tuned single-class models.
- **Stream-copy cutting** (`-c copy`) is used by default for fast lossless cuts. Pass `--reencode` for frame-accurate cuts.
- **Velocity filter** is fps-aware: displacement is scaled by elapsed time, so 30fps and 60fps footage get equivalent thresholds.
- **Two-stage pipeline:** ball detector (tracking) + frame classifier (segmentation). Decoupling them means the detector can be tuned for recall without breaking segmentation quality.
- **Windows + AMD GPU:** PyTorch does not support ROCm on Windows, so the AMD GPU goes unused and inference falls back to CPU. Native Linux (dual-boot) or WSL2 with ROCm are the paths to GPU acceleration on AMD hardware (see CLAUDE.md for setup details).
