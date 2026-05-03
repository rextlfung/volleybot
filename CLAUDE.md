# Volleybot

Automatic volleyball video editing toolkit. Detects rallies in fixed-angle amateur footage and produces clean-cut video with dead time removed.

## Project goals (priority order)
1. **Clean cuts** — remove dead time between rallies
2. **Highlight reels** — best-of compilation
3. **Game statistics** — per-rally events (future)

## Footage characteristics
- ~100 videos, 2-3 hours each
- Fixed camera from the back/endline on a tripod, slightly elevated
- Almost all 1080p60 on YouTube (download with yt-dlp at format 299+140)
- Multiple indoor gyms — green curtain (gym1/20220805), different backgrounds at gym2/gym3
- Recreational play — no referee whistles, continuous audio chatter (audio is NOT a reliable segmentation signal)

## Setup

```bash
uv sync --all-groups      # install deps + dev deps
uv run pytest             # run tests
```

Requires ffmpeg and yt-dlp to be installed:
```bash
# Windows (winget):
winget install Gyan.FFmpeg
winget install yt-dlp.yt-dlp

# macOS (brew):
# brew install ffmpeg yt-dlp
```

### GPU acceleration

Scripts auto-detect the best device (`cuda > mps > cpu`) via `volleybot.device.best_device()`.
You can always override with `--device cpu|cuda|mps`.

**Windows + AMD GPU (RX 6700 XT):** PyTorch does not support ROCm on Windows — CPU only.
To use the AMD GPU, run the pipeline under WSL2 (see below).

### AMD GPU via WSL2 + ROCm (RX 6700 XT — UNTESTED, RDNA2 not officially supported)

> **Warning:** AMD's official ROCm WSL2 support targets RDNA3 (RX 7000 series) and select
> Pro cards. The RX 6700 XT (RDNA2, gfx1031) may work with `HSA_OVERRIDE_GFX_VERSION=10.3.0`
> but this is unsupported. Native Linux (dual-boot) is the more reliable path for ROCm on RDNA2.

If you want to try WSL2:

1. **Install WSL2 Ubuntu 22.04** and the AMD Windows GPU driver (≥ 23.40)
2. **Inside WSL2, install ROCm:**
   ```bash
   # Add AMD ROCm apt repo (check https://rocm.docs.amd.com for current instructions)
   sudo apt install rocm-hip-sdk
   ```
3. **Install PyTorch with ROCm support** (replace rocm6.2 with latest available):
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
   ```
4. **Verify:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # Should print: True  AMD Radeon RX 6700 XT
   ```
5. If verified, `best_device()` will return `"cuda"` automatically and all scripts use the GPU.

## Project structure

```
src/volleybot/
  detection.py        # load CSV, filter_detections(), smoothed_mask(), fill_short_gaps()
  segmentation.py     # rally state machine (detection mask → Segment list)
  cutter.py           # ffmpeg video cutting and concat
  classification.py   # load classification CSV, classification_mask()

scripts/
  # Detection
  yolo_ball_detect_mps.py             # YOLO stream-based detection with MPS support (USE THIS)

  # Analysis
  analyze_detections.py               # timeline plots + sample frames from YOLO CSV
  compare_models.py                   # compare two YOLO models: detection rate, speed, side-by-side video
  compare_phases.py                   # comparison figures between two detection CSVs (phase 0 vs 1)

  # Labeling & fine-tuning (ball detection)
  sample_frames_for_labeling.py       # sample frames from videos for Roboflow upload
  preview_detections.py               # annotate sampled frames with YOLO boxes for QA
  finetune_yolo.py                    # fine-tune YOLOv8 on a Roboflow-exported dataset

  # Labeling & fine-tuning (play-state classifier)
  sample_frames_for_classification.py # sample frames for live/dead labeling (transition-biased)
  classify_frames.py                  # run YOLOv8-cls on a video → per-frame live/dead CSV
  finetune_classifier.py              # fine-tune YOLOv8n-cls on Roboflow Classification export

  # Pipeline
  cut_rallies.py                      # end-to-end: detect → segment → cut [MAIN ENTRYPOINT]
                                      # supports --classifier-model / --classifier-csv

data/                       # raw video files (gitignored)
outputs/                    # all generated files (gitignored)
  <stem>/
    detections.csv          # YOLO per-frame ball detections
    classification.csv      # per-frame live/dead classifier output
    annotated.mp4           # YOLO-annotated video
    rallies/                # individual rally clip files
    <stem>_all_rallies.mp4  # concatenated highlight reel
  labeling/                 # sampled frames for ball detection labeling
  labeling_cls/             # sampled frames for live/dead classification labeling
  analysis/                 # plots and comparison figures
tests/                      # pytest unit tests (36 passing)
```

## Downloading videos from YouTube

```bash
# Always specify format — default yt-dlp quality is too low (720p30)
yt-dlp -f "299+140" --merge-output-format mp4 -o "data/<name>.mp4" <URL>

# Or generically:
yt-dlp -f "bestvideo[height<=1080]+bestaudio/best" --merge-output-format mp4 -o "data/<name>.mp4" <URL>
```

## Running the pipeline

### Step 1: Detect ball in a video
```bash
uv run python scripts/yolo_ball_detect_mps.py --input outputs/clips/test_2min.mp4 --device cpu
```

### Step 2: Analyze detections
```bash
uv run python scripts/analyze_detections.py --csv outputs/clips/detections_mps.csv --video outputs/clips/test_2min.mp4
```

### Step 3: End-to-end cut (detect + segment + cut)
```bash
uv run python scripts/cut_rallies.py --input data/20220805g1.mp4 --concat
```

### Options
- Stream-copy (`-c copy`) for fast lossless cuts
- Re-encode option (`--reencode`) for frame-accurate cuts when needed

## Key design decisions

### Visual-first segmentation
Audio is not useful for rally segmentation in this footage (recreational play with continuous chatter). All segmentation is visual:
- Ball detection (YOLOv8x, COCO sports ball class 32)
- Short gap filling (≤0.5s) to handle missed frames during fast play
- State machine: DEAD → RALLY on ball detection; RALLY → DEAD after 3s gap
- Pre-roll 1.5s + post-roll 2.0s padding around raw detection windows

### Ball detection approach
YOLOv8 stream-based detection (`yolo_ball_detect_mps.py`):
- Speed at 640px: CPU=3.1 fps, MPS=6.7 fps; `model.predict(stream=True)` runs at ~8 fps end-to-end
- Always pass `device="mps"` for Apple Silicon
- Fine-tuned YOLOv8n reaches 38.6 fps on MPS (6× faster than COCO YOLOv8x)

## Phase 1 findings (from data/20220805g1.mp4, 2-min test clip at minutes 9-11)

### YOLO ball detection results
- **32.4% raw detection rate** on 7338 frames — in the "needs improvement" range
- **53.3% after 0.5s gap-fill** — shows the detector finds bursts of the ball but misses a lot
- False positive types found:
  - Floor markings detected as ball at conf=1.00 (ball on floor)
  - Player body parts (knee pads, shoes) at low conf
  - Max detected "ball" width = 188px (clearly not the ball)
- Filtering `conf≥0.25, width 15-80px, cy<72% of frame height` cleans this up significantly

### Segmentation results
| Method | Segments | Play time | Notes |
|--------|----------|-----------|-------|
| Raw YOLO (conf≥0.15) | 5 | 115s/120s | One 57s mega-segment (too large) |
| Filtered YOLO (conf≥0.25, size+pos) | 6 | 105s/120s | More precise breaks |
| BG-subtract (vel+arc filtered) | 3 | 116s/120s | Fewer, wider segments |
| Hybrid (union) | 1 | 120s/120s | Too aggressive, useless |

### What's working
- End-to-end pipeline: detect → segment → cut → concat runs correctly
- 6 rally clip files produced at `outputs/clips/rallies_filtered/`
- Concatenated highlight reel: `outputs/clips/test_2min_filtered_rallies.mp4`
- BG-subtract at 36-47fps is viable for quick pre-segmentation

### What needs improvement
1. **YOLO false positives on floor** — add `cy < 72% * frame_height` filter (done in `filter_detections()`)
2. **YOLO misses ball during fast rallies** — motion blur at 60fps kills detection on spikes
3. **Large merged segment (26-83s)** — need shorter `dead_gap_s` OR more reliable detection
4. **MPS speed** — fine-tuned YOLOv8n reaches 38.6 fps on MPS (see Phase 2 results)

### Recommended next steps
1. ~~Watch rally clips~~ — done, cut boundaries are reasonable
2. ~~Build labeled dataset and fine-tune~~ — done (see Phase 2 below)

## Fine-tuning workflow (Phase 2)

Goal: fine-tune YOLOv8n on volleyball-specific footage to replace the COCO
pretrained YOLOv8x. Even a nano model fine-tuned on domain data typically beats
a pretrained xlarge on this fixed-angle footage.

### Step 1: Sample frames for each venue
```bash
# Gym 1 (already downloaded) — uses detections CSV for biased sampling:
uv run python scripts/sample_frames_for_labeling.py \
    --inputs data/20220805g1.mp4 \
    --n-frames 200 \
    --csv outputs/clips/detections.csv

# Gym 2 and 3 (after downloading):
uv run python scripts/sample_frames_for_labeling.py \
    --inputs data/gym2.mp4 data/gym3.mp4 \
    --n-frames 150

# Output: outputs/labeling/<stem>/frame_XXXXXX.jpg
```

### Step 2: QA detection coverage (optional but recommended)
```bash
uv run python scripts/preview_detections.py \
    --frames-dir outputs/labeling/20220805g1 \
    --csv outputs/clips/detections.csv
# Output: outputs/labeling/20220805g1_preview/ — green=detected, red=missed
```

### Step 3: Label in Roboflow
1. Upload `outputs/labeling/` to Roboflow (one project, all venues)
2. Use **Auto Label → COCO sports ball** to pre-label, then correct mistakes
3. One class only: `volleyball`
4. Target: ~300-500 labeled frames total across all venues
5. Export as **YOLOv8** format → download zip → unzip to `data/roboflow_dataset/`

### Step 4: Fine-tune
```bash
uv run python scripts/finetune_yolo.py \
    --data data/roboflow_dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --device cpu
# Best weights: runs/detect/volleybot/weights/best.pt
```

### Step 5: Compare against baseline
```bash
uv run python scripts/compare_models.py \
    --video outputs/clips/test_10s.mp4 \
    --model-a yolov8x.pt \
    --model-b runs/detect/volleybot/weights/best.pt \
    --out-video outputs/analysis/model_comparison.mp4
```

### Step 6: Run full pipeline with fine-tuned model
```bash
uv run python scripts/cut_rallies.py \
    --input data/20220805g1.mp4 \
    --model runs/detect/volleybot/weights/best.pt \
    --concat
# Note: ball-class is auto-detected (0 for fine-tuned, 32 for COCO models)
```

## Phase 2 results (fine-tuned YOLOv8n, 2026-05-02)

Labeled ~500 frames across 3 gyms in Roboflow (1 class: `volleyball`).
Roboflow augmentation expanded to 915 train / 87 val / 44 test images.
Fine-tuned YOLOv8n for 50 epochs on MPS (~45 min).

| Metric | Phase 0 (YOLOv8x COCO) | Phase 1 (YOLOv8n fine-tuned) |
|--------|------------------------|------------------------------|
| Raw detection rate (2-min clip) | 32.4% | **47.1%** |
| Detection rate (10s clip) | 66.4% | **83.7%** |
| Inference speed (MPS) | 6.3 fps | **38.6 fps** |
| Best mAP50 (val) | — | 0.523 (epoch 30) |

Key takeaway: a nano model fine-tuned on domain footage **beats a pretrained xlarge** by +17pp
detection rate and runs **6× faster**. Recall is still ~48% — further improvement requires more
labeled data, especially for fast spikes (motion-blurred frames).

Fine-tuned weights: `runs/detect/volleybot/weights/best.pt` (gitignored — retrain with `finetune_yolo.py`)
Comparison figures: `outputs/analysis/phase_comparison/`

### Root cause of poor segmentation
The fine-tuned ball detector has high recall but also detects the ball during dead time (serve setup, ball rolling on floor). No conf/dead_gap combination fixed this — the detector can't distinguish live from dead play from ball position alone.

### Recommended next steps
1. ~~Train ball detector~~ — done; kept for tracking/highlights/statistics
2. Label live/dead frames and train a frame-level play-state classifier (Phase 3)

### Two-stage segmentation (Phase 3)
Ball detection alone gives poor segmentation when the fine-tuned model is too sensitive (detects ball during dead time too). Solution: decouple tracking from segmentation.

- **Ball detector** → `detections.csv` — used for future highlight reels + statistics
- **Frame classifier** → `classification.csv` — drives segmentation (live/dead, full-frame context)

The classifier sees the full frame (player postures, court state, ball position) and can distinguish a serve setup (dead) from an active rally (live) even when both have a ball visible.

## Phase 3 plan: frame-level live/dead classifier

Goal: replace ball-detection-based segmentation with a fine-tuned YOLOv8n-cls model
that classifies each frame as "live" (rally in progress) or "dead" (between points).

### Step 1: Sample frames for classification labeling
```bash
# With ball detection CSV (transition-biased sampling — most valuable):
uv run python scripts/sample_frames_for_classification.py \
    --inputs data/20220805g1.mp4 data/20230110.mp4 data/20250508g1.mp4 \
    --n-frames 200 \
    --csv outputs/20220805g1/detections.csv

# Output: outputs/labeling_cls/<stem>/frame_XXXXXX.jpg
```

### Step 2: Label in Roboflow (Classification project)
1. Create a new **Classification** project (not Detection)
2. Upload `outputs/labeling_cls/` frames
3. Label each frame as `live` or `dead`
   - **LIVE**: from the moment the server tosses the ball until the point ends
   - **DEAD**: everything else (rotation, server bouncing ball, ball retrieval)
   - Key transitions: serve toss (dead→live), ball hitting floor/going out (live→dead)
4. Target: ~400–600 frames total, well-balanced between live/dead
5. Export as **Folder Structure / YOLOv8** → unzip to `data/roboflow_classification/`

### Step 3: Fine-tune
```bash
uv run python scripts/finetune_classifier.py \
    --data data/roboflow_classification \
    --epochs 30 --device cpu
# Best weights: runs/classify/volleybot_cls/weights/best.pt
```

### Step 4: Classify frames
```bash
uv run python scripts/classify_frames.py \
    --input data/20220805g1.mp4 \
    --model runs/classify/volleybot_cls/weights/best.pt \
    --device cpu
# Output: outputs/20220805g1/classification.csv
```

### Step 5: Cut rallies using classifier
```bash
uv run python scripts/cut_rallies.py \
    --input data/20220805g1.mp4 \
    --classifier-model runs/classify/volleybot_cls/weights/best.pt \
    --concat
# Ball detection still runs → outputs/20220805g1/detections.csv
# Classification drives segmentation → outputs/20220805g1/rallies/
```
