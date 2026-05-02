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
brew install ffmpeg yt-dlp
```

## Project structure

```
src/volleybot/
  detection.py      # load CSV, filter_detections() (conf/size/position), smoothed_mask()
  segmentation.py   # rally state machine (detection mask → Segment list)
  cutter.py         # ffmpeg video cutting and concat
  tracking.py       # trajectory filters: velocity + parabolic arc

scripts/
  # Detection
  yolo_ball_detect_mps.py       # YOLO stream-based detection with MPS support (USE THIS)
  yolo_ball_detect.py           # original per-frame YOLO (slower, legacy)
  bg_subtract_detect.py         # background subtraction detector (fast, noisy)
  quick_speed_test.py           # CPU vs MPS inference speed benchmark

  # Analysis
  analyze_detections.py         # timeline plots + sample frames from YOLO CSV
  analyze_bg_filtered.py        # bg-subtract with velocity/arc filters
  analyze_yolo_filtered.py      # YOLO filter parameter comparison
  compare_detectors.py          # side-by-side YOLO vs bg-subtract plots
  compare_models.py             # compare two YOLO models: detection rate, speed, side-by-side video
  compare_phases.py             # generate comparison figures between two detection CSVs (phase 0 vs 1)
  plot_audio.py                 # audio waveform/spectrogram visualizer

  # Labeling & fine-tuning
  sample_frames_for_labeling.py # sample frames from videos for Roboflow upload
  preview_detections.py         # annotate sampled frames with YOLO boxes for QA
  finetune_yolo.py              # fine-tune YOLOv8 on a Roboflow-exported dataset

  # Pipeline
  cut_rallies.py                # end-to-end: detect → filter → segment → cut [MAIN ENTRYPOINT]
  cut_filtered_rallies.py       # one-off: cut with filtered YOLO detections
  post_yolo_analysis.py         # run full analysis after YOLO finishes

data/                       # raw video files (gitignored)
outputs/                    # all generated files (gitignored)
  frames/                   # sampled frames for inspection
  audio/                    # extracted audio + plots
  clips/
    test_2min.mp4                       # 2-min test clip (minutes 9-11)
    test_2min_annotated.mp4             # YOLO annotated video
    test_2min_all_rallies.mp4           # highlight reel (raw YOLO, 5 segments)
    test_2min_filtered_rallies.mp4      # highlight reel (filtered YOLO, 6 segments) ← review this
    rallies/                            # individual clips (raw)
    rallies_filtered/                   # individual clips (filtered) ← review these
    detections.csv                      # YOLO per-frame detections
    detections_bg.csv                   # bg-subtract detections
  analysis/
    detection_timeline.png              # YOLO detection over time
    rally_segments.png                  # segmentation visualization
    detector_comparison.png             # YOLO vs bg-subtract side-by-side
    sample_frames/                      # detected + missed example frames
tests/                      # pytest unit tests (28 passing)
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
uv run python scripts/yolo_ball_detect_mps.py --input outputs/clips/test_2min.mp4 --device mps
```

### Step 2: Analyze detections
```bash
uv run python scripts/analyze_detections.py --csv outputs/clips/detections_mps.csv --video outputs/clips/test_2min.mp4
```

### Step 3: End-to-end cut (detect + segment + cut)
```bash
uv run python scripts/cut_rallies.py --input data/20220805g1.mp4 --concat
```

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
4. **MPS speed** — see benchmark results in `scripts/benchmark_detection.py`

### Recommended next steps
1. ~~Watch rally clips~~ — done, cut boundaries are reasonable
2. ~~Build labeled dataset and fine-tune~~ — done (see Phase 2 below)

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

### Recommended next steps
1. Run full pipeline on all 3 gym videos with the fine-tuned model
2. Review rally cuts — identify failure modes (missed spikes, false positives)
3. Add more labeled frames for hard cases, retrain for Phase 3

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
    --device mps
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

## Key design decisions

### Visual-first segmentation
Audio is not useful for rally segmentation in this footage (recreational play with continuous chatter). All segmentation is visual:
- Ball detection (YOLOv8x, COCO sports ball class 32)
- Short gap filling (≤0.5s) to handle missed frames during fast play
- State machine: DEAD → RALLY on ball detection; RALLY → DEAD after 3s gap
- Pre-roll 1.5s + post-roll 2.0s padding around raw detection windows

### Ball detection approach
Two detectors are implemented — use both and compare:

**1. YOLOv8x (COCO sports ball class 32)**
- More precise, fewer false positives
- Speed at 640px: CPU=3.1 fps, MPS=6.7 fps (2.1× speedup) — bottleneck is per-frame Python overhead
- The `model.predict(stream=True)` pipeline in `yolo_ball_detect_mps.py` runs at ~8 fps end-to-end
- Always pass `device="mps"` for Apple Silicon
- Script: `scripts/yolo_ball_detect_mps.py`

**2. Background subtraction (MOG2 + static diff)**
- Blazing fast: 36-47 fps on CPU (no GPU needed)
- 94% raw recall but ~78% of those are false positives (player arms/bodies)
- After velocity filter: drops to ~22% detection but produces plausible rally segments
- After arc filter: further noise reduction, same segments
- Key parameters: `SEARCH_BOT_FRAC=0.75` (search y=10-75%), `MAX_JUMP_PX=90`
- Script: `scripts/bg_subtract_detect.py`

**Trajectory filtering** (`src/volleybot/tracking.py`):
- `apply_velocity_filter()`: remove detections that jump >90px/frame (physically impossible)
- `apply_arc_filter()`: remove detections that don't fit local parabolic arc
- Both filters work on any detector's output

**Recommended pipeline**: use YOLOv8 as primary + bg subtract (velocity-filtered) as gap-filler.

### Video cutting
- Stream-copy (`-c copy`) for fast lossless cuts
- Re-encode option (`--reencode`) for frame-accurate cuts when needed
