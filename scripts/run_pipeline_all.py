"""Run cut_rallies.py on all 3 game videos sequentially."""

import subprocess
import sys
from pathlib import Path

VIDEOS = ["data/20220805g1.mp4", "data/20230110.mp4", "data/20250508g1.mp4"]
ROOT = Path(__file__).parent.parent

for video in VIDEOS:
    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {video}", flush=True)
    print(f"{'='*60}", flush=True)
    result = subprocess.run(
        [sys.executable, "scripts/cut_rallies.py", "--input", video, "--concat"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"FAILED: {video} (exit {result.returncode})", flush=True)
    else:
        print(f"DONE: {video}", flush=True)

print("\nAll videos processed.", flush=True)
