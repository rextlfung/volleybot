"""Plot waveform + log-mel spectrogram for the sample audio.

Run: uv run python scripts/plot_audio.py
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO = Path("outputs/audio/audio_16k_mono.wav")
OUT_DIR = Path("outputs/audio")


def main() -> None:
    y, sr = librosa.load(AUDIO, sr=None, mono=True)
    duration = len(y) / sr
    print(f"loaded {AUDIO}: {len(y):,} samples, sr={sr}, duration={duration:.1f}s")

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    print(f"rms: mean={rms.mean():.4f} max={rms.max():.4f}")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(np.arange(len(y)) / sr, y, linewidth=0.3, color="steelblue")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("waveform")
    axes[0].set_ylim(-1, 1)

    axes[1].plot(rms_times, rms, linewidth=0.6, color="darkorange")
    axes[1].set_ylabel("RMS energy")
    axes[1].set_title("short-term loudness (RMS)")

    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
        ax=axes[2], fmax=sr // 2, cmap="magma",
    )
    axes[2].set_title("log-mel spectrogram")
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB", pad=0.01)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()

    out_path = OUT_DIR / "audio_overview.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
