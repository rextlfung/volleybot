"""GPU device auto-detection utility."""

from __future__ import annotations


def best_device() -> str:
    """Return the best available PyTorch device string: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
