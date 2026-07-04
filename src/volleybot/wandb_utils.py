from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from typing import Generator

import wandb


@contextmanager
def wandb_run(args: argparse.Namespace) -> Generator[None, None, None]:
    """Context manager: init wandb if args.wandb_project is set, finish on exit."""
    active = bool(getattr(args, "wandb_project", ""))
    if active:
        wandb.init(
            project=args.wandb_project,
            name=args.name,
            config={
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "patience": args.patience,
            },
        )
    try:
        yield
    finally:
        if active:
            exc_type = sys.exc_info()[0]
            try:
                wandb.finish(exit_code=0 if exc_type is None else 1)
            except Exception:
                if exc_type is None:
                    raise
