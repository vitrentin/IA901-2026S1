"""Project-wide defaults used by train/eval/data modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RUNS_DIR: Path = PROJECT_ROOT / "runs"

YOLO_AUG_KWARGS: dict[str, Any] = {
    "epochs":   300,
    "imgsz":    640,
    "device":   0,
    "batch":    16,
    "workers":  4,
    "amp":      True,
    "hsv_h":    0.015,
    "hsv_s":    0.25,
    "hsv_v":    0.15,
    "degrees":  10.0,
    "translate": 0.1,
    "scale":    0.2,
    "shear":    0.0,
    "fliplr":   0.5,
    "flipud":   0.0,
    "mosaic":   1.0,
    "erasing":  0.1,
}

DEFAULT_MODEL_NAME: str = "yolo"

MODEL_WEIGHTS: dict[str, str] = {
    "yolo":     "yolo11m.pt",
    "yolo_raw": "yolo11m.pt",
    "sam":      "sam2.1_b.pt",
    "sam_raw":  "sam2.1_b.pt",
}

WANDB_PROJECT: str = os.environ.get("WANDB_PROJECT", "bus-passenger-count")
WANDB_ENTITY: str | None = os.environ.get("WANDB_ENTITY")
LOG_N_TEST_IMAGES: int = 10

DEFAULT_SEED: int = 42


def to_wandb_config(extra: dict | None = None) -> dict:
    """Build a flat wandb config payload from static + per-run values."""
    payload = {
        "yolo_aug_kwargs": YOLO_AUG_KWARGS,
        "wandb_project":   WANDB_PROJECT,
        "wandb_entity":    WANDB_ENTITY,
        "seed":            DEFAULT_SEED,
    }
    if extra:
        payload.update(extra)
    return payload
