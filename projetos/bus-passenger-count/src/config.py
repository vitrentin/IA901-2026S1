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
DEFAULT_DATA_YAML: Path = PROCESSED_DIR / "passenger-detection-bus" / "data.yaml"

# Public Roboflow dataset for pipeline tests; override via environment.
ROBOFLOW_WORKSPACE: str = os.environ.get("ROBOFLOW_WORKSPACE", "bus-project-frdgz")
ROBOFLOW_PROJECT: str = os.environ.get("ROBOFLOW_PROJECT", "passenger-detection-on-a-bus-qgljh")
ROBOFLOW_VERSION: int = int(os.environ.get("ROBOFLOW_VERSION", "1"))
ROBOFLOW_FORMAT: str = os.environ.get("ROBOFLOW_FORMAT", "yolov11")

SPLIT: dict[str, float] = {
    "train": 0.70,
    "valid": 0.20,
    "test":  0.10,
}

ROBOFLOW_PREPROC: dict[str, Any] = {
    "auto_orient": True,
    "resize": {"strategy": "fit_black_edges", "size": [640, 640]},
    "modify_classes": {"passenger": "person"},
    "isolate_objects": False,
    "grayscale": False,
    "auto_contrast": False,
    "filter_null": False,
    "static_crop": None,
    "tile": None,
    "dynamic_crop": False,
    "filter_by_tag": None,
}

ROBOFLOW_AUG: dict[str, Any] = {
    "flip": {"horizontal": True, "vertical": False},
    "mosaic": True,
    "brightness_pct": 15,
    "exposure_pct": 10,
    "cutout": {"count": 3, "size_pct": 10},
    "rotation_deg": 10,
    "blur_px": 1,
    "motion_blur": {"px": 50, "angle_deg": 0, "frames": 1},
    "hue_pct": 15,
    "saturation_pct": 25,
    "crop_pct_range": [0, 20],
    "shear_deg": 0,
    "noise_pct": 0,
    "camera_gain_sigma": 0,
    "probabilistic_grayscale_pct": 0,
}

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
        "split":            SPLIT,
        "roboflow_preproc": ROBOFLOW_PREPROC,
        "roboflow_aug":     ROBOFLOW_AUG,
        "yolo_aug_kwargs":  YOLO_AUG_KWARGS,
        "roboflow_workspace": ROBOFLOW_WORKSPACE,
        "roboflow_project":   ROBOFLOW_PROJECT,
        "roboflow_version":   ROBOFLOW_VERSION,
        "roboflow_format":    ROBOFLOW_FORMAT,
        "wandb_project":      WANDB_PROJECT,
        "wandb_entity":       WANDB_ENTITY,
        "seed":             DEFAULT_SEED,
    }
    if extra:
        payload.update(extra)
    return payload
