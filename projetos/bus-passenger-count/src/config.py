"""Caminhos do projeto e configurações do Weights & Biases."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR      = PROJECT_ROOT / "runs"

WANDB_PROJECT                  = os.environ.get("WANDB_PROJECT", "bus-passenger-count")
WANDB_ENTITY                   = os.environ.get("WANDB_ENTITY")
LOG_N_WANDB_TEST_PREDICTIONS   = int(os.environ.get("LOG_N_WANDB_TEST_PREDICTIONS", "10"))
WANDB_MAX_IMAGE_SIDE           = int(os.environ.get("WANDB_MAX_IMAGE_SIDE", "1280"))
WANDB_IMAGE_JPEG_QUALITY       = int(os.environ.get("WANDB_IMAGE_JPEG_QUALITY", "75"))

VIZ_CONF         = 0.25
VIZ_IOU          = 0.70
VIZ_AGNOSTIC_NMS = True

PERSON_CLASS_ID = 0

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
