"""Caminhos do projeto e configurações do Weights & Biases."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR      = PROJECT_ROOT / "data"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR      = PROJECT_ROOT / "runs"

WANDB_PROJECT                  = os.environ.get("WANDB_PROJECT", "bus-passenger-count")
WANDB_ENTITY                   = os.environ.get("WANDB_ENTITY")
LOG_N_WANDB_TEST_PREDICTIONS   = 10

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
