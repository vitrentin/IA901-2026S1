"""Caminhos do projeto e configurações do Weights & Biases."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Caminhos do projeto
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RUNS_DIR: Path = PROJECT_ROOT / "runs"

# Configuração para o Weights and Biases
WANDB_PROJECT: str = os.environ.get("WANDB_PROJECT", "bus-passenger-count")
WANDB_ENTITY: Optional[str] = os.environ.get("WANDB_ENTITY")
LOG_N_TEST_IMAGES: int = 10

ROBOFLOW_API_KEY: str = os.environ.get("ROBOFLOW_API_KEY", "")
