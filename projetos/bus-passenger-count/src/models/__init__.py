"""Model registry used by train/eval entrypoints."""

from __future__ import annotations

import importlib
from types import ModuleType

_AVAILABLE = ("yolo", "yolo_raw")


def get_model(name: str) -> ModuleType:
    """Return the model module registered under `name`."""
    if name not in _AVAILABLE:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_AVAILABLE)}"
        )
    return importlib.import_module(f"models.{name}")


def available() -> tuple[str, ...]:
    return _AVAILABLE
