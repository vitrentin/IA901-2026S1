"""Raw COCO-pretrained YOLO baseline (no fine-tuning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config

MODEL_KEY = "yolo_raw"


def build(cfg: dict[str, Any]):
    from ultralytics import YOLO
    weights = cfg.get("weights") or config.MODEL_WEIGHTS[MODEL_KEY]
    return YOLO(weights)


def train(cfg: dict[str, Any], data_yaml: str | Path, run_dir: str | Path) -> dict:
    raise NotImplementedError(
        "yolo_raw is a fine-tuning-free baseline. Run eval directly."
    )


def predict(
    cfg: dict[str, Any],
    data_yaml: str | Path,
    run_dir: str | Path,
    weights: str | Path | None = None,
) -> dict:
    """Evaluate base YOLO weights on the dataset test split."""
    data_yaml = Path(data_yaml).resolve()
    run_dir = Path(run_dir).resolve()

    model = build({"weights": weights or cfg.get("weights")})
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(run_dir),
        name=f"{MODEL_KEY}_eval",
        exist_ok=True,
    )

    flat: dict[str, float] = {}
    md = getattr(metrics, "results_dict", None) or {}
    for k, v in md.items():
        try:
            flat[f"test/{k}"] = float(v)
        except (TypeError, ValueError):
            continue
    return {"model": model, "metrics": flat}
