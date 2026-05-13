"""YOLO model adapter used by train/eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config

MODEL_KEY = "yolo"


def build(cfg: dict[str, Any]):
    """Instantiate a YOLO model with selected weights."""
    from ultralytics import YOLO
    weights = cfg.get("weights") or config.MODEL_WEIGHTS[MODEL_KEY]
    print(f"yolo:     loading base weights -> {weights}")
    return YOLO(weights)


def train(cfg: dict[str, Any], data_yaml: str | Path, run_dir: str | Path) -> dict:
    """Train YOLO and return output paths."""
    data_yaml = Path(data_yaml).resolve()
    run_dir   = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build(cfg)

    aug = dict(config.YOLO_AUG_KWARGS)
    aug.update(cfg.get("yolo_overrides", {}))
    aug["data"]    = str(data_yaml)
    aug["project"] = str(run_dir)
    aug["name"]    = MODEL_KEY
    aug["exist_ok"] = True

    print(f"yolo:     training for {aug.get('epochs')} epochs on {data_yaml}")
    results = model.train(**aug)

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else run_dir / MODEL_KEY
    best     = save_dir / "weights" / "best.pt"
    if not best.exists():
        last = save_dir / "weights" / "last.pt"
        best = last if last.exists() else best
    print(f"yolo:     done. best weights -> {best}")
    return {
        "weights":  str(best),
        "save_dir": str(save_dir),
    }


def predict(
    cfg: dict[str, Any],
    data_yaml: str | Path,
    run_dir: str | Path,
    weights: str | Path | None = None,
) -> dict:
    """Run test split evaluation and return flat metrics."""
    from ultralytics import YOLO
    data_yaml = Path(data_yaml).resolve()
    run_dir   = Path(run_dir).resolve()

    weights = weights or cfg.get("weights") or config.MODEL_WEIGHTS[MODEL_KEY]
    print(f"yolo:     loading weights for eval -> {weights}")
    model = YOLO(str(weights))

    metrics = model.val(
        data     = str(data_yaml),
        split    = "test",
        project  = str(run_dir),
        name     = f"{MODEL_KEY}_eval",
        exist_ok = True,
    )

    flat: dict[str, float] = {}
    md = getattr(metrics, "results_dict", None) or {}
    for k, v in md.items():
        try:
            flat[f"test/{k}"] = float(v)
        except (TypeError, ValueError):
            continue
    return {"model": model, "metrics": flat}
