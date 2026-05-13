"""Evaluation entrypoint for trained or baseline models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
import wandb_utils
from models import get_model


def _resolve_weights(run_dir: Path, override: str | Path | None) -> str | None:
    if override:
        return str(override)
    wfile = run_dir / "weights.txt"
    if wfile.exists():
        return wfile.read_text().strip() or None
    return None


def run(
    run_dir: str | Path,
    data_yaml: str | Path,
    model_name: str = config.DEFAULT_MODEL_NAME,
    n_samples: int = config.LOG_N_TEST_IMAGES,
    cfg: dict[str, Any] | None = None,
    weights: str | Path | None = None,
) -> dict:
    """Evaluate and return the flat metrics dict logged to wandb."""
    cfg       = dict(cfg or {})
    run_dir   = Path(run_dir).resolve()
    data_yaml = Path(data_yaml).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_path = _resolve_weights(run_dir, weights)
    print(f"eval:     weights -> {weights_path or '(model default)'}")

    if not wandb_utils.is_active():
        notes = cfg.get("notes")
        tags = cfg.get("tags") or None
        cfg_extra = {
            k: v for k, v in cfg.items()
            if k not in ("notes", "tags")
        }
        wandb_utils.init_run(
            wandb_config = config.to_wandb_config({
                "experiment_id": run_dir.name,
                "model_name":    model_name,
                "mode":          "eval_only",
                **cfg_extra,
            }),
            run_name = f"{run_dir.name}_eval",
            run_dir  = run_dir,
            notes    = notes,
            tags     = tags,
        )

    model_mod = get_model(model_name)
    try:
        result = model_mod.predict(cfg, data_yaml, run_dir, weights=weights_path)
    except NotImplementedError as e:
        print(f"eval:     {model_name}.predict is not implemented yet ({e}).")
        wandb_utils.finish_run({})
        return {}

    metrics    = result.get("metrics", {}) if isinstance(result, dict) else {}
    yolo_model = result.get("model") if isinstance(result, dict) else None

    if metrics:
        wandb_utils.log_metrics(metrics)
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("eval:     test metrics:")
        for k, v in metrics.items():
            print(f"          {k:40s} {v:.4f}")

    if yolo_model is not None and n_samples > 0:
        wandb_utils.log_test_predictions(
            predictor = yolo_model,
            data_yaml = data_yaml,
            n         = n_samples,
        )

    wandb_utils.finish_run(metrics)
    return metrics
