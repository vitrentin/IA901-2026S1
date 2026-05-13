"""Training entrypoint that creates run dirs and wandb runs."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config
import wandb_utils
from models import get_model


def _git_info() -> dict[str, str]:
    def _cmd(args):
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True,
                cwd=str(config.PROJECT_ROOT),
            ).strip()
        except Exception:
            return "unknown"
    return {
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
    }


def _make_run_dir(experiment_id: str) -> tuple[Path, str]:
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id   = experiment_id or "run"
    run_name  = f"{safe_id}_{ts}"
    run_dir   = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def _log_yolo_history(save_dir: Path) -> None:
    """Log YOLO epoch history from results.csv into the active wandb run."""
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return

    logged = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = None
            raw_epoch = row.get("epoch")
            if raw_epoch is not None:
                try:
                    step = int(float(raw_epoch)) + 1
                except (TypeError, ValueError):
                    step = None

            payload: dict[str, float] = {}
            for k, v in row.items():
                key = (k or "").strip()
                if not key or key == "epoch":
                    continue
                try:
                    payload[key] = float(v)
                except (TypeError, ValueError):
                    continue
            if payload:
                wandb_utils.log_metrics(payload, step=step)
                logged += 1

    if logged:
        print(f"train:    logged {logged} epoch rows from {csv_path.name}")


def run(
    model_name: str = config.DEFAULT_MODEL_NAME,
    cfg: dict[str, Any] | None = None,
    data_yaml: str | Path | None = None,
) -> Path:
    """Train `model_name` on `data_yaml` and return the run directory."""
    cfg = dict(cfg or {})

    experiment_id = cfg.get("experiment_id", "")
    notes         = cfg.get("notes")
    tags          = cfg.get("tags") or []

    run_dir, run_name = _make_run_dir(experiment_id)
    print(f"train:    run dir -> {run_dir}")

    wandb_cfg = config.to_wandb_config({
        "experiment_id":   experiment_id,
        "model_name":      model_name,
        "notes":           notes,
        "tags":            tags,
        **_git_info(),
        **{k: v for k, v in cfg.items()
           if k not in ("experiment_id", "notes", "tags")},
    })
    wandb_utils.init_run(
        wandb_config = wandb_cfg,
        run_name     = run_name,
        run_dir      = run_dir,
        notes        = notes,
        tags         = tags or None,
    )

    model_mod = get_model(model_name)
    t0 = time.time()
    try:
        result = model_mod.train(cfg, data_yaml, run_dir)
    except NotImplementedError as e:
        print(f"train:    {model_name} has no training step ({e}). Skipping.")
        result = {"weights": cfg.get("weights") or config.MODEL_WEIGHTS.get(model_name, "")}
    dt = time.time() - t0
    print(f"train:    finished in {dt:.1f}s")

    weights_path = result.get("weights", "")
    save_dir = result.get("save_dir", "")
    if save_dir:
        _log_yolo_history(Path(save_dir))

    if weights_path:
        (run_dir / "weights.txt").write_text(str(weights_path) + "\n")
    (run_dir / "train_result.json").write_text(json.dumps(
        {k: str(v) for k, v in result.items()}, indent=2
    ))

    wandb_utils.log_metrics({"train/duration_sec": dt})
    return run_dir
