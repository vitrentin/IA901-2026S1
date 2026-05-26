"""Training entrypoint — infraestrutura de run dir e wandb."""

import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from src import config
from src import wandb_utils


def _git_info():
    def _cmd(args):
        return subprocess.check_output(
            args, stderr=subprocess.DEVNULL, text=True,
            cwd=str(config.PROJECT_ROOT),
        ).strip()
    return {
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
    }


def _make_run_dir(experiment_id):
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_id or 'run'}_{ts}"
    run_dir  = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def _log_yolo_history(save_dir):
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return
    logged = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["epoch"])) + 1 if row.get("epoch") else None
            payload = {}
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
        print(f"train:    {logged} épocas registradas no wandb")


def run(experiment_id, model, data_yaml, train_config=None):
    """Cria run dir, inicia wandb, treina o modelo e retorna o run dir."""
    run_dir, run_name = _make_run_dir(experiment_id)
    print(f"train:    run dir -> {run_dir}")

    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "data_yaml":     str(data_yaml),
            "train_config":  train_config,
            **_git_info(),
        },
        run_name=run_name,
        run_dir=run_dir,
    )

    params = dict(train_config or {})
    params["data"]     = str(data_yaml)
    params["project"]  = str(run_dir)
    params["name"]     = "train"
    params["exist_ok"] = True

    t0 = time.time()
    results = model.train(**params)
    dt = time.time() - t0
    print(f"train:    concluído em {dt:.1f}s")

    save_dir = Path(results.save_dir)
    best     = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"

    _log_yolo_history(save_dir)
    (run_dir / "weights.txt").write_text(str(best) + "\n")
    wandb_utils.log_metrics({"train/duration_sec": dt})
    return run_dir
