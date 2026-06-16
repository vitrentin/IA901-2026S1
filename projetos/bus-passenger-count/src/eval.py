"""Evaluation entrypoint — infraestrutura de wandb e métricas."""

import json
from pathlib import Path

from src import config
from src import wandb_utils
import yaml


def find_run(experiment_id):
    """Retorna (run_dir, weights_path) para o experiment_id dado, ou (None, None)."""
    if not config.RUNS_DIR.exists():
        return None, None
    matches = sorted(
        [d for d in config.RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith(f"{experiment_id}_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not matches:
        return None, None
    run_dir = matches[-1]
    wfile   = run_dir / "weights.txt"
    weights = wfile.read_text().strip() if wfile.exists() else None
    return run_dir, weights


def _resolve_data_arg(data_spec, run_dir):
    if isinstance(data_spec, dict):
        data_file = Path(run_dir) / "_data_runtime.yaml"
        data_file.write_text(yaml.safe_dump(data_spec, sort_keys=False), encoding="utf-8")
        return str(data_file)
    return str(data_spec)


def run(experiment_id, model, data_spec, n_samples=config.LOG_N_TEST_IMAGES, run_dir=None):
    """Avalia o modelo no split de teste e retorna o dicionário de métricas."""
    if run_dir is None:
        run_dir, _ = find_run(experiment_id)
    if run_dir is None:
        run_dir = config.RUNS_DIR / f"eval_{experiment_id}"
        print(f"eval:     nenhum run de treino encontrado — criando {run_dir.name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "data":          data_spec,
        },
        run_name=f"{run_dir.name}_eval",
        run_dir=run_dir,
    )

    results = model.val(
        data     = _resolve_data_arg(data_spec, run_dir),
        split    = "test",
        project  = str(run_dir),
        name     = "eval",
        exist_ok = True,
    )

    md      = getattr(results, "results_dict", None) or {}
    metrics = {f"test/{k}": float(v) for k, v in md.items()}

    if metrics:
        wandb_utils.log_metrics(metrics)
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("eval:     métricas:")
        for k, v in metrics.items():
            print(f"          {k:40s} {v:.4f}")

    if n_samples > 0:
        wandb_utils.log_test_predictions(
            predictor = model,
            data_spec = data_spec,
            n         = n_samples,
        )

    wandb_utils.finish_run(metrics)
    return metrics
