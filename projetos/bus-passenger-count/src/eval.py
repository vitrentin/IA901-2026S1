"""Avaliacao de experimentos."""

import json
from pathlib import Path

from src import config
from src import datasets
from src import wandb_utils
import yaml


def find_run(experiment_id):
    """Retorna o run mais recente e o arquivo de pesos."""
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


def run(
    experiment_id,
    model,
    data_spec,
    n_wandb_samples=config.LOG_N_WANDB_TEST_PREDICTIONS,
    run_dir=None,
):
    """Avalia o modelo no split de teste."""
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

    if n_wandb_samples > 0:
        wandb_utils.log_test_predictions(
            predictor = model,
            data_spec = data_spec,
            n         = n_wandb_samples,
        )

    wandb_utils.finish_run(metrics)
    return metrics


def _eval_one_dataset(model, dataset_name, stage, run_dir, n_wandb_samples):
    """Avalia um unico dataset."""
    data_spec = datasets.prepare([dataset_name], stage=stage)
    split     = "test" if "test" in data_spec else "val"

    results = model.val(
        data     = _resolve_data_arg(data_spec, run_dir),
        split    = split,
        project  = str(run_dir),
        name     = f"eval_{dataset_name}",
        exist_ok = True,
    )
    md      = getattr(results, "results_dict", None) or {}
    metrics = {k: float(v) for k, v in md.items()}

    wandb_utils.log_metrics({f"test/{dataset_name}/{k}": v for k, v in metrics.items()})
    print(f"eval:     {dataset_name} ({split}):")
    for k, v in metrics.items():
        print(f"            {k:32s} {v:.4f}")

    if n_wandb_samples > 0:
        wandb_utils.log_test_predictions(
            predictor = model,
            data_spec = data_spec,
            n         = n_wandb_samples,
            panel_key = f"test/{dataset_name}/predictions",
        )
    return metrics


def run_experiment(cfg, n_wandb_samples=config.LOG_N_WANDB_TEST_PREDICTIONS, weights_override=""):
    """Avalia o experimento em cada dataset de `eval_datasets`."""
    from ultralytics import YOLO

    experiment_id = cfg["experiment_id"]
    stage         = cfg.get("dataset_stage", "interim")
    eval_datasets = cfg["eval_datasets"]

    run_dir, weights = find_run(experiment_id)
    weights = weights_override or weights or cfg["weights"]
    if run_dir is None:
        run_dir = config.RUNS_DIR / f"eval_{experiment_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"eval:     nenhum run de treino — avaliando pesos base '{weights}'")
    print(f"eval:     pesos -> {weights}")

    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "strategy":      cfg.get("strategy"),
            "weights":       str(weights),
            "eval_datasets": eval_datasets,
        },
        run_name=f"{run_dir.name}_eval",
        run_dir=run_dir,
    )

    model = YOLO(str(weights))
    all_metrics = {}
    for dataset_name in eval_datasets:
        all_metrics[dataset_name] = _eval_one_dataset(
            model, dataset_name, stage, run_dir, n_wandb_samples
        )

    (run_dir / "test_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    summary = {}
    for dataset_name, metrics in all_metrics.items():
        for k, v in metrics.items():
            if "mAP50(B)" in k or "mAP50-95(B)" in k:
                summary[f"{dataset_name}/{k}"] = v
    wandb_utils.finish_run(summary)
    return all_metrics
