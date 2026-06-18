"""Avaliacao de experimentos."""

import json
from pathlib import Path

from src import config
from src import datasets
from src import wandb_utils
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _headline_metrics(results_dict, count_metrics):
    """Small, slide-ready metric set for a single-class detection/counting task."""
    p = float(results_dict.get("metrics/precision(B)", 0.0))
    r = float(results_dict.get("metrics/recall(B)", 0.0))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "mAP50":      float(results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95":   float(results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "precision":  p,
        "recall":     r,
        "F1":         f1,
        "count_mae":  count_metrics["count_mae"],
        "count_rmse": count_metrics["count_rmse"],
    }


def _count_metrics(model, data_spec):
    """Per-image count error at the operating threshold (the counting objective)."""
    img_dir   = wandb_utils._resolve_test_image_dir(data_spec)
    label_dir = img_dir.parent / "labels"
    abs_err, sq_err = [], []
    for r in model.predict(
        source       = str(img_dir),
        conf         = config.VIZ_CONF,
        iou          = config.VIZ_IOU,
        classes      = [config.PERSON_CLASS_ID],
        agnostic_nms = config.VIZ_AGNOSTIC_NMS,
        stream       = True,
        verbose      = False,
    ):
        pred_n = len(r.boxes) if r.boxes is not None else 0
        gt_n   = len(wandb_utils._read_yolo_labels(label_dir / f"{Path(r.path).stem}.txt"))
        d = abs(pred_n - gt_n)
        abs_err.append(d)
        sq_err.append(d * d)
    n = len(abs_err) or 1
    return {
        "count_mae":  sum(abs_err) / n,
        "count_rmse": (sum(sq_err) / n) ** 0.5,
        "n_images":   len(abs_err),
    }


def _log_curves(save_dir, dataset_name):
    """Upload Ultralytics' PR and F1-confidence curves (justify the threshold)."""
    for fname, key in [("PR_curve.png", "pr_curve"), ("F1_curve.png", "f1_curve")]:
        png = Path(save_dir) / fname
        if png.exists():
            wandb_utils.log_image(png, key=f"test/{dataset_name}/{key}")


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
        classes  = [config.PERSON_CLASS_ID],
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
    """Avalia um unico dataset. Retorna (headline, full_metrics)."""
    data_spec = datasets.prepare([dataset_name], stage=stage)
    split     = "test" if "test" in data_spec else "val"

    results = model.val(
        data     = _resolve_data_arg(data_spec, run_dir),
        split    = split,
        classes  = [config.PERSON_CLASS_ID],
        project  = str(run_dir),
        name     = f"eval_{dataset_name}",
        exist_ok = True,
    )
    md       = {k: float(v) for k, v in (getattr(results, "results_dict", None) or {}).items()}
    counts   = _count_metrics(model, data_spec)
    headline = _headline_metrics(md, counts)

    wandb_utils.log_metrics({f"test/{dataset_name}/{k}": v for k, v in headline.items()})
    _log_curves(results.save_dir, dataset_name)

    print(f"eval:     {dataset_name} ({split}):")
    for k, v in headline.items():
        print(f"            {k:12s} {v:.4f}")

    if n_wandb_samples > 0:
        wandb_utils.log_test_predictions(
            predictor = model,
            data_spec = data_spec,
            n         = n_wandb_samples,
            panel_key = f"test/{dataset_name}/predictions",
        )
    return headline, {**md, **counts, "F1": headline["F1"]}


def run_experiment(cfg, n_wandb_samples=config.LOG_N_WANDB_TEST_PREDICTIONS, weights_override=""):
    """Avalia o experimento em cada dataset de `eval_datasets`."""
    from ultralytics import YOLO

    wandb_utils.disable_ultralytics_autolog()
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
            "dataset_stage": stage,
            "weights":       str(weights),
            "eval_datasets": eval_datasets,
        },
        run_name=f"{run_dir.name}_eval",
        run_dir=run_dir,
    )

    model = YOLO(str(weights))
    columns     = ["dataset", "mAP50", "mAP50-95", "precision", "recall", "F1",
                   "count_mae", "count_rmse"]
    all_metrics = {}
    headlines   = {}
    for dataset_name in eval_datasets:
        headline, full = _eval_one_dataset(
            model, dataset_name, stage, run_dir, n_wandb_samples
        )
        headlines[dataset_name]   = headline
        all_metrics[dataset_name] = full

    (run_dir / "test_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    table_rows = [[ds] + [h[c] for c in columns[1:]] for ds, h in headlines.items()]
    wandb_utils.log_summary_table(table_rows, columns)

    # No reprefixed summary: `wandb.log("test/<ds>/<metric>")` already populates
    # the runs-table summary columns, so re-logging them would just duplicate.
    wandb_utils.finish_run()
    return all_metrics
