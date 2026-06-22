"""Avaliacao de experimentos."""

import json
from pathlib import Path

from src import config
from src import datasets
from src import wandb_utils
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FULL_DATASET_TEST_ONLY = "onibus-unicamp-private"


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
        "count_me":   count_metrics["count_me"],  # signed error
        "count_rmse": count_metrics["count_rmse"],
        "tp":         count_metrics["tp"],
        "fp":         count_metrics["fp"],
        "fn":         count_metrics["fn"],
        "pred_n":     count_metrics["pred_n"],
        "gt_n":       count_metrics["gt_n"],
        "mean_conf":  count_metrics["mean_conf"],
        "precision_cnt": count_metrics["precision_cnt"],
        "recall_cnt":    count_metrics["recall_cnt"],
        "f1_cnt":        count_metrics["f1_cnt"],
    }


def _iou_matrix(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return []
    out = []
    for p in pred_boxes:
        px1, py1, px2, py2 = p
        pa = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
        row = []
        for g in gt_boxes:
            gx1, gy1, gx2, gy2 = g
            ga = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)
            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
            union = pa + ga - inter
            row.append(inter / union if union > 0 else 0.0)
        out.append(row)
    return out


def _greedy_tp(pred_boxes, gt_boxes, iou_thr=0.5):
    if not pred_boxes or not gt_boxes:
        return 0
    ious = _iou_matrix(pred_boxes, gt_boxes)
    candidates = []
    for pi, row in enumerate(ious):
        for gi, iou in enumerate(row):
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    if not candidates:
        return 0
    candidates.sort(reverse=True)
    used_p = set()
    used_g = set()
    tp = 0
    for _, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
    return tp


def _count_metrics(model, data_spec, iou_match_thr=0.5):
    """Per-image count error at the operating threshold (the counting objective)."""
    abs_err, sq_err, signed_err = [], [], []
    total_pred = 0
    total_gt = 0
    total_tp = 0
    all_conf = []
    if isinstance(data_spec, dict) and data_spec.get("_dataset_name") == FULL_DATASET_TEST_ONLY:
        split_dirs = data_spec.get("test") or []
        if not isinstance(split_dirs, list):
            split_dirs = [split_dirs]
        image_dirs = [Path(p) for p in split_dirs]
    else:
        image_dirs = [Path(wandb_utils._resolve_test_image_dir(data_spec))]

    for img_dir in image_dirs:
        label_dir = wandb_utils._resolve_label_dir(img_dir)
        for r in model.predict(
            source       = str(img_dir),
            conf         = config.VIZ_CONF,
            iou          = config.VIZ_IOU,
            classes      = [config.PERSON_CLASS_ID],
            agnostic_nms = config.VIZ_AGNOSTIC_NMS,
            stream       = True,
            verbose      = False,
        ):
            pred_boxes = r.boxes.xyxyn.cpu().tolist() if (r.boxes is not None and len(r.boxes) > 0) else []
            pred_conf = r.boxes.conf.cpu().tolist() if (r.boxes is not None and len(r.boxes) > 0) else []
            gt_labels = wandb_utils._read_yolo_labels(label_dir / f"{Path(r.path).stem}.txt")
            gt_boxes = []
            for _, cx, cy, bw, bh in gt_labels:
                x1 = cx - (bw / 2.0)
                y1 = cy - (bh / 2.0)
                x2 = cx + (bw / 2.0)
                y2 = cy + (bh / 2.0)
                gt_boxes.append([x1, y1, x2, y2])

            pred_n = len(pred_boxes)
            gt_n   = len(gt_boxes)
            tp_n = _greedy_tp(pred_boxes, gt_boxes, iou_thr=iou_match_thr)

            signed_d = pred_n - gt_n  # positive = overcounting, negative = undercounting
            d = abs(signed_d)
            abs_err.append(d)
            sq_err.append(d * d)
            signed_err.append(signed_d)
            total_pred += pred_n
            total_gt += gt_n
            total_tp += tp_n
            all_conf.extend(pred_conf)
    n = len(abs_err) or 1
    fp = total_pred - total_tp
    fn = total_gt - total_tp
    precision_cnt = (total_tp / total_pred) if total_pred > 0 else 0.0
    recall_cnt = (total_tp / total_gt) if total_gt > 0 else 0.0
    f1_cnt = (
        (2 * precision_cnt * recall_cnt / (precision_cnt + recall_cnt))
        if (precision_cnt + recall_cnt) > 0 else 0.0
    )
    return {
        "count_mae":  sum(abs_err) / n,
        "count_me":   sum(signed_err) / n,  # signed mean error (+ = overcount, - = undercount)
        "count_rmse": (sum(sq_err) / n) ** 0.5,
        "n_images":   len(abs_err),
        "tp":         total_tp,
        "fp":         fp,
        "fn":         fn,
        "pred_n":     total_pred,
        "gt_n":       total_gt,
        "mean_conf":  (sum(all_conf) / len(all_conf)) if all_conf else 0.0,
        "precision_cnt": precision_cnt,
        "recall_cnt":    recall_cnt,
        "f1_cnt":        f1_cnt,
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


def _eval_one_dataset(model, dataset_name, stage, run_dir, n_wandb_samples, private_eval_scope="test_only"):
    """Avalia um unico dataset. Retorna (headline, full_metrics)."""
    data_spec = datasets.prepare([dataset_name], stage=stage)
    if dataset_name == FULL_DATASET_TEST_ONLY and private_eval_scope == "all_splits":
        root = config.DATA_DIR / stage / dataset_name
        all_split_imgs = []
        for split in ("train", "valid", "test"):
            img_dir = datasets._split_images(root, split)
            if img_dir:
                all_split_imgs.append(str(img_dir))
        if all_split_imgs:
            data_spec = dict(data_spec)
            data_spec["test"] = all_split_imgs if len(all_split_imgs) > 1 else all_split_imgs[0]
    if isinstance(data_spec, dict):
        data_spec["_dataset_name"] = dataset_name
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
    private_eval_scope = cfg.get("private_eval_scope", "test_only")

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
            "private_eval_scope": private_eval_scope,
        },
        run_name=f"{run_dir.name}_eval",
        run_dir=run_dir,
    )

    model = YOLO(str(weights))
    columns     = [
        "experiment_id", "run_name", "dataset",
        "mAP50", "mAP50-95", "precision", "recall", "F1",
        "count_mae", "count_me", "count_rmse",
        "tp", "fp", "fn", "pred_n", "gt_n", "mean_conf",
        "precision_cnt", "recall_cnt", "f1_cnt",
    ]
    all_metrics = {}
    headlines   = {}
    for dataset_name in eval_datasets:
        headline, full = _eval_one_dataset(
            model, dataset_name, stage, run_dir, n_wandb_samples, private_eval_scope=private_eval_scope
        )
        headlines[dataset_name]   = headline
        all_metrics[dataset_name] = full

    (run_dir / "test_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    table_rows = [
        [experiment_id, run_dir.name, ds] + [h[c] for c in columns[3:]]
        for ds, h in headlines.items()
    ]
    wandb_utils.log_summary_table(table_rows, columns)

    # No reprefixed summary: `wandb.log("test/<ds>/<metric>")` already populates
    # the runs-table summary columns, so re-logging them would just duplicate.
    wandb_utils.finish_run()
    return all_metrics
