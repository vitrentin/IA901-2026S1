"""Treino de experimentos."""

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from src import config
from src import datasets
from src import experiments
from src import wandb_utils

CUSTOM_ALBUMENTATIONS_KEYS = {
    "blur_px",
    "blur_p",
    "motion_blur_px",
    "motion_blur_p",
    "cutout_holes",
    "cutout_size",
    "cutout_p",
}
_ORIGINAL_ALBUMENTATIONS_CLASS = None


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
            epoch = int(float(row["epoch"])) + 1 if row.get("epoch") else None
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
                if epoch is not None:
                    payload["epoch"] = epoch
                wandb_utils.log_metrics(payload)
                logged += 1
    if logged:
        print(f"train:    {logged} épocas registradas no wandb")


def _resolve_data_arg(data_spec, run_dir):
    if isinstance(data_spec, dict):
        data_file = Path(run_dir) / "_data_runtime.yaml"
        data_file.write_text(yaml.safe_dump(data_spec, sort_keys=False), encoding="utf-8")
        return str(data_file)
    return str(data_spec)


def _best_weights(results):
    save_dir = Path(results.save_dir)
    best     = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    return save_dir, best


def _split_augment_params(augment):
    if not augment:
        return {}, {}
    yolo_aug = {}
    custom_aug = {}
    for k, v in augment.items():
        if k in CUSTOM_ALBUMENTATIONS_KEYS:
            custom_aug[k] = v
        else:
            yolo_aug[k] = v
    return yolo_aug, custom_aug


def _build_custom_albumentations(custom_aug):
    if not custom_aug:
        return []

    import albumentations as A

    transforms = []
    blur_px = int(custom_aug.get("blur_px", 0) or 0)
    motion_blur_px = int(custom_aug.get("motion_blur_px", 0) or 0)
    cutout_holes = int(custom_aug.get("cutout_holes", 0) or 0)
    cutout_size = float(custom_aug.get("cutout_size", 0.0) or 0.0)

    if blur_px > 0 and blur_px % 2 == 0:
        blur_px += 1
    if motion_blur_px > 0 and motion_blur_px % 2 == 0:
        motion_blur_px += 1

    if blur_px > 0:
        transforms.append(A.Blur(blur_limit=(blur_px, blur_px), p=float(custom_aug.get("blur_p", 0.15))))
    if motion_blur_px > 0:
        transforms.append(
            A.MotionBlur(blur_limit=(motion_blur_px, motion_blur_px), p=float(custom_aug.get("motion_blur_p", 0.15)))
        )
    if cutout_holes > 0 and cutout_size > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(cutout_holes, cutout_holes),
                hole_height_range=(cutout_size, cutout_size),
                hole_width_range=(cutout_size, cutout_size),
                p=float(custom_aug.get("cutout_p", 0.5)),
            )
        )

    return transforms


def _patch_albumentations(custom_aug):
    global _ORIGINAL_ALBUMENTATIONS_CLASS

    from ultralytics.data import augment as ul_aug

    if _ORIGINAL_ALBUMENTATIONS_CLASS is None:
        _ORIGINAL_ALBUMENTATIONS_CLASS = ul_aug.Albumentations

    if not custom_aug:
        ul_aug.Albumentations = _ORIGINAL_ALBUMENTATIONS_CLASS
        return

    transforms = _build_custom_albumentations(custom_aug)
    if not transforms:
        ul_aug.Albumentations = _ORIGINAL_ALBUMENTATIONS_CLASS
        return

    base_class = _ORIGINAL_ALBUMENTATIONS_CLASS

    class PatchedAlbumentations(base_class):
        def __init__(self, p=1.0, transforms=None):
            if transforms is None:
                transforms = list(_build_custom_albumentations(custom_aug))
            super().__init__(p=p, transforms=transforms)

    ul_aug.Albumentations = PatchedAlbumentations
    print(f"train:    custom albumentations ativas: {', '.join(custom_aug.keys())}")


def _train_stage(model, data_spec, train_params, run_dir, name, augment=None):
    """Treina uma etapa e retorna os melhores pesos."""
    params = dict(train_params or {})
    yolo_aug, custom_aug = _split_augment_params(augment)
    params.update(yolo_aug)
    _patch_albumentations(custom_aug)
    params["data"]     = _resolve_data_arg(data_spec, run_dir)
    params["project"]  = str(run_dir)
    params["name"]     = name
    params["exist_ok"] = True

    t0 = time.time()
    results = model.train(**params)
    dt = time.time() - t0
    print(f"train:    etapa '{name}' concluída em {dt:.1f}s")

    save_dir, best = _best_weights(results)
    _log_yolo_history(save_dir)
    return best, dt


def run(experiment_id, model, data_spec, train_config=None):
    """Treina um modelo e retorna o run dir."""
    run_dir, run_name = _make_run_dir(experiment_id)
    print(f"train:    run dir -> {run_dir}")

    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "data":          data_spec,
            "train_config":  train_config,
            **_git_info(),
        },
        run_name=run_name,
        run_dir=run_dir,
    )

    best, dt = _train_stage(model, data_spec, train_config, run_dir, "train")
    (run_dir / "weights.txt").write_text(str(best) + "\n")
    wandb_utils.log_metrics({"train/duration_sec": dt})
    wandb_utils.finish_run({"train/duration_sec": dt})
    return run_dir


def _prepare_and_summarize(names, stage):
    data_spec = datasets.prepare(names, stage=stage)
    print(f"train:    datasets de treino ({stage}): {names}")
    for name in names:
        for split in ["train", "valid", "test"]:
            img_dir = config.DATA_DIR / stage / name / split / "images"
            if img_dir.exists():
                n_img = len(list(img_dir.glob("*.*")))
                print(f"            {name}/{split}: {n_img} imagens")
    return data_spec


def run_experiment(cfg):
    """Roda experimento por `strategy` (`direct`, `two_stage`, `baseline`)."""
    from ultralytics import YOLO

    wandb_utils.disable_ultralytics_autolog()
    experiment_id = cfg["experiment_id"]
    strategy      = cfg.get("strategy", "direct")
    stage         = cfg.get("dataset_stage", "interim")

    if strategy == "baseline":
        print(f"train:    estratégia 'baseline' — sem treino para {experiment_id}")
        return None

    run_dir, run_name = _make_run_dir(experiment_id)
    print(f"train:    run dir -> {run_dir}")
    experiments.save_resolved(cfg, run_dir)

    if strategy == "two_stage":
        weights = cfg["weights"]
        for i, st in enumerate(cfg["stages"], start=1):
            st_name   = st.get("name", f"stage{i}")
            st_stage  = st.get("dataset_stage", stage)
            data_spec = _prepare_and_summarize(st["train_datasets"], st_stage)
            params    = dict(st.get("train_config", {}))
            if st.get("freeze") is not None:
                params["freeze"] = st["freeze"]
            augment = st.get("augment", cfg.get("augment"))

            wandb_utils.init_run(
                wandb_config={
                    "experiment_id": experiment_id,
                    "strategy":      strategy,
                    "stage":         st_name,
                    "stage_index":   i,
                    "dataset_stage": st_stage,
                    "datasets":      st["train_datasets"],
                    "data":          data_spec,
                    "train_config":  params,
                    "augment":       augment,
                    **_git_info(),
                },
                run_name=f"{run_name}_{st_name}",
                run_dir=run_dir,
            )
            model = YOLO(weights)
            best, dt = _train_stage(model, data_spec, params, run_dir, st_name, augment)
            wandb_utils.finish_run({"train/duration_sec": dt})
            weights = str(best)

        (run_dir / "weights.txt").write_text(str(weights) + "\n")
        return run_dir

    data_spec = _prepare_and_summarize(cfg["train_datasets"], stage)
    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "strategy":      strategy,
            "dataset_stage": stage,
            "datasets":      cfg["train_datasets"],
            "data":          data_spec,
            "train_config":  cfg.get("train_config"),
            "augment":       cfg.get("augment"),
            **_git_info(),
        },
        run_name=run_name,
        run_dir=run_dir,
    )
    model = YOLO(cfg["weights"])
    best, dt = _train_stage(
        model, data_spec, cfg.get("train_config"), run_dir, "train", cfg.get("augment")
    )
    (run_dir / "weights.txt").write_text(str(best) + "\n")
    wandb_utils.log_metrics({"train/duration_sec": dt})
    wandb_utils.finish_run({"train/duration_sec": dt})
    return run_dir
