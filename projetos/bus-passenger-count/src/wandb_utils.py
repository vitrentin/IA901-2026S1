"""Small wandb wrapper used by train/eval modules."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml

import config

def _active_run():
    try:
        import wandb
        return wandb.run
    except Exception:
        return None


def init_run(
    wandb_config: dict[str, Any],
    run_name: str,
    run_dir: str | Path,
    notes: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Start a wandb run; returns False if wandb is unavailable."""
    try:
        import wandb
        project = os.environ.get("WANDB_PROJECT") or config.WANDB_PROJECT
        entity = os.environ.get("WANDB_ENTITY") or config.WANDB_ENTITY
        if entity and "/" in entity:
            # Common mistake: passing "entity/project" in WANDB_ENTITY.
            entity = entity.split("/", 1)[0].strip()
            print(f"wandb:    normalized entity to '{entity}' from WANDB_ENTITY")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project = project,
            entity  = entity,
            name    = run_name,
            config  = wandb_config,
            dir     = str(run_dir),
            notes   = notes,
            tags    = tags,
            reinit  = "finish_previous",
        )
        print(f"wandb:    run '{run_name}' -> {run.url}")
        return True
    except Exception as e:
        print(f"wandb:    init failed ({e}) — continuing without WandB")
        return False


def is_active() -> bool:
    return _active_run() is not None


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    if not is_active():
        return
    try:
        import wandb
        payload = {k: v for k, v in metrics.items() if v is not None}
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)
    except Exception as e:
        print(f"wandb:    log_metrics failed ({e})")


def finish_run(summary: dict[str, Any] | None = None) -> None:
    run = _active_run()
    if run is None:
        return
    try:
        import wandb
        if summary:
            final = {f"final/{k}": v for k, v in summary.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if final:
                wandb.log(final)
                for k, v in final.items():
                    run.summary[k] = v
        wandb.finish()
        print("wandb:    run finished")
    except Exception as e:
        print(f"wandb:    finish failed ({e})")


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into normalized box tuples."""
    if not label_path.exists():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        out.append((c, x, y, w, h))
    return out


def _yolo_to_minmax(x: float, y: float, w: float, h: float) -> dict:
    """Convert normalized YOLO center boxes to wandb min/max boxes."""
    return {
        "minX": max(x - w / 2.0, 0.0),
        "maxX": min(x + w / 2.0, 1.0),
        "minY": max(y - h / 2.0, 0.0),
        "maxY": min(y + h / 2.0, 1.0),
    }


def _resolve_test_image_dir(data_yaml: Path) -> Path:
    """Resolve the test image directory from a YOLO dataset yaml."""
    data_yaml = Path(data_yaml)
    with data_yaml.open() as f:
        spec = yaml.safe_load(f)

    split_dir = spec.get("test") or spec.get("val")
    if split_dir is None:
        raise FileNotFoundError(f"{data_yaml} defines neither 'test' nor 'val' split")

    p = Path(split_dir)
    if p.is_absolute() and p.exists():
        return p

    candidates: list[Path] = []

    # Candidate 1: Ultralytics-style resolution using optional `path`.
    base = Path(spec.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    candidates.append((base / p).resolve())

    # Candidate 2: direct relative to data.yaml directory.
    candidates.append((data_yaml.parent / p).resolve())

    # Candidate 3: sanitize leading ".." segments that some exports include.
    normalized_parts = [part for part in p.parts if part not in ("..", ".")]
    if normalized_parts:
        candidates.append((data_yaml.parent / Path(*normalized_parts)).resolve())

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve split directory '{split_dir}' from {data_yaml}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _to_wandb_image_source(result_obj, fallback_path: Path):
    """Return an RGB image source for wandb.Image."""
    img = getattr(result_obj, "orig_img", None)
    if img is None:
        return str(fallback_path)
    try:
        if getattr(img, "ndim", 0) == 3 and img.shape[2] == 3:
            # Ultralytics/OpenCV images are usually BGR; wandb expects RGB.
            return img[:, :, ::-1].copy()
    except Exception:
        pass
    return img


def log_test_predictions(
    predictor,
    data_yaml: str | Path,
    n: int = 10,
    names: dict[int, str] | None = None,
    conf: float = 0.25,
    panel_key: str = "test/predictions",
    seed: int = 0,
) -> None:
    """Upload a sample of test images with prediction and GT boxes."""
    if not is_active():
        print("wandb:    no active run — skipping test prediction panel")
        return

    try:
        import wandb
    except ImportError:
        print("wandb:    not installed — skipping log_test_predictions")
        return

    try:
        data_yaml = Path(data_yaml)
        with data_yaml.open() as f:
            spec = yaml.safe_load(f)
        class_names = names or (
            spec["names"] if isinstance(spec.get("names"), dict)
            else {i: n for i, n in enumerate(spec.get("names", []))}
        )

        img_dir   = _resolve_test_image_dir(data_yaml)
        label_dir = img_dir.parent / "labels"

        all_imgs = sorted(
            [p for p in img_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        )
        if not all_imgs:
            print(f"wandb:    no images found in {img_dir} — skipping prediction panel")
            return

        rng = random.Random(seed)
        sample = rng.sample(all_imgs, min(n, len(all_imgs)))
        print(f"wandb:    uploading {len(sample)} test predictions from {img_dir}")

        wandb_images: list[Any] = []
        for img_path in sample:
            results = predictor.predict(source=str(img_path), conf=conf, verbose=False)
            if not results:
                continue
            r = results[0]

            pred_boxes: list[dict] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xywhn = r.boxes.xywhn.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for (x, y, w, h), c, p in zip(xywhn, clses, confs):
                    pred_boxes.append({
                        "position":    _yolo_to_minmax(float(x), float(y), float(w), float(h)),
                        "class_id":    int(c),
                        "box_caption": f"{class_names.get(int(c), c)} {p:.2f}",
                        "scores":      {"conf": float(p)},
                    })

            gt_boxes: list[dict] = []
            label_path = label_dir / f"{img_path.stem}.txt"
            for c, x, y, w, h in _read_yolo_labels(label_path):
                gt_boxes.append({
                    "position":    _yolo_to_minmax(x, y, w, h),
                    "class_id":    int(c),
                    "box_caption": class_names.get(int(c), str(c)),
                })

            wandb_images.append(wandb.Image(
                _to_wandb_image_source(r, img_path),
                boxes = {
                    "predictions":  {"box_data": pred_boxes,  "class_labels": class_names},
                    "ground_truth": {"box_data": gt_boxes,    "class_labels": class_names},
                },
                caption = img_path.name,
            ))

        if wandb_images:
            wandb.log({panel_key: wandb_images})
            print(f"wandb:    logged {len(wandb_images)} images under '{panel_key}'")
    except Exception as e:
        print(f"wandb:    log_test_predictions failed ({e})")
