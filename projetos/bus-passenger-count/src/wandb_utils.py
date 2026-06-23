"""Wandb wrapper used by train/eval modules."""

import os
import random
from pathlib import Path
from uuid import uuid4

import yaml

from src import config

try:
    import wandb
except ImportError:
    wandb = None


def _active_run():
    return wandb.run if wandb is not None else None


def disable_ultralytics_autolog():
    """Turn off Ultralytics' built-in W&B callback so we log once, on epochs."""
    from ultralytics import settings
    if settings.get("wandb"):
        settings.update({"wandb": False})
        print("wandb:    disabled Ultralytics built-in W&B logging")


def init_run(wandb_config, run_name, run_dir, notes=None, tags=None):
    """Start a wandb run; returns False if wandb is unavailable."""
    if wandb is None:
        return False
    project = os.environ.get("WANDB_PROJECT") or config.WANDB_PROJECT
    entity  = os.environ.get("WANDB_ENTITY")  or config.WANDB_ENTITY
    if entity and "/" in entity:
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
    # Use epoch as the x-axis for every metric so curves from datasets with
    # different #steps overlay cleanly.
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    print(f"wandb:    run '{run_name}' -> {run.url}")
    return True


def is_active():
    return _active_run() is not None


def log_metrics(metrics, step=None):
    if not is_active():
        return
    payload = {k: v for k, v in metrics.items() if v is not None}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def finish_run(summary=None):
    run = _active_run()
    if run is None:
        return
    # Store headline values once, as run summary (shown in the runs table for
    # cross-run comparison). No redundant `final/` chart section.
    if summary:
        for k, v in summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                run.summary[k] = v
    wandb.finish()
    print("wandb:    run finished")


def log_image(path, key):
    """Upload a single image (e.g. a PR/F1 curve PNG) to wandb."""
    if not is_active():
        return
    wandb.log({key: wandb.Image(str(path))})


def log_summary_table(rows, columns, key="test/summary"):
    """Log one slide-ready table (one row per dataset)."""
    if not is_active():
        return
    wandb.log({key: wandb.Table(columns=list(columns), data=rows)})


def _read_yolo_labels(label_path):
    if not label_path.exists():
        return []

    def _parse_label_xywh(parts):
        coords = [float(v) for v in parts[1:]]

        if len(coords) == 4:
            return coords

        if len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max_x - min_x
            h = max_y - min_y
            if w <= 0 or h <= 0:
                return None
            cx = min_x + (w / 2.0)
            cy = min_y + (h / 2.0)
            return [cx, cy, w, h]

        return None

    out = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        xywh = _parse_label_xywh(parts)
        if xywh is None:
            continue
        x, y, w, h = xywh
        out.append((c, x, y, w, h))
    return out


def _yolo_to_minmax(x, y, w, h):
    return {
        "minX": max(x - w / 2.0, 0.0),
        "maxX": min(x + w / 2.0, 1.0),
        "minY": max(y - h / 2.0, 0.0),
        "maxY": min(y + h / 2.0, 1.0),
    }


def _resolve_test_image_dir(data_spec):
    if isinstance(data_spec, dict):
        spec = data_spec
        base_dir = Path.cwd()
    else:
        data_yaml = Path(data_spec)
        with data_yaml.open() as f:
            spec = yaml.safe_load(f)
        base_dir = data_yaml.parent

    split_dirs = spec.get("test") or spec.get("val")
    if split_dirs is None:
        raise FileNotFoundError("Data spec defines neither 'test' nor 'val' split")
    if not isinstance(split_dirs, list):
        split_dirs = [split_dirs]

    def _resolve_one(split_dir):
        p = Path(split_dir)
        if p.is_absolute() and p.exists():
            return p.resolve()

        candidates = []

        base = Path(spec.get("path", base_dir))
        if not base.is_absolute():
            base = (base_dir / base).resolve()
        candidates.append((base / p).resolve())

        candidates.append((base_dir / p).resolve())

        normalized_parts = [part for part in p.parts if part not in ("..", ".")]
        if normalized_parts:
            candidates.append((base_dir / Path(*normalized_parts)).resolve())

        for c in candidates:
            if c.exists():
                return c
        return None

    tried = []
    for split_dir in split_dirs:
        resolved = _resolve_one(split_dir)
        tried.append(str(split_dir))
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        f"Could not resolve split directories {tried}"
    )


def _resolve_label_dir(img_dir):
    img_dir = Path(img_dir)

    if img_dir.name.lower() == "images":
        split_dir = img_dir.parent
        if split_dir.name.lower() in ("train", "valid", "val", "test"):
            candidate = split_dir / "labels"
            if candidate.exists():
                return candidate

    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() != "images":
            continue
        candidate = Path(*parts[:i], "labels", *parts[i + 1:])
        if candidate.exists():
            return candidate

    return img_dir.parent / "labels"


def _to_wandb_image_source(result_obj, fallback_path):
    img = getattr(result_obj, "orig_img", None)
    if img is None:
        return str(fallback_path)
    if hasattr(img, "ndim") and img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()
    return img


def _downscale_for_wandb(image_source, max_side):
    """Reduce very large images before upload to avoid quota blowups."""
    if max_side <= 0:
        return image_source
    if isinstance(image_source, str):
        return image_source
    if not hasattr(image_source, "shape") or len(image_source.shape) < 2:
        return image_source

    h, w = image_source.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_source

    # Fast integer-step downsampling keeps dependencies minimal.
    step = (longest + max_side - 1) // max_side
    return image_source[::step, ::step].copy()


def _jpeg_for_wandb(image_source, quality):
    """Materialize as JPEG so wandb media is much smaller than PNG."""
    if quality >= 100:
        return image_source
    run = _active_run()
    if run is None:
        return image_source

    try:
        import cv2
    except ImportError:
        return image_source

    if isinstance(image_source, str):
        bgr = cv2.imread(image_source, cv2.IMREAD_COLOR)
        if bgr is None:
            return image_source
    else:
        if not hasattr(image_source, "shape"):
            return image_source
        if len(image_source.shape) == 2:
            bgr = image_source
        elif len(image_source.shape) == 3 and image_source.shape[2] == 3:
            bgr = image_source[:, :, ::-1].copy()
        else:
            return image_source

    cache_dir = Path(run.dir) / "media_jpeg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{uuid4().hex}.jpg"
    ok = cv2.imwrite(
        str(out_path),
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    return str(out_path) if ok else image_source


def log_test_predictions(predictor, data_spec, n=10, names=None,
                         conf=None, iou=None, panel_key="test/predictions", seed=0):
    """Upload a sample of test images with prediction and GT boxes."""
    if not is_active():
        print("wandb:    no active run — skipping test prediction panel")
        return

    conf = config.VIZ_CONF if conf is None else conf
    iou  = config.VIZ_IOU if iou is None else iou

    img_dir   = _resolve_test_image_dir(data_spec)
    label_dir = _resolve_label_dir(img_dir)

    all_imgs = sorted(
        [p for p in img_dir.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    )
    if not all_imgs:
        print(f"wandb:    no images found in {img_dir} — skipping prediction panel")
        return

    rng    = random.Random(seed)
    labeled_imgs = [
        p for p in all_imgs
        if (label_dir / f"{p.stem}.txt").exists()
    ]
    pool = labeled_imgs if labeled_imgs else all_imgs
    sample = rng.sample(pool, min(n, len(pool)))
    print(
        f"wandb:    uploading {len(sample)} test predictions from {img_dir} "
        f"(labeled images: {len(labeled_imgs)}/{len(all_imgs)})"
    )

    wandb_images = []
    for img_path in sample:
        results = predictor.predict(
            source=str(img_path), conf=conf, iou=iou,
            classes=[config.PERSON_CLASS_ID],
            agnostic_nms=config.VIZ_AGNOSTIC_NMS, verbose=False,
        )
        if not results:
            continue
        r = results[0]

        pred_boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xywhn = r.boxes.xywhn.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x, y, w, h), c, p in zip(xywhn, clses, confs):
                pred_boxes.append({
                    "position":    _yolo_to_minmax(float(x), float(y), float(w), float(h)),
                    "class_id":    0,
                    "box_caption": f"person_pred {p:.2f}",
                    "scores":      {"conf": float(p)},
                })

        gt_boxes = []
        label_path = label_dir / f"{img_path.stem}.txt"
        for c, x, y, w, h in _read_yolo_labels(label_path):
            gt_boxes.append({
                "position":    _yolo_to_minmax(x, y, w, h),
                "class_id":    1,
                "box_caption": "GT_ref",
            })

        image_source = _to_wandb_image_source(r, img_path)
        image_source = _downscale_for_wandb(
            image_source=image_source,
            max_side=config.WANDB_MAX_IMAGE_SIDE,
        )
        image_source = _jpeg_for_wandb(
            image_source=image_source,
            quality=config.WANDB_IMAGE_JPEG_QUALITY,
        )
        wandb_images.append(wandb.Image(
            image_source,
            boxes={
                "person_pred": {"box_data": pred_boxes, "class_labels": {0: "person_pred"}},
                "gt_ref": {"box_data": gt_boxes, "class_labels": {1: "GT_ref"}},
            },
            caption=img_path.name,
        ))

    if wandb_images:
        wandb.log({panel_key: wandb_images})
        print(
            f"wandb:    logged {len(wandb_images)} images under '{panel_key}' "
            "(single combined panel: person_pred + gt_ref)"
        )
