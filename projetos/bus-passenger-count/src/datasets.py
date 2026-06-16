"""Download e preparação dos datasets."""

import shutil
from pathlib import Path

import yaml

from src import config

CLASS_NAMES        = ["person"]
_DEFAULT_STAGE     = "processed"
_DOWNLOAD_COMPLETE = ".download_complete"
_STAGE_DIRS        = {
    "interim":   config.INTERIM_DIR,
    "processed": config.PROCESSED_DIR,
}
_ROBOFLOW_SOURCES = {
    "inside-bus-view": {
        "workspace": "cristians-workspace-9z79y",
        "project": "inside-bus-view-interim-clean-20260615-151915-3vue1",
        "keep_classes": {"person", "0"},
        "versions": {
            "interim": 1,
            "processed": None,
        },
    },
    "passenger-deakin": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-deakin-interim-clean-20260615-151915-pmurz",
        "keep_classes": {"person", "0"},
        "versions": {
            "interim": 1,
            "processed": None,
        },
    },
    "passenger-detection-bus": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-detection-bus-interim-clean-20260615-152755-ilzsy",
        "keep_classes": {"person", "0"},
        "versions": {
            "interim": 1,
            "processed": None,
        },
    },
}


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def available(stage=_DEFAULT_STAGE):
    """Datasets já baixados em data/<stage>/."""
    stage_dir = _STAGE_DIRS[stage]
    if not stage_dir.exists():
        return []
    return sorted(
        d.name for d in stage_dir.iterdir()
        if d.is_dir() and (d / _DOWNLOAD_COMPLETE).exists()
    )


def _find_child(parent, name):
    if not parent.exists():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.name.lower() == target:
            return child
    return None


def _split_images(root, split):
    split_dir = _find_child(root, split)
    if split_dir is None or not split_dir.is_dir():
        return None
    images = _find_child(split_dir, "images")
    return images.resolve() if images and images.is_dir() else None


def _has_yolo_layout(root):
    return root.exists() and all(_split_images(root, s) for s in ("train", "valid"))


def _is_downloaded(root):
    return (root / _DOWNLOAD_COMPLETE).exists() and _has_yolo_layout(root)


def _locate_dataset_root(start):
    if _has_yolo_layout(start):
        return start
    for child in start.rglob("*"):
        if child.is_dir() and _has_yolo_layout(child):
            return child
    raise RuntimeError(f"Nenhum split train/valid encontrado sob {start}")


def _flatten_into(root, source):
    if source == root:
        return
    for item in source.iterdir():
        dst = root / item.name
        if not dst.exists():
            shutil.move(str(item), str(dst))
    source.rmdir()


def _parse_label_row(parts):
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


def _normalize_to_single_person_class(root, keep_classes):
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return

    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = spec.get("names", [])
    if isinstance(names, dict):
        id_to_name = {int(k): str(v).lower() for k, v in names.items()}
    else:
        id_to_name = {i: str(v).lower() for i, v in enumerate(names)}

    keep_ids = {idx for idx, name in id_to_name.items() if name in keep_classes}

    converted = 0
    dropped = 0
    for split in ("train", "valid", "test"):
        lbl_dir = root / split / "labels"
        if not lbl_dir.exists():
            continue
        for label_path in lbl_dir.glob("*.txt"):
            rows = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id not in keep_ids:
                    dropped += 1
                    continue
                xywh = _parse_label_row(parts)
                if xywh is None:
                    dropped += 1
                    continue
                rows.append("0 " + " ".join(f"{v:.6f}" for v in xywh))
                converted += 1
            label_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"datasets:   labels normalizados ({converted} caixas, {dropped} descartadas)")


# ---------------------------------------------------------------------------
# Download API
# ---------------------------------------------------------------------------

def download_roboflow(name, stage=_DEFAULT_STAGE, force=False, model_format="yolov8"):
    """Baixa um dataset do Roboflow para data/<stage>/<name>/."""
    root = _STAGE_DIRS[stage] / name
    if not force and _is_downloaded(root):
        print(f"datasets: {name} já baixado em {root}")
        return root

    if name not in _ROBOFLOW_SOURCES:
        raise KeyError(
            f"Dataset '{name}' não mapeado em _ROBOFLOW_SOURCES."
        )
    source = _ROBOFLOW_SOURCES[name]
    version = source["versions"].get(stage)
    if not version:
        raise ValueError(
            f"Não há versão Roboflow configurada para '{name}' no stage '{stage}'."
        )

    api_key = (config.ROBOFLOW_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY ausente no .env.")

    from roboflow import Roboflow

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    print(
        f"datasets: baixando {name} "
        f"({source['workspace']}/{source['project']}/{version}) ..."
    )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source["workspace"]).project(source["project"])
    project.version(int(version)).download(
        model_format,
        location=str(root),
        overwrite=True,
    )

    _flatten_into(root, _locate_dataset_root(root))
    _normalize_to_single_person_class(root, source["keep_classes"])
    (root / _DOWNLOAD_COMPLETE).write_text("ok\n", encoding="utf-8")
    print(f"datasets: {name} pronto em {root}")
    return root


def download_selected(stage=_DEFAULT_STAGE, names=None, force=False, model_format="yolov8"):
    """Baixa uma lista de datasets mapeados no Roboflow."""
    names = list(names or _ROBOFLOW_SOURCES.keys())
    downloaded = []
    for name in names:
        downloaded.append(
            download_roboflow(
                name=name,
                stage=stage,
                force=force,
                model_format=model_format,
            )
        )
    return downloaded


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def prepare(names, stage=_DEFAULT_STAGE):
    """Monta o config de dados para YOLO e retorna um dict."""
    if isinstance(names, str):
        names = [names]
    names = list(names)

    roots = []
    for name in names:
        root = _STAGE_DIRS[stage] / name
        if not _is_downloaded(root):
            raise FileNotFoundError(
                f"Dataset '{name}' não encontrado em {root}. Execute o notebook 1 primeiro."
            )
        roots.append(root)

    train_imgs = [p for p in (_split_images(r, "train") for r in roots) if p]
    val_imgs   = [p for p in (_split_images(r, "valid") for r in roots) if p]
    test_imgs  = [p for p in (_split_images(r, "test")  for r in roots) if p]

    if not train_imgs:
        raise RuntimeError(f"Nenhum split 'train/images' para {names}.")
    if not val_imgs:
        raise RuntimeError(f"Nenhum split 'valid/images' para {names}.")

    single = len(names) == 1
    payload = {
        "train": str(train_imgs[0]) if single else [str(p) for p in train_imgs],
        "val":   str(val_imgs[0])   if single else [str(p) for p in val_imgs],
        "nc":    len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    if test_imgs:
        payload["test"] = str(test_imgs[0]) if single else [str(p) for p in test_imgs]

    return payload
