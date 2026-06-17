"""Download e preparação dos datasets."""

import json
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image

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
_CROWDHUMAN_FILES = {
    "zips": [
        "CrowdHuman_train01.zip",
        "CrowdHuman_train02.zip",
        "CrowdHuman_train03.zip",
        "CrowdHuman_val.zip",
    ],
    "annotations": {
        "train": "annotation_train.odgt",
        "valid": "annotation_val.odgt",
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


def _is_image_file(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _extract_images_from_zip(zip_path, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if not _is_image_file(member_path):
                continue
            target = dst_dir / member_path.name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def _build_yolo_labels_from_odgt(annotation_path, images_dir, labels_dir):
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_stem = {img.stem: img for img in images_dir.iterdir() if img.is_file() and _is_image_file(img)}

    written = 0
    dropped = 0
    missing_images = 0
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        image_id = str(item.get("ID", "")).strip()
        image_path = by_stem.get(image_id)
        if image_path is None:
            missing_images += 1
            continue

        width, height = Image.open(image_path).size
        rows = []
        for box in item.get("gtboxes", []):
            if str(box.get("tag", "")).lower() != "person":
                dropped += 1
                continue
            if int((box.get("extra") or {}).get("ignore", 0)) == 1:
                dropped += 1
                continue

            fbox = box.get("fbox")
            if not fbox or len(fbox) != 4:
                dropped += 1
                continue

            x, y, w, h = [float(v) for v in fbox]
            if w <= 1 or h <= 1:
                dropped += 1
                continue

            x1 = max(0.0, min(x, float(width)))
            y1 = max(0.0, min(y, float(height)))
            x2 = max(0.0, min(x + w, float(width)))
            y2 = max(0.0, min(y + h, float(height)))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                dropped += 1
                continue

            cx = (x1 + (bw / 2.0)) / float(width)
            cy = (y1 + (bh / 2.0)) / float(height)
            nw = bw / float(width)
            nh = bh / float(height)
            rows.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            written += 1

        (labels_dir / f"{image_path.stem}.txt").write_text("\n".join(rows), encoding="utf-8")

    return written, dropped, missing_images


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


def download_crowdhuman(
    repo_id="sshao0516/CrowdHuman",
    stage=_DEFAULT_STAGE,
    force=False,
    name="crowdhuman",
):
    """Baixa o CrowdHuman do Hugging Face e converte anotações para YOLO."""
    root = _STAGE_DIRS[stage] / name
    if not force and _is_downloaded(root):
        print(f"datasets: {name} já baixado em {root}")
        return root

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    valid_images = root / "valid" / "images"
    valid_labels = root / "valid" / "labels"

    from huggingface_hub import hf_hub_download

    print(f"datasets: baixando {name} ({repo_id}) ...")
    local_files = {}
    for filename in _CROWDHUMAN_FILES["zips"]:
        local_files[filename] = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            force_download=force,
        )
    for filename in _CROWDHUMAN_FILES["annotations"].values():
        local_files[filename] = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            force_download=force,
        )

    total_train = 0
    for filename in _CROWDHUMAN_FILES["zips"][:3]:
        total_train += _extract_images_from_zip(Path(local_files[filename]), train_images)
    total_valid = _extract_images_from_zip(
        Path(local_files[_CROWDHUMAN_FILES["zips"][3]]),
        valid_images,
    )
    print(f"datasets:   imagens extraídas (train={total_train}, valid={total_valid})")

    tr_written, tr_dropped, tr_missing = _build_yolo_labels_from_odgt(
        Path(local_files[_CROWDHUMAN_FILES["annotations"]["train"]]),
        train_images,
        train_labels,
    )
    va_written, va_dropped, va_missing = _build_yolo_labels_from_odgt(
        Path(local_files[_CROWDHUMAN_FILES["annotations"]["valid"]]),
        valid_images,
        valid_labels,
    )
    print(
        "datasets:   labels CrowdHuman convertidos "
        f"(train={tr_written}, valid={va_written}, "
        f"descartados={tr_dropped + va_dropped}, faltantes={tr_missing + va_missing})"
    )

    data_yaml = {
        "train": str(train_images.resolve()),
        "val": str(valid_images.resolve()),
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

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
