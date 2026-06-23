"""Dataset download and processing utilities for notebooks.

Conventions:
- RAW: downloaded source datasets (minimal handling)
- PROCESSED: normalized/filtered final datasets
"""

import json
import os
import random
import re
import shutil
import stat
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from roboflow import Roboflow

from src import config
from src import datasets

RAW_DOWNLOAD_SOURCES = {
    "passenger_detection_bus": {
        "workspace": "bus-project-frdgz",
        "project": "passenger-detection-on-a-bus-qgljh",
        "keep_classes": {"passenger", "person", "occupied"},
        "raw_name": "passenger-detection-bus",
    },
    "inside_bus_view": {
        "workspace": "seat-occupancy",
        "project": "inside-bus-view",
        "keep_classes": {"occupied", "person", "passenger"},
        "raw_name": "inside-bus-view",
    },
    "passenger_deakin": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-mmpbi-kmo7r",
        "keep_classes": {"passenger", "person", "occupied"},
        "raw_name": "passenger-deakin",
    },
    "onibus_unicamp_private": {
        "workspace": "cristians-workspace-9z79y",
        "project": "bus-unicamp-ds4sm",
        "keep_classes": {"person"},
        "raw_name": "onibus-unicamp-private",
    },
}

PROCESSED_SOURCES = {
    "passenger_detection_bus": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-detection-bus-clean",
        "raw_name": "passenger-detection-bus",
    },
    "inside_bus_view": {
        "workspace": "cristians-workspace-9z79y",
        "project": "inside-bus-view-clean",
        "raw_name": "inside-bus-view",
    },
    "passenger_deakin": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-deakin-clean",
        "raw_name": "passenger-deakin",
    },
    "onibus_unicamp_private": {
        "workspace": "cristians-workspace-9z79y",
        "project": "bus-unicamp-ds4sm",
        "raw_name": "onibus-unicamp-private",
    },
}

RAW_PREPROCESS_SOURCES = {
    k: v for k, v in RAW_DOWNLOAD_SOURCES.items()
    if k != "onibus_unicamp_private"
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


def _on_rm_error(func, path, exc_info):
    # Common on Windows when files are read-only.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _safe_rmtree(path, retries=5, delay_s=0.5):
    """Best-effort recursive delete with retries for transient locks."""
    path = Path(path)
    if not path.exists():
        return
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except PermissionError as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(delay_s * attempt)
    raise last_err


def _load_class_names(data_yaml):
    spec = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = spec.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def _iter_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts], key=lambda p: p.name)


def _parse_timestamp_from_name(name):
    m = re.search(r"(\d{17}|\d{14})", name)
    if not m:
        return None
    raw = m.group(1)
    if len(raw) == 17:
        return datetime.strptime(raw[:14], "%Y%m%d%H%M%S")
    return datetime.strptime(raw, "%Y%m%d%H%M%S")


def _select_images(source_name, images, deakin_stride, inside_min_minutes):
    if source_name == "passenger_deakin":
        exact_drop = {"aa_png.rf.a4e45e180810d76cd80daa6d8c896815.jpg"}
        dropped = {
            "numeric_only": 0, "video_watch": 0, "youtube": 0, "frameN": 0,
            "frame_X": 0, "head_X": 0, "aa_png": 0,
        }
        grouped_num = {}
        grouped_letter = {}
        passthrough = []

        for img in images:
            n = img.name
            if n in exact_drop:
                dropped["aa_png"] += 1
                continue
            if re.match(r"^\d+_jpg\.rf\.", n):
                dropped["numeric_only"] += 1
                continue
            if n.startswith("VIDEO-WATCH_-CCTV"):
                dropped["video_watch"] += 1
                continue
            if n.startswith("youtube-"):
                dropped["youtube"] += 1
                continue
            if re.match(r"^frame\d+_jpg\.rf\.", n):
                dropped["frameN"] += 1
                continue
            if re.match(r"^frame_\d+_jpg\.rf\.", n):
                dropped["frame_X"] += 1
                continue
            if re.match(r"^head_\d+_png\.rf\.", n):
                dropped["head_X"] += 1
                continue

            m_num = re.match(r"^(\d+)-(\d+)_jpg\.rf\.", n)
            if m_num:
                grouped_num.setdefault(m_num.group(1), []).append((int(m_num.group(2)), img))
                continue

            m_let = re.match(r"^([A-Za-z])-(\d+)-_jpg\.rf\.", n)
            if m_let:
                grouped_letter.setdefault(m_let.group(1).lower(), []).append((int(m_let.group(2)), img))
                continue

            passthrough.append(img)

        selected = list(passthrough)

        def pick_three(items):
            items = sorted(items, key=lambda x: x[0])
            if len(items) <= 3:
                return [img for _, img in items]
            mid = len(items) // 2
            return [items[0][1], items[mid][1], items[-1][1]]

        for group in grouped_num.values():
            selected.extend(pick_three(group))
        for group in grouped_letter.values():
            selected.extend(pick_three(group))
        return selected

    if source_name == "inside_bus_view":
        out = []
        min_delta = timedelta(minutes=inside_min_minutes)
        last_ts = None
        for img in images:
            ts = _parse_timestamp_from_name(img.name)
            if ts is None:
                out.append(img)
                continue
            if last_ts is None or (ts - last_ts) >= min_delta:
                out.append(img)
                last_ts = ts
        return out

    return images


def _resplit_items(items, train_ratio=0.7, valid_ratio=0.15, seed=42):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = items[:n_train]
    valid = items[n_train:n_train + n_valid]
    test = items[n_train + n_valid:]
    return {"train": train, "valid": valid, "test": test}


def _normalize_label_rows(label_path, id_to_name, keep_names):
    keep_ids = {k for k, v in id_to_name.items() if v.lower() in keep_names}
    out = []
    if not label_path.exists():
        return out
    for row in label_path.read_text(encoding="utf-8").splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split()
        cls_id = int(float(parts[0]))
        if cls_id not in keep_ids:
            continue
        out.append("0 " + " ".join(parts[1:5]))
    return out


def _write_data_yaml(root):
    payload = {
        "train": "../train/images",
        "val": "../valid/images",
        "nc": 1,
        "names": ["person"],
    }
    if (root / "test" / "images").exists():
        payload["test"] = "../test/images"
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _find_downloaded_yolo_root(raw_download_source_dir):
    # Roboflow may place YOLO files directly in `location` or inside a child folder.
    if (raw_download_source_dir / "data.yaml").exists():
        return raw_download_source_dir
    candidates = [d for d in raw_download_source_dir.iterdir() if d.is_dir() and (d / "data.yaml").exists()]
    if not candidates:
        raise FileNotFoundError(f"Nenhum download YOLO encontrado em {raw_download_source_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _download_roboflow_sources_to_dir(
    target_root,
    sources,
    pdb_version=2,
    inside_version=5,
    deakin_version=3,
    private_version=1,
    api_key="",
):
    api_key = (api_key or config.ROBOFLOW_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY ausente.")
    target_root.mkdir(parents=True, exist_ok=True)

    versions = {
        "passenger_detection_bus": pdb_version,
        "inside_bus_view": inside_version,
        "passenger_deakin": deakin_version,
        "onibus_unicamp_private": private_version,
    }

    download_root = target_root.parent / "_roboflow_downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    for source_name, meta in sources.items():
        dst_outer = download_root / source_name
        if dst_outer.exists():
            _safe_rmtree(dst_outer)
        project = rf.workspace(meta["workspace"]).project(meta["project"])
        version = versions[source_name]
        print(f"[download] {source_name}: version {version}")
        project.version(version).download("yolov8", location=str(dst_outer), overwrite=True)

        src_root = _find_downloaded_yolo_root(dst_outer)
        dst_root = target_root / meta["raw_name"]
        if dst_root.exists():
            _safe_rmtree(dst_root)
        shutil.copytree(src_root, dst_root)
        # datasets.prepare() requires this marker to consider the dataset ready.
        (dst_root / ".download_complete").write_text("ok\n", encoding="utf-8")


def download_roboflow_raw(pdb_version=2, inside_version=5, deakin_version=3, private_version=1, api_key=""):
    return _download_roboflow_sources_to_dir(
        target_root=config.RAW_DIR,
        sources=RAW_DOWNLOAD_SOURCES,
        pdb_version=pdb_version,
        inside_version=inside_version,
        deakin_version=deakin_version,
        private_version=private_version,
        api_key=api_key,
    )


def download_roboflow_processed(pdb_version=1, inside_version=1, deakin_version=1, private_version=1, api_key=""):
    return _download_roboflow_sources_to_dir(
        target_root=config.PROCESSED_DIR,
        sources=PROCESSED_SOURCES,
        pdb_version=pdb_version,
        inside_version=inside_version,
        deakin_version=deakin_version,
        private_version=private_version,
        api_key=api_key,
    )


def download_crowdhuman_raw(repo_id="sshao0516/CrowdHuman", force=False, raw_name="crowdhuman"):
    root = config.RAW_DIR / raw_name
    if root.exists() and not force:
        return root
    if root.exists():
        _safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for filename in _CROWDHUMAN_FILES["zips"]:
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=str(root), force_download=force)
    for filename in _CROWDHUMAN_FILES["annotations"].values():
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=str(root), force_download=force)
    return root


def process_roboflow_raw_to_processed(deakin_stride=40, inside_min_minutes=1):
    out_root = config.PROCESSED_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    for source_name, meta in RAW_PREPROCESS_SOURCES.items():
        src_root = config.RAW_DIR / meta["raw_name"]
        if not src_root.exists():
            raise FileNotFoundError(f"Dataset raw ausente: {src_root}")

        id_to_name = _load_class_names(src_root / "data.yaml")
        all_items = []
        for split in ("train", "valid", "test"):
            src_img = src_root / split / "images"
            src_lbl = src_root / split / "labels"
            if not src_img.exists():
                continue
            images = _iter_images(src_img)
            for img in images:
                rows = _normalize_label_rows(src_lbl / f"{img.stem}.txt", id_to_name, meta["keep_classes"])
                if rows:
                    all_items.append({"img": img, "rows": rows, "split": split})

        selected_imgs = _select_images(source_name, [item["img"] for item in all_items], deakin_stride, inside_min_minutes)
        selected_names = {p.name for p in selected_imgs}
        selected_items = [item for item in all_items if item["img"].name in selected_names]

        if source_name in {"passenger_deakin", "inside_bus_view"}:
            buckets = _resplit_items(selected_items)
        else:
            buckets = {"train": [], "valid": [], "test": []}
            for item in selected_items:
                buckets[item["split"]].append(item)

        dst_root = out_root / meta["raw_name"]
        if dst_root.exists():
            _safe_rmtree(dst_root)
        for split, items in buckets.items():
            dst_img = dst_root / split / "images"
            dst_lbl = dst_root / split / "labels"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for item in items:
                img = item["img"]
                shutil.copy2(img, dst_img / img.name)
                (dst_lbl / f"{img.stem}.txt").write_text("\n".join(item["rows"]), encoding="utf-8")

        _write_data_yaml(dst_root)
        (dst_root / ".download_complete").write_text("ok\n", encoding="utf-8")


def promote_crowdhuman_raw_to_processed(raw_name="crowdhuman", processed_name="crowdhuman", force=False):
    raw_root = config.RAW_DIR / raw_name
    if not raw_root.exists():
        raise FileNotFoundError(f"CrowdHuman raw não encontrado em {raw_root}.")

    root = config.PROCESSED_DIR / processed_name
    if root.exists() and not force:
        return root
    if root.exists():
        _safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    valid_images = root / "valid" / "images"
    valid_labels = root / "valid" / "labels"

    for filename in _CROWDHUMAN_FILES["zips"][:3]:
        datasets._extract_images_from_zip(raw_root / filename, train_images)
    datasets._extract_images_from_zip(raw_root / _CROWDHUMAN_FILES["zips"][3], valid_images)

    train_ann = raw_root / _CROWDHUMAN_FILES["annotations"]["train"]
    valid_ann = raw_root / _CROWDHUMAN_FILES["annotations"]["valid"]
    datasets._build_yolo_labels_from_odgt(train_ann, train_images, train_labels)
    datasets._build_yolo_labels_from_odgt(valid_ann, valid_images, valid_labels)

    data_yaml = {
        "train": str(train_images.resolve()),
        "val": str(valid_images.resolve()),
        "nc": 1,
        "names": ["person"],
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (root / ".download_complete").write_text("ok\n", encoding="utf-8")
    return root


def ensure_crowdhuman_processed(force=False):
    return datasets.download_crowdhuman(
        repo_id="sshao0516/CrowdHuman",
        stage="processed",
        force=force,
        name="crowdhuman",
    )
