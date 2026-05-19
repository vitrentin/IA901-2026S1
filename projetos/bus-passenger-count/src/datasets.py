"""Pick datasets by name; fetch the processed payload on demand.

Datasets live on Google Drive (per-dataset shared folder, YOLO layout) or on
Hugging Face (CrowdHuman, redistribution-restricted). This module:

* Resolves friendly names (e.g. ``"InsideBusView"``, ``"CrowdHuman"``) to
  canonical keys via a small registry; matching is case- and separator-
  insensitive (``"InsideBusView" == "inside_bus_view" == "inside-bus-view"``).
* Downloads missing data into ``data/processed/<name>/`` on first use.
* Tolerates split directories with mixed capitalization (``Train`` vs ``train``).
* Converts CrowdHuman's ODGT annotations to YOLO labels (single ``person`` class).
* Generates a YOLO ``data.yaml`` with absolute paths, supporting either a
  single dataset or a merged training set across multiple datasets.

Example:

    >>> import datasets
    >>> yaml_path = datasets.prepare("InsideBusView")
    >>> yaml_path = datasets.prepare(["InsideBusView", "CrowdHuman"])
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import yaml

import config


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Canonical key -> dataset info. Source is determined by which key is set:
#   * ``gdrive_folder_url`` — shared Drive folder containing the YOLO layout
#     (preferred for small datasets; gdown has a ~50-file Drive folder cap).
#   * ``gdrive_zip_url`` — single shared .zip on Drive, extracted on download.
#   * ``hf_repo`` (+ ``hf_zips`` and ``hf_odgt``) — Hugging Face dataset where
#     image zips and ODGT annotations are downloaded individually and the
#     ODGT boxes are converted to YOLO labels (CrowdHuman).
#
# ``dir`` is the destination folder under ``data/processed/``.
REGISTRY: dict[str, dict] = {
    "inside-bus-view": {
        "dir":               "inside-bus-view",
        "gdrive_folder_url": "https://drive.google.com/drive/folders/1fvW1eP3QAAht62inlsjjUxRERNOcq2jH",
    },
    "passenger-deakin": {
        "dir":               "passenger-deakin",
        "gdrive_folder_url": "https://drive.google.com/drive/folders/1F-LB5fE6owvsw3vq_zXcblI2M_4g4NyE",
    },
    "passenger-detection-bus": {
        "dir":               "passenger-detection-bus",
        "gdrive_folder_url": "https://drive.google.com/drive/folders/1QFeVXeULj_IqQLLiR7u5f4AR-gXZpjrP",
    },
    "crowdhuman": {
        "dir":     "crowdhuman",
        "hf_repo": "sshao0516/CrowdHuman",
        # Map upstream split name -> our local split name + the HF files used.
        "hf_zips": {
            "train": (
                "CrowdHuman_train01.zip",
                "CrowdHuman_train02.zip",
                "CrowdHuman_train03.zip",
            ),
            "valid": ("CrowdHuman_val.zip",),
        },
        "hf_odgt": {
            "train": "annotation_train.odgt",
            "valid": "annotation_val.odgt",
        },
    },
}

# Aliases for less-obvious user inputs (case/separator are already normalized).
_ALIAS_OVERRIDES: dict[str, str] = {
    "passengerdetectiononabus": "passenger-detection-bus",
}

# All datasets in this project are remapped to a single "person" class.
CLASS_NAMES: list[str] = ["person"]

_SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _canonical(name: str) -> str:
    """Resolve a user-provided name to a key in ``REGISTRY``."""
    key = _norm(name)
    if key in _ALIAS_OVERRIDES:
        return _ALIAS_OVERRIDES[key]
    for canon in REGISTRY:
        if _norm(canon) == key:
            return canon
    raise KeyError(
        f"Unknown dataset: {name!r}. Known: {sorted(REGISTRY)}"
    )


def available() -> list[str]:
    """Return the canonical names known to the registry."""
    return sorted(REGISTRY)


# ---------------------------------------------------------------------------
# Filesystem helpers (case-insensitive YOLO layout discovery)
# ---------------------------------------------------------------------------

def _find_child(parent: Path, name: str) -> Path | None:
    if not parent.exists():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.name.lower() == target:
            return child
    return None


def _has_yolo_layout(root: Path) -> bool:
    if not root.exists():
        return False
    return any(_find_child(root, s) for s in _SPLITS)


def _locate_dataset_root(start: Path) -> Path:
    """Find the first directory at/under ``start`` that has YOLO splits."""
    if _has_yolo_layout(start):
        return start
    for child in start.iterdir():
        if child.is_dir() and _has_yolo_layout(child):
            return child
    raise RuntimeError(
        f"Downloaded data under {start} does not contain train/valid/test split folders."
    )


def _flatten_into(root: Path, source: Path) -> None:
    """Move every child of ``source`` up into ``root`` and remove ``source``."""
    if source == root:
        return
    for item in source.iterdir():
        dst = root / item.name
        if dst.exists():
            continue
        shutil.move(str(item), str(dst))
    try:
        source.rmdir()
    except OSError:
        pass


def _split_images(root: Path, split: str) -> Path | None:
    split_dir = _find_child(root, split)
    if split_dir is None or not split_dir.is_dir():
        return None
    images = _find_child(split_dir, "images")
    return images.resolve() if images and images.is_dir() else None


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------

def _download_gdrive(info: dict, target: Path) -> None:
    import gdown

    if "gdrive_zip_url" in info:
        zip_path = target / "_dataset.zip"
        print(f"datasets: downloading zip {info['gdrive_zip_url']} -> {zip_path}")
        gdown.download(url=info["gdrive_zip_url"], output=str(zip_path), quiet=False, fuzzy=True)
        print(f"datasets: extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        zip_path.unlink(missing_ok=True)
        return

    print(f"datasets: downloading folder {info['gdrive_folder_url']} -> {target}")
    gdown.download_folder(
        url=info["gdrive_folder_url"],
        output=str(target),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )


def _download_huggingface_crowdhuman(info: dict, target: Path) -> None:
    """Download CrowdHuman zips + ODGT and produce a YOLO layout under ``target``."""
    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo = info["hf_repo"]
    print(f"datasets: downloading HF dataset {repo}")

    for split, zips in info["hf_zips"].items():
        img_dir = target / split / "images"
        lbl_dir = target / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for zname in zips:
            zpath = hf_hub_download(repo_id=repo, repo_type="dataset", filename=zname)
            print(f"datasets:   extracting {zname} -> {img_dir}")
            with zipfile.ZipFile(zpath) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    if not member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    out = img_dir / Path(member.filename).name
                    if out.exists():
                        continue
                    with zf.open(member) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

        odgt_name = info["hf_odgt"].get(split)
        if not odgt_name:
            continue
        odgt_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=odgt_name)
        print(f"datasets:   converting {odgt_name} -> {lbl_dir}")
        _odgt_to_yolo(Path(odgt_path), img_dir, lbl_dir, Image)


def _odgt_to_yolo(odgt: Path, img_dir: Path, lbl_dir: Path, Image) -> None:
    """Convert one CrowdHuman ODGT file into YOLO label files (class 0 = person)."""
    converted = 0
    for line in odgt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        img_path = img_dir / f"{rec['ID']}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            iw, ih = im.size

        rows: list[str] = []
        for box in rec.get("gtboxes", []):
            if box.get("tag") != "person":
                continue
            if box.get("extra", {}).get("ignore"):
                continue
            x, y, w, h = box["fbox"]
            x0 = max(x, 0)
            y0 = max(y, 0)
            x1 = min(x + w, iw)
            y1 = min(y + h, ih)
            if x1 <= x0 or y1 <= y0:
                continue
            cx = ((x0 + x1) / 2.0) / iw
            cy = ((y0 + y1) / 2.0) / ih
            bw = (x1 - x0) / iw
            bh = (y1 - y0) / ih
            rows.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (lbl_dir / f"{rec['ID']}.txt").write_text("\n".join(rows), encoding="utf-8")
        converted += 1
    print(f"datasets:   wrote {converted} label files in {lbl_dir}")


def _download(info: dict, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if "hf_repo" in info:
        _download_huggingface_crowdhuman(info, target)
    elif "gdrive_zip_url" in info or "gdrive_folder_url" in info:
        _download_gdrive(info, target)
    else:
        raise ValueError(f"Registry entry has no known source URL: {info}")


def _dataset_root(name: str, force_download: bool = False) -> Path:
    """Return a local directory containing YOLO splits for ``name``."""
    info = REGISTRY[_canonical(name)]
    root = config.PROCESSED_DIR / info["dir"]

    if not force_download and _has_yolo_layout(root):
        return root

    _download(info, root)

    # Drive folder downloads often nest under the original folder name
    # (e.g. ``Processed/``). Flatten so split dirs sit directly under ``root``.
    actual = _locate_dataset_root(root)
    _flatten_into(root, actual)
    return root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare(
    names: str | Iterable[str],
    force_download: bool = False,
) -> Path:
    """Ensure datasets are local and return the path to a YOLO ``data.yaml``.

    With one name, the yaml is written inside that dataset's folder and uses
    string paths so downstream YOLO tooling treats it like a single source.
    With multiple names, a merged yaml is written under
    ``data/processed/_combined/<name1>+<name2>+...`` using list paths.
    """
    if isinstance(names, str):
        names = [names]
    canonical = [_canonical(n) for n in names]
    if not canonical:
        raise ValueError("prepare() requires at least one dataset name.")

    roots = [_dataset_root(n, force_download=force_download) for n in canonical]

    train_imgs = [p for p in (_split_images(r, "train") for r in roots) if p]
    val_imgs   = [p for p in (_split_images(r, "valid") for r in roots) if p]
    test_imgs  = [p for p in (_split_images(r, "test")  for r in roots) if p]

    if not train_imgs:
        raise RuntimeError(f"No 'train/images' found for {canonical}.")
    if not val_imgs:
        raise RuntimeError(f"No 'valid/images' found for {canonical}.")

    single = len(canonical) == 1
    payload: dict[str, object] = {
        "train": str(train_imgs[0]) if single else [str(p) for p in train_imgs],
        "val":   str(val_imgs[0])   if single else [str(p) for p in val_imgs],
        "nc":    len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    if test_imgs:
        payload["test"] = str(test_imgs[0]) if single else [str(p) for p in test_imgs]

    if single:
        out_dir = roots[0]
    else:
        out_dir = config.PROCESSED_DIR / "_combined" / "+".join(canonical)
        out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"datasets: data.yaml -> {out}")
    return out.resolve()
