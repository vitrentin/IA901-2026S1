"""Download e preparação dos datasets."""

import json
import re
import shutil
import zipfile
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


def _gdrive_download_url(url):
    match = re.search(r"/d/([^/]+)", url)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"
    return url


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


# ---------------------------------------------------------------------------
# Download API
# ---------------------------------------------------------------------------

def download_gdrive(name, url, stage=_DEFAULT_STAGE, force=False):
    """Baixa um dataset do Google Drive e extrai em data/<stage>/<name>/."""
    root = _STAGE_DIRS[stage] / name
    if not force and _is_downloaded(root):
        print(f"datasets: {name} já baixado em {root}")
        return root

    import gdown
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "_dataset.zip"

    print(f"datasets: baixando {name} ...")
    result = gdown.download(
        url    = _gdrive_download_url(url),
        output = str(zip_path),
        quiet  = False,
        fuzzy  = True,
    )
    if result is None:
        raise RuntimeError(f"Falha ao baixar {url}")

    print(f"datasets: extraindo {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink(missing_ok=True)

    _flatten_into(root, _locate_dataset_root(root))
    (root / _DOWNLOAD_COMPLETE).write_text("ok\n", encoding="utf-8")
    print(f"datasets: {name} pronto em {root}")
    return root


def download_crowdhuman(hf_repo, stage=_DEFAULT_STAGE, force=False):
    """Baixa o CrowdHuman do HuggingFace e converte anotações ODGT → YOLO."""
    HF_ZIPS = {
        "train": ("CrowdHuman_train01.zip", "CrowdHuman_train02.zip", "CrowdHuman_train03.zip"),
        "valid": ("CrowdHuman_val.zip",),
    }
    HF_ODGT = {
        "train": "annotation_train.odgt",
        "valid": "annotation_val.odgt",
    }

    root = _STAGE_DIRS[stage] / "crowdhuman"
    if not force and _is_downloaded(root):
        print(f"datasets: crowdhuman já baixado em {root}")
        return root

    from huggingface_hub import hf_hub_download
    from PIL import Image

    print(f"datasets: baixando CrowdHuman de {hf_repo}")

    for split, zips in HF_ZIPS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for zname in zips:
            zpath = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=zname)
            print(f"datasets:   extraindo {zname}")
            with zipfile.ZipFile(zpath) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    if not member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    out = img_dir / Path(member.filename).name
                    if not out.exists():
                        with zf.open(member) as src, out.open("wb") as dst:
                            shutil.copyfileobj(src, dst)

        odgt_name = HF_ODGT.get(split)
        if odgt_name:
            odgt_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=odgt_name)
            print(f"datasets:   convertendo {odgt_name}")
            _odgt_to_yolo(Path(odgt_path), img_dir, lbl_dir, Image)

    (root / _DOWNLOAD_COMPLETE).write_text("ok\n", encoding="utf-8")
    print(f"datasets: crowdhuman pronto em {root}")
    return root


def _odgt_to_yolo(odgt, img_dir, lbl_dir, Image):
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
        rows = []
        for box in rec.get("gtboxes", []):
            if box.get("tag") != "person" or box.get("extra", {}).get("ignore"):
                continue
            x, y, w, h = box["fbox"]
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, iw), min(y + h, ih)
            if x1 <= x0 or y1 <= y0:
                continue
            cx = ((x0 + x1) / 2) / iw
            cy = ((y0 + y1) / 2) / ih
            bw = (x1 - x0) / iw
            bh = (y1 - y0) / ih
            rows.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{rec['ID']}.txt").write_text("\n".join(rows), encoding="utf-8")
        converted += 1
    print(f"datasets:   {converted} labels escritos em {lbl_dir}")


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def prepare(names, stage=_DEFAULT_STAGE):
    """Gera o data.yaml a partir dos datasets já baixados; retorna o Path do yaml."""
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

    if single:
        out_dir = roots[0]
    else:
        out_dir = _STAGE_DIRS[stage] / "_combined" / "+".join(names)
        out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out.resolve()
