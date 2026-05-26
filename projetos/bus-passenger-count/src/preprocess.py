"""Preprocessing helpers that turn interim YOLO datasets into processed data.

The current public bus datasets were prepared in Roboflow and uploaded as
``data/processed`` archives. This module keeps that boundary reproducible:
change preprocessing here, rerun ``notebooks/2_preprocess_datasets.ipynb``,
then train/evaluate with the same downstream notebooks.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from src import config
from src import datasets


_SPLITS = ("train", "valid", "test")
_STAGE_DIRS = {
    "interim": config.INTERIM_DIR,
    "processed": config.PROCESSED_DIR,
}

RECIPES: dict[str, list[str]] = {
    "inside-bus-view": [
        "Keep YOLO splits layout.",
        "Normalize labels to a single class: person.",
        "Use this point to replace Roboflow export if preprocessing changes.",
    ],
    "passenger-deakin": [
        "Keep sampled frames in YOLO splits layout.",
        "Normalize labels to a single class: person.",
    ],
    "passenger-detection-bus": [
        "Keep YOLO splits layout.",
        "Normalize labels to a single class: person.",
    ],
    "crowdhuman": [
        "CrowdHuman's ODGT to YOLO conversion happens in datasets.prepare().",
        "This recipe remains here to explicit the preprocessing step.",
    ],
}


def _stage_dir(stage: str) -> Path:
    return _STAGE_DIRS[stage]


def _copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst} already exists. Set overwrite=True to rebuild."
            )
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(".download_complete", "_dataset.zip", "*.cache"),
    )


def _normalize_label_file(path: Path) -> int:
    """Rewrite YOLO labels so every kept annotation uses class 0 (person)."""
    rows: list[str] = []
    changed = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        if parts[0] != "0":
            changed += 1
        rows.append(" ".join(["0", *parts[1:5]]))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return changed


def _write_single_class_yaml(root: Path) -> Path:
    payload: dict[str, object] = {
        "train": str((root / "train" / "images").resolve()),
        "val": str((root / "valid" / "images").resolve()),
        "nc": 1,
        "names": ["person"],
    }
    if (root / "test" / "images").exists():
        payload["test"] = str((root / "test" / "images").resolve())
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def _manifest(
    dataset: str,
    source: Path,
    target: Path,
    label_files: int,
    relabeled_rows: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source.resolve()),
        "target": str(target.resolve()),
        "class_names": datasets.CLASS_NAMES,
        "recipe": RECIPES.get(dataset, ["Normalize YOLO labels to class 0: person."]),
        "label_files": label_files,
        "relabeled_rows": relabeled_rows,
    }


def run(
    names: str | Iterable[str],
    source_stage: str = "interim",
    target_stage: str = "processed",
    overwrite: bool = False,
) -> list[Path]:
    """Build processed datasets from an existing source stage.

    The function expects each input dataset to already contain YOLO-style
    ``train/valid/test`` split folders under ``data/<source_stage>/<name>``.
    """
    if isinstance(names, str):
        names = [names]

    outputs: list[Path] = []
    source_root = _stage_dir(source_stage)
    target_root = _stage_dir(target_stage)

    for name in names:
        if name not in datasets.REGISTRY:
            raise KeyError(f"Unknown dataset: {name!r}. Use one of: {datasets.available()}")
        source = source_root / name
        target = target_root / name
        if not source.exists():
            raise FileNotFoundError(
                f"{source} does not exist. Run the download notebook first "
                f"with stage={source_stage!r}."
            )

        print(f"preprocess: {name}: {source_stage} -> {target_stage}")
        _copy_tree(source, target, overwrite=overwrite)

        label_files = 0
        relabeled_rows = 0
        for split in _SPLITS:
            labels = target / split / "labels"
            if not labels.exists():
                continue
            for label_file in labels.glob("*.txt"):
                label_files += 1
                relabeled_rows += _normalize_label_file(label_file)

        data_yaml = _write_single_class_yaml(target)
        manifest = _manifest(
            dataset=name,
            source=source,
            target=target,
            label_files=label_files,
            relabeled_rows=relabeled_rows,
        )
        (target / "preprocessing_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        (target / ".download_complete").write_text("ok\n", encoding="utf-8")
        outputs.append(data_yaml.resolve())
        print(f"preprocess: wrote {data_yaml}")

    return outputs

