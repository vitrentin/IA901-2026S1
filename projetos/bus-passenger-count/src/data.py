"""Dataset resolver/downloader with optional Roboflow pull."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

import config


def _load_env() -> None:
    """Load local `.env` once so notebooks/scripts share defaults."""
    load_dotenv(config.PROJECT_ROOT / ".env", override=False)


def download_dataset(
    workspace: str | None = None,
    project: str | None = None,
    version: int | None = None,
    fmt: str | None = None,
    force_download: bool = False,
) -> Path:
    """Return `data.yaml`, using local cache unless `force_download=True`."""
    workspace = workspace or config.ROBOFLOW_WORKSPACE
    project   = project   or config.ROBOFLOW_PROJECT
    version   = version   if version is not None else config.ROBOFLOW_VERSION
    fmt       = fmt       or config.ROBOFLOW_FORMAT

    on_disk = config.DEFAULT_DATA_YAML
    if not force_download and on_disk.exists():
        print(f"data:     using on-disk dataset -> {on_disk}")
        return on_disk.resolve()

    _load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    from roboflow import Roboflow

    target_dir = config.PROCESSED_DIR / f"{project}-v{version}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"data:     downloading {workspace}/{project} v{version} ({fmt}) -> {target_dir}")
    try:
        rf = Roboflow(api_key=api_key) if api_key else Roboflow()
        dataset = (
            rf.workspace(workspace)
              .project(project)
              .version(version)
              .download(fmt, location=str(target_dir))
        )
    except Exception as e:
        raise RuntimeError(
            "Roboflow download failed. If the dataset requires auth, set "
            "ROBOFLOW_API_KEY in `.env` and try again."
        ) from e

    yaml_path = Path(dataset.location) / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Roboflow download did not produce {yaml_path}")
    print(f"data:     dataset ready -> {yaml_path}")
    return yaml_path.resolve()
