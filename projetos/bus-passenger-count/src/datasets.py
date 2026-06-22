"""Módulo para download, extração e preparação de conjuntos de dados(datasets)."""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union, Tuple

import yaml

from src import config

CLASS_NAMES: List[str] = ["person"]
_DEFAULT_STAGE: str = "processed"
_DOWNLOAD_COMPLETE: str = ".download_complete"

_STAGE_DIRS: Dict[str, Path] = {
    "raw":       config.RAW_DIR,
    "interim":   config.INTERIM_DIR,
    "processed": config.PROCESSED_DIR,
}

_ROBOFLOW_SOURCES: Dict[str, Dict[str, Any]] = {
    "inside-bus-view": {
        "workspace": "cristians-workspace-9z79y",
        "project": "inside-bus-view-interim-clean-20260615-151915-3vue1",
        "keep_classes": {"person", "0"},
        "versions": {
            "raw": 1,
            "interim": None,
            "processed": None,
        },
    },
    "passenger-deakin": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-deakin-interim-clean-20260615-151915-pmurz",
        "keep_classes": {"person", "0"},
        "versions": {
            "raw": 1,
            "interim": None,
            "processed": None,
        },
    },
    "passenger-detection-bus": {
        "workspace": "cristians-workspace-9z79y",
        "project": "passenger-detection-bus-interim-clean-20260615-152755-ilzsy",
        "keep_classes": {"person", "0"},
        "versions": {
            "raw": 1,
            "interim": None,
            "processed": None,
        },
    },
}


# -------------------------------------------------------------
# Funções de utilidade para manipulação de datasets locais
# -------------------------------------------------------------

def available(stage: str = _DEFAULT_STAGE) -> List[str]:
    f"""Retorna uma lista de conjuntos de dados já baixados em data/<stage>/."""
    stage_dir: Path = _STAGE_DIRS[stage]
    if not stage_dir.exists():
        return []
    return sorted(
        d.name for d in stage_dir.iterdir()
        if d.is_dir() and (d / _DOWNLOAD_COMPLETE).exists()
    )


def _find_child(parent: Path, name: str) -> Optional[Path]:
    f"""Encontra um diretório filho pelo nome (sem distinção entre maiúsculas e minúsculas)."""
    if not parent.exists():
        return None
    target: str = name.lower()
    for child in parent.iterdir():
        if child.name.lower() == target:
            return child
    return None


def _split_images(root: Path, split: str) -> Optional[Path]:
    f"""Localiza o diretório de imagens para uma divisão específica (treino/validação/teste)."""
    split_dir: Optional[Path] = _find_child(root, split)
    if split_dir is None or not split_dir.is_dir():
        return None
    images: Optional[Path] = _find_child(split_dir, "images")
    return images.resolve() if images and images.is_dir() else None


def _has_yolo_layout(root: Path) -> bool:
    f"""Verifica se a estrutura de diretórios corresponde ao layout do YOLO."""
    return root.exists() and all(_split_images(root, s) for s in ("train", "valid"))

def _is_downloaded(root: Path) -> bool:
    f"""Valida se o conjunto de dados foi totalmente baixado e está estruturado."""
    return (root / _DOWNLOAD_COMPLETE).exists() and _has_yolo_layout(root)

def _locate_dataset_root(start: Path) -> Path:
    f"""Realiza uma busca recursiva pelo diretório raiz do conjunto de dados YOLO."""
    if _has_yolo_layout(start):
        return start
    for child in start.rglob("*"):
        if child.is_dir() and _has_yolo_layout(child):
            return child
    raise RuntimeError(f"Nenhum train/valid divisão encontrada em {start}")


def _flatten_into(root: Path, source: Path) -> None:
    f"""Move todo o conteúdo do diretório de origem para o diretório raiz."""
    if source == root:
        return
    for item in source.iterdir():
        dst: Path = root / item.name
        if not dst.exists():
            shutil.move(str(item), str(dst))
    source.rmdir()


def _parse_label_row(parts: List[str]) -> Optional[List[float]]:
    f"""Analisa as coordenadas das linhas dos rótulos YOLO e as normaliza."""
    coords: List[float] = [float(v) for v in parts[1:]]
    if len(coords) == 4:
        return coords

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs: List[float] = coords[0::2]
        ys: List[float] = coords[1::2]
        min_x: float = min(xs)
        max_x: float = max(xs)
        min_y: float = min(ys)
        max_y: float = max(ys)
        w: float = max_x - min_x
        h: float = max_y - min_y
        
        if w <= 0 or h <= 0:
            return None
            
        cx: float = min_x + (w / 2.0)
        cy: float = min_y + (h / 2.0)
        return [cx, cy, w, h]

    return None


def _normalize_to_single_person_class(root: Path, keep_classes: Set[str]) -> None:
    f"""Filtra os rótulos do conjunto de dados para manter apenas a classe especificada (por exemplo, 'pessoa')."""
    data_yaml: Path = root / "data.yaml"
    if not data_yaml.exists():
        return

    spec: Dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = spec.get("names", [])
    
    id_to_name: Dict[int, str]
    if isinstance(names, dict):
        id_to_name = {int(k): str(v).lower() for k, v in names.items()}
    else:
        id_to_name = {i: str(v).lower() for i, v in enumerate(names)}

    keep_ids: Set[int] = {idx for idx, name in id_to_name.items() if name in keep_classes}

    converted: int = 0
    dropped: int = 0
    
    for split in ("train", "valid", "test"):
        lbl_dir: Path = root / split / "labels"
        if not lbl_dir.exists():
            continue
            
        for label_path in lbl_dir.glob("*.txt"):
            rows: List[str] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts: List[str] = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                cls_id: int = int(float(parts[0]))
                if cls_id not in keep_ids:
                    dropped += 1
                    continue
                    
                xywh: Optional[List[float]] = _parse_label_row(parts)
                if xywh is None:
                    dropped += 1
                    continue
                    
                rows.append("0 " + " ".join(f"{v:.6f}" for v in xywh))
                converted += 1
                
            label_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"datasets: rótulos(labels) normalizados({converted} caixas(boxes) guardadas, {dropped} baixados(dropped)).")


# -------------------------------------------------------------
# Download API
# -------------------------------------------------------------

def download_roboflow(name: str, stage: str = _DEFAULT_STAGE, force: bool = False, model_format: str = "yolov8") -> Path:
    f"""Faz o download de um conjunto de dados do Roboflow para data/<stage>/<name>/."""
    root: Path = _STAGE_DIRS[stage] / name
    
    if not force and _is_downloaded(root):
        print(f"datasets: {name} já está baixado em {root}")
        return root

    if name not in _ROBOFLOW_SOURCES:
        raise KeyError(f"Dataset '{name}' não está mapeado em _ROBOFLOW_SOURCES.")
        
    source: Dict[str, Any] = _ROBOFLOW_SOURCES[name]
    version: Optional[int] = source["versions"].get(stage)
    
    if not version:
        raise ValueError(f"Nenhuma versão do Roboflow configurada para '{name}' na etapa '{stage}'.")

    api_key: str = (config.ROBOFLOW_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY está faltando no arquivo .env .")

    from roboflow import Roboflow

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"datasets: downloading {name} ({source['workspace']}/{source['project']}/{version}) ...")
    
    rf: Roboflow = Roboflow(api_key=api_key)
    project = rf.workspace(source["workspace"]).project(source["project"])
    project.version(int(version)).download(
        model_format,
        location=str(root),
        overwrite=True,
    )

    _flatten_into(root, _locate_dataset_root(root))
    _normalize_to_single_person_class(root, source["keep_classes"])
    (root / _DOWNLOAD_COMPLETE).write_text("ok\n", encoding="utf-8")
    
    print(f"datasets: {name} está pronto em {root}")
    return root


def download_selected(stage: str = _DEFAULT_STAGE, names: Optional[List[str]] = None, force: bool = False, model_format: str = "yolov8") -> List[Path]:
    f"""Faz o download de uma lista de conjuntos de dados mapeados do Roboflow."""
    dataset_names: List[str] = list(names or _ROBOFLOW_SOURCES.keys())
    downloaded: List[Path] = []
    
    for name in dataset_names:
        downloaded.append(
            download_roboflow(
                name=name,
                stage=stage,
                force=force,
                model_format=model_format,
            )
        )
    return downloaded


# -------------------------------------------------------------
# data.yaml
# -------------------------------------------------------------

def prepare(names: Union[str, List[str]], stage: str = _DEFAULT_STAGE) -> Dict[str, Any]:
    f"""Compila a configuração de dados do YOLO e a retorna como um dicionário."""
    dataset_names: List[str] = [names] if isinstance(names, str) else list(names)
    roots: List[Path] = []
    
    for name in dataset_names:
        root: Path = _STAGE_DIRS[stage] / name
        if not _is_downloaded(root):
            raise FileNotFoundError(f"Dataset '{name}' Não encontrado em {root}. Execute o Notebook 1 de Download primeiro.")
        roots.append(root)

    train_imgs: List[Path] = [p for r in roots if (p := _split_images(r, "train"))]
    val_imgs: List[Path] = [p for r in roots if (p := _split_images(r, "valid"))]
    test_imgs: List[Path] = [p for r in roots if (p := _split_images(r, "test"))]

    if not train_imgs:
        raise RuntimeError(f"Nenhum 'train/images' divisão encontrada para {dataset_names}.")
    if not val_imgs:
        raise RuntimeError(f"Nenhum 'valid/images' divisão encontrada para {dataset_names}.")

    single: bool = len(dataset_names) == 1
    payload: Dict[str, Any] = {
        "train": str(train_imgs[0]) if single else [str(p) for p in train_imgs],
        "val":   str(val_imgs[0])   if single else [str(p) for p in val_imgs],
        "nc":    len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    
    if test_imgs:
        payload["test"] = str(test_imgs[0]) if single else [str(p) for p in test_imgs]

    return payload