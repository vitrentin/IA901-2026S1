"""Wrapper do Wandb usado pelos módulos de treinamento e avaliação."""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import yaml

from src import config

try:
    import wandb
except ImportError:
    wandb = None


def _active_run() -> Any:
    f"""Retorna o objeto de execução wandb ativo, se inicializado."""
    return wandb.run if wandb is not None else None


def init_run(wandb_config: Dict[str, Any], run_name: str, run_dir: Path, notes: Optional[str] = None, tags: Optional[List[str]] = None) -> bool:
    f"""Inicia uma execução do wandb; retorna False se o wandb não estiver disponível."""
    if wandb is None:
        return False
        
    project: str = os.environ.get("WANDB_PROJECT") or config.WANDB_PROJECT
    entity: Optional[str] = os.environ.get("WANDB_ENTITY") or config.WANDB_ENTITY
    
    if entity and "/" in entity:
        entity = entity.split("/", 1)[0].strip()
        print(f"wandb: entidade normalizada para '{entity}' de WANDB_ENTITY")

    run_dir_path: Path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    
    run = wandb.init(
        project = project,
        entity = entity,
        name = run_name,
        config = wandb_config,
        dir = str(run_dir_path),
        notes = notes,
        tags = tags,
        reinit = "finish_previous",
    )
    print(f"wandb: run '{run_name}' -> {run.url}")
    return True


def is_active() -> bool:
    f"""Verifica se uma sessão de rastreamento do wandb está atualmente ativa."""
    return _active_run() is not None

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    f"""Registra as métricas na execução ativa do wandb."""
    if not is_active():
        return
        
    payload: Dict[str, Any] = {k: v for k, v in metrics.items() if v is not None}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def finish_run(summary: Optional[Dict[str, Any]] = None) -> None:
    f"""Fecha a execução do wandb e registra as métricas resumidas finais."""
    run: Any = _active_run()
    if run is None:
        return
        
    if summary:
        final: Dict[str, Any] = {
            f"final/{k}": v for k, v in summary.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if final:
            wandb.log(final)
            for k, v in final.items():
                run.summary[k] = v
                
    wandb.finish()
    print("wandb: execução(run) finalizada")


def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    f"""Lê caixas delimitadoras no formato YOLO a partir de um arquivo de texto."""
    if not label_path.exists():
        return []
        
    out: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts: List[str] = line.strip().split()
        if len(parts) < 5:
            continue
            
        c: int = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        out.append((c, x, y, w, h))
        
    return out


def _yolo_to_minmax(x: float, y: float, w: float, h: float) -> Dict[str, float]:
    f"""Converte as coordenadas do centro YOLO para o formato mínimo/máximo."""
    return {
        "minX": max(x - w / 2.0, 0.0),
        "maxX": min(x + w / 2.0, 1.0),
        "minY": max(y - h / 2.0, 0.0),
        "maxY": min(y + h / 2.0, 1.0),
    }

def _resolve_test_image_dir(data_spec: Union[Dict[str, Any], str, Path]) -> Path:
    f"""Resolve o diretório contendo imagens de teste a partir da especificação de dados."""
    spec: Dict[str, Any]
    base_dir: Path
    
    if isinstance(data_spec, dict):
        spec = data_spec
        base_dir = Path.cwd()
    else:
        data_yaml: Path = Path(data_spec)
        with data_yaml.open() as f:
            spec = yaml.safe_load(f)
        base_dir = data_yaml.parent

    split_dirs: Union[str, List[str], None] = spec.get("test") or spec.get("val")
    if split_dirs is None:
        raise FileNotFoundError("A especificação de dados não define nenhuma divisão 'test' nem 'val' ")
        
    if not isinstance(split_dirs, list):
        split_dirs = [split_dirs]

    def _resolve_one(split_dir: str) -> Optional[Path]:
        p: Path = Path(split_dir)
        if p.is_absolute() and p.exists():
            return p.resolve()

        candidates: List[Path] = []
        base: Path = Path(spec.get("path", base_dir))
        
        if not base.is_absolute():
            base = (base_dir / base).resolve()
            
        candidates.append((base / p).resolve())
        candidates.append((base_dir / p).resolve())

        normalized_parts: List[str] = [part for part in p.parts if part not in ("..", ".")]
        if normalized_parts:
            candidates.append((base_dir / Path(*normalized_parts)).resolve())

        for c in candidates:
            if c.exists():
                return c
        return None

    tried: List[str] = []
    for split_dir in split_dirs:
        resolved: Optional[Path] = _resolve_one(str(split_dir))
        tried.append(str(split_dir))
        if resolved is not None:
            return resolved

    raise FileNotFoundError(f"Não foi possível resolver diretórios divididos {tried}")


def _to_wandb_image_source(result_obj: Any, fallback_path: Path) -> Any:
    f"""Extrai o array de imagem dos resultados de previsão YOLO ou faz fallback para o caminho do arquivo."""
    img: Any = getattr(result_obj, "orig_img", None)
    if img is None:
        return str(fallback_path)
    if hasattr(img, "ndim") and img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()
    return img


def log_test_predictions(predictor: Any, data_spec: Union[Dict[str, Any], str, Path], n: int = 10, names: Optional[Dict[int, str]] = None, conf: float = 0.05, panel_key: str = "test/predictions", seed: int = 0) -> None:
    f"""Envia uma amostra de imagens de teste com caixas delimitadoras de previsão e de verdade fundamental para o wandb."""
    if not is_active():
        print("wandb: Nenhuma execução ativa - ignorando o painel de previsão de testes.")
        return

    spec: Dict[str, Any]
    if isinstance(data_spec, dict):
        spec = data_spec
    else:
        with Path(data_spec).open() as f:
            spec = yaml.safe_load(f)
            
    class_names: Dict[int, str] = names or (
        spec["names"] if isinstance(spec.get("names"), dict)
        else {i: n for i, n in enumerate(spec.get("names", []))}
    )

    img_dir: Path = _resolve_test_image_dir(data_spec)
    label_dir: Path = img_dir.parent / "labels"

    all_imgs: List[Path] = sorted(
        [p for p in img_dir.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    )
    
    if not all_imgs:
        print(f"wandb: Nenhuma imagem encontrada em {img_dir} - ignorando o painel de previsão")
        return

    rng: random.Random = random.Random(seed)
    labeled_imgs: List[Path] = [
        p for p in all_imgs
        if (label_dir / f"{p.stem}.txt").exists()
    ]
    
    pool: List[Path] = labeled_imgs if labeled_imgs else all_imgs
    sample: List[Path] = rng.sample(pool, min(n, len(pool)))
    
    print(f"wandb: Carregando {len(sample)} previsões de teste de {img_dir} (imagens rotuladas: {len(labeled_imgs)}/{len(all_imgs)})")

    wandb_images: List[Any] = []
    for img_path in sample:
        results: Any = predictor.predict(source=str(img_path), conf=conf, verbose=False)
        if not results:
            continue
            
        r: Any = results[0]
        pred_boxes: List[Dict[str, Any]] = []
        
        if r.boxes is not None and len(r.boxes) > 0:
            xywhn: Any = r.boxes.xywhn.cpu().numpy()
            clses: Any = r.boxes.cls.cpu().numpy().astype(int)
            confs: Any = r.boxes.conf.cpu().numpy()
            
            for (x, y, w, h), c, p in zip(xywhn, clses, confs):
                pred_boxes.append({
                    "position": _yolo_to_minmax(float(x), float(y), float(w), float(h)),
                    "class_id": int(c),
                    "box_caption": f"{class_names.get(int(c), str(c))} {p:.2f}",
                    "scores": {"conf": float(p)},
                })

        gt_boxes: List[Dict[str, Any]] = []
        label_path: Path = label_dir / f"{img_path.stem}.txt"
        
        for c, x, y, w, h in _read_yolo_labels(label_path):
            gt_boxes.append({
                "position": _yolo_to_minmax(x, y, w, h),
                "class_id": int(c),
                "box_caption": class_names.get(int(c), str(c)),
            })

        wandb_images.append(wandb.Image(
            _to_wandb_image_source(r, img_path),
            boxes={
                "predictions": {"box_data": pred_boxes,  "class_labels": class_names},
                "ground_truth": {"box_data": gt_boxes,    "class_labels": class_names},
            },
            caption=img_path.name,
        ))

    if wandb_images:
        wandb.log({panel_key: wandb_images})
        print(f"wandb: registrado {len(wandb_images)} imagens em '{panel_key}'")