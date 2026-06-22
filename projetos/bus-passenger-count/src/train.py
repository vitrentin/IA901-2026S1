"""Ponto de entrada de treinamento - infraestrutura de execução dir e wandb."""

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import yaml

from src import config
from src import wandb_utils


def _git_info() -> Dict[str, str]:
    f"""
    Recupera o branch git atual e o hash do commit para rastreamento de experimentos.
    """
    def _cmd(args: List[str]) -> str:
        return subprocess.check_output(
            args, stderr=subprocess.DEVNULL, text=True,
            cwd=str(config.PROJECT_ROOT),
        ).strip()
        
    return {
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
    }


def _make_run_dir(experiment_id: str) -> Tuple[Path, str]:
    f"""
    Cria o diretório de execução e gera um nome de execução exclusivo com registro de data e hora.
    """
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name: str = f"{experiment_id or 'run'}_{ts}"
    run_dir: Path = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir, run_name


def _log_yolo_history(save_dir: Path) -> None:
    f"""
    Analisa o arquivo results.csv do YOLO e registra as métricas históricas de treinamento no wandb.
    """
    csv_path: Path = save_dir / "results.csv"
    if not csv_path.exists():
        return
        
    logged: int = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step: Optional[int] = int(float(row["epoch"])) + 1 if row.get("epoch") else None
            payload: Dict[str, float] = {}
            
            for k, v in row.items():
                key: str = (k or "").strip()
                if not key or key == "epoch":
                    continue
                try:
                    payload[key] = float(v)
                except (TypeError, ValueError):
                    continue
                    
            if payload:
                wandb_utils.log_metrics(payload, step=step)
                logged += 1
                
    if logged:
        print(f"train: {logged} epochs conectado no wandb")


def _resolve_data_arg(data_spec: Any, run_dir: Path) -> str:
    f"""
    Converte a especificação do conjunto de dados em um arquivo YAML de tempo de execução, caso um dicionário seja fornecido.
    """
    if isinstance(data_spec, dict):
        data_file: Path = Path(run_dir) / "_data_runtime.yaml"
        data_file.write_text(yaml.safe_dump(data_spec, sort_keys=False), encoding="utf-8")
        return str(data_file)
        
    return str(data_spec)


def run(experiment_id: str, model: Any, data_spec: Any, train_config: Optional[Dict[str, Any]] = None) -> Path:
    f"""
    Cria o diretório de execução, inicializa o registro de logs do wandb, treina o modelo YOLO e salva os pesos.
    """
    run_dir: Path
    run_name: str
    run_dir, run_name = _make_run_dir(experiment_id)
    print(f"train: 'run dir' -> {run_dir}")

    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "data":          data_spec,
            "train_config":  train_config,
            **_git_info(),
        },
        run_name=run_name,
        run_dir=run_dir,
    )

    params: Dict[str, Any] = dict(train_config or {})
    params["data"] = _resolve_data_arg(data_spec, run_dir)
    params["project"] = str(run_dir)
    params["name"] = "train"
    params["exist_ok"] = True

    t0: float = time.time()
    results: Any = model.train(**params)
    dt: float = time.time() - t0
    print(f"train: completado em {dt:.1f}s")

    save_dir: Path = Path(results.save_dir)
    best: Path = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"

    _log_yolo_history(save_dir)
    (run_dir / "weights.txt").write_text(str(best) + "\n")
    
    wandb_utils.log_metrics({"train/duration_sec": dt})
    wandb_utils.finish_run({"train/duration_sec": dt})
    
    return run_dir
