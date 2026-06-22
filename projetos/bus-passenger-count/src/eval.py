"""Ponto de entrada para avaliação — infraestrutura wandb e execução de métricas."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import yaml

from src import config
from src import wandb_utils


def find_run(experiment_id: str) -> Tuple[Optional[Path], Optional[str]]:
    f"""Retorna o diretório de execução (run_dir) e o caminho dos pesos (weights_path) para o ID do experimento fornecido, ou (None, None) se estiver faltando."""
    if not config.RUNS_DIR.exists():
        return None, None
        
    matches: list[Path] = sorted(
        [d for d in config.RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith(f"{experiment_id}_")],
        key=lambda d: d.stat().st_mtime,
    )
    
    if not matches:
        return None, None
        
    run_dir: Path = matches[-1]
    wfile: Path = run_dir / "weights.txt"
    weights: Optional[str] = wfile.read_text().strip() if wfile.exists() else None
    
    return run_dir, weights


def _resolve_data_arg(data_spec: Any, run_dir: Path) -> str:
    f"""Se necessário, converte a especificação do conjunto de dados em um arquivo YAML de tempo de execução."""
    if isinstance(data_spec, dict):
        data_file: Path = Path(run_dir) / "_data_runtime.yaml"
        data_file.write_text(yaml.safe_dump(data_spec, sort_keys=False), encoding="utf-8")
        return str(data_file)
    return str(data_spec)


def run(experiment_id: str, model: Any, data_spec: Any, n_samples: int = config.LOG_N_TEST_IMAGES, run_dir: Optional[Path] = None) -> Dict[str, float]:
    f"""Avalia o modelo na divisão de teste e retorna o dicionário de métricas."""
    if run_dir is None:
        run_dir, _ = find_run(experiment_id)
        
    if run_dir is None:
        run_dir = config.RUNS_DIR / f"eval_{experiment_id}"
        print(f"avaliação: nenhuma execução de treinamento encontrada - criando {run_dir.name}")
        
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_utils.init_run(
        wandb_config={
            "experiment_id": experiment_id,
            "data": data_spec,
        },
        run_name=f"{run_dir.name}_eval",
        run_dir=run_dir,
    )

    results: Any = model.val(
        data = _resolve_data_arg(data_spec, run_dir),
        split = "test",
        project = str(run_dir),
        name = "eval",
        exist_ok = True,
    )

    md: Dict[str, Any] = getattr(results, "results_dict", None) or {}
    metrics: Dict[str, float] = {f"test/{k}": float(v) for k, v in md.items()}

    if metrics:
        wandb_utils.log_metrics(metrics)
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("avaliação(eval): métricas:")
        for k, v in metrics.items():
            print(f" {k:40s} {v:.4f}")

    if n_samples > 0:
        wandb_utils.log_test_predictions(
            predictor = model,
            data_spec = data_spec,
            n = n_samples,
        )

    wandb_utils.finish_run(metrics)
    return metrics
