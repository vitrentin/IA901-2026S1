# Código e notebooks

Este diretório contém os módulos usados pelos notebooks. A ideia é manter os
notebooks curtos e trocar comportamento nos módulos:

- `datasets.py`: usa as chaves exatas do registro de datasets, baixa artefatos do Drive/Hugging Face
  e monta um dicionário de config para YOLO (sem arquivo `data.yaml`).
- `preprocess.py`: transforma `data/interim` em `data/processed` e registra um
  `preprocessing_manifest.json`.
- `train.py`: cria runs, inicializa Weights & Biases e chama o adaptador do
  modelo.
- `eval.py`: avalia modelos treinados ou baselines e registra métricas/imagens.
- `models/`: adaptadores de modelo (`yolo`, `yolo_raw`).

## Fluxo dos notebooks

| Notebook | Objetivo | Artefato principal |
| --- | --- | --- |
| `notebooks/1_download_datasets.ipynb` | Baixar `interim` ou `processed` | diretórios `data/<stage>/<dataset>/` |
| `notebooks/2_preprocess_datasets.ipynb` | Reconstruir `processed` a partir de `interim`, se necessário | `data/processed/<dataset>/preprocessing_manifest.json` |
| `notebooks/3_train.ipynb` | Treinar um experimento | `runs/<experiment>/weights.txt` |
| `notebooks/4_validate_test.ipynb` | Validar/testar e fazer cross-dataset | `runs/<experiment>/test_metrics.json` |

Para criar um novo experimento, duplique apenas `3_train.ipynb` e altere a
célula de configuração. Download, preprocessing e validação continuam
compartilhados.

## Setup

No diretório `projetos/bus-passenger-count`:

```powershell
pip install uv
uv venv
uv sync
copy .env.example .env
uv run wandb login
```

Para abrir Jupyter:

```powershell
uv run jupyter lab
```

Ou abra os notebooks no Cursor/VS Code usando o interpretador:

```text
.venv\Scripts\python.exe
```

Se quiser registrar um kernel nomeado:

```powershell
uv run python -m ipykernel install --user --name bus-passenger-count --display-name "Python (bus-passenger-count)"
```

## Seleção de datasets

Na célula de configuração dos notebooks, defina:

```python
DATASETS = datasets.available()
DATASET_STAGE = "processed"
```

Use `DATASET_STAGE = "interim"` quando for reconstruir preprocessing. Os dados
brutos com rótulos/classes originais ficam documentados em `data/raw/`, mas não
entram no pipeline executável. Os nomes devem ser exatamente as chaves retornadas
por `datasets.available()`. Use `FORCE_DOWNLOAD = True` para baixar novamente.

Com múltiplos datasets, `datasets.prepare()` retorna um config combinado em memória.

Mais detalhes de W&B ficam em `../docs/WANDB.md`.
