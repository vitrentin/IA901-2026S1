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
| `notebooks/4_evaluate.ipynb` | Validar/testar e fazer cross-dataset | `runs/<experiment>/test_metrics.json` |

## Experimentos (replicabilidade)

Cada experimento é um YAML em `experiments/<id>.yaml`.
Nos notebooks 3 e 4, você só troca o experimento carregado.

```python
from src import experiments
cfg = experiments.load("e2-all")            # carrega do registro
cfg = experiments.load("e2-all", override)  # mescla um dict colado na célula
```

- `load(name, override)` aplica override no YAML (`train_config` e `augment`
  fazem merge por chave; o resto substitui).
- `load(None, dict)` roda 100% inline, sem arquivo.
- `train.run_experiment(cfg)` salva a config resolvida em `runs/<run>/experiment.yaml`.

Estratégias suportadas (`strategy` no YAML):

| strategy | o que faz |
| --- | --- |
| `baseline` | não treina; avalia os pesos base (zero-shot) |
| `direct` | fine-tuning direto em `train_datasets` |
| `two_stage` | etapas sequenciais (`stages`); os pesos passam de uma etapa para a seguinte |

`eval.run_experiment(cfg)` avalia cada item de `eval_datasets` e salva os
resultados em `runs/<run>/test_metrics.json`.

### Processamento e augmentation

- **Augmentation (online):** bloco opcional `augment` no YAML, repassado direto ao
  `model.train()` do Ultralytics (`fliplr`, `mosaic`, `hsv_*`, `degrees`, `scale`,
  `erasing`, `shear`, ...). Para desligar tudo, zere as chaves (ver
  `experiments/e2-all-aug-off.yaml`).
- **Pré-processamento (offline):** escolha `dataset_stage: interim` (sem
  processamento) ou `processed`. A versão `processed` vem do export do Roboflow
  com pré-processamento aplicado (grayscale, auto-contrast, resize, etc.); baixe-a
  no notebook 1 antes de rodar os experimentos `*-proc`.

## Setup

No diretório `projetos/bus-passenger-count`:

```powershell
pip install uv
uv venv
uv sync
copy .env.example .env
uv run wandb login
```

Configure `.env`:

- `ROBOFLOW_API_KEY`: obrigatório para baixar datasets do Roboflow.
- Hugging Face público (ex.: CrowdHuman): não precisa de API key.

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
