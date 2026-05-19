# Setup e execução

## 1) Instalar `uv`

No Windows (PowerShell):

```powershell
pip install uv
```

## 2) Criar ambiente e instalar dependências

No diretório `projetos/bus-passenger-count`:

```powershell
uv venv
uv sync
```

## 3) Configurar ambiente

```powershell
copy .env.example .env
```

Use `WANDB_ENTITY` e `WANDB_PROJECT` em `.env` para registrar no workspace do time.

## 4) Login no Weights & Biases (uma vez)

```powershell
uv run wandb login
```

## 5) Usar o notebook com o Python do `.venv`

Opção A: iniciar Jupyter via `uv`:

```powershell
uv run jupyter lab
```

Opção B (Cursor/VS Code): abrir `notebooks/bus-passenger-count.ipynb` e selecionar o interpretador:

- Windows: `.venv\Scripts\python.exe`

Se quiser registrar um kernel nomeado:

```powershell
uv run python -m ipykernel install --user --name bus-passenger-count --display-name "Python (bus-passenger-count)"
```

## 6) Executar pipeline

No notebook:

- Edite a célula **Experiment Config** (ex.: `MODEL_NAME`, `RUN_TRAINING`, `TRAIN_OVERRIDES` com `epochs`/`batch`/`imgsz`).
- Rode as células de cima para baixo.
- Para baseline sem treino: `MODEL_NAME = "yolo_raw"` e `RUN_TRAINING = False`.

## 7) Selecionar datasets

Na célula **Experiment Config**, defina um ou mais datasets. Eles são preparados
em `data/processed/` e podem vir do Drive ou do Hugging Face, conforme
`src/datasets.py`.

```python
DATASETS = ["InsideBusView"]
DATASETS = ["CrowdHuman", "InsideBusView", "PassengerDetectionBus"]
```

Nomes aceitam variações de caixa e separador (`InsideBusView`,
`inside-bus-view`, `inside_bus_view`). Use `datasets.available()` para listar
opções e `FORCE_DOWNLOAD = True` para baixar de novo. Com múltiplos datasets,
o notebook gera um `data.yaml` combinado em `data/processed/_combined/`.