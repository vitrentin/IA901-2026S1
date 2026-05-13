# Setup e execução (uv + notebook)

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

## 3) Configurar variáveis de ambiente

```powershell
copy .env.example .env
```

- Para dataset público de teste (já configurado por padrão), não é obrigatório `ROBOFLOW_API_KEY`.
- Para datasets privados no Roboflow, preencha `ROBOFLOW_API_KEY` no `.env`.
- Para log centralizado no mesmo workspace do time, use os mesmos valores de:
  - `WANDB_ENTITY`
  - `WANDB_PROJECT`

## 4) Login no Weights & Biases (uma vez)

```powershell
uv run wandb login
```

## 5) Usar o notebook com o Python do `.venv`

Opção A (recomendada): iniciar Jupyter via `uv`:

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