# Weights & Biases

## Login

1. Get your personal API key at <https://wandb.ai/authorize>.
2. Login once:

```powershell
uv run wandb login
```

Notebook login is also fine (`wandb.login()`).

## Logged data

- Static config from `src/config.py` (YOLO augmentation kwargs, seed)
- Per-run metadata (`experiment_id`, `datasets`, tags, notes, git branch/commit)
- Training metrics from Ultralytics callback
- Eval metrics under `test/*` and `final/*`
- Sample prediction panel under `test/predictions`

## Disable

Set `WANDB_MODE=disabled` before running.
