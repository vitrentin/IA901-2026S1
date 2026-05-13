# Weights & Biases setup

## One-time setup

1. Get your personal API key at <https://wandb.ai/authorize>.
2. Login once:

```powershell
uv run wandb login
```

Notebook login is also fine (`wandb.login()`).

## What each run logs

- Static config from `src/config.py` (split, roboflow settings, YOLO kwargs)
- Per-run metadata (`experiment_id`, tags, notes, git branch/commit)
- Training metrics from Ultralytics callback
- Eval metrics under `test/*` and `final/*`
- Sample prediction panel under `test/predictions`

## Turn wandb off

Set `WANDB_MODE=disabled` before running. The code continues without failing.
