"""Helpers para carregar configs em `experiments/*.yaml`."""

from src import config
import yaml

EXPERIMENTS_DIR = config.PROJECT_ROOT / "experiments"

_MERGE_KEYS = {"train_config", "augment"}


def path(name):
    return EXPERIMENTS_DIR / f"{name}.yaml"


def available():
    """Lista os experimentos disponíveis."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml"))


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if k in _MERGE_KEYS and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def load(name=None, override=None):
    """Carrega um experimento por nome e aplica override opcional."""
    cfg = {}
    if name:
        cfg = yaml.safe_load(path(name).read_text(encoding="utf-8")) or {}
    if override:
        cfg = _merge(cfg, override) if cfg else dict(override)
    return cfg


def resolve(name=None, config=None):
    """Resolve config completa inline ou nome + override parcial."""
    if config:
        if "experiment_id" in config:
            return dict(config)
        return load(name, config)
    return load(name)


def save_resolved(cfg, run_dir):
    """Salva a config resolvida do run."""
    out = run_dir / "experiment.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
