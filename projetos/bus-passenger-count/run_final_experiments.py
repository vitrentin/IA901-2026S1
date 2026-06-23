"""
One-shot final experiment suite.

Run:
  uv run python run_final_experiments.py

The suite follows the project story:
1) zero-shot baselines,
2) single-domain public training,
3) two-public-dataset baseline,
4) augmentation,
5) two-stage CrowdHuman transfer,
6) private-domain adaptation,
7) final nano/medium/large size trade-off.
"""

import copy
import json
import traceback
import argparse
from datetime import datetime
from pathlib import Path

from src import eval as eval_module
from src import experiments
from src import train


RESULTS_JSON = Path("final_experiment_results.json")
REPORT_MD = Path("FINAL_EXPERIMENT_RESULTS.md")

BASELINE_EXPERIMENTS = [
    "e0-baseline-medium",
    "e0-baseline-nano",
    "e0-baseline-large",
]

SINGLE_DOMAIN_EXPERIMENTS = [
    "e1-dkn",
    "e1-ibv",
]

PUBLIC_BASELINE_EXPERIMENTS = [
    "e1-public-noaug",
]

AUGMENTATION_EXPERIMENTS = [
    "e2-public-default",
    "e2-public-busaug",
]

TWO_STAGE_EXPERIMENTS = [
    "e3-public-crowd-default",
    "e3-public-crowd-busaug",
]

PRIVATE = "onibus-unicamp-private"
ALL_EVAL_DATASETS = [
    "passenger-detection-bus",
    "inside-bus-view",
    "passenger-deakin",
    PRIVATE,
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def metric(ds_metrics, key, default=0.0):
    try:
        return float(ds_metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def summarize_result(eval_result, eval_datasets):
    rows = []
    maes = []
    maps = []
    for ds in eval_datasets:
        metrics = (eval_result or {}).get(ds, {})
        mae = metric(metrics, "count_mae", 999999.0)
        m50 = metric(metrics, "metrics/mAP50(B)", 0.0)
        rows.append(
            {
                "dataset": ds,
                "mAP50": m50,
                "F1": metric(metrics, "F1", 0.0),
                "count_mae": mae,
                "count_me": metric(metrics, "count_me", 0.0),
                "count_rmse": metric(metrics, "count_rmse", 0.0),
            }
        )
        maes.append(mae)
        maps.append(m50)
    return {
        "avg_count_mae": sum(maes) / len(maes),
        "avg_mAP50": sum(maps) / len(maps),
        "rows": rows,
    }


def better(candidate, current):
    if current is None:
        return True
    c = candidate["summary"]
    b = current["summary"]
    if abs(c["avg_count_mae"] - b["avg_count_mae"]) > 1e-9:
        return c["avg_count_mae"] < b["avg_count_mae"]
    return c["avg_mAP50"] > b["avg_mAP50"]


def run_one(cfg):
    exp_id = cfg["experiment_id"]
    eval_datasets = cfg.get("eval_datasets") or ALL_EVAL_DATASETS

    log("=" * 80)
    log(f"Starting {exp_id}")
    log("=" * 80)

    if cfg.get("strategy") != "baseline":
        run_dir = train.run_experiment(cfg)
        log(f"Training completed: {run_dir}")
    else:
        log("Baseline: evaluating pretrained weights without training")

    eval_result = eval_module.run_experiment(cfg)
    summary = summarize_result(eval_result, eval_datasets)
    log(
        f"Finished {exp_id}: avg_count_mae={summary['avg_count_mae']:.4f}, "
        f"avg_mAP50={summary['avg_mAP50']:.4f}"
    )
    return {
        "experiment_id": exp_id,
        "config": cfg,
        "eval_result": eval_result,
        "summary": summary,
    }


def run_named(name):
    return run_one(experiments.load(name))


def add_private_to_training(cfg, exp_id, weights="yolo11m.pt"):
    out = copy.deepcopy(cfg)
    out["experiment_id"] = exp_id
    out["weights"] = weights
    out["eval_datasets"] = list(ALL_EVAL_DATASETS)

    if out.get("strategy") == "baseline":
        raise ValueError("Cannot adapt a baseline config.")

    if out.get("strategy") == "two_stage":
        stages = out.get("stages") or []
        last = stages[-1]
        train_datasets = list(last.get("train_datasets") or [])
        if PRIVATE not in train_datasets:
            train_datasets.append(PRIVATE)
        last["train_datasets"] = train_datasets
        return out

    train_datasets = list(out.get("train_datasets") or [])
    if PRIVATE not in train_datasets:
        train_datasets.append(PRIVATE)
    out["train_datasets"] = train_datasets
    return out


def with_model(cfg, exp_id, weights):
    out = copy.deepcopy(cfg)
    out["experiment_id"] = exp_id
    out["weights"] = weights
    return out


def write_report(records, failures, best_aug, best_public, private_medium, best_final):
    lines = []
    lines.append("# Final Experiment Results")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("- Dataset stage: `processed`")
    lines.append("- Main selection criterion: lower average `count_mae` across all test datasets.")
    lines.append("- Tie-breaker: higher average `mAP50`.")
    lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        for exp_id, err in failures:
            lines.append(f"- `{exp_id}`: `{err}`")
        lines.append("")

    lines.append("## Decisions")
    lines.append("")
    if best_aug:
        lines.append(
            f"- Best public augmentation/direct baseline: `{best_aug['experiment_id']}` "
            f"(avg_count_mae={best_aug['summary']['avg_count_mae']:.4f}, "
            f"avg_mAP50={best_aug['summary']['avg_mAP50']:.4f})."
        )
    if best_public:
        lines.append(
            f"- Best public-only medium model after two-stage check: `{best_public['experiment_id']}` "
            f"(avg_count_mae={best_public['summary']['avg_count_mae']:.4f}, "
            f"avg_mAP50={best_public['summary']['avg_mAP50']:.4f})."
        )
    if private_medium:
        lines.append(
            f"- Private-adapted medium model: `{private_medium['experiment_id']}` "
            f"(avg_count_mae={private_medium['summary']['avg_count_mae']:.4f}, "
            f"avg_mAP50={private_medium['summary']['avg_mAP50']:.4f})."
        )
    if best_final:
        lines.append(
            f"- Best final size/adaptation candidate: `{best_final['experiment_id']}` "
            f"(avg_count_mae={best_final['summary']['avg_count_mae']:.4f}, "
            f"avg_mAP50={best_final['summary']['avg_mAP50']:.4f})."
        )
    lines.append("")

    lines.append("## Per-dataset Metrics")
    lines.append("")
    lines.append("| experiment | dataset | mAP50 | F1 | count_mae | count_me | count_rmse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        for row in rec["summary"]["rows"]:
            lines.append(
                f"| {rec['experiment_id']} | {row['dataset']} | "
                f"{row['mAP50']:.4f} | {row['F1']:.4f} | "
                f"{row['count_mae']:.4f} | {row['count_me']:.4f} | {row['count_rmse']:.4f} |"
            )
    lines.append("")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_run(records, failures, cfg_or_name):
    exp_id = cfg_or_name if isinstance(cfg_or_name, str) else cfg_or_name["experiment_id"]
    try:
        rec = run_named(cfg_or_name) if isinstance(cfg_or_name, str) else run_one(cfg_or_name)
        records.append(rec)
        return rec
    except Exception as e:
        failures.append((exp_id, str(e)))
        log(traceback.format_exc())
        return None


def _label(cfg_or_name):
    if isinstance(cfg_or_name, str):
        cfg = experiments.load(cfg_or_name)
    else:
        cfg = cfg_or_name
    strategy = cfg.get("strategy", "direct")
    weights = cfg.get("weights", "")
    stage = cfg.get("dataset_stage", "")
    train_datasets = []
    if strategy == "two_stage":
        train_datasets = [
            f"{st.get('name')}: {', '.join(st.get('train_datasets') or [])}"
            for st in cfg.get("stages", []) or []
        ]
    else:
        train_datasets = cfg.get("train_datasets") or []
    return {
        "experiment_id": cfg["experiment_id"],
        "strategy": strategy,
        "weights": weights,
        "dataset_stage": stage,
        "train_datasets": train_datasets,
        "eval_datasets": cfg.get("eval_datasets") or ALL_EVAL_DATASETS,
    }


def _print_dry_run():
    names = []
    names.extend(BASELINE_EXPERIMENTS)
    names.extend(SINGLE_DOMAIN_EXPERIMENTS)
    names.extend(PUBLIC_BASELINE_EXPERIMENTS)
    names.extend(AUGMENTATION_EXPERIMENTS)
    names.extend([
        "<dynamic e4-private-adapt-medium from best e1/e2 public result>",
        "<dynamic e5-private-adapt-nano from e4-private-adapt-medium>",
        "<dynamic e5-private-adapt-large from e4-private-adapt-medium>",
    ])
    names.extend(TWO_STAGE_EXPERIMENTS)

    log("DRY RUN: final experiment execution order")
    for i, name in enumerate(names, start=1):
        if name.startswith("<dynamic"):
            log(f"{i:02d}. {name}")
            continue
        info = _label(name)
        log(
            f"{i:02d}. {info['experiment_id']} | {info['strategy']} | "
            f"{info['weights']} | {info['dataset_stage']}"
        )
        log(f"    train: {info['train_datasets'] or '(none)'}")
        log(f"    eval:  {info['eval_datasets']}")


def main(dry_run=False):
    if dry_run:
        _print_dry_run()
        return 0

    records = []
    failures = []

    log("Final suite: story-driven experiment sequence")
    log("All YAML configs use dataset_stage=processed.")

    for name in BASELINE_EXPERIMENTS:
        try_run(records, failures, name)

    for name in SINGLE_DOMAIN_EXPERIMENTS:
        try_run(records, failures, name)

    public_direct = []
    for name in PUBLIC_BASELINE_EXPERIMENTS + AUGMENTATION_EXPERIMENTS:
        rec = try_run(records, failures, name)
        if rec:
            public_direct.append(rec)

    best_aug = None
    for rec in public_direct:
        if better(rec, best_aug):
            best_aug = rec

    # Run the private adaptation and size comparison before the expensive
    # two-stage experiments so useful final-candidate results appear earlier.
    best_public = best_aug

    private_medium = None
    if best_public:
        cfg = add_private_to_training(best_public["config"], "e4-private-adapt-medium", "yolo11m.pt")
        private_medium = try_run(records, failures, cfg)

    best_final = private_medium
    if private_medium:
        for exp_id, weights in [
            ("e5-private-adapt-nano", "yolo11n.pt"),
            ("e5-private-adapt-large", "yolo11l.pt"),
        ]:
            cfg = with_model(private_medium["config"], exp_id, weights)
            rec = try_run(records, failures, cfg)
            if rec and better(rec, best_final):
                best_final = rec

    for name in TWO_STAGE_EXPERIMENTS:
        rec = try_run(records, failures, name)
        if rec and better(rec, best_public):
            best_public = rec

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": records,
        "failures": failures,
        "best_aug": best_aug,
        "best_public": best_public,
        "private_medium": private_medium,
        "best_final": best_final,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(records, failures, best_aug, best_public, private_medium, best_final)

    log("=" * 80)
    log(f"Wrote {RESULTS_JSON}")
    log(f"Wrote {REPORT_MD}")
    log(f"Finished with {len(failures)} failures")
    log("=" * 80)
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load configs and print execution order without training/evaluation.",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
