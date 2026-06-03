#!/usr/bin/env python3
"""
================================================================================
Two-Way OpenML Regression Benchmark  (SLAutoML vs FLAML)
================================================================================
Benchmarks two systems on OpenML regression datasets and produces
comparison plots + tables:

  1. SLAutoML   — script.py (nested-CV, Optuna-tuned)
  2. FLAML      — AutoGluon_FLAML.py (AutoML)

NOTE: AutoGluon is excluded by default — it does not support Python 3.14.
      Re-enable with --with-autogluon once you switch to Python ≤ 3.12.

SUITE CHOICES (--suite):
  269  → OpenML-Reg19   (curated regression, ~18 tasks, RECOMMENDED)
  297  → OpenML-CTR23   (mixed; only regression tasks are kept)
  custom → hard-coded CUSTOM_TASK_IDS list below

DATASET FILTERING:
  --max-rows 10000   Skip tasks whose dataset exceeds this many rows (default 10 000)
                     Keeps RAM usage manageable for large AutoML systems.
  --min-rows 50      Skip tasks that are too small for nested CV.

MEMORY GUARD:
  The script checks available RAM before loading each dataset and skips the task
  if headroom is below --min-free-gb (default 2.0 GB).

OUTPUT:
  benchmark_results/
  ├── benchmark_summary_<timestamp>.csv     ← flat comparison table (one row / system / task)
  ├── benchmark_summary_<timestamp>.json    ← full JSON log (rolling checkpoint)
  ├── comparison_r2_<timestamp>.png         ← grouped-bar chart: R² per dataset
  ├── comparison_rmse_<timestamp>.png       ← grouped-bar chart: RMSE per dataset
  ├── comparison_scatter_<timestamp>.png    ← scatter: SLAutoML R² vs FLAML R²
  └── task_<id>_<name>/                     ← per-task artefacts from each system

USAGE EXAMPLES:
  # Quick smoke test — first 3 tasks from OpenML-Reg19, fast mode
  python benchmark_compare.py --max-tasks 3 --fast

  # Full suite run (may take several hours)
  python benchmark_compare.py --suite 269

  # Specific tasks, 300-second budget per AutoML system
  python benchmark_compare.py --task-ids 2295 4823 --time-budget 300

  # Enable AutoGluon (requires Python ≤ 3.12 and autogluon-tabular installed)
  python benchmark_compare.py --with-autogluon --max-tasks 5 --ag-excluded-model-types NN_TORCH

  # Limit dataset size to 8 000 rows and require ≥ 3 GB free RAM
  python benchmark_compare.py --max-rows 8000 --min-free-gb 3.0

  # Skip specific known-broken or very large tasks
  python benchmark_compare.py --skip-tasks 168350 14969

OPTIONS:
  --suite INT                OpenML benchmark suite ID (default: 269)
  --task-ids INT [INT ...]   Run only these specific task IDs (overrides --suite)
  --max-tasks INT            Limit to first N tasks (quick tests)
  --skip-tasks INT [INT ...] Task IDs to exclude
  --max-rows INT             Maximum dataset rows; larger tasks are skipped (default: 10 000)
  --min-rows INT             Minimum dataset rows (default: 50)
  --fast                     Fast mode: fewer CV folds / Optuna trials / models
  --time-budget INT          Wall-clock seconds per AutoML system per task (default: 180)
  --output-dir STR           Root output directory (default: benchmark_results)
  --min-free-gb FLOAT        Minimum free RAM in GB before loading a task (default: 2.0)
  --no-slautoml              Skip the SLAutoML (script.py) system
  --no-flaml                 Skip FLAML
  --with-autogluon           Enable AutoGluon (disabled by default; requires Python ≤ 3.12)
  --ag-excluded-model-types STR [...]  AutoGluon model types to exclude
  --ag-presets STR           AutoGluon preset (default: medium_quality)
  --no-plots                 Skip generating matplotlib plots at the end
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── OpenML ────────────────────────────────────────────────────────────────────
try:
    import openml
    from openml.tasks import TaskType
except ImportError:
    print("ERROR: openml not installed.  Run:  pip install openml")
    sys.exit(1)

# ── psutil for memory checks ──────────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Our custom pipeline ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from script import (
        MLConfig,
        FeatureConfig,
        TargetConfig,
        from_dataframe,
    )
    HAS_SLAUTOML = True
except ImportError as _e:
    HAS_SLAUTOML = False
    _SLAUTOML_ERR = _e

# ── AutoGluon / FLAML wrappers ────────────────────────────────────────────────
try:
    from AutoGluon_FLAML import (
        BenchmarkConfig,
        BenchmarkRunner,
        HAS_AUTOGLUON,
        HAS_FLAML,
        prepare_xy,
        train_test_split,
        make_cv_splitter,
        plan_time_budgets,
        BackendFactory,
        compute_metrics,
        align_proba_to_classes,
    )
    HAS_AG_FLAML_MODULE = True
except ImportError as _e:
    HAS_AG_FLAML_MODULE = False
    HAS_AUTOGLUON = False
    HAS_FLAML = False
    _AG_FLAML_ERR = _e

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Benchmark3Way")

# ── Curated fallback task list ────────────────────────────────────────────────
CUSTOM_TASK_IDS: list[int] = [
    2295,    # abalone            (4 177 × 8)
    4823,    # space_ga           (3 107 × 6)
    4729,    # kin8nm             (8 192 × 8)
    4990,    # cpu_small          (8 192 × 12)
    7222,    # delta_elevators    (9 517 × 6)
    7589,    # ailerons           (13 750 × 33)
    5636,    # pol                (15 000 × 26)
]

# System name constants
SYS_SL    = "SLAutoML"
SYS_AG    = "AutoGluon"
SYS_FLAML = "FLAML"
ALL_SYSTEMS         = [SYS_SL, SYS_AG, SYS_FLAML]   # full list (used for palette / plots)
DEFAULT_SYSTEMS     = [SYS_SL, SYS_FLAML]             # AG excluded by default

PALETTE = {
    SYS_SL:    "#4C9BE8",   # blue
    SYS_AG:    "#E87D4C",   # orange
    SYS_FLAML: "#4CE87D",   # green
}


# =============================================================================
# UTILITIES
# =============================================================================

def get_free_ram_gb() -> float:
    """Return available RAM in GB (0 if psutil not installed)."""
    if not HAS_PSUTIL:
        return float("inf")
    return psutil.virtual_memory().available / (1024 ** 3)


def get_process_ram_gb() -> float:
    """Return RSS memory used by the current process in GB."""
    if not HAS_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


# =============================================================================
# OPENML HELPERS
# =============================================================================

def fetch_regression_tasks(suite_id: int) -> list[int]:
    """Fetch regression task IDs from an OpenML benchmark suite."""
    logger.info(f"Fetching OpenML suite {suite_id}...")
    try:
        suite    = openml.study.get_suite(suite_id=suite_id)
        task_ids = list(suite.tasks)
        logger.info(f"Suite {suite_id}: {len(task_ids)} tasks total. Filtering to regression...")

        regression_ids: list[int] = []
        failed = 0
        for tid in task_ids:
            try:
                task = openml.tasks.get_task(tid, download_data=False)
                if task.task_type_id == TaskType.SUPERVISED_REGRESSION:
                    regression_ids.append(tid)
            except Exception as e:
                logger.warning(f"  Could not inspect task {tid}: {e}")
                failed += 1

        logger.info(
            f"Found {len(regression_ids)} regression tasks "
            f"({failed} failed to inspect)."
        )
        return regression_ids

    except Exception as e:
        logger.error(f"Failed to load suite {suite_id}: {e}")
        logger.info("Falling back to CUSTOM_TASK_IDS.")
        return list(CUSTOM_TASK_IDS)


def load_openml_task(
    task_id: int,
    max_rows: int,
    min_rows: int,
) -> tuple[pd.DataFrame, str, dict] | None:
    """
    Load one OpenML regression task.

    Returns (df, target_col, meta) or None when the task should be skipped
    (too large, too small, wrong type, or load error).
    """
    logger.info(f"  Loading OpenML task {task_id}...")
    try:
        task    = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        X, y, _categorical_mask, _feature_names = dataset.get_data(
            target=task.target_name,
            dataset_format="dataframe",
        )

        target_col = task.target_name or "target"
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_col)
        else:
            y = y.rename(target_col)

        df = X.copy()
        df[target_col] = y.values

        n, p = len(df), X.shape[1]
        meta = {
            "task_id":       task_id,
            "dataset_name":  dataset.name,
            "n_samples":     n,
            "n_features":    p,
            "target_column": target_col,
            "openml_url":    f"https://www.openml.org/t/{task_id}",
        }
        logger.info(
            f"  → {dataset.name} | n={n} | p={p} | target='{target_col}'"
        )

        if n < min_rows:
            logger.warning(f"  Skipping task {task_id}: only {n} rows (< {min_rows}).")
            return None
        if n > max_rows:
            logger.warning(f"  Skipping task {task_id}: {n} rows exceeds --max-rows {max_rows}.")
            return None

        return df, target_col, meta

    except Exception as e:
        logger.error(f"  Failed to load task {task_id}: {e}")
        return None


# =============================================================================
# SYSTEM RUNNERS
# =============================================================================

def run_slautoml(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    fast: bool,
) -> dict:
    """Run the custom SLAutoML pipeline (script.py) and return metric dict."""
    if not HAS_SLAUTOML:
        return {"status": "skipped_not_installed", "system": SYS_SL}

    start = time.perf_counter()
    try:
        if fast:
            ml_cfg = MLConfig(
                random_state=42,
                n_outer_splits=3,
                n_inner_splits=3,
                n_optuna_trials=15,
                optuna_use_pruning=True,
                optuna_n_startup_trials=5,
                n_jobs=-1,
                output_dir=output_dir,
                compute_shap=False,
                compute_permutation_importance=False,
                compute_conformal=False,
                track_feature_stability=False,
                use_multitask=False,
                use_stacking=False,
                save_best_models=False,
                save_oof_predictions=False,
                save_config_snapshot=False,
                include_models=["dummy", "ridge", "hist_gbrt", "random_forest",
                                "xgboost", "lightgbm"],
                feature_strategies_override=["full", "full_scaled"],
            )
        else:
            ml_cfg = MLConfig(
                random_state=42,
                n_outer_splits=5,
                n_inner_splits=4,
                n_optuna_trials=40,
                optuna_use_pruning=True,
                optuna_n_startup_trials=8,
                n_jobs=-1,
                output_dir=output_dir,
                compute_shap=False,
                compute_permutation_importance=False,
                compute_conformal=False,
                track_feature_stability=False,
                use_multitask=False,
                use_stacking=False,
                save_best_models=False,
                save_oof_predictions=False,
                save_config_snapshot=True,
                include_models=["dummy", "ridge", "elasticnet", "bayesian_ridge",
                                "random_forest", "hist_gbrt", "xgboost", "lightgbm"],
                feature_strategies_override=["full", "full_scaled", "tree_select"],
            )

        feat_cfg = FeatureConfig(
            max_cardinality_for_categorical=50,
            use_variance_threshold=True,
            use_correlation_filter=False,
            expand_datetime=True,
        )
        tgt_cfg = TargetConfig(
            auto_transform=True,
            skew_threshold=1.0,
            preferred_auto_transform="yeo-johnson",
        )

        rm = from_dataframe(
            df=df,
            target_columns=target_col,
            ml_config=ml_cfg,
            feat_config=feat_cfg,
            tgt_config=tgt_cfg,
        ).run()

        best_df = rm.get_best_per_label()
        elapsed = time.perf_counter() - start

        result: dict = {"system": SYS_SL, "status": "success", "elapsed_sec": round(elapsed, 1)}
        if not best_df.empty:
            best = best_df.iloc[0]
            result["r2"]           = float(best.get("oof_r2",   np.nan))
            result["rmse"]         = float(best.get("oof_rmse", np.nan))
            result["mae"]          = float(best.get("oof_mae",  np.nan))
            result["best_model"]   = str(best.get("model", "?"))
            result["best_strategy"]= str(best.get("feature_strategy", "?"))
            result["n_experiments"]= len(rm.to_dataframe())

        logger.info(
            f"  ✓ {SYS_SL} done in {elapsed:.0f}s | "
            f"OOF R²={result.get('r2', float('nan')):.4f}"
        )
        return result

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(f"  ✗ {SYS_SL} FAILED: {exc}")
        return {
            "system": SYS_SL, "status": "error",
            "elapsed_sec": round(elapsed, 1),
            "error": str(exc), "traceback": traceback.format_exc(),
        }


def _run_automl_on_df(
    df: pd.DataFrame,
    target_col: str,
    backend: str,
    output_dir: str,
    time_budget: int,
    ag_presets: str,
    ag_excluded_model_types: list[str],
) -> dict:
    """
    Shared helper: run AutoGluon or FLAML via BenchmarkRunner on an in-memory
    DataFrame by writing it to a temporary CSV.
    """
    if not HAS_AG_FLAML_MODULE:
        return {"status": "skipped_not_installed", "system": backend}

    system_name = SYS_AG if backend == "autogluon" else SYS_FLAML
    start = time.perf_counter()

    # Save DataFrame to a temp CSV outside output_dir so BenchmarkRunner doesn't delete it
    tmp_csv = Path(output_dir).parent / f"_tmp_data_{backend}.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv, index=False)

    try:
        cfg = BenchmarkConfig(
            backend=backend,
            data_path=str(tmp_csv),
            target=target_col,
            task="regression",
            time_budget=time_budget,
            output_dir=output_dir,
            n_splits=3,
            oof_budget_fraction=0.60,
            random_state=42,
            overwrite=True,
            keep_fold_models=False,
            # AutoGluon specific
            ag_presets=ag_presets,
            ag_excluded_model_types=ag_excluded_model_types,
            ag_verbosity=1,
            # FLAML specific
            flaml_n_jobs=-1,
        )

        payload = BenchmarkRunner(cfg).run()
        elapsed = time.perf_counter() - start

        oof     = payload.get("oof_metrics", {})
        holdout = payload.get("holdout_metrics", {})

        result = {
            "system":      system_name,
            "status":      "success",
            "elapsed_sec": round(elapsed, 1),
            # Primary metric: OOF (same protocol across all systems)
            "r2":          float(oof.get("r2",   np.nan)),
            "rmse":        float(oof.get("rmse", np.nan)),
            "mae":         float(oof.get("mae",  np.nan)),
            # Holdout metrics (bonus)
            "holdout_r2":   float(holdout.get("r2",   np.nan)),
            "holdout_rmse": float(holdout.get("rmse", np.nan)),
            "holdout_mae":  float(holdout.get("mae",  np.nan)),
        }
        logger.info(
            f"  ✓ {system_name} done in {elapsed:.0f}s | "
            f"OOF R²={result.get('r2', float('nan')):.4f}"
        )
        return result

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(f"  ✗ {system_name} FAILED: {exc}")
        return {
            "system": system_name, "status": "error",
            "elapsed_sec": round(elapsed, 1),
            "error": str(exc), "traceback": traceback.format_exc(),
        }
    finally:
        # Clean up temp CSV
        try:
            tmp_csv.unlink(missing_ok=True)
        except Exception:
            pass


def run_autogluon(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    time_budget: int,
    ag_presets: str,
    ag_excluded_model_types: list[str],
) -> dict:
    if not HAS_AUTOGLUON:
        return {"system": SYS_AG, "status": "skipped_not_installed"}
    return _run_automl_on_df(
        df=df, target_col=target_col, backend="autogluon",
        output_dir=output_dir, time_budget=time_budget,
        ag_presets=ag_presets,
        ag_excluded_model_types=ag_excluded_model_types,
    )


def run_flaml(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    time_budget: int,
) -> dict:
    if not HAS_FLAML:
        return {"system": SYS_FLAML, "status": "skipped_not_installed"}
    return _run_automl_on_df(
        df=df, target_col=target_col, backend="flaml",
        output_dir=output_dir, time_budget=time_budget,
        ag_presets="medium_quality",       # unused for FLAML
        ag_excluded_model_types=[],
    )


# =============================================================================
# PER-TASK ORCHESTRATOR
# =============================================================================

def run_task(
    task_id: int,
    output_root: Path,
    args: argparse.Namespace,
) -> list[dict]:
    """
    Run all requested systems on one OpenML task.

    Returns a list of result dicts (one per system) — always, never raises.
    Each dict contains the task metadata + system-specific metrics.
    """
    base_meta: dict = {
        "task_id":      task_id,
        "dataset_name": f"task_{task_id}",
        "n_samples":    None,
        "n_features":   None,
        "target_column": None,
        "openml_url":   f"https://www.openml.org/t/{task_id}",
    }

    # ── Memory check ─────────────────────────────────────────────────────────
    free_gb = get_free_ram_gb()
    proc_gb = get_process_ram_gb()
    logger.info(
        f"  RAM: {free_gb:.1f} GB free | "
        f"{proc_gb:.1f} GB used by this process"
    )
    if free_gb < args.min_free_gb:
        logger.warning(
            f"  Skipping task {task_id}: only {free_gb:.1f} GB free RAM "
            f"(< --min-free-gb {args.min_free_gb})."
        )
        skip = {**base_meta, "status": "skipped_low_ram",
                "free_ram_gb_at_skip": round(free_gb, 2)}
        active_systems = [s for s in ALL_SYSTEMS
                         if not (s == SYS_AG and getattr(args, "with_autogluon", False) is False)]
        return [{**skip, "system": s} for s in active_systems]

    # ── Load data ─────────────────────────────────────────────────────────────
    loaded = load_openml_task(task_id, max_rows=args.max_rows, min_rows=args.min_rows)
    if loaded is None:
        skip = {**base_meta, "status": "skipped_size"}
        active_systems = [s for s in ALL_SYSTEMS
                         if not (s == SYS_AG and not getattr(args, "with_autogluon", False))]
        return [{**skip, "system": s} for s in active_systems]

    df, target_col, meta = loaded
    base_meta.update(meta)

    task_dir = output_root / f"task_{task_id}_{meta['dataset_name'][:30]}"
    task_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    # ── Run each system ───────────────────────────────────────────────────────
    if not args.no_slautoml:
        logger.info(f"  [{SYS_SL}] Starting...")
        r = run_slautoml(
            df=df,
            target_col=target_col,
            output_dir=str(task_dir / "slautoml"),
            fast=args.fast,
        )
        results.append({**base_meta, **r})

    if args.with_autogluon:
        logger.info(f"  [{SYS_AG}] Starting...")
        r = run_autogluon(
            df=df,
            target_col=target_col,
            output_dir=str(task_dir / "autogluon"),
            time_budget=args.time_budget,
            ag_presets=args.ag_presets,
            ag_excluded_model_types=args.ag_excluded_model_types or [],
        )
        results.append({**base_meta, **r})

    if not args.no_flaml:
        logger.info(f"  [{SYS_FLAML}] Starting...")
        r = run_flaml(
            df=df,
            target_col=target_col,
            output_dir=str(task_dir / "flaml"),
            time_budget=args.time_budget,
        )
        results.append({**base_meta, **r})

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def _safe_metric(df: pd.DataFrame, system: str, metric: str) -> list[float]:
    sub = df[df["system"] == system]
    return [float(x) if pd.notna(x) else float("nan") for x in sub[metric].tolist()]


def make_plots(summary_df: pd.DataFrame, output_root: Path, run_ts: str) -> None:
    """Generate comparison bar charts and scatter plots."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed — skipping plots. pip install matplotlib")
        return

    ok = summary_df[summary_df["status"] == "success"].copy()
    if ok.empty:
        logger.warning("No successful runs to plot.")
        return

    # Only keep tasks where at least one system succeeded
    datasets = ok["dataset_name"].unique().tolist()
    if not datasets:
        return

    systems_present = [s for s in ALL_SYSTEMS if s in ok["system"].unique()]
    x = np.arange(len(datasets))
    n_sys = len(systems_present)
    width = 0.8 / max(n_sys, 1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # ── 1. Grouped-bar: R² ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.4), 5))
    for i, sys_name in enumerate(systems_present):
        sub = ok[ok["system"] == sys_name].set_index("dataset_name")
        vals = [float(sub.loc[d, "r2"]) if d in sub.index else float("nan") for d in datasets]
        offset = (i - n_sys / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=sys_name,
                      color=PALETTE.get(sys_name, "#888888"), alpha=0.85)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("OOF R²", fontsize=10)
    ax.set_title("OOF R² Comparison — SLAutoML vs AutoGluon vs FLAML", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    fig.tight_layout()
    path = output_root / f"comparison_r2_{run_ts}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {path}")

    # ── 2. Grouped-bar: RMSE ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.4), 5))
    for i, sys_name in enumerate(systems_present):
        sub = ok[ok["system"] == sys_name].set_index("dataset_name")
        vals = [float(sub.loc[d, "rmse"]) if d in sub.index else float("nan") for d in datasets]
        offset = (i - n_sys / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=sys_name,
                      color=PALETTE.get(sys_name, "#888888"), alpha=0.85)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("OOF RMSE (lower is better)", fontsize=10)
    ax.set_title("OOF RMSE Comparison — SLAutoML vs AutoGluon vs FLAML", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_root / f"comparison_rmse_{run_ts}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {path}")

    # ── 3. Scatter: SLAutoML R² vs each competitor ──────────────────────────
    if SYS_SL in systems_present and len(systems_present) > 1:
        competitors = [s for s in systems_present if s != SYS_SL]
        fig, axes = plt.subplots(1, len(competitors),
                                 figsize=(5 * len(competitors), 4.5),
                                 squeeze=False)
        for col, comp in enumerate(competitors):
            ax = axes[0][col]
            sl_vals, comp_vals, names = [], [], []
            for d in datasets:
                sl_sub   = ok[(ok["system"] == SYS_SL)   & (ok["dataset_name"] == d)]
                comp_sub = ok[(ok["system"] == comp)      & (ok["dataset_name"] == d)]
                if sl_sub.empty or comp_sub.empty:
                    continue
                sl_r2   = float(sl_sub["r2"].iloc[0])
                comp_r2 = float(comp_sub["r2"].iloc[0])
                if np.isfinite(sl_r2) and np.isfinite(comp_r2):
                    sl_vals.append(sl_r2)
                    comp_vals.append(comp_r2)
                    names.append(d)

            if sl_vals:
                sc = ax.scatter(sl_vals, comp_vals, c=[PALETTE.get(comp, "#888")],
                                alpha=0.8, s=80, edgecolors="white", linewidths=0.5)
                for xi, yi, name in zip(sl_vals, comp_vals, names):
                    ax.annotate(name[:12], (xi, yi), textcoords="offset points",
                                xytext=(4, 4), fontsize=6, alpha=0.7)

                lims = [min(sl_vals + comp_vals) - 0.05, max(sl_vals + comp_vals) + 0.05]
                ax.plot(lims, lims, "--", color="gray", lw=0.8, alpha=0.6, label="y = x")
                ax.set_xlim(lims); ax.set_ylim(lims)
                ax.set_xlabel(f"{SYS_SL} OOF R²", fontsize=9)
                ax.set_ylabel(f"{comp} OOF R²", fontsize=9)
                ax.set_title(f"{SYS_SL} vs {comp}", fontsize=10, fontweight="bold")
                ax.legend(fontsize=8)
                # Count wins
                sl_wins  = sum(1 for a, b in zip(sl_vals, comp_vals) if a > b)
                comp_wins= sum(1 for a, b in zip(sl_vals, comp_vals) if b > a)
                ax.text(0.02, 0.98, f"{SYS_SL} wins: {sl_wins} | {comp} wins: {comp_wins}",
                        transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        fig.suptitle("Scatter: OOF R² Head-to-Head", fontsize=11, fontweight="bold")
        fig.tight_layout()
        path = output_root / f"comparison_scatter_{run_ts}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved: {path}")

    # ── 4. Win-rate summary bar ──────────────────────────────────────────────
    if len(systems_present) > 1:
        wins: dict[str, int] = {s: 0 for s in systems_present}
        for d in datasets:
            sub = ok[ok["dataset_name"] == d]
            if sub.empty:
                continue
            sub_r2 = sub.dropna(subset=["r2"])
            if sub_r2.empty:
                continue
            best_sys = sub_r2.loc[sub_r2["r2"].idxmax(), "system"]
            if best_sys in wins:
                wins[best_sys] += 1

        fig, ax = plt.subplots(figsize=(5, 3.5))
        sys_names = list(wins.keys())
        win_counts = [wins[s] for s in sys_names]
        colors = [PALETTE.get(s, "#888") for s in sys_names]
        bars = ax.bar(sys_names, win_counts, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, win_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of datasets won (highest R²)", fontsize=9)
        ax.set_title("Win Count by System (best OOF R²)", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(win_counts or [1]) + 1)
        fig.tight_layout()
        path = output_root / f"comparison_wins_{run_ts}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved: {path}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table(summary_df: pd.DataFrame) -> None:
    """Print a rich comparison table: one row per dataset, columns per system."""
    ok = summary_df[summary_df["status"] == "success"].copy()
    if ok.empty:
        print("No successful results to display.")
        return

    datasets     = ok["dataset_name"].unique().tolist()
    systems_have = [s for s in ALL_SYSTEMS if s in ok["system"].unique()]

    rows = []
    for d in datasets:
        sub  = ok[ok["dataset_name"] == d]
        n    = sub["n_samples"].iloc[0] if not sub.empty else None
        p    = sub["n_features"].iloc[0] if not sub.empty else None
        row: dict = {"dataset": d[:20], "n": n, "p": p}
        for sys_name in systems_have:
            s_sub = sub[sub["system"] == sys_name]
            if s_sub.empty or pd.isna(s_sub["r2"].iloc[0]):
                row[f"{sys_name}_R2"]   = "—"
                row[f"{sys_name}_RMSE"] = "—"
                row[f"{sys_name}_sec"]  = "—"
            else:
                row[f"{sys_name}_R2"]   = f"{s_sub['r2'].iloc[0]:.4f}"
                row[f"{sys_name}_RMSE"] = f"{s_sub['rmse'].iloc[0]:.4f}"
                row[f"{sys_name}_sec"]  = f"{s_sub['elapsed_sec'].iloc[0]:.0f}"
        rows.append(row)

    table = pd.DataFrame(rows)
    sep = "=" * max(80, len(table.columns) * 14)
    print(f"\n{sep}")
    print("BENCHMARK COMPARISON TABLE  (OOF metrics)")
    print(sep)
    print(table.to_string(index=False))
    print(sep)

    # Aggregate stats (mean across successful tasks)
    print("\nMean OOF metrics across tasks where all systems succeeded:")
    agg_rows = []
    for sys_name in systems_have:
        sub = ok[ok["system"] == sys_name]
        agg_rows.append({
            "system": sys_name,
            "mean_R2":   f"{sub['r2'].mean():.4f}" if "r2" in sub else "—",
            "mean_RMSE": f"{sub['rmse'].mean():.4f}" if "rmse" in sub else "—",
            "mean_MAE":  f"{sub['mae'].mean():.4f}" if "mae" in sub else "—",
            "mean_time_sec": f"{sub['elapsed_sec'].mean():.0f}" if "elapsed_sec" in sub else "—",
            "n_wins_R2": sum(1 for d in datasets
                              if not ok[(ok["system"]==sys_name) & (ok["dataset_name"]==d)].empty
                              and ok[(ok["system"]==sys_name) & (ok["dataset_name"]==d)]["r2"].notna().any()
                              and ok[(ok["dataset_name"]==d)]["r2"].idxmax() in
                                 ok[(ok["system"]==sys_name) & (ok["dataset_name"]==d)].index),
        })
    print(pd.DataFrame(agg_rows).to_string(index=False))
    print()


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenML regression benchmark: SLAutoML vs FLAML (AutoGluon opt-in via --with-autogluon).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset selection
    p.add_argument("--suite",     type=int, default=269, metavar="ID",
                   help="OpenML benchmark suite ID (default: 269 = OpenML-Reg19)")
    p.add_argument("--task-ids", type=int, nargs="+", default=None, metavar="ID",
                   help="Override: run only these specific OpenML task IDs")
    p.add_argument("--max-tasks", type=int, default=None, metavar="N",
                   help="Limit to the first N tasks (useful for smoke tests)")
    p.add_argument("--skip-tasks", type=int, nargs="+", default=None, metavar="ID",
                   help="Task IDs to skip entirely")

    # Size / memory limits
    p.add_argument("--max-rows", type=int, default=10_000, metavar="N",
                   help="Skip tasks whose dataset has more than N rows (default: 10 000)")
    p.add_argument("--min-rows", type=int, default=50, metavar="N",
                   help="Skip tasks with fewer than N rows (default: 50)")
    p.add_argument("--min-free-gb", type=float, default=2.0, metavar="GB",
                   help="Minimum free RAM in GB before loading a task (default: 2.0)")

    # Benchmark settings
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: fewer CV folds / Optuna trials / models")
    p.add_argument("--time-budget", type=int, default=180, metavar="SEC",
                   help="Wall-clock seconds per AutoML system per task (default: 180)")
    p.add_argument("--output-dir", type=str, default="benchmark_results", metavar="DIR",
                   help="Root output directory (default: benchmark_results)")

    # System toggles
    p.add_argument("--no-slautoml",   action="store_true", help="Skip SLAutoML (script.py)")
    p.add_argument("--with-autogluon", action="store_true", default=False,
                   help="Enable AutoGluon (disabled by default; requires Python ≤ 3.12 "
                        "with autogluon-tabular installed).")
    p.add_argument("--no-flaml",      action="store_true", help="Skip FLAML")

    # AutoGluon settings
    p.add_argument("--ag-presets", type=str, default="medium_quality",
                   help="AutoGluon preset (default: medium_quality). Options: "
                        "medium_quality / high_quality / best_quality")
    p.add_argument("--ag-excluded-model-types", type=str, nargs="*", default=None,
                   metavar="TYPE",
                   help="AutoGluon model types to exclude "
                        "(e.g. --ag-excluded-model-types NN_TORCH).")

    # Output
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating matplotlib comparison plots")

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Print availability summary ────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Three-Way OpenML Regression Benchmark")
    logger.info(f"  Timestamp   : {run_ts}")
    logger.info(f"  Output dir  : {output_root.resolve()}")
    logger.info(f"  Fast mode   : {args.fast}")
    logger.info(f"  Max rows    : {args.max_rows}")
    logger.info(f"  Time budget : {args.time_budget}s per AutoML system")
    logger.info(f"  Min free RAM: {args.min_free_gb} GB")
    logger.info("Systems:")
    logger.info(f"  {SYS_SL}  : {'✓ available' if HAS_SLAUTOML else '✗ NOT installed'}")
    logger.info(f"  {SYS_AG}  : {'✓ available' if HAS_AUTOGLUON else '✗ NOT installed (or not enabled)'}"
                f"  {'← use --with-autogluon to enable' if not args.with_autogluon else ''}")
    logger.info(f"  {SYS_FLAML}       : {'✓ available' if HAS_FLAML else '✗ NOT installed'}")
    if not HAS_PSUTIL:
        logger.warning("  psutil not installed — RAM checks disabled (pip install psutil)")
    logger.info("=" * 72)

    # Warn if nothing to run
    if args.no_slautoml and not args.with_autogluon and args.no_flaml:
        logger.error("All systems disabled via --no-* flags. Nothing to do.")
        sys.exit(1)

    # ── Resolve task IDs ──────────────────────────────────────────────────────
    if args.task_ids:
        task_ids = args.task_ids
        logger.info(f"Using user-supplied task IDs: {task_ids}")
    else:
        task_ids = fetch_regression_tasks(args.suite)

    skip_set = set(args.skip_tasks or [])
    task_ids = [t for t in task_ids if t not in skip_set]

    if args.max_tasks is not None:
        task_ids = task_ids[:args.max_tasks]
        logger.info(f"Limiting to first {args.max_tasks} tasks.")

    logger.info(f"Tasks to benchmark: {len(task_ids)}")

    # ── Run each task ─────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    checkpoint_path = output_root / f"benchmark_summary_{run_ts}.json"

    for i, task_id in enumerate(task_ids, start=1):
        logger.info(f"\n{'─'*70}")
        logger.info(f"[{i}/{len(task_ids)}] Task ID: {task_id}")
        logger.info(f"{'─'*70}")

        rows = run_task(task_id, output_root, args)
        all_rows.extend(rows)

        # Rolling checkpoint after every task
        with open(checkpoint_path, "w") as f:
            json.dump(all_rows, f, indent=2, default=str)

    # ── Build summary DataFrame ───────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)

    # ── Print comparison table ────────────────────────────────────────────────
    print_comparison_table(summary_df)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = output_root / f"benchmark_summary_{run_ts}.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"\nSummary CSV : {csv_path}")
    logger.info(f"Full JSON   : {checkpoint_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("\nGenerating comparison plots...")
        make_plots(summary_df, output_root, run_ts)
    else:
        logger.info("Skipping plots (--no-plots).")

    # ── Final stats ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 72)
    for sys_name in ALL_SYSTEMS:
        sub = summary_df[
            (summary_df["system"] == sys_name) & (summary_df["status"] == "success")
        ]
        if sub.empty:
            continue
        logger.info(
            f"  {sys_name:12s}: {len(sub)} successful tasks | "
            f"mean OOF R²={sub['r2'].mean():.4f} | "
            f"mean RMSE={sub['rmse'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
