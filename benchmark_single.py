#!/usr/bin/env python3
"""
================================================================================
Single Model OpenML Benchmarking Script for SLAutoML
================================================================================
Allows evaluating and tuning a single specific model (e.g., xgboost, ridge, etc.)
from your custom SLAutoML pipeline on OpenML regression tasks.

All configuration options are defined as constants at the top of the script
for easy editing, and are also fully customizable via command-line arguments.
"""

from __future__ import annotations

import argparse
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
    print("ERROR: openml is not installed. Run: pip install openml")
    sys.exit(1)

# ── psutil for RAM check ──────────────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── SLAutoML Pipeline Imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from script import (
        MLConfig,
        FeatureConfig,
        TargetConfig,
        from_dataframe,
    )
    HAS_SLAUTOML = True
except ImportError as e:
    HAS_SLAUTOML = False
    SLAUTOML_ERR = e


# =============================================================================
# DEFAULT CONFIGURATION CONSTANTS
# =============================================================================

# --- OpenML Benchmark Settings ---
DEFAULT_SUITE = 269          # OpenML-Reg19 regression benchmark suite
DEFAULT_TASK_IDS = None      # Optional: list of specific task IDs to run, e.g. [361099, 361273]
DEFAULT_MAX_TASKS = None     # Optional: limit execution to the first N tasks
DEFAULT_SKIP_TASKS = []      # List of task IDs to skip
DEFAULT_MAX_ROWS = 10000     # Skip tasks whose dataset exceeds this row count
DEFAULT_MIN_ROWS = 50        # Skip tasks that are too small for cross-validation
DEFAULT_MIN_FREE_GB = 2.0    # Skip tasks if available system RAM is below this threshold

# --- Model Selection Settings ---
MODEL_NAME = "xgboost"       # The single model to benchmark (e.g., "xgboost", "lightgbm", "catboost", "ridge")

# --- MLConfig Settings ---
RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4
N_OPTUNA_TRIALS = 60
OPTUNA_TIMEOUT_SEC = None
OPTUNA_N_STARTUP_TRIALS = 10
OPTUNA_USE_PRUNING = True
N_JOBS = -1
OUTPUT_DIR = "benchmark_single_results"
SAVE_BEST_MODELS = True
SAVE_OOF_PREDICTIONS = True
SAVE_CONFIG_SNAPSHOT = True

# --- Conformal Prediction Settings ---
COMPUTE_CONFORMAL = True
CONFORMAL_ALPHA = 0.10
CONFORMAL_CALIBRATION_FRACTION = 0.20

# --- SHAP Settings ---
COMPUTE_SHAP = True
SHAP_SAMPLE_SIZE = 200
SHAP_KERNEL_FALLBACK = True

# --- Permutation Importance Settings ---
COMPUTE_PERMUTATION_IMPORTANCE = False
PERMUTATION_N_REPEATS = 5

# --- Feature Stability Settings ---
TRACK_FEATURE_STABILITY = True

# --- Preprocessing & FeatureConfig Settings ---
MAX_CARDINALITY_FOR_CATEGORICAL = 50
MAX_MISSING_FRACTION = 0.90
EXPAND_DATETIME = True
NUMERIC_IMPUTER_STRATEGY = "median"
CATEGORICAL_IMPUTER_STRATEGY = "most_frequent"
NUMERIC_SCALER = "standard"  # options: standard | robust | quantile | none
USE_VARIANCE_THRESHOLD = True
VARIANCE_THRESHOLD = 1e-8
USE_CORRELATION_FILTER = False
CORRELATION_THRESHOLD = 0.98
SELECTKBEST_K = 30
TREE_SELECTOR_THRESHOLD = "median"
TREE_SELECTOR_N_ESTIMATORS = 200
PCA_VARIANCE_RATIO = 0.95
APPLY_CLR_TRANSFORM = False
CLR_COLUMN_PATTERN = ""
CLR_PSEUDOCOUNT = 1e-6

# --- TargetConfig Settings ---
AUTO_TRANSFORM = True
SKEW_THRESHOLD = 1.0
PREFERRED_AUTO_TRANSFORM = "yeo-johnson"  # options: log1p | yeo-johnson | box-cox | quantile-normal


# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SingleModelBenchmark")


# =============================================================================
# OPENML UTILITIES
# =============================================================================

def fetch_regression_tasks(suite_id: int) -> list[int]:
    """Fetch regression task IDs from an OpenML benchmark suite."""
    logger.info(f"Fetching OpenML suite {suite_id}...")
    try:
        suite = openml.study.get_suite(suite_id=suite_id)
        task_ids = list(suite.tasks)
        logger.info(f"Suite {suite_id}: {len(task_ids)} tasks total. Inspecting for regression tasks...")

        regression_ids: list[int] = []
        for tid in task_ids:
            try:
                task = openml.tasks.get_task(tid, download_data=False)
                if task.task_type_id == TaskType.SUPERVISED_REGRESSION:
                    regression_ids.append(tid)
            except Exception as e:
                logger.warning(f"  Could not inspect task {tid}: {e}")

        logger.info(f"Found {len(regression_ids)} regression tasks in suite {suite_id}.")
        return regression_ids
    except Exception as e:
        logger.error(f"Failed to load suite {suite_id}: {e}")
        return []


def load_openml_task(
    task_id: int,
    max_rows: int,
    min_rows: int,
) -> tuple[pd.DataFrame, str, dict] | None:
    """Load and return dataset for an OpenML regression task."""
    logger.info(f"Loading OpenML task {task_id}...")
    try:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        X, y, _, _ = dataset.get_data(
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
            "task_id": task_id,
            "dataset_name": dataset.name,
            "n_samples": n,
            "n_features": p,
            "target_column": target_col,
            "openml_url": f"https://www.openml.org/t/{task_id}",
        }
        logger.info(f"  → Loaded: {dataset.name} | n={n} | p={p} | target='{target_col}'")

        if n < min_rows:
            logger.warning(f"  Skipping task {task_id}: only {n} rows (< {min_rows}).")
            return None
        if n > max_rows:
            logger.warning(f"  Skipping task {task_id}: {n} rows exceeds limit of {max_rows}.")
            return None

        return df, target_col, meta
    except Exception as e:
        logger.error(f"  Failed to load task {task_id}: {e}")
        return None


# =============================================================================
# MAIN BENCHMARK ENGINE
# =============================================================================

def main() -> None:
    if not HAS_SLAUTOML:
        logger.error("Could not import SLAutoML's script.py pipeline!")
        logger.error(f"Reason: {SLAUTOML_ERR}")
        sys.exit(1)

    # ── Command Line Argument Parsing ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Benchmark a single model from SLAutoML on OpenML regression tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # OpenML / Task Settings
    parser.add_argument("--suite", type=int, default=DEFAULT_SUITE, help="OpenML benchmark suite ID")
    parser.add_argument("--task-ids", type=int, nargs="+", default=DEFAULT_TASK_IDS, help="Specific OpenML task IDs to run (overrides --suite)")
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS, help="Limit execution to first N tasks")
    parser.add_argument("--skip-tasks", type=int, nargs="+", default=DEFAULT_SKIP_TASKS, help="Task IDs to skip")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Skip tasks whose dataset exceeds this row count")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS, help="Skip tasks that are too small for cross-validation")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB, help="Skip task if free system RAM falls below this threshold")

    # Model Settings
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model to evaluate (e.g., xgboost, lightgbm, ridge, random_forest)")

    # MLConfig Settings
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-outer-splits", type=int, default=N_OUTER_SPLITS, help="Outer folds (estimates performance)")
    parser.add_argument("--n-inner-splits", type=int, default=N_INNER_SPLITS, help="Inner folds (tunes hyperparameters)")
    parser.add_argument("--n-optuna-trials", type=int, default=N_OPTUNA_TRIALS, help="Optuna HPO trials per model configuration")
    parser.add_argument("--optuna-timeout-sec", type=int, default=OPTUNA_TIMEOUT_SEC, help="Optuna timeout in seconds")
    parser.add_argument("--optuna-n-startup-trials", type=int, default=OPTUNA_N_STARTUP_TRIALS, help="Optuna startup random trials")
    parser.add_argument("--no-optuna-pruning", action="store_false", dest="optuna_use_pruning", help="Disable Optuna early pruning")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="CPU cores to use (-1 for all)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Root folder for model and metric outputs")
    parser.add_argument("--no-save-models", action="store_false", dest="save_best_models", help="Do not save serialized model files")
    parser.add_argument("--no-save-oof", action="store_false", dest="save_oof_predictions", help="Do not save OOF prediction files")
    parser.add_argument("--no-save-config", action="store_false", dest="save_config_snapshot", help="Do not save pipeline configuration snapshot JSON")

    # Conformal Settings
    parser.add_argument("--no-conformal", action="store_false", dest="compute_conformal", help="Disable conformal intervals")
    parser.add_argument("--conformal-alpha", type=float, default=CONFORMAL_ALPHA, help="Nominal coverage error (e.g., 0.10 for 90 percent confidence)")
    parser.add_argument("--conformal-calibration-fraction", type=float, default=CONFORMAL_CALIBRATION_FRACTION)

    # SHAP Settings
    parser.add_argument("--no-shap", action="store_false", dest="compute_shap", help="Disable SHAP importance computation")
    parser.add_argument("--shap-sample-size", type=int, default=SHAP_SAMPLE_SIZE)
    parser.add_argument("--no-shap-kernel-fallback", action="store_false", dest="shap_kernel_fallback", help="Disable slow SHAP KernelExplainer fallback")

    # Permutation Importance Settings
    parser.add_argument("--compute-permutation-importance", action="store_true", default=COMPUTE_PERMUTATION_IMPORTANCE, help="Compute model-agnostic permutation importance")
    parser.add_argument("--permutation-n-repeats", type=int, default=PERMUTATION_N_REPEATS)

    # Feature Stability Settings
    parser.add_argument("--no-feature-stability", action="store_false", dest="track_feature_stability", help="Disable tracking feature selection stability across folds")

    # Preprocessing / FeatureConfig Settings
    parser.add_argument("--max-cardinality-for-categorical", type=int, default=MAX_CARDINALITY_FOR_CATEGORICAL)
    parser.add_argument("--max-missing-fraction", type=float, default=MAX_MISSING_FRACTION)
    parser.add_argument("--no-expand-datetime", action="store_false", dest="expand_datetime", help="Disable datetime feature expansion")
    parser.add_argument("--numeric-imputer-strategy", type=str, default=NUMERIC_IMPUTER_STRATEGY, help="median | mean | most_frequent | constant")
    parser.add_argument("--categorical-imputer-strategy", type=str, default=CATEGORICAL_IMPUTER_STRATEGY, help="most_frequent | constant")
    parser.add_argument("--numeric-scaler", type=str, default=NUMERIC_SCALER, help="standard | robust | quantile | none")
    parser.add_argument("--no-variance-threshold", action="store_false", dest="use_variance_threshold", help="Disable variance filtering")
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)
    parser.add_argument("--use-correlation-filter", action="store_true", default=USE_CORRELATION_FILTER, help="Enable collinear feature dropping")
    parser.add_argument("--correlation-threshold", type=float, default=CORRELATION_THRESHOLD)
    parser.add_argument("--selectkbest-k", type=int, default=SELECTKBEST_K, help="Features to retain in KBest selector")
    parser.add_argument("--tree-selector-threshold", type=str, default=TREE_SELECTOR_THRESHOLD, help="ExtraTrees feature selector threshold")
    parser.add_argument("--tree-selector-n-estimators", type=int, default=TREE_SELECTOR_N_ESTIMATORS)
    parser.add_argument("--pca-variance-ratio", type=float, default=PCA_VARIANCE_RATIO)
    parser.add_argument("--apply-clr-transform", action="store_true", default=APPLY_CLR_TRANSFORM, help="Enable Centered Log-Ratio transform")
    parser.add_argument("--clr-column-pattern", type=str, default=CLR_COLUMN_PATTERN)
    parser.add_argument("--clr-pseudocount", type=float, default=CLR_PSEUDOCOUNT)

    # TargetConfig Settings
    parser.add_argument("--no-auto-transform", action="store_false", dest="auto_transform", help="Disable auto target skew transform")
    parser.add_argument("--skew-threshold", type=float, default=SKEW_THRESHOLD)
    parser.add_argument("--preferred-auto-transform", type=str, default=PREFERRED_AUTO_TRANSFORM, help="log1p | yeo-johnson | box-cox | quantile-normal")

    args = parser.parse_args()

    # ── Resolve Tasks to Run ──────────────────────────────────────────────────
    if args.task_ids:
        task_list = list(args.task_ids)
        logger.info(f"Using explicitly provided Task IDs: {task_list}")
    else:
        task_list = fetch_regression_tasks(args.suite)
        if not task_list:
            logger.error("No regression tasks found. Exiting.")
            sys.exit(1)

    # Remove skipped tasks
    if args.skip_tasks:
        task_list = [tid for tid in task_list if tid not in args.skip_tasks]
        logger.info(f"Skip-list applied. Tasks remaining: {task_list}")

    # Limit to max-tasks
    if args.max_tasks and len(task_list) > args.max_tasks:
        task_list = task_list[:args.max_tasks]
        logger.info(f"Limited execution to first {args.max_tasks} tasks: {task_list}")

    # ── Map Arguments to Config Objects ───────────────────────────────────────
    # Initialize output directory with timestamp for this execution run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_root = Path(args.output_dir) / f"run_{timestamp}_{args.model}"
    run_output_root.mkdir(parents=True, exist_ok=True)

    ml_cfg = MLConfig(
        random_state=args.random_state,
        n_outer_splits=args.n_outer_splits,
        n_inner_splits=args.n_inner_splits,
        n_optuna_trials=args.n_optuna_trials,
        optuna_timeout_sec=args.optuna_timeout_sec,
        optuna_n_startup_trials=args.optuna_n_startup_trials,
        optuna_use_pruning=args.optuna_use_pruning,
        n_jobs=args.n_jobs,
        output_dir=str(run_output_root),
        save_best_models=args.save_best_models,
        save_oof_predictions=args.save_oof_predictions,
        save_config_snapshot=args.save_config_snapshot,
        compute_conformal=args.compute_conformal,
        conformal_alpha=args.conformal_alpha,
        conformal_calibration_fraction=args.conformal_calibration_fraction,
        compute_shap=args.compute_shap,
        shap_sample_size=args.shap_sample_size,
        shap_kernel_fallback=args.shap_kernel_fallback,
        compute_permutation_importance=args.compute_permutation_importance,
        permutation_n_repeats=args.permutation_n_repeats,
        track_feature_stability=args.track_feature_stability,
        include_models=[args.model],  # RESTRICT TO THE SINGLE MODEL TO BENCHMARK!
    )

    feat_cfg = FeatureConfig(
        max_cardinality_for_categorical=args.max_cardinality_for_categorical,
        max_missing_fraction=args.max_missing_fraction,
        expand_datetime=args.expand_datetime,
        numeric_imputer_strategy=args.numeric_imputer_strategy,
        categorical_imputer_strategy=args.categorical_imputer_strategy,
        numeric_scaler=args.numeric_scaler,
        use_variance_threshold=args.use_variance_threshold,
        variance_threshold=args.variance_threshold,
        use_correlation_filter=args.use_correlation_filter,
        correlation_threshold=args.correlation_threshold,
        selectkbest_k=args.selectkbest_k,
        tree_selector_threshold=args.tree_selector_threshold,
        tree_selector_n_estimators=args.tree_selector_n_estimators,
        pca_variance_ratio=args.pca_variance_ratio,
        apply_clr_transform=args.apply_clr_transform,
        clr_column_pattern=args.clr_column_pattern,
        clr_pseudocount=args.clr_pseudocount,
    )

    # ── Execute Benchmark Across Tasks ────────────────────────────────────────
    results_summary: List[Dict[str, Any]] = []

    logger.info("=" * 80)
    logger.info(f"Starting Single Model Benchmark: {args.model.upper()}")
    logger.info(f"Targeting: {len(task_list)} task(s)")
    logger.info(f"Results will be stored in: {run_output_root.resolve()}")
    logger.info("=" * 80)

    for i, task_id in enumerate(task_list, start=1):
        logger.info("-" * 80)
        logger.info(f"Task {i}/{len(task_list)}: ID {task_id}")
        logger.info("-" * 80)

        # 1. Available Memory Check
        if HAS_PSUTIL:
            free_gb = psutil.virtual_memory().available / (1024 ** 3)
            logger.info(f"  System RAM headroom: {free_gb:.2f} GB (Required: {args.min_free_gb:.2f} GB)")
            if free_gb < args.min_free_gb:
                logger.warning(f"  Skipping task {task_id}: insufficient free memory.")
                results_summary.append({
                    "task_id": task_id, "dataset_name": "?", "status": "skipped_memory",
                    "r2": np.nan, "rmse": np.nan, "mae": np.nan, "elapsed_sec": 0
                })
                continue

        # 2. Fetch and Load Task
        loaded = load_openml_task(task_id, max_rows=args.max_rows, min_rows=args.min_rows)
        if not loaded:
            results_summary.append({
                "task_id": task_id, "dataset_name": "?", "status": "skipped_size_or_error",
                "r2": np.nan, "rmse": np.nan, "mae": np.nan, "elapsed_sec": 0
            })
            continue

        df, target_col, meta = loaded
        dataset_name = meta["dataset_name"]

        # Override Target Config specifically for this task's target column name
        tgt_cfg = TargetConfig(
            target_columns=[target_col],
            auto_transform=args.auto_transform,
            skew_threshold=args.skew_threshold,
            preferred_auto_transform=args.preferred_auto_transform,
        )

        # Override output_dir per task to keep artifact folders clean and isolated
        task_output_dir = run_output_root / f"task_{task_id}_{dataset_name}"
        ml_cfg.output_dir = str(task_output_dir)

        start_time = time.perf_counter()
        try:
            # 3. Instantiate and run pipeline on the dataset
            pipeline = from_dataframe(
                df=df,
                target_columns=target_col,
                ml_config=ml_cfg,
                feat_config=feat_cfg,
                tgt_config=tgt_cfg,
            )
            rm = pipeline.run()
            elapsed = time.perf_counter() - start_time

            # 4. Extract metrics
            best_df = rm.get_best_per_label()
            if not best_df.empty:
                best = best_df.iloc[0]
                r2 = float(best.get("oof_r2", np.nan))
                rmse = float(best.get("oof_rmse", np.nan))
                mae = float(best.get("oof_mae", np.nan))
                best_strategy = best.get("feature_strategy", "default")
                logger.info(f"  ✓ SUCCESS: {args.model} completed in {elapsed:.1f}s | OOF R²={r2:.4f}")
                results_summary.append({
                    "task_id": task_id, "dataset_name": dataset_name, "status": "success",
                    "r2": r2, "rmse": rmse, "mae": mae, "best_strategy": best_strategy,
                    "elapsed_sec": round(elapsed, 1)
                })
            else:
                logger.warning("  ✗ FAILURE: Pipeline finished but no best model was recorded.")
                results_summary.append({
                    "task_id": task_id, "dataset_name": dataset_name, "status": "no_model",
                    "r2": np.nan, "rmse": np.nan, "mae": np.nan, "elapsed_sec": round(elapsed, 1)
                })

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(f"  ✗ FAILED to run task {task_id}: {exc}")
            traceback.print_exc()
            results_summary.append({
                "task_id": task_id, "dataset_name": dataset_name, "status": "error",
                "r2": np.nan, "rmse": np.nan, "mae": np.nan, "elapsed_sec": round(elapsed, 1),
                "error": str(exc)
            })

    # ── Report Overall Summary ────────────────────────────────────────────────
    summary_df = pd.DataFrame(results_summary)
    summary_csv_path = run_output_root / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Results and summary saved to: {summary_csv_path.resolve()}\n")

    # Format beautiful markdown-like CLI output table
    print(summary_df.to_string(
        index=False,
        columns=["task_id", "dataset_name", "status", "r2", "rmse", "mae", "elapsed_sec"]
    ))
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
