#!/usr/bin/env python3
"""
================================================================================
AutoML Benchmark Harness — v2.0
================================================================================

A production-grade, type-hinted wrapper around AutoGluon and FLAML for rapid
benchmarking against a custom supervised tabular-learning pipeline.

DESIGN GOALS
    · Unified CLI / dataclass config; backend selectable at runtime (--backend).
    · Factory pattern: AutoMLBackend ABC with AutoGluonBackend / FLAMLBackend,
      registered in a plain dict for zero-friction extensibility.
    · Source-agnostic ingestion (CSV, TSV, Parquet, Feather) via pandas before
      passing to backend-specific structures (TabularDataset for AutoGluon).
    · External K-fold OOF evaluation so both backends run under the SAME CV
      protocol, making oof_metrics directly comparable across libraries.
    · Native time budgeting:
          time_limit  → AutoGluon  TabularPredictor.fit()
          time_budget → FLAML      AutoML.fit()
    · Standardised outputs regardless of backend:
          metrics.json          R², RMSE, MAE, MedAE, MAPE (regression) /
                                accuracy, balanced_accuracy, F1, precision,
                                recall, log-loss, ROC-AUC (classification)
          oof_predictions.csv   row-aligned OOF predictions + fold IDs
          config_snapshot.json  full BenchmarkConfig for reproducibility
          load_model.py         backend-specific model loading code snippet
          artifacts/final/      AutoGluon predictor dir or flaml_model.pkl
    · Graceful degradation: optional imports wrapped in try/except ImportError;
      calling an uninstalled backend raises ImportError with install guidance.
    · --demo flag: smoke test on synthetic data (no real dataset required).

IMPORTANT CAVEATS
    OOF TIME DILUTION
        With n_splits=5 and oof_budget_fraction=0.60, each fold receives only
        ~12% (0.60/5) of total_budget. Per-fold models may be substantially
        weaker than the final model; treat fold-level metrics as lower bounds
        on the final model's true capability, not as its representative score.

    METRIC COMPARABILITY
        AutoGluon's internal leaderboard val_score and FLAML's best_loss are
        qualitatively different quantities — different data splits, different
        aggregation logic. This harness reports them only as supplementary
        metadata under "backend_native_info". The primary "oof_metrics" and
        "holdout_metrics" keys use the same external CV protocol for every
        backend and are the fields you should use for cross-backend comparison.

    DEPENDENCY CONFLICTS
        AutoGluon and FLAML may conflict on shared transitive dependencies.
        Consider isolated virtual environments (one per backend) for stability.

    macOS APPLE SILICON (M1 / M2 / M3)
        AutoGluon's NN_TORCH models need Metal Performance Shaders or a CPU
        fallback. Pass --ag_excluded_model_types NN_TORCH to skip them safely.
        FLAML's LightGBM-based estimators work natively on arm64.

    DISK SPACE
        AutoGluon best_quality with multi-layer stacking can write multiple
        gigabytes per run. Point --output_dir at a volume with ≥ 10 GB free.
        No run-resumption logic is implemented; interrupted runs restart from
        scratch.

INSTALLATION
    pip install pandas numpy scikit-learn pyarrow fastparquet   # always
    pip install "autogluon.tabular[all]"                        # AutoGluon
    pip install "flaml[automl]"                                 # FLAML

COMMAND-LINE EXAMPLES
    # Regression — AutoGluon, macOS-safe (excludes NN_TORCH):
    python automl_benchmark.py \\
        --backend autogluon --data_path housing.csv --target price \\
        --task regression --time_budget 300 \\
        --ag_excluded_model_types NN_TORCH --output_dir runs/ag_housing

    # Regression — FLAML, 3-minute budget:
    python automl_benchmark.py \\
        --backend flaml --data_path housing.parquet --target price \\
        --task regression --time_budget 180 --output_dir runs/flaml_housing

    # Classification — FLAML with a specific estimator list:
    python automl_benchmark.py \\
        --backend flaml --data_path titanic.csv --target Survived \\
        --task classification --time_budget 120 \\
        --flaml_estimators lgbm xgboost rf --output_dir runs/flaml_titanic

    # Drop leakage columns; task auto-inferred from target:
    python automl_benchmark.py \\
        --backend autogluon --data_path data.csv --target target \\
        --drop_columns id timestamp leaked_future --time_budget 300

    # Built-in smoke test (no dataset needed):
    python automl_benchmark.py --demo --backend autogluon
    python automl_benchmark.py --demo --backend flaml

PYTHON API
    from automl_benchmark import run_automl_benchmark

    metrics = run_automl_benchmark(
        backend="flaml",
        data_path="housing.csv",
        target="price",
        task="regression",
        time_budget=300,
        output_dir="runs/flaml_api",
    )
    print(metrics["oof_metrics"])       # {'r2': 0.82, 'rmse': 0.49, ...}
    print(metrics["holdout_metrics"])   # same schema on the stratified holdout
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import shutil
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# =============================================================================
# Optional backend imports — graceful degradation
# =============================================================================

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    HAS_AUTOGLUON = True
    _AG_IMPORT_ERR: Optional[ImportError] = None
except ImportError as _e:
    HAS_AUTOGLUON = False
    TabularDataset = None   # type: ignore[assignment]
    TabularPredictor = None  # type: ignore[assignment]
    _AG_IMPORT_ERR = _e

try:
    from flaml import AutoML as _FLAMLAutoML
    HAS_FLAML = True
    _FLAML_IMPORT_ERR: Optional[ImportError] = None
except ImportError as _e:
    HAS_FLAML = False
    _FLAMLAutoML = None   # type: ignore[assignment]
    _FLAML_IMPORT_ERR = _e

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AutoMLBenchmark")
warnings.filterwarnings("ignore")

__version__ = "2.0.0"
__all__ = [
    "BenchmarkConfig",
    "AutoMLBackend",
    "AutoGluonBackend",
    "FLAMLBackend",
    "BackendFactory",
    "BenchmarkRunner",
    "run_automl_benchmark",
]

# Minimum per-fold training budget to avoid truly degenerate models.
_MIN_FOLD_BUDGET_SEC: int = 15


# =============================================================================
# §0  CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """
    Unified runtime configuration.  Every field maps 1-to-1 with a CLI
    argument, making experiments fully reproducible from code or the terminal.

    Time-budget allocation (OOF + final fit)
    -----------------------------------------
    oof_total   = time_budget × oof_budget_fraction
    final_total = time_budget × (1 − oof_budget_fraction)
    per_fold    = max(_MIN_FOLD_BUDGET_SEC, oof_total / n_splits)

    The minimum per-fold floor ensures each fold trains at least a minimal
    model; it may cause total elapsed time to slightly exceed the budget on
    very small budgets (e.g. ≤ 30 s with 5 folds).
    """

    # ── Core ─────────────────────────────────────────────────────────────────
    backend:     str = "autogluon"  # "autogluon" | "flaml"
    data_path:   str = ""
    target:      str = ""
    task:        str = "auto"       # auto | regression | classification | binary | multiclass
    time_budget: int = 300          # total wall-clock training budget in seconds
    output_dir:  str = "automl_benchmark"

    # ── Data loading ──────────────────────────────────────────────────────────
    file_format:  Optional[str] = None  # auto-detected from extension; override if needed
    csv_sep:      str = ","
    csv_encoding: str = "utf-8"
    drop_columns: List[str] = field(default_factory=list)

    # ── Evaluation / split ────────────────────────────────────────────────────
    test_size:           float = 0.20
    random_state:        int   = 42
    n_splits:            int   = 5     # CV folds for external OOF evaluation
    oof_budget_fraction: float = 0.60  # share of time_budget for OOF CV folds

    # ── Task inference heuristics ─────────────────────────────────────────────
    max_classes_for_classification: int = 20

    # ── Output / operational ──────────────────────────────────────────────────
    keep_fold_models: bool = False  # retain per-fold artifacts (large disk use)
    overwrite:        bool = False  # delete and recreate output_dir

    # ── AutoGluon-specific ────────────────────────────────────────────────────
    ag_presets:              str            = "best_quality"
    ag_excluded_model_types: List[str]      = field(default_factory=list)
    ag_num_bag_folds:        Optional[int]  = None  # None = preset default
    ag_num_stack_levels:     Optional[int]  = None  # None = preset default
    ag_verbosity:            int            = 2

    # ── FLAML-specific ────────────────────────────────────────────────────────
    flaml_estimator_list: Optional[List[str]] = None  # None = FLAML auto-search
    flaml_n_jobs:         int                 = -1
    flaml_log_file:       str                 = ""    # "" = disabled

    # ── Backend-native optimization metric ────────────────────────────────────
    eval_metric: Optional[str] = None  # None = sensible per-task default

    def validate(self) -> None:
        """Run early validation with actionable error messages."""
        self.backend = self.backend.lower().strip()
        self.task    = self.task.lower().strip()

        if self.backend not in {"autogluon", "flaml"}:
            raise ValueError(
                f"--backend must be 'autogluon' or 'flaml', got '{self.backend}'."
            )
        valid_tasks = {"auto", "regression", "classification", "binary", "multiclass"}
        if self.task not in valid_tasks:
            raise ValueError(f"--task must be one of {sorted(valid_tasks)}.")
        if self.time_budget <= 0:
            raise ValueError("--time_budget must be a positive integer (seconds).")
        if self.n_splits < 2:
            raise ValueError("--n_splits must be >= 2 for OOF evaluation.")
        if not (0.05 <= self.oof_budget_fraction <= 0.95):
            raise ValueError("--oof_budget_fraction must be between 0.05 and 0.95.")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("--test_size must be strictly between 0.0 and 1.0.")
        if self.data_path:
            p = Path(self.data_path)
            if not p.exists():
                raise FileNotFoundError(f"Data file not found: {p.resolve()}")


# =============================================================================
# §1  UTILITIES
# =============================================================================

def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert scientific-Python objects to JSON-safe values.

    Handles Path, str/int/bool, non-finite floats (→ None), numpy scalars,
    numpy arrays, pandas Series / DataFrames, dicts, and generic iterables.
    Prevents the silent serialization failures common when writing AutoML
    metadata (which often contains numpy scalars, Paths, or ±inf).
    """
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def sanitize_name(name: Any) -> str:
    """Return a filesystem- and column-safe identifier string."""
    text = re.sub(r"[^\w.\-]+", "_", str(name)).strip("_")
    return text or "value"


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_empty_dir(path: Path) -> None:
    """Create a directory, deleting any existing content first."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    """Write *payload* as JSON with to_jsonable() applied for safe serialization."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, sort_keys=False)
    logger.debug("JSON written: %s", path)


def normalize_cli_list(values: Optional[Sequence[str]]) -> List[str]:
    """
    Normalize argparse ``nargs='*'`` inputs, supporting both:
        --flag a b c      →  ['a', 'b', 'c']
        --flag a,b,c      →  ['a', 'b', 'c']
    """
    if not values:
        return []
    out: List[str] = []
    for val in values:
        out.extend(x.strip() for x in val.split(",") if x.strip())
    return out


# =============================================================================
# §2  DATA LOADING AND TASK INFERENCE
# =============================================================================

_FORMAT_LOADERS: Dict[str, Any] = {
    "csv":     lambda p, cfg: pd.read_csv(p, sep=cfg.csv_sep, encoding=cfg.csv_encoding),
    "tsv":     lambda p, cfg: pd.read_csv(p, sep="\t",        encoding=cfg.csv_encoding),
    "parquet": lambda p, _:   pd.read_parquet(p),
    "pq":      lambda p, _:   pd.read_parquet(p),
    "feather": lambda p, _:   pd.read_feather(p),
}


def load_data(config: BenchmarkConfig) -> pd.DataFrame:
    """
    Load a tabular data file and return a pandas DataFrame.

    File format is inferred from the extension; override via
    ``BenchmarkConfig.file_format``.  Supported: CSV, TSV, Parquet, Feather.
    """
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    fmt = (config.file_format or path.suffix.lstrip(".")).lower()
    if fmt not in _FORMAT_LOADERS:
        raise ValueError(
            f"Unsupported format '.{fmt}'. Supported: {sorted(_FORMAT_LOADERS)}"
        )

    logger.info("Loading %s: %s", fmt.upper(), path)
    df: pd.DataFrame = _FORMAT_LOADERS[fmt](path, config)
    df.columns = [str(c) for c in df.columns]

    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not supported: {dups}")

    logger.info("  → %d rows × %d columns", len(df), df.shape[1])
    return df


def _is_integer_like(y: pd.Series) -> bool:
    """Return True when all finite numeric values in *y* are integer-valued."""
    arr = pd.to_numeric(y.dropna(), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr.size > 0 and bool(np.all(np.isclose(arr, np.round(arr))))


def infer_task_type(y: pd.Series, requested: str, max_classes: int) -> str:
    """
    Infer the canonical task type from the target series.

    Returns one of: 'regression', 'binary', 'multiclass'.

    When ``requested='auto'`` the heuristics are:
      - Non-numeric or boolean dtype            → classification
      - Numeric, integer-like, low-cardinality  → classification
      - Otherwise                               → regression
    """
    requested = requested.lower().strip()
    n_unique = int(y.dropna().nunique())
    if n_unique < 2:
        raise ValueError("Target must have at least two distinct non-null values.")

    if requested == "regression":
        return "regression"
    if requested == "binary":
        if n_unique != 2:
            raise ValueError(
                f"--task binary requested but target has {n_unique} unique values."
            )
        return "binary"
    if requested == "multiclass":
        return "multiclass"
    if requested == "classification":
        return "binary" if n_unique == 2 else "multiclass"

    # requested == "auto"
    if pd.api.types.is_bool_dtype(y):
        return "binary"
    if pd.api.types.is_numeric_dtype(y):
        if n_unique <= max_classes and _is_integer_like(y):
            return "binary" if n_unique == 2 else "multiclass"
        return "regression"
    # Object / string: probe for numeric-string targets
    y_numeric = pd.to_numeric(y.dropna(), errors="coerce")
    if y_numeric.notna().all():
        if n_unique <= max_classes and _is_integer_like(y_numeric):
            return "binary" if n_unique == 2 else "multiclass"
        return "regression"
    return "binary" if n_unique == 2 else "multiclass"


def prepare_xy(
    df: pd.DataFrame,
    config: BenchmarkConfig,
) -> Tuple[pd.DataFrame, pd.Series, str, Optional[List[Any]]]:
    """
    Validate, clean, and split the DataFrame into (X, y, task_type, classes).

    Drops rows with missing targets, resolves task type, applies
    ``drop_columns``, and returns the ordered class list for classification.
    """
    if config.target not in df.columns:
        raise ValueError(
            f"Target '{config.target}' not found. "
            f"Available columns: {list(df.columns[:30])}"
        )

    n_before = len(df)
    df = df.dropna(subset=[config.target]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with missing target.", n_dropped)

    y_raw = df[config.target].copy()
    task_type = infer_task_type(
        y=y_raw,
        requested=config.task,
        max_classes=config.max_classes_for_classification,
    )
    logger.info("Task type: %s", task_type)

    if task_type == "regression":
        y: pd.Series = pd.to_numeric(y_raw, errors="coerce")
        bad = int(y.isna().sum())
        if bad:
            raise ValueError(
                f"Regression target contains {bad} non-numeric values after coercion."
            )
        y = y.astype(float)
    else:
        y = y_raw

    drop_set = {config.target} | set(config.drop_columns)
    missing = [c for c in config.drop_columns if c not in df.columns]
    if missing:
        logger.warning("Requested drop columns not found (ignored): %s", missing)

    feature_cols = [c for c in df.columns if c not in drop_set]
    if not feature_cols:
        raise ValueError("No feature columns remain after dropping target / drop_columns.")

    X = df[feature_cols].reset_index(drop=True)
    y = y.reset_index(drop=True)

    classes: Optional[List[Any]] = None
    if task_type != "regression":
        classes = sorted(y.dropna().unique().tolist(), key=lambda v: str(v))

    logger.info(
        "Dataset ready: %d rows × %d features  target='%s'",
        len(X), X.shape[1], config.target,
    )
    if classes:
        logger.info("Classes: %s", classes)

    return X, y, task_type, classes


# =============================================================================
# §3  PROBABILITY ALIGNMENT
# =============================================================================

def align_proba_to_classes(
    proba: Any,
    target_classes: Sequence[Any],
    backend_classes: Optional[Sequence[Any]] = None,
) -> Optional[np.ndarray]:
    """
    Align backend probability outputs to a canonical global class ordering.

    Handles four forms of backend output:
      1. pandas DataFrame with class-label columns → re-order to target_classes
      2. 1-D array (binary positive-class proba)  → expand to (n, 2)
      3. 2-D array with backend_classes provided  → re-order columns to match
      4. 2-D array without backend_classes        → assumed already aligned

    Returns an (n, len(target_classes)) float64 array or None when
    the alignment cannot be determined (e.g. shape mismatch, None input).
    """
    if proba is None:
        return None

    n_classes = len(target_classes)

    # ── Case 1: DataFrame with class-label columns ────────────────────────────
    if isinstance(proba, pd.DataFrame):
        out = np.full((len(proba), n_classes), np.nan, dtype=float)
        col_lookup = {str(c): c for c in proba.columns}
        for j, cls in enumerate(target_classes):
            col = col_lookup.get(str(cls))
            if col is not None:
                out[:, j] = pd.to_numeric(proba[col], errors="coerce").to_numpy(float)
        return out

    arr = np.asarray(proba, dtype=float)

    # ── Case 2: 1-D binary positive-class probabilities ──────────────────────
    if arr.ndim == 1 and n_classes == 2:
        return np.column_stack([1.0 - arr, arr])

    if arr.ndim != 2:
        return None

    # ── Case 3: 2-D with known backend class order ────────────────────────────
    if backend_classes is not None and len(backend_classes) == arr.shape[1]:
        backend_map = {str(cls): i for i, cls in enumerate(backend_classes)}
        out = np.full((arr.shape[0], n_classes), np.nan, dtype=float)
        for j, cls in enumerate(target_classes):
            src = backend_map.get(str(cls))
            if src is not None:
                out[:, j] = arr[:, src]
        return out

    # ── Case 4: assume already aligned ────────────────────────────────────────
    if arr.shape[1] == n_classes:
        return arr

    return None  # shape mismatch; alignment indeterminate


# =============================================================================
# §4  METRICS
# =============================================================================

def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Standard regression metric suite, NaN-safe."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    n_valid = int(mask.sum())

    if n_valid < 2:
        return {k: float("nan") for k in ("n", "r2", "mae", "medae", "mse", "rmse", "mape")}

    yt_m, yp_m = yt[mask], yp[mask]
    mse = float(mean_squared_error(yt_m, yp_m))
    metrics: Dict[str, float] = {
        "n":     float(n_valid),
        "r2":    float(r2_score(yt_m, yp_m)),
        "mae":   float(mean_absolute_error(yt_m, yp_m)),
        "medae": float(median_absolute_error(yt_m, yp_m)),
        "mse":   mse,
        "rmse":  float(np.sqrt(mse)),
    }
    try:
        metrics["mape"] = float(mean_absolute_percentage_error(yt_m, yp_m))
    except Exception:
        metrics["mape"] = float("nan")
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    classes: Sequence[Any],
) -> Dict[str, Any]:
    """Standard classification metric suite with per-class probability support."""
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    mask = pd.notna(yt) & pd.notna(yp)
    yt_m, yp_m = yt[mask], yp[mask]

    if len(yt_m) == 0:
        return {"n": 0, "accuracy": float("nan"), "f1_macro": float("nan")}

    metrics: Dict[str, Any] = {
        "n":                 int(len(yt_m)),
        "accuracy":          float(accuracy_score(yt_m, yp_m)),
        "balanced_accuracy": float(balanced_accuracy_score(yt_m, yp_m)),
        "f1_macro":          float(f1_score(yt_m, yp_m, average="macro",    zero_division=0)),
        "f1_weighted":       float(f1_score(yt_m, yp_m, average="weighted", zero_division=0)),
        "precision_macro":   float(precision_score(yt_m, yp_m, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(yt_m, yp_m, average="macro",    zero_division=0)),
    }

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        proba_m = proba[mask]
        if (
            proba_m.ndim == 2
            and proba_m.shape[1] == len(classes)
            and np.isfinite(proba_m).all()
        ):
            try:
                metrics["log_loss"] = float(
                    log_loss(yt_m, proba_m, labels=list(classes))
                )
            except Exception as exc:
                metrics["log_loss_error"] = str(exc)

            try:
                if len(classes) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(yt_m, proba_m[:, 1]))
                else:
                    metrics["roc_auc_ovr_macro"] = float(
                        roc_auc_score(
                            yt_m, proba_m, labels=list(classes),
                            multi_class="ovr", average="macro",
                        )
                    )
            except Exception as exc:
                metrics["roc_auc_error"] = str(exc)

    return metrics


def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Dispatch to the appropriate metric function based on *task_type*."""
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    return compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        classes=list(classes or []),
    )


# =============================================================================
# §5  TIME-BUDGET PLANNING
# =============================================================================

def plan_time_budgets(config: BenchmarkConfig, n_splits: int) -> Tuple[int, int]:
    """
    Allocate the total ``time_budget`` between OOF folds and the final fit.

    Returns ``(fold_budget_sec, final_budget_sec)``.
    The per-fold budget is floored at ``_MIN_FOLD_BUDGET_SEC`` with a warning.
    """
    oof_total   = max(1, int(config.time_budget * config.oof_budget_fraction))
    final_total = max(1, int(config.time_budget * (1.0 - config.oof_budget_fraction)))
    fold_raw    = oof_total // max(1, n_splits)

    if fold_raw < _MIN_FOLD_BUDGET_SEC:
        logger.warning(
            "Computed per-fold budget (%ds) is below the minimum (%ds).  "
            "Flooring to minimum.  Consider increasing --time_budget or "
            "reducing --n_splits.",
            fold_raw, _MIN_FOLD_BUDGET_SEC,
        )
        fold_raw = _MIN_FOLD_BUDGET_SEC

    return fold_raw, final_total


def make_cv_splitter(
    y: pd.Series,
    task_type: str,
    n_splits: int,
    random_state: int,
) -> Tuple[Any, int, bool]:
    """
    Build a CV splitter appropriate for the task.

    Returns ``(splitter, effective_n_splits, is_stratified)``.
    Falls back from StratifiedKFold to KFold when any class has < 2 samples.
    """
    n_samples = len(y)
    effective  = min(n_splits, n_samples)

    if task_type != "regression":
        min_class = int(y.value_counts(dropna=False).min())
        if min_class >= 2:
            effective = min(effective, min_class)
            return (
                StratifiedKFold(n_splits=effective, shuffle=True, random_state=random_state),
                effective,
                True,
            )
        logger.warning(
            "At least one class has < 2 samples; falling back to non-stratified KFold."
        )

    return (
        KFold(n_splits=effective, shuffle=True, random_state=random_state),
        effective,
        False,
    )


# =============================================================================
# §6  ABSTRACT BACKEND (FACTORY PATTERN)
# =============================================================================

class AutoMLBackend(ABC):
    """
    Abstract base class for AutoML backends.

    Concrete subclasses encapsulate one library's API so that the
    BenchmarkRunner orchestrator remains backend-agnostic.

    Interface contract
    ------------------
    train(X_train, y_train, time_budget)
        Fit under a native wall-clock budget.  Concrete implementations pass
        ''time_budget'' directly to the library's fit parameter.

    predict(X)
        Return a 1-D array of predictions (labels for classification).

    predict_proba(X)
        Return (n, n_classes) probabilities or None (regression / unsupported).

    classes_  [property]
        Class ordering for probability matrices; None for regression.

    save_artifact()
        Persist the best model to self.artifact_dir; return artifact path.

    best_model_info  [property]
        Backend-specific metadata (leaderboard, best config, etc.).

    evaluate(X_test, y_test, target_classes)  [concrete — do NOT override]
        Calls predict / predict_proba, aligns probabilities, dispatches
        compute_metrics.  Returns a standardised metric dict.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        task_type: str,
        artifact_dir: Path,
    ) -> None:
        self.config       = config
        self.task_type    = task_type
        self.artifact_dir = artifact_dir
        self._is_trained  = False
        self._train_time  = 0.0

    @abstractmethod
    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, time_budget: int
    ) -> None:
        """Fit the AutoML model with a native time budget (seconds)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1-D predictions; X must not contain the target column."""

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        """
        Return class probabilities (n, n_classes) or None.
        Default implementation returns None (correct for regression).
        Override in classification backends.
        """
        return None

    @property
    def classes_(self) -> Optional[List[Any]]:
        """Class ordering used in predict_proba columns; None for regression."""
        return None

    @abstractmethod
    def save_artifact(self) -> Path:
        """Persist best model to self.artifact_dir. Return the artifact path."""

    @property
    def best_model_info(self) -> Dict[str, Any]:
        """Backend-specific model selection metadata for inclusion in metrics.json."""
        return {}

    # ── Shared concrete evaluation — subclasses must NOT override ────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_classes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute standardised holdout metrics.

        Calls predict() and predict_proba(), aligns class probabilities via
        align_proba_to_classes(), then dispatches to compute_metrics().
        This method is the single source of metric computation truth across
        all backends.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"{type(self).__name__} is not trained. Call .train() first."
            )
        y_pred = self.predict(X_test)
        y_proba: Optional[np.ndarray] = None

        if self.task_type != "regression" and target_classes is not None:
            raw = self.predict_proba(X_test)
            if raw is not None:
                y_proba = align_proba_to_classes(
                    proba=raw,
                    target_classes=target_classes,
                    backend_classes=self.classes_,
                )

        return compute_metrics(
            task_type=self.task_type,
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            y_proba=y_proba,
            classes=target_classes,
        )

    @property
    def train_time(self) -> float:
        """Actual wall-clock training time in seconds."""
        return self._train_time


# =============================================================================
# §7  AUTOGLUON BACKEND
# =============================================================================

class AutoGluonBackend(AutoMLBackend):
    """
    AutoGluon TabularPredictor backend.

    Implementation notes
    --------------------
    · time_limit is enforced natively by AutoGluon's internal scheduler.
    · presets='best_quality' (default) enables bagging + multi-layer stacking.
    · Models are auto-persisted to artifact_dir during fit; save_artifact()
      simply records metadata and returns that directory.
    · predict_proba() requests a pandas DataFrame (as_pandas=True) which
      align_proba_to_classes() handles via Case 1 (class-label columns).
    · On macOS M1/M2/M3, pass ag_excluded_model_types=['NN_TORCH'] to skip
      PyTorch-dependent models that require MPS configuration.
    """

    # Generic eval_metric names → AutoGluon metric identifiers
    _AG_METRIC_MAP: Dict[str, str] = {
        "r2":       "r2",
        "rmse":     "root_mean_squared_error",
        "mse":      "mean_squared_error",
        "mae":      "mean_absolute_error",
        "medae":    "median_absolute_error",
        "mape":     "mean_absolute_percentage_error",
        "accuracy": "accuracy",
        "f1":       "f1_macro",
        "f1_macro": "f1_macro",
        "roc_auc":  "roc_auc",
        "log_loss": "log_loss",
    }
    _DEFAULT_AG_METRIC: Dict[str, str] = {
        "regression": "root_mean_squared_error",
        "binary":     "roc_auc",
        "multiclass": "accuracy",
    }

    def __init__(
        self, config: BenchmarkConfig, task_type: str, artifact_dir: Path
    ) -> None:
        if not HAS_AUTOGLUON:
            raise ImportError(
                "AutoGluon is not installed.\n"
                "  pip install 'autogluon.tabular[all]'\n"
                f"  Original error: {_AG_IMPORT_ERR}"
            )
        super().__init__(config, task_type, artifact_dir)
        self._predictor: Any = None

    def _ag_eval_metric(self) -> str:
        if self.config.eval_metric:
            key = self.config.eval_metric.lower()
            mapped = self._AG_METRIC_MAP.get(key)
            if mapped is None:
                logger.warning(
                    "[AutoGluon] eval_metric='%s' not in metric map; passing as-is. "
                    "Known: %s", key, sorted(self._AG_METRIC_MAP),
                )
                return key
            return mapped
        return self._DEFAULT_AG_METRIC[self.task_type]

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, time_budget: int
    ) -> None:
        ensure_empty_dir(self.artifact_dir)

        # AutoGluon requires a single DataFrame with the target column included
        train_df = X_train.copy()
        train_df[self.config.target] = y_train.to_numpy()

        eval_metric = self._ag_eval_metric()
        logger.info(
            "[AutoGluon] problem_type=%s | eval_metric=%s | presets=%s | "
            "time_limit=%ds | model_path=%s",
            self.task_type, eval_metric, self.config.ag_presets,
            time_budget, self.artifact_dir,
        )

        predictor = TabularPredictor(   # type: ignore[misc]
            label=self.config.target,
            problem_type=self.task_type,
            eval_metric=eval_metric,
            path=str(self.artifact_dir),
            verbosity=self.config.ag_verbosity,
        )

        fit_kwargs: Dict[str, Any] = {
            "train_data": TabularDataset(train_df),   # type: ignore[operator]
            "time_limit": int(max(1, time_budget)),
            "presets":    self.config.ag_presets,
        }
        if self.config.ag_excluded_model_types:
            fit_kwargs["excluded_model_types"] = self.config.ag_excluded_model_types
        if self.config.ag_num_bag_folds is not None:
            fit_kwargs["num_bag_folds"] = self.config.ag_num_bag_folds
        if self.config.ag_num_stack_levels is not None:
            fit_kwargs["num_stack_levels"] = self.config.ag_num_stack_levels

        t0 = time.perf_counter()
        self._predictor = predictor.fit(**fit_kwargs)
        self._train_time = time.perf_counter() - t0
        self._is_trained = True

        try:
            n_models = len(self._predictor.model_names())
            best     = self._predictor.get_model_best()
            logger.info(
                "[AutoGluon] Trained %d models in %.1fs | best: %s",
                n_models, self._train_time, best,
            )
        except Exception:
            logger.info("[AutoGluon] Training complete in %.1fs", self._train_time)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("AutoGluonBackend.predict() called before train().")
        return self._predictor.predict(TabularDataset(X.copy())).to_numpy()   # type: ignore[operator]

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        if self.task_type == "regression" or self._predictor is None:
            return None
        try:
            # as_pandas=True → DataFrame with class-label columns;
            # align_proba_to_classes() handles this via Case 1.
            return self._predictor.predict_proba(
                TabularDataset(X.copy()), as_pandas=True   # type: ignore[operator]
            )
        except TypeError:
            # Older AutoGluon versions do not support as_pandas kwarg
            return self._predictor.predict_proba(TabularDataset(X.copy()))   # type: ignore[operator]
        except Exception as exc:
            logger.warning("[AutoGluon] predict_proba failed: %s", exc)
            return None

    @property
    def classes_(self) -> Optional[List[Any]]:
        if self._predictor is None or self.task_type == "regression":
            return None
        for attr in ("class_labels", "class_labels_internal", "classes_"):
            try:
                val = getattr(self._predictor, attr, None)
                if val is not None:
                    return list(val)
            except Exception:
                pass
        return None

    def save_artifact(self) -> Path:
        """AutoGluon auto-saves during fit; record metadata and return the directory."""
        metadata = {
            "backend":       "autogluon",
            "artifact_type": "TabularPredictor directory",
            "load_snippet":  f"TabularPredictor.load({str(self.artifact_dir)!r})",
        }
        write_json(metadata, self.artifact_dir / "artifact_metadata.json")
        return self.artifact_dir

    @property
    def best_model_info(self) -> Dict[str, Any]:
        if self._predictor is None:
            return {}
        info: Dict[str, Any] = {}
        try:
            info["best_model"] = str(self._predictor.get_model_best())
        except Exception:
            pass
        try:
            lb = self._predictor.leaderboard(silent=True)
            info["leaderboard_head"] = lb.head(10).to_dict(orient="records")
        except Exception:
            pass
        return info


# =============================================================================
# §8  FLAML BACKEND
# =============================================================================

class FLAMLBackend(AutoMLBackend):
    """
    FLAML AutoML backend.

    Implementation notes
    --------------------
    · FLAML accepts 'regression' or 'classification' as task (not 'binary' /
      'multiclass'); this class maps transparently.
    · Default classification metric is 'accuracy' (most broadly interpretable).
      Override via config.eval_metric.
    · time_budget is enforced natively by FLAML's optimizer controller.
    · Datetime columns are detected at fit time and converted to numeric
      timestamps.  The list is saved on self so inference applies the same
      transform — preventing silent feature-type leakage.
    · Fit retry: some FLAML versions reject n_jobs in fit().  A TypeError
      triggers an automatic retry without n_jobs (older-version guard).
    · The fitted AutoML object is pickled for persistence and reloading.
    """

    _FLAML_TASK_MAP: Dict[str, str] = {
        "regression": "regression",
        "binary":     "classification",
        "multiclass": "classification",
    }
    _DEFAULT_FLAML_METRIC: Dict[str, str] = {
        "regression": "rmse",
        "binary":     "accuracy",
        "multiclass": "accuracy",
    }

    def __init__(
        self, config: BenchmarkConfig, task_type: str, artifact_dir: Path
    ) -> None:
        if not HAS_FLAML:
            raise ImportError(
                "FLAML is not installed.\n"
                "  pip install 'flaml[automl]'\n"
                f"  Original error: {_FLAML_IMPORT_ERR}"
            )
        super().__init__(config, task_type, artifact_dir)
        self.automl: Any = None
        self._feature_columns: Optional[List[str]] = None
        self._datetime_columns: List[str] = []

    def _flaml_metric(self) -> str:
        if self.config.eval_metric:
            return self.config.eval_metric.lower()
        return self._DEFAULT_FLAML_METRIC[self.task_type]

    def _prepare_X(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Leakage-free feature preprocessing.

        At *fit time* (fit=True): records the feature column list and detects
        datetime columns for numeric conversion.
        At *inference time* (fit=False): applies the same transformations to
        guarantee consistency.  Only columns seen at fit time are retained,
        preventing silent schema drift.
        """
        Xp = X.copy()

        if fit:
            self._feature_columns  = list(Xp.columns)
            self._datetime_columns = [
                c for c in Xp.columns
                if pd.api.types.is_datetime64_any_dtype(Xp[c])
            ]

        if self._feature_columns is not None:
            Xp = Xp[[c for c in self._feature_columns if c in Xp.columns]].copy()

        for col in self._datetime_columns:
            if col in Xp.columns:
                dts = pd.to_datetime(Xp[col], errors="coerce")
                Xp[col] = dts.view("int64").astype(float)
                Xp.loc[dts.isna(), col] = np.nan   # preserve NaN semantics

        return Xp

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, time_budget: int
    ) -> None:
        ensure_empty_dir(self.artifact_dir)

        flaml_task   = self._FLAML_TASK_MAP[self.task_type]
        flaml_metric = self._flaml_metric()
        Xp           = self._prepare_X(X_train, fit=True)

        logger.info(
            "[FLAML] task=%s | metric=%s | time_budget=%ds | path=%s",
            flaml_task, flaml_metric, time_budget, self.artifact_dir,
        )

        self.automl = _FLAMLAutoML()   # type: ignore[misc]
        settings: Dict[str, Any] = {
            "time_budget": int(max(1, time_budget)),
            "metric":      flaml_metric,
            "task":        flaml_task,
            "seed":        self.config.random_state,
            "verbose":     1,
            "n_jobs":      self.config.flaml_n_jobs,
        }
        if self.config.flaml_log_file:
            settings["log_file_name"] = self.config.flaml_log_file
        if self.config.flaml_estimator_list:
            settings["estimator_list"] = self.config.flaml_estimator_list

        t0 = time.perf_counter()
        try:
            self.automl.fit(X_train=Xp, y_train=y_train, **settings)
        except TypeError as exc:
            # Guard: older FLAML versions may not support n_jobs in fit()
            if "n_jobs" in settings:
                logger.warning(
                    "[FLAML] Retrying fit without n_jobs (TypeError: %s).", exc
                )
                settings.pop("n_jobs")
                self.automl.fit(X_train=Xp, y_train=y_train, **settings)
            else:
                raise

        self._train_time = time.perf_counter() - t0
        self._is_trained = True
        logger.info(
            "[FLAML] Training complete in %.1fs | best estimator: %s",
            self._train_time,
            getattr(self.automl, "best_estimator", "unknown"),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.automl is None:
            raise RuntimeError("FLAMLBackend.predict() called before train().")
        Xp = self._prepare_X(X, fit=False)
        return np.asarray(self.automl.predict(Xp))

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        if self.task_type == "regression" or self.automl is None:
            return None
        if not hasattr(self.automl, "predict_proba"):
            return None
        try:
            Xp = self._prepare_X(X, fit=False)
            return np.asarray(self.automl.predict_proba(Xp), dtype=float)
        except Exception as exc:
            logger.warning("[FLAML] predict_proba failed: %s", exc)
            return None

    @property
    def classes_(self) -> Optional[List[Any]]:
        if self.automl is None or self.task_type == "regression":
            return None
        # Walk common attribute paths across FLAML versions
        for obj in (
            self.automl,
            getattr(self.automl, "model", None),
            getattr(getattr(self.automl, "model", None), "estimator", None),
        ):
            if obj is None:
                continue
            try:
                val = getattr(obj, "classes_", None)
                if val is not None:
                    return list(val)
            except Exception:
                pass
        return None

    def save_artifact(self) -> Path:
        if self.automl is None:
            raise RuntimeError("FLAMLBackend.save_artifact() called before train().")
        ensure_dir(self.artifact_dir)
        model_path = self.artifact_dir / "flaml_model.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(self.automl, fh)
        metadata = {
            "backend":       "flaml",
            "artifact_type": "pickle",
            "model_path":    str(model_path),
            "load_snippet":  f"pickle.load(open({str(model_path)!r}, 'rb'))",
        }
        write_json(metadata, self.artifact_dir / "artifact_metadata.json")
        logger.info("[FLAML] Model pickled to: %s", model_path)
        return model_path

    @property
    def best_model_info(self) -> Dict[str, Any]:
        if self.automl is None:
            return {}
        info: Dict[str, Any] = {}
        for attr in (
            "best_estimator", "best_config", "best_loss",
            "best_config_train_time", "best_iteration",
        ):
            try:
                info[attr] = getattr(self.automl, attr)
            except Exception:
                pass
        return info


# =============================================================================
# §9  FACTORY
# =============================================================================

_BACKEND_REGISTRY: Dict[str, type] = {
    "autogluon": AutoGluonBackend,
    "flaml":     FLAMLBackend,
}
"""
Registry mapping backend names → concrete AutoMLBackend subclasses.

Adding a new library (e.g., H2O, Ludwig) requires only:
  1. Subclass AutoMLBackend with train / predict / predict_proba / save_artifact.
  2. Add one entry:  _BACKEND_REGISTRY["h2o"] = H2OBackend
The CLI, factory, and runner pick it up automatically.
"""


class BackendFactory:
    """Factory that creates AutoMLBackend instances from a BenchmarkConfig."""

    @staticmethod
    def create(
        config: BenchmarkConfig,
        task_type: str,
        artifact_dir: Path,
    ) -> AutoMLBackend:
        """
        Instantiate the concrete backend named by ``config.backend``.

        Raises ValueError for unknown backends, ImportError for uninstalled ones.
        """
        key = config.backend.lower()
        if key not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Unknown backend '{config.backend}'. "
                f"Available: {sorted(_BACKEND_REGISTRY)}"
            )
        return _BACKEND_REGISTRY[key](
            config=config, task_type=task_type, artifact_dir=artifact_dir
        )


# =============================================================================
# §10  BENCHMARK RUNNER (ORCHESTRATOR)
# =============================================================================

class BenchmarkRunner:
    """
    Top-level orchestrator: load → split → OOF CV → final model → write outputs.

    Evaluation protocol
    -------------------
    1. Stratified holdout split (default 80/20).
    2. External K-fold OOF CV on the training portion → oof_metrics.
       Both AutoGluon and FLAML run under the SAME CV protocol, so their
       oof_metrics are directly comparable.
    3. Final model trained on all training data within the remaining budget
       → holdout_metrics on the held-out test partition.
    4. Standardised output files written to output_dir.

    AutoGluon's leaderboard scores and FLAML's best_loss are captured as
    supplementary metadata under "backend_native_info" in metrics.json.
    They are NOT labelled as OOF metrics because they are qualitatively
    different quantities (different splits, different aggregation).
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config    = config
        self.run_dir   = Path(config.output_dir)
        self.folds_dir = self.run_dir / "artifacts" / "folds"
        self.final_dir = self.run_dir / "artifacts" / "final"

    def _prepare_output_dir(self) -> None:
        if self.run_dir.exists() and self.config.overwrite:
            logger.info("Overwriting output directory: %s", self.run_dir)
            shutil.rmtree(self.run_dir)
        ensure_dir(self.run_dir)
        ensure_dir(self.run_dir / "artifacts")

    def _save_loader_snippet(self, artifact_path: Path) -> None:
        """Write a backend-specific Python snippet for downstream model loading."""
        loader = self.run_dir / "load_model.py"
        if self.config.backend == "autogluon":
            code = (
                '"""Load the saved AutoGluon model."""\n'
                "from autogluon.tabular import TabularPredictor\n\n"
                f"predictor = TabularPredictor.load({str(artifact_path)!r})\n"
                "# preds = predictor.predict(new_df)\n"
            )
        else:
            code = (
                '"""Load the saved FLAML model."""\n'
                "import pickle\n\n"
                f"with open({str(artifact_path)!r}, 'rb') as f:\n"
                "    automl = pickle.load(f)\n"
                "# preds = automl.predict(new_df)\n"
            )
        loader.write_text(code, encoding="utf-8")
        logger.info("Model loading snippet → %s", loader)

    def run(self) -> Dict[str, Any]:
        """
        Execute the full benchmark pipeline.

        Returns the metrics payload dict (same content as metrics.json).
        """
        self.config.validate()
        self._prepare_output_dir()

        # Persist config immediately for post-hoc reproducibility
        write_json(asdict(self.config), self.run_dir / "config_snapshot.json")

        # ── 1. Load and prepare data ──────────────────────────────────────────
        df = load_data(self.config)
        X, y, task_type, classes = prepare_xy(df, self.config)

        # ── 2. Stratified holdout split ───────────────────────────────────────
        stratify = y if task_type != "regression" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )
        X_train = X_train.reset_index(drop=True)
        X_test  = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)
        logger.info(
            "Holdout split: train=%d, test=%d (test_size=%.0f%%)",
            len(X_train), len(X_test), self.config.test_size * 100,
        )

        # ── 3. External OOF cross-validation ─────────────────────────────────
        splitter, n_splits, is_stratified = make_cv_splitter(
            y=y_train, task_type=task_type,
            n_splits=self.config.n_splits, random_state=self.config.random_state,
        )
        fold_budget, final_budget = plan_time_budgets(self.config, n_splits)

        logger.info("=" * 72)
        logger.info(
            "External OOF CV: %d folds | fold_budget=%ds | "
            "final_budget=%ds | stratified=%s",
            n_splits, fold_budget, final_budget, is_stratified,
        )
        logger.info("=" * 72)

        n = len(y_train)
        if task_type == "regression":
            oof_pred  = np.full(n, np.nan, dtype=float)
            oof_proba = None
        else:
            oof_pred  = np.full(n, None, dtype=object)
            oof_proba = np.full((n, len(classes or [])), np.nan, dtype=float)

        fold_ids: np.ndarray = np.full(n, -1, dtype=int)
        fold_records: List[Dict[str, Any]] = []

        split_iter = (
            splitter.split(X_train, y_train) if is_stratified
            else splitter.split(X_train)
        )

        for fold_idx, (tr_idx, va_idx) in enumerate(split_iter, start=1):
            logger.info(
                "Fold %d/%d — train=%d  val=%d",
                fold_idx, n_splits, len(tr_idx), len(va_idx),
            )

            fold_dir = self.folds_dir / f"fold_{fold_idx:03d}"
            backend  = BackendFactory.create(
                config=self.config, task_type=task_type, artifact_dir=fold_dir
            )

            t0 = time.perf_counter()
            backend.train(
                X_train=X_train.iloc[tr_idx],
                y_train=y_train.iloc[tr_idx],
                time_budget=fold_budget,
            )
            elapsed = time.perf_counter() - t0

            fold_pred   = backend.predict(X_train.iloc[va_idx])
            oof_pred[va_idx] = fold_pred

            fold_proba: Optional[np.ndarray] = None
            if task_type != "regression" and classes is not None:
                raw = backend.predict_proba(X_train.iloc[va_idx])
                fold_proba = align_proba_to_classes(
                    proba=raw, target_classes=classes, backend_classes=backend.classes_
                )
                if fold_proba is not None and oof_proba is not None:
                    oof_proba[va_idx, :] = fold_proba

            fold_ids[va_idx] = fold_idx

            fold_metrics = compute_metrics(
                task_type=task_type,
                y_true=y_train.iloc[va_idx].to_numpy(),
                y_pred=fold_pred,
                y_proba=fold_proba,
                classes=classes,
            )

            record: Dict[str, Any] = {
                "fold":            fold_idx,
                "n_train":         int(len(tr_idx)),
                "n_val":           int(len(va_idx)),
                "time_budget_sec": fold_budget,
                "elapsed_sec":     round(elapsed, 3),
                "metrics":         fold_metrics,
                "backend_native_info": backend.best_model_info,
            }

            if self.config.keep_fold_models:
                record["artifact_path"] = str(backend.save_artifact())
            else:
                shutil.rmtree(fold_dir, ignore_errors=True)

            fold_records.append(record)
            brief = {
                k: round(float(v), 4)
                for k, v in fold_metrics.items()
                if k in {"r2", "rmse", "mae", "accuracy", "f1_macro"}
                and isinstance(v, (int, float))
            }
            logger.info("  → fold %d: %s  (%.1fs)", fold_idx, brief, elapsed)

        # ── Aggregate OOF metrics ─────────────────────────────────────────────
        if task_type == "regression":
            oof_pred_f  = oof_pred.astype(float)
            valid_mask  = (fold_ids >= 0) & np.isfinite(oof_pred_f)
            proba_valid: Optional[np.ndarray] = None
        else:
            valid_mask = fold_ids >= 0
            if oof_proba is not None:
                pb = oof_proba[valid_mask]
                proba_valid = pb if np.isfinite(pb).all() else None
            else:
                proba_valid = None

        if not valid_mask.any():
            raise RuntimeError(
                "No valid OOF predictions produced; cannot compute OOF metrics."
            )

        oof_metrics = compute_metrics(
            task_type=task_type,
            y_true=y_train[valid_mask].to_numpy(),
            y_pred=(oof_pred_f[valid_mask] if task_type == "regression"
                    else oof_pred[valid_mask]),
            y_proba=proba_valid,
            classes=classes,
        )
        logger.info("=" * 72)
        logger.info(
            "OOF metrics: %s",
            {k: round(float(v), 4) for k, v in oof_metrics.items()
             if isinstance(v, (int, float))},
        )

        # Write OOF predictions CSV
        oof_df = pd.DataFrame({
            "fold":   fold_ids,
            "y_true": y_train.to_numpy(),
            "y_pred": oof_pred,
        })
        if task_type != "regression" and oof_proba is not None and classes is not None:
            for j, cls in enumerate(classes):
                oof_df[f"proba_{sanitize_name(cls)}"] = oof_proba[:, j]
        oof_df.to_csv(self.run_dir / "oof_predictions.csv", index=False)

        # ── 4. Final model — trained on the full training partition ───────────
        logger.info("=" * 72)
        logger.info(
            "Training final model — %d rows | budget=%ds",
            len(X_train), final_budget,
        )
        logger.info("=" * 72)

        final_backend = BackendFactory.create(
            config=self.config, task_type=task_type, artifact_dir=self.final_dir
        )
        t0 = time.perf_counter()
        final_backend.train(X_train=X_train, y_train=y_train, time_budget=final_budget)
        final_elapsed = time.perf_counter() - t0

        holdout_metrics = final_backend.evaluate(X_test, y_test, classes)
        artifact_path   = final_backend.save_artifact()
        self._save_loader_snippet(artifact_path)

        logger.info(
            "Holdout metrics: %s",
            {k: round(float(v), 4) for k, v in holdout_metrics.items()
             if isinstance(v, (int, float))},
        )

        # ── 5. Write standardised metrics.json ───────────────────────────────
        payload: Dict[str, Any] = {
            "schema_version":  "2.0",
            "backend":         self.config.backend,
            "task_type":       task_type,
            "target":          self.config.target,
            "n_train":         int(len(X_train)),
            "n_test":          int(len(X_test)),
            "n_features":      int(X.shape[1]),
            "classes":         classes,
            "cv": {
                "n_splits":            n_splits,
                "stratified":          is_stratified,
                "fold_budget_sec":     fold_budget,
                "final_budget_sec":    final_budget,
                "oof_budget_fraction": self.config.oof_budget_fraction,
            },
            # Primary comparable metrics (same protocol for every backend):
            "oof_metrics":     oof_metrics,
            "holdout_metrics": holdout_metrics,
            # Per-fold breakdown:
            "fold_metrics": fold_records,
            # Final model metadata:
            "final_model": {
                "artifact_path":       str(artifact_path),
                "elapsed_sec":         round(final_elapsed, 3),
                # Supplementary metadata — NOT equivalent to oof_metrics:
                "backend_native_info": final_backend.best_model_info,
            },
            "outputs": {
                "metrics_json":         str(self.run_dir / "metrics.json"),
                "oof_predictions_csv":  str(self.run_dir / "oof_predictions.csv"),
                "config_snapshot_json": str(self.run_dir / "config_snapshot.json"),
                "loader_snippet":       str(self.run_dir / "load_model.py"),
            },
        }

        write_json(payload, self.run_dir / "metrics.json")
        logger.info("All outputs written to: %s", self.run_dir.resolve())

        return payload


# =============================================================================
# §11  CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the full CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Unified AutoML benchmark harness (AutoGluon & FLAML).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Control
    p.add_argument(
        "--demo", action="store_true",
        help="Smoke-test on synthetic data — no real dataset required.",
    )

    # Core
    p.add_argument(
        "--backend", choices=sorted(_BACKEND_REGISTRY),
        help="AutoML backend to use.",
    )
    p.add_argument("--data_path",
                   help="Path to data (CSV, TSV, Parquet, Feather).")
    p.add_argument("--target", help="Target column name.")
    p.add_argument(
        "--task", default="auto",
        choices=["auto", "regression", "classification", "binary", "multiclass"],
        help="Task type; 'auto' infers from the target.",
    )
    p.add_argument("--time_budget", type=int, default=300,
                   help="Total training budget in seconds.")
    p.add_argument("--output_dir", default="automl_benchmark",
                   help="Directory for outputs.")

    # Data
    p.add_argument("--file_format", default=None,
                   choices=["csv", "tsv", "parquet", "pq", "feather"],
                   help="Force file format (inferred from extension by default).")
    p.add_argument("--csv_sep", default=",", help="CSV delimiter.")
    p.add_argument("--csv_encoding", default="utf-8")
    p.add_argument("--drop_columns", nargs="*", default=[],
                   help="Columns to drop before training (space- or comma-separated).")

    # Evaluation
    p.add_argument("--test_size",   type=float, default=0.20,
                   help="Holdout fraction.")
    p.add_argument("--n_splits",    type=int,   default=5,
                   help="OOF CV folds.")
    p.add_argument("--oof_budget_fraction", type=float, default=0.60,
                   help="Share of time_budget for OOF folds.")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--eval_metric", default=None,
                   help="Backend-native optimization metric (e.g. rmse, roc_auc).")

    # Output
    p.add_argument("--keep_fold_models", action="store_true",
                   help="Keep per-fold model artifacts (large disk use).")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete and recreate output_dir before running.")

    # AutoGluon group
    ag = p.add_argument_group("AutoGluon options")
    ag.add_argument("--ag_presets", default="best_quality",
                    help="AutoGluon preset (best_quality / high_quality / medium_quality).")
    ag.add_argument("--ag_excluded_model_types", nargs="*", default=[],
                    help="Model types to skip — e.g. --ag_excluded_model_types NN_TORCH "
                         "(recommended on macOS Apple Silicon).")
    ag.add_argument("--ag_num_bag_folds", type=int, default=None)
    ag.add_argument("--ag_num_stack_levels", type=int, default=None)
    ag.add_argument("--ag_verbosity", type=int, default=2)

    # FLAML group
    fl = p.add_argument_group("FLAML options")
    fl.add_argument("--flaml_estimators", nargs="*", default=None,
                    help="FLAML estimator allow-list — e.g. lgbm xgboost rf.")
    fl.add_argument("--flaml_n_jobs", type=int, default=-1)
    fl.add_argument("--flaml_log_file", default="",
                    help="Path for FLAML training log ('' = disabled).")

    return p


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert a parsed Namespace to a BenchmarkConfig."""
    return BenchmarkConfig(
        backend=args.backend or "autogluon",
        data_path=args.data_path or "",
        target=args.target or "",
        task=args.task,
        time_budget=args.time_budget,
        output_dir=args.output_dir,
        file_format=args.file_format,
        csv_sep=args.csv_sep,
        csv_encoding=args.csv_encoding,
        drop_columns=normalize_cli_list(args.drop_columns),
        test_size=args.test_size,
        n_splits=args.n_splits,
        oof_budget_fraction=args.oof_budget_fraction,
        random_state=args.random_state,
        eval_metric=args.eval_metric,
        keep_fold_models=args.keep_fold_models,
        overwrite=args.overwrite,
        ag_presets=args.ag_presets,
        ag_excluded_model_types=normalize_cli_list(args.ag_excluded_model_types),
        ag_num_bag_folds=args.ag_num_bag_folds,
        ag_num_stack_levels=args.ag_num_stack_levels,
        ag_verbosity=args.ag_verbosity,
        flaml_estimator_list=normalize_cli_list(args.flaml_estimators) or None,
        flaml_n_jobs=args.flaml_n_jobs,
        flaml_log_file=args.flaml_log_file,
    )


# =============================================================================
# §12  PUBLIC PYTHON API
# =============================================================================

def run_automl_benchmark(
    backend: str,
    data_path: str,
    target: str,
    task: str = "auto",
    time_budget: int = 300,
    output_dir: str = "automl_benchmark",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Python API entry point — run a benchmark without the CLI.

    Parameters
    ----------
    backend : str
        'autogluon' or 'flaml'.
    data_path : str
        Path to a data file (CSV, TSV, Parquet, or Feather).
    target : str
        Name of the target column.
    task : str, optional
        'auto', 'regression', 'classification', 'binary', or 'multiclass'.
    time_budget : int, optional
        Total wall-clock training budget in seconds.
    output_dir : str, optional
        Directory for all outputs.
    **kwargs
        Any additional BenchmarkConfig fields as keyword arguments.

    Returns
    -------
    dict
        Full metrics payload (same content as metrics.json).

    Examples
    --------
    >>> from automl_benchmark import run_automl_benchmark
    >>> m = run_automl_benchmark(
    ...     backend="flaml", data_path="housing.csv",
    ...     target="price", task="regression", time_budget=120,
    ... )
    >>> print(m["oof_metrics"]["r2"])
    """
    cfg = BenchmarkConfig(
        backend=backend,
        data_path=data_path,
        target=target,
        task=task,
        time_budget=time_budget,
        output_dir=output_dir,
        **kwargs,
    )
    return BenchmarkRunner(cfg).run()


# =============================================================================
# §13  DEMO / SMOKE TEST
# =============================================================================

def _run_demo(backend: str, output_dir: str = "automl_demo") -> None:
    """
    Smoke test using sklearn's synthetic regression data.

    Validates that the selected backend is installed, callable, and produces
    all expected output files — without needing any external dataset.
    Useful for verifying installation on a new machine.
    """
    logger.info("=" * 72)
    logger.info(
        "DEMO MODE — backend='%s' | synthetic regression (500 rows, 10 features)",
        backend,
    )
    logger.info("=" * 72)

    X_np, y_np = make_regression(
        n_samples=500, n_features=10, noise=0.2, random_state=42
    )
    demo_df = pd.DataFrame(
        X_np, columns=[f"feat_{i}" for i in range(X_np.shape[1])]
    )
    demo_df["target"] = y_np

    demo_dir = Path(output_dir)
    ensure_dir(demo_dir)
    demo_csv = demo_dir / "_demo_data.csv"
    demo_df.to_csv(demo_csv, index=False)

    cfg = BenchmarkConfig(
        backend=backend,
        data_path=str(demo_csv),
        target="target",
        task="regression",
        time_budget=60,
        output_dir=output_dir,
        n_splits=3,
        overwrite=True,
        ag_verbosity=1,
        # NN_TORCH excluded by default for broad macOS compatibility
        ag_excluded_model_types=["NN_TORCH"],
    )

    metrics = BenchmarkRunner(cfg).run()

    oof     = metrics.get("oof_metrics", {})
    holdout = metrics.get("holdout_metrics", {})
    logger.info("Demo complete.")
    logger.info(
        "  OOF     R²=%.4f  RMSE=%.4f",
        oof.get("r2", float("nan")), oof.get("rmse", float("nan")),
    )
    logger.info(
        "  Holdout R²=%.4f  RMSE=%.4f",
        holdout.get("r2", float("nan")), holdout.get("rmse", float("nan")),
    )
    logger.info("Outputs → %s", demo_dir.resolve())


# =============================================================================
# §14  ENTRY POINT
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args   = parser.parse_args()

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if args.demo:
        backend = (args.backend or "autogluon").lower()
        if backend == "autogluon" and not HAS_AUTOGLUON:
            logger.error(
                "AutoGluon not installed: pip install 'autogluon.tabular[all]'"
            )
            sys.exit(1)
        if backend == "flaml" and not HAS_FLAML:
            logger.error("FLAML not installed: pip install 'flaml[automl]'")
            sys.exit(1)
        try:
            _run_demo(backend=backend, output_dir=args.output_dir)
        except Exception as exc:
            logger.error("Demo failed: %s", exc, exc_info=True)
            sys.exit(1)
        return

    # ── Validate required arguments for normal mode ───────────────────────────
    if not args.backend:
        parser.error("--backend is required (or use --demo).")
    if not args.data_path:
        parser.error("--data_path is required (or use --demo).")
    if not args.target:
        parser.error("--target is required (or use --demo).")

    # Fail fast with a clear message if the chosen backend is not installed
    if args.backend == "autogluon" and not HAS_AUTOGLUON:
        logger.error(
            "AutoGluon not installed.\n"
            "  pip install 'autogluon.tabular[all]'\n"
            "  Original error: %s", _AG_IMPORT_ERR,
        )
        sys.exit(1)
    if args.backend == "flaml" and not HAS_FLAML:
        logger.error(
            "FLAML not installed.\n"
            "  pip install 'flaml[automl]'\n"
            "  Original error: %s", _FLAML_IMPORT_ERR,
        )
        sys.exit(1)

    cfg = config_from_args(args)
    try:
        metrics = BenchmarkRunner(cfg).run()

        oof     = metrics.get("oof_metrics", {})
        holdout = metrics.get("holdout_metrics", {})

        sep = "=" * 72
        print(f"\n{sep}")
        print(f"Backend   : {metrics['backend']}")
        print(f"Task      : {metrics['task_type']}")
        print(f"Output    : {cfg.output_dir}")

        if metrics["task_type"] == "regression":
            def _fmt(d: Dict, k: str) -> str:
                v = d.get(k)
                return f"{v:.6f}" if isinstance(v, float) else str(v)
            print(
                f"OOF       : R²={_fmt(oof,'r2')}  "
                f"RMSE={_fmt(oof,'rmse')}  MAE={_fmt(oof,'mae')}"
            )
            print(
                f"Holdout   : R²={_fmt(holdout,'r2')}  "
                f"RMSE={_fmt(holdout,'rmse')}  MAE={_fmt(holdout,'mae')}"
            )
        else:
            print(
                f"OOF       : accuracy={oof.get('accuracy','n/a')}  "
                f"f1_macro={oof.get('f1_macro','n/a')}"
            )
            print(
                f"Holdout   : accuracy={holdout.get('accuracy','n/a')}  "
                f"f1_macro={holdout.get('f1_macro','n/a')}"
            )
        print(f"{sep}\n")

    except Exception as exc:
        logger.error("Benchmark run failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
