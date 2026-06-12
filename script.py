"""
================================================================================
SOTA Tabular Supervised Learning Pipeline — v5.0
================================================================================
A production-grade, fully domain-agnostic supervised learning framework for
tabular regression. Fully pluggable, leakage-free, source-agnostic.

DATA SOURCES (all pluggable via DataSource ABC):
  - CSV, TSV, Excel (.xlsx/.xls), Parquet, Feather, HDF5, JSON, JSONL
  - SQLite (single query)
  - PostgreSQL / MySQL / Oracle / MSSQL via SQLAlchemy (optional)
  - In-memory pd.DataFrame
  - Custom user-defined sources (subclass DataSource ABC)

METHODOLOGY (SOTA):
  - True Nested CV: outer GroupKFold/KFold + inner CV + Optuna TPE
  - Bayesian HPO with MedianPruner
  - Leakage-safe per-fold target transforms (log1p, Yeo-Johnson, Box-Cox,
    quantile-normal) with automatic inverse on predict
  - Automatic column type inference (numeric / categorical / datetime / drop)
  - Automatic categorical encoding: OneHot for linear models, Ordinal for trees
  - Datetime expansion to numeric parts (year, month, day, hour, etc.)
  - Optional CLR transform for compositional features (user opt-in)
  - Correlation filter (removes near-duplicate features)
  - Multi-task: RegressorChain + MultiOutputRegressor
  - Split conformal prediction (calibration data isolated before HPO)
  - Feature stability tracking with real post-preprocessor names
  - Per-fold SHAP (TreeExplainer / LinearExplainer / KernelExplainer fallback)
  - Per-fold permutation importance (optional)
  - OOF predictions + bootstrap 95% CIs
  - Stacking meta-learner on OOF base predictions

MODELS (graceful optional imports):
  - Dummy, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor,
    LinearSVR, SVR, KNeighborsRegressor
  - RandomForest, ExtraTrees, HistGradientBoosting
  - XGBoost, LightGBM, CatBoost (optional)
  - TabPFN (optional, SOTA for small datasets)
  - User-registered custom models via ModelRegistry.register()

ARCHITECTURE (fully pluggable ABCs):
  - DataSource ABC → FileDataSource, SQLiteDataSource, SQLAlchemyDataSource,
                      InMemoryDataSource
  - TypeInferenceEngine   → auditable column-role classification
  - FeatureEngineer ABC   → PassthroughEngineer (default), custom subclasses
  - LabelExtractor ABC    → ColumnLabelExtractor (default), custom subclasses
  - PreprocessorBuilder   → column-type-aware ColumnTransformer construction
  - DatasetBuilder        → composes the above into a Dataset
  - ModelRegistry, PipelineFactory, OptunaTuner
  - TargetTransformer     → per-fold invertible target scaling
  - CLRTransformer, ILRTransformer (compositional data, opt-in)
  - CorrelationFilter, DatetimeExpander
  - ConformalPredictor    → split conformal with pre-HPO calibration isolation
  - FeatureStabilityTracker
  - SHAPComputer
  - MultiTaskRunner, StackingRunner
  - NestedCVRunner        → core engine
  - ResultsManager        → Parquet + CSV + per-fold artifacts + config snapshot
  - TabularMLPipeline     → top-level orchestrator

CONVENIENCE FACTORIES:
  - from_dataframe(df, targets, ...)
  - from_csv(path, targets, ...)
  - from_parquet(path, targets, ...)
  - from_excel(path, targets, ...)
  - from_sqlite(db_path, query, targets, ...)
  - from_sqlalchemy(uri, query, targets, ...)
================================================================================
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import sqlite3
import sys
import time
import warnings
import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import optuna
import pandas as pd
from scipy.stats import skew as scipy_skew

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR, LinearSVR

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

try:
    from sqlalchemy import create_engine, text as sa_text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# TargetEncoder was added in scikit-learn 1.3
try:
    from sklearn.preprocessing import TargetEncoder
    HAS_TARGET_ENCODER = True
except ImportError:
    HAS_TARGET_ENCODER = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TabularML")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

@dataclass
class DataSourceConfig:
    """Unified configuration for any tabular data source."""
    # File-based
    file_path: Optional[str] = None
    file_format: Optional[str] = None   # auto-detected from extension if None
    csv_sep: str = ","
    csv_encoding: str = "utf-8"
    excel_sheet: Union[str, int] = 0
    hdf5_key: str = "data"
    json_orient: Optional[str] = None
    # SQL
    sqlite_path: Optional[str] = None
    sqlalchemy_uri: Optional[str] = None
    sql_query: Optional[str] = None


@dataclass
class FeatureConfig:
    """
    Feature preprocessing and selection configuration.
    All defaults are intentionally domain-agnostic.
    """
    # Column type inference
    max_cardinality_for_categorical: int = 50
    max_missing_fraction: float = 0.90
    # Datetime expansion
    expand_datetime: bool = True
    datetime_column_name_patterns: List[str] = field(default_factory=lambda: [
        "date", "time", "timestamp", "year", "month", "day",
    ])
    # Encoding
    onehot_handle_unknown: str = "ignore"
    onehot_max_categories: Optional[int] = 50
    # Imputation
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    # Scaling
    numeric_scaler: str = "standard"   # standard | robust | quantile | none
    # Filters
    use_variance_threshold: bool = True
    variance_threshold: float = 1e-8
    use_correlation_filter: bool = False
    correlation_threshold: float = 0.98
    # Feature selection
    selectkbest_k: int = 30
    tree_selector_threshold: str = "median"
    tree_selector_n_estimators: int = 200
    # PCA
    pca_variance_ratio: float = 0.95
    # Compositional (CLR) — opt-in only
    apply_clr_transform: bool = False
    clr_column_pattern: str = ""        # regex; only used when apply_clr_transform=True
    clr_pseudocount: float = 1e-6
    # Manual column type overrides: {regex_pattern: "numeric"|"categorical"|"drop"}
    column_type_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class TargetConfig:
    """Target/label configuration."""
    target_columns: List[str] = field(default_factory=list)
    # Per-target explicit transform overrides
    target_transforms: Dict[str, str] = field(default_factory=dict)
    # Auto-selection of target transform by skewness
    auto_transform: bool = True
    skew_threshold: float = 1.0
    preferred_auto_transform: str = "yeo-johnson"  # log1p | yeo-johnson | box-cox | quantile-normal
    # Targets with all-positive values can safely use log1p
    positive_targets: Optional[List[str]] = None


@dataclass
class MLConfig:
    """Machine learning, CV, HPO, and output configuration."""
    random_state: int = 42
    n_outer_splits: int = 5
    n_inner_splits: int = 4
    n_optuna_trials: int = 60
    optuna_timeout_sec: Optional[int] = None
    optuna_n_startup_trials: int = 10
    optuna_use_pruning: bool = True
    # Group-aware CV
    use_group_cv: bool = False
    group_column: Optional[str] = None
    sample_weight_column: Optional[str] = None
    drop_columns: List[str] = field(default_factory=list)
    # Resources
    n_jobs: int = -1
    # I/O
    output_dir: str = "ml_results"
    save_best_models: bool = True
    save_oof_predictions: bool = True
    save_config_snapshot: bool = True
    # Conformal prediction
    compute_conformal: bool = True
    conformal_alpha: float = 0.10
    conformal_mode: str = "strict"   # "strict" | "empirical"
    conformal_calibration_fraction: float = 0.20
    # SHAP
    compute_shap: bool = True
    shap_sample_size: int = 200
    shap_kernel_fallback: bool = True
    # Permutation importance
    compute_permutation_importance: bool = False
    permutation_n_repeats: int = 5
    # Feature stability
    track_feature_stability: bool = True
    # Multi-task
    use_multitask: bool = False
    # Stacking
    use_stacking: bool = False
    # Model selection
    include_models: Optional[List[str]] = None   # None → all available
    exclude_models: List[str] = field(default_factory=list)
    # Feature strategy override
    feature_strategies_override: Optional[List[str]] = None
    # Minimum valid samples required per target
    min_samples_per_label: int = 30


# =============================================================================
# 1. UTILITIES
# =============================================================================

def sanitize_name(name: str) -> str:
    """Sanitize a column name to a safe identifier (alphanumeric + underscore)."""
    cleaned = re.sub(r"[^\w]+", "_", str(name)).strip("_")
    return cleaned if cleaned else "col"


def sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with sanitized, deduplicated column names."""
    df = df.copy()
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in df.columns:
        base = sanitize_name(c)
        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
    df.columns = new_cols
    return df


def bootstrap_ci(
    data: np.ndarray,
    stat: Callable = np.median,
    n_resamples: int = 5_000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI at level (1 - alpha)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(x, size=(n_resamples, x.size), replace=True)
    stats = np.apply_along_axis(stat, 1, samples)
    return (
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metric suite (all finite-value safe)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {k: float("nan") for k in ("r2", "mae", "medae", "mse", "rmse", "mape")}
    yt, yp = y_true[mask], y_pred[mask]
    mse = float(mean_squared_error(yt, yp))
    out: Dict[str, float] = {
        "r2":    float(r2_score(yt, yp)),
        "mae":   float(mean_absolute_error(yt, yp)),
        "medae": float(median_absolute_error(yt, yp)),
        "mse":   mse,
        "rmse":  float(np.sqrt(mse)),
    }
    try:
        out["mape"] = float(mean_absolute_percentage_error(yt, yp))
    except Exception:
        out["mape"] = float("nan")
    return out


def detect_abs_skewness(y: pd.Series) -> float:
    """Absolute skewness of non-null values."""
    vals = y.dropna().values
    if len(vals) < 4:
        return 0.0
    try:
        return float(abs(scipy_skew(vals)))
    except Exception:
        return 0.0


# =============================================================================
# 2. COLUMN TYPE INFERENCE
# =============================================================================

class TypeInferenceEngine:
    """
    Scans a DataFrame and assigns a role to every column:
      'numeric' | 'categorical' | 'datetime' | 'drop'

    The result is explicit and auditable via .column_roles.
    Roles can be overridden via FeatureConfig.column_type_overrides (regex).
    """

    def __init__(self, feat_config: FeatureConfig) -> None:
        self.cfg = feat_config
        self.column_roles: Dict[str, str] = {}

    def fit(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> "TypeInferenceEngine":
        exclude = set(exclude_columns or [])
        overrides = self.cfg.column_type_overrides

        for col in df.columns:
            if col in exclude:
                self.column_roles[col] = "drop"
                continue

            # Override via regex
            forced = self._match_override(col, overrides)
            if forced is not None:
                self.column_roles[col] = forced
                continue

            # Missing-value gate
            if df[col].isna().mean() > self.cfg.max_missing_fraction:
                logger.debug(f"  Dropping '{col}': exceeds max_missing_fraction.")
                self.column_roles[col] = "drop"
                continue

            self.column_roles[col] = self._infer_role(col, df[col])

        counts = {r: sum(v == r for v in self.column_roles.values())
                  for r in ("numeric", "categorical", "datetime",
                            "high_cardinality", "drop")}
        logger.info(
            f"  Type inference: {counts['numeric']} numeric, "
            f"{counts['categorical']} categorical, "
            f"{counts['datetime']} datetime, "
            f"{counts['high_cardinality']} high-cardinality, "
            f"{counts['drop']} dropped."
        )
        return self

    def _match_override(self, col: str, overrides: Dict[str, str]) -> Optional[str]:
        for pattern, role in overrides.items():
            try:
                if re.search(pattern, col):
                    return role
            except re.error:
                if pattern == col:
                    return role
        return None

    def _infer_role(self, col: str, series: pd.Series) -> str:
        """Heuristic column role assignment."""
        # Explicit datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Numeric dtype
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            n_unique = series.nunique(dropna=True)
            # Low-cardinality integer → treat as categorical
            if pd.api.types.is_integer_dtype(series) and n_unique <= self.cfg.max_cardinality_for_categorical:
                return "categorical"
            # Check column name patterns for datetime
            col_lower = col.lower()
            if any(p in col_lower for p in self.cfg.datetime_column_name_patterns):
                return "datetime"
            return "numeric"

        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return "categorical"

        # Object / string
        if series.dtype == object or hasattr(series, "cat"):
            n_unique = series.nunique(dropna=True)
            if n_unique <= self.cfg.max_cardinality_for_categorical:
                return "categorical"
            # Try parsing as numeric
            try:
                pd.to_numeric(series.dropna(), errors="raise")
                return "numeric"
            except (ValueError, TypeError):
                pass
            # Try parsing as datetime
            try:
                pd.to_datetime(series.dropna().iloc[:100], errors="raise")
                return "datetime"
            except Exception:
                pass
            # High-cardinality string → target-encode instead of dropping.
            # TargetEncoder converts each level to the mean of the target,
            # computed exclusively on the training fold (leakage-safe via CV).
            logger.debug(
                f"  High-cardinality string column detected "
                f"({n_unique} unique values) — role='high_cardinality'."
            )
            return "high_cardinality"

        return "drop"

    def columns_of_role(self, role: str) -> List[str]:
        return [c for c, r in self.column_roles.items() if r == role]

    def override(self, column: str, role: str) -> "TypeInferenceEngine":
        self.column_roles[column] = role
        return self


# =============================================================================
# 3. CUSTOM TRANSFORMERS
# =============================================================================

class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered Log-Ratio transform for compositional/simplex data. Stateless."""

    def __init__(self, pseudocount: float = 1e-6) -> None:
        self.pseudocount = pseudocount

    def fit(self, X, y=None) -> "CLRTransformer":
        return self

    def transform(self, X) -> np.ndarray:
        X_arr = np.clip(np.asarray(X, dtype=float), 0.0, None) + self.pseudocount
        log_X = np.log(X_arr)
        return log_X - log_X.mean(axis=1, keepdims=True)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.array([f"clr__{f}" for f in input_features])
        return np.array([])


class ILRTransformer(BaseEstimator, TransformerMixin):
    """Isometric Log-Ratio (Helmert basis). D parts → D-1 orthonormal coordinates."""

    def __init__(self, pseudocount: float = 1e-6) -> None:
        self.pseudocount = pseudocount
        self._basis: Optional[np.ndarray] = None

    @staticmethod
    def _helmert_basis(D: int) -> np.ndarray:
        H = np.zeros((D - 1, D))
        for i in range(D - 1):
            n = i + 1
            H[i, :n] = 1.0 / n
            H[i, n] = -1.0
        return H / np.linalg.norm(H, axis=1, keepdims=True)

    def fit(self, X, y=None) -> "ILRTransformer":
        self._basis = self._helmert_basis(np.asarray(X).shape[1])
        return self

    def transform(self, X) -> np.ndarray:
        if self._basis is None:
            raise RuntimeError("ILRTransformer not fitted.")
        X_arr = np.clip(np.asarray(X, dtype=float), 0.0, None) + self.pseudocount
        return (self._basis @ np.log(X_arr).T).T

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self._basis is None:
            return np.array([])
        return np.array([f"ilr_{i}" for i in range(self._basis.shape[0])])


class DatetimeExpander(BaseEstimator, TransformerMixin):
    """
    Expands datetime columns into numeric parts.
    Accepts pd.DataFrame input (used as the transformer for datetime columns
    inside a ColumnTransformer with the column list passed explicitly).
    """

    _PARTS = ("year", "month", "day", "dayofweek",
               "hour", "minute", "second", "dayofyear", "quarter", "is_weekend")

    def __init__(self) -> None:
        self._columns: List[str] = []

    def fit(self, X, y=None) -> "DatetimeExpander":
        if isinstance(X, pd.DataFrame):
            self._columns = list(X.columns)
        else:
            self._columns = [f"col_{i}" for i in range(np.asarray(X).shape[1])]
        return self

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self._columns)
        parts = []
        for col in self._columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            parts.extend([
                dt.dt.year.fillna(0).astype(float).values,
                dt.dt.month.fillna(0).astype(float).values,
                dt.dt.day.fillna(0).astype(float).values,
                dt.dt.dayofweek.fillna(0).astype(float).values,
                dt.dt.hour.fillna(0).astype(float).values,
                dt.dt.minute.fillna(0).astype(float).values,
                dt.dt.second.fillna(0).astype(float).values,
                dt.dt.dayofyear.fillna(0).astype(float).values,
                dt.dt.quarter.fillna(0).astype(float).values,
                (dt.dt.dayofweek >= 5).astype(float).values,
            ])
        return np.column_stack(parts) if parts else np.zeros((len(df), 0))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        cols = input_features if input_features is not None else self._columns
        names = []
        for col in cols:
            for part in self._PARTS:
                names.append(f"{col}__{part}")
        return np.array(names)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Removes one column from each pair with |Pearson r| >= threshold.
    Fitted on training data only — leakage-free.
    """

    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold
        self._keep_idx: Optional[np.ndarray] = None
        self._n_input: int = 0

    def fit(self, X, y=None) -> "CorrelationFilter":
        X_arr = np.asarray(X, dtype=float)
        self._n_input = X_arr.shape[1]
        if self._n_input < 2:
            self._keep_idx = np.arange(self._n_input)
            return self
        df_tmp = pd.DataFrame(X_arr)               # NO fillna: corr() is
                                                   # pairwise-complete already
        stds = df_tmp.std(axis=0, ddof=0).to_numpy()
        # Tolerance-based zero/near-zero variance detection (no exact ==)
        zero_var = np.where(~np.isfinite(stds) | (stds < 1e-12))[0]
        drop: set = set(int(i) for i in zero_var)
        if drop:
            logger.debug(
                f"  CorrelationFilter: {len(drop)} near-zero-variance "
                f"column(s) dropped before correlation check."
            )

        corr = df_tmp.corr(method="pearson", min_periods=2).abs().to_numpy()
        corr = np.nan_to_num(corr, nan=0.0)        # NaN ⇒ "not correlated"

        n = corr.shape[0]
        for j in range(n):
            if j in drop:
                continue
            for i in range(j + 1, n):
                if i not in drop and corr[j, i] >= self.threshold:
                    drop.add(i)
        self._keep_idx = np.array([i for i in range(n) if i not in drop])
        logger.debug(
            f"  CorrelationFilter: keeping {len(self._keep_idx)}/{n} features."
        )
        return self


    def transform(self, X) -> np.ndarray:
        if self._keep_idx is None:
            raise RuntimeError("CorrelationFilter not fitted.")
        return np.asarray(X, dtype=float)[:, self._keep_idx]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if self._keep_idx is None:
            return np.array([], dtype=int)
        if indices:
            return self._keep_idx
        mask = np.zeros(self._n_input, dtype=bool)
        if len(self._keep_idx):
            mask[self._keep_idx] = True
        return mask

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self._keep_idx is None:
            raise RuntimeError("CorrelationFilter not fitted.")
        if input_features is None:
            input_features = np.array(
                [f"x{i}" for i in range(self._n_input)], dtype=object
            )
        return np.asarray(input_features, dtype=object)[self._keep_idx]


class ToCategoryDtype(BaseEstimator, TransformerMixin):
    """
    Casts all columns to pandas 'category' dtype with the category sets
    FROZEN at fit time.

    Why freezing matters: LightGBM splits on the *integer codes* of pandas
    categoricals. A plain .astype('category') at transform time would
    produce a different code→level mapping on test data and silently
    corrupt predictions. Frozen categories guarantee identical codes;
    unseen test levels map to code -1 (NaN), which LightGBM treats as missing.
    """

    def fit(self, X, y=None) -> "ToCategoryDtype":
        Xdf = pd.DataFrame(X)
        self.columns_ = list(Xdf.columns)
        self.categories_ = {
            c: pd.Index(pd.unique(Xdf[c].astype(str))) for c in self.columns_
        }
        return self

    def transform(self, X) -> pd.DataFrame:
        Xdf = pd.DataFrame(X).copy()
        Xdf.columns = self.columns_
        for c in self.columns_:
            Xdf[c] = pd.Categorical(
                Xdf[c].astype(str), categories=self.categories_[c]
            )
        return Xdf

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            self.columns_ if input_features is None else input_features,
            dtype=object,
        )


class CatBoostNativeRegressor(BaseEstimator, RegressorMixin):
    """
    Clone-safe CatBoost wrapper with native categorical handling.

    sklearn contract: __init__ stores its arguments verbatim, so
    get_params()/set_params() round-trip exactly and sklearn.clone() works
    (the old adapter crashed clone()). Categorical columns are detected at
    fit() time from pandas dtypes — no positional index bookkeeping.
    """

    def __init__(self, catboost_params: Optional[Dict[str, Any]] = None) -> None:
        self.catboost_params = catboost_params

    @staticmethod
    def _cat_columns(X: pd.DataFrame) -> List[str]:
        return [
            c for c in X.columns
            if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object
        ]

    def _as_catboost_frame(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        # CatBoost rejects floats/NaN inside cat features → force str
        for c in getattr(self, "cat_columns_", self._cat_columns(X)):
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y, **fit_params) -> "CatBoostNativeRegressor":
        p = dict(self.catboost_params or {})
        if "bagging_temperature" in p and "subsample" in p:
            raise ValueError(
                "CatBoost: 'bagging_temperature' (Bayesian bootstrap) and "
                "'subsample' (Bernoulli/MVS) are mutually exclusive — fix the "
                "search space / instantiator."
            )
        X = pd.DataFrame(X)
        self.cat_columns_ = self._cat_columns(X)
        X = self._as_catboost_frame(X)
        self.model_ = CatBoostRegressor(**p)
        self.model_.fit(X, y, cat_features=self.cat_columns_ or None, **fit_params)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(self._as_catboost_frame(X))


class TreeFeatureSelector(BaseEstimator, TransformerMixin):
    """ExtraTrees-based feature selector. Leakage-free (fitted on train only)."""

    def __init__(
        self,
        random_state: int = 42,
        threshold: str = "median",
        n_estimators: int = 200,
        max_features: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_features = max_features
        self._selector: Optional[SelectFromModel] = None

    def fit(self, X, y) -> "TreeFeatureSelector":
        et = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state, n_jobs=-1,
        )
        self._selector = SelectFromModel(
            estimator=et, threshold=self.threshold, max_features=self.max_features,
        )
        self._selector.fit(X, y)
        return self

    def transform(self, X):
        if self._selector is None:
            raise RuntimeError("TreeFeatureSelector not fitted.")
        return self._selector.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        if self._selector is None:
            return np.array([], dtype=bool)
        return self._selector.get_support(indices=indices)


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """SelectKBest with mutual information regression."""

    def __init__(self, k: int = 30, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state
        self._selector: Optional[SelectKBest] = None

    def fit(self, X, y) -> "MutualInfoSelector":
        rs = self.random_state
        k = min(self.k, np.asarray(X).shape[1])
        self._selector = SelectKBest(
            score_func=lambda X_, y_: mutual_info_regression(X_, y_, random_state=rs),
            k=k,
        )
        self._selector.fit(X, y)
        return self

    def transform(self, X):
        if self._selector is None:
            raise RuntimeError("MutualInfoSelector not fitted.")
        return self._selector.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        if self._selector is None:
            return np.array([], dtype=bool)
        return self._selector.get_support(indices=indices)


class TargetTransformer:
    """
    Per-fold invertible target transform — fitted exclusively on training data.
    Methods: 'none' | 'log1p' | 'yeo-johnson' | 'box-cox' | 'quantile-normal'
    """

    SUPPORTED = ("none", "log1p", "yeo-johnson", "box-cox", "quantile-normal")

    def __init__(self, method: str = "none", random_state: int = 42) -> None:
        if method not in self.SUPPORTED:
            raise ValueError(f"Unknown transform '{method}'. Choose from: {self.SUPPORTED}")
        self.method = method
        self.random_state = random_state
        self._pt: Optional[PowerTransformer] = None
        self._qt: Optional[QuantileTransformer] = None
        self._shift: float = 0.0

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).copy()
        if self.method == "none":
            return y
        if self.method == "log1p":
            self._shift = max(0.0, -float(np.nanmin(y)))
            return np.log1p(y + self._shift)
        if self.method == "yeo-johnson":
            self._pt = PowerTransformer(method="yeo-johnson", standardize=False)
            return self._pt.fit_transform(y.reshape(-1, 1)).ravel()
        if self.method == "box-cox":
            # Box-Cox requires strictly positive values
            self._shift = max(0.0, -float(np.nanmin(y))) + 1e-6
            self._pt = PowerTransformer(method="box-cox", standardize=False)
            return self._pt.fit_transform((y + self._shift).reshape(-1, 1)).ravel()
        if self.method == "quantile-normal":
            self._qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, max(10, len(y))),
                random_state=self.random_state,
            )
            return self._qt.fit_transform(y.reshape(-1, 1)).ravel()
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).copy()
        if self.method == "none":
            return y
        if self.method == "log1p":
            return np.expm1(y) - self._shift
        if self.method in ("yeo-johnson", "box-cox") and self._pt is not None:
            result = self._pt.inverse_transform(y.reshape(-1, 1)).ravel()
            if self.method == "box-cox":
                result = result - self._shift
            return result
        if self.method == "quantile-normal" and self._qt is not None:
            return self._qt.inverse_transform(y.reshape(-1, 1)).ravel()
        return y

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply an already-fitted transform to new targets."""
        y = np.asarray(y, dtype=float).copy()
        if self.method == "none":
            return y
        if self.method == "log1p":
            return np.log1p(y + self._shift)
        if self.method == "yeo-johnson" and self._pt is not None:
            return self._pt.transform(y.reshape(-1, 1)).ravel()
        if self.method == "box-cox" and self._pt is not None:
            return self._pt.transform((y + self._shift).reshape(-1, 1)).ravel()
        if self.method == "quantile-normal" and self._qt is not None:
            return self._qt.transform(y.reshape(-1, 1)).ravel()
        return y

    @staticmethod
    def auto_select(y: pd.Series, tgt_cfg: TargetConfig, label_name: str) -> str:
        """Heuristically choose a target transform."""
        override = tgt_cfg.target_transforms.get(label_name)
        if override:
            logger.info(f"  Target '{label_name}': explicit transform '{override}'.")
            return override
        if not tgt_cfg.auto_transform:
            return "none"
        skewness = detect_abs_skewness(y)
        if skewness <= tgt_cfg.skew_threshold:
            return "none"
        choice = tgt_cfg.preferred_auto_transform
        # Safety: log1p needs non-negative; box-cox needs positive
        if choice == "log1p":
            if tgt_cfg.positive_targets and label_name in tgt_cfg.positive_targets:
                pass
            elif y.dropna().min() < 0:
                choice = "yeo-johnson"
        if choice == "box-cox" and y.dropna().min() <= 0:
            choice = "yeo-johnson"
        logger.info(
            f"  Target '{label_name}': auto transform '{choice}' "
            f"(|skew|={skewness:.2f})."
        )
        return choice


# =============================================================================
# 4. DATA SOURCES (PLUGGABLE)
# =============================================================================

class DataSource(ABC):
    """Abstract base. Returns a single flat DataFrame."""

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


class InMemoryDataSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def load(self) -> pd.DataFrame:
        return self._df.copy()


class FileDataSource(DataSource):
    """
    Loads CSV/TSV/Parquet/Feather/Excel/HDF5/JSON/JSONL.
    Auto-detects format from file extension when file_format is None.
    """

    SUPPORTED = frozenset({
        "csv", "tsv", "parquet", "feather", "xlsx", "xls",
        "h5", "hdf5", "json", "jsonl",
    })

    def __init__(self, config: DataSourceConfig) -> None:
        if not config.file_path:
            raise ValueError("file_path required for FileDataSource.")
        self.config = config
        self.path = Path(config.file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        ext = self.path.suffix.lstrip(".").lower()
        self.fmt = (config.file_format or ext).lower()
        if self.fmt == "tsv":
            self.fmt = "csv"
            # override separator for TSV without mutating config
            self._sep = "\t"
        else:
            self._sep = config.csv_sep
        if self.fmt not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported format '{self.fmt}'. Supported: {sorted(self.SUPPORTED)}"
            )

    def load(self) -> pd.DataFrame:
        c = self.config
        fmt = self.fmt
        logger.info(f"Loading {fmt.upper()}: {self.path}")
        if fmt == "csv":
            return pd.read_csv(self.path, sep=self._sep, encoding=c.csv_encoding)
        if fmt == "parquet":
            return pd.read_parquet(self.path)
        if fmt == "feather":
            return pd.read_feather(self.path)
        if fmt in ("xlsx", "xls"):
            return pd.read_excel(self.path, sheet_name=c.excel_sheet)
        if fmt in ("h5", "hdf5"):
            return pd.read_hdf(self.path, key=c.hdf5_key)
        if fmt == "json":
            return pd.read_json(self.path, orient=c.json_orient)
        if fmt == "jsonl":
            return pd.read_json(self.path, lines=True)
        raise RuntimeError(f"Unhandled format: {fmt}")


class SQLiteDataSource(DataSource):
    def __init__(self, config: DataSourceConfig) -> None:
        if not config.sqlite_path:
            raise ValueError("sqlite_path required.")
        if not config.sql_query:
            raise ValueError("sql_query required.")
        if not Path(config.sqlite_path).exists():
            raise FileNotFoundError(f"SQLite DB not found: {config.sqlite_path}")
        self.config = config

    def load(self) -> pd.DataFrame:
        logger.info(f"Querying SQLite: {self.config.sqlite_path}")
        with sqlite3.connect(self.config.sqlite_path) as conn:
            return pd.read_sql_query(self.config.sql_query, conn)


class SQLAlchemyDataSource(DataSource):
    def __init__(self, config: DataSourceConfig) -> None:
        if not HAS_SQLALCHEMY:
            raise ImportError("Install sqlalchemy: pip install sqlalchemy")
        if not config.sqlalchemy_uri:
            raise ValueError("sqlalchemy_uri required.")
        if not config.sql_query:
            raise ValueError("sql_query required.")
        self.config = config

    def load(self) -> pd.DataFrame:
        logger.info(f"Querying via SQLAlchemy: {self.config.sqlalchemy_uri}")
        engine = create_engine(self.config.sqlalchemy_uri)
        with engine.connect() as conn:
            return pd.read_sql_query(sa_text(self.config.sql_query), conn)


def make_data_source(config: DataSourceConfig) -> DataSource:
    """Factory: select the right DataSource from a DataSourceConfig."""
    if config.sqlite_path:
        return SQLiteDataSource(config)
    if config.sqlalchemy_uri:
        return SQLAlchemyDataSource(config)
    if config.file_path:
        return FileDataSource(config)
    raise ValueError("No valid data source specified in DataSourceConfig.")


# =============================================================================
# 5. FEATURE ENGINEERS & LABEL EXTRACTORS (PLUGGABLE ABCs)
# =============================================================================

class FeatureEngineer(ABC):
    """Abstract feature engineer. Returns a cleaned feature DataFrame."""
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    @abstractmethod
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame: ...


class PassthroughEngineer(FeatureEngineer):
    """Default: pass the raw DataFrame through (sanitize column names only)."""

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        return sanitize_dataframe_columns(df)


class LabelExtractor(ABC):
    """Abstract label extractor. Returns a DataFrame of target columns only."""
    def __init__(self, config: TargetConfig) -> None:
        self.config = config

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame: ...


class ColumnLabelExtractor(LabelExtractor):
    """Extract targets already present as columns in the (sanitized) DataFrame."""

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.target_columns:
            raise ValueError("TargetConfig.target_columns must be non-empty.")
        target_san = [sanitize_name(c) for c in self.config.target_columns]
        missing = [c for c in target_san if c not in df.columns]
        if missing:
            raise ValueError(
                f"Target columns not found after sanitisation: {missing}\n"
                f"Available columns: {list(df.columns)[:30]}"
            )
        return df[target_san].copy()


# =============================================================================
# 6. DATASET (container) + DATASET BUILDER
# =============================================================================

@dataclass
class Dataset:
    """Container for all ML-ready data produced by DatasetBuilder."""
    df_features: pd.DataFrame
    df_targets: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    high_cardinality_columns: List[str] = field(default_factory=list)   # target-encoded HC strings
    groups: Optional[pd.Series] = None
    sample_weights: Optional[pd.Series] = None


class DatasetBuilder:
    """
    Composes DataSource + FeatureEngineer + LabelExtractor → Dataset.

    Column-type inference is performed here (once) and stored in the Dataset
    so that PipelineFactory can build per-type preprocessing without repeating work.
    """

    def __init__(
        self,
        data_source: DataSource,
        feat_config: FeatureConfig,
        tgt_config: TargetConfig,
        ml_config: MLConfig,
        feature_engineer: Optional[FeatureEngineer] = None,
        label_extractor: Optional[LabelExtractor] = None,
    ) -> None:
        self.ds = data_source
        self.fc = feat_config
        self.tc = tgt_config
        self.mc = ml_config
        self.feature_engineer = feature_engineer or PassthroughEngineer(feat_config)
        self.label_extractor  = label_extractor  or ColumnLabelExtractor(tgt_config)

    def build(self) -> Dataset:
        logger.info("Building dataset...")
        raw_df = self.ds.load()
        df = self.feature_engineer.engineer(raw_df)

        # Extract targets
        df_targets = self.label_extractor.extract(df)
        target_cols = list(df_targets.columns)

        # Determine columns to completely exclude from features
        exclude = set(target_cols)
        if self.mc.group_column:
            exclude.add(sanitize_name(self.mc.group_column))
        if self.mc.sample_weight_column:
            exclude.add(sanitize_name(self.mc.sample_weight_column))
        for c in self.mc.drop_columns:
            exclude.add(sanitize_name(c))

        # Extract groups
        groups: Optional[pd.Series] = None
        gcol = sanitize_name(self.mc.group_column) if self.mc.group_column else None
        if gcol and gcol in df.columns:
            groups = df[gcol].copy().reset_index(drop=True)
            logger.info(f"  Group column '{gcol}': {groups.nunique()} unique groups.")
        elif gcol:
            logger.warning(f"  Group column '{self.mc.group_column}' not found — disabling group CV.")

        # Extract sample weights
        sample_weights: Optional[pd.Series] = None
        wcol = sanitize_name(self.mc.sample_weight_column) if self.mc.sample_weight_column else None
        if wcol and wcol in df.columns:
            sample_weights = pd.to_numeric(df[wcol], errors="coerce").reset_index(drop=True)
            logger.info(f"  Sample weight column '{wcol}' attached.")
        elif wcol:
            logger.warning(f"  Weight column '{self.mc.sample_weight_column}' not found — ignoring.")

        # Infer column types for the feature columns
        type_engine = TypeInferenceEngine(self.fc)
        type_engine.fit(df, exclude_columns=list(exclude))

        num_cols  = type_engine.columns_of_role("numeric")
        cat_cols  = type_engine.columns_of_role("categorical")
        dt_cols   = type_engine.columns_of_role("datetime")
        hc_cols   = type_engine.columns_of_role("high_cardinality")
        feature_cols = num_cols + cat_cols + dt_cols + hc_cols

        df_features = df[feature_cols].reset_index(drop=True)
        df_targets = df_targets.reset_index(drop=True)

        logger.info(
            f"Dataset ready: {len(df_features)} samples × {len(feature_cols)} features "
            f"(numeric={len(num_cols)}, categorical={len(cat_cols)}, "
            f"datetime={len(dt_cols)}, high_cardinality={len(hc_cols)}) "
            f"× {len(target_cols)} targets"
            + (f" × {groups.nunique()} groups" if groups is not None else "")
            + "."
        )
        return Dataset(
            df_features=df_features,
            df_targets=df_targets,
            feature_columns=feature_cols,
            target_columns=target_cols,
            numeric_columns=num_cols,
            categorical_columns=cat_cols,
            datetime_columns=dt_cols,
            high_cardinality_columns=hc_cols,
            groups=groups,
            sample_weights=sample_weights,
        )


# =============================================================================
# 7. MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Central model registry. Supports user-registered custom models via .register().
    """

    LINEAR_MODELS  = frozenset({"ridge", "lasso", "elasticnet", "bayesian_ridge",
                                 "huber", "linear_svr", "svr", "knn"})
    TREE_MODELS    = frozenset({"random_forest", "extra_trees", "hist_gbrt",
                                 "xgboost", "lightgbm", "catboost"})
    SPECIAL_MODELS = frozenset({"dummy", "tabpfn"})

    def __init__(self, random_state: int = 42, n_jobs: int = -1) -> None:
        self.rs = random_state
        self.nj = n_jobs
        self._registry: Dict[str, Type[BaseEstimator]] = {}
        self._instantiators: Dict[str, Callable[[Dict[str, Any]], BaseEstimator]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for name, cls in [
            ("dummy",          DummyRegressor),
            ("ridge",          Ridge),
            ("lasso",          Lasso),
            ("elasticnet",     ElasticNet),
            ("bayesian_ridge", BayesianRidge),
            ("huber",          HuberRegressor),
            ("linear_svr",     LinearSVR),
            ("svr",            SVR),
            ("knn",            KNeighborsRegressor),
            ("random_forest",  RandomForestRegressor),
            ("extra_trees",    ExtraTreesRegressor),
            ("hist_gbrt",      HistGradientBoostingRegressor),
        ]:
            self.register(name, cls)
        if HAS_XGB:
            self.register("xgboost", xgb.XGBRegressor)
            logger.info("XGBoost registered.")
        if HAS_LGBM:
            self.register("lightgbm", lgb.LGBMRegressor)
            logger.info("LightGBM registered.")
        if HAS_CATBOOST:
            self.register("catboost", CatBoostRegressor)
            logger.info("CatBoost registered.")
        if HAS_TABPFN:
            self.register("tabpfn", TabPFNRegressor)
            logger.info("TabPFN registered.")

    def register(
        self,
        name: str,
        cls: Type[BaseEstimator],
        instantiator: Optional[Callable[[Dict[str, Any]], BaseEstimator]] = None,
    ) -> None:
        """Register a model. Supply a custom `instantiator(params)` for non-standard constructors."""
        self._registry[name] = cls
        if instantiator is not None:
            self._instantiators[name] = instantiator

    def instantiate(self, name: str, params: Dict[str, Any]) -> BaseEstimator:
        if name in self._instantiators:
            return self._instantiators[name](params)
        cls = self._registry[name]
        rs, nj = self.rs, self.nj
        if name == "dummy":
            return cls(strategy="mean")
        if name in ("huber", "knn"):
            return cls(**params)
        if name in ("svr", "linear_svr"):
            return cls(**params)
        if name in ("ridge", "lasso", "elasticnet"):
            return cls(**params, random_state=rs)
        if name == "bayesian_ridge":
            return cls(**params)
        if name in ("random_forest", "extra_trees"):
            return cls(**params, random_state=rs, n_jobs=nj)
        if name == "hist_gbrt":
            return cls(**params, random_state=rs)
        if HAS_XGB and name == "xgboost":
            return xgb.XGBRegressor(
                **params, random_state=rs, n_jobs=nj,
                objective="reg:squarederror", tree_method="hist", verbosity=0,
            )
        if HAS_LGBM and name == "lightgbm":
            return lgb.LGBMRegressor(**params, random_state=rs, n_jobs=nj, verbose=-1)
        if HAS_CATBOOST and name == "catboost":
            return CatBoostRegressor(**params, random_seed=rs, silent=True)
        if HAS_TABPFN and name == "tabpfn":
            return cls()
        # Generic fallback
        try:
            return cls(**params)
        except TypeError:
            return cls()

    def available_names(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        names = list(self._registry.keys())
        if include:
            names = [n for n in names if n in include]
        if exclude:
            names = [n for n in names if n not in exclude]
        return names


# =============================================================================
# 8. PREPROCESSOR BUILDER + PIPELINE FACTORY
# =============================================================================

class PreprocessorBuilder:
    """
    Builds a leakage-safe ColumnTransformer for a given dataset's column types.

    - Numeric:     impute (median) → optional variance filter → optional scale
    - Categorical: impute (most_frequent) → OneHot (linear) or Ordinal (tree)
    - Datetime:    DatetimeExpander → optional scale
    - CLR:         impute → CLRTransformer → StandardScaler  (opt-in)
    """

    def __init__(
        self,
        feat_config: FeatureConfig,
        random_state: int = 42,
        for_native_cat: bool = False,
    ) -> None:
        self.fc = feat_config
        self.rs = random_state
        self.for_native_cat = for_native_cat

    def _make_scaler(self, strategy: str):
        return {
            "standard": StandardScaler(),
            "robust":   RobustScaler(),
            "quantile": QuantileTransformer(
                output_distribution="normal", random_state=self.rs
            ),
            "none": None,
        }.get(strategy, StandardScaler())

    def build(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        datetime_cols: List[str],
        with_scale: bool,
        for_linear: bool,
        clr_cols: Optional[List[str]] = None,
        high_cardinality_cols: Optional[List[str]] = None,
    ) -> ColumnTransformer:
        clr_cols = clr_cols or []
        plain_numeric = [c for c in numeric_cols if c not in clr_cols]
        transformers: list = []

        # ── Numeric ──────────────────────────────────────────────────────────
        if plain_numeric:
            steps: list = [("imputer", SimpleImputer(strategy=self.fc.numeric_imputer_strategy))]
            if self.fc.use_variance_threshold:
                steps.append(("var_thresh", VarianceThreshold(threshold=self.fc.variance_threshold)))
            if with_scale:
                scaler = self._make_scaler(self.fc.numeric_scaler)
                if scaler is not None:
                    steps.append(("scaler", scaler))
            if self.fc.use_correlation_filter:
                steps.append(("corr_filter", CorrelationFilter(self.fc.correlation_threshold)))
            transformers.append(("num", Pipeline(steps), plain_numeric))

        # ── Categorical ───────────────────────────────────────────────────────
        if categorical_cols:
            cat_steps: list = [
                ("imputer", SimpleImputer(strategy=self.fc.categorical_imputer_strategy)),
            ]
            if for_linear:
                # OneHotEncoder is necessary for linear models to be well-specified
                try:
                    ohe = OneHotEncoder(
                        handle_unknown=self.fc.onehot_handle_unknown,
                        max_categories=self.fc.onehot_max_categories,
                        sparse_output=False,
                    )
                except TypeError:
                    # sklearn < 1.2
                    ohe = OneHotEncoder(
                        handle_unknown=self.fc.onehot_handle_unknown,
                        sparse=False,
                    )
                cat_steps.append(("encoder", ohe))
                if with_scale:
                    cat_steps.append(("scaler", StandardScaler()))
            elif self.for_native_cat:
                # CatBoost / LightGBM: impute, then cast to FROZEN pandas
                # 'category' dtype. Categorical-ness travels with the dtype —
                # the old positional-index arithmetic broke whenever
                # VarianceThreshold / CorrelationFilter changed the numeric
                # block's output width.
                cat_steps.append(("to_category", ToCategoryDtype()))
            else:
                # Generic tree-based models (RF, XGB, etc.): Ordinal is correct
                # because they treat every split threshold numerically.
                cat_steps.append((
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ))
            transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

        # ── Datetime ─────────────────────────────────────────────────────────
        if datetime_cols and self.fc.expand_datetime:
            dt_steps: list = [("expand", DatetimeExpander())]
            if with_scale:
                dt_steps.append(("scaler", StandardScaler()))
            transformers.append(("dt", Pipeline(dt_steps), datetime_cols))

        # ── High-cardinality strings (Target Encoding) ────────────────────────
        # TargetEncoder (sklearn ≥ 1.3) uses cross-fitting internally to avoid
        # within-fold leakage.  For older sklearn we fall back to OrdinalEncoder
        # so the pipeline still runs — just without the quality boost.
        hc_cols = [c for c in (high_cardinality_cols or []) if c not in clr_cols]
        if hc_cols:
            if HAS_TARGET_ENCODER:
                hc_encoder = TargetEncoder(
                    target_type="continuous",
                    smooth="auto",           # empirical Bayes shrinkage
                    cv=5,
                )
            else:
                logger.warning(
                    "scikit-learn < 1.3: TargetEncoder unavailable. "
                    "High-cardinality columns will use OrdinalEncoder as fallback. "
                    "Upgrade to sklearn ≥ 1.3 for proper target encoding."
                )
                hc_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            hc_steps: list = [
                ("imputer", SimpleImputer(
                    strategy=self.fc.categorical_imputer_strategy
                )),
                ("encoder", hc_encoder),
            ]
            if with_scale:
                hc_steps.append(("scaler", StandardScaler()))
            transformers.append(("hc", Pipeline(hc_steps), hc_cols))

        # ── CLR (opt-in) ──────────────────────────────────────────────────────
        if clr_cols:
            clr_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=self.fc.clr_pseudocount)),
                ("clr",     CLRTransformer(pseudocount=self.fc.clr_pseudocount)),
                ("scaler",  StandardScaler()),
            ])
            transformers.append(("clr", clr_pipe, clr_cols))

        if not transformers:
            # Safety fallback
            hc_cols_fb = [c for c in (high_cardinality_cols or []) if c not in clr_cols]
            all_cols = plain_numeric + categorical_cols + datetime_cols + hc_cols_fb
            if all_cols:
                transformers = [("passthrough", SimpleImputer(strategy="median"), all_cols)]
            else:
                raise ValueError("No usable feature columns found.")

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        if self.for_native_cat:
            # DataFrame output preserves dtypes so LightGBM ('auto') and
            # CatBoostNativeRegressor detect categorical columns natively.
            ct.set_output(transform="pandas")
        return ct


class PipelineFactory:
    """Builds leakage-safe sklearn Pipelines per (model, feature_strategy)."""

    _NATIVE_CAT_MODELS: frozenset = frozenset(
        (["lightgbm"] if HAS_LGBM else []) +
        (["catboost"] if HAS_CATBOOST else [])
    )
    # Strategies that feed numeric-only selectors downstream — native cat
    # handling (category dtype columns) must be disabled for these.
    _SELECTOR_STRATEGIES: frozenset = frozenset(
        {"tree_select", "selectkbest", "selectkbest_mi", "pca"}
    )

    def __init__(self, feat_config: FeatureConfig, random_state: int = 42) -> None:
        self.fc = feat_config
        self.rs = random_state

    @staticmethod
    def strategies_for_model(model_name: str) -> List[str]:
        """Smart default feature strategy list per model family."""
        if model_name in ModelRegistry.SPECIAL_MODELS:
            return ["full"]
        if model_name in ModelRegistry.LINEAR_MODELS:
            return ["full_scaled", "full_robust", "selectkbest", "selectkbest_mi", "pca"]
        if model_name in ModelRegistry.TREE_MODELS:
            return ["full", "tree_select"]
        return ["full"]

    def build(
        self,
        model_name: str,
        model: BaseEstimator,
        feature_strategy: str,
        ds: "Dataset",
    ) -> Pipeline:
        is_linear  = model_name in ModelRegistry.LINEAR_MODELS
        needs_scale = is_linear or feature_strategy in (
            "full_scaled", "full_robust", "pca", "selectkbest", "selectkbest_mi",
        )
        use_robust = (feature_strategy == "full_robust")

        # Never mutate self.fc — take a local deep copy for this build only.
        # This is safe under any parallelism (Joblib, Optuna parallel trials)
        # and immune to exceptions leaving shared state corrupted.
        local_fc = copy.deepcopy(self.fc)
        if use_robust:
            local_fc.numeric_scaler = "robust"         # mutates only the local copy

        # CLR columns (opt-in)
        clr_cols: List[str] = []
        if local_fc.apply_clr_transform and local_fc.clr_column_pattern:
            try:
                pat = re.compile(local_fc.clr_column_pattern)
                clr_cols = [c for c in ds.numeric_columns if pat.search(c)]
            except re.error:
                logger.warning(f"Invalid clr_column_pattern: '{local_fc.clr_column_pattern}'")

        # Native-cat output contains raw category columns. Downstream numeric
        # selectors (f_regression, ExtraTrees, MI, PCA) would crash on them or
        # silently invalidate column positions → native handling only for
        # selector-free strategies; otherwise fall back to OrdinalEncoder.
        for_native_cat = (
            model_name in self._NATIVE_CAT_MODELS
            and bool(ds.categorical_columns)
            and feature_strategy not in self._SELECTOR_STRATEGIES
        )

        local_preproc_builder = PreprocessorBuilder(local_fc, self.rs, for_native_cat=for_native_cat)
        preprocessor = local_preproc_builder.build(
            numeric_cols=ds.numeric_columns,
            categorical_cols=ds.categorical_columns,
            datetime_cols=ds.datetime_columns,
            with_scale=needs_scale,
            for_linear=is_linear,
            clr_cols=clr_cols,
            high_cardinality_cols=ds.high_cardinality_columns,
        )

        if for_native_cat and model_name == "catboost":
            # Clone-safe wrapper; cat columns detected from dtypes at fit time.
            model = CatBoostNativeRegressor(catboost_params=model.get_params())
        # LightGBM needs no wrapper: with DataFrame input, its default
        # categorical_feature='auto' uses 'category'-dtype columns natively.

        steps: list = [("preprocessor", preprocessor)]

        # ── Feature selection / reduction ─────────────────────────────────────
        if feature_strategy == "tree_select":
            steps.append(("select", TreeFeatureSelector(
                random_state=self.rs,
                threshold=self.fc.tree_selector_threshold,
                n_estimators=self.fc.tree_selector_n_estimators,
            )))
        elif feature_strategy == "selectkbest":
            steps.append(("select", SelectKBest(
                score_func=f_regression,
                k=self.fc.selectkbest_k,
            )))
        elif feature_strategy == "selectkbest_mi":
            steps.append(("select", MutualInfoSelector(
                k=self.fc.selectkbest_k, random_state=self.rs,
            )))
        elif feature_strategy == "pca" and is_linear:
            steps.append(("pca", PCA(n_components=self.fc.pca_variance_ratio)))

        steps.append(("model", model))
        return Pipeline(steps)


# =============================================================================
# 9. CONFORMAL PREDICTOR
# =============================================================================

class ConformalPredictor:
    """
    Split conformal prediction intervals.
    Marginal coverage guarantee: P(y ∈ [ŷ - q̂, ŷ + q̂]) ≥ 1 - α
    for exchangeable data (finite-sample corrected quantile level).

    NOTE: For rigorous coverage, the calibration data must be isolated
    BEFORE any HPO/fitting on the same fold's training data. This is
    handled in NestedCVRunner._proper_cal_split().
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self._q_hat: Optional[float] = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        n = len(residuals)
        if (n + 1) * (1 - self.alpha) > n:
            logger.warning(
                f"Conformal: n_cal={n} too small for alpha={self.alpha} "
                f"(need n \u2265 {int(np.ceil((1 - self.alpha) / self.alpha))}); "
                "coverage guarantee does not hold."
            )
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self._q_hat = float(np.quantile(residuals, level, method="higher"))
        return self._q_hat

    def predict_interval(
        self, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._q_hat is None:
            raise RuntimeError("ConformalPredictor must be calibrated first.")
        p = np.asarray(y_pred, dtype=float)
        return p - self._q_hat, p + self._q_hat

    @property
    def interval_width(self) -> Optional[float]:
        return None if self._q_hat is None else 2 * self._q_hat


# =============================================================================
# 10. FEATURE STABILITY TRACKER
# =============================================================================

class FeatureStabilityTracker:
    """
    Records which features survive selection across outer folds.

    Memory-efficient design: instead of storing one dictionary per
    (feature, fold, model, strategy) record, we maintain a rolling mean
    incrementally.  Peak memory is O(unique_features × unique_groups)
    rather than O(features × folds × strategies × labels).
    """

    def __init__(self) -> None:
        # Key: (label, model_name, strategy, feature_name)
        # Value: [running_sum, running_count]
        self._stats: Dict[Tuple[str, str, str, str], List[float]] = {}

    def record(
        self,
        feature_names: List[str],
        selected_mask: np.ndarray,
        fold_idx: int,          # kept for API compatibility; not stored
        label: str,
        model_name: str,
        strategy: str,
    ) -> None:
        n = min(len(feature_names), len(selected_mask))
        for i in range(n):
            key = (label, model_name, strategy, feature_names[i])
            entry = self._stats.get(key)
            if entry is None:
                self._stats[key] = [float(selected_mask[i]), 1.0]
            else:
                entry[0] += float(selected_mask[i])
                entry[1] += 1.0

    def summary(self) -> pd.DataFrame:
        if not self._stats:
            return pd.DataFrame()
        rows = [
            {
                "label":           key[0],
                "model":           key[1],
                "strategy":        key[2],
                "feature":         key[3],
                "stability_score": val[0] / val[1],
            }
            for key, val in self._stats.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values(["label", "stability_score"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def save(self, output_dir: Path) -> None:
        df = self.summary()
        if not df.empty:
            df.to_csv(output_dir / "feature_stability.csv", index=False)
            logger.info("Feature stability report saved.")



# =============================================================================
# 11. OPTUNA TUNER (inner CV)
# =============================================================================

class OptunaTuner:
    """Bayesian HPO via Optuna TPE. Each trial builds a fresh pipeline."""

    def __init__(self, ml_config: MLConfig) -> None:
        self.cfg = ml_config

    def _suggest(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        if model_name == "dummy":
            return {}
        if model_name == "ridge":
            return {"alpha": trial.suggest_float("alpha", 1e-5, 1e4, log=True)}
        if model_name == "lasso":
            return {"alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True), "max_iter": 10_000}
        if model_name == "elasticnet":
            return {
                "alpha":    trial.suggest_float("alpha",    1e-5, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0,  1.0),
                "max_iter": 10_000,
            }
        if model_name == "bayesian_ridge":
            return {
                "alpha_1":  trial.suggest_float("alpha_1",  1e-7, 1e-3, log=True),
                "alpha_2":  trial.suggest_float("alpha_2",  1e-7, 1e-3, log=True),
                "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-3, log=True),
                "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-3, log=True),
            }
        if model_name == "huber":
            return {
                "epsilon": trial.suggest_float("epsilon", 1.01, 5.0),
                "alpha":   trial.suggest_float("alpha",   1e-5, 10.0, log=True),
            }
        if model_name == "linear_svr":
            return {
                "C":       trial.suggest_float("C",       1e-3, 1e3, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
                "max_iter": 5_000,
            }
        if model_name == "svr":
            return {
                "C":       trial.suggest_float("C",       1e-3, 1e3, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
                "kernel":  trial.suggest_categorical("kernel", ["rbf", "poly"]),
                "gamma":   trial.suggest_categorical("gamma",  ["scale", "auto"]),
            }
        if model_name == "knn":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
                "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p":           trial.suggest_int("p", 1, 2),
            }
        if model_name in ("random_forest", "extra_trees"):
            return {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth":         trial.suggest_int("max_depth",    3,   30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  1, 10),
                "max_features":      trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
            }
        if model_name == "hist_gbrt":
            return {
                "learning_rate":     trial.suggest_float("learning_rate",   5e-3, 0.3, log=True),
                "max_depth":         trial.suggest_int("max_depth",         3,   20),
                "max_iter":          trial.suggest_int("max_iter",          100, 1000, step=100),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  5,   50),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
                "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes",    15,  255),
            }
        if HAS_XGB and model_name == "xgboost":
            return {
                "n_estimators":     trial.suggest_int("n_estimators",   100, 2000, step=100),
                "max_depth":        trial.suggest_int("max_depth",      2,   10),
                "learning_rate":    trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample",    0.5,  1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "reg_alpha":        trial.suggest_float("reg_alpha",   1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda",  1e-8, 10.0, log=True),
                "gamma":            trial.suggest_float("gamma",       1e-8, 5.0,  log=True),
            }
        if HAS_LGBM and model_name == "lightgbm":
            return {
                "n_estimators":      trial.suggest_int("n_estimators",   100, 2000, step=100),
                "learning_rate":     trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                "max_depth":         trial.suggest_int("max_depth",      3,   15),
                "num_leaves":        trial.suggest_int("num_leaves",     15,  255),
                "subsample":         trial.suggest_float("subsample",    0.5,  1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha",   1e-8, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda",  1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
        if HAS_CATBOOST and model_name == "catboost":
            params = {
                "iterations":    trial.suggest_int("iterations",    100, 2000, step=100),
                "depth":         trial.suggest_int("depth",         3,   10),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-8, 20.0, log=True),
                # subsample and bagging_temperature are MUTUALLY EXCLUSIVE in
                # CatBoost: bagging_temperature requires bootstrap_type='Bayesian',
                # subsample requires Bernoulli/MVS. Suggesting both (the old code)
                # raised CatBoostError on every fit → all trials returned inf.
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
            }
            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                )
            else:
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            return params
        return {}

    def optimize(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: Optional[pd.Series],
        registry: ModelRegistry,
        feature_strategy: str,
        ds: Dataset,
        pipeline_factory: PipelineFactory,
        study_name: str,
    ) -> Tuple[Pipeline, Dict[str, Any], float]:
        """
        Inner-CV Optuna study.
        Each trial builds a FRESH pipeline — no fitted-state reuse.
        Returns (best_pipeline_fitted_on_full_X_train, best_params, best_cv_rmse).
        """
        n_groups = groups_train.nunique() if groups_train is not None else 0
        use_grp = (
            self.cfg.use_group_cv
            and groups_train is not None
            and n_groups >= self.cfg.n_inner_splits
        )
        n_splits = max(2, min(self.cfg.n_inner_splits, len(y_train) // 3))
        if use_grp:
            inner_cv = GroupKFold(n_splits=n_splits)
            splits = list(inner_cv.split(X_train, y_train, groups_train))
        else:
            inner_cv = KFold(n_splits=n_splits, shuffle=True,
                             random_state=self.cfg.random_state)
            splits = list(inner_cv.split(X_train, y_train))

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest(trial, model_name)
            rmses: List[float] = []
            for step_i, (tr, va) in enumerate(splits):
                model = registry.instantiate(model_name, params)
                pipe  = pipeline_factory.build(model_name, model, feature_strategy, ds)
                try:
                    pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
                    preds = pipe.predict(X_train.iloc[va])
                    rmses.append(float(np.sqrt(mean_squared_error(y_train.iloc[va], preds))))
                except Exception:
                    return float("inf")
                if self.cfg.optuna_use_pruning:
                    trial.report(float(np.mean(rmses)), step=step_i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            return float(np.mean(rmses))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.cfg.random_state,
                n_startup_trials=self.cfg.optuna_n_startup_trials,
            ),
            pruner=(
                optuna.pruners.MedianPruner(n_warmup_steps=1)
                if self.cfg.optuna_use_pruning else optuna.pruners.NopPruner()
            ),
            study_name=study_name,
        )
        study.optimize(
            objective,
            n_trials=self.cfg.n_optuna_trials,
            timeout=self.cfg.optuna_timeout_sec,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_model  = registry.instantiate(model_name, best_params)
        best_pipe   = pipeline_factory.build(model_name, best_model, feature_strategy, ds)
        best_pipe.fit(X_train, y_train)
        return best_pipe, best_params, float(study.best_value)


# =============================================================================
# 12. RESULT CONTAINERS
# =============================================================================

@dataclass
class FoldResult:
    fold_index:       int
    label:            str
    model_name:       str
    feature_strategy: str
    target_transform: str
    r2:               float
    mae:              float
    medae:            float
    mse:              float
    rmse:             float
    mape:             float
    n_train:          int
    n_test:           int
    best_params:      Dict[str, Any]
    y_pred:           np.ndarray
    y_test:           np.ndarray
    elapsed_sec:      float = 0.0
    conformal_coverage: Optional[float] = None
    conformal_width:    Optional[float] = None
    selected_features:  List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    label:            str
    model_name:       str
    feature_strategy: str
    target_transform: str = "none"
    fold_results:     List[FoldResult] = field(default_factory=list)

    mean_r2:   float = 0.0
    std_r2:    float = 0.0
    mean_rmse: float = 0.0
    std_rmse:  float = 0.0
    mean_mae:  float = 0.0
    std_mae:   float = 0.0
    oof_r2:    float = 0.0
    oof_rmse:  float = 0.0
    oof_mae:   float = 0.0
    oof_mape:  float = 0.0
    ci95_rmse: Tuple[float, float] = (0.0, 0.0)
    n_samples: int = 0
    mean_conformal_coverage: Optional[float] = None
    mean_conformal_width:    Optional[float] = None

    def summarize(self, n_samples: int) -> "ExperimentResult":
        if not self.fold_results:
            return self
        r2s   = np.array([f.r2   for f in self.fold_results], dtype=float)
        rmses = np.array([f.rmse for f in self.fold_results], dtype=float)
        maes  = np.array([f.mae  for f in self.fold_results], dtype=float)
        self.n_samples = n_samples
        self.mean_r2   = float(np.nanmean(r2s));   self.std_r2   = float(np.nanstd(r2s))
        self.mean_rmse = float(np.nanmean(rmses)); self.std_rmse = float(np.nanstd(rmses))
        self.mean_mae  = float(np.nanmean(maes));  self.std_mae  = float(np.nanstd(maes))
        self.ci95_rmse = bootstrap_ci(rmses, stat=np.median, random_state=0)

        all_yt = np.concatenate([f.y_test for f in self.fold_results])
        all_yp = np.concatenate([f.y_pred for f in self.fold_results])
        oof = compute_metrics(all_yt, all_yp)
        self.oof_r2, self.oof_rmse = oof["r2"], oof["rmse"]
        self.oof_mae, self.oof_mape = oof["mae"], oof["mape"]

        cov = [f.conformal_coverage for f in self.fold_results
               if f.conformal_coverage is not None]
        wid = [f.conformal_width for f in self.fold_results
               if f.conformal_width is not None]
        if cov: self.mean_conformal_coverage = float(np.mean(cov))
        if wid: self.mean_conformal_width    = float(np.mean(wid))
        return self


# =============================================================================
# 13. RESULTS MANAGER
# =============================================================================

class ResultsManager:
    def __init__(self, output_dir: str = "ml_results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("shap", "models", "fold_predictions", "permutation"):
            (self.output_dir / sub).mkdir(exist_ok=True)
        self._results: List[ExperimentResult] = []

    def add(self, r: ExperimentResult) -> None:
        self._results.append(r)

    def all_results(self) -> List[ExperimentResult]:
        return list(self._results)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([{
            "label":                   r.label,
            "model":                   r.model_name,
            "feature_strategy":        r.feature_strategy,
            "target_transform":        r.target_transform,
            "n_samples":               r.n_samples,
            "mean_r2":                 r.mean_r2,
            "std_r2":                  r.std_r2,
            "mean_rmse":               r.mean_rmse,
            "std_rmse":                r.std_rmse,
            "mean_mae":                r.mean_mae,
            "std_mae":                 r.std_mae,
            "oof_r2":                  r.oof_r2,
            "oof_rmse":                r.oof_rmse,
            "oof_mae":                 r.oof_mae,
            "oof_mape":                r.oof_mape,
            "ci95_rmse_lo":            r.ci95_rmse[0],
            "ci95_rmse_hi":            r.ci95_rmse[1],
            "mean_conformal_coverage": r.mean_conformal_coverage,
            "mean_conformal_width":    r.mean_conformal_width,
        } for r in self._results])

    def fold_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._results:
            for f in r.fold_results:
                rows.append({
                    "label":              f.label,
                    "model":              f.model_name,
                    "feature_strategy":   f.feature_strategy,
                    "target_transform":   f.target_transform,
                    "fold":               f.fold_index,
                    "r2":                 f.r2,
                    "mae":                f.mae,
                    "rmse":               f.rmse,
                    "mape":               f.mape,
                    "n_train":            f.n_train,
                    "n_test":             f.n_test,
                    "elapsed_sec":        f.elapsed_sec,
                    "conformal_coverage": f.conformal_coverage,
                    "conformal_width":    f.conformal_width,
                    "best_params":        json.dumps(f.best_params, default=str),
                })
        return pd.DataFrame(rows)

    def save_fold_predictions(self, result: ExperimentResult) -> None:
        rows = []
        for f in result.fold_results:
            for yt, yp in zip(f.y_test, f.y_pred):
                rows.append({
                    "label": f.label, "model": f.model_name,
                    "feature_strategy": f.feature_strategy,
                    "fold": f.fold_index, "y_true": float(yt), "y_pred": float(yp),
                })
        if not rows:
            return
        fname = (
            f"preds__{sanitize_name(result.label)}__{result.model_name}"
            f"__{result.feature_strategy}.csv"
        )
        pd.DataFrame(rows).to_csv(
            self.output_dir / "fold_predictions" / fname, index=False
        )

    def save_config_snapshot(self, configs: Dict[str, Any]) -> None:
        try:
            with open(self.output_dir / "config_snapshot.json", "w") as fh:
                json.dump(configs, fh, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save config snapshot: {e}")

    def save(self) -> None:
        df      = self.to_dataframe()
        df_fold = self.fold_dataframe()
        if not df.empty:
            try:
                df.to_parquet(self.output_dir / "results_summary.parquet", index=False)
            except Exception:
                pass
            df.to_csv(self.output_dir / "results_summary.csv", index=False)
        if not df_fold.empty:
            df_fold.to_csv(self.output_dir / "fold_results.csv", index=False)
        logger.info(f"Results saved to '{self.output_dir}'.")

    def get_best_per_label(self, by: str = "oof_rmse") -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        ascending = by not in ("oof_r2", "mean_r2")
        if ascending:
            idx = df.groupby("label")[by].idxmin()
        else:
            idx = df.groupby("label")[by].idxmax()
        return df.loc[idx].reset_index(drop=True)

    def print_summary(self) -> None:
        df = self.to_dataframe()
        if df.empty:
            logger.warning("No results to display.")
            return
        sep = "=" * 120
        print(f"\n{sep}\nNESTED CV RESULTS SUMMARY\n{sep}")
        for label in df["label"].unique():
            sub = df[df["label"] == label].sort_values("oof_rmse")
            print(f"\n▶ Target: {label}  (n={sub['n_samples'].iloc[0]})")
            print(
                sub[[
                    "model", "feature_strategy", "target_transform",
                    "mean_r2", "std_r2", "mean_rmse", "std_rmse",
                    "oof_r2", "oof_rmse",
                    "ci95_rmse_lo", "ci95_rmse_hi",
                    "mean_conformal_coverage", "mean_conformal_width",
                ]].to_string(index=False, float_format="{:.4f}".format)
            )
        best = df.sort_values("oof_rmse").iloc[0]
        print(f"\n{sep}")
        print(
            f"🏆 Best overall: [{best['label']}] "
            f"{best['model']} / {best['feature_strategy']} / "
            f"{best['target_transform']} "
            f"→ OOF RMSE={best['oof_rmse']:.4f}  OOF R²={best['oof_r2']:.4f}"
        )
        print(f"{sep}\n")


# =============================================================================
# 14. SHAP COMPUTER
# =============================================================================

class SHAPComputer:
    """
    Per-fold SHAP feature importances.
    - TreeExplainer   : tree-based models
    - LinearExplainer : linear models
    - KernelExplainer : fallback for arbitrary models (slow, sampled)
    """

    def __init__(self, ml_cfg: MLConfig, output_dir: Path) -> None:
        self.cfg = ml_cfg
        self.out = output_dir

    def _get_post_preprocessor_names(
        self, pipeline: Pipeline, fallback_n: int
    ) -> List[str]:
        """Attempt to recover real feature names after the preprocessor step."""
        try:
            pre = pipeline.named_steps.get("preprocessor")
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                raw = list(pre.get_feature_names_out())
                # Strip ColumnTransformer prefixes (e.g., "num__x" → "x")
                return [re.sub(r"^[^_]+__", "", n) for n in raw]
        except Exception:
            pass
        return [f"feature_{i}" for i in range(fallback_n)]

    def compute_and_save(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        label: str,
        model_name: str,
        feature_strategy: str,
        fold_idx: int,
    ) -> None:
        if not (self.cfg.compute_shap and HAS_SHAP):
            return
        if model_name == "dummy":
            return
        try:
            fitted_model = pipeline.named_steps["model"]
            cat_cols: List[str] = getattr(fitted_model, "cat_columns_", [])
            fitted_model = getattr(fitted_model, "model_", fitted_model)  # unwrap CatBoostNativeRegressor

            pre_steps = [(n, s) for n, s in pipeline.steps if n != "model"]
            pre_pipe  = Pipeline(pre_steps) if pre_steps else None
            X_trans   = pre_pipe.transform(X_train) if pre_pipe is not None else X_train
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            is_frame = isinstance(X_trans, pd.DataFrame)
            if not is_frame:
                X_trans = np.asarray(X_trans, dtype=float)   # only safe when no cats

            n = min(self.cfg.shap_sample_size, X_trans.shape[0])
            rng = np.random.default_rng(self.cfg.random_state)
            idx = rng.choice(X_trans.shape[0], n, replace=False)
            X_sample = X_trans.iloc[idx] if is_frame else X_trans[idx]

            is_linear = model_name in ModelRegistry.LINEAR_MODELS
            is_tree   = model_name in ModelRegistry.TREE_MODELS
            try:
                if HAS_CATBOOST and isinstance(fitted_model, CatBoostRegressor) and cat_cols:
                    from catboost import Pool
                    Xs = pd.DataFrame(X_sample).copy()
                    for c in cat_cols:
                        Xs[c] = Xs[c].astype(str)
                    shap_values = fitted_model.get_feature_importance(
                        type="ShapValues", data=Pool(Xs, cat_features=cat_cols)
                    )[:, :-1]                      # last column = expected value
                elif is_tree:
                    explainer = shap.TreeExplainer(fitted_model)
                    shap_values = explainer.shap_values(X_sample)
                elif is_linear:
                    explainer = shap.LinearExplainer(
                        fitted_model, X_trans,
                        feature_perturbation="interventional",
                    )
                    shap_values = explainer.shap_values(X_sample)
                else:
                    raise ValueError("Use KernelExplainer fallback.")
            except Exception:
                if not self.cfg.shap_kernel_fallback:
                    raise
                X_bg = X_trans.iloc[:50] if is_frame else X_trans[:min(50, X_trans.shape[0])]
                bg = shap.sample(X_bg, min(50, X_bg.shape[0]))
                explainer   = shap.KernelExplainer(fitted_model.predict, bg)
                shap_values = explainer.shap_values(X_sample, nsamples=100)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs   = np.abs(shap_values).mean(axis=0)
            feat_names = self._get_post_preprocessor_names(pipeline, len(mean_abs))

            pd.DataFrame({
                "feature": feat_names[:len(mean_abs)],
                "mean_abs_shap": mean_abs,
            }).sort_values("mean_abs_shap", ascending=False).to_csv(
                self.out / "shap" /
                f"shap__{sanitize_name(label)}__{model_name}"
                f"__{feature_strategy}__fold{fold_idx}.csv",
                index=False,
            )
        except Exception as exc:
            logger.warning(
                f"SHAP failed [{model_name}/{feature_strategy}/fold{fold_idx}]: {exc}"
            )


# =============================================================================
# 15. MULTI-TASK RUNNER
# =============================================================================

class MultiTaskRunner:
    def __init__(
        self, ml_cfg: MLConfig, registry: ModelRegistry,
        feat_cfg: FeatureConfig, rm: ResultsManager,
    ) -> None:
        self.cfg = ml_cfg
        self.reg = registry
        self.fc  = feat_cfg
        self.rm  = rm

    def run(self, ds: Dataset) -> None:
        if not self.cfg.use_multitask or len(ds.target_columns) < 2:
            return
        valid_mask = ds.df_targets.notna().all(axis=1)
        n_valid = int(valid_mask.sum())
        if n_valid < self.cfg.n_outer_splits * 3:
            logger.warning(f"Multi-task skipped: only {n_valid} complete rows.")
            return
        logger.info(f"\n{'='*70}\nMULTI-TASK LEARNING (n={n_valid}, targets={len(ds.target_columns)})\n{'='*70}")

        X = ds.df_features.loc[valid_mask].reset_index(drop=True)
        Y = ds.df_targets.loc[valid_mask].reset_index(drop=True)
        g = ds.groups.loc[valid_mask].reset_index(drop=True) if ds.groups is not None else None

        use_grp = (self.cfg.use_group_cv and g is not None
                   and g.nunique() >= self.cfg.n_outer_splits)
        cv = (GroupKFold(self.cfg.n_outer_splits) if use_grp
              else KFold(self.cfg.n_outer_splits, shuffle=True,
                         random_state=self.cfg.random_state))

        # Use a simple numeric-only preprocessor (fit per fold, no leakage)
        multi_models = {
            "extra_trees_multioutput": MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=300, random_state=self.cfg.random_state, n_jobs=-1)
            ),
            "ridge_chain": RegressorChain(
                base_estimator=Ridge(random_state=self.cfg.random_state),
                order="random", random_state=self.cfg.random_state,
            ),
        }
        for name, model in multi_models.items():
            oof = np.full((n_valid, len(ds.target_columns)), np.nan)
            split_iter = cv.split(X, Y, g) if use_grp else cv.split(X, Y)
            for tr, te in split_iter:
                # Fit preprocessor on training fold only
                pre = PreprocessorBuilder(self.fc).build(
                    numeric_cols=ds.numeric_columns,
                    categorical_cols=ds.categorical_columns,
                    datetime_cols=ds.datetime_columns,
                    with_scale=True,
                    for_linear=False,
                    high_cardinality_cols=None,   # TargetEncoder needs 1-D y;
                )                                 # HC cols dropped in multitask
                Xtr = pre.fit_transform(X.iloc[tr])
                Xte = pre.transform(X.iloc[te])
                m = clone(model)
                m.fit(Xtr, Y.iloc[tr].values)
                oof[te] = m.predict(Xte)

            for j, lbl in enumerate(ds.target_columns):
                yt, yp = Y[lbl].values, oof[:, j]
                ok = np.isfinite(yt) & np.isfinite(yp)
                if ok.sum() < 5:
                    continue
                met = compute_metrics(yt[ok], yp[ok])
                res = ExperimentResult(
                    label=lbl, model_name=f"multitask_{name}",
                    feature_strategy="multitask", target_transform="none",
                )
                res.fold_results.append(FoldResult(
                    fold_index=0, label=lbl,
                    model_name=f"multitask_{name}",
                    feature_strategy="multitask", target_transform="none",
                    r2=met["r2"], mae=met["mae"], medae=met["medae"],
                    mse=met["mse"], rmse=met["rmse"], mape=met["mape"],
                    n_train=n_valid, n_test=0, best_params={},
                    y_pred=yp[ok], y_test=yt[ok],
                ))
                res.summarize(n_valid)
                self.rm.add(res)
                logger.info(f"  [{name}] {lbl}: OOF R²={res.oof_r2:.4f}  RMSE={res.oof_rmse:.4f}")


# =============================================================================
# 16. STACKING RUNNER
# =============================================================================

class StackingRunner:
    def __init__(
        self, ml_cfg: MLConfig, registry: ModelRegistry,
        pf: PipelineFactory, rm: ResultsManager,
    ) -> None:
        self.cfg = ml_cfg
        self.reg = registry
        self.pf  = pf
        self.rm  = rm

    def run(self, ds: Dataset, base_model_names: List[str]) -> None:
        if not self.cfg.use_stacking or len(base_model_names) < 2:
            return
        logger.info(f"\n{'='*70}\nSTACKING ENSEMBLE\n{'='*70}")

        for label in ds.target_columns:
            mask   = ds.df_targets[label].notna()
            n_valid = int(mask.sum())
            if n_valid < self.cfg.n_outer_splits * 4:
                continue
            X = ds.df_features.loc[mask].reset_index(drop=True)
            y = ds.df_targets.loc[mask, label].astype(float).reset_index(drop=True)
            g = ds.groups.loc[mask].reset_index(drop=True) if ds.groups is not None else None

            use_grp = (self.cfg.use_group_cv and g is not None
                       and g.nunique() >= self.cfg.n_outer_splits)
            cv = (GroupKFold(self.cfg.n_outer_splits) if use_grp
                  else KFold(self.cfg.n_outer_splits, shuffle=True,
                             random_state=self.cfg.random_state))

            oof = np.full((n_valid, len(base_model_names)), np.nan)
            for m_idx, mname in enumerate(base_model_names):
                preds = np.full(n_valid, np.nan)
                split_iter = cv.split(X, y, g) if use_grp else cv.split(X, y)
                for tr, te in split_iter:
                    try:
                        pipe = self.pf.build(mname, self.reg.instantiate(mname, {}), "full", ds)
                        pipe.fit(X.iloc[tr], y.iloc[tr])
                        preds[te] = pipe.predict(X.iloc[te])
                    except Exception as e:
                        logger.warning(f"Stacking base [{mname}] failed: {e}")
                oof[:, m_idx] = preds

            valid_rows = np.all(np.isfinite(oof), axis=1)
            if valid_rows.sum() < 10:
                logger.warning(f"Stacking skipped for '{label}': too few valid OOF rows.")
                continue

            meta = RidgeCV(cv=5)
            meta.fit(oof[valid_rows], y.values[valid_rows])
            meta_preds = meta.predict(oof[valid_rows])
            met = compute_metrics(y.values[valid_rows], meta_preds)

            res = ExperimentResult(
                label=label, model_name="stacking_ridgecv",
                feature_strategy="stacking", target_transform="none",
            )
            res.fold_results.append(FoldResult(
                fold_index=0, label=label,
                model_name="stacking_ridgecv",
                feature_strategy="stacking", target_transform="none",
                r2=met["r2"], mae=met["mae"], medae=met["medae"],
                mse=met["mse"], rmse=met["rmse"], mape=met["mape"],
                n_train=int(valid_rows.sum()), n_test=0,
                best_params={"base_models": base_model_names},
                y_pred=meta_preds, y_test=y.values[valid_rows],
            ))
            res.summarize(int(valid_rows.sum()))
            self.rm.add(res)
            logger.info(f"  [{label}]: OOF R²={res.oof_r2:.4f}  RMSE={res.oof_rmse:.4f}")


# =============================================================================
# 17. NESTED CV RUNNER (core engine)
# =============================================================================

class NestedCVRunner:
    """
    Full Nested Cross-Validation engine.

    Outer loop: GroupKFold / KFold  → unbiased generalisation estimate
    Inner loop: GroupKFold / KFold + Optuna TPE → HPO

    Per fold:
      1. Split outer training data into proper-train + calibration sets
         BEFORE any HPO (so calibration data never influences HPO).
      2. Run HPO (Optuna) on proper-train using inner CV.
      3. Strict mode (default): report best_pipe directly (no refit) and
         calibrate the ConformalPredictor on the held-out calibration set;
         the finite-sample coverage guarantee P(y ∈ [ŷ−q̂, ŷ+q̂]) ≥ 1−α holds
         exactly because cal + test residuals are exchangeable draws from the
         SAME model.
         Empirical mode: refit on the full outer training fold and calibrate;
         coverage is empirical only (no formal guarantee).
      4. Compute metrics, SHAP, permutation importance.
    """

    def __init__(
        self,
        ml_cfg:   MLConfig,
        tgt_cfg:  TargetConfig,
        feat_cfg: FeatureConfig,
        registry: ModelRegistry,
        pf:       PipelineFactory,
        tuner:    OptunaTuner,
        rm:       ResultsManager,
        shap:     SHAPComputer,
        stab:     FeatureStabilityTracker,
    ) -> None:
        self.cfg    = ml_cfg
        self.tcfg   = tgt_cfg
        self.fcfg   = feat_cfg
        self.reg    = registry
        self.pf     = pf
        self.tuner  = tuner
        self.rm     = rm
        self.shap   = shap
        self.stab   = stab

    # ── helpers ──────────────────────────────────────────────────────────────

    def _outer_cv(self, groups: Optional[pd.Series], n_samples: int):
        n_groups = groups.nunique() if groups is not None else 0
        n_splits = max(2, min(self.cfg.n_outer_splits, n_samples // 3))
        use_grp = (
            self.cfg.use_group_cv and groups is not None
            and n_groups >= n_splits
        )
        if use_grp:
            logger.info(f"  Outer GroupKFold (k={n_splits}, {n_groups} groups).")
            return GroupKFold(n_splits=n_splits), True
        logger.info(f"  Outer KFold (k={n_splits}).")
        return KFold(n_splits=n_splits, shuffle=True,
                     random_state=self.cfg.random_state), False

    def _proper_cal_split(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        g_tr: Optional[pd.Series],
        fold_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split outer-fold training data into proper-train and calibration sets.
        Calibration data is isolated BEFORE HPO — the most rigorous approach for
        split conformal prediction (following Response A's methodological insight).
        """
        n = len(y_tr)
        frac = self.cfg.conformal_calibration_fraction
        if n < 20 or frac <= 0:
            return np.arange(n), np.array([], dtype=int)

        if (self.cfg.use_group_cv and g_tr is not None
                and g_tr.nunique(dropna=True) >= 2):
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=frac, random_state=fold_seed,
            )
            proper_idx, cal_idx = next(splitter.split(X_tr, y_tr, g_tr))
        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=frac, random_state=fold_seed,
            )
            proper_idx, cal_idx = next(splitter.split(X_tr, y_tr))
        return proper_idx, cal_idx

    def _record_stability(
        self,
        pipeline: Pipeline,
        shap_computer: SHAPComputer,   # re-use its name-recovery method
        fold_idx: int, label: str,
        model_name: str, strategy: str,
    ) -> List[str]:
        if not self.cfg.track_feature_stability:
            return []
        for step_name in ("select", "corr", "pca"):
            step = pipeline.named_steps.get(step_name)
            if step is not None and hasattr(step, "get_support"):
                try:
                    mask  = step.get_support()
                    names = shap_computer._get_post_preprocessor_names(
                        pipeline, len(mask)
                    )
                    if len(mask) == len(names):
                        selected = [names[i] for i, m in enumerate(mask) if m]
                        self.stab.record(names, mask.astype(int),
                                         fold_idx, label, model_name, strategy)
                        return selected
                except Exception:
                    pass
        return []

    def _resolve_strategies(self, model_name: str) -> List[str]:
        if self.cfg.feature_strategies_override:
            return self.cfg.feature_strategies_override
        return PipelineFactory.strategies_for_model(model_name)

    # ── single experiment ─────────────────────────────────────────────────────

    def run_single(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series],
        ds: Dataset,
        label: str,
        model_name: str,
        feature_strategy: str,
    ) -> ExperimentResult:
        exp = ExperimentResult(
            label=label, model_name=model_name,
            feature_strategy=feature_strategy,
            target_transform="auto",
        )
        outer_cv, use_grp = self._outer_cv(groups, len(y))
        Xr = X.reset_index(drop=True)
        yr = y.reset_index(drop=True)
        gr = groups.reset_index(drop=True) if groups is not None else None

        split_iter = outer_cv.split(Xr, yr, gr) if use_grp else outer_cv.split(Xr, yr)

        for fold_idx, (tr_idx, te_idx) in enumerate(split_iter, start=1):
            t0 = time.perf_counter()
            X_tr_full = Xr.iloc[tr_idx]
            y_tr_full = yr.iloc[tr_idx]
            X_te      = Xr.iloc[te_idx]
            y_te      = yr.iloc[te_idx]
            g_tr_full = gr.iloc[tr_idx] if gr is not None else None

            # ── 1. Pre-isolate calibration data (before any HPO!) ─────────────
            fold_seed   = self.cfg.random_state + fold_idx
            proper_idx, cal_idx = self._proper_cal_split(
                X_tr_full, y_tr_full, g_tr_full, fold_seed
            )
            if len(cal_idx) >= 5 and len(proper_idx) >= self.cfg.n_inner_splits * 2:
                X_hpo = X_tr_full.iloc[proper_idx]
                y_hpo = y_tr_full.iloc[proper_idx]
                g_hpo = g_tr_full.iloc[proper_idx] if g_tr_full is not None else None
            else:
                X_hpo, y_hpo, g_hpo = X_tr_full, y_tr_full, g_tr_full
                cal_idx = np.array([], dtype=int)

            # ── 2. Resolve target transform using ONLY this fold's training data ──
            target_transform = TargetTransformer.auto_select(
                y_tr_full, self.tcfg, label
            )
            # Update ExperimentResult metadata on first fold (representative value)
            if fold_idx == 1:
                exp.target_transform = target_transform
            logger.info(f"    Fold {fold_idx}: target_transform='{target_transform}'")

            tt_hpo      = TargetTransformer(target_transform, random_state=self.cfg.random_state)
            y_hpo_t = pd.Series(tt_hpo.fit_transform(y_hpo.values), index=y_hpo.index)

            study_name = (
                f"{sanitize_name(label)}__{model_name}"
                f"__{feature_strategy}__fold{fold_idx}"
            )
            try:
                best_pipe, best_params, _ = self.tuner.optimize(
                    model_name=model_name,
                    X_train=X_hpo, y_train=y_hpo_t,
                    groups_train=g_hpo,
                    registry=self.reg,
                    feature_strategy=feature_strategy,
                    ds=ds,
                    pipeline_factory=self.pf,
                    study_name=study_name,
                )
            except Exception as exc:
                logger.error(f"HPO failed [{study_name}]: {exc}")
                continue

            # ── 3. Final model + transformer: ONE consistent pair ────────────
            strict_conformal = (
                self.cfg.compute_conformal
                and self.cfg.conformal_mode == "strict"
                and len(cal_idx) >= 5
            )
            if strict_conformal:
                # Strict split conformal: do NOT refit on calibration rows.
                # The reported model IS best_pipe, so fold metrics, point
                # predictions and intervals all refer to the SAME model and
                # transform, and the finite-sample guarantee
                #   P(y ∈ [ŷ − q̂, ŷ + q̂]) ≥ 1 − α
                # holds exactly.
                final_pipe  = best_pipe
                tt_eval     = tt_hpo
                n_train_eff = len(proper_idx)
            else:
                # Empirical mode: refit on the full outer training fold
                # (calibration guarantee weakened if conformal runs here).
                tt_final = TargetTransformer(target_transform,
                                             random_state=self.cfg.random_state)
                y_tr_full_t = pd.Series(
                    tt_final.fit_transform(y_tr_full.values), index=y_tr_full.index
                )
                final_model = self.reg.instantiate(model_name, best_params)
                final_pipe  = self.pf.build(model_name, final_model,
                                            feature_strategy, ds)
                final_pipe.fit(X_tr_full, y_tr_full_t)
                tt_eval     = tt_final
                n_train_eff = len(tr_idx)

            # ── 4. Predict on the outer test fold ────────────────────────────
            y_pred_t = final_pipe.predict(X_te)
            y_pred   = tt_eval.inverse_transform(np.asarray(y_pred_t, dtype=float))
            y_te_arr = y_te.to_numpy(dtype=float)
            metrics  = compute_metrics(y_te_arr, y_pred)
            elapsed  = time.perf_counter() - t0

            # ── 5. Conformal (same model + transform as the reported preds) ──
            cov, wid = None, None
            if self.cfg.compute_conformal and len(cal_idx) >= 5:
                try:
                    y_cal_true   = y_tr_full.iloc[cal_idx].to_numpy(dtype=float)
                    y_cal_pred_t = final_pipe.predict(X_tr_full.iloc[cal_idx])
                    y_cal_pred   = tt_eval.inverse_transform(
                        np.asarray(y_cal_pred_t, dtype=float)
                    )
                    cp = ConformalPredictor(alpha=self.cfg.conformal_alpha)
                    cp.calibrate(y_cal_true, y_cal_pred)
                    lo, hi = cp.predict_interval(y_pred)   # SAME y_pred as metrics
                    cov = float(np.mean((y_te_arr >= lo) & (y_te_arr <= hi)))
                    wid = cp.interval_width
                except Exception as exc:
                    logger.warning(f"Conformal failed [fold{fold_idx}]: {exc}")

            # ── 6. Feature stability ──────────────────────────────────────────
            selected = self._record_stability(
                final_pipe, self.shap, fold_idx, label, model_name, feature_strategy
            )

            exp.fold_results.append(FoldResult(
                fold_index=fold_idx, label=label,
                model_name=model_name, feature_strategy=feature_strategy,
                target_transform=target_transform,
                r2=metrics["r2"], mae=metrics["mae"], medae=metrics["medae"],
                mse=metrics["mse"], rmse=metrics["rmse"], mape=metrics["mape"],
                n_train=n_train_eff, n_test=len(te_idx),
                best_params=best_params, y_pred=y_pred, y_test=y_te_arr,
                elapsed_sec=elapsed,
                conformal_coverage=cov, conformal_width=wid,
                selected_features=selected,
            ))

            # ── 7. SHAP ───────────────────────────────────────────────────────
            self.shap.compute_and_save(
                final_pipe, X_tr_full, label, model_name, feature_strategy, fold_idx
            )

            # ── 8. Permutation importance ─────────────────────────────────────
            if self.cfg.compute_permutation_importance:
                try:
                    # final_pipe predicts in TRANSFORMED target space →
                    # score against the transformed y, not the raw y.
                    y_te_t = pd.Series(tt_eval.transform(y_te_arr), index=y_te.index)
                    pi = sk_permutation_importance(
                        final_pipe, X_te, y_te_t,
                        n_repeats=self.cfg.permutation_n_repeats,
                        random_state=self.cfg.random_state,
                        n_jobs=self.cfg.n_jobs,
                    )
                    pd.DataFrame({
                        "feature":          list(X_te.columns),
                        "importance_mean":  pi.importances_mean,
                        "importance_std":   pi.importances_std,
                    }).sort_values("importance_mean", ascending=False).to_csv(
                        self.rm.output_dir / "permutation" /
                        f"perm__{sanitize_name(label)}__{model_name}"
                        f"__{feature_strategy}__fold{fold_idx}.csv",
                        index=False,
                    )
                except Exception as e:
                    logger.warning(f"Permutation importance failed [fold{fold_idx}]: {e}")

            # ── 9. Save model ─────────────────────────────────────────────────
            if self.cfg.save_best_models:
                try:
                    with open(self.rm.output_dir / "models" / f"{study_name}.pkl", "wb") as fh:
                        pickle.dump(final_pipe, fh)
                except Exception as exc:
                    logger.debug(f"Could not pickle model: {exc}")

            logger.info(
                f"    Fold {fold_idx}: R²={metrics['r2']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"({elapsed:.1f}s)"
            )

        return exp.summarize(n_samples=len(y))

    # ── main loop ─────────────────────────────────────────────────────────────

    def run_all(self, ds: Dataset) -> ResultsManager:
        names  = self.reg.available_names(
            include=self.cfg.include_models, exclude=self.cfg.exclude_models,
        )
        t0 = time.perf_counter()

        if (self.cfg.use_group_cv and ds.high_cardinality_columns
                and HAS_TARGET_ENCODER):
            logger.warning(
                "Group CV + TargetEncoder: sklearn's internal cross-fitting is "
                "NOT group-aware \u2192 residual target leakage across groups inside "
                "training folds. Consider drop_columns for high-cardinality "
                "features or a custom group-aware encoder."
            )

        for l_idx, label in enumerate(ds.target_columns, start=1):
            logger.info(f"\n{'='*70}\n[{l_idx}/{len(ds.target_columns)}] Target: {label}\n{'='*70}")
            mask   = ds.df_targets[label].notna()
            n_valid = int(mask.sum())
            if n_valid < self.cfg.min_samples_per_label:
                logger.warning(
                    f"  Skipping '{label}': {n_valid} valid samples "
                    f"(need ≥ {self.cfg.min_samples_per_label})."
                )
                continue
            X = ds.df_features.loc[mask].copy()
            y = ds.df_targets.loc[mask, label].astype(float).copy()
            g = ds.groups.loc[mask].copy() if ds.groups is not None else None

            logger.info(f"  n_valid={n_valid}")

            for model_name in names:
                logger.info(f"\n  ── Model: {model_name} ──")
                for fs in self._resolve_strategies(model_name):
                    logger.info(f"    Strategy: {fs}")
                    try:
                        res = self.run_single(
                            X=X, y=y, groups=g, ds=ds,
                            label=label, model_name=model_name,
                            feature_strategy=fs,
                        )
                        if res.fold_results:
                            self.rm.add(res)
                            if self.cfg.save_oof_predictions:
                                self.rm.save_fold_predictions(res)
                            logger.info(
                                f"    → OOF RMSE={res.oof_rmse:.4f}  "
                                f"OOF R²={res.oof_r2:.4f}  "
                                f"CI95=[{res.ci95_rmse[0]:.4f}, {res.ci95_rmse[1]:.4f}]"
                            )
                    except Exception as exc:
                        logger.error(
                            f"    FAILED [{label}/{model_name}/{fs}]: {exc}",
                            exc_info=True,
                        )

        logger.info(f"\nTotal NestedCV runtime: {time.perf_counter() - t0:.1f}s")
        self.stab.save(self.rm.output_dir)
        self.rm.print_summary()
        self.rm.save()
        return self.rm


# =============================================================================
# 18. TOP-LEVEL ORCHESTRATOR
# =============================================================================

class TabularMLPipeline:
    """
    Top-level orchestrator. Fully domain-agnostic.

    Typical usage
    -------------
    # From any DataFrame:
    results = from_dataframe(df, target_columns=["price", "yield"]).run()

    # From a file:
    results = from_csv("data.csv", ["target"]).run()

    # From SQL:
    results = from_sqlite("db.sqlite", "SELECT * FROM data", ["target"]).run()

    # With custom components:
    pipeline = TabularMLPipeline(
        data_source=MyCustomDataSource(),
        feature_engineer=MyDomainEngineer(feat_cfg),
        ...
    )
    pipeline.register_model("my_model", MyEstimator, my_instantiator)
    results = pipeline.run()
    """

    def __init__(
        self,
        data_source: DataSource,
        feat_config: Optional[FeatureConfig] = None,
        tgt_config:  Optional[TargetConfig]  = None,
        ml_config:   Optional[MLConfig]      = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        label_extractor:  Optional[LabelExtractor]  = None,
    ) -> None:
        self.feat_cfg = feat_config or FeatureConfig()
        self.tgt_cfg  = tgt_config  or TargetConfig()
        self.ml_cfg   = ml_config   or MLConfig()

        if not self.tgt_cfg.target_columns:
            raise ValueError(
                "TargetConfig.target_columns must be non-empty. "
                "Specify the column name(s) you want to predict."
            )

        self.ds_builder = DatasetBuilder(
            data_source=data_source,
            feat_config=self.feat_cfg,
            tgt_config=self.tgt_cfg,
            ml_config=self.ml_cfg,
            feature_engineer=feature_engineer,
            label_extractor=label_extractor,
        )
        self.registry = ModelRegistry(self.ml_cfg.random_state, self.ml_cfg.n_jobs)
        self.pf       = PipelineFactory(self.feat_cfg, self.ml_cfg.random_state)
        self.tuner    = OptunaTuner(self.ml_cfg)
        self.rm       = ResultsManager(self.ml_cfg.output_dir)
        self.shap_cmp = SHAPComputer(self.ml_cfg, Path(self.ml_cfg.output_dir))
        self.stab     = FeatureStabilityTracker()

        self.cv_runner = NestedCVRunner(
            ml_cfg=self.ml_cfg, tgt_cfg=self.tgt_cfg, feat_cfg=self.feat_cfg,
            registry=self.registry, pf=self.pf, tuner=self.tuner, rm=self.rm,
            shap=self.shap_cmp, stab=self.stab,
        )
        self.mt_runner    = MultiTaskRunner(self.ml_cfg, self.registry, self.feat_cfg, self.rm)
        self.stack_runner = StackingRunner(self.ml_cfg, self.registry, self.pf, self.rm)

    def register_model(
        self,
        name: str,
        cls: Type[BaseEstimator],
        instantiator: Optional[Callable[[Dict[str, Any]], BaseEstimator]] = None,
    ) -> "TabularMLPipeline":
        """Register a custom regressor. Chainable."""
        self.registry.register(name, cls, instantiator)
        return self

    def run(self, dataset: Optional[Dataset] = None) -> ResultsManager:
        logger.info("=" * 70)
        logger.info("TabularML Pipeline v5.0 — starting")
        logger.info("=" * 70)

        ds = dataset or self.ds_builder.build()

        if self.ml_cfg.save_config_snapshot:
            self.rm.save_config_snapshot({
                "ml_config":   asdict(self.ml_cfg),
                "feat_config": asdict(self.feat_cfg),
                "tgt_config":  asdict(self.tgt_cfg),
                "n_features":  len(ds.feature_columns),
                "n_numeric":   len(ds.numeric_columns),
                "n_categorical": len(ds.categorical_columns),
                "n_datetime":  len(ds.datetime_columns),
                "n_targets":   len(ds.target_columns),
                "n_samples":   len(ds.df_features),
            })

        # 1. Single-task nested CV
        self.cv_runner.run_all(ds)

        # 2. Multi-task learning
        self.mt_runner.run(ds)

        # 3. Stacking ensemble
        stack_base = self.registry.available_names(
            include=self.ml_cfg.include_models,
            exclude=self.ml_cfg.exclude_models + ["dummy", "svr", "linear_svr", "tabpfn"],
        )
        self.stack_runner.run(ds, stack_base)

        self.rm.save()
        return self.rm


# =============================================================================
# 19. CONVENIENCE FACTORIES
# =============================================================================

def from_dataframe(
    df: pd.DataFrame,
    target_columns: Union[str, List[str]],
    group_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    sample_weight_column: Optional[str] = None,
    ml_config:   Optional[MLConfig]   = None,
    feat_config: Optional[FeatureConfig] = None,
    tgt_config:  Optional[TargetConfig]  = None,
) -> TabularMLPipeline:
    """
    Build a pipeline from any in-memory pandas DataFrame.

    Example
    -------
    results = from_dataframe(
        df=my_df,
        target_columns=["price", "yield"],
        group_column="batch_id",
    ).run()
    """
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    tc = tgt_config or TargetConfig()
    tc.target_columns = targets
    ml = ml_config or MLConfig()
    if group_column:
        ml.group_column = group_column
        ml.use_group_cv = True
    if drop_columns:
        ml.drop_columns = drop_columns
    if sample_weight_column:
        ml.sample_weight_column = sample_weight_column
    return TabularMLPipeline(
        data_source=InMemoryDataSource(df),
        feat_config=feat_config,
        tgt_config=tc,
        ml_config=ml,
    )


def from_csv(
    file_path: str,
    target_columns: Union[str, List[str]],
    group_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    sep: str = ",",
    encoding: str = "utf-8",
    ml_config:   Optional[MLConfig]   = None,
    feat_config: Optional[FeatureConfig] = None,
    tgt_config:  Optional[TargetConfig]  = None,
) -> TabularMLPipeline:
    """
    Build a pipeline from a CSV or TSV file.

    Example
    -------
    results = from_csv("housing.csv", ["price"], group_column="city").run()
    results = from_csv("data.tsv", ["yield"], sep="\\t").run()
    """
    cfg = DataSourceConfig(file_path=file_path, file_format="csv",
                           csv_sep=sep, csv_encoding=encoding)
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    tc = tgt_config or TargetConfig()
    tc.target_columns = targets
    ml = ml_config or MLConfig()
    if group_column:
        ml.group_column = group_column
        ml.use_group_cv = True
    if drop_columns:
        ml.drop_columns = drop_columns
    return TabularMLPipeline(
        data_source=FileDataSource(cfg),
        feat_config=feat_config, tgt_config=tc, ml_config=ml,
    )


def from_parquet(
    file_path: str,
    target_columns: Union[str, List[str]],
    group_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    ml_config:   Optional[MLConfig]   = None,
    feat_config: Optional[FeatureConfig] = None,
    tgt_config:  Optional[TargetConfig]  = None,
) -> TabularMLPipeline:
    """Build a pipeline from a Parquet file."""
    cfg = DataSourceConfig(file_path=file_path, file_format="parquet")
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    tc = tgt_config or TargetConfig()
    tc.target_columns = targets
    ml = ml_config or MLConfig()
    if group_column:
        ml.group_column = group_column
        ml.use_group_cv = True
    if drop_columns:
        ml.drop_columns = drop_columns
    return TabularMLPipeline(
        data_source=FileDataSource(cfg),
        feat_config=feat_config, tgt_config=tc, ml_config=ml,
    )


def from_excel(
    file_path: str,
    target_columns: Union[str, List[str]],
    sheet: Union[str, int] = 0,
    group_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    ml_config:   Optional[MLConfig]   = None,
    feat_config: Optional[FeatureConfig] = None,
    tgt_config:  Optional[TargetConfig]  = None,
) -> TabularMLPipeline:
    """Build a pipeline from an Excel file."""
    ext = Path(file_path).suffix.lstrip(".").lower() or "xlsx"
    cfg = DataSourceConfig(file_path=file_path, file_format=ext, excel_sheet=sheet)
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    tc = tgt_config or TargetConfig()
    tc.target_columns = targets
    ml = ml_config or MLConfig()
    if group_column:
        ml.group_column = group_column
        ml.use_group_cv = True
    if drop_columns:
        ml.drop_columns = drop_columns
    return TabularMLPipeline(
        data_source=FileDataSource(cfg),
        feat_config=feat_config, tgt_config=tc, ml_config=ml,
    )


def from_sqlite(
    db_path: str,
    sql_query: str,
    target_columns: Union[str, List[str]],
    group_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    ml_config:   Optional[MLConfig]   = None,
    feat_config: Optional[FeatureConfig] = None,
    tgt_config:  Optional[TargetConfig]  = None,
) -> TabularMLPipeline:
    """
    Build a pipeline from a SQLite database.

    Example
    -------
    results = from_sqlite(
        db_path="experiments.db",
        sql_query="SELECT * FROM results WHERE split='train'",
        target_columns=["yield"],
    ).run()
    """
    cfg = DataSourceConfig(sqlite_path=db_path, sql_query=sql_query)
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    tc = tgt_config or TargetConfig()
    tc.target_columns = targets
    ml = ml_config or MLConfig()
    if group_column:
        ml.group_column = group_column
        ml.use_group_cv = True
    if drop_columns:
        ml.drop_columns = drop_columns
    return TabularMLPipeline(
        data_source=SQLiteDataSource(cfg),
        feat_config=feat_config, tgt_config=tc, ml_config=ml,
    )


def from_sqlalchemy(
    uri: str,
    sql_query: str,
    target_columns: Union[str, List[str]],
    group_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    ml_config:   Optional[MLConfig]   = None,
    feat_config: Optional[FeatureConfig] = None,
    tgt_config:  Optional[TargetConfig]  = None,
) -> TabularMLPipeline:
    """
    Build a pipeline from any SQLAlchemy-supported database.

    Example
    -------
    results = from_sqlalchemy(
        uri="postgresql://user:pw@host/db",
        sql_query="SELECT * FROM measurements",
        target_columns=["efficiency"],
    ).run()
    """
    cfg = DataSourceConfig(sqlalchemy_uri=uri, sql_query=sql_query)
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    tc = tgt_config or TargetConfig()
    tc.target_columns = targets
    ml = ml_config or MLConfig()
    if group_column:
        ml.group_column = group_column
        ml.use_group_cv = True
    if drop_columns:
        ml.drop_columns = drop_columns
    return TabularMLPipeline(
        data_source=SQLAlchemyDataSource(cfg),
        feat_config=feat_config, tgt_config=tc, ml_config=ml,
    )


# =============================================================================
# 20. ENTRY POINT
# =============================================================================

def main() -> None:
    """
    CLI entry point.

    Usage
    -----
    python pipeline.py demo
    python pipeline.py csv      <file>    <target1> [target2 ...]
    python pipeline.py tsv      <file>    <target1> [target2 ...]
    python pipeline.py parquet  <file>    <target1> [target2 ...]
    python pipeline.py excel    <file>    <target1> [target2 ...]  [--sheet=N]
    python pipeline.py sqlite   <db_path> <sql_or_table> <target1> [target2 ...]
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    def quick_cfg(output: str = "ml_results") -> MLConfig:
        return MLConfig(
            random_state=42,
            n_outer_splits=5,
            n_inner_splits=3,
            n_optuna_trials=30,
            optuna_use_pruning=True,
            use_group_cv=False,
            n_jobs=-1,
            output_dir=output,
            compute_shap=True,
            shap_sample_size=100,
            compute_conformal=True,
            compute_permutation_importance=False,
            use_multitask=False,
            use_stacking=False,
            save_best_models=True,
            track_feature_stability=True,
        )

    if mode == "demo":
        logger.info("Running demo on California Housing dataset...")
        from sklearn.datasets import fetch_california_housing
        data   = fetch_california_housing(as_frame=True)
        df     = data.frame.copy()
        ml     = quick_cfg("ml_results_demo")
        ml.include_models = ["dummy", "ridge", "random_forest", "hist_gbrt"]
        ml.feature_strategies_override = ["full", "full_scaled", "tree_select"]
        ml.n_optuna_trials = 20
        results = from_dataframe(
            df=df, target_columns="MedHouseVal", ml_config=ml,
        ).run()

    elif mode in ("csv", "tsv"):
        if len(sys.argv) < 4:
            print(f"Usage: pipeline.py {mode} <file> <target1> [target2 ...]")
            sys.exit(1)
        sep = "\t" if mode == "tsv" else ","
        results = from_csv(
            file_path=sys.argv[2],
            target_columns=sys.argv[3:],
            sep=sep,
            ml_config=quick_cfg(),
        ).run()

    elif mode == "parquet":
        if len(sys.argv) < 4:
            print("Usage: pipeline.py parquet <file> <target1> [target2 ...]")
            sys.exit(1)
        results = from_parquet(
            file_path=sys.argv[2],
            target_columns=sys.argv[3:],
            ml_config=quick_cfg(),
        ).run()

    elif mode == "excel":
        if len(sys.argv) < 4:
            print("Usage: pipeline.py excel <file> <target1> [target2 ...] [--sheet=N]")
            sys.exit(1)
        sheet: Union[str, int] = 0
        targets: List[str] = []
        for a in sys.argv[3:]:
            if a.startswith("--sheet="):
                v = a.split("=", 1)[1]
                sheet = int(v) if v.isdigit() else v
            else:
                targets.append(a)
        results = from_excel(
            file_path=sys.argv[2], target_columns=targets,
            sheet=sheet, ml_config=quick_cfg(),
        ).run()

    elif mode == "sqlite":
        if len(sys.argv) < 5:
            print("Usage: pipeline.py sqlite <db_path> <sql_or_table> <target1> [target2 ...]")
            sys.exit(1)
        q = sys.argv[3]
        if not q.lower().strip().startswith("select"):
            q = f"SELECT * FROM {q}"
        results = from_sqlite(
            db_path=sys.argv[2], sql_query=q,
            target_columns=sys.argv[4:], ml_config=quick_cfg(),
        ).run()

    else:
        print(f"Unknown mode '{mode}'. Modes: demo | csv | tsv | parquet | excel | sqlite")
        sys.exit(1)

    print("\nBest configuration per target:")
    print(results.get_best_per_label().to_string(index=False))


if __name__ == "__main__":
    main()