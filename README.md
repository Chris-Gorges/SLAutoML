# User Manual — `SLAutoML` Supervised Learning Script

> **Important:** This pipeline is for **regression only**, not classification. All target columns must be numeric or convertible to numeric.

---

# Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Core ML Concepts](#2-core-ml-concepts)
3. [Data Flow & Architecture](#3-data-flow--architecture)
4. [Configuration Reference](#4-configuration-reference)
5. [Running the Script & Examples](#5-running-the-script--examples)
6. [Available Models](#6-available-models)
7. [Feature Engineering & Preprocessing](#7-feature-engineering--preprocessing)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Evaluation & Metrics](#9-evaluation--metrics)
10. [Output Files](#10-output-files)
11. [Advanced Features](#11-advanced-features)
12. [Function & Class Reference](#12-function--class-reference)
13. [Recommended Workflows](#13-recommended-workflows)
14. [Common Errors & Fixes](#14-common-errors--fixes)
15. [Best Practices](#15-best-practices)

---

# 1. High-Level Overview

This script is a **production-grade, fully domain-agnostic supervised regression pipeline** for tabular data. You point it at a data source, tell it your target column(s), and it handles everything else automatically.

At a high level, the pipeline performs:

1. **Data loading** — from CSV, TSV, Excel, Parquet, Feather, HDF5, JSON, JSONL, SQLite, SQLAlchemy databases, or an in-memory `pandas.DataFrame`.
2. **Data engineering** — sanitizes column names, extracts targets, drops irrelevant columns, infers feature types.
3. **Preprocessing** — numeric imputation and scaling, categorical encoding, datetime expansion, optional CLR transform for compositional data, optional correlation filtering.
4. **Model training** — runs many regression models with nested cross-validation and Bayesian hyperparameter tuning (Optuna).
5. **Evaluation** — computes $R^2$, MAE, MedAE, MSE, RMSE, MAPE; produces out-of-fold (OOF) predictions and bootstrap confidence intervals.
6. **Explainability & uncertainty** — optional SHAP importance, optional permutation importance, optional conformal prediction intervals.
7. **Advanced modes** — multi-target regression, stacking ensembles.
8. **Output** — saves models, predictions, SHAP files, and summary CSVs to an output directory.

The most common usage is simply:

```python
results = from_csv("data.csv", target_columns="target").run()
```

---

# 2. Core ML Concepts

> Read this section before configuring the pipeline. It defines the vocabulary used throughout.

<details>
<summary><strong>2.1 Cross-Validation (CV) and Folds</strong></summary>

**Cross-validation** is a technique for estimating how well a model will generalize to unseen data. The dataset is split into $k$ roughly equal parts called **folds**.

- In each round, one fold is held out as the **test set** and the remaining $k-1$ folds are used for **training**.
- This is repeated $k$ times so every sample is tested exactly once.
- The metric (e.g., RMSE) is averaged across all $k$ folds.

The number of folds is controlled by `n_outer_splits` (default: `5`).

</details>

<details>
<summary><strong>2.2 Nested Cross-Validation</strong></summary>

Standard CV has a problem: if you use the same data to both tune hyperparameters and estimate performance, your performance estimate will be **optimistically biased**.

**Nested CV** solves this by using two loops:

```
Outer CV loop  →  estimates generalization performance (unbiased)
    Inner CV loop  →  tunes hyperparameters (Optuna)
```

- The **outer test fold** is never seen during hyperparameter tuning.
- The **inner loop** runs entirely inside the outer training data.

This pipeline uses true nested CV. The key parameters are:

| Parameter | Role |
|---|---|
| `n_outer_splits` | Number of outer folds (for performance estimation) |
| `n_inner_splits` | Number of inner folds (for hyperparameter tuning) |
| `n_optuna_trials` | Number of Optuna trials per inner study |

</details>

<details>
<summary><strong>2.3 Out-of-Fold (OOF) Predictions</strong></summary>

**Out-of-fold (OOF) predictions** are assembled by collecting each outer test fold's predictions and concatenating them. Every row in the dataset appears in exactly one test fold, so the full dataset gets a predicted value that was made by a model that never saw that row during training.

OOF predictions are more informative than simple fold-metric averages because they reflect the full distribution of errors, including rare cases.

The pipeline reports:
- `oof_r2`, `oof_rmse`, `oof_mae`, `oof_mape`

OOF prediction files are saved to `fold_predictions/` for further analysis.

</details>

<details>
<summary><strong>2.4 Group-Aware Splitting</strong></summary>

Standard KFold assumes rows are **independent**. Many real datasets violate this:

- Multiple measurements from the same patient
- Multiple experiments in the same batch
- Multiple time windows from the same sensor
- Repeated measurements from the same location

If you split such data randomly, the same source can appear in both train and test, causing the model to "memorize" group-specific patterns. This inflates performance estimates.

**Group-aware splitting** (`GroupKFold`) keeps all rows from the same group in either train or test — never both. Enable it via `group_column`:

```python
from_dataframe(df, target_columns="y", group_column="patient_id")
```

</details>

<details>
<summary><strong>2.5 SHAP (Feature Importance)</strong></summary>

**SHAP (SHapley Additive exPlanations)** is a method for explaining model predictions. Each feature gets a SHAP value that represents its contribution to a specific prediction.

The pipeline reports `mean_abs_shap` — the average absolute SHAP value across the test sample — as a feature importance ranking.

Depending on the model type, different explainers are used:

| Model family | SHAP explainer | Speed |
|---|---|---|
| Tree models | `TreeExplainer` | Fast |
| Linear models | `LinearExplainer` | Fast |
| Other models | `KernelExplainer` (fallback) | Slow |

SHAP requires `pip install shap`.

</details>

<details>
<summary><strong>2.6 Conformal Prediction Intervals</strong></summary>

Standard models output a single point prediction $\hat{y}$. **Conformal prediction** adds statistically grounded uncertainty intervals.

The pipeline uses **split conformal prediction**:

1. A **calibration set** is isolated from the outer training fold *before* any hyperparameter tuning (so it is never used for fitting or tuning).
2. The fitted model predicts on the calibration set and residuals $|y - \hat{y}|$ are computed.
3. A quantile $\hat{q}$ is computed from those residuals at level $\lceil(n+1)(1-\alpha)\rceil / n$.
4. For any new prediction: the interval is $[\hat{y} - \hat{q},\; \hat{y} + \hat{q}]$.

With `conformal_alpha=0.10`, you get nominal **90% coverage**. With `conformal_alpha=0.05`, you get nominal **95% coverage**.

This guarantee is **finite-sample valid** for exchangeable data (no distributional assumptions required).

</details>

<details>
<summary><strong>2.7 Target Transforms</strong></summary>

Many regression targets are right-skewed (e.g., prices, counts, concentrations). Linear models and some tree models perform better when the target is approximately normally distributed.

The pipeline can automatically detect skewness and apply a reversible transformation **per fold** (fitted only on training data — no leakage). Predictions are inverse-transformed back to the original scale before metrics are computed.

Available transforms:

| Transform | Use case |
|---|---|
| `"none"` | Target is already approximately normal |
| `"log1p"` | Non-negative, strongly right-skewed targets (e.g., counts, prices) |
| `"yeo-johnson"` | Skewed targets that can include zero or negative values |
| `"box-cox"` | Strictly positive skewed targets; slightly more powerful than Yeo-Johnson for positives |
| `"quantile-normal"` | Nonlinear transform — forces the target to a near-normal distribution; use when other transforms fail |

</details>

<details>
<summary><strong>2.8 Feature Strategies</strong></summary>

A **feature strategy** controls how features are preprocessed and selected before being passed to a model. The same model can be tried with different strategies in a single run, and the best combination is selected.

Examples:
- `full_scaled` — use all features with standard scaling (good default for linear models)
- `tree_select` — select features using ExtraTrees importance before fitting
- `pca` — compress features using PCA before fitting a linear model
- `selectkbest` — keep only the top $k$ features by F-statistic

Different model families get different default strategies automatically (see [Section 7.3](#73-feature-strategies)).

</details>

---

# 3. Data Flow & Architecture

## 3.1 End-to-End Data Flow

```
Raw Data Source
    │
    ▼
DataSource.load()          ← CSV / DataFrame / SQLite / etc.
    │
    ▼
FeatureEngineer.engineer() ← sanitize column names; custom feature creation
    │
    ▼
LabelExtractor.extract()   ← pull out target columns
    │
    ▼
TypeInferenceEngine.fit()  ← classify each column: numeric / categorical / datetime / drop
    │
    ▼
Dataset                    ← container: df_features, df_targets, column roles, groups, weights
    │
    ▼
NestedCVRunner.run_all()   ← outer CV loop per target × model × strategy
    │
    ├── per outer fold:
    │       │
    │       ├── _proper_cal_split()     ← isolate calibration set BEFORE HPO
    │       ├── OptunaTuner.optimize()  ← inner CV + Bayesian HPO
    │       ├── refit on full outer train
    │       ├── TargetTransformer (inverse)
    │       ├── compute_metrics()
    │       ├── ConformalPredictor.calibrate() + predict_interval()
    │       ├── SHAPComputer.compute_and_save()
    │       └── permutation importance (optional)
    │
    ▼
ResultsManager             ← aggregate, save CSV/Parquet, print summary
    │
    ▼
MultiTaskRunner (optional) ← RegressorChain, MultiOutputRegressor
StackingRunner (optional)  ← OOF meta-features → RidgeCV
```

## 3.2 Component Map

| Pipeline stage | Script component(s) |
|---|---|
| Data loading | `DataSource`, `FileDataSource`, `SQLiteDataSource`, `SQLAlchemyDataSource`, `InMemoryDataSource` |
| Feature creation | `FeatureEngineer`, `PassthroughEngineer` |
| Label extraction | `LabelExtractor`, `ColumnLabelExtractor` |
| Type inference | `TypeInferenceEngine` |
| Dataset assembly | `DatasetBuilder`, `Dataset` |
| Preprocessing | `PreprocessorBuilder`, `DatetimeExpander`, `CLRTransformer`, `CorrelationFilter` |
| Feature selection | `TreeFeatureSelector`, `MutualInfoSelector`, `SelectKBest`, `PCA` |
| Model registry | `ModelRegistry`, `PipelineFactory` |
| HPO | `OptunaTuner` |
| CV engine | `NestedCVRunner` |
| Evaluation | `compute_metrics`, `ExperimentResult`, `ResultsManager` |
| Uncertainty | `ConformalPredictor` |
| Explainability | `SHAPComputer`, permutation importance |
| Multi-target | `MultiTaskRunner` |
| Stacking | `StackingRunner` |
| Orchestration | `TabularMLPipeline` |

## 3.3 Column Name Sanitization

All column names are automatically sanitized to safe Python identifiers via `sanitize_dataframe_columns`. This happens before any other processing.

| Original name | Sanitized name |
|---|---|
| `"Sale Price"` | `Sale_Price` |
| `"temp°C"` | `temp_C` |
| `"feature-1"` | `feature_1` |
| `"my column"` + `"my-column"` | `my_column`, `my_column_1` |

> **Important:** Your `target_columns` argument is also sanitized internally. Pass the original column name — e.g., `"Sale Price"` — and the script will find `Sale_Price` automatically.

## 3.4 Column Type Inference

`TypeInferenceEngine` assigns every column one of four roles:

| Role | Meaning |
|---|---|
| `numeric` | Continuous numeric feature |
| `categorical` | Low-cardinality integer or string |
| `datetime` | Date/time feature (expanded to numeric parts) |
| `drop` | Excluded from modeling |

Logic applied in order:
1. Columns in `drop_columns`, group column, weight column, and target columns → `drop`
2. Columns matching `column_type_overrides` regex patterns → forced role
3. Missing fraction > `max_missing_fraction` → `drop`
4. Explicit datetime dtype → `datetime`
5. Numeric dtype with ≤ `max_cardinality_for_categorical` unique values → `categorical`
6. Numeric dtype → `numeric` (or `datetime` if name contains patterns like `"date"`, `"time"`)
7. Boolean → `categorical`
8. Object/string with ≤ `max_cardinality_for_categorical` unique values → `categorical`
9. Object/string parseable as numeric → `numeric`
10. Object/string parseable as datetime → `datetime`
11. High-cardinality string → `drop`

You can always override the inference:

```python
feat = FeatureConfig(
    column_type_overrides={
        "customer_id": "drop",
        "zip_code": "categorical",
        "temperature_raw": "numeric",
    }
)
```

---

# 4. Configuration Reference

The pipeline is controlled by three dataclasses. Create them explicitly for full control, or use the defaults for a quick start.

## 4.1 `DataSourceConfig`

Controls how data is loaded. You usually do not create this directly — use the convenience factories (`from_csv`, `from_dataframe`, etc.) instead.

| Field | Default | Purpose |
|---|---|---|
| `file_path` | `None` | Path to CSV, TSV, Excel, Parquet, JSON, etc. |
| `file_format` | `None` | Explicit format override (auto-detected from extension if `None`) |
| `csv_sep` | `","` | CSV column separator |
| `csv_encoding` | `"utf-8"` | CSV file encoding |
| `excel_sheet` | `0` | Excel sheet name or index |
| `hdf5_key` | `"data"` | HDF5 dataset key |
| `sqlite_path` | `None` | SQLite database file path |
| `sqlalchemy_uri` | `None` | SQLAlchemy connection URI |
| `sql_query` | `None` | SQL query string |

---

## 4.2 `FeatureConfig`

Controls all preprocessing, encoding, and feature selection behavior.

### Column Type Inference

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `max_cardinality_for_categorical` | `50` | Any integer > 0 | Keep under 100. High-cardinality columns (e.g., zip codes, product IDs) explode one-hot encoding into huge sparse matrices, wasting RAM and time. If you have IDs with thousands of unique values, use `column_type_overrides` to force them to `"drop"`. |
| `max_missing_fraction` | `0.90` | Float 0.0–1.0 | Drops columns that are mostly empty. Lower to `0.50` if you want to keep only reliable features. Raise toward `0.99` if you have intentionally sparse columns where missingness itself is informative (but encode missingness explicitly as a binary feature first). |
| `column_type_overrides` | `{}` | Dict of `{regex_pattern: "numeric"\|"categorical"\|"drop"}` | Use regex patterns to force whole groups of columns. For example `{"^id_": "drop"}` drops all columns starting with `id_`. Essential for excluding leaky or irrelevant columns. |

### Imputation

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `numeric_imputer_strategy` | `"median"` | `"median"`, `"mean"`, `"most_frequent"`, `"constant"` | **Median** is robust to extreme outliers (e.g., sensor spikes, long tails) and is the safe default. Use `"mean"` only if your data is normally distributed with few outliers. Use `"constant"` (combined with a `fill_value`) when missingness is meaningful and you want a sentinel value. |
| `categorical_imputer_strategy` | `"most_frequent"` | `"most_frequent"`, `"constant"` | `"most_frequent"` is the standard choice. Use `"constant"` to impute with a dedicated `"missing"` category, which preserves the information that a value was absent. |

### Scaling

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `numeric_scaler` | `"standard"` | `"standard"`, `"robust"`, `"quantile"`, `"none"` | **`"standard"`** (Z-score normalization) is the baseline for linear models, SVMs, and KNN — always required for distance-based or regularized models. **`"robust"`** uses median and IQR instead of mean and std; use when data has heavy outliers that would distort Z-scores. **`"quantile"`** forces a near-normal or uniform distribution; use when data is highly skewed even after target transform. **`"none"`** is appropriate for tree-based models only (they are scale-invariant). |

### Filtering

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `use_variance_threshold` | `True` | `True`, `False` | Removes near-constant columns (all values identical). Almost always beneficial. Only turn off if you have a specific reason to keep zero-variance features. |
| `variance_threshold` | `1e-8` | Float ≥ 0 | The threshold for "near-constant." The default is extremely conservative (essentially removes only perfectly constant columns). Raise to `0.01` to also remove very low-variance features. |
| `use_correlation_filter` | `False` | `True`, `False` | Turn on (with `correlation_threshold` ~0.95) when features are highly collinear — e.g., multiple temperature sensors close to each other, or redundant engineered features. Reduces matrix rank problems for linear models and speeds up tree models slightly. Has no effect on model accuracy if regularization is strong. |
| `correlation_threshold` | `0.98` | Float 0.0–1.0 | Pairs with $\|r\| \geq$ this value lose one member. `0.98` is very conservative (removes near-duplicates only). Lower to `0.90` to be more aggressive. Fitted per fold — leakage-free. |

### Feature Selection

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `selectkbest_k` | `30` | Any integer > 0 | Limits the dataset to the top $k$ features by score. Useful when dealing with hundreds or thousands of variables to prevent the curse of dimensionality and speed up training significantly. Set $k$ to roughly 10–50 for typical tabular datasets. Used by the `selectkbest` and `selectkbest_mi` strategies. |
| `tree_selector_threshold` | `"median"` | `"median"`, `"mean"`, float | Controls how many features `tree_select` retains. `"median"` keeps approximately half. Lower values (e.g., `"0.1*mean"`) keep fewer features. Used by the `tree_select` strategy. |
| `tree_selector_n_estimators` | `200` | Integer > 0 | Number of trees in the ExtraTrees selector. More trees = more stable selection but slower. |
| `pca_variance_ratio` | `0.95` | Float 0.0–1.0 | PCA retains enough components to explain this fraction of total variance. `0.95` means "keep 95% of the information." Great for highly correlated physical measurements (e.g., spectroscopy). Lower to `0.80` for more compression. Used by the `pca` strategy (linear models only). |

### Datetime & Compositional

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `expand_datetime` | `True` | `True`, `False` | Converts each datetime column into year, month, day, day-of-week, hour, minute, second, day-of-year, quarter, is-weekend. Disable only if you want to handle datetime features manually. |
| `apply_clr_transform` | `False` | `True`, `False` | Enable the Centered Log-Ratio (CLR) transform for compositional features — i.e., features that represent proportions summing to 1 (chemical compositions, microbiome abundances, budget shares). See [Section 7.5](#75-clr-transform-for-compositional-data). |
| `clr_column_pattern` | `""` | Regex string | Regex selecting which numeric columns are compositional. Example: `"^comp_"` selects all columns starting with `comp_`. Only used when `apply_clr_transform=True`. |
| `clr_pseudocount` | `1e-6` | Small positive float | Added before taking log to avoid $\log(0)$. The default is appropriate for proportions; increase if your data has exact zeros that are structurally meaningful. |

---

## 4.3 `TargetConfig`

Controls target column selection and automatic transformations.

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `target_columns` | `[]` | List of column name strings | The column(s) you want to predict. Must be set. |
| `target_transforms` | `{}` | Dict `{column_name: transform_name}` | Explicit override per target. Takes priority over `auto_transform`. |
| `auto_transform` | `True` | `True`, `False` | Automatically selects a transform based on skewness. Disable if you want full manual control or if your target has already been normalized. |
| `skew_threshold` | `1.0` | Float > 0 | Absolute skewness must exceed this value for auto-transform to activate. Raise to `2.0` to only transform very strongly skewed targets; lower to `0.5` to transform mildly skewed ones. |
| `preferred_auto_transform` | `"yeo-johnson"` | `"log1p"`, `"yeo-johnson"`, `"box-cox"`, `"quantile-normal"` | The transform chosen when auto-transform triggers. `"yeo-johnson"` is the safest default (handles zero and negative values). Use `"log1p"` if you know targets are always non-negative and the log scale is physically meaningful (e.g., concentrations, prices). |
| `positive_targets` | `None` | List of column name strings | Declares which targets are always non-negative, allowing `log1p` to be used for those targets during auto-selection. |

**Available transforms and when to use each:**

| Transform | Use case |
|---|---|
| `"none"` | Target is already approximately normal; no transformation needed |
| `"log1p"` | Non-negative, strongly right-skewed targets (prices, counts, concentrations). The `+1` avoids $\log(0)$. Inverse is $\exp(\cdot) - 1$. |
| `"yeo-johnson"` | Skewed targets that may include zero or negative values. The safest general-purpose choice. |
| `"box-cox"` | Strictly positive skewed targets. Theoretically optimal for positive data but requires $y > 0$. If any value $\leq 0$, the pipeline automatically falls back to Yeo-Johnson. |
| `"quantile-normal"` | Nonlinear transform that forces the empirical distribution to match a normal distribution. Use as a last resort when other transforms fail to reduce skewness adequately. |

Example — explicit transforms:

```python
from pipeline import TargetConfig

target_cfg = TargetConfig(
    target_columns=["price", "count"],
    target_transforms={
        "price": "log1p",   # price is always positive
        "count": "yeo-johnson",
    },
    positive_targets=["price"],
)
```

---

## 4.4 `MLConfig`

Controls CV structure, HPO, resources, output, and optional features.

### Cross-Validation

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `n_outer_splits` | `5` | Integer ≥ 2 | Number of outer CV folds. `5` is standard. Use `3` for large datasets or quick exploration; `10` for small datasets where variance in the estimate matters. Reduce if runtime is prohibitive. |
| `n_inner_splits` | `4` | Integer ≥ 2 | Number of inner CV folds used during Optuna tuning. `4` is a good balance. Lower to `2` or `3` to reduce runtime at the cost of less stable HPO. |
| `use_group_cv` | `False` | `True`, `False` | Set to `True` when rows are not independent (see [Section 2.4](#24-group-aware-splitting)). Set automatically when `group_column` is provided via convenience factories. |
| `group_column` | `None` | Column name string | The column defining groups (e.g., `"patient_id"`, `"batch_id"`). |
| `sample_weight_column` | `None` | Column name string | Column of per-sample weights. Currently extracted into the `Dataset` as metadata but **not yet passed to `.fit()`**. |

### Hyperparameter Optimization

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `n_optuna_trials` | `60` | Integer > 0 | Number of Optuna trials per model/strategy/fold combination. `60` is a reasonable default for a thorough search. Use `10`–`20` during exploration and `100`+ for final production benchmarking. More trials = better HPO but longer runtime. |
| `optuna_timeout_sec` | `None` | Integer (seconds) or `None` | Hard time limit per Optuna study. Useful when runtime is constrained. `None` means no limit — HPO runs until `n_optuna_trials` is exhausted. |
| `optuna_n_startup_trials` | `10` | Integer > 0 | Number of random trials before TPE starts. Ensures adequate exploration before exploitation begins. |
| `optuna_use_pruning` | `True` | `True`, `False` | Enables Optuna's `MedianPruner`, which stops unpromising trials early. Almost always beneficial. Disable only if pruning causes instability with a specific model. |

### Resources & I/O

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `n_jobs` | `-1` | Integer or `-1` | Parallel jobs for models that support it. `-1` uses all CPU cores. Set to `1` for debugging (makes stack traces readable). |
| `output_dir` | `"ml_results"` | Directory path string | Where all output files are saved. Use a descriptive name per experiment. |
| `save_best_models` | `True` | `True`, `False` | Saves each fold's fitted pipeline as a `.pkl` file. Disable to save disk space if you only need metrics. |
| `save_oof_predictions` | `True` | `True`, `False` | Saves OOF prediction CSVs to `fold_predictions/`. Almost always worth keeping. |

### Conformal Prediction

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `compute_conformal` | `True` | `True`, `False` | Enables split conformal prediction intervals. Adds very little compute overhead. Recommended for any deployment use case. |
| `conformal_alpha` | `0.10` | Float 0.0–1.0 | Miscoverage level. `0.10` → nominal 90% coverage. `0.05` → nominal 95% coverage. Choose based on the cost of missing an interval. |
| `conformal_calibration_fraction` | `0.20` | Float 0.0–1.0 | Fraction of the outer training fold reserved as a calibration set, isolated *before* HPO. `0.20` means 20% of training data is used for calibration. Lower if you have very little data. |

### SHAP & Permutation Importance

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `compute_shap` | `True` | `True`, `False` | Enables SHAP feature importance. Requires `pip install shap`. Disable to significantly speed up the pipeline if interpretability is not needed. |
| `shap_sample_size` | `200` | Integer > 0 | Number of training rows sampled for SHAP computation per fold. Lower to `50`–`100` if SHAP is slow (especially with `KernelExplainer`). |
| `shap_kernel_fallback` | `True` | `True`, `False` | Falls back to the slow `KernelExplainer` for models that don't support Tree/LinearExplainer. Set to `False` to skip SHAP for such models instead. |
| `compute_permutation_importance` | `False` | `True`, `False` | Computes model-agnostic permutation importance per fold. More expensive than SHAP for large datasets. Useful as a model-agnostic cross-check. |
| `permutation_n_repeats` | `5` | Integer > 0 | Number of shuffles per feature. Higher → more stable estimates but slower. |

### Advanced Modes & Model Selection

| Field | Default | Options | Best Practice / When to Use |
|---|---|---|---|
| `include_models` | `None` | List of model name strings, or `None` | `None` runs all registered models. Specify a list to run only those models — essential for fast exploration. |
| `exclude_models` | `[]` | List of model name strings | Models to skip even if they are registered. Useful for excluding slow models (e.g., `"svr"`, `"tabpfn"`) from a quick run. |
| `feature_strategies_override` | `None` | List of strategy name strings, or `None` | `None` uses the smart default strategies per model family. Override to force all models to use the same strategies — useful for controlled experiments. |
| `min_samples_per_label` | `30` | Integer > 0 | Targets with fewer valid rows than this are skipped. Reduce carefully for small datasets. |
| `use_multitask` | `False` | `True`, `False` | Enables `RegressorChain` and `MultiOutputRegressor` when ≥ 2 targets are present. |
| `use_stacking` | `False` | `True`, `False` | Enables a `RidgeCV` stacking ensemble on OOF predictions from base models. |
| `track_feature_stability` | `True` | `True`, `False` | Records how often each feature is selected across folds. Saved to `feature_stability.csv`. |
| `random_state` | `42` | Any integer | Global random seed for reproducibility. |

---

# 5. Running the Script & Examples

## 5.1 Installation

Core dependencies:

```bash
pip install numpy pandas scipy scikit-learn optuna
```

Recommended optional dependencies:

```bash
pip install pyarrow openpyxl shap xgboost lightgbm catboost sqlalchemy
```

Optional model:

```bash
pip install tabpfn
```

Format-specific dependencies:

| File type | Dependency |
|---|---|
| Parquet / Feather | `pyarrow` or `fastparquet` |
| Excel | `openpyxl` |
| HDF5 | `tables` |
| SQLAlchemy databases | `sqlalchemy` + database driver |

---

## 5.2 Built-in Demo

```bash
python pipeline.py demo
```

Uses the California Housing dataset and trains `dummy`, `ridge`, `random_forest`, `hist_gbrt`. Results are written to `ml_results_demo/`.

---

## 5.3 CLI Usage

```bash
# CSV
python pipeline.py csv housing.csv price

# Multiple targets
python pipeline.py csv experiments.csv yield purity

# TSV
python pipeline.py tsv data.tsv target

# Parquet
python pipeline.py parquet data.parquet target

# Excel
python pipeline.py excel data.xlsx target --sheet=0

# SQLite
python pipeline.py sqlite experiments.db "SELECT * FROM measurements" yield
```

---

## 5.4 Python API Examples

### Fast Baseline Run

Use this to check that data loads correctly and get a performance baseline quickly.

```python
import pandas as pd
from pipeline import from_csv, MLConfig

ml = MLConfig(
    output_dir="fast_results",
    include_models=["dummy", "ridge"],
    n_outer_splits=3,
    n_inner_splits=2,
    n_optuna_trials=10,
    compute_shap=False,
    compute_conformal=False,
)

results = from_csv(
    file_path="data.csv",
    target_columns="target",
    ml_config=ml,
).run()

print(results.get_best_per_label())
```

---

### Strong Tree-Based Benchmark

```python
from pipeline import from_csv, MLConfig

ml = MLConfig(
    output_dir="tree_results",
    include_models=[
        "random_forest", "extra_trees",
        "hist_gbrt", "xgboost", "lightgbm", "catboost",
    ],
    n_outer_splits=5,
    n_inner_splits=4,
    n_optuna_trials=60,
    compute_shap=True,
    compute_conformal=True,
)

results = from_csv("data.csv", target_columns="target", ml_config=ml).run()
```

---

### Group-Aware Modeling

Use when observations from the same group must not appear in both train and test data.

```python
from pipeline import from_dataframe, MLConfig

ml = MLConfig(
    output_dir="group_results",
    include_models=["ridge", "random_forest", "hist_gbrt"],
    n_outer_splits=5,
    n_inner_splits=3,
)

results = from_dataframe(
    df=df,
    target_columns="outcome",
    group_column="patient_id",  # sets use_group_cv=True automatically
    ml_config=ml,
).run()
```

---

### Multiple Target Columns

```python
from pipeline import from_csv, MLConfig

ml = MLConfig(
    output_dir="multi_target_results",
    include_models=["ridge", "random_forest"],
    use_multitask=True,
)

results = from_csv(
    file_path="experiments.csv",
    target_columns=["yield", "purity", "strength"],
    ml_config=ml,
).run()
```

The script trains separate single-target models for each target, and also runs multi-task models if `use_multitask=True`.

---

### Stacking Ensemble

```python
from pipeline import from_csv, MLConfig

ml = MLConfig(
    output_dir="stacking_results",
    include_models=["ridge", "random_forest", "hist_gbrt"],
    use_stacking=True,
    compute_shap=False,
)

results = from_csv("data.csv", target_columns="target", ml_config=ml).run()
```

Stacking fits a `RidgeCV` meta-model on top of OOF predictions from the base models.

---

### Custom Preprocessing Configuration

```python
from pipeline import from_csv, MLConfig, FeatureConfig, TargetConfig

feat = FeatureConfig(
    use_correlation_filter=True,
    correlation_threshold=0.95,
    numeric_scaler="robust",       # robust to outliers
    selectkbest_k=25,
    max_missing_fraction=0.50,     # only keep columns with < 50% missing
    column_type_overrides={
        "sample_id": "drop",
        "zip_code": "categorical",
    },
)

target_cfg = TargetConfig(
    target_columns=["price"],
    target_transforms={"price": "log1p"},
    positive_targets=["price"],
)

ml = MLConfig(
    output_dir="custom_results",
    include_models=["ridge", "lasso", "random_forest", "lightgbm"],
    n_optuna_trials=50,
    compute_shap=True,
)

results = from_csv(
    file_path="housing.csv",
    target_columns="price",
    ml_config=ml,
    feat_config=feat,
    tgt_config=target_cfg,
).run()
```

---

### From an In-Memory DataFrame

```python
import pandas as pd
from pipeline import from_dataframe, MLConfig

df = pd.read_csv("housing.csv")

ml = MLConfig(
    output_dir="housing_results",
    include_models=["ridge", "random_forest", "hist_gbrt"],
    n_optuna_trials=30,
    compute_shap=True,
)

results = from_dataframe(
    df=df,
    target_columns="price",
    ml_config=ml,
).run()

print(results.get_best_per_label())
```

---

### From SQLite

```python
from pipeline import from_sqlite, MLConfig

results = from_sqlite(
    db_path="experiments.sqlite",
    sql_query="SELECT * FROM measurements WHERE split = 'train'",
    target_columns="yield",
    ml_config=MLConfig(output_dir="sqlite_results"),
).run()
```

---

### From a PostgreSQL / SQLAlchemy Database

```python
from pipeline import from_sqlalchemy, MLConfig

results = from_sqlalchemy(
    uri="postgresql://user:password@localhost/mydb",
    sql_query="SELECT * FROM measurements",
    target_columns="efficiency",
    ml_config=MLConfig(output_dir="pg_results"),
).run()
```

---

### Disable Automatic Target Transformation

```python
from pipeline import TargetConfig

target_cfg = TargetConfig(
    target_columns=["target"],
    auto_transform=False,    # keep target as-is
)
```

---

### Use Compositional Features with CLR

Suppose columns named `comp_A`, `comp_B`, `comp_C` are parts of a composition (sum to 1).

```python
from pipeline import FeatureConfig

feat = FeatureConfig(
    apply_clr_transform=True,
    clr_column_pattern="^comp_",
    clr_pseudocount=1e-6,
)
```

---

### Register a Custom Model

```python
from sklearn.ensemble import GradientBoostingRegressor
from pipeline import from_dataframe, MLConfig

ml = MLConfig(include_models=["my_gbr", "ridge"])

pipeline = from_dataframe(df=df, target_columns="target", ml_config=ml)

pipeline.register_model(
    name="my_gbr",
    cls=GradientBoostingRegressor,
)

results = pipeline.run()
```

For models requiring special construction:

```python
def make_model(params):
    return MyRegressor(special_argument=True, **params)

pipeline.register_model(
    name="my_model",
    cls=MyRegressor,
    instantiator=make_model,
)
```

---

### Minimal Practical Template

```python
import pandas as pd
from pipeline import from_dataframe, MLConfig, FeatureConfig, TargetConfig

df = pd.read_csv("my_data.csv")

ml = MLConfig(
    output_dir="my_results",
    include_models=["dummy", "ridge", "random_forest", "hist_gbrt"],
    n_outer_splits=5,
    n_inner_splits=3,
    n_optuna_trials=30,
    compute_shap=True,
    compute_conformal=True,
)

feat = FeatureConfig(
    use_correlation_filter=True,
    correlation_threshold=0.98,
    numeric_scaler="standard",
)

target = TargetConfig(
    target_columns=["target"],
    auto_transform=True,
)

results = from_dataframe(
    df=df,
    target_columns="target",
    ml_config=ml,
    feat_config=feat,
    tgt_config=target,
).run()

summary = results.to_dataframe()
best = results.get_best_per_label(by="oof_rmse")
print(best)
```

---

# 6. Available Models

## 6.1 Always-Available Models

| Model name | Class | Type | Notes |
|---|---|---|---|
| `dummy` | `DummyRegressor` | Baseline | Predicts the mean. Always include this. |
| `ridge` | `Ridge` | Linear (L2) | Fast and reliable. Good first linear baseline. |
| `lasso` | `Lasso` | Linear (L1, sparse) | Performs implicit feature selection; useful with many irrelevant features. |
| `elasticnet` | `ElasticNet` | Linear (L1 + L2) | Combines Ridge and Lasso; use when both sparsity and stability matter. |
| `bayesian_ridge` | `BayesianRidge` | Bayesian linear | Automatically regularizes; provides posterior uncertainty. |
| `huber` | `HuberRegressor` | Robust linear | Less sensitive to outliers in the target than Ridge. |
| `linear_svr` | `LinearSVR` | Linear SVM | Scales well to large datasets; requires scaling. |
| `svr` | `SVR` | Kernel SVM | Can model nonlinear patterns; slow for large datasets. |
| `knn` | `KNeighborsRegressor` | Distance-based | Nonparametric; requires scaling and good feature selection. |
| `random_forest` | `RandomForestRegressor` | Bagged trees | Robust, parallelizable, good default tree model. |
| `extra_trees` | `ExtraTreesRegressor` | Extremely randomized trees | Faster than Random Forest; often comparable accuracy. |
| `hist_gbrt` | `HistGradientBoostingRegressor` | Gradient boosting | Handles missing values natively; often best built-in tree model. |

## 6.2 Optional Models

These are registered only if the package is installed.

| Model name | Package | Notes |
|---|---|---|
| `xgboost` | `xgboost` | Excellent all-around gradient boosting; GPU support available. |
| `lightgbm` | `lightgbm` | Very fast on large datasets; good default for competitions. |
| `catboost` | `catboost` | Native categorical feature support; minimal preprocessing needed. |
| `tabpfn` | `tabpfn` | Transformer-based; best for small datasets (< ~1000 rows); no HPO. |

```bash
pip install xgboost lightgbm catboost
```

## 6.3 What Preprocessing Do Models Need?

All preprocessing is handled automatically. Here is what the pipeline applies per model family:

**Linear and distance/kernel models** (`ridge`, `lasso`, `elasticnet`, `bayesian_ridge`, `huber`, `linear_svr`, `svr`, `knn`):

```
numeric:     impute → variance filter → scale → optional correlation filter
categorical: impute → one-hot encode → optional scale
datetime:    expand → scale
```

These models require scaling because they are sensitive to feature magnitude.

**Tree-based models** (`random_forest`, `extra_trees`, `hist_gbrt`, `xgboost`, `lightgbm`, `catboost`):

```
numeric:     impute → optional variance filter
categorical: impute → ordinal encode
datetime:    expand
```

Tree models do **not** require scaling. Categorical columns are ordinal-encoded (not one-hot) because trees can handle ordinal integers efficiently and one-hot encoding is wasteful for trees.

**`tabpfn`**: Receives `full` strategy, no HPO tuning.

**`dummy`**: Requires only a valid numeric target and enough rows for splitting.

---

# 7. Feature Engineering & Preprocessing

## 7.1 Custom Feature Engineering

The default `PassthroughEngineer` only sanitizes column names. You can add domain-specific features by subclassing `FeatureEngineer`:

```python
from pipeline import FeatureEngineer

class MyFeatureEngineer(FeatureEngineer):
    def engineer(self, df):
        df = df.copy()
        df["area"] = df["length"] * df["width"]
        df["income_debt_ratio"] = df["income"] / df["debt"].replace(0, float("nan"))
        return df
```

Pass it to `TabularMLPipeline` directly.

## 7.2 Numeric Preprocessing

```
SimpleImputer (median)
    → VarianceThreshold
    → Scaler (standard / robust / quantile / none)
    → CorrelationFilter (optional)
```

Scaling options:

| `numeric_scaler` | Transformer | Formula |
|---|---|---|
| `"standard"` | `StandardScaler` | $z = (x - \mu) / \sigma$ |
| `"robust"` | `RobustScaler` | $z = (x - \text{median}) / \text{IQR}$ |
| `"quantile"` | `QuantileTransformer` | Maps to normal distribution |
| `"none"` | No scaling | — |

## 7.3 Feature Strategies

A feature strategy is a named preprocessing + selection configuration applied on top of the base preprocessor.

**Default strategies for linear models:**

| Strategy | What it does | When to use |
|---|---|---|
| `full_scaled` | All features, standard scaling | Default for linear models |
| `full_robust` | All features, robust scaling | When data has heavy outliers |
| `selectkbest` | Top $k$ features by F-regression | Many features, want interpretability |
| `selectkbest_mi` | Top $k$ features by mutual information | Nonlinear dependencies |
| `pca` | PCA after preprocessing | Highly correlated features (e.g., spectroscopy) |

**Default strategies for tree models:**

| Strategy | What it does | When to use |
|---|---|---|
| `full` | All preprocessed features | Default |
| `tree_select` | ExtraTrees importance selection | Many redundant features |

**Default strategies for `dummy` and `tabpfn`:** `full` only.

**Override strategies** — force all models to use the same strategies:

```python
ml = MLConfig(
    feature_strategies_override=["full", "selectkbest"],
)
```

> **Note:** Be careful overriding strategies across model families. `pca` and `full_scaled` are not meaningful for tree models, and `tree_select` requires a tree pre-training step which adds significant compute.

## 7.4 Datetime Expansion

`DatetimeExpander` converts each datetime column into 10 numeric features:

| Expanded feature | Example (`order_date` = 2024-03-15 14:30:00) |
|---|---|
| `order_date__year` | 2024 |
| `order_date__month` | 3 |
| `order_date__day` | 15 |
| `order_date__dayofweek` | 4 (Friday) |
| `order_date__hour` | 14 |
| `order_date__minute` | 30 |
| `order_date__second` | 0 |
| `order_date__dayofyear` | 75 |
| `order_date__quarter` | 1 |
| `order_date__is_weekend` | 0 |

## 7.5 CLR Transform for Compositional Data

Use when features represent parts of a whole (chemical compositions, microbiome proportions, budget shares, mineral fractions). The **Centered Log-Ratio (CLR)** transform converts compositional proportions into unconstrained real-valued coordinates:

$$
\operatorname{clr}(x_i) = \log(x_i) - \frac{1}{D}\sum_{j=1}^{D}\log(x_j)
$$

where $D$ is the number of compositional parts. A pseudocount is added before taking the log to handle exact zeros.

```python
feat = FeatureConfig(
    apply_clr_transform=True,
    clr_column_pattern="^comp_",   # regex matching compositional columns
    clr_pseudocount=1e-6,
)
```

---

# 8. Hyperparameter Tuning

## 8.1 How Nested CV + Optuna Works

```
For each outer fold:
    1. Isolate calibration set from outer training data (before HPO)
    2. Run Optuna study on remaining outer training data:
           For each trial:
               suggest hyperparameters (TPE sampler)
               inner KFold / GroupKFold
               build fresh pipeline, fit on inner train, score on inner val
               report average inner RMSE to Optuna
    3. Refit best pipeline on full outer training set
    4. Predict on outer test fold
    5. Calibrate conformal predictor on calibration set
```

**Optuna algorithm:** `TPESampler` (Tree-structured Parzen Estimator) — a Bayesian optimization algorithm that builds a probabilistic model of good vs. bad hyperparameter regions and samples accordingly.

**Pruning:** `MedianPruner` stops unpromising trials early after the first inner fold, saving substantial computation.

**Optimization objective:** Minimize average inner-fold RMSE.

## 8.2 Tuned Hyperparameters by Model

<details>
<summary><strong>Linear models</strong></summary>

| Model | Hyperparameter | Search range | Notes |
|---|---|---|---|
| `ridge` | `alpha` | $[10^{-5}, 10^4]$ (log) | Regularization strength. Higher = more shrinkage. |
| `lasso` | `alpha` | $[10^{-5}, 10]$ (log) | Sparsity strength. |
| `elasticnet` | `alpha` | $[10^{-5}, 10]$ (log) | Overall regularization. |
| `elasticnet` | `l1_ratio` | $[0, 1]$ | Mix of L1 and L2. `1.0` = pure Lasso. |
| `bayesian_ridge` | `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` | $[10^{-7}, 10^{-3}]$ (log) | Gamma prior shape/rate parameters. |
| `huber` | `epsilon` | $[1.01, 5]$ | Robustness threshold. Lower = more robust. |
| `huber` | `alpha` | $[10^{-5}, 10]$ (log) | Regularization. |

</details>

<details>
<summary><strong>Support vector models</strong></summary>

| Model | Hyperparameter | Search range | Notes |
|---|---|---|---|
| `linear_svr` | `C` | $[10^{-3}, 10^3]$ (log) | Regularization; lower = stronger. |
| `linear_svr` | `epsilon` | $[10^{-4}, 1]$ (log) | Tube width in SVR loss. |
| `svr` | `C` | $[10^{-3}, 10^3]$ (log) | |
| `svr` | `epsilon` | $[10^{-4}, 1]$ (log) | |
| `svr` | `kernel` | `rbf`, `poly` | `rbf` is usually better for smooth data. |
| `svr` | `gamma` | `scale`, `auto` | |

</details>

<details>
<summary><strong>KNN</strong></summary>

| Model | Hyperparameter | Search range | Notes |
|---|---|---|---|
| `knn` | `n_neighbors` | $[2, 50]$ | More neighbors = smoother predictions. |
| `knn` | `weights` | `uniform`, `distance` | `distance` weights closer neighbors more. |
| `knn` | `p` | 1, 2 | Distance metric: 1 = Manhattan, 2 = Euclidean. |

</details>

<details>
<summary><strong>Tree ensembles</strong></summary>

| Model | Hyperparameter | Search range |
|---|---|---|
| `random_forest` / `extra_trees` | `n_estimators` | 100–1000 (step 100) |
| | `max_depth` | 3–30 |
| | `min_samples_split` | 2–20 |
| | `min_samples_leaf` | 1–10 |
| | `max_features` | `sqrt`, `log2`, `None` |
| `hist_gbrt` | `learning_rate` | $[0.005, 0.3]$ (log) |
| | `max_depth` | 3–20 |
| | `max_iter` | 100–1000 (step 100) |
| | `min_samples_leaf` | 5–50 |
| | `l2_regularization` | $[10^{-8}, 10]$ (log) |
| | `max_leaf_nodes` | 15–255 |

</details>

<details>
<summary><strong>Gradient boosting libraries</strong></summary>

| Model | Hyperparameter | Search range |
|---|---|---|
| `xgboost` | `n_estimators` | 100–2000 |
| | `max_depth` | 2–10 |
| | `learning_rate` | $[0.005, 0.3]$ (log) |
| | `subsample` | 0.5–1.0 |
| | `colsample_bytree` | 0.3–1.0 |
| | `min_child_weight` | 1–20 |
| | `reg_alpha`, `reg_lambda` | $[10^{-8}, 10]$ (log) |
| | `gamma` | $[10^{-8}, 5]$ (log) |
| `lightgbm` | `n_estimators` | 100–2000 |
| | `learning_rate` | $[0.005, 0.3]$ (log) |
| | `max_depth` | 3–15 |
| | `num_leaves` | 15–255 |
| | `subsample`, `colsample_bytree` | 0.5–1.0 / 0.3–1.0 |
| | `reg_alpha`, `reg_lambda` | $[10^{-8}, 10]$ (log) |
| | `min_child_samples` | 5–100 |
| `catboost` | `iterations` | 100–2000 |
| | `depth` | 3–10 |
| | `learning_rate` | $[0.005, 0.3]$ (log) |
| | `l2_leaf_reg` | $[10^{-8}, 20]$ (log) |
| | `subsample` | 0.5–1.0 |
| | `bagging_temperature` | 0.0–1.0 |

</details>

---

# 9. Evaluation & Metrics

## 9.1 Regression Metrics

| Metric | Formula | Lower = better? | Notes |
|---|---|---|---|
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | No (higher = better) | 1 = perfect, 0 = mean predictor, < 0 = worse than mean |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Yes | Interpretable; average error in target units |
| MedAE | $\operatorname{median}_i\|y_i - \hat{y}_i\|$ | Yes | Robust to outliers; better than MAE when extreme errors occur |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Yes | Penalizes large errors heavily |
| RMSE | $\sqrt{\text{MSE}}$ | Yes | Same units as target; main HPO objective |
| MAPE | $\frac{1}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Yes | Percentage error; unstable when $y_i \approx 0$ |

## 9.2 Out-of-Fold vs. Mean-Fold Metrics

The pipeline reports both:

- **Mean-fold metrics** (`mean_r2`, `mean_rmse`, `std_rmse`, etc.) — the average of per-fold metrics.
- **OOF metrics** (`oof_r2`, `oof_rmse`, `oof_mae`, `oof_mape`) — computed on the full concatenated OOF predictions.

**OOF metrics are generally more informative** because they capture the full distribution of errors (including rare test cases), while mean-fold metrics can be misleading when fold sizes are unequal or when outlier folds dominate.

## 9.3 Bootstrap Confidence Intervals

For RMSE, the pipeline computes a 95% bootstrap confidence interval over the per-fold RMSE values:

```
ci95_rmse_lo    ci95_rmse_hi
```

Use this to assess how stable the RMSE estimate is across folds. Wide intervals suggest high variance across folds (possibly due to small dataset size or strong group structure).

## 9.4 Conformal Coverage

The pipeline reports:

| Column | Meaning |
|---|---|
| `mean_conformal_coverage` | Fraction of test values that fall inside the predicted interval (target: $1 - \alpha$) |
| `mean_conformal_width` | Average interval width in original target units |

A well-calibrated model will show `mean_conformal_coverage` ≈ $1 - \alpha$. Coverage significantly below the target indicates the model is overconfident on the test fold.

---

# 10. Output Files

The pipeline creates the following structure under `output_dir` (default: `ml_results/`):

```
ml_results/
├── config_snapshot.json       ← all config parameters + dataset shape
├── results_summary.csv        ← one row per target × model × strategy
├── results_summary.parquet    ← same, in Parquet format
├── fold_results.csv           ← one row per outer fold
├── feature_stability.csv      ← feature selection frequency across folds
├── models/                    ← fitted pipelines as .pkl files
│   └── price__ridge__full_scaled__fold1.pkl
├── shap/                      ← SHAP importance CSVs per fold
│   └── shap__price__ridge__full_scaled__fold1.csv
├── fold_predictions/          ← OOF prediction CSVs per model/strategy
│   └── preds__price__ridge__full_scaled.csv
└── permutation/               ← permutation importance CSVs (if enabled)
    └── perm__price__ridge__full_scaled__fold1.csv
```

## 10.1 `results_summary.csv` — Key Columns

| Column | Meaning |
|---|---|
| `label` | Target column name |
| `model` | Model name |
| `feature_strategy` | Feature strategy used |
| `target_transform` | Target transform applied |
| `n_samples` | Valid rows for this target |
| `mean_r2` / `std_r2` | Mean and std of fold $R^2$ |
| `mean_rmse` / `std_rmse` | Mean and std of fold RMSE |
| `mean_mae` / `std_mae` | Mean and std of fold MAE |
| `oof_r2` | Out-of-fold $R^2$ |
| `oof_rmse` | Out-of-fold RMSE |
| `oof_mae` | Out-of-fold MAE |
| `oof_mape` | Out-of-fold MAPE |
| `ci95_rmse_lo` / `ci95_rmse_hi` | 95% bootstrap CI for fold RMSE |
| `mean_conformal_coverage` | Mean fraction of test values inside prediction intervals |
| `mean_conformal_width` | Mean interval width |

## 10.2 `fold_results.csv`

One row per outer fold. Contains fold-level metrics, train/test sizes, runtime, best hyperparameters (JSON string), and conformal coverage/width per fold.

## 10.3 `fold_predictions/`

OOF prediction CSVs. Columns: `label`, `model`, `feature_strategy`, `fold`, `y_true`, `y_pred`.

Use these to plot residuals, examine failure cases, or build custom ensembles.

## 10.4 `models/`

Fitted scikit-learn pipelines (full preprocessing + model) saved as pickle files. These can be loaded and used directly for inference:

```python
import pickle
import pandas as pd

with open("ml_results/models/price__ridge__full_scaled__fold1.pkl", "rb") as f:
    pipe = pickle.load(f)

new_preds = pipe.predict(new_df)
```

---

# 11. Advanced Features

## 11.1 Multi-Target Regression

When ≥ 2 targets are specified and `use_multitask=True`, the pipeline runs:

- `MultiOutputRegressor` wrapping `ExtraTreesRegressor` — fits independent models per target but evaluates jointly.
- `RegressorChain` wrapping `Ridge` — chains targets so earlier target predictions become features for later targets. The chain order is random; useful when targets are correlated.

Results appear in `results_summary.csv` with `model` = `multitask_extra_trees_multioutput` or `multitask_ridge_chain`.

## 11.2 Stacking Ensemble

When `use_stacking=True`, the pipeline:

1. Collects OOF predictions from all base models (excluding `dummy`, `svr`, `linear_svr`, `tabpfn`).
2. Trains a `RidgeCV` meta-model on those OOF predictions.
3. Reports the stacking result in `results_summary.csv` with `model` = `stacking_ridgecv`.

The meta-model is trained on OOF predictions, so there is no train/test leakage.

## 11.3 Feature Stability Tracking

When `track_feature_stability=True`, the pipeline records which features survive selection in each outer fold for strategies like `tree_select`, `selectkbest`, and `selectkbest_mi`.

Output: `feature_stability.csv` with columns `label`, `model`, `strategy`, `feature`, `stability_score` (fraction of folds in which the feature was selected).

Features with `stability_score = 1.0` are selected in every fold — those are the most reliable. Features with `stability_score < 0.3` are marginal and may be excluded safely.

---

# 12. Function & Class Reference

<details>
<summary><strong>Utility functions</strong></summary>

| Function | Purpose |
|---|---|
| `sanitize_name(name)` | Converts a string to a safe Python identifier |
| `sanitize_dataframe_columns(df)` | Sanitizes and deduplicates all column names in a DataFrame |
| `bootstrap_ci(data, stat, n_resamples, alpha)` | Computes a percentile bootstrap confidence interval |
| `compute_metrics(y_true, y_pred)` | Returns dict with `r2`, `mae`, `medae`, `mse`, `rmse`, `mape` |
| `detect_abs_skewness(y)` | Returns the absolute skewness of a pandas Series |

</details>

<details>
<summary><strong>Data source classes</strong></summary>

| Class | Purpose |
|---|---|
| `DataSource` | Abstract base class |
| `InMemoryDataSource` | Loads from an existing DataFrame |
| `FileDataSource` | Loads CSV, TSV, Excel, Parquet, Feather, HDF5, JSON, JSONL |
| `SQLiteDataSource` | Loads from a SQLite query |
| `SQLAlchemyDataSource` | Loads from any SQLAlchemy-compatible database |
| `make_data_source(config)` | Factory selecting the right source from a `DataSourceConfig` |

</details>

<details>
<summary><strong>Feature engineering & label classes</strong></summary>

| Class | Purpose |
|---|---|
| `FeatureEngineer` | Abstract base class — subclass to add custom features |
| `PassthroughEngineer` | Default: only sanitizes column names |
| `LabelExtractor` | Abstract base class for target extraction |
| `ColumnLabelExtractor` | Extracts named target columns from the DataFrame |

</details>

<details>
<summary><strong>Dataset classes</strong></summary>

| Class | Purpose |
|---|---|
| `Dataset` | Container: features, targets, column roles, groups, weights |
| `DatasetBuilder` | Composes DataSource + FeatureEngineer + LabelExtractor → Dataset |

</details>

<details>
<summary><strong>Preprocessing classes</strong></summary>

| Class | Purpose |
|---|---|
| `TypeInferenceEngine` | Classifies columns as numeric / categorical / datetime / drop |
| `PreprocessorBuilder` | Builds a column-type-aware `ColumnTransformer` |
| `DatetimeExpander` | Expands datetime columns into 10 numeric parts |
| `CLRTransformer` | Centered log-ratio transform for compositional data |
| `ILRTransformer` | Isometric log-ratio transform (defined but not used by default pipeline) |
| `CorrelationFilter` | Removes highly correlated features (leakage-free, per fold) |
| `TargetTransformer` | Applies and inverts per-fold target transformations |

</details>

<details>
<summary><strong>Feature selection classes</strong></summary>

| Class | Purpose |
|---|---|
| `TreeFeatureSelector` | ExtraTrees-based feature selector (`tree_select` strategy) |
| `MutualInfoSelector` | Mutual information feature selector (`selectkbest_mi` strategy) |
| `SelectKBest` | F-regression feature selector (`selectkbest` strategy) |
| `PCA` | Dimensionality reduction (`pca` strategy) |

</details>

<details>
<summary><strong>Modeling & HPO classes</strong></summary>

| Class | Purpose |
|---|---|
| `ModelRegistry` | Registers and instantiates models; supports custom models |
| `PipelineFactory` | Builds full preprocessing + selection + model pipelines |
| `OptunaTuner` | Runs inner-CV Bayesian HPO |
| `NestedCVRunner` | Core engine: outer CV loop + per-fold calibration/HPO/evaluation |
| `MultiTaskRunner` | Runs multi-target models |
| `StackingRunner` | Runs stacking ensemble on OOF predictions |
| `TabularMLPipeline` | Top-level orchestrator |

</details>

<details>
<summary><strong>Results & interpretation classes</strong></summary>

| Class | Purpose |
|---|---|
| `FoldResult` | Stores one fold's metrics and predictions |
| `ExperimentResult` | Stores one model/strategy/target experiment; calls `.summarize()` |
| `ResultsManager` | Aggregates results; saves CSV/Parquet/pkl; prints summary |
| `SHAPComputer` | Computes and saves SHAP importances per fold |
| `FeatureStabilityTracker` | Tracks feature selection frequency across folds |
| `ConformalPredictor` | Split conformal prediction intervals |

</details>

---

# 13. Recommended Workflows

## 13.1 Step 1 — Sanity Check (5 minutes)

Always start here to verify data loads correctly and get a rough baseline:

```python
ml = MLConfig(
    output_dir="explore",
    include_models=["dummy", "ridge"],
    n_outer_splits=3,
    n_inner_splits=2,
    n_optuna_trials=5,
    compute_shap=False,
    compute_conformal=False,
)
```

Check: Does it run? What is the `dummy` RMSE? Is it much better with `ridge`?

## 13.2 Step 2 — Broader Exploration

Once data loads cleanly, compare more model families:

```python
ml = MLConfig(
    output_dir="explore2",
    include_models=["ridge", "random_forest", "hist_gbrt"],
    n_outer_splits=3,
    n_inner_splits=2,
    n_optuna_trials=20,
    compute_shap=False,
)
```

## 13.3 Step 3 — Full Benchmark

Run a thorough comparison with all desired models:

```python
ml = MLConfig(
    output_dir="benchmark",
    include_models=[
        "ridge", "lasso", "elasticnet",
        "random_forest", "extra_trees",
        "hist_gbrt", "xgboost", "lightgbm", "catboost",
    ],
    n_outer_splits=5,
    n_inner_splits=4,
    n_optuna_trials=60,
    compute_shap=True,
    compute_conformal=True,
)
```

## 13.4 Step 4 — Select Best Model

```python
summary = results.to_dataframe()
best = results.get_best_per_label(by="oof_rmse")
print(best)
```

Choose models based on:
1. Low `oof_rmse` and high `oof_r2`
2. Stable performance (`std_rmse` not too large relative to `mean_rmse`)
3. Reasonable conformal coverage (close to $1 - \alpha$)
4. Interpretability requirements (SHAP / feature stability)
5. Inference time constraints

## 13.5 Runtime Estimation

Total training cost scales roughly as:

$$
N_{\text{targets}} \times N_{\text{models}} \times N_{\text{strategies}} \times N_{\text{outer}} \times N_{\text{trials}} \times N_{\text{inner}}
$$

For example, 2 targets × 6 models × 2 strategies × 5 outer folds × 60 trials × 4 inner folds = **14,400 model fits**. This can run anywhere from minutes to hours depending on dataset size and model type. Reduce `n_optuna_trials`, `n_outer_splits`, or `include_models` to control runtime.

---

# 14. Common Errors & Fixes

<details>
<summary><strong>Target column not found</strong></summary>

**Error:**
```
Target columns not found after sanitisation
```

**Cause:** Target name misspelled, or column name changes during sanitization (e.g., `"Sale Price"` → `Sale_Price`).

**Fix:**
```python
print(df.columns.tolist())
# pass the original name; sanitization is applied internally
results = from_dataframe(df, target_columns="Sale Price")
```

</details>

<details>
<summary><strong>No usable feature columns</strong></summary>

**Cause:** All columns were dropped (too much missing data, high cardinality, or all columns excluded).

**Fix:**
```python
feat = FeatureConfig(
    max_missing_fraction=0.99,
    max_cardinality_for_categorical=200,
)
```
Or use `column_type_overrides` to explicitly force columns to `"numeric"` or `"categorical"`.

</details>

<details>
<summary><strong>Optional model not available</strong></summary>

**Cause:** Requesting `lightgbm`, `xgboost`, or `catboost` without installing the package.

**Fix:**
```bash
pip install lightgbm xgboost catboost
```

</details>

<details>
<summary><strong>Too few samples</strong></summary>

**Cause:** Target has fewer valid rows than `min_samples_per_label` (default: 30).

**Fix:**
```python
ml = MLConfig(min_samples_per_label=10)
```
Use with caution — very few samples make CV estimates unreliable.

</details>

<details>
<summary><strong>Group CV not activating</strong></summary>

**Cause:** `use_group_cv=True` but too few groups for the number of folds, or group column not found.

**Symptoms:** Log message saying group CV was skipped; falls back to regular KFold.

**Fix:** Ensure the group column exists in the DataFrame, and that the number of unique groups ≥ `n_outer_splits`. If not, reduce `n_outer_splits`.

</details>

<details>
<summary><strong>SHAP is very slow</strong></summary>

**Cause:** `KernelExplainer` fallback for non-tree, non-linear models.

**Fix:**
```python
ml = MLConfig(
    compute_shap=False,   # disable entirely, or
    shap_sample_size=50,  # reduce sample size
    shap_kernel_fallback=False,  # skip KernelExplainer for non-tree/linear models
)
```

</details>

<details>
<summary><strong>MAPE is very large or NaN</strong></summary>

**Cause:** Target contains values near zero.

**Fix:** Use RMSE or MAE instead of MAPE as your primary metric. MAPE is included for completeness but is unreliable for targets that pass through zero.

</details>

---

# 15. Best Practices

1. **Always include `dummy`** — it tells you whether real models improve over a simple mean predictor. A model that barely beats `dummy` is a red flag.

2. **Use group CV when rows are not independent** — for patients, batches, machines, sensors, experiments, or any repeated-measurement design. Not using group CV in these settings will make all models appear better than they truly are.

3. **Start fast, scale up** — use `n_optuna_trials=10` and 2–3 models during debugging. Only increase trials for final benchmarking.

4. **Prefer `oof_rmse` over `mean_rmse`** for model selection — it is more robust and captures the full error distribution.

5. **Examine fold-level stability** — a model with `mean_rmse=0.5` and `std_rmse=0.4` is far less reliable than one with `mean_rmse=0.6` and `std_rmse=0.05`.

6. **Inspect OOF predictions** — load files from `fold_predictions/` and plot residuals. Look for systematic patterns (e.g., errors concentrated in a specific target range) that suggest missing features or model misspecification.

7. **Drop leaky columns explicitly:**

   ```python
   from_dataframe(
       df=df,
       target_columns="revenue",
       drop_columns=["revenue_after_tax", "future_status", "target_encoded_col"],
   )
   ```

8. **Check conformal coverage** — if `mean_conformal_coverage` is much lower than $1 - \alpha$, the model is overconfident on the test fold (likely due to distribution shift or too small a calibration set).

9. **Use `robust` scaler for outlier-heavy industrial or sensor data** — it is more stable than standard Z-score normalization when sensor spikes are present.

10. **For high-dimensional data (hundreds of features), enable correlation filtering and feature selection** — this prevents the curse of dimensionality and often improves both speed and accuracy:

    ```python
    feat = FeatureConfig(
        use_correlation_filter=True,
        correlation_threshold=0.95,
        selectkbest_k=40,
    )
    ml = MLConfig(
        feature_strategies_override=["selectkbest", "selectkbest_mi", "tree_select"],
    )
    ```
