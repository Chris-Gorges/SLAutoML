# User Manual — `SLAutoML` Supervised Learning Script

This script is a **supervised tabular regression pipeline**. It loads tabular data, extracts target columns, preprocesses features, runs multiple regression models, tunes hyperparameters with nested cross-validation, evaluates performance, saves predictions/models/results, and optionally computes SHAP explanations, conformal prediction intervals, multi-task models, and stacking ensembles.

> **Important:** This pipeline is for **regression**, not classification. Targets must be numeric or convertible to numeric.

---

## 1. What the Script Does

At a high level, the pipeline performs:

1. **Data loading**
   - From CSV, TSV, Excel, Parquet, Feather, HDF5, JSON, JSONL, SQLite, SQLAlchemy databases, or an in-memory `pandas.DataFrame`.

2. **Data engineering**
   - Sanitizes column names.
   - Extracts target columns.
   - Drops target, group, weight, and user-specified columns from features.
   - Infers feature column types.

3. **Feature engineering / preprocessing**
   - Numeric imputation, scaling, variance filtering, optional correlation filtering.
   - Categorical imputation and encoding.
   - Datetime expansion into numeric date parts.
   - Optional CLR transform for compositional data.
   - Optional feature selection or PCA.

4. **Model training**
   - Runs many regression models.
   - Uses nested cross-validation for unbiased performance estimation.
   - Uses Optuna for Bayesian hyperparameter tuning.

5. **Evaluation**
   - Measures $R^2$, MAE, median absolute error, MSE, RMSE, and MAPE.
   - Produces out-of-fold predictions.
   - Computes bootstrap confidence intervals for RMSE.

6. **Explainability and uncertainty**
   - Optional SHAP feature importance.
   - Optional permutation importance.
   - Optional split conformal prediction intervals.

7. **Optional advanced modes**
   - Multi-target regression.
   - Stacking ensemble.

---

## 2. Main Supervised Learning Concepts in This Script

| Supervised learning stage | Script components |
|---|---|
| Data loading | `DataSource`, `FileDataSource`, `SQLiteDataSource`, `SQLAlchemyDataSource`, `InMemoryDataSource` |
| Data engineering | `DatasetBuilder`, `FeatureEngineer`, `LabelExtractor` |
| Feature engineering | `PreprocessorBuilder`, `DatetimeExpander`, `CLRTransformer`, `CorrelationFilter` |
| Feature selection | `TreeFeatureSelector`, `MutualInfoSelector`, `SelectKBest`, `PCA` |
| Model selection | `ModelRegistry`, `PipelineFactory` |
| Hyperparameter tuning | `OptunaTuner` |
| Train/test splitting | `NestedCVRunner` |
| Evaluation | `compute_metrics`, `ExperimentResult`, `ResultsManager` |
| Explainability | `SHAPComputer`, permutation importance |
| Uncertainty intervals | `ConformalPredictor` |
| Multi-target learning | `MultiTaskRunner` |
| Stacking | `StackingRunner` |
| Orchestration | `TabularMLPipeline` |

---

## 3. Quick Start

### 3.1 Run the built-in demo

Save the script as, for example, `pipeline.py`, then run:

```bash
python pipeline.py demo
```

This uses the California Housing dataset and trains:

- `dummy`
- `ridge`
- `random_forest`
- `hist_gbrt`

Results are written to:

```text
ml_results_demo/
```

---

### 3.2 Run on a CSV file

```bash
python pipeline.py csv housing.csv price
```

For multiple targets:

```bash
python pipeline.py csv experiments.csv yield purity
```

For TSV:

```bash
python pipeline.py tsv data.tsv target
```

---

### 3.3 Use from Python with a DataFrame

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

## 4. Installation Requirements

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
| Parquet | `pyarrow` or `fastparquet` |
| Feather | `pyarrow` |
| Excel | `openpyxl` |
| HDF5 | `tables` |
| SQLAlchemy databases | `sqlalchemy` plus database driver |

---

## 5. Configuration Classes

The script uses three main configuration dataclasses.

---

## 5.1 `DataSourceConfig`

Controls how data is loaded.

Important fields:

| Field | Purpose |
|---|---|
| `file_path` | Path to CSV, TSV, Excel, Parquet, JSON, etc. |
| `file_format` | Optional explicit format |
| `csv_sep` | CSV separator |
| `csv_encoding` | CSV encoding |
| `excel_sheet` | Excel sheet name or index |
| `sqlite_path` | SQLite database path |
| `sqlalchemy_uri` | SQLAlchemy database URI |
| `sql_query` | SQL query |

Usually, you do not create this directly. Use helper functions like `from_csv`, `from_dataframe`, or `from_sqlite`.

---

## 5.2 `FeatureConfig`

Controls feature preprocessing and feature selection.

Important fields:

| Field | Default | Meaning |
|---|---:|---|
| `max_cardinality_for_categorical` | `50` | Integer/string columns with at most this many unique values may be treated as categorical |
| `max_missing_fraction` | `0.90` | Columns with more missing values than this are dropped |
| `expand_datetime` | `True` | Expand dates into year, month, day, etc. |
| `numeric_imputer_strategy` | `"median"` | Missing numeric value strategy |
| `categorical_imputer_strategy` | `"most_frequent"` | Missing categorical value strategy |
| `numeric_scaler` | `"standard"` | Numeric scaling strategy |
| `use_variance_threshold` | `True` | Remove almost-constant numeric features |
| `use_correlation_filter` | `False` | Remove highly correlated features |
| `correlation_threshold` | `0.98` | Correlation cutoff |
| `selectkbest_k` | `30` | Number of features for `selectkbest` strategies |
| `tree_selector_threshold` | `"median"` | Threshold for tree-based feature selector |
| `pca_variance_ratio` | `0.95` | PCA keeps this fraction of variance |
| `apply_clr_transform` | `False` | Enable centered log-ratio transform |
| `clr_column_pattern` | `""` | Regex selecting compositional features |
| `column_type_overrides` | `{}` | Force columns to numeric/categorical/drop |

Example:

```python
from pipeline import FeatureConfig

feat = FeatureConfig(
    use_correlation_filter=True,
    correlation_threshold=0.95,
    numeric_scaler="robust",
    selectkbest_k=20,
    column_type_overrides={
        "zip_code": "categorical",
        "internal_id": "drop",
    },
)
```

---

## 5.3 `TargetConfig`

Controls target/label handling.

Important fields:

| Field | Default | Meaning |
|---|---:|---|
| `target_columns` | `[]` | Target columns to predict |
| `target_transforms` | `{}` | Explicit transform per target |
| `auto_transform` | `True` | Automatically transform skewed targets |
| `skew_threshold` | `1.0` | Absolute skewness threshold |
| `preferred_auto_transform` | `"yeo-johnson"` | Preferred transform for skewed targets |
| `positive_targets` | `None` | Targets allowed to use `log1p` |

Available target transforms:

| Transform | Use case |
|---|---|
| `"none"` | No target transformation |
| `"log1p"` | Non-negative, strongly right-skewed targets |
| `"yeo-johnson"` | Skewed targets, can handle zero/negative values |
| `"box-cox"` | Strictly positive skewed targets |
| `"quantile-normal"` | Nonlinear transform to approximately normal target distribution |

Example:

```python
from pipeline import TargetConfig

target_cfg = TargetConfig(
    target_columns=["price"],
    target_transforms={
        "price": "log1p",
    },
    positive_targets=["price"],
)
```

---

## 5.4 `MLConfig`

Controls model training, cross-validation, hyperparameter tuning, output, SHAP, conformal prediction, and advanced modes.

Important fields:

| Field | Default | Meaning |
|---|---:|---|
| `random_state` | `42` | Random seed |
| `n_outer_splits` | `5` | Outer CV folds |
| `n_inner_splits` | `4` | Inner CV folds for hyperparameter tuning |
| `n_optuna_trials` | `60` | Number of Optuna trials per model/strategy/fold |
| `optuna_timeout_sec` | `None` | Optional time limit per Optuna study |
| `use_group_cv` | `False` | Use group-aware splitting |
| `group_column` | `None` | Group column name |
| `sample_weight_column` | `None` | Sample weight column name |
| `drop_columns` | `[]` | Columns to exclude from features |
| `n_jobs` | `-1` | Parallel jobs where supported |
| `output_dir` | `"ml_results"` | Output directory |
| `compute_conformal` | `True` | Compute conformal prediction intervals |
| `conformal_alpha` | `0.10` | Miscoverage level; `0.10` means nominal $90\%$ intervals |
| `compute_shap` | `True` | Compute SHAP explanations |
| `compute_permutation_importance` | `False` | Compute permutation importance |
| `use_multitask` | `False` | Enable multi-target models |
| `use_stacking` | `False` | Enable stacking |
| `include_models` | `None` | If set, only run these models |
| `exclude_models` | `[]` | Models to skip |
| `feature_strategies_override` | `None` | Override automatic feature strategies |
| `min_samples_per_label` | `30` | Minimum valid rows per target |

Example:

```python
from pipeline import MLConfig

ml = MLConfig(
    output_dir="results_fast",
    include_models=["ridge", "random_forest", "lightgbm"],
    n_outer_splits=5,
    n_inner_splits=3,
    n_optuna_trials=25,
    compute_shap=False,
    compute_conformal=True,
)
```

---

# 6. Available Machine Learning Models

The model registry includes built-in scikit-learn models and optional external models.

## 6.1 Always available models

| Model name | Class | Type |
|---|---|---|
| `dummy` | `DummyRegressor` | Baseline |
| `ridge` | `Ridge` | Linear regularized regression |
| `lasso` | `Lasso` | Sparse linear regression |
| `elasticnet` | `ElasticNet` | Linear regression with L1/L2 penalty |
| `bayesian_ridge` | `BayesianRidge` | Bayesian linear regression |
| `huber` | `HuberRegressor` | Robust linear regression |
| `linear_svr` | `LinearSVR` | Linear support vector regression |
| `svr` | `SVR` | Kernel support vector regression |
| `knn` | `KNeighborsRegressor` | Distance-based regression |
| `random_forest` | `RandomForestRegressor` | Bagged decision trees |
| `extra_trees` | `ExtraTreesRegressor` | Extremely randomized trees |
| `hist_gbrt` | `HistGradientBoostingRegressor` | Gradient boosting |

---

## 6.2 Optional models

These are registered only if the package is installed.

| Model name | Package | Class |
|---|---|---|
| `xgboost` | `xgboost` | `xgb.XGBRegressor` |
| `lightgbm` | `lightgbm` | `lgb.LGBMRegressor` |
| `catboost` | `catboost` | `CatBoostRegressor` |
| `tabpfn` | `tabpfn` | `TabPFNRegressor` |

Example:

```bash
pip install xgboost lightgbm catboost
```

---

# 7. What Inputs Do Models Need?

All models receive:

- `X`: feature table
- `y`: numeric regression target

The script automatically converts raw tabular data into model-ready numeric arrays using the preprocessing pipeline.

## 7.1 Linear and distance/kernel models

Models:

- `ridge`
- `lasso`
- `elasticnet`
- `bayesian_ridge`
- `huber`
- `linear_svr`
- `svr`
- `knn`

These generally need:

- Numeric features
- Scaled features
- One-hot encoded categorical variables
- Imputed missing values

The script handles this automatically:

```text
numeric columns:
    impute -> variance filter -> scale

categorical columns:
    impute -> one-hot encode -> optional scale

datetime columns:
    expand -> scale
```

These models are sensitive to feature scale, so the pipeline uses scaling for them.

---

## 7.2 Tree-based models

Models:

- `random_forest`
- `extra_trees`
- `hist_gbrt`
- `xgboost`
- `lightgbm`
- `catboost`

These generally need:

- Numeric features
- Missing values handled by the preprocessor
- Categorical variables encoded numerically

The script uses:

```text
numeric columns:
    impute -> optional variance filter

categorical columns:
    impute -> ordinal encode

datetime columns:
    expand
```

Tree models usually do **not** require scaling.

---

## 7.3 `dummy`

`dummy` predicts the mean of the training target. It is used as a baseline.

It requires only:

- A valid target column
- At least enough rows for splitting

---

## 7.4 `tabpfn`

`tabpfn` is intended for smaller tabular datasets.

The script does not tune hyperparameters for `tabpfn`.

Practical recommendations:

- Use it on small to medium datasets.
- Avoid extremely high-dimensional one-hot encoded inputs.
- Consider reducing features before using it.

---

# 8. Data Engineering Flow

The data flow is:

```text
DataSource
    -> FeatureEngineer
    -> LabelExtractor
    -> DatasetBuilder
    -> TypeInferenceEngine
    -> Dataset
    -> NestedCVRunner
```

---

## 8.1 Data loading functions

### `from_dataframe`

Use an in-memory `pandas.DataFrame`.

```python
results = from_dataframe(
    df=df,
    target_columns="target",
).run()
```

---

### `from_csv`

Use CSV or TSV files.

```python
results = from_csv(
    file_path="data.csv",
    target_columns=["target"],
).run()
```

For TSV:

```python
results = from_csv(
    file_path="data.tsv",
    target_columns="target",
    sep="\t",
).run()
```

---

### `from_parquet`

```python
results = from_parquet(
    file_path="data.parquet",
    target_columns="target",
).run()
```

---

### `from_excel`

```python
results = from_excel(
    file_path="data.xlsx",
    sheet="Sheet1",
    target_columns="target",
).run()
```

---

### `from_sqlite`

```python
results = from_sqlite(
    db_path="experiments.sqlite",
    sql_query="SELECT * FROM measurements",
    target_columns="yield",
).run()
```

---

### `from_sqlalchemy`

```python
results = from_sqlalchemy(
    uri="postgresql://user:password@localhost/mydb",
    sql_query="SELECT * FROM measurements",
    target_columns="efficiency",
).run()
```

---

## 8.2 Column name sanitization

The function `sanitize_dataframe_columns` converts column names to safe Python-like names.

Examples:

| Original | Sanitized |
|---|---|
| `"Sale Price"` | `Sale_Price` |
| `"temp°C"` | `temp_C` |
| `"feature-1"` | `feature_1` |

If duplicate names appear after sanitization, suffixes are added.

Example:

```text
"my column"
"my-column"
```

becomes:

```text
my_column
my_column_1
```

This matters because target names are also sanitized.

---

## 8.3 Target extraction

`ColumnLabelExtractor` extracts the target columns from the sanitized DataFrame.

If your original target column is:

```text
Sale Price
```

you may pass:

```python
target_columns="Sale Price"
```

The script sanitizes it internally to:

```text
Sale_Price
```

---

# 9. Column Type Inference

`TypeInferenceEngine` assigns every column one of four roles:

| Role | Meaning |
|---|---|
| `numeric` | Continuous numeric feature |
| `categorical` | Category or low-cardinality integer |
| `datetime` | Date/time feature |
| `drop` | Excluded from modeling |

The inference logic considers:

- Missing fraction
- Data type
- Number of unique values
- Column name patterns such as `"date"`, `"time"`, `"year"`
- Manual overrides

Example override:

```python
feat = FeatureConfig(
    column_type_overrides={
        "customer_id": "drop",
        "zip_code": "categorical",
        "temperature": "numeric",
    }
)
```

---

# 10. Feature Engineering and Preprocessing

The default `PassthroughEngineer` does not create new domain-specific features. It only sanitizes column names.

You can add your own feature engineering by subclassing `FeatureEngineer`.

Example:

```python
from pipeline import FeatureEngineer

class MyFeatureEngineer(FeatureEngineer):
    def engineer(self, df):
        df = df.copy()
        df["area"] = df["length"] * df["width"]
        df["ratio"] = df["income"] / df["debt"].replace(0, float("nan"))
        return df
```

Then use it with `TabularMLPipeline` directly.

---

## 10.1 Numeric preprocessing

Numeric columns go through:

```text
SimpleImputer
    -> VarianceThreshold
    -> optional scaler
    -> optional CorrelationFilter
```

The default imputation strategy is median.

Scaling options:

| `numeric_scaler` | Transformer |
|---|---|
| `"standard"` | `StandardScaler` |
| `"robust"` | `RobustScaler` |
| `"quantile"` | `QuantileTransformer` |
| `"none"` | No scaling |

---

## 10.2 Categorical preprocessing

For linear models:

```text
SimpleImputer
    -> OneHotEncoder
    -> optional StandardScaler
```

For tree models:

```text
SimpleImputer
    -> OrdinalEncoder
```

Why different?

- Linear models need one-hot encoding because category numbers would imply false ordering.
- Tree models can usually handle ordinal-encoded categories reasonably well.

---

## 10.3 Datetime preprocessing

`DatetimeExpander` converts each datetime column into:

- year
- month
- day
- day of week
- hour
- minute
- second
- day of year
- quarter
- weekend flag

Example:

```text
order_date
```

becomes:

```text
order_date__year
order_date__month
order_date__day
order_date__dayofweek
...
```

---

## 10.4 CLR transform for compositional data

`CLRTransformer` is useful when features represent parts of a whole, such as:

- chemical composition percentages
- microbiome proportions
- budget shares
- mineral fractions

Enable it with:

```python
feat = FeatureConfig(
    apply_clr_transform=True,
    clr_column_pattern="^composition_",
    clr_pseudocount=1e-6,
)
```

The centered log-ratio transform is:

$$
\operatorname{clr}(x_i) = \log(x_i) - \frac{1}{D}\sum_{j=1}^{D}\log(x_j)
$$

where $D$ is the number of compositional parts.

---

# 11. Feature Selection and Feature Strategies

The script can run different feature strategies per model.

## 11.1 Default strategies for linear models

For models in `ModelRegistry.LINEAR_MODELS`:

```text
full_scaled
full_robust
selectkbest
selectkbest_mi
pca
```

| Strategy | Meaning |
|---|---|
| `full_scaled` | Use all features with standard scaling |
| `full_robust` | Use all features with robust scaling |
| `selectkbest` | Select top $k$ features using linear regression F-test |
| `selectkbest_mi` | Select top $k$ features using mutual information |
| `pca` | Apply PCA after preprocessing |

---

## 11.2 Default strategies for tree models

For models in `ModelRegistry.TREE_MODELS`:

```text
full
tree_select
```

| Strategy | Meaning |
|---|---|
| `full` | Use all preprocessed features |
| `tree_select` | Use `ExtraTreesRegressor` feature importance to select features |

---

## 11.3 Default strategies for special models

For `dummy` and `tabpfn`:

```text
full
```

---

## 11.4 Override feature strategies

You can force the same feature strategies for all models:

```python
ml = MLConfig(
    include_models=["ridge", "random_forest"],
    feature_strategies_override=["full", "selectkbest"],
)
```

Be careful: not all strategies are equally meaningful for all model types.

---

# 12. How Hyperparameter Tuning Works

Hyperparameter tuning is handled by `OptunaTuner`.

The script uses **nested cross-validation**:

```text
Outer CV:
    estimates model performance

Inner CV:
    tunes hyperparameters using Optuna
```

This helps avoid optimistic bias. The test fold in the outer loop is never used for tuning.

---

## 12.1 Outer CV

The outer cross-validation loop estimates generalization performance.

If group CV is disabled:

```python
KFold(
    n_splits=n_outer_splits,
    shuffle=True,
    random_state=random_state,
)
```

If group CV is enabled and enough groups exist:

```python
GroupKFold(n_splits=n_outer_splits)
```

Outer test data is held out for final fold evaluation.

---

## 12.2 Inner CV

The inner loop tunes hyperparameters.

If group CV is disabled:

```python
KFold(
    n_splits=n_inner_splits,
    shuffle=True,
    random_state=random_state,
)
```

If group CV is enabled and enough groups exist:

```python
GroupKFold(n_splits=n_inner_splits)
```

Each Optuna trial:

1. Suggests hyperparameters.
2. Builds a fresh model pipeline.
3. Fits on inner training split.
4. Predicts on inner validation split.
5. Computes RMSE.
6. Returns average inner RMSE to Optuna.

Optuna minimizes RMSE.

---

## 12.3 Optuna algorithm

The script uses:

```python
optuna.samplers.TPESampler
```

TPE means Tree-structured Parzen Estimator, a Bayesian optimization algorithm.

It also optionally uses:

```python
optuna.pruners.MedianPruner
```

The pruner stops bad trials early.

---

## 12.4 Which models are tuned?

The script calls Optuna for every model and strategy, but some models have no tunable parameters.

| Model | Tuned? |
|---|---|
| `dummy` | No meaningful tuning |
| `ridge` | Yes |
| `lasso` | Yes |
| `elasticnet` | Yes |
| `bayesian_ridge` | Yes |
| `huber` | Yes |
| `linear_svr` | Yes |
| `svr` | Yes |
| `knn` | Yes |
| `random_forest` | Yes |
| `extra_trees` | Yes |
| `hist_gbrt` | Yes |
| `xgboost` | Yes, if installed |
| `lightgbm` | Yes, if installed |
| `catboost` | Yes, if installed |
| `tabpfn` | No hyperparameters tuned in this script |

---

## 12.5 Tuned hyperparameters by model

### Linear models

| Model | Hyperparameters |
|---|---|
| `ridge` | `alpha` |
| `lasso` | `alpha` |
| `elasticnet` | `alpha`, `l1_ratio` |
| `bayesian_ridge` | `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` |
| `huber` | `epsilon`, `alpha` |

### Support vector models

| Model | Hyperparameters |
|---|---|
| `linear_svr` | `C`, `epsilon` |
| `svr` | `C`, `epsilon`, `kernel`, `gamma` |

### KNN

| Model | Hyperparameters |
|---|---|
| `knn` | `n_neighbors`, `weights`, `p` |

### Tree ensembles

| Model | Hyperparameters |
|---|---|
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `extra_trees` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `hist_gbrt` | `learning_rate`, `max_depth`, `max_iter`, `min_samples_leaf`, `l2_regularization`, `max_leaf_nodes` |

### Gradient boosting libraries

| Model | Hyperparameters |
|---|---|
| `xgboost` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma` |
| `lightgbm` | `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_samples` |
| `catboost` | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `subsample`, `bagging_temperature` |

---

# 13. Train/Test Splits and Leakage Prevention

The script is designed to avoid data leakage.

## 13.1 Outer train/test split

For each outer fold:

```text
data
    -> outer training set
    -> outer test set
```

The outer test set is only used for final evaluation of that fold.

---

## 13.2 Inner tuning split

Inside the outer training set:

```text
outer training set
    -> inner training folds
    -> inner validation folds
```

Optuna only sees the outer training data.

---

## 13.3 Conformal calibration split

For conformal prediction, the script further splits the outer training data:

```text
outer training set
    -> proper training data
    -> calibration data
```

Calibration data is isolated **before** hyperparameter tuning.

This is important because conformal intervals require a clean calibration set that has not influenced model fitting or tuning.

---

## 13.4 Group-aware splitting

If you provide a group column:

```python
results = from_dataframe(
    df=df,
    target_columns="target",
    group_column="patient_id",
).run()
```

Then the script uses group-aware splitting when possible.

This means rows from the same group are kept together in either train or test folds.

Use group CV when rows are not independent, for example:

- Multiple rows per patient
- Multiple measurements per machine
- Multiple experiments per batch
- Time-series windows from the same source
- Repeated measurements from the same location

---

# 14. Metrics and Error Measures

The function `compute_metrics` computes standard regression metrics.

## 14.1 $R^2$

Coefficient of determination.

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Higher is better.

- $R^2 = 1$ means perfect prediction.
- $R^2 = 0$ means similar to predicting the mean.
- $R^2 < 0$ means worse than predicting the mean.

---

## 14.2 MAE

Mean absolute error.

$$
\operatorname{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
$$

Lower is better.

Good when you want an interpretable average absolute error in target units.

---

## 14.3 Median absolute error

Median of absolute errors.

$$
\operatorname{MedAE} = \operatorname{median}_i |y_i - \hat{y}_i|
$$

Lower is better.

More robust to outliers than MAE.

---

## 14.4 MSE

Mean squared error.

$$
\operatorname{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2
$$

Lower is better.

Penalizes large errors heavily.

---

## 14.5 RMSE

Root mean squared error.

$$
\operatorname{RMSE} = \sqrt{\operatorname{MSE}}
$$

Lower is better.

Same unit as the target.

This is the main optimization objective during hyperparameter tuning.

---

## 14.6 MAPE

Mean absolute percentage error.

$$
\operatorname{MAPE} = \frac{1}{n}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

Lower is better.

Be careful when target values are zero or close to zero.

---

## 14.7 Out-of-fold metrics

The script also combines predictions from all outer test folds into out-of-fold predictions.

It reports:

- `oof_r2`
- `oof_rmse`
- `oof_mae`
- `oof_mape`

These are often more useful than simple averages of fold metrics.

---

## 14.8 Bootstrap confidence interval

For RMSE, the script computes a bootstrap confidence interval over fold RMSE values.

Output columns:

```text
ci95_rmse_lo
ci95_rmse_hi
```

This gives an approximate $95\%$ confidence interval for median fold RMSE.

---

# 15. Conformal Prediction Intervals

If `compute_conformal=True`, the script computes split conformal intervals.

For each prediction $\hat{y}$, it creates an interval:

$$
[\hat{y} - \hat{q}, \hat{y} + \hat{q}]
$$

where $\hat{q}$ is a quantile of absolute calibration residuals.

With:

```python
conformal_alpha=0.10
```

the intended coverage is approximately $90\%$.

The script reports:

| Output | Meaning |
|---|---|
| `mean_conformal_coverage` | Fraction of test values inside intervals |
| `mean_conformal_width` | Average interval width |

Example:

```python
ml = MLConfig(
    compute_conformal=True,
    conformal_alpha=0.05,
)
```

This requests nominal $95\%$ prediction intervals.

---

# 16. SHAP and Feature Importance

## 16.1 SHAP

If `compute_shap=True` and `shap` is installed, the script computes per-fold SHAP importance.

It uses:

| Model type | SHAP explainer |
|---|---|
| Tree models | `TreeExplainer` |
| Linear models | `LinearExplainer` |
| Other models | `KernelExplainer` fallback |

SHAP files are saved to:

```text
output_dir/shap/
```

Example file:

```text
shap__price__random_forest__full__fold1.csv
```

Each SHAP file contains:

```text
feature, mean_abs_shap
```

---

## 16.2 Permutation importance

Enable with:

```python
ml = MLConfig(
    compute_permutation_importance=True,
    permutation_n_repeats=10,
)
```

Files are saved to:

```text
output_dir/permutation/
```

Permutation importance measures how much model performance decreases when a feature is randomly shuffled.

---

## 16.3 Feature stability

If `track_feature_stability=True`, the script records how frequently a feature is selected across folds.

Output:

```text
output_dir/feature_stability.csv
```

Important mainly for strategies such as:

- `tree_select`
- `selectkbest`
- `selectkbest_mi`

---

# 17. Output Files

The pipeline creates an output directory, default:

```text
ml_results/
```

Inside it:

```text
ml_results/
    config_snapshot.json
    results_summary.csv
    results_summary.parquet
    fold_results.csv
    feature_stability.csv
    models/
    shap/
    fold_predictions/
    permutation/
```

## 17.1 `results_summary.csv`

One row per:

```text
target + model + feature strategy
```

Important columns:

| Column | Meaning |
|---|---|
| `label` | Target column |
| `model` | Model name |
| `feature_strategy` | Feature strategy |
| `target_transform` | Target transform used |
| `n_samples` | Valid rows for this target |
| `mean_r2` | Average fold $R^2$ |
| `std_r2` | Standard deviation of fold $R^2$ |
| `mean_rmse` | Average fold RMSE |
| `std_rmse` | Standard deviation of fold RMSE |
| `mean_mae` | Average fold MAE |
| `oof_r2` | Out-of-fold $R^2$ |
| `oof_rmse` | Out-of-fold RMSE |
| `oof_mae` | Out-of-fold MAE |
| `oof_mape` | Out-of-fold MAPE |
| `ci95_rmse_lo` | Lower bootstrap CI for RMSE |
| `ci95_rmse_hi` | Upper bootstrap CI for RMSE |
| `mean_conformal_coverage` | Mean conformal coverage |
| `mean_conformal_width` | Mean conformal interval width |

---

## 17.2 `fold_results.csv`

One row per outer fold.

Includes:

- Fold number
- Train size
- Test size
- Metrics
- Runtime
- Best hyperparameters
- Conformal coverage and width

---

## 17.3 `fold_predictions/`

Contains out-of-fold predictions.

Example:

```text
preds__price__random_forest__full.csv
```

Columns:

```text
label, model, feature_strategy, fold, y_true, y_pred
```

---

## 17.4 `models/`

If `save_best_models=True`, fitted pipelines are saved as pickle files.

Example:

```text
price__random_forest__full__fold1.pkl
```

These are full scikit-learn pipelines including preprocessing and model.

---

# 18. Examples

## 18.1 Fast baseline run

Use only a baseline and Ridge regression.

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

## 18.2 Strong tree-based modeling

```python
from pipeline import from_csv, MLConfig

ml = MLConfig(
    output_dir="tree_results",
    include_models=[
        "random_forest",
        "extra_trees",
        "hist_gbrt",
        "xgboost",
        "lightgbm",
        "catboost",
    ],
    n_outer_splits=5,
    n_inner_splits=3,
    n_optuna_trials=50,
    compute_shap=True,
)

results = from_csv(
    file_path="data.csv",
    target_columns="target",
    ml_config=ml,
).run()
```

If optional packages are not installed, those models are simply not registered.

---

## 18.3 Group-aware modeling

Use this when observations from the same group must not appear in both train and test data.

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
    group_column="patient_id",
    ml_config=ml,
).run()
```

This sets:

```python
use_group_cv=True
```

automatically.

---

## 18.4 Multiple target columns

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

The script will:

1. Train separate single-target models for each target.
2. Optionally run multi-task models if `use_multitask=True`.

---

## 18.5 Stacking ensemble

```python
from pipeline import from_csv, MLConfig

ml = MLConfig(
    output_dir="stacking_results",
    include_models=["ridge", "random_forest", "hist_gbrt"],
    use_stacking=True,
    compute_shap=False,
)

results = from_csv(
    file_path="data.csv",
    target_columns="target",
    ml_config=ml,
).run()
```

Stacking uses out-of-fold predictions from base models and trains a `RidgeCV` meta-model.

---

## 18.6 Custom target transformation

```python
from pipeline import from_csv, MLConfig, TargetConfig

target_cfg = TargetConfig(
    target_columns=["price"],
    target_transforms={
        "price": "log1p",
    },
    positive_targets=["price"],
)

ml = MLConfig(
    output_dir="price_results",
    include_models=["ridge", "random_forest", "lightgbm"],
)

results = from_csv(
    file_path="housing.csv",
    target_columns="price",
    tgt_config=target_cfg,
    ml_config=ml,
).run()
```

---

## 18.7 Disable automatic target transformation

```python
from pipeline import TargetConfig

target_cfg = TargetConfig(
    target_columns=["target"],
    auto_transform=False,
)
```

---

## 18.8 Use correlation filtering

```python
from pipeline import FeatureConfig

feat = FeatureConfig(
    use_correlation_filter=True,
    correlation_threshold=0.95,
)
```

This removes near-duplicate features inside each training fold.

---

## 18.9 Use feature selection

```python
from pipeline import MLConfig, FeatureConfig

feat = FeatureConfig(
    selectkbest_k=25,
)

ml = MLConfig(
    include_models=["ridge", "lasso", "elasticnet"],
    feature_strategies_override=["selectkbest", "selectkbest_mi"],
)
```

---

## 18.10 Use compositional features with CLR

Suppose columns named `comp_A`, `comp_B`, `comp_C` are parts of a composition.

```python
from pipeline import FeatureConfig

feat = FeatureConfig(
    apply_clr_transform=True,
    clr_column_pattern="^comp_",
)
```

---

## 18.11 Run from SQLite

```python
from pipeline import from_sqlite, MLConfig

ml = MLConfig(
    output_dir="sqlite_results",
    include_models=["ridge", "random_forest"],
)

results = from_sqlite(
    db_path="my_database.sqlite",
    sql_query="SELECT * FROM training_table",
    target_columns="target",
    ml_config=ml,
).run()
```

---

## 18.12 Register a custom model

Your model should be compatible with the scikit-learn estimator API.

```python
from sklearn.ensemble import GradientBoostingRegressor
from pipeline import from_dataframe, MLConfig

ml = MLConfig(
    include_models=["my_gbr"],
)

pipeline = from_dataframe(
    df=df,
    target_columns="target",
    ml_config=ml,
)

pipeline.register_model(
    name="my_gbr",
    cls=GradientBoostingRegressor,
)

results = pipeline.run()
```

For models requiring special construction, provide an instantiator:

```python
def make_model(params):
    return MyRegressor(
        special_argument=True,
        **params,
    )

pipeline.register_model(
    name="my_model",
    cls=MyRegressor,
    instantiator=make_model,
)
```

---

# 19. Function and Class Reference

## 19.1 Utility functions

| Function | Purpose |
|---|---|
| `sanitize_name` | Converts a name to a safe identifier |
| `sanitize_dataframe_columns` | Sanitizes and deduplicates DataFrame column names |
| `bootstrap_ci` | Computes percentile bootstrap confidence intervals |
| `compute_metrics` | Computes regression metrics |
| `detect_abs_skewness` | Computes absolute skewness of a target |

---

## 19.2 Data source classes

| Class | Purpose |
|---|---|
| `DataSource` | Abstract base class for data loading |
| `InMemoryDataSource` | Loads from an existing DataFrame |
| `FileDataSource` | Loads CSV, TSV, Excel, Parquet, Feather, HDF5, JSON, JSONL |
| `SQLiteDataSource` | Loads from SQLite query |
| `SQLAlchemyDataSource` | Loads from SQLAlchemy-compatible database |
| `make_data_source` | Factory that chooses a data source based on config |

---

## 19.3 Feature and label classes

| Class | Purpose |
|---|---|
| `FeatureEngineer` | Abstract base class for feature engineering |
| `PassthroughEngineer` | Default feature engineer |
| `LabelExtractor` | Abstract base class for label extraction |
| `ColumnLabelExtractor` | Extracts target columns from DataFrame |

---

## 19.4 Dataset classes

| Class | Purpose |
|---|---|
| `Dataset` | Stores features, targets, column roles, groups, weights |
| `DatasetBuilder` | Builds a `Dataset` from source, feature engineer, and label extractor |

---

## 19.5 Preprocessing classes

| Class | Purpose |
|---|---|
| `TypeInferenceEngine` | Classifies columns as numeric/categorical/datetime/drop |
| `PreprocessorBuilder` | Builds column-aware preprocessing pipeline |
| `DatetimeExpander` | Expands datetime columns |
| `CLRTransformer` | Centered log-ratio transform |
| `ILRTransformer` | Isometric log-ratio transform |
| `CorrelationFilter` | Removes highly correlated features |
| `TargetTransformer` | Applies reversible target transformations |

Note: `ILRTransformer` is defined but not currently used by the default pipeline.

---

## 19.6 Feature selection classes

| Class | Purpose |
|---|---|
| `TreeFeatureSelector` | ExtraTrees-based feature selector |
| `MutualInfoSelector` | Mutual information feature selector |
| `SelectKBest` | Used directly for F-test selection |
| `PCA` | Used for dimensionality reduction |

---

## 19.7 Modeling classes

| Class | Purpose |
|---|---|
| `ModelRegistry` | Registers and instantiates models |
| `PipelineFactory` | Builds full preprocessing + model pipelines |
| `OptunaTuner` | Runs inner-CV hyperparameter tuning |
| `NestedCVRunner` | Runs full nested cross-validation |
| `MultiTaskRunner` | Runs multi-target models |
| `StackingRunner` | Runs stacking ensemble |
| `TabularMLPipeline` | Top-level orchestrator |

---

## 19.8 Results and interpretation classes

| Class | Purpose |
|---|---|
| `FoldResult` | Stores one fold's result |
| `ExperimentResult` | Stores one model/strategy/target experiment |
| `ResultsManager` | Saves and summarizes results |
| `SHAPComputer` | Computes SHAP importances |
| `FeatureStabilityTracker` | Tracks feature selection stability |
| `ConformalPredictor` | Computes split conformal intervals |

---

# 20. Recommended Workflows

## 20.1 First exploratory run

Use few models and few trials:

```python
ml = MLConfig(
    output_dir="explore",
    include_models=["dummy", "ridge", "random_forest"],
    n_outer_splits=3,
    n_inner_splits=2,
    n_optuna_trials=10,
    compute_shap=False,
    compute_conformal=False,
)
```

Purpose:

- Check data loads correctly.
- Check target names.
- Check runtime.
- Get baseline performance.

---

## 20.2 More serious benchmark

```python
ml = MLConfig(
    output_dir="benchmark",
    include_models=[
        "ridge",
        "elasticnet",
        "random_forest",
        "extra_trees",
        "hist_gbrt",
        "xgboost",
        "lightgbm",
        "catboost",
    ],
    n_outer_splits=5,
    n_inner_splits=4,
    n_optuna_trials=60,
    compute_shap=True,
    compute_conformal=True,
)
```

Purpose:

- Compare many model families.
- Get nested CV performance.
- Save SHAP explanations.
- Save prediction intervals.

---

## 20.3 Production candidate selection

After running the benchmark:

```python
summary = results.to_dataframe()
best = results.get_best_per_label(by="oof_rmse")

print(best)
```

Choose models based on:

1. Low `oof_rmse`
2. High `oof_r2`
3. Stable performance across folds
4. Reasonable conformal coverage
5. Interpretability needs
6. Training and prediction time

---

# 21. Important Notes and Limitations

## 21.1 Regression only

This script is not designed for classification. Do not use categorical class labels as targets.

---

## 21.2 Sample weights are extracted but not used

The script extracts `sample_weight_column` into the `Dataset`, but the current model-fitting code does **not** pass sample weights into `.fit()`.

So this field is currently metadata only unless you modify the training code.

---

## 21.3 High-cardinality categorical columns may be dropped

String/object columns with too many unique values are classified as `drop` by default.

You can change this by increasing:

```python
FeatureConfig(max_cardinality_for_categorical=...)
```

or forcing a type:

```python
FeatureConfig(
    column_type_overrides={
        "product_id": "categorical",
    }
)
```

Be careful: one-hot encoding high-cardinality columns can create huge feature matrices.

---

## 21.4 Runtime can grow quickly

Total training cost is roughly proportional to:

```text
targets
× models
× feature strategies
× outer folds
× Optuna trials
× inner folds
```

For example:

```text
2 targets × 6 models × 2 strategies × 5 outer folds × 60 trials × 4 inner folds
```

can be very expensive.

For fast experimentation, reduce:

```python
n_optuna_trials
n_outer_splits
n_inner_splits
include_models
feature_strategies_override
```

---

## 21.5 SHAP can be slow

SHAP, especially `KernelExplainer`, can be slow.

For faster runs:

```python
ml = MLConfig(
    compute_shap=False,
)
```

or reduce:

```python
shap_sample_size=50
```

---

## 21.6 MAPE can be unstable near zero

If target values are zero or close to zero, MAPE may be very large or misleading.

Prefer RMSE, MAE, or MedAE in those cases.

---

# 22. Common Errors and Fixes

## Error: target column not found

Example message:

```text
Target columns not found after sanitisation
```

Cause:

- Target name is misspelled.
- Column name changed during sanitization.
- Target column not present in loaded data.

Fix:

```python
print(df.columns)
```

Check the exact column name.

---

## Error: no usable feature columns found

Cause:

- All columns were dropped.
- Target was the only usable column.
- Missing values exceeded threshold.
- High-cardinality strings were dropped.

Fix:

```python
feat = FeatureConfig(
    max_missing_fraction=0.99,
    max_cardinality_for_categorical=200,
)
```

or use type overrides.

---

## Error: optional model not available

If you request:

```python
include_models=["lightgbm"]
```

but LightGBM is not installed, it will not be registered.

Fix:

```bash
pip install lightgbm
```

---

## Error: too few samples

The script skips targets with fewer than:

```python
min_samples_per_label
```

valid samples.

Default:

```python
30
```

Fix:

```python
ml = MLConfig(
    min_samples_per_label=10,
)
```

Use with caution.

---

## Error: group CV not used

If `use_group_cv=True` but there are too few groups, the script falls back to regular KFold.

Fix:

- Make sure the group column exists.
- Make sure there are at least as many unique groups as folds.
- Reduce `n_outer_splits`.

---

# 23. Minimal Practical Template

```python
import pandas as pd

from pipeline import (
    from_dataframe,
    MLConfig,
    FeatureConfig,
    TargetConfig,
)

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

print(summary)
print(best)
```

---

# 24. Best Practices

1. **Always start with `dummy`**
   - It tells you whether real models improve over a simple mean predictor.

2. **Use group CV if observations are correlated**
   - For patients, batches, machines, locations, users, or repeated experiments.

3. **Reduce models during debugging**
   - Use `include_models`.

4. **Increase Optuna trials only after the pipeline works**
   - Start with `10` or `20`.
   - Increase to `60+` for final benchmarking.

5. **Inspect out-of-fold predictions**
   - Look at files in `fold_predictions/`.

6. **Do not choose models only by one metric**
   - Compare `oof_rmse`, `oof_mae`, `oof_r2`, fold stability, and interpretability.

7. **Watch for leakage**
   - Do not include columns that directly encode the target.
   - Use `drop_columns` for IDs, post-outcome variables, or derived target variables.

Example:

```python
from_dataframe(
    df=df,
    target_columns="revenue",
    drop_columns=["revenue_after_tax", "future_status", "target_encoded_field"],
)
```

---

# 25. Summary

This script is a full supervised regression framework for tabular data. It provides:

- Flexible data loading
- Automatic column type detection
- Leakage-safe preprocessing
- Multiple regression models
- Nested cross-validation
- Optuna hyperparameter tuning
- Target transformations
- Feature selection
- SHAP explainability
- Conformal prediction intervals
- Out-of-fold predictions
- Multi-target and stacking options
- Saved models and result reports

The most common usage is:

```python
results = from_csv(
    file_path="data.csv",
    target_columns="target",
).run()
```

For serious experiments, configure `MLConfig`, `FeatureConfig`, and `TargetConfig` explicitly and inspect `results_summary.csv`, `fold_results.csv`, and out-of-fold predictions.
