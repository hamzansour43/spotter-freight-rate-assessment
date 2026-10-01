# EDA Findings

## Dataset

- Dataset contains **48,000 labeled loads**.
- Target variable: `posted_rate`.
- Features include pickup and delivery locations and coordinates, distance, equipment, weight, date, market index, and quote signal.
- Date range: **2025-01-01 to 2025-10-31**; all dates converted to datetime successfully, with no parse failures.

## Data quality

- Missing values: `weight` has **300** (0.625%) and `market_index` has **374** (0.779%) missing values. Other columns have none.
- No duplicate rows detected.
- **292** rows have negative `weight`; the minimum is **-47,500**. These signs/values should be investigated against the source data before correction.
- No `distance` values are less than or equal to zero.
- No `posted_rate` values are less than or equal to zero.
- The `equipment` column has three observed categories: `Dry Van` (27,202), `Reefer` (12,045), and `Flatbed` (8,753). No missing or unexpected equipment categories were observed.

## Relationships and distributions

- Freight rate increases with distance (Pearson correlation with `posted_rate`: **0.909**).
- The `posted_rate` distribution is right-skewed, not normally distributed, with high-end outlier candidates. Its maximum is **25,533**, compared with a 75th percentile of **3,330.75**.

## Recommended validation approach

- Use a chronological holdout rather than a random split, because freight-rate prediction is intended to generalize to future loads and random splitting could let later-period patterns inform evaluation on earlier dates.
- Split the **304 unique labeled dates** at **2025-09-01**: train on **2025-01-01 through 2025-08-31** (38,477 rows) and validate on **2025-09-01 through 2025-10-31** (9,523 rows). This is an 80/20 split by dates and keeps all loads from each date together.
- The separate `validation.csv` contains **12,000 unlabeled loads dated 2025-11-01 through 2025-12-31**, with no dates overlapping the labeled data. Keep it untouched for final prediction; it cannot be used to calculate validation metrics because it has no `posted_rate`.
- Use **MAE** as the primary validation metric (typical absolute rate error, less dominated by extreme values) and report **RMSE** as a secondary metric to show sensitivity to large errors. For a final estimate, consider rolling-origin validation across several earlier date cutoffs, then evaluate once on the final Sep-Oct holdout.
- Before model fitting, investigate the **292 negative weights**. Do not silently treat these as valid physical weights; confirm whether they are sign errors and correct or mark them missing according to the source/business rules. Fit all preprocessing only on the training partition.

## Engineered features and encoding

- Added calendar features: `month`, `day_of_week` (Monday=0), ISO `week`, and `quarter`.
- Added `distance_squared` and `weight_per_mile`. The latter preserves the existing negative weights, so clean or resolve those source values before relying on it for modeling.
- Added `lane` by combining pickup and delivery city names (for example, `Richmond_Baltimore`).
- For CatBoost, `cat_features` identifies `pickup`, `delivery`, `equipment`, and `lane`; matching feature matrices are prepared for the chronological train/holdout and final unlabeled validation set.
- For other models, a `ColumnTransformer` one-hot encodes those same categorical features with `handle_unknown="ignore"` and passes numeric features through. It is fit only on the chronological training partition, then applied to both holdout and final validation data. The resulting sparse-compatible matrices each have **4,144 features**.

## Linear regression baseline

- Fit `LinearRegression` on the chronological training partition (**38,477** rows; Jan-Aug 2025) and evaluated on the held-out future period (**9,523** rows; Sep-Oct 2025).
- The model uses the engineered numeric features and one-hot-encoded `pickup`, `delivery`, `equipment`, and `lane`; the encoder and numeric median imputer are fit on the training partition only.
- For this baseline only, negative `weight` values and their derived `weight_per_mile` values are treated as missing, then imputed with training-partition numeric medians. Original source data is not overwritten. This is a modeling assumption pending confirmation of those source records.
- Holdout metrics: **MAE $187.14**, **RMSE $649.07**, **MAPE 8.37%**. MAPE is reported as the mean absolute percentage error; the high RMSE relative to MAE suggests some larger errors influence squared-error performance. These are baseline results, not a final model-selection claim.

## Tree-model comparison

- Compared Random Forest and CatBoost on the same chronological split and engineered features as the linear baseline. The final unlabeled validation set was not used for model selection.
- Random Forest used median imputation for numeric inputs, one-hot encoding for categorical inputs, 300 trees, `max_features=0.5`, and `min_samples_leaf=2`.
- CatBoost used native categorical handling for `pickup`, `delivery`, `equipment`, and `lane`, with 800 iterations, depth 8, and learning rate 0.05. Numeric missing values were left for CatBoost's native handling.
- Both tree models use modeling copies where negative `weight` and `weight_per_mile` values are set to missing; the source features remain unchanged.

| Model | Holdout MAE | Holdout RMSE | Holdout MAPE |
|---|---:|---:|---:|
| LinearRegression | $187.14 | $649.07 | 8.37% |
| RandomForest | $139.19 | $655.53 | 6.07% |
| CatBoost | **$131.57** | **$642.17** | 6.12% |

- CatBoost is the current best by the primary metric, MAE, improving on the linear baseline by **$55.57 (29.7%)** and on Random Forest by **$7.62 (5.5%)**. It also has the lowest RMSE; Random Forest has marginally lower MAPE.
- These are initial model settings, not tuned results. Keep CatBoost as the leading candidate and validate with rolling-origin splits before finalizing model selection.

## Final fit on all labeled data

- After selecting CatBoost from the chronological holdout comparison, refit it with the same initial settings on **all 48,000 labeled rows** in `train-test.csv`, including the former Sep-Oct holdout.
- Generate `predicted_posted_rate` for all **12,000** rows in the separate unlabeled validation file, retaining `load_id` in `final_validation_predictions`.
- As in the model comparison, negative `weight` values and derived `weight_per_mile` values are treated as missing in the model input copies; original datasets remain unchanged.
- The final validation set has no target labels, so these predictions are not scored here. The holdout metrics above remain the model-selection estimate; the full-data model is for final inference.
