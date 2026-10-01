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
