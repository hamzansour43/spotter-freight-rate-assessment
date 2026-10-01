# EDA Findings

- Dataset contains **48,000 labeled loads**.
- Target variable: `posted_rate`.
- Features include pickup and delivery locations and coordinates, distance, equipment, weight, date, market index, and quote signal.
- Missing values: `weight` has 300 rows (0.625%) and `market_index` has 374 rows (0.779%) missing values. Other columns have none.
- No duplicate rows detected.
- Freight rate increases with distance (Pearson correlation with `posted_rate`: **0.909**).
- Date range: **2025-01-01 to 2025-10-31**.
