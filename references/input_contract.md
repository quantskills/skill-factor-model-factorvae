# Input Contract

`data.panel_path` points to one CSV or Parquet table. Every row represents one
stock on one decision date.

```yaml
data:
  panel_path: /path/to/panel.parquet
  date_column: date
  ticker_column: ticker
  label_column: label
  feature_columns: [factor_value, factor_quality]
splits:
  train: [20160104, 20231229]
  valid: [20240102, 20241231]
  test: [20250102, 20261231]
```

`date` must be integer-like `YYYYMMDD`. `(date, ticker)` must be unique.
Feature columns are numeric and explicitly listed. `label` is required for
training and validation; test rows may contain null labels when only future
signals are needed. Split ranges are chronological and non-overlapping.

The model uses only the preceding `preprocessing.lookback` feature rows for a
prediction. It does not infer factor direction or calculate a trading return.
