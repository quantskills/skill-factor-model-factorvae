# Output Contract

A completed run directory contains:

- `resolved_config.yaml`: immutable effective configuration;
- `input_manifest.json`: panel shape and feature names;
- `training_metrics.jsonl`: one record per epoch;
- `best_checkpoint.pt`: selected PyTorch state;
- `validation_summary.json`: checkpoint selection result;
- `signals_<split>.parquet`: optional exported predictions.

The default signal file has exactly these columns:

```text
date,ticker,signal
```

With `predict --include-uncertainty`, the separately named
`signals_<split>_with_uncertainty.parquet` additionally contains `prediction_std`.

`signal` is a floating-point cross-sectional score. `prediction_std` is the
model's decoded predictive standard deviation on the label scale used by the
run. Neither field is a portfolio return or an investment recommendation.
