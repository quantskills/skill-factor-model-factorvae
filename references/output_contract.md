# Output Contract

A completed run directory contains:

- `resolved_config.yaml`: immutable effective configuration;
- `input_manifest.json`: panel shape and feature names;
- `training_metrics.jsonl`: one record per epoch;
- `best_checkpoint.pt`: selected PyTorch state;
- `validation_summary.json`: checkpoint selection result;
- `signals_<split>.parquet`: optional exported predictions.

Every signal file has exactly these columns:

```text
date,ticker,signal
```

`signal` is a floating-point cross-sectional score. It is intentionally not a
portfolio return, risk forecast, or investment recommendation.
