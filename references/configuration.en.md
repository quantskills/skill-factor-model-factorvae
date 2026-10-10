# Configuration Reference

Configuration is YAML. Omitted fields use the defaults below; paths are resolved relative to the configuration file.

## Data and splits

| Key | Default | Meaning |
|---|---:|---|
| `data.panel_path` | required | Path to a CSV or Parquet long-form factor panel. |
| `data.date_column` | required | Integer-like `YYYYMMDD` date column. |
| `data.ticker_column` | required | Instrument identifier column. |
| `data.label_column` | required | Training and validation target column. |
| `data.feature_columns` | required | Numeric factor columns used as model features. |
| `splits.train` | required | `[start_date, end_date]` training interval. |
| `splits.valid` | required | Validation interval. |
| `splits.test` | required | Signal-export interval. Splits are chronological and non-overlapping. |

## Preprocessing

| Key | Default | Meaning |
|---|---:|---|
| `preprocessing.lookback` | `20` | Historical trading dates per prediction; minimum `2`. |
| `preprocessing.standardize_features` | `true` | Cross-sectionally standardise each factor on each date. |
| `preprocessing.standardize_labels` | `true` | Cross-sectionally standardise labels on each date. |

## Model

| Key | Default | Meaning |
|---|---:|---|
| `model.hidden_size` | `64` | Hidden width of the GRU encoder and internal networks. |
| `model.latent_factors` | `8` | Number of latent factors. |
| `model.portfolios` | `16` | Number of dynamic portfolios in the posterior path. |
| `model.dropout` | `0.1` | Feature-encoder dropout probability. |
| `model.kl_weight` | `0.01` | Final posterior-prior KL loss weight. |
| `model.kl_warmup_epochs` | `10` | Epochs used to ramp the KL weight to its final value. |

## Training and output

| Key | Default | Meaning |
|---|---:|---|
| `training.epochs` | `20` | Training epochs. |
| `training.learning_rate` | `0.001` | AdamW learning rate. |
| `training.weight_decay` | `0.0001` | AdamW weight decay. |
| `training.seed` | `42` | Random seed. |
| `training.device` | `cpu` | `cpu`, `cuda`, or `auto`; `auto` prefers available CUDA. |
| `training.selection_metric` | `mean_pearson` | Checkpoint selection metric; this release fixes it to validation mean Pearson IC. |
| `output_root` | `runs` | Root directory for run outputs. |
| `run_name` | timestamp | Optional fixed run directory name; an existing name fails closed. |

## Uncertainty export

The default command writes the portable three-column signal file:

```bash
python scripts/run_model.py predict --run-dir <run_dir> --split test
```

Use `--include-uncertainty` to write `signals_test_with_uncertainty.parquet`, which also contains `prediction_std`:

```bash
python scripts/run_model.py predict   --run-dir <run_dir> --split test --include-uncertainty
```
