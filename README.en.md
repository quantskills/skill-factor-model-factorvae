# FactorVAE Alpha Model

## Architecture

![skill-factor-model-factorvae architecture](assets/factorvae-architecture.svg)


`skill-factor-model-factorvae` is a standalone, installable model for synthesising a user-supplied multi-factor panel into cross-sectional stock signals.

## Method and paper reference

This project is informed by:

> Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li. **FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns.** AAAI 2022, pp. 4468-4476. DOI: [10.1609/aaai.v36i4.20369](https://doi.org/10.1609/aaai.v36i4.20369).

This repository is an **unofficial independent reproduction and generic factor-panel adaptation**, not an implementation released by the paper authors. It retains the prior-posterior dynamic-factor design, dynamic portfolios, Gaussian alpha-beta decoder, and KL objective while leaving the dataset, label horizon, time splits, and hyperparameters configurable. See [paper mapping](references/paper_mapping.md).

## Capabilities

- Loads CSV or Parquet multi-factor panels.
- Encodes each stock from its historical factor sequence.
- Learns posterior latent factors from labels during training and prior latent factors from historical features.
- Uses a probabilistic `alpha + beta × latent factor` decoder.
- Selects checkpoints by validation mean Pearson IC.
- Exports portable `date,ticker,signal` Parquet files without a backtesting dependency.

## Quick start

```bash
python -m pip install -e ".[test]"
python examples/make_synthetic_panel.py   --output examples/generated/synthetic_panel.csv
python scripts/run_model.py check --config examples/smoke.yaml
python scripts/run_model.py train --config examples/smoke.yaml
```

Then export and verify a test split:

```bash
python scripts/run_model.py predict   --run-dir outputs/smoke/<run_id> --split test
python scripts/run_model.py verify   --run-dir outputs/smoke/<run_id>
```

Run the checked-in synthetic smoke flow and unit tests with:

```bash
python scripts/run_smoke.py
python -m pytest -q
```

## Input and output

The input is one long-form CSV or Parquet panel with unique `(date, ticker)` rows. Configure one or more numeric factor columns and a label explicitly in YAML. See the [input contract](references/input_contract.md) and the complete [configuration reference](references/configuration.en.md).

A completed run writes a resolved configuration, input manifest, epoch metrics, selected checkpoint, validation summary, and optional signal export. Signal files always contain:

```text
date,ticker,signal
20250102,600000,0.1837
```

See the [output contract](references/output_contract.md) for the full artifact definition.

The default FactorVAE export remains a portable three-column signal. To also export decoded predictive standard deviation, run:

```bash
python scripts/run_model.py predict \
  --run-dir <run_id> --split test --include-uncertainty
```

## Validation

| Check | Command | Expected result |
|---|---|---|
| Configuration and data | `python scripts/run_model.py check --config <config.yaml>` | Prints `status: valid` with panel dimensions and split availability. |
| Synthetic smoke flow | `python scripts/run_smoke.py` | Generates synthetic data, validates configuration, and trains a small model. |
| Unit tests | `python -m pytest -q` | All tests pass, including standard signals, optional uncertainty export, and run verification. |
| Run artifacts | `python scripts/run_model.py verify --run-dir <run_id>` | Prints `status: verified` after required artifacts are present. |

## Research use

**Data source:** users supply factor panels and labels at runtime; the repository contains only synthetic examples. **Assumptions and parameters:** splits, labels, lookback, model dimensions, and training settings are explicit YAML parameters. **Limitations and risk boundary:** output is a cross-sectional ranking signal, not portfolio construction, execution, or a return guarantee. This project is for research and education and does not constitute investment advice.

## License

Licensed under [GPL-3.0-only](LICENSE).
