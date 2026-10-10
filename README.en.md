# FactorVAE Alpha Model

A standalone FactorVAE model for synthesising multi-factor panels into
cross-sectional stock signals. It accepts a user-supplied panel and exports a
portable `date,ticker,signal` file.

```bash
python -m pip install -e .
python examples/make_synthetic_panel.py --output examples/generated/synthetic_panel.csv
python scripts/run_model.py check --config examples/smoke.yaml
python scripts/run_model.py train --config examples/smoke.yaml
```

Use `predict --run-dir <run-dir> --split test` to export the selected split.
See `references/` for the data and output contracts.

## Research Use

**Data source:** users supply factor panels and labels at runtime; the repository contains only synthetic examples. **Assumptions and parameters:** splits, labels, lookback, model dimensions, and training settings are explicit YAML parameters. **Limitations and risk boundary:** output is a cross-sectional ranking signal, not portfolio construction, execution, or a return guarantee. This project is for research and education and does not constitute investment advice.
