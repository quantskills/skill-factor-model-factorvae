from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from factorvae_alpha.config import load_config
from factorvae_alpha.data import load_panel
from factorvae_alpha.model import FactorVAE
from factorvae_alpha.trainer import predict, train, verify


def _config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-04", periods=36).strftime("%Y%m%d").astype(int)
    rows = []
    for date in dates:
        for ticker in range(1, 11):
            first, second = rng.normal(size=2)
            rows.append({"date": int(date), "ticker": ticker, "first": first, "second": second,
                         "label": 0.4 * first - 0.2 * second + rng.normal(scale=0.15)})
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False)
    config = {
        "data": {"panel_path": str(panel_path), "date_column": "date", "ticker_column": "ticker",
                 "label_column": "label", "feature_columns": ["first", "second"]},
        "splits": {"train": [int(dates[0]), int(dates[18])], "valid": [int(dates[19]), int(dates[26])],
                   "test": [int(dates[27]), int(dates[-1])]},
        "preprocessing": {"lookback": 4, "standardize_features": True, "standardize_labels": True},
        "model": {"hidden_size": 8, "latent_factors": 2, "portfolios": 3, "dropout": 0.0,
                  "kl_weight": 0.01, "kl_warmup_epochs": 1},
        "training": {"epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0, "seed": 4, "device": "cpu"},
        "output_root": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_train_predict_and_verify(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = train(config)
    signals = predict(run_dir, "test")
    frame = pd.read_parquet(signals)
    assert list(frame.columns) == ["date", "ticker", "signal"]
    assert len(frame) > 0
    uncertainty = pd.read_parquet(predict(run_dir, "test", include_uncertainty=True))
    assert list(uncertainty.columns) == ["date", "ticker", "signal", "prediction_std"]
    assert uncertainty["prediction_std"].ge(0).all()
    assert verify(run_dir)["status"] == "verified"


def test_forward_has_no_label_argument(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    panel = load_panel(config)
    _, values, _ = panel.batch(panel.lookback - 1, require_label=False)
    model = FactorVAE(values.shape[-1], config["model"])
    output = model(torch.as_tensor(values))
    assert output["prediction"].shape[0] == values.shape[0]
