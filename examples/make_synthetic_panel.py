from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range("2020-01-02", periods=54).strftime("%Y%m%d").astype(int)
    rows = []
    for date_index, date in enumerate(dates):
        market = np.sin(date_index / 7.0)
        for ticker in range(1001, 1017):
            quality, value, momentum = rng.normal(size=3)
            label = 0.35 * quality - 0.2 * value + 0.15 * momentum + 0.08 * market + rng.normal(scale=0.2)
            rows.append({"date": int(date), "ticker": ticker, "factor_quality": quality, "factor_value": value,
                         "factor_momentum": momentum, "label": label})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
