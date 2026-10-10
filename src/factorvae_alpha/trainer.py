from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import PanelData, load_panel
from .io import set_seed, timestamped_run_dir, write_json, write_yaml
from .model import FactorVAE


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
    return device


def _pearson(prediction: np.ndarray, labels: np.ndarray) -> float | None:
    if len(prediction) < 3 or np.std(prediction) < 1e-12 or np.std(labels) < 1e-12:
        return None
    return float(np.corrcoef(prediction, labels)[0, 1])


def _evaluate(model: FactorVAE, panel: PanelData, positions: np.ndarray, device: torch.device) -> dict:
    model.eval()
    correlations: list[float] = []
    with torch.no_grad():
        for position in positions:
            _, features, labels = panel.batch(int(position), require_label=True)
            if labels is None or len(labels) < 3:
                continue
            output = model(torch.as_tensor(features, device=device))
            score = _pearson(output["prediction"].detach().cpu().numpy(), labels)
            if score is not None:
                correlations.append(score)
    if not correlations:
        raise ValueError("validation contains no cross-sections with a defined Pearson IC")
    return {"mean_pearson": float(np.mean(correlations)), "pearson_std": float(np.std(correlations, ddof=0)),
            "valid_dates": len(correlations)}


def train(config_path: str | Path) -> Path:
    config = load_config(config_path)
    panel = load_panel(config)
    train_positions = panel.positions(config["splits"]["train"], require_label=True)
    valid_positions = panel.positions(config["splits"]["valid"], require_label=True)
    if not len(train_positions) or not len(valid_positions):
        raise ValueError("train and valid splits must contain labeled positions after lookback")
    set_seed(int(config["training"]["seed"]))
    device = _device(config["training"]["device"])
    model = FactorVAE(panel.features.shape[-1], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    run_dir = timestamped_run_dir(config["output_root"], config.get("run_name"))
    write_yaml(run_dir / "resolved_config.yaml", config)
    write_json(run_dir / "input_manifest.json", {"panel_path": config["data"]["panel_path"],
                                                    "panel_summary": panel.summary(config["splits"]),
                                                    "feature_columns": panel.feature_columns})
    metrics_path = run_dir / "training_metrics.jsonl"
    best_score = -float("inf")
    best_epoch = -1
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        losses = []
        for position in train_positions:
            _, features, labels = panel.batch(int(position), require_label=True)
            if labels is None or len(labels) < 3:
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = model(torch.as_tensor(features, device=device))
            loss, _ = model.objective(outputs, torch.as_tensor(labels, device=device), epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid = _evaluate(model, panel, valid_positions, device)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **valid}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record["mean_pearson"] > best_score:
            best_score, best_epoch = record["mean_pearson"], epoch + 1
            torch.save({"model_state": model.state_dict(), "model_settings": config["model"],
                        "input_size": panel.features.shape[-1], "best_epoch": best_epoch,
                        "best_valid_pearson": best_score}, run_dir / "best_checkpoint.pt")
    summary = {"best_epoch": best_epoch, "best_valid_pearson": best_score, "device": str(device),
               "status": "complete"}
    write_json(run_dir / "validation_summary.json", summary)
    write_json(run_dir / "run_manifest.json", {"status": "complete", "run_dir": str(run_dir),
                                                  "artifacts": ["resolved_config.yaml", "input_manifest.json",
                                                                "training_metrics.jsonl", "best_checkpoint.pt",
                                                                "validation_summary.json"]})
    return run_dir


def _load_run(run_dir: str | Path) -> tuple[dict, PanelData, FactorVAE, torch.device]:
    root = Path(run_dir).resolve()
    config = load_config(root / "resolved_config.yaml")
    panel = load_panel(config)
    device = _device(config["training"]["device"])
    checkpoint = torch.load(root / "best_checkpoint.pt", map_location=device, weights_only=True)
    model = FactorVAE(int(checkpoint["input_size"]), checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, panel, model, device


def predict(run_dir: str | Path, split: str, include_uncertainty: bool = False) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    config, panel, model, device = _load_run(run_dir)
    positions = panel.positions(config["splits"][split], require_label=False)
    rows = []
    with torch.no_grad():
        for position in positions:
            stocks, features, _ = panel.batch(int(position), require_label=False)
            if not len(stocks):
                continue
            outputs = model(torch.as_tensor(features, device=device))
            payload = {"date": int(panel.dates[position]), "ticker": panel.tickers[stocks],
                       "signal": outputs["prediction"].detach().cpu().numpy().astype("float64")}
            if include_uncertainty:
                payload["prediction_std"] = outputs["prediction_std"].detach().cpu().numpy().astype("float64")
            rows.append(pd.DataFrame(payload))
    if not rows:
        raise ValueError(f"no predictable rows in split {split}")
    suffix = "_with_uncertainty" if include_uncertainty else ""
    destination = Path(run_dir) / f"signals_{split}{suffix}.parquet"
    output = pd.concat(rows, ignore_index=True)
    output.to_parquet(destination, index=False)
    write_json(Path(run_dir) / f"signals_{split}{suffix}_summary.json", {"rows": int(len(output)), "dates": int(output.date.nunique()),
                                                                           "includes_uncertainty": include_uncertainty,
                                                                           "path": str(destination)})
    return destination


def verify(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    required = ["resolved_config.yaml", "input_manifest.json", "training_metrics.jsonl", "best_checkpoint.pt",
                "validation_summary.json", "run_manifest.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing run artifacts: {', '.join(missing)}")
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    return {"status": "verified", "run_dir": str(root.resolve()), "best_valid_pearson": summary["best_valid_pearson"]}
