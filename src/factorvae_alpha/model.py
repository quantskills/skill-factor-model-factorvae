from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_size, hidden_size), nn.SiLU())
        self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [stocks, history, features]")
        _, hidden = self.recurrent(self.projection(inputs))
        return self.dropout(hidden[-1])


class FactorVAE(nn.Module):
    """Probabilistic dynamic factor model with prior-posterior learning."""

    def __init__(self, input_size: int, settings: dict) -> None:
        super().__init__()
        self.settings = dict(settings)
        hidden = int(settings["hidden_size"])
        self.factor_count = int(settings["latent_factors"])
        portfolios = int(settings["portfolios"])
        dropout = float(settings.get("dropout", 0.1))
        self.encoder = SequenceEncoder(input_size, hidden, dropout)
        self.portfolio_weights = nn.Linear(hidden, portfolios)
        self.posterior = nn.Sequential(nn.Linear(portfolios, hidden), nn.SiLU(), nn.Linear(hidden, self.factor_count * 2))
        self.prior_queries = nn.Parameter(torch.randn(self.factor_count, hidden) / math.sqrt(hidden))
        self.prior_network = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.alpha_network = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.beta_network = nn.Linear(hidden, self.factor_count)

    @staticmethod
    def positive_scale(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw).clamp(min=1e-4, max=10.0)

    def forward(self, inputs: torch.Tensor) -> dict:
        """Return prior-based predictions using historical features only."""
        hidden = self.encoder(inputs)
        attention = (hidden @ self.prior_queries.T / math.sqrt(hidden.shape[-1])).softmax(dim=0)
        prior = self.prior_network(attention.T @ hidden)
        prior_mean, prior_scale = prior[:, 0], self.positive_scale(prior[:, 1])
        alpha_parameters = self.alpha_network(hidden)
        alpha, specific_scale = alpha_parameters[:, 0], self.positive_scale(alpha_parameters[:, 1])
        beta = self.beta_network(hidden)
        prediction = alpha + beta @ prior_mean
        variance = specific_scale.square() + beta.square() @ prior_scale.square()
        return {"prediction": prediction, "prediction_std": variance.sqrt(), "hidden": hidden,
                "prior_mean": prior_mean, "prior_scale": prior_scale, "alpha": alpha,
                "specific_scale": specific_scale, "beta": beta}

    def objective(self, outputs: dict, labels: torch.Tensor, epoch: int) -> tuple[torch.Tensor, dict]:
        hidden = outputs["hidden"]
        weights = self.portfolio_weights(hidden).softmax(dim=0)
        portfolio_returns = weights.T @ labels
        posterior = self.posterior(portfolio_returns)
        posterior_mean, posterior_raw = posterior.chunk(2, dim=-1)
        posterior_scale = self.positive_scale(posterior_raw)
        reconstruction_mean = outputs["alpha"] + outputs["beta"] @ posterior_mean
        reconstruction_variance = outputs["specific_scale"].square() + outputs["beta"].square() @ posterior_scale.square()
        reconstruction = 0.5 * (((labels - reconstruction_mean).square() / reconstruction_variance) + reconstruction_variance.log()).mean()
        ratio = posterior_scale / outputs["prior_scale"]
        distance = (posterior_mean - outputs["prior_mean"]) / outputs["prior_scale"]
        kl = 0.5 * (ratio.square() + distance.square() - 1.0 - 2.0 * ratio.log()).mean()
        warmup = max(1, int(self.settings.get("kl_warmup_epochs", 10)))
        coefficient = float(self.settings.get("kl_weight", 0.01)) * min(1.0, (epoch + 1) / warmup)
        return reconstruction + coefficient * kl, {"reconstruction_loss": float(reconstruction.detach()),
                                                     "kl_loss": float(kl.detach()), "kl_coefficient": coefficient}
