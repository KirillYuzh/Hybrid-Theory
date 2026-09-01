from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kyt_engine.features._utils import prepare_features


class _BehavioralVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self._mu = nn.Linear(32, latent_dim)
        self._logvar = nn.Linear(32, latent_dim)
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._encoder(x)
        mu, logvar = self._mu(h), self._logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self._decoder(z)
        return recon, mu, logvar


class BehavioralVAEDetector:
    def __init__(
        self,
        latent_dim: int = 16,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        contamination: float = 0.5,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._contamination = contamination
        self._random_state = random_state
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self._net: _BehavioralVAE | None = None
        self._threshold: float = 0.5
        self._mean: np.ndarray = np.array([])
        self._std: np.ndarray = np.array([])
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BehavioralVAEDetector:
        torch.manual_seed(self._random_state)
        np.random.seed(self._random_state)

        X_df, y_s = prepare_features(X, pd.Series(y) if not isinstance(y, pd.Series) else y)
        self._feature_names = list(X_df.columns)
        X_np = X_df.to_numpy(dtype=np.float32)

        licit_mask = np.asarray(y_s) == 0
        X_licit = X_np[licit_mask]

        self._mean = np.mean(X_licit, axis=0)
        self._std = np.std(X_licit, axis=0)
        self._std[self._std < 1e-8] = 1.0

        X_scaled = (X_licit - self._mean) / self._std

        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        n_features = X_scaled.shape[1]
        self._net = _BehavioralVAE(n_features, self._latent_dim).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        recon_criterion = nn.MSELoss(reduction="sum")

        self._net.train()
        for _ in range(self._epochs):
            for (batch,) in loader:
                batch = batch.to(self._device, non_blocking=True)
                recon, mu, logvar = self._net(batch)
                recon_loss = recon_criterion(recon, batch)
                kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self._net.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
            recon, mu, logvar = self._net(x_t)
            recon_err = torch.mean((recon - x_t) ** 2, dim=1)
            kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores = (recon_err + kl).cpu().numpy()

        quantile = 1.0 - self._contamination
        self._threshold = float(np.quantile(scores, quantile))

        return self

    def _anomaly_score(self, X_np: np.ndarray) -> np.ndarray:
        X_scaled = (X_np - self._mean) / self._std
        self._net.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
            recon, mu, logvar = self._net(x_t)
            recon_err = torch.mean((recon - x_t) ** 2, dim=1)
            kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores = recon_err + kl
        return scores.cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before predict_proba")
        X_df, _ = prepare_features(X)
        X_np = X_df.to_numpy(dtype=np.float32)
        scores = self._anomaly_score(X_np)
        probs = np.clip(scores / (self._threshold + 1e-9), 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
