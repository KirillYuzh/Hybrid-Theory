from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kyt_engine.features._utils import prepare_features


class _AE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        enc_dim = max(latent_dim * 2, input_dim // 3)
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, latent_dim),
            nn.ReLU(),
        )
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(self._encoder(x))


class AutoencoderDetector:
    def __init__(
        self,
        latent_dim: int = 32,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        contamination: float = 0.05,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._contamination = contamination
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self._net: _AE | None = None
        self._threshold: float = 0.5
        self._mean: np.ndarray = np.array([])
        self._std: np.ndarray = np.array([])
        self._feature_names: list[str] = []

    def _fit_scaler(self, X_licit: np.ndarray) -> None:
        self._mean = np.mean(X_licit, axis=0)
        self._std = np.std(X_licit, axis=0)
        self._std[self._std < 1e-8] = 1.0

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> AutoencoderDetector:
        X_df, y_s = prepare_features(X, pd.Series(y) if not isinstance(y, pd.Series) else y)
        self._feature_names = list(X_df.columns)
        X_np = X_df.to_numpy(dtype=np.float32)

        licit_mask = (np.asarray(y_s) == 0)
        X_licit = X_np[licit_mask]

        self._fit_scaler(X_licit)
        X_scaled = self._transform(X_licit)

        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

        n_features = X_scaled.shape[1]
        self._net = _AE(n_features, self._latent_dim).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        self._net.train()
        for _ in range(self._epochs):
            for (batch,) in loader:
                batch = batch.to(self._device, non_blocking=True)
                recon = self._net(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self._net.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
            recon = self._net(x_t)
            errors = torch.mean((recon - x_t) ** 2, dim=1).cpu()
            errors_np = errors.numpy()

        quantile = 1.0 - self._contamination
        self._threshold = float(np.quantile(errors_np, quantile))

        return self

    def _reconstruction_error(self, X_np: np.ndarray) -> np.ndarray:
        X_scaled = self._transform(X_np)
        self._net.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
            recon = self._net(x_t)
            errors = torch.mean((recon - x_t) ** 2, dim=1).cpu()
        return errors.numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_df, _ = prepare_features(X)
        X_np = X_df.to_numpy(dtype=np.float32)
        errors = self._reconstruction_error(X_np)

        probs = np.clip(errors / (self._threshold + 1e-9), 0.0, 1.0)
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
