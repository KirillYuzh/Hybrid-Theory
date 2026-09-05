import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.dec3(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


class VAEDetector:
    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        contamination: float = 0.05,
        device: str = "cpu",
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.contamination = contamination
        self.device = device
        self.model: Optional[VAE] = None
        self.scaler = StandardScaler()
        self.threshold: float = 0.0
        self.is_fitted: bool = False
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "VAEDetector":
        # Use only normal (licit) samples for training
        if y is not None:
            normal_mask = y == 0
            if not normal_mask.any():
                # No normal samples, use all
                X_train = X
            else:
                X_train = X[normal_mask]
        else:
            X_train = X
        
        # Prepare data
        self._feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X_train.to_numpy(dtype=np.float64))
        
        # Create model
        input_dim = X_scaled.shape[1]
        self.model = VAE(input_dim, self.latent_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Training loop
        self.model.train()
        n_samples = X_tensor.shape[0]
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(0, n_samples, self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch)
                loss = vae_loss(recon, batch, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Compute anomaly threshold on training data
        self.model.eval()
        with torch.no_grad():
            recon, mu, logvar = self.model(X_tensor)
            recon_errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
            kl_errors = -0.5 * np.sum(1 + logvar.cpu().numpy() - mu.cpu().numpy() ** 2 - np.exp(logvar.cpu().numpy()), axis=1)
            anomaly_scores = recon_errors + kl_errors
            self.threshold = float(np.percentile(anomaly_scores, 100 * (1 - self.contamination)))
        
        self.is_fitted = True
        return self

    def _compute_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("VAEDetector must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X.to_numpy(dtype=np.float64))
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            recon, mu, logvar = self.model(X_tensor)
            recon_errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
            kl_errors = -0.5 * np.sum(1 + logvar.cpu().numpy() - mu.cpu().numpy() ** 2 - np.exp(logvar.cpu().numpy()), axis=1)
            anomaly_scores = recon_errors + kl_errors
        
        return anomaly_scores

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._compute_anomaly_scores(X)
        # Normalize to [0, 1] range using threshold
        probs = np.clip(scores / self.threshold, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names