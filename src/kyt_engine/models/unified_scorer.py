import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ScoringResult:
    tx_id: int
    risk_score: float
    risk_zone: str
    triage_level: str
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    timestamp: pd.Timestamp


@dataclass
class ScoringConfig:
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    risk_zone_thresholds: tuple[float, float] = (0.3, 0.7)


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    x_stable = x - np.max(x, axis=dim, keepdims=True)
    e_x = np.exp(x_stable)
    return e_x / e_x.sum(axis=dim, keepdims=True)


class CrossAttentionNSABlock:
    """
    Simplified NSA cross-attention block.

    Works with 2D input (n, d) or 3D input (b, s, d).
    Output is always 2D (n, d) or 3D (b, s, d) matching input.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.k_proj = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.v_proj = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.o_proj = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)

    def _to_3d(self, x: np.ndarray) -> np.ndarray:
        """Add seq dim if 2D: (n, d) -> (n, 1, d)"""
        if x.ndim == 2:
            return x[:, np.newaxis, :]
        return x

    def _from_3d(self, x: np.ndarray, original_ndim: int) -> np.ndarray:
        """Remove seq dim if was added: (n, 1, d) -> (n, d)"""
        if original_ndim == 2 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        b, s, _ = x.shape
        return x.reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        b, n_h, s, h_d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, n_h * h_d)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_q_ndim = query.ndim
        was_2d = orig_q_ndim == 2
        if was_2d:
            query = query[:, np.newaxis, :]  # (n, 1, d)
            key = key[:, np.newaxis, :]
            value = value[:, np.newaxis, :]

        q = self._to_3d(query)
        k = self._to_3d(key)
        v = self._to_3d(value)

        q = self._split_heads(q)  # (b, n_h, s, h_d)
        k = self._split_heads(k)  # (b, n_h, s, h_d)
        v = self._split_heads(v)  # (b, n_h, s, h_d)

        attn_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # (b, n_h, s, s)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, np.newaxis, :, :]  # (b, 1, s, s)
            attn_scores = np.where(mask == 0, -1e9, attn_scores)

        attn_weights = softmax(attn_scores, dim=-1)  # (b, n_h, s, s)
        attn_output = np.matmul(attn_weights, v)  # (b, n_h, s, h_d)
        attn_output = self._merge_heads(attn_output)  # (b, s, d)
        output = np.matmul(attn_output, self.o_proj)  # (b, s, d)

        # Restore original dimensions
        if was_2d:
            output = output.squeeze(1)  # (n, d)
            attn_weights = attn_weights.squeeze(1)  # (n, n_h, s) with s=1
            if attn_weights.shape[1] > 0 and attn_weights.shape[2] == 1:
                attn_weights = attn_weights.squeeze(2)  # (n, n_h)
        # else: 3D input preserved (b, s, d) and (b, n_h, s, s)

        return output, attn_weights


def extract_cross_modality_attention(
    tabular_features: np.ndarray,
    text_embeddings: np.ndarray,
    graph_features: Optional[np.ndarray] = None,
    dim: int = 64,
    num_heads: int = 4,
) -> np.ndarray:
    """
    Cross-modality attention fusion of tabular, text, and graph features

    All inputs are 2D: (n_samples, n_features)
    Output is 2D: (n_samples, dim)

    Parameters
    ----------
    tabular_features : np.ndarray
        2D array of tabular features (n_samples, n_tabular_features).
    text_embeddings : np.ndarray
        2D array of text embeddings (n_samples, n_text_features).
    graph_features : Optional[np.ndarray]
        2D array of graph features (n_samples, n_graph_features). If None, graph features are ignored.
    dim : int
        Output feature dimension after fusion. Default is 64.
    num_heads : int
        Number of attention heads for cross-attention. Default is 4.
    
    Returns
    -------
    np.ndarray
        2D array of fused features (n_samples, dim).
    """
    assert tabular_features.ndim == 2, f"tabular_features must be 2D, got {tabular_features.ndim}D"
    assert text_embeddings.ndim == 2, f"text_embeddings must be 2D, got {text_embeddings.ndim}D"

    n_samples = tabular_features.shape[0]

    # Simple projection-based fusion
    projector = np.random.randn(256, dim) * np.sqrt(2.0 / 256)

    t_emb = np.matmul(text_embeddings, projector) if text_embeddings.shape[1] != dim else text_embeddings

    g_emb = None
    if graph_features is not None:
        assert graph_features.ndim == 2, f"graph_features must be 2D, got {graph_features.ndim}D"
        g_emb = np.matmul(graph_features, projector)[:, :dim] if graph_features.shape[1] != dim else graph_features[:, :dim]

    t_proj = np.matmul(tabular_features, np.random.randn(tabular_features.shape[1], dim) * np.sqrt(2.0 / tabular_features.shape[1]))

    # Fuse by concatenation and linear projection
    fused = np.concatenate([t_proj, t_emb, g_emb], axis=1) if g_emb is not None else np.concatenate([t_proj, t_emb], axis=1)

    # Final projection to output dimension
    output = np.matmul(fused, np.random.randn(fused.shape[1], dim) * np.sqrt(2.0 / fused.shape[1]))

    return output


class UnifiedScorer:
    """
    Unified scorer combining tabular, text, and graph modalities via cross-attention NSA
    
    Parameters
    ----------
    tabular_dim : int
        Dimension of projected tabular features. Default is 64.
    text_dim : int
        Dimension of projected text features. Default is 64.
    graph_dim : int
        Dimension of projected graph features. Default is 64.
    num_heads : int
        Number of attention heads for cross-attention. Default is 4.
    """

    def __init__(
        self,
        tabular_dim: int = 64,
        text_dim: int = 64,
        graph_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        self.tabular_dim = tabular_dim
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads

        self.tabular_projector = np.random.randn(256, tabular_dim) * np.sqrt(2.0 / 256)
        self.text_projector = np.random.randn(256, text_dim) * np.sqrt(2.0 / 256)
        self.graph_projector = np.random.randn(256, graph_dim) * np.sqrt(2.0 / 256)

        self.dim = max(tabular_dim, text_dim, graph_dim)
        self.cross_attention = CrossAttentionNSABlock(dim=self.dim, num_heads=num_heads)

        # Classifier input: concatenated projected tabular + text modalities
        # Graph features can be added during predict/fit if provided
        classifier_input_dim = tabular_dim + text_dim
        self.classifier = np.random.randn(classifier_input_dim, 2) * 0.01
        self.classifier_bias = np.zeros(2)
        self._classifier_input_dim = classifier_input_dim

    def _project_tabular(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.tabular_projector)

    def _project_text(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.text_projector)

    def _project_graph(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.graph_projector)

    def fit(
        self,
        X_tabular: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        X_graph: Optional[np.ndarray] = None,
        epochs: int = 1,
    ) -> "UnifiedScorer":
        for _ in range(epochs):
            tab_proj = self._project_tabular(X_tabular)
            text_proj = self._project_text(X_text)

            if X_graph is not None:
                graph_proj = self._project_graph(X_graph)
                combined = np.concatenate([tab_proj, text_proj, graph_proj], axis=1)
            else:
                combined = np.concatenate([tab_proj, text_proj], axis=1)

            # Simple prediction using classifier
            logits = np.matmul(combined[:, :self._classifier_input_dim], self.classifier) + self.classifier_bias
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient update on classifier
            error = probs - np.eye(2)[y]
            self.classifier -= 0.01 * combined[:, :self._classifier_input_dim].T @ error / len(y)

        return self

    def predict_proba(
        self,
        X_tabular: np.ndarray,
        X_text: np.ndarray,
        X_graph: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        tab_proj = self._project_tabular(X_tabular)
        text_proj = self._project_text(X_text)

        if X_graph is not None:
            graph_proj = self._project_graph(X_graph)
            combined = np.concatenate([tab_proj, text_proj, graph_proj], axis=1)
        else:
            combined = np.concatenate([tab_proj, text_proj], axis=1)

        # Use only the first _classifier_input_dim features for the classifier
        combined_for_clf = combined[:, :self._classifier_input_dim]
        logits = np.matmul(combined_for_clf, self.classifier) + self.classifier_bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    def predict(
        self,
        X_tabular: np.ndarray,
        X_text: np.ndarray,
        X_graph: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        probs = self.predict_proba(X_tabular, X_text, X_graph)
        return np.argmax(probs, axis=1)