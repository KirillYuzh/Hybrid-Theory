import numpy as np
import pytest

from kyt_engine.models.unified_scorer import (
    CrossAttentionNSABlock,
    extract_cross_modality_attention,
    UnifiedScorer,
    softmax,
)


def test_softmax_basic():
    x = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    result = softmax(x, dim=-1)
    assert np.allclose(result.sum(axis=-1), 1.0)
    assert result.shape == x.shape


def test_cross_attention_nsa_block_basic():
    block = CrossAttentionNSABlock(dim=16, num_heads=4)
    batch_size, seq_len, dim = 2, 5, 16
    query = np.random.randn(batch_size, seq_len, dim)
    key = np.random.randn(batch_size, seq_len, dim)
    value = np.random.randn(batch_size, seq_len, dim)

    output, attn_weights = block.forward(query, key, value)
    assert output.shape == (batch_size, seq_len, dim)
    assert attn_weights.shape == (batch_size, block.num_heads, seq_len, seq_len)


def test_cross_attention_nsa_block_with_mask():
    block = CrossAttentionNSABlock(dim=16, num_heads=4)
    batch_size, seq_len, dim = 2, 5, 16
    query = np.random.randn(batch_size, seq_len, dim)
    key = np.random.randn(batch_size, seq_len, dim)
    value = np.random.randn(batch_size, seq_len, dim)
    mask = np.ones((batch_size, seq_len, seq_len))
    mask[:, :, -1] = 0

    output, attn_weights = block.forward(query, key, value, mask=mask)
    assert output.shape == (batch_size, seq_len, dim)


def test_extract_cross_modality_attention_basic():
    tabular = np.random.randn(3, 256)
    text = np.random.randn(3, 256)
    result = extract_cross_modality_attention(tabular, text, dim=64, num_heads=4)
    assert result.shape == (3, 64)


def test_extract_cross_modality_attention_with_graph():
    tabular = np.random.randn(3, 256)
    text = np.random.randn(3, 256)
    graph = np.random.randn(3, 256)
    result = extract_cross_modality_attention(tabular, text, graph, dim=64, num_heads=4)
    assert result.shape == (3, 64)


def test_unified_scorer_init():
    scorer = UnifiedScorer(tabular_dim=64, text_dim=64, graph_dim=64, num_heads=4)
    assert scorer is not None
    assert scorer.tabular_dim == 64


def test_unified_scorer_predict_proba_basic():
    scorer = UnifiedScorer(tabular_dim=64, text_dim=64, graph_dim=64, num_heads=4)

    X_tabular = np.random.randn(10, 256)
    X_text = np.random.randn(10, 256)

    probs = scorer.predict_proba(X_tabular, X_text)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_unified_scorer_predict_basic():
    scorer = UnifiedScorer(tabular_dim=64, text_dim=64, graph_dim=64, num_heads=4)

    X_tabular = np.random.randn(10, 256)
    X_text = np.random.randn(10, 256)

    preds = scorer.predict(X_tabular, X_text)
    assert preds.shape == (10,)
    assert all(p in [0, 1] for p in preds)


def test_unified_scorer_with_graph():
    scorer = UnifiedScorer(tabular_dim=64, text_dim=64, graph_dim=64, num_heads=4)

    X_tabular = np.random.randn(10, 256)
    X_text = np.random.randn(10, 256)
    X_graph = np.random.randn(10, 256)

    probs = scorer.predict_proba(X_tabular, X_text, X_graph)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_unified_scorer_fit_basic():
    scorer = UnifiedScorer(tabular_dim=64, text_dim=64, graph_dim=64, num_heads=4)

    X_tabular = np.random.randn(20, 256)
    X_text = np.random.randn(20, 256)
    y = np.random.randint(0, 2, size=20)

    scorer.fit(X_tabular, X_text, y, epochs=1)
    probs = scorer.predict_proba(X_tabular, X_text)
    assert probs.shape == (20, 2)