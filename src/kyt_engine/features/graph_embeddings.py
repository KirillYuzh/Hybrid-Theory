from __future__ import annotations

import logging
import os
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)

logger = logging.getLogger(__name__)


def _biased_walk(
    g: nx.Graph,
    start: int,
    walk_length: int,
    p: float,
    q: float,
    rng: random.Random,
) -> list[int]:
    nbr_cache: dict[int, list[int]] = {}
    edge_set: set[tuple[int, int]] = set()
    walk = [start]
    for _ in range(walk_length - 1):
        curr = walk[-1]
        if curr not in nbr_cache:
            nbr_cache[curr] = list(g.neighbors(curr))
        nbrs = nbr_cache[curr]
        if not nbrs:
            break
        prev = walk[-2] if len(walk) > 1 else curr
        weights = []
        for nbr in nbrs:
            if nbr == prev:
                weights.append(1.0 / p)
            else:
                edge = (prev, nbr) if prev < nbr else (nbr, prev)
                if edge not in edge_set:
                    if g.has_edge(prev, nbr):
                        edge_set.add(edge)
                if edge in edge_set:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
        total = sum(weights)
        probs = [w / total for w in weights]
        walk.append(rng.choices(nbrs, weights=probs, k=1)[0])
    return walk


def _generate_walks(
    g: nx.Graph,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    rng: random.Random,
) -> list[list[str]]:
    nodes = list(g.nodes())
    all_walks: list[list[str]] = []
    logger.info("Generating %d walks of length %d for %d nodes", num_walks, walk_length, len(nodes))
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = _biased_walk(g, node, walk_length, p, q, rng)
            all_walks.append([str(w) for w in walk])
    return all_walks


class _SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor, negatives: torch.Tensor):
        center_emb = self.embeddings(center)
        context_emb = self.output_embeddings(context)
        neg_emb = self.output_embeddings(negatives)
        pos_score = torch.sum(center_emb * context_emb, dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return (pos_loss + neg_loss).mean()


def _train_skipgram(
    walks: list[list[str]],
    embed_dim: int,
    window: int = 10,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 0.025,
    neg_samples: int = 5,
    min_count: int = 1,
) -> dict[str, np.ndarray]:
    freq: dict[str, int] = defaultdict(int)
    for w in walks:
        for token in w:
            freq[token] += 1

    vocab = [t for t, c in freq.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    pairs: list[tuple[int, int]] = []
    for walk in walks:
        indices = [word2idx[t] for t in walk if t in word2idx]
        for i, center in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i != j:
                    pairs.append((center, indices[j]))

    if not pairs:
        logger.warning("No training pairs generated")
        return {w: np.zeros(embed_dim) for w in vocab}

    centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    dataset = TensorDataset(centers, contexts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    freq_arr = np.array([freq.get(vocab[i], 1) for i in range(vocab_size)], dtype=np.float64)
    neg_dist = freq_arr ** 0.75
    neg_dist /= neg_dist.sum()

    model = _SkipGram(vocab_size, embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("Training skip-gram: %d pairs, %d vocab, %d epochs", len(pairs), vocab_size, epochs)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for center_batch, context_batch in loader:
            neg_indices = np.random.choice(
                vocab_size, size=(len(center_batch), neg_samples), replace=True, p=neg_dist
            )
            neg_tensor = torch.tensor(neg_indices, dtype=torch.long)
            loss = model(center_batch, context_batch, neg_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(center_batch)
        if (epoch + 1) % max(1, epochs // 2) == 0 or epoch == epochs - 1:
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, total_loss / len(pairs))

    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(vocab_size)
        all_emb = model.embeddings(all_indices).numpy()

    return {vocab[i]: all_emb[i] for i in range(vocab_size)}


def build_node2vec_embeddings(
    edges: pd.DataFrame,
    tx_ids: list[int] | np.ndarray,
    dimensions: int = 64,
    walk_length: int = 40,
    num_walks: int = 10,
    random_state: int = 42,
    p: float = 1.0,
    q: float = 1.0,
) -> pd.DataFrame:
    """Build Node2Vec embeddings for transaction graph.

    Args:
        edges: DataFrame with columns [txId1, txId2]
        tx_ids: list of transaction IDs to generate embeddings for
        dimensions: embedding dimension (default 64)
        walk_length: length of random walks
        num_walks: number of walks per node

    Returns:
        DataFrame indexed by txId with columns n2v_0..n2v_{dimensions-1}
    """
    if dimensions <= 0:
        raise ValueError(f"dimensions must be > 0, got {dimensions}")

    rng = random.Random(random_state)

    if len(edges) == 0:
        logger.warning("No edges provided, returning zero embeddings")
        df = pd.DataFrame(
            np.zeros((len(tx_ids), dimensions), dtype=np.float64),
            index=pd.Index(tx_ids, name="txId"),
            columns=[f"n2v_{i}" for i in range(dimensions)],
        )
        return df

    logger.info("Building NetworkX graph from %d edges", len(edges))
    g = nx.from_pandas_edgelist(edges, "txId1", "txId2")

    walks = _generate_walks(g, num_walks, walk_length, p, q, rng)

    emb_map = _train_skipgram(walks, embed_dim=dimensions, window=10, epochs=5)

    logger.info("Extracting embeddings for %d target nodes", len(tx_ids))
    embeddings = {}
    missing = 0
    for tx in tx_ids:
        key = str(tx)
        if key in emb_map:
            embeddings[tx] = emb_map[key]
        else:
            embeddings[tx] = np.zeros(dimensions)
            missing += 1

    if missing > 0:
        logger.warning("%d nodes not in graph, using zero vectors", missing)

    df = pd.DataFrame(embeddings).T
    df.columns = [f"n2v_{i}" for i in range(dimensions)]
    df.index.name = "txId"
    return df


if __name__ == "__main__":
    edges = pd.DataFrame({
        "txId1": [1, 2, 3, 4, 5, 1, 2],
        "txId2": [2, 3, 4, 5, 1, 3, 5],
    })
    tx_ids = [1, 2, 3, 4, 5, 6]
    emb = build_node2vec_embeddings(edges, tx_ids, dimensions=8, num_walks=5)
    print(f"Shape: {emb.shape}")
    print(f"Columns: {list(emb.columns)}")
    print(emb.head())
