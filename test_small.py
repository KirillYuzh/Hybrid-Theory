# test_small.py — verify all 4 streams on small data
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
print("Python:", sys.version, flush=True)

# === Test 1: Stream D — Node2Vec on tiny graph ===
print("\n=== Stream D: Node2Vec ===", flush=True)
import pandas as pd
import numpy as np
edges = pd.DataFrame({"txId1": [1,2,3,4,5,1,2], "txId2": [2,3,4,5,1,3,5]})
from kyt_engine.features.graph_embeddings import build_node2vec_embeddings
emb = build_node2vec_embeddings(edges, [1,2,3,4,5,6], dimensions=8, num_walks=5)
assert emb.shape == (6, 8), f"Expected (6,8), got {emb.shape}"
assert emb.loc[6].sum() == 0.0, "Node 6 should be zeros"
print("  PASSED", flush=True)

# === Test 2: Stream C — VAE ===
print("\n=== Stream C: VAE ===", flush=True)
from kyt_engine.models.autoencoder import VAEDetector
X = np.random.randn(500, 167).astype(np.float32)
y = np.array([0]*450 + [1]*50)
vae = VAEDetector(latent_dim=16, epochs=5, batch_size=128, device="cpu")
vae.fit(pd.DataFrame(X), y)
proba = vae.predict_proba(pd.DataFrame(X))
assert proba.shape == (500, 2), f"Expected (500,2), got {proba.shape}"
assert 0 <= proba[:,1].mean() <= 1, "Probas should be in [0,1]"
print("  PASSED", flush=True)

# === Test 3: Stream C — Incremental demo (import only) ===
print("\n=== Stream C: Incremental demo ===", flush=True)
from kyt_engine.training.demo_incremental import run_incremental_demo
print("  Import OK", flush=True)

# === Test 4: Stream B — Pseudo-labeling ===
print("\n=== Stream B: Supplement ===", flush=True)
from kyt_engine.training.supplement import augment_illicit, pseudo_label
X_small = pd.DataFrame(np.random.randn(100, 10))
y_small = np.array([0]*80 + [1]*20)
X_aug, y_aug = augment_illicit(X_small, y_small, n_copies=2, noise_std=0.01)
assert len(X_aug) == 100 + 40, f"Expected 140, got {len(X_aug)}"
assert y_aug.sum() == 20 * 3, f"Expected 60 illicit, got {y_aug.sum()}"
print("  PASSED", flush=True)

# === Test 5: Stream A — Temporal split (unit test) ===
print("\n=== Stream A: Temporal split ===", flush=True)
from kyt_engine.training.train_real import temporal_split
df = pd.DataFrame({
    "txId": range(200),
    "time_step": np.tile(np.arange(1, 51), 4)[:200],
    **{f"f{i}": np.random.randn(200) for i in range(165)}
})
label_map = {i: (0 if i < 160 else 1) for i in range(200)}
result = temporal_split(df, label_map, train_end=36, val_end=44)
assert "X_train" in result
assert "X_val" in result
assert "X_test" in result
assert "time_step" not in result["feature_cols"], "time_step should not be a feature"
assert "label" not in result["feature_cols"], "label should not be a feature"
print(f"  Train: {len(result['X_train'])}, Val: {len(result['X_val'])}, Test: {len(result['X_test'])}", flush=True)
print(f"  Features: {len(result['feature_cols'])}", flush=True)
print("  PASSED", flush=True)

# === Test 6: Stream A — Drift analysis (unit test) ===
print("\n=== Stream A: Drift analysis ===", flush=True)
from kyt_engine.training.train_real import drift_analysis
from lightgbm import LGBMClassifier
X_dummy = pd.DataFrame(np.random.randn(100, 10))
y_dummy = np.array([0]*80 + [1]*20)
m = LGBMClassifier(n_estimators=10, verbose=-1, n_jobs=1)
m.fit(X_dummy, y_dummy)
df_drift = pd.DataFrame({
    "txId": range(50),
    "time_step": np.tile(np.arange(1, 11), 5)[:50],
    **{f"f{i}": np.random.randn(50) for i in range(10)}
})
lm_drift = {i: (0 if i < 40 else 1) for i in range(50)}
dr = drift_analysis(m, df_drift, lm_drift, [f"f{i}" for i in range(10)])
print(f"  Drift steps evaluated: {len(dr)}", flush=True)
print("  PASSED", flush=True)

print("\n=== ALL SMALL DATA TESTS PASSED ===", flush=True)
