"""Full pipeline: Behavioral → K-Score → Triage → External Labels → Active Learning."""

import time, numpy as np, pandas as pd, sys, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.flush()

# === LOAD ===
print("=== 1. Loading Elliptic ===", flush=True)
t0 = time.time()
features = pd.read_csv('data/raw/elliptic_txs_features.csv', header=None,
                        names=['txId','time_step'] + [f'f{i}' for i in range(165)])
classes = pd.read_csv('data/raw/elliptic_txs_classes.csv')
classes['label'] = classes['class'].map({'1':0,'2':1}).dropna().astype(int)
label_map = dict(zip(classes['txId'], classes['label']))
edges = pd.read_csv('data/raw/elliptic_txs_edgelist.csv')

tx_set = set(features['txId'])
valid = edges[edges['txId1'].isin(tx_set) & edges['txId2'].isin(tx_set)]
out_deg = valid.groupby('txId1').size().reset_index(name='out_degree').rename(columns={'txId1':'txId'})
in_deg = valid.groupby('txId2').size().reset_index(name='in_degree').rename(columns={'txId2':'txId'})
features = features.merge(out_deg, on='txId', how='left').merge(in_deg, on='txId', how='left')
features['in_degree'] = features['in_degree'].fillna(0).astype(int)
features['out_degree'] = features['out_degree'].fillna(0).astype(int)
features['label'] = features['txId'].map(label_map)
features = features.dropna(subset=['label'])
features['label'] = features['label'].astype(int)
features = features.drop_duplicates(subset=['txId'])
print(f"  {len(features)} labeled txs in {time.time()-t0:.1f}s (illicit={int(features['label'].sum())})", flush=True)

# === K-SCORE ===
print("\n=== 2. K-Score ===", flush=True)
t0 = time.time()
from kyt_engine.models.kscore import KScoreCalculator
feat_cols = [c for c in features.columns if c not in ('txId','label','time_step','from_address','to_address')]
ksc = KScoreCalculator(baseline_window=6)
ksc.fit(features, features['time_step'])
k_scores = ksc.score(features)
zones = ksc.classify(k_scores)
print(f"  Computed in {time.time()-t0:.1f}s, mean={k_scores.mean():.3f}", flush=True)
print(f"  Zones: {zones.value_counts().to_dict()}", flush=True)

# === TRIAGE ===
print("\n=== 3. Triage ===", flush=True)
from kyt_engine.models.triage import TriageSystem
# Use mock proba/entropy (in real system these come from LightGBM)
proba_mock = np.where(features['label'].values == 1, 0.7, 0.3) + np.random.normal(0, 0.1, len(features))
proba_mock = np.clip(proba_mock, 0, 1)
entropy_mock = -(proba_mock * np.log2(np.clip(proba_mock, 1e-10, 1-1e-10)) + 
                 (1-proba_mock) * np.log2(np.clip(1-proba_mock, 1e-10, 1-1e-10)))
tsys = TriageSystem()
triage_result = tsys.triage(k_scores, pd.Series(proba_mock), pd.Series(entropy_mock))
print(f"  {tsys.statistics(triage_result)}", flush=True)

# === EXTERNAL LABELS ===
print("\n=== 4. External Labels ===", flush=True)
from kyt_engine.data.scraper import ExternalLabelStore, LabeledAddress
store = ExternalLabelStore()
# Simulate scraped data (in production, run_full_scrape() would fetch real data)
sample_labels = [
    LabeledAddress(addr, 'illicit', 'ofac', '2026-09-01', 'sanctions')
    for addr in [f'0x{i:040x}' for i in range(10)]
]
sample_labels += [
    LabeledAddress(addr, 'illicit', 'cryptoscamdb', '2026-09-01', 'phishing')
    for addr in [f'0x{i:040x}' for i in range(10, 20)]
]
df_labels = store.merge(sample_labels)
store.save(df_labels)
illicit_addrs = store.get_illicit_addresses()
print(f"  {len(df_labels)} external labels, {len(illicit_addrs)} unique illicit addresses", flush=True)

# === LIGHTGBM BASELINE ===
print("\n=== 5. LightGBM Baseline (165 features) ===", flush=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

feat_cols_165 = [f'f{i}' for i in range(165)] + ['in_degree', 'out_degree']
X = features[feat_cols_165].replace([np.inf,-np.inf], np.nan).fillna(0)
y = features['label'].values.astype(int)

# Temporal split
train_mask = features['time_step'] <= 36
val_mask = (features['time_step'] > 36) & (features['time_step'] <= 44)
test_mask = features['time_step'] > 44

X_tr, y_tr = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_te, y_te = X[test_mask], y[test_mask]
print(f"  Split: train={len(X_tr)} val={len(X_val)} test={len(X_te)}", flush=True)

lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                       class_weight='balanced', verbose=-1, n_jobs=1, random_state=42)
t0 = time.time()
lgbm.fit(X_tr, y_tr)
print(f"  Trained in {time.time()-t0:.1f}s", flush=True)

# Validation
proba_val = lgbm.predict_proba(X_val)[:, 1]
pred_val = (proba_val >= 0.5).astype(int)
print("  Validation report:")
print(classification_report(y_val, pred_val, target_names=['licit','illicit']))
val_auc_roc = roc_auc_score(y_val, proba_val)
val_auc_pr = average_precision_score(y_val, proba_val)
print(f"  AUC-ROC: {val_auc_roc:.4f}")
print(f"  AUC-PR:  {val_auc_pr:.4f}")

# Test
proba_te = lgbm.predict_proba(X_te)[:, 1]
pred_te = (proba_te >= 0.5).astype(int)
print("  Test report:")
print(classification_report(y_te, pred_te, target_names=['licit','illicit']))
test_auc_roc = roc_auc_score(y_te, proba_te)
test_auc_pr = average_precision_score(y_te, proba_te)
print(f"  AUC-ROC: {test_auc_roc:.4f}")
print(f"  AUC-PR:  {test_auc_pr:.4f}")

# === ACTIVE LEARNING SAMPLE ===
print("\n=== 6. Active Learning Selection ===", flush=True)
from kyt_engine.training.active_learning import UncertaintySampler
k_score_vals = k_scores.values
sampler = UncertaintySampler(top_k=1000)
sel = sampler.select_samples(proba_val, k_score_vals[:len(X_val)], X_val)
high = sum(1 for s in sel if s.priority == 'HIGH')
med = sum(1 for s in sel if s.priority == 'MEDIUM')
low = sum(1 for s in sel if s.priority == 'LOW')
print(f"  Selected {len(sel)} samples: HIGH={high}, MEDIUM={med}, LOW={low}", flush=True)

# === SUMMARY ===
from sklearn.metrics import precision_score, recall_score, f1_score
val_prec = precision_score(y_val, pred_val, zero_division=0)
val_rec = recall_score(y_val, pred_val, zero_division=0)
val_f1 = f1_score(y_val, pred_val, zero_division=0)
test_prec = precision_score(y_te, pred_te, zero_division=0)
test_rec = recall_score(y_te, pred_te, zero_division=0)
test_f1 = f1_score(y_te, pred_te, zero_division=0)

# Confusion matrices
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val, pred_val)
cm_test = confusion_matrix(y_te, pred_te)

# Feature importance
importance = pd.Series(lgbm.feature_importances_, index=feat_cols_165)
importance = importance.sort_values(ascending=False)
top20 = importance.head(20)

print("\n" + "="*60, flush=True)
print("FULL PIPELINE COMPLETE", flush=True)
print("="*60, flush=True)
print(f"  Total labeled txs: {len(features)}", flush=True)
print(f"  Illicit: {int(y.sum())} ({y.mean()*100:.1f}%)", flush=True)
print(f"  K-Score mean: {k_scores.mean():.3f}", flush=True)
print(f"  Zones: {zones.value_counts().to_dict()}", flush=True)
print(f"  External labels: {len(df_labels)}", flush=True)
print(f"", flush=True)
print(f"  --- VAL (t=37-44, n={len(X_val)}) ---", flush=True)
print(f"  Precision: {val_prec:.4f}", flush=True)
print(f"  Recall:    {val_rec:.4f}", flush=True)
print(f"  F1:        {val_f1:.4f}", flush=True)
print(f"  AUC-ROC:   {val_auc_roc:.4f}", flush=True)
print(f"  AUC-PR:    {val_auc_pr:.4f}", flush=True)
print(f"  CM: {cm_val.tolist()}", flush=True)
print(f"", flush=True)
print(f"  --- TEST (t=45-49, n={len(X_te)}) ---", flush=True)
print(f"  Precision: {test_prec:.4f}", flush=True)
print(f"  Recall:    {test_rec:.4f}", flush=True)
print(f"  F1:        {test_f1:.4f}", flush=True)
print(f"  AUC-ROC:   {test_auc_roc:.4f}", flush=True)
print(f"  AUC-PR:    {test_auc_pr:.4f}", flush=True)
print(f"  CM: {cm_test.tolist()}", flush=True)
print(f"", flush=True)
print(f"  Active Learning samples: {len(sel)} (HIGH={high})", flush=True)
print(f"", flush=True)
print(f"  Top-20 features:", flush=True)
for feat, imp in top20.items():
    print(f"    {feat}: {imp}", flush=True)
