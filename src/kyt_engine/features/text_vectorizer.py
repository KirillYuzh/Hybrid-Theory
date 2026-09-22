import numpy as np
import pandas as pd
from collections import Counter


def _hash_token(token: str, num_buckets: int = 2**16) -> int:
    h = hash(token) & 0x7fffffff
    return h % num_buckets


def _tf_idf_weights(df: pd.DataFrame, text_col: str = "text_description",
                    doc_col: str = "address") -> tuple[pd.Series, pd.Series]:
    n_docs = df[doc_col].nunique()
    term_doc_counts = df.groupby(doc_col)[text_col].count()
    idf = np.log(n_docs / (term_doc_counts + 1))
    term_freq = df.groupby(doc_col)[text_col].apply(
        lambda s: s.value_counts(normalize=True)
    )
    tf_idf = term_freq.join(idf, on=text_col).fillna(0.0)
    return tf_idf, idf


def nsA_text_vectorize(
    df: pd.DataFrame,
    text_col: str = "text_description",
    doc_col: str = "address",
    num_features: int = 256,
) -> pd.DataFrame:
    if df.empty or text_col not in df.columns:
        n_docs = df[doc_col].nunique() if doc_col in df.columns else 0
        return pd.DataFrame(
            np.zeros((n_docs, num_features)), index=df[doc_col].unique()[:n_docs] if n_docs > 0 else [],
            dtype=np.float64
        )

    addresses = df[doc_col].unique().tolist()
    feature_rows = []

    for addr in addresses:
        addr_df = df[df[doc_col] == addr].copy()
        text_series = addr_df[text_col].fillna("").astype(str)
        text_values = text_series.values

        if len(text_values) == 0:
            zeros = np.zeros(num_features)
            feature_rows.append(zeros)
            continue

        vec = np.zeros(num_features)
        for text in text_values:
            if isinstance(text, str):
                tokens = text.lower().split()
                for token in tokens:
                    bucket = _hash_token(token, num_features)
                    vec[bucket] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        feature_rows.append(vec)

    result = pd.DataFrame(
        feature_rows, index=addresses, dtype=np.float64
    )
    result.columns = [f"text_nsA_dim_{i}" for i in range(num_features)]
    return result


def extract_text_features(
    df: pd.DataFrame,
    text_col: str = "text_description",
    doc_col: str = "address",
    num_features: int = 256,
) -> pd.DataFrame:
    if df.empty or text_col not in df.columns:
        return pd.DataFrame(index=df[doc_col].unique()[:0] if doc_col in df.columns else [], dtype=np.float64)

    vec_df = nsA_text_vectorize(df, text_col=text_col, doc_col=doc_col, num_features=num_features)

    all_features: list[dict[str, float]] = []
    for addr in df[doc_col].unique():
        addr_df = df[df[doc_col] == addr].copy()
        text_series = addr_df[text_col].fillna("").astype(str)
        base_feats = {
            "text_length": int(text_series.str.len().sum()) if hasattr(text_series, 'str') else 0,
            "text_word_count": int(text_series.str.split().apply(len).sum()) if hasattr(text_series, 'str') else 0,
            "text_unique_words": len(set(
                w.lower() for t in text_series.astype(str) for w in t.split()
            )),
        }
        merged = {**base_feats, **dict(zip(vec_df.columns, vec_df.loc[addr]))}
        all_features.append(merged)

    result = pd.DataFrame(all_features, index=vec_df.index)
    for col in result.columns:
        result[col] = result[col].astype(np.float64).fillna(0.0)
    return result