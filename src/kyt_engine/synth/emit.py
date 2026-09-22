from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .graph import GeneratedGraph
from .stats import CLASS_TO_LABEL, N_FEATURES


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(
    config: GeneratorConfig,
    graph: GeneratedGraph,
    features: np.ndarray,
    out_dir: Path,
) -> None:
    """Emit the three CSVs in Elliptic++ format (spillety.loader contract) + manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, 2 + N_FEATURES)]
    feats = pd.DataFrame(features, columns=cols)
    feats["time_step"] = feats["time_step"].astype(int)
    feats["txId"] = feats["txId"].astype(int)
    feats.to_csv(out_dir / "elliptic_txs_features.csv", header=False, index=False)

    classes = pd.DataFrame(
        {
            "txId": [nd.tx_id for nd in graph.nodes],
            "class": [CLASS_TO_LABEL[nd.cls] for nd in graph.nodes],
        }
    )
    classes.to_csv(out_dir / "elliptic_txs_classes.csv", index=False)

    edge_df = pd.DataFrame(graph.edges, columns=["txId1", "txId2"])
    edge_df.to_csv(out_dir / "elliptic_txs_edgelist.csv", index=False)

    ground_truth = [
        {"txId": nd.tx_id, "scheme": nd.scheme, "role": nd.role, "class": nd.cls}
        for nd in graph.nodes
    ]
    manifest = {
        "seed": config.seed,
        "config": config.to_dict(),
        "sha256": {
            "features": _sha256(out_dir / "elliptic_txs_features.csv"),
            "classes": _sha256(out_dir / "elliptic_txs_classes.csv"),
            "edgelist": _sha256(out_dir / "elliptic_txs_edgelist.csv"),
        },
        "ground_truth": ground_truth,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
