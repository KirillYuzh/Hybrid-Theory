from pathlib import Path

import pandas as pd

from kyt_engine.data.validators import validate_columns, validate_file_exists

REQUIRED_NODES = ["txId", "timestamp"]
REQUIRED_EDGES = ["txId1", "txId2"]
REQUIRED_CLASSES = ["txId", "label"]

NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"
CLASSES_FILE = "classes.csv"


def load_nodes(data_dir: Path) -> pd.DataFrame:
    path = data_dir / NODES_FILE
    validate_file_exists(path)
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_NODES, NODES_FILE)
    return df


def load_edges(data_dir: Path) -> pd.DataFrame:
    path = data_dir / EDGES_FILE
    validate_file_exists(path)
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_EDGES, EDGES_FILE)
    return df


def load_classes(data_dir: Path) -> pd.DataFrame:
    path = data_dir / CLASSES_FILE
    validate_file_exists(path)
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_CLASSES, CLASSES_FILE)
    return df


def load_elliptic(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(data_dir)
    validate_file_exists(base)
    return {
        "nodes": load_nodes(base),
        "edges": load_edges(base),
        "classes": load_classes(base),
    }
