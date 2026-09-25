import hashlib
import json
import string
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .behavior import get_action, get_chain, get_profile, registry_sha256
from .stats import N_STEPS as TIME_STEP_MAX
from .stats import STEP_MIN as TIME_STEP_MIN

VALID_CLASSES = {"1", "2", "unknown"}


def _read_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat_path = root / "elliptic_txs_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"features not found: {feat_path}")

    features = pd.read_csv(feat_path, header=None)
    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    edgelist = pd.read_csv(root / "elliptic_txs_edgelist.csv")

    n_cols = features.shape[1]
    features.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, n_cols)]
    return features, classes, edgelist


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _validate_manifest_hashes(root: Path) -> None:
    manifest_path = root / "manifest.json"
    _check(manifest_path.exists(), "manifest.json is missing")
    manifest = _load_json(manifest_path)
    hashes = manifest.get("sha256")
    _check(isinstance(hashes, dict), "manifest sha256 must be a mapping")
    paths = {
        "features": root / "elliptic_txs_features.csv",
        "classes": root / "elliptic_txs_classes.csv",
        "edgelist": root / "elliptic_txs_edgelist.csv",
        "edge_attributes": root / "elliptic_txs_edge_attributes.csv",
    }
    _check(set(hashes) == set(paths), "manifest CSV hash keys are incomplete")
    for key, path in paths.items():
        _check(hashes[key] == _sha256(path), f"manifest hash mismatch: {key}")


def validate_dataset(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate the Elliptic++ CSV dataset contract.

    Features must have 167 columns, contiguous local IDs, and time steps 1..49.
    Classes may contain ``1``, ``2``, or ``unknown``; every edgelist endpoint must
    resolve to a feature ID, and manifest hashes must match the CSV files.

    Parameters
    ----------
    root : pathlib.Path
        Directory containing the three Elliptic CSV files and ``manifest.json``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Validated features, classes, and edgelist tables.

    Raises
    ------
    FileNotFoundError
        If a required dataset file is missing.
    ValueError
        If a schema, value, reference, manifest, or hash check fails.
    """
    features, classes, edgelist = _read_dataset(root)

    _check(features.shape[1] == 167, f"features has {features.shape[1]} columns, expected 167")
    _check(
        classes.shape == (len(features), 2),
        f"classes shape {classes.shape}, expected ({len(features)}, 2)",
    )
    _check(list(classes.columns) == ["txId", "class"], list(classes.columns))
    _check(edgelist.shape[1] == 2, f"edgelist has {edgelist.shape[1]} columns, expected 2")
    _check(list(edgelist.columns) == ["txId1", "txId2"], list(edgelist.columns))
    if not edgelist.empty:
        _check(edgelist["txId1"].dtype.kind == "i", edgelist["txId1"].dtype)
        _check(edgelist["txId2"].dtype.kind == "i", edgelist["txId2"].dtype)
    _check(features["time_step"].dtype.kind == "i", features["time_step"].dtype)
    _check(features["txId"].dtype.kind == "i", features["txId"].dtype)
    _check(features["txId"].is_unique, "dup txId in features")
    _check(
        features["txId"].to_numpy(dtype=np.int64).tolist() == list(range(len(features))),
        "txId must be contiguous local row IDs",
    )
    _check(
        features["time_step"].between(TIME_STEP_MIN, TIME_STEP_MAX).all(),
        "time_step out of 1..49",
    )
    _check(np.isfinite(features.iloc[:, 2:].to_numpy(dtype=float)).all(), "non-finite feature")
    class_vals = set(classes["class"].astype(str).unique())
    _check(class_vals <= VALID_CLASSES, class_vals)
    _check(classes["txId"].dtype.kind == "i", classes["txId"].dtype)
    _check(classes["txId"].is_unique, "dup txId in classes")
    _check(
        classes["txId"].to_numpy(dtype=np.int64).tolist()
        == features["txId"].to_numpy(dtype=np.int64).tolist(),
        "txId order mismatch feat/classes",
    )
    tx_ids = set(features["txId"])
    _check(edgelist["txId1"].isin(tx_ids).all(), "orphan txId1 in edgelist")
    _check(edgelist["txId2"].isin(tx_ids).all(), "orphan txId2 in edgelist")
    _validate_manifest_hashes(root)
    return features, classes, edgelist


def validate_edge_attributes(root: Path) -> pd.DataFrame:
    """Validate edge-attribute row alignment and value ranges.

    Parameters
    ----------
    root : pathlib.Path
        Directory containing the edge list and edge-attribute CSV files.

    Returns
    -------
    pandas.DataFrame
        Validated ``txId1``, ``txId2``, ``amount``, and ``timestamp`` table aligned
        row-for-row with the edgelist.

    Raises
    ------
    FileNotFoundError
        If the edge-attribute CSV or edgelist is missing.
    ValueError
        If columns, row alignment, endpoint types, or numeric ranges are invalid.
    """
    path = root / "elliptic_txs_edge_attributes.csv"
    if not path.exists():
        raise FileNotFoundError(f"edge attributes not found: {path}")
    edgelist = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    attrs = pd.read_csv(path)
    _check(list(attrs.columns) == ["txId1", "txId2", "amount", "timestamp"], list(attrs.columns))
    _check(
        len(attrs) == len(edgelist),
        f"edge attributes rows {len(attrs)}, edgelist {len(edgelist)}",
    )
    if not edgelist.empty:
        for column in ("txId1", "txId2"):
            _check(
                pd.api.types.is_integer_dtype(attrs[column]),
                f"{column} must contain integer IDs",
            )
            _check(
                np.isfinite(attrs[column].to_numpy(dtype=float)).all(),
                f"{column} must be finite",
            )
        _check(attrs["txId1"].equals(edgelist["txId1"]), "txId1 mismatch with edgelist")
        _check(attrs["txId2"].equals(edgelist["txId2"]), "txId2 mismatch with edgelist")
    if not attrs.empty:
        _check(pd.api.types.is_integer_dtype(attrs["timestamp"]), "timestamp must contain integers")
    for column in ("amount", "timestamp"):
        values = pd.to_numeric(attrs[column], errors="coerce").to_numpy(dtype=float)
        _check(np.isfinite(values).all(), f"{column} must be finite")
    _check((attrs["amount"] >= 0).all() and attrs["amount"].notna().all(), "amount must be >= 0")
    _check(attrs["timestamp"].ge(0).all(), "timestamp must be >= 0")
    return attrs


def validate_semantic_features(root: Path) -> pd.DataFrame:
    features, classes, edgelist = validate_dataset(root)
    attrs = validate_edge_attributes(root)
    from .features_semantic import build_semantic_features
    from .graph import GeneratedGraph, TxNode

    class_by_tx = dict(zip(classes["txId"], classes["class"], strict=True))
    graph = GeneratedGraph(
        nodes=[],
        edges=[
            (int(source), int(target))
            for source, target in edgelist[["txId1", "txId2"]].itertuples(index=False, name=None)
        ],
    )
    graph.nodes = [
        TxNode(
            tx_id=int(row.txId),
            step=int(row.time_step),
            cls={"1": "illicit", "2": "licit"}.get(class_by_tx[int(row.txId)], "unknown"),
            role="semantic",
            scheme="semantic",
        )
        for row in features.itertuples(index=False)
    ]
    manifest = _load_json(root / "manifest.json")
    if manifest.get("config", {}).get("graph", {}).get("mode") == "behavior":
        chain_by_tx = {
            int(item["txId"]): item["chain_id"]
            for item in manifest["behavior"]["chain_metadata"]["node_records"]
        }
        graph.node_meta = [
            SimpleNamespace(tx_id=node.tx_id, chain_id=chain_by_tx[node.tx_id])
            for node in graph.nodes
        ]
    expected = build_semantic_features(graph, attrs.to_dict("records"))
    actual = features.iloc[:, 2:].to_numpy(dtype=float)
    _check(actual.shape == expected.shape, "semantic feature width mismatch")
    _check(np.allclose(actual, expected, rtol=1e-9, atol=1e-9), "semantic feature matrix mismatch")
    return features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_behavior_dataset(root: Path) -> dict:
    features, classes, edgelist = validate_dataset(root)
    attrs = validate_edge_attributes(root)
    manifest = _load_json(root / "manifest.json")
    _check(
        manifest.get("config", {}).get("graph", {}).get("mode") == "behavior",
        "behavior config marker is missing",
    )
    behavior = manifest.get("behavior")
    _check(isinstance(behavior, dict), "behavior manifest is missing")
    required = {
        "behavioral_profiles",
        "entities",
        "events",
        "instances",
        "drift_schedule",
        "chain_metadata",
        "sanctions_anchor_map",
        "calibration",
    }
    _check(required <= set(behavior), "behavior manifest fields are incomplete")
    _check(
        behavior.get("contract_version") == "behavior.p0.v1", "behavior contract version mismatch"
    )
    _check(
        behavior["behavioral_profiles"].get("registry_version") == 1, "registry version mismatch"
    )
    _check(
        behavior["behavioral_profiles"].get("registry_sha256") == registry_sha256(),
        "behavior registry hash mismatch",
    )
    _check(
        behavior.get("rng_contract", {}).get("generator") == "PCG64",
        "behavior RNG contract mismatch",
    )
    tx_ids = set(features["txId"])
    config = manifest.get("config")
    _check(isinstance(config, dict), "manifest config is missing")
    _check(config.get("seed") == manifest.get("seed"), "manifest seed mismatch")
    _check(config.get("n_txs") == len(features), "manifest n_txs mismatch")
    _check(config.get("graph", {}).get("mode") == "behavior", "manifest graph mode mismatch")
    behavior_config = config.get("behavior")
    _check(isinstance(behavior_config, dict), "manifest behavior config is missing")
    _check(
        behavior_config.get("contract_version") == behavior["contract_version"],
        "manifest behavior contract mismatch",
    )
    quotas = behavior_config.get("profile_quotas")
    _check(isinstance(quotas, dict), "manifest profile quotas are missing")
    _check(
        sum(item.get("tx_quota", -1) for item in quotas.values()) == len(features),
        "manifest profile quota total mismatch",
    )
    _check(
        all(set(item) == {"entity_count", "tx_quota"} for item in quotas.values()),
        "manifest profile quota fields mismatch",
    )
    _check(
        config.get("features", {}).get("mode") == manifest.get("feature_semantics", {}).get("mode"),
        "manifest feature mode mismatch",
    )
    _check(config.get("features", {}).get("mode") == "semantic", "manifest feature mode is invalid")
    validate_semantic_features(root)
    gnn_config = config.get("gnn")
    _check(isinstance(gnn_config, dict), "manifest GNN config is missing")
    _check(
        gnn_config.get("backend") in {"auto", "full", "sgc"}
        and isinstance(gnn_config.get("seeds"), list)
        and bool(gnn_config["seeds"])
        and all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in gnn_config["seeds"]
        )
        and len(set(gnn_config["seeds"])) == len(gnn_config["seeds"]),
        "manifest GNN config is invalid",
    )
    edge_config = config.get("edge_attributes")
    _check(isinstance(edge_config, dict), "manifest edge config is missing")
    _check(
        isinstance(edge_config.get("timestamp_scale"), int)
        and not isinstance(edge_config.get("timestamp_scale"), bool)
        and edge_config["timestamp_scale"] >= 0,
        "manifest edge timestamp config is invalid",
    )
    configured_chains = behavior_config.get("chains")
    _check(isinstance(configured_chains, dict), "manifest chain config is missing")
    _check(
        set(configured_chains) == {"bitcoin", "ethereum", "tron"},
        "manifest chain config is incomplete",
    )
    _check(
        all(
            set(value) == {"enabled"} and isinstance(value["enabled"], bool)
            for value in configured_chains.values()
        ),
        "manifest chain config is invalid",
    )
    enabled_chain_ids = {
        get_chain(chain_key)["chain_id"]
        for chain_key, value in configured_chains.items()
        if value["enabled"]
    }
    _check(bool(enabled_chain_ids), "manifest has no enabled behavior chain")
    _check(
        behavior["drift_schedule"].get("configured") == behavior_config.get("drift"),
        "manifest behavior drift config mismatch",
    )
    _check(
        isinstance(config.get("anchors"), dict)
        and isinstance(config.get("holdout"), dict)
        and isinstance(config["holdout"].get("entries"), list),
        "manifest retrieval config is invalid",
    )
    _check(not edgelist.duplicated().any(), "behavior duplicate edge")
    _check(
        not (edgelist["txId1"] == edgelist["txId2"]).any(),
        "behavior self-loop edge",
    )
    edge_pairs = set(zip(edgelist["txId1"].astype(int), edgelist["txId2"].astype(int), strict=True))
    step_by_tx = features.set_index("txId")["time_step"].to_dict()
    chain_ids = {get_chain(key)["chain_id"] for key in ("bitcoin", "ethereum", "tron")}

    entity_rows = behavior["entities"]
    entity_ids = [item["entity_id"] for item in entity_rows]
    _check(len(entity_ids) == len(set(entity_ids)), "entity IDs are not unique")
    entity_by_id = {item["entity_id"]: item for item in entity_rows}
    native_entity_ids = []
    seed = int(manifest["seed"])
    for entity in entity_rows:
        profile = get_profile(entity["profile_id"])
        _check(entity["profile_version"] == profile.version, "entity profile version mismatch")
        _check(entity["entity_type"] == profile.entity_type, "entity type mismatch")
        _check(entity["chain_id"] in chain_ids, "entity chain is unknown")
        _check(entity["chain_id"] in enabled_chain_ids, "entity uses a disabled chain")
        _check(isinstance(entity["native_entity_id"], str), "native entity ID is invalid")
        _check(
            entity["native_entity_id"].startswith("synthetic-"),
            "native entity ID is not synthetic",
        )
        expected_native_entity_id = hashlib.sha256(
            f"{seed}:behavior:entity:{entity['entity_id']}:{entity['chain_id']}".encode()
        ).hexdigest()[:32]
        _check(
            entity["native_entity_id"] == f"synthetic-{expected_native_entity_id}",
            "native entity ID derivation mismatch",
        )
        native_entity_ids.append(entity["native_entity_id"])
    _check(len(native_entity_ids) == len(set(native_entity_ids)), "native entity ID is duplicated")
    entity_counts = Counter(item["profile_id"] for item in entity_rows)
    _check(
        all(
            entity_counts[profile_id] == item["entity_count"] for profile_id, item in quotas.items()
        ),
        "manifest entity quota mismatch",
    )

    node_profiles = behavior["behavioral_profiles"]["nodes"]
    profile_by_tx = {item["txId"]: item for item in node_profiles}
    _check(len(profile_by_tx) == len(node_profiles), "behavior profile txId is duplicated")
    _check(set(profile_by_tx) == tx_ids, "behavior profile coverage mismatch")
    profile_counts = Counter(item["profile_id"] for item in node_profiles)
    _check(
        all(profile_counts[profile_id] == item["tx_quota"] for profile_id, item in quotas.items()),
        "manifest tx quota mismatch",
    )
    from .anchors import (
        _articulation_points,
        _central_anchor,
        _eccentricity,
        _full_adjacency,
        _too_close,
    )
    from .graph import GeneratedGraph, TxNode

    projection_nodes = [
        TxNode(
            tx_id,
            int(step_by_tx[tx_id]),
            get_action(node["behavior_params"]["action_id"]).target_class,
            node["behavior_params"]["action_id"],
            node["profile_id"],
        )
        for tx_id, node in profile_by_tx.items()
    ]
    projection_graph = GeneratedGraph(
        nodes=projection_nodes,
        edges=[
            (int(source), int(target))
            for source, target in edgelist[["txId1", "txId2"]].itertuples(index=False, name=None)
        ],
    )
    for tx_id, node in profile_by_tx.items():
        profile = get_profile(node["profile_id"])
        action = get_action(node["behavior_params"]["action_id"])
        _check(action.action_id in profile.action_ids, "profile action is not allowlisted")
        _check(profile.profile_id in action.actor_profiles, "action actor mismatch")
        _check(node["entity_type"] == profile.entity_type, "profile entity type mismatch")
        _check(node["class_policy"] == profile.class_policy, "profile class policy mismatch")
        _check(
            node["behavior_params"].get("amount_policy_id") == action.amount_policy,
            "node amount policy mismatch",
        )
        _check(
            node["behavior_params"].get("profile_amount_policy_id") == profile.amount_policy_id,
            "profile amount policy mismatch",
        )
        _check(node["behavior_params"].get("channel") == action.channel, "node channel mismatch")
        _check(node["chain_id"] in chain_ids, "node chain is unknown")
        _check(set(node["entity_ids"]) <= set(entity_by_id), "node entity is dangling")
        participant_chains = {
            entity_by_id[entity_id]["chain_id"] for entity_id in node["entity_ids"]
        }
        if action.edge_kind != "bridge_link":
            _check(participant_chains == {node["chain_id"]}, "non-bridge node crossed chains")

    events = behavior["events"]
    event_ids = [item["event_id"] for item in events]
    target_ids = [item["target_tx_id"] for item in events]
    _check(event_ids == list(range(len(features))), "event IDs are not contiguous")
    _check(len(set(target_ids)) == len(target_ids), "event target is duplicated")
    _check(set(target_ids) == tx_ids, "event target coverage mismatch")
    event_by_id = {item["event_id"]: item for item in events}
    event_by_target = {item["target_tx_id"]: item for item in events}
    for event in events:
        profile = get_profile(event["profile_id"])
        action = get_action(event["action_id"])
        _check(action.action_id in profile.action_ids, "event action is not allowlisted")
        _check(event["actor_entity_id"] in entity_by_id, "behavior actor is dangling")
        _check(
            entity_by_id[event["actor_entity_id"]]["profile_id"] == event["profile_id"],
            "event actor entity mismatch",
        )
        if action.edge_kind == "bridge_link":
            actor_chain = entity_by_id[event["actor_entity_id"]]["chain_id"]
            target_chain = event["chain_id"]
            _check(actor_chain != target_chain, "bridge event is not cross-chain")
            _check(
                event.get("bridge_id") == f"{actor_chain}->{target_chain}",
                "bridge event ID is missing or invalid",
            )
        else:
            _check(event.get("bridge_id") is None, "non-bridge event has bridge ID")
        _check(profile.profile_id in action.actor_profiles, "event actor profile mismatch")
        _check(
            isinstance(event.get("amount_minor"), int)
            and not isinstance(event.get("amount_minor"), bool)
            and event["amount_minor"] >= 0,
            "event amount is invalid",
        )
        counterparties = event["counterparty_entity_ids"]
        _check(counterparties, "event has no counterparty")
        _check(len(counterparties) == len(set(counterparties)), "event counterparty is duplicated")
        for entity_id in counterparties:
            _check(entity_id in entity_by_id, "behavior counterparty is dangling")
            counterparty_profile = entity_by_id[entity_id]["profile_id"]
            _check(
                counterparty_profile in action.counterparty_profiles,
                "event counterparty violates action allowlist",
            )
        sources = event["source_tx_ids"]
        source_entity_ids = event.get("source_entity_ids", [])
        _check(isinstance(source_entity_ids, list), "event source entities are invalid")
        _check(
            len(source_entity_ids) == len(set(source_entity_ids)),
            "event source entity is duplicated",
        )
        _check(bool(source_entity_ids) == bool(sources), "event source entity coverage mismatch")
        for source_entity_id in source_entity_ids:
            _check(source_entity_id in entity_by_id, "event source entity is dangling")
        _check(len(sources) == len(set(sources)), "event source is duplicated")
        for source in sources:
            _check(source in event_by_target, "behavior source is dangling")
            source_event = event_by_target[source]
            _check(source_event["event_id"] < event["event_id"], "behavior source is not causal")
            _check(step_by_tx[source] <= event["step"], "behavior source is in the future")
            _check(
                source_entity_ids == [source_event["target_entity_id"]],
                "event source entity derivation mismatch",
            )
            _check(
                (int(source), int(event["target_tx_id"])) in edge_pairs,
                "behavior source has no matching edge",
            )
        target = profile_by_tx[event["target_tx_id"]]
        target_entity_id = event["target_entity_id"]
        _check(target_entity_id in entity_by_id, "event target entity is dangling")
        _check(target_entity_id in target["entity_ids"], "event target entity mismatch")
        expected_target_entity = (
            event["actor_entity_id"]
            if action.action_id in {"hub_in", "mix_in"}
            else counterparties[0]
        )
        _check(target_entity_id == expected_target_entity, "event target derivation mismatch")
        _check(event["step"] == step_by_tx[event["target_tx_id"]], "event step mismatch")
        _check(event["chain_id"] == target["chain_id"], "event chain mismatch")
        _check(
            entity_by_id[target_entity_id]["chain_id"] == event["chain_id"],
            "event target entity chain mismatch",
        )
        event_entity_ids = {
            event["actor_entity_id"],
            *event["counterparty_entity_ids"],
            *source_entity_ids,
            target_entity_id,
        }
        event_chains = {entity_by_id[entity_id]["chain_id"] for entity_id in event_entity_ids}
        _check(
            set(target["entity_ids"]) == event_entity_ids,
            "node participant provenance mismatch",
        )
        event_chains.update(profile_by_tx[source]["chain_id"] for source in event["source_tx_ids"])
        for source in sources:
            _check(
                set(profile_by_tx[source]["entity_ids"]) & event_entity_ids,
                "event source entity is not in participants",
            )
        if action.edge_kind != "bridge_link":
            _check(
                event_chains == {event["chain_id"]},
                "non-bridge event crossed chains",
            )
        _check(
            target["behavior_params"]["action_id"] == event["action_id"], "event action mismatch"
        )
        _check(target["profile_id"] == event["profile_id"], "event profile mismatch")

    entity_steps: dict[int, list[int]] = {entity_id: [] for entity_id in entity_by_id}
    for event in events:
        participants = {
            event["actor_entity_id"],
            *event["counterparty_entity_ids"],
            *event.get("source_entity_ids", []),
        }
        for entity_id in participants:
            entity_steps[entity_id].append(event["step"])
    for entity_id, entity in entity_by_id.items():
        steps = entity_steps[entity_id]
        _check(bool(steps), "entity has no event provenance")
        _check(
            entity["first_step"] == min(steps) and entity["last_step"] == max(steps),
            "entity temporal bounds mismatch",
        )

    instances = behavior["instances"]
    instance_ids = [item["instance_id"] for item in instances]
    _check(instance_ids == list(range(len(instances))), "instance IDs are not contiguous")
    instance_by_id = {item["instance_id"]: item for item in instances}
    for instance in instances:
        for field in ("event_ids", "node_ids", "edge_ids", "entity_ids"):
            _check(
                len(instance[field]) == len(set(instance[field])),
                f"instance {field} is duplicated",
            )
        _check(set(instance["event_ids"]) <= set(event_by_id), "instance event is dangling")
        event_rows = [event_by_id[event_id] for event_id in instance["event_ids"]]
        _check(bool(event_rows), "instance has no events")
        _check(
            all(event["instance_id"] == instance["instance_id"] for event in event_rows),
            "instance event mismatch",
        )
        event_profiles = {get_profile(event["profile_id"]).profile_id for event in event_rows}
        _check(
            set(instance.get("profile_ids", (instance["profile_id"],))) == event_profiles,
            "instance profile set mismatch",
        )
        _check(instance["profile_id"] in event_profiles, "instance primary profile mismatch")
        _check(
            {event["target_tx_id"] for event in event_rows} == set(instance["node_ids"]),
            "instance node coverage mismatch",
        )
        expected_steps = [step_by_tx[event["target_tx_id"]] for event in event_rows]
        _check(
            instance["temporal_window"] == [min(expected_steps), max(expected_steps)],
            "instance temporal window mismatch",
        )
        expected_chain_path = list(dict.fromkeys(event["chain_id"] for event in event_rows))
        _check(instance["chain_path"] == expected_chain_path, "instance chain path mismatch")
        _check(
            set(instance["edge_ids"])
            == {
                edge_id
                for edge_id, (source, target) in enumerate(
                    edgelist[["txId1", "txId2"]].itertuples(index=False, name=None)
                )
                if event_by_target[target]["instance_id"] == instance["instance_id"]
                and source in event_by_target[target]["source_tx_ids"]
            },
            "behavior instance edge coverage mismatch",
        )
        _check(set(instance["entity_ids"]) <= set(entity_by_id), "instance entity is dangling")
        _check(
            set(instance["entity_ids"])
            == {
                entity_id
                for event in event_rows
                for entity_id in profile_by_tx[event["target_tx_id"]]["entity_ids"]
            },
            "instance entity coverage mismatch",
        )
    _check(
        {event["instance_id"] for event in events} == set(instance_by_id),
        "instance coverage mismatch",
    )
    anchor_cfg = config.get("anchors")
    _check(isinstance(anchor_cfg, dict), "manifest anchor config is missing")
    per_1000_nodes = anchor_cfg.get("per_1000_nodes")
    min_anchor_distance = anchor_cfg.get("min_anchor_distance")
    _check(
        isinstance(per_1000_nodes, int)
        and not isinstance(per_1000_nodes, bool)
        and isinstance(min_anchor_distance, int)
        and not isinstance(min_anchor_distance, bool)
        and per_1000_nodes >= 0
        and min_anchor_distance >= 0,
        "manifest anchor config is invalid",
    )
    anchor_target = per_1000_nodes * len(features) // 1000
    full_adjacency = _full_adjacency(projection_graph)
    selected_anchors: list[int] = []
    expected_anchor_by_instance: dict[int, dict[str, Any]] = {}
    selection_full = False
    for instance in instances:
        instance_id = instance["instance_id"]
        node_ids = list(instance["node_ids"])
        edge_ids = list(instance["edge_ids"])
        anchor = (
            _central_anchor(node_ids, projection_graph, edge_ids)
            if config["anchors"].get("prefer_central_anchors", True)
            else node_ids[0]
        )
        if selection_full:
            anchored = False
        else:
            anchored = not any(
                _too_close(full_adjacency, anchor, kept, min_anchor_distance)
                for kept in selected_anchors
            )
            if anchored:
                selected_anchors.append(anchor)
            if len(selected_anchors) >= anchor_target:
                selection_full = True
        expected_anchor_by_instance[instance_id] = {
            "anchor_node_id": anchor,
            "anchor_role": profile_by_tx[anchor]["behavior_params"]["action_id"],
            "max_anchor_distance": _eccentricity(projection_graph, node_ids, edge_ids, anchor),
            "articulation_points": _articulation_points(projection_graph, node_ids, edge_ids),
            "anchored": anchored,
        }
    instance_ground_truth = manifest.get("instance_ground_truth", [])
    ground_instance_by_id = {item["instance_id"]: item for item in instance_ground_truth}
    _check(
        set(ground_instance_by_id) == set(instance_by_id)
        and len(ground_instance_by_id) == len(instance_ground_truth),
        "instance ground truth coverage mismatch",
    )
    for instance_id, instance in instance_by_id.items():
        ground_instance = ground_instance_by_id[instance_id]
        _check(
            ground_instance["node_ids"] == list(instance["node_ids"]),
            "instance ground truth node mismatch",
        )
        _check(
            ground_instance["edge_ids"] == list(instance["edge_ids"]),
            "instance ground truth edge mismatch",
        )
        for field in ("profile_id", "profile_ids", "entity_ids", "event_ids", "chain_path"):
            _check(
                ground_instance.get(field) == instance.get(field),
                f"instance ground truth {field} mismatch",
            )
        _check(
            ground_instance.get("temporal_window") == list(instance["temporal_window"]),
            "instance ground truth temporal window mismatch",
        )
        _check(
            ground_instance.get("pattern_type") == instance["profile_id"],
            "instance ground truth pattern mismatch",
        )
        expected_anchor = expected_anchor_by_instance[instance_id]
        _check(
            ground_instance.get("anchor_node_id") == expected_anchor["anchor_node_id"],
            "instance ground truth anchor mismatch",
        )
        _check(
            ground_instance.get("anchor_role") == expected_anchor["anchor_role"],
            "instance ground truth anchor role mismatch",
        )
        _check(
            ground_instance.get("entity_type") == get_profile(instance["profile_id"]).entity_type,
            "instance ground truth entity type mismatch",
        )
        _check(
            ground_instance.get("max_anchor_distance") == expected_anchor["max_anchor_distance"],
            "instance ground truth anchor distance mismatch",
        )
        _check(
            ground_instance.get("articulation_points") == expected_anchor["articulation_points"],
            "instance ground truth articulation points mismatch",
        )
        _check(
            ground_instance.get("anchored") == expected_anchor["anchored"],
            "instance ground truth anchor selection mismatch",
        )
        holdout_entries = config.get("holdout", {}).get("entries", [])
        expected_holdout = False
        expected_train_excluded = False
        for entry in holdout_entries:
            if entry.get("pattern") != instance["profile_id"]:
                continue
            if max(instance["temporal_window"][0], entry["step_min"]) <= min(
                instance["temporal_window"][1], entry["step_max"]
            ):
                expected_holdout = True
                expected_train_excluded = bool(entry.get("train_excluded", True))
                break
        _check(
            ground_instance.get("holdout") == expected_holdout,
            "instance ground truth holdout mismatch",
        )
        _check(
            ground_instance.get("train_excluded") == expected_train_excluded,
            "instance ground truth train exclusion mismatch",
        )
        for field in ("anchored", "holdout", "train_excluded"):
            _check(
                isinstance(ground_instance.get(field), bool),
                f"instance ground truth {field} is invalid",
            )

    ground_truth = manifest.get("ground_truth", [])
    ground_by_tx = {item["txId"]: item for item in ground_truth}
    csv_class_by_tx = {
        tx_id: str(value) for tx_id, value in zip(classes["txId"], classes["class"], strict=True)
    }
    _check(
        set(ground_by_tx) == tx_ids and len(ground_by_tx) == len(ground_truth),
        "ground truth coverage mismatch",
    )
    for tx_id, node in profile_by_tx.items():
        ground = ground_by_tx[tx_id]
        event = event_by_target[tx_id]
        action = get_action(node["behavior_params"]["action_id"])
        expected_class = action.target_class
        _check(ground["event_id"] == event["event_id"], "ground truth event mismatch")
        _check(ground["scheme"] == node["profile_id"], "ground truth profile mismatch")
        _check(ground["role"] == action.action_id, "ground truth role mismatch")
        _check(ground["class"] == expected_class, "ground truth class mismatch")
        _check(ground["instance_id"] == event["instance_id"], "ground truth instance mismatch")
        _check(ground["entity_ids"] == node["entity_ids"], "ground truth entity mismatch")
        _check(ground["chain_id"] == node["chain_id"], "ground truth chain mismatch")
        expected_csv_class = {"illicit": "1", "licit": "2"}.get(expected_class, "unknown")
        _check(
            csv_class_by_tx[tx_id] == expected_csv_class,
            "ground truth CSV class mismatch",
        )

    edge_ground_truth = manifest.get("edge_attribute_ground_truth")
    _check(isinstance(edge_ground_truth, list), "edge ground truth is missing")
    _check(len(edge_ground_truth) == len(edgelist), "edge ground truth coverage mismatch")
    edge_ground_by_id = {item["edge_id"]: item for item in edge_ground_truth}
    _check(
        set(edge_ground_by_id) == set(range(len(edgelist)))
        and len(edge_ground_by_id) == len(edge_ground_truth),
        "edge ground truth IDs are invalid",
    )
    for edge_id, edge_ground in edge_ground_by_id.items():
        edge = edgelist.iloc[edge_id]
        event = event_by_target[int(edge["txId2"])]
        action = get_action(event["action_id"])
        _check(edge_ground["txId1"] == edge["txId1"], "edge ground truth source mismatch")
        _check(edge_ground["txId2"] == edge["txId2"], "edge ground truth target mismatch")
        _check(edge_ground["event_id"] == event["event_id"], "edge ground truth event mismatch")
        _check(
            edge_ground["instance_id"] == event["instance_id"],
            "edge ground truth instance mismatch",
        )
        _check(edge_ground["role"] == action.action_id, "edge ground truth role mismatch")
        _check(
            edge_ground["pattern_type"]
            == ground_instance_by_id[event["instance_id"]]["pattern_type"],
            "edge ground truth pattern mismatch",
        )
        _check(
            np.isclose(float(edge_ground["amount"]), float(attrs.iloc[edge_id]["amount"])),
            "edge ground truth amount mismatch",
        )
        if behavior["chain_metadata"]["edge_records"]:
            _check(
                edge_ground.get("chain_id") == event["chain_id"],
                "edge ground truth chain mismatch",
            )
            _check(
                edge_ground.get("edge_kind") == action.edge_kind,
                "edge ground truth kind mismatch",
            )

    chain = behavior["chain_metadata"]
    expected_chains = [
        {"chain_key": key, **get_chain(key)} for key in ("bitcoin", "ethereum", "tron")
    ]
    _check(chain.get("chains") == expected_chains, "chain registry mismatch")
    _check(chain.get("identity_scope") == "synthetic", "chain identity scope mismatch")
    node_records = chain["node_records"]
    node_record_by_tx = {item["txId"]: item for item in node_records}
    _check(len(node_record_by_tx) == len(node_records), "chain node txId is duplicated")
    _check(set(node_record_by_tx) == tx_ids, "chain node coverage mismatch")
    native_tx_ids = []
    for tx_id, record in node_record_by_tx.items():
        node = profile_by_tx[tx_id]
        _check(record["chain_id"] == node["chain_id"], "chain node record mismatch")
        _check(record["native_id_is_synthetic"] is True, "chain native ID is not synthetic")
        seed = int(manifest["seed"])
        expected_tx_id = hashlib.sha256(
            f"{seed}:behavior:tx:{tx_id}:{record['chain_id']}".encode()
        ).hexdigest()
        if record["chain_id"] == "eip155:1":
            expected_tx_id = f"0x{expected_tx_id}"
        native_entity = next(
            entity_by_id[entity_id]
            for entity_id in node["entity_ids"]
            if entity_by_id[entity_id]["chain_id"] == record["chain_id"]
        )
        expected_entity_id = hashlib.sha256(
            f"{seed}:behavior:entity:{native_entity['entity_id']}:{record['chain_id']}".encode()
        ).hexdigest()[:32]
        _check(record["native_tx_id"] == expected_tx_id, "chain native tx ID derivation mismatch")
        _check(
            record["native_entity_id"] == f"synthetic-{expected_entity_id}",
            "chain native entity ID derivation mismatch",
        )
        native_tx_id = record["native_tx_id"]
        _check(isinstance(native_tx_id, str), "chain native tx ID is invalid")
        hex_id = native_tx_id[2:] if native_tx_id.startswith("0x") else native_tx_id
        _check(
            len(hex_id) == 64 and all(value in string.hexdigits for value in hex_id),
            "chain native tx ID encoding is invalid",
        )
        native_tx_ids.append(native_tx_id)
    _check(len(native_tx_ids) == len(set(native_tx_ids)), "native tx ID is duplicated")
    edge_records = chain["edge_records"]
    edge_record_by_id = {item["edge_id"]: item for item in edge_records}
    _check(
        len(edge_records) == len(edgelist) and set(edge_record_by_id) == set(range(len(edgelist))),
        "chain edge coverage mismatch",
    )
    for edge_id, record in edge_record_by_id.items():
        expected_edge = edgelist.iloc[edge_id]
        _check(record["txId1"] == expected_edge["txId1"], "chain edge source mismatch")
        _check(record["txId2"] == expected_edge["txId2"], "chain edge target mismatch")
        event = event_by_id[record["event_id"]]
        action = get_action(event["action_id"])
        _check(record["kind"] == action.edge_kind, "chain edge kind mismatch")
        _check(record["role"] == action.action_id, "chain edge role mismatch")
        _check(record["timestamp"] == event["step"], "chain edge timestamp metadata mismatch")
        expected_timestamp = config["edge_attributes"]["timestamp_scale"] * (
            max(step_by_tx[record["txId1"]], step_by_tx[record["txId2"]]) - 1
        )
        _check(
            attrs.iloc[edge_id]["timestamp"] == expected_timestamp,
            "edge attribute timestamp mismatch",
        )
        _check(
            record["source_chain_id"] == node_record_by_tx[record["txId1"]]["chain_id"],
            "chain edge source chain mismatch",
        )
        _check(
            record["target_chain_id"] == node_record_by_tx[record["txId2"]]["chain_id"],
            "chain edge target chain mismatch",
        )
        _check(record["native_amount_minor"] is None, "native amount must remain metadata-only")
        if record["bridge_id"] is not None:
            _check(record["kind"] == "bridge_link", "bridge ID on non-bridge edge")
            _check(
                record["source_chain_id"] != record["target_chain_id"],
                "bridge edge is not cross-chain",
            )
        elif record["source_chain_id"] != record["target_chain_id"]:
            _check(False, "cross-chain edge has no bridge ID")
        _check(
            np.isclose(record["csv_amount"], event["amount_minor"] / 100.0),
            "event amount mismatch",
        )
        _check(
            np.isclose(record["csv_amount"], attrs.iloc[edge_id]["amount"]),
            "edge amount mismatch",
        )
        _check(event["target_tx_id"] == record["txId2"], "chain edge event mismatch")
        _check(record["txId1"] in event["source_tx_ids"], "chain edge source event mismatch")
        _check(record["bridge_id"] == event.get("bridge_id"), "edge/event bridge mismatch")
    edge_by_event_id = {
        record["event_id"]: record["edge_id"] for record in edge_record_by_id.values()
    }
    expected_bridges = []
    for event in events:
        if event.get("bridge_id") is None:
            continue
        source_chain = entity_by_id[event["actor_entity_id"]]["chain_id"]
        target_chain = event["chain_id"]
        _check(source_chain != target_chain, "bridge event is not cross-chain")
        expected_bridges.append(
            {
                "bridge_id": event["bridge_id"],
                "event_id": event["event_id"],
                "edge_id": edge_by_event_id.get(event["event_id"]),
                "source_chain_id": source_chain,
                "target_chain_id": target_chain,
            }
        )
    actual_bridges = sorted(chain["bridge_records"], key=lambda record: record["event_id"])
    _check(actual_bridges == expected_bridges, "bridge record coverage mismatch")
    _check(
        len(actual_bridges) == len({record["event_id"] for record in actual_bridges}),
        "bridge record is duplicated",
    )
    for instance in instances:
        expected_edges = {
            edge_id
            for edge_id, record in edge_record_by_id.items()
            if event_by_id[record["event_id"]]["instance_id"] == instance["instance_id"]
        }
        _check(set(instance["edge_ids"]) == expected_edges, "instance edge coverage mismatch")
    drift = behavior["drift_schedule"]
    behavior_config = manifest.get("config", {}).get("behavior")
    _check(isinstance(behavior_config, dict), "behavior config provenance is missing")
    configured_drift = behavior_config.get("drift")
    _check(isinstance(configured_drift, dict), "behavior drift config is missing")
    _check(drift.get("configured") == configured_drift, "behavior drift config mismatch")
    realized_phases = drift.get("realized", [])
    _check(
        [item.get("phase_id") for item in realized_phases]
        == [item["phase_id"] for item in configured_drift["phases"]],
        "behavior drift phase coverage mismatch",
    )
    for phase in configured_drift["phases"]:
        realized = next(item for item in realized_phases if item["phase_id"] == phase["phase_id"])
        _check(
            (realized["step_min"], realized["step_max"]) == (phase["step_min"], phase["step_max"]),
            "behavior drift phase bounds mismatch",
        )
        phase_nodes = [
            node
            for node in profile_by_tx.values()
            if phase["step_min"] <= step_by_tx[node["txId"]] <= phase["step_max"]
        ]
        _check(realized["node_count"] == len(phase_nodes), "behavior drift count mismatch")
        expected_class_counts = {
            value: sum(
                get_action(node["behavior_params"]["action_id"]).target_class == value
                for node in phase_nodes
            )
            for value in ("illicit", "licit")
        }
        _check(
            realized.get("class_counts", {}).get("illicit") == expected_class_counts["illicit"]
            and realized.get("class_counts", {}).get("licit") == expected_class_counts["licit"],
            "behavior drift class counts mismatch",
        )
        expected_profile_counts = {
            profile_id: sum(node["profile_id"] == profile_id for node in phase_nodes)
            for profile_id in sorted(behavior_config["profile_quotas"])
        }
        _check(
            realized.get("profile_counts") == expected_profile_counts,
            "behavior drift profile counts mismatch",
        )
        expected_chain_counts = {
            chain_id: sum(node["chain_id"] == chain_id for node in phase_nodes)
            for chain_id in sorted(chain_ids)
        }
        _check(
            realized.get("chain_counts") == expected_chain_counts,
            "behavior drift chain counts mismatch",
        )
        if phase["illicit_keep_probability"] == 0:
            _check(
                all(
                    get_profile(node["profile_id"]).class_policy != "illicit"
                    or get_profile(node["profile_id"]).negative_kind == "novel_illicit"
                    for node in phase_nodes
                ),
                "behavior hard suppression mismatch",
            )
        _check(
            all(
                get_profile(node["profile_id"]).negative_kind != "novel_illicit"
                or node["profile_id"] in phase["novel_profile_ids"]
                for node in phase_nodes
            ),
            "behavior novel phase mismatch",
        )
    sanctions = behavior["sanctions_anchor_map"]
    _check(sanctions.get("status") == "provenance_only", "sanctions status mismatch")
    _check(sanctions.get("real_address_payload_loaded") is False, "real sanctions payload loaded")
    _check(sanctions.get("records") == [], "sanctions records are not empty")
    calibration = behavior["calibration"]
    _check(calibration.get("status") == "not_evaluated", "calibration status mismatch")
    _check(calibration.get("unknown_policy") == "exclude", "calibration unknown policy mismatch")
    _check(isinstance(chain.get("bridge_records"), list), "bridge records schema mismatch")
    _check(
        sum(item["node_count"] for item in behavior["drift_schedule"]["realized"]) == len(features),
        "behavior drift realization does not cover nodes",
    )
    return manifest
