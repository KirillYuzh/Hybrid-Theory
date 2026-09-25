import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from .config import GeneratorConfig
from .drift import (
    BehaviorDriftError,
    allocate_step_slots,
    phase_at,
    profile_is_eligible,
    validate_drift_schedule,
)
from .graph import GeneratedGraph, TxNode
from .stats import VolumeProfile


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    actor_profiles: tuple[str, ...]
    counterparty_profiles: tuple[str, ...]
    target_class: str
    edge_kind: str
    amount_policy: str
    channel: str


@dataclass(frozen=True)
class BehaviorProfile:
    profile_id: str
    version: int
    entity_type: str
    class_policy: str
    negative_kind: str
    action_ids: tuple[str, ...]
    counterparty_profile_ids: tuple[str, ...]
    chain_weights: tuple[tuple[str, float], ...]
    activity_weight: float
    amount_policy_id: str


@dataclass(frozen=True)
class EntityRecord:
    entity_id: int
    profile_id: str
    profile_version: int
    entity_type: str
    chain_id: str
    native_entity_id: str
    first_step: int | None = None
    last_step: int | None = None


@dataclass
class EntityState:
    entity_id: int
    profile_id: str
    chain_id: str
    last_tx_id: int | None = None


@dataclass(frozen=True)
class BehaviorEvent:
    event_id: int
    instance_id: int
    step: int
    actor_entity_id: int
    counterparty_entity_ids: tuple[int, ...]
    action_id: str
    profile_id: str
    source_tx_ids: tuple[int, ...]
    target_tx_id: int
    target_entity_id: int
    amount_minor: int
    chain_id: str
    bridge_id: str | None = None
    source_entity_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BehaviorNode:
    tx_id: int
    event_id: int
    entity_ids: tuple[int, ...]
    chain_id: str


@dataclass(frozen=True)
class BehaviorEdge:
    edge_id: int
    txId1: int
    txId2: int
    event_id: int
    chain_id: str
    kind: str
    role: str
    amount_minor: int
    timestamp: int
    bridge_id: str | None = None


@dataclass(frozen=True)
class BehaviorInstance:
    instance_id: int
    profile_id: str
    entity_ids: tuple[int, ...]
    event_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    edge_ids: tuple[int, ...]
    temporal_window: tuple[int, int]
    chain_path: tuple[str, ...]
    profile_ids: tuple[str, ...] = ()


@dataclass
class BehaviorGraph:
    graph: GeneratedGraph
    node_meta: list[BehaviorNode] = field(default_factory=list)
    edge_meta: list[BehaviorEdge] = field(default_factory=list)
    entities: list[EntityRecord] = field(default_factory=list)
    events: list[BehaviorEvent] = field(default_factory=list)
    instances: list[BehaviorInstance] = field(default_factory=list)

    @property
    def nodes(self) -> list[TxNode]:
        return self.graph.nodes

    @property
    def edges(self) -> list[tuple[int, int]]:
        return self.graph.edges


_CHAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "bitcoin": {
        "chain_id": "bip122:000000000019d6689c085ae165831e93",
        "family": "utxo",
        "native_asset_id": "bitcoin:native:BTC",
        "decimals": 8,
    },
    "ethereum": {
        "chain_id": "eip155:1",
        "family": "account",
        "native_asset_id": "ethereum:native:ETH",
        "decimals": 18,
    },
    "tron": {
        "chain_id": "eip155:728126428",
        "family": "account",
        "native_asset_id": "tron:native:TRX",
        "decimals": 6,
    },
}

_PROFILE_REGISTRY: dict[str, BehaviorProfile] = {
    "ordinary_wallet": BehaviorProfile(
        "ordinary_wallet",
        1,
        "individual",
        "licit",
        "ordinary",
        ("transfer", "bridge_link"),
        ("ordinary_wallet", "individual", "exchange_hub", "wallet_provider"),
        (("bitcoin", 0.4), ("ethereum", 0.3), ("tron", 0.3)),
        1.0,
        "medium_uniform",
    ),
    "exchange_hub": BehaviorProfile(
        "exchange_hub",
        1,
        "exchange",
        "licit",
        "licit_negative",
        ("hub_in", "hub_out", "bridge_link"),
        ("ordinary_wallet", "individual", "exchange_hub", "wallet_provider"),
        (("bitcoin", 0.34), ("ethereum", 0.33), ("tron", 0.33)),
        0.8,
        "exchange_uniform",
    ),
    "miner_payout": BehaviorProfile(
        "miner_payout",
        1,
        "miner",
        "licit",
        "licit_negative",
        ("payout", "bridge_link"),
        ("miner_payout", "individual", "exchange_hub", "ordinary_wallet"),
        (("bitcoin", 1.0),),
        0.7,
        "payout_regular",
    ),
    "wallet_provider": BehaviorProfile(
        "wallet_provider",
        1,
        "wallet_provider",
        "licit",
        "licit_negative",
        ("micropayment", "bridge_link"),
        ("wallet_provider", "individual", "ordinary_wallet", "exchange_hub"),
        (("bitcoin", 0.34), ("ethereum", 0.33), ("tron", 0.33)),
        0.9,
        "micro_low_variance",
    ),
    "individual": BehaviorProfile(
        "individual",
        1,
        "individual",
        "licit",
        "ordinary",
        ("transfer", "bridge_link"),
        ("ordinary_wallet", "individual", "exchange_hub", "wallet_provider"),
        (("bitcoin", 0.34), ("ethereum", 0.33), ("tron", 0.33)),
        0.35,
        "medium_uniform",
    ),
    "mixer": BehaviorProfile(
        "mixer",
        1,
        "illicit_actor",
        "illicit",
        "illicit",
        ("mix_in", "mix_out", "bridge_link"),
        ("ordinary_wallet", "individual", "exchange_hub"),
        (("bitcoin", 0.5), ("ethereum", 0.25), ("tron", 0.25)),
        0.4,
        "mixer_fee",
    ),
    "wash_round_trip": BehaviorProfile(
        "wash_round_trip",
        1,
        "illicit_actor",
        "illicit",
        "novel_illicit",
        ("round_trip", "bridge_link"),
        ("ordinary_wallet", "individual", "exchange_hub"),
        (("bitcoin", 0.5), ("ethereum", 0.25), ("tron", 0.25)),
        0.2,
        "round_trip_uniform",
    ),
}

_ACTION_REGISTRY: dict[str, ActionDefinition] = {
    "transfer": ActionDefinition(
        "transfer",
        ("ordinary_wallet", "individual"),
        ("ordinary_wallet", "individual"),
        "licit",
        "transfer",
        "medium_uniform",
        "on_chain",
    ),
    "hub_in": ActionDefinition(
        "hub_in",
        ("exchange_hub",),
        ("ordinary_wallet", "individual"),
        "licit",
        "transfer",
        "exchange_uniform",
        "on_chain",
    ),
    "hub_out": ActionDefinition(
        "hub_out",
        ("exchange_hub",),
        ("wallet_provider", "ordinary_wallet"),
        "licit",
        "transfer",
        "exchange_uniform",
        "on_chain",
    ),
    "payout": ActionDefinition(
        "payout",
        ("miner_payout",),
        ("individual", "exchange_hub"),
        "licit",
        "transfer",
        "payout_regular",
        "on_chain",
    ),
    "micropayment": ActionDefinition(
        "micropayment",
        ("wallet_provider",),
        ("individual", "wallet_provider"),
        "licit",
        "transfer",
        "micro_low_variance",
        "on_chain",
    ),
    "mix_in": ActionDefinition(
        "mix_in",
        ("mixer",),
        ("ordinary_wallet", "individual"),
        "illicit",
        "transfer",
        "mixer_fee",
        "on_chain",
    ),
    "mix_out": ActionDefinition(
        "mix_out",
        ("mixer",),
        ("exchange_hub", "ordinary_wallet"),
        "illicit",
        "transfer",
        "mixer_fee",
        "on_chain",
    ),
    "round_trip": ActionDefinition(
        "round_trip",
        ("wash_round_trip",),
        ("ordinary_wallet", "individual"),
        "illicit",
        "round_trip",
        "round_trip_uniform",
        "on_chain",
    ),
    "bridge_link": ActionDefinition(
        "bridge_link",
        (
            "ordinary_wallet",
            "exchange_hub",
            "miner_payout",
            "wallet_provider",
            "individual",
            "mixer",
            "wash_round_trip",
        ),
        ("ordinary_wallet", "exchange_hub"),
        "licit",
        "bridge_link",
        "normalized_uniform",
        "bridge",
    ),
}


def get_profile(profile_id: str) -> BehaviorProfile:
    try:
        return _PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown behavior profile: {profile_id}") from exc


def get_action(action_id: str) -> ActionDefinition:
    try:
        return _ACTION_REGISTRY[action_id]
    except KeyError as exc:
        raise ValueError(f"unknown behavior action: {action_id}") from exc


def get_chain(chain_key: str) -> dict[str, Any]:
    try:
        return dict(_CHAIN_REGISTRY[chain_key])
    except KeyError as exc:
        raise ValueError(f"unknown behavior chain: {chain_key}") from exc


def validate_registry() -> None:
    if set(_CHAIN_REGISTRY) != {"bitcoin", "ethereum", "tron"}:
        raise ValueError("behavior chain registry is incomplete")
    for profile in _PROFILE_REGISTRY.values():
        for action_id in profile.action_ids:
            action = get_action(action_id)
            if profile.profile_id not in action.actor_profiles:
                raise ValueError(f"profile is not allowed for action: {profile.profile_id}")
            if not set(action.counterparty_profiles) <= set(profile.counterparty_profile_ids):
                raise ValueError(f"action counterparty is outside profile policy: {action_id}")
        for profile_id in profile.counterparty_profile_ids:
            if profile_id not in _PROFILE_REGISTRY:
                raise ValueError(f"profile references unknown counterparty: {profile_id}")
        for chain_key, weight in profile.chain_weights:
            if chain_key not in _CHAIN_REGISTRY:
                raise ValueError(f"profile references unknown chain: {chain_key}")
            if weight <= 0:
                raise ValueError(f"profile has non-positive chain weight: {chain_key}")
    for action in _ACTION_REGISTRY.values():
        for profile_id in action.actor_profiles + action.counterparty_profiles:
            if profile_id not in _PROFILE_REGISTRY:
                raise ValueError(f"action references unknown profile: {profile_id}")
    if len(_ACTION_REGISTRY) != 9:
        raise ValueError("behavior action registry is incomplete")


def registry_sha256() -> str:
    payload = {
        "contract_version": "behavior.p0.v1",
        "profiles": [asdict(profile) for _, profile in sorted(_PROFILE_REGISTRY.items())],
        "actions": [asdict(action) for _, action in sorted(_ACTION_REGISTRY.items())],
        "chains": {key: get_chain(key) for key in sorted(_CHAIN_REGISTRY)},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def derive_rng(seed: int, namespace: str) -> np.random.Generator:
    """Derive an independent random generator for a seed namespace.

    Parameters
    ----------
    seed : int
        Base seed shared by all behavior RNG namespaces.
    namespace : str
        Stable namespace that separates this generator from other RNG streams.

    Returns
    -------
    numpy.random.Generator
        Generator seeded from the first eight SHA-256 digest bytes interpreted as
        a little-endian unsigned integer.
    """
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _validate_behavior_config(config: GeneratorConfig) -> None:
    """Validate the registry, drift, quota, and chain constraints.

    Returns
    -------
    None
        Indicates that the configuration satisfies the behavior contract.
    """
    if config.graph.mode != "behavior":
        raise ValueError("behavior simulator requires graph.mode=behavior")
    validate_registry()
    validate_drift_schedule(
        config.behavior.drift, known_profile_ids=set(config.behavior.profile_quotas)
    )
    quotas = config.behavior.profile_quotas
    if sum(item.tx_quota for item in quotas.values()) != config.n_txs:
        raise BehaviorDriftError("behavior profile quotas must sum to n_txs")
    for profile_id, quota in quotas.items():
        profile = get_profile(profile_id)
        if quota.entity_count < 0 or quota.tx_quota < 0:
            raise ValueError(f"negative behavior quota: {profile_id}")
        if (quota.entity_count == 0) != (quota.tx_quota == 0):
            raise ValueError(f"entity_count and tx_quota must be paired: {profile_id}")
        if profile.class_policy not in {"licit", "illicit"}:
            raise ValueError(f"invalid behavior class policy: {profile_id}")
    for phase in config.behavior.drift["phases"]:
        for profile_id in phase["novel_profile_ids"]:
            if get_profile(profile_id).negative_kind != "novel_illicit":
                raise ValueError(f"non-novel profile marked novel: {profile_id}")
    for chain_key, settings in config.behavior.chains.items():
        get_chain(chain_key)
        if set(settings) != {"enabled"}:
            raise ValueError(f"invalid behavior chain settings: {chain_key}")
    enabled_chains = {
        key for key, settings in config.behavior.chains.items() if settings["enabled"]
    }
    if not enabled_chains:
        raise ValueError("at least one behavior chain must be enabled")
    if len(enabled_chains) < 2:
        raise ValueError("bridge_link requires at least two enabled behavior chains")


def _native_entity_id(seed: int, entity_id: int, chain_id: str) -> str:
    payload = f"{seed}:behavior:entity:{entity_id}:{chain_id}"
    return "synthetic-" + hashlib.sha256(payload.encode()).hexdigest()[:32]


def _choose_chain(profile: BehaviorProfile, enabled: set[str], rng: np.random.Generator) -> str:
    weights = [
        (key, weight) for key, weight in profile.chain_weights if key in enabled and weight > 0
    ]
    if not weights:
        raise ValueError(f"profile has no enabled chain: {profile.profile_id}")
    total = sum(weight for _, weight in weights)
    threshold = float(rng.random()) * total
    cumulative = 0.0
    for chain_key, weight in weights:
        cumulative += weight
        if threshold <= cumulative:
            return chain_key
    return weights[-1][0]


def _build_entities(
    config: GeneratorConfig, rng: np.random.Generator
) -> tuple[list[EntityState], dict[str, list[EntityState]]]:
    enabled = {key for key, value in config.behavior.chains.items() if value["enabled"]}
    states: list[EntityState] = []
    by_profile: dict[str, list[EntityState]] = {}
    next_id = 0
    for profile_id in sorted(config.behavior.profile_quotas):
        quota = config.behavior.profile_quotas[profile_id]
        profile = get_profile(profile_id)
        by_profile[profile_id] = []
        for _ in range(quota.entity_count):
            chain_id = get_chain(_choose_chain(profile, enabled, rng))["chain_id"]
            state = EntityState(next_id, profile_id, chain_id)
            states.append(state)
            by_profile[profile_id].append(state)
            next_id += 1
    return states, by_profile


def _eligible(
    profile: BehaviorProfile,
    step: int,
    schedule: dict[str, Any],
    rng: np.random.Generator,
) -> bool:
    if not profile_is_eligible(profile, step, schedule):
        return False
    if profile.class_policy == "illicit" and profile.negative_kind != "novel_illicit":
        probability = phase_at(schedule, step)["illicit_keep_probability"]
        if probability < 1 and float(rng.random()) >= probability:
            return False
    return True


def _assign_profiles(
    config: GeneratorConfig,
    slots: list[int],
    schedule: dict[str, Any],
    schedule_rng: np.random.Generator,
    drift_rng: np.random.Generator,
) -> list[str]:
    """Assign each profile quota to eligible step slots without replacement.

    Parameters
    ----------
    config : GeneratorConfig
        Validated behavior configuration containing the exact profile quotas.
    slots : list[int]
        One-based time step for each transaction slot.
    schedule : dict[str, Any]
        Validated drift schedule used to determine phase eligibility.
    schedule_rng : numpy.random.Generator
        Generator used to select eligible slot indices.
    drift_rng : numpy.random.Generator
        Generator used for phase-dependent illicit-profile eligibility.

    Returns
    -------
    list[str]
        Profile IDs in slot order, with one assignment for every transaction.

    Raises
    ------
    BehaviorDriftError
        If a profile has too few eligible slots or any slot remains unassigned.
    """
    assignments: list[str | None] = [None] * len(slots)
    available = set(range(len(slots)))
    profile_ids = sorted(
        config.behavior.profile_quotas,
        key=lambda value: (get_profile(value).negative_kind != "novel_illicit", value),
    )
    for profile_id in profile_ids:
        quota = config.behavior.profile_quotas[profile_id].tx_quota
        if quota == 0:
            continue
        profile = get_profile(profile_id)
        eligible = [
            index
            for index in sorted(available)
            if _eligible(profile, slots[index], schedule, drift_rng)
        ]
        if len(eligible) < quota:
            raise BehaviorDriftError(f"not enough eligible steps for profile: {profile_id}")
        selected = schedule_rng.choice(np.asarray(eligible), size=quota, replace=False)
        for index in selected:
            index = int(index)
            assignments[index] = profile_id
            available.remove(index)
    if available:
        raise BehaviorDriftError("behavior scheduler left unassigned slots")
    return [str(profile_id) for profile_id in assignments if profile_id is not None]


def _sample_amount(policy: str, rng: np.random.Generator) -> int:
    ranges = {
        "medium_uniform": (100, 10_000),
        "exchange_uniform": (10, 3_000),
        "payout_regular": (500, 50_000),
        "micro_low_variance": (1, 200),
        "mixer_fee": (10_000, 100_000),
        "round_trip_uniform": (1_000, 20_000),
        "normalized_uniform": (1, 1_000),
    }
    if policy not in ranges:
        raise ValueError(f"unknown behavior amount policy: {policy}")
    low, high = ranges[policy]
    return int(rng.integers(low, high + 1))


def _choose_counterparty(
    action: ActionDefinition,
    actor: EntityState,
    by_profile: dict[str, list[EntityState]],
    rng: np.random.Generator,
) -> EntityState:
    candidates = [
        state
        for profile_id in action.counterparty_profiles
        for state in by_profile.get(profile_id, [])
        if state.entity_id != actor.entity_id
    ]
    if not candidates:
        raise ValueError(f"action has no counterparty: {action.action_id}")
    same_chain = [state for state in candidates if state.chain_id == actor.chain_id]
    cross_chain = [state for state in candidates if state.chain_id != actor.chain_id]
    if action.edge_kind == "bridge_link":
        if not cross_chain:
            raise ValueError("bridge action has no cross-chain counterparty")
        pool = cross_chain
    else:
        pool = same_chain or candidates
    weights = np.asarray(
        [get_profile(state.profile_id).activity_weight for state in pool], dtype=float
    )
    weights /= weights.sum()
    return pool[int(rng.choice(len(pool), p=weights))]


def _source_and_target(
    action: ActionDefinition,
    actor: EntityState,
    counterparty: EntityState,
    states: list[EntityState],
) -> tuple[EntityState | None, EntityState]:
    """Choose a causal source and target for an action.

    The action direction determines the target. A previous transaction is reused
    as the source when its chain is valid for the action; otherwise another
    eligible counterparty is searched and a bootstrap event may remain.

    Parameters
    ----------
    action : ActionDefinition
        Action that defines target direction and bridge semantics.
    actor : EntityState
        Entity selected as the event actor.
    counterparty : EntityState
        Entity selected from the action's counterparty policy.
    states : list[EntityState]
        Population available for a fallback causal source.

    Returns
    -------
    tuple[EntityState | None, EntityState]
        Optional source entity and the required target entity.
    """
    action_id = action.action_id
    if action_id in {"hub_in", "mix_in"}:
        preferred = counterparty
        target = actor
    else:
        preferred = actor
        target = counterparty
    if action.edge_kind == "bridge_link":
        if preferred.chain_id != target.chain_id and preferred.last_tx_id is not None:
            return preferred, target
    else:
        if preferred.chain_id != target.chain_id:
            preferred = None
        if preferred is not None and preferred.last_tx_id is not None:
            return preferred, target
    for candidate in states:
        candidate_chain_matches = (
            candidate.chain_id != target.chain_id
            if action.edge_kind == "bridge_link"
            else candidate.chain_id == target.chain_id
        )
        if (
            candidate.entity_id not in {actor.entity_id, target.entity_id}
            and candidate.profile_id in action.counterparty_profiles
            and candidate_chain_matches
            and candidate.last_tx_id is not None
        ):
            return candidate, target
    return None, target


def build_behavior_graph(config: GeneratorConfig, stats: VolumeProfile) -> BehaviorGraph:
    """Build a causal behavior graph and aggregate events into instances.

    Population, scheduling, transitions, amounts, and drift use independent RNG
    namespaces. Source-connected events are grouped by union-find so bootstrap
    events remain valid single-event components.

    Parameters
    ----------
    config : GeneratorConfig
        Validated generator configuration and canonical seed.
    stats : VolumeProfile
        Non-negative volume profile containing exactly 49 time steps.

    Returns
    -------
    BehaviorGraph
        Generated graph with node, edge, entity, event, and instance metadata.

    Raises
    ------
    ValueError
        If the configuration, volume shape, or edge metadata violates the contract.
    BehaviorDriftError
        If scheduling or causal constraints cannot produce exactly ``n_txs`` events.
    """
    _validate_behavior_config(config)
    if len(stats.volume) != 49:
        raise ValueError("behavior simulation requires a 49-step volume profile")
    population_rng = derive_rng(config.seed, "behavior:population")
    schedule_rng = derive_rng(config.seed, "behavior:schedule")
    transition_rng = derive_rng(config.seed, "behavior:transition")
    amount_rng = derive_rng(config.seed, "behavior:amount")
    drift_rng = derive_rng(config.seed, "behavior:drift")
    entities, by_profile = _build_entities(config, population_rng)
    slots = allocate_step_slots(stats.volume, config.n_txs)
    schedule = validate_drift_schedule(config.behavior.drift)
    assigned = _assign_profiles(config, slots, schedule, schedule_rng, drift_rng)
    nodes: list[TxNode] = []
    node_meta: list[BehaviorNode] = []
    edges: list[tuple[int, int]] = []
    edge_meta: list[BehaviorEdge] = []
    events: list[BehaviorEvent] = []
    tx_events: dict[int, int] = {}
    event_parent = list(range(config.n_txs))
    event_rank = [0] * config.n_txs

    def find_event(event_id: int) -> int:
        root = event_id
        while event_parent[root] != root:
            root = event_parent[root]
        while event_parent[event_id] != event_id:
            next_id = event_parent[event_id]
            event_parent[event_id] = root
            event_id = next_id
        return root

    def union_events(left_id: int, right_id: int) -> None:
        left_root = find_event(left_id)
        right_root = find_event(right_id)
        if left_root == right_root:
            return
        if event_rank[left_root] < event_rank[right_root]:
            left_root, right_root = right_root, left_root
        event_parent[right_root] = left_root
        if event_rank[left_root] == event_rank[right_root]:
            event_rank[left_root] += 1

    entity_steps: dict[int, list[int | None]] = {
        entity.entity_id: [None, None] for entity in entities
    }

    def mark(entity_id: int, step: int) -> None:
        current = entity_steps[entity_id]
        current[0] = step if current[0] is None else min(current[0], step)
        current[1] = step if current[1] is None else max(current[1], step)

    for step, profile_id in zip(slots, assigned):
        profile = get_profile(profile_id)
        action_id = profile.action_ids[int(transition_rng.integers(len(profile.action_ids)))]
        action = get_action(action_id)
        actor = by_profile[profile_id][int(transition_rng.integers(len(by_profile[profile_id])))]
        counterparty = _choose_counterparty(action, actor, by_profile, transition_rng)
        # Cross-chain non-bridge actions are replaced by the allowlisted bridge action.
        if action.edge_kind != "bridge_link" and counterparty.chain_id != actor.chain_id:
            action = get_action("bridge_link")
            action_id = action.action_id
            counterparty = _choose_counterparty(action, actor, by_profile, transition_rng)
        source, target = _source_and_target(action, actor, counterparty, entities)
        participant_chains = {actor.chain_id, counterparty.chain_id, target.chain_id}
        if source is not None:
            participant_chains.add(source.chain_id)
        if action.edge_kind != "bridge_link" and len(participant_chains) != 1:
            raise BehaviorDriftError("non-bridge action crossed chain boundary")
        bridge_id = (
            f"{actor.chain_id}->{target.chain_id}"
            if action.edge_kind == "bridge_link" and len(participant_chains) > 1
            else None
        )
        if source is not None and source.last_tx_id is None:
            source = None
        source_tx = None if source is None else source.last_tx_id
        if source_tx is not None and source_tx not in tx_events:
            raise BehaviorDriftError("behavior source transaction does not exist")
        event_id = len(events)
        if source_tx is not None:
            union_events(event_id, tx_events[source_tx])
        instance_id = -1
        target_id = len(nodes)
        target_chain = (
            target.chain_id if source is None or bridge_id is not None else source.chain_id
        )
        amount_minor = _sample_amount(action.amount_policy, amount_rng)
        node = TxNode(target_id, step, action.target_class, action_id, profile_id)
        nodes.append(node)
        participants = {actor.entity_id, target.entity_id, counterparty.entity_id}
        if source is not None:
            participants.add(source.entity_id)
        entity_ids = tuple(sorted(participants))
        node_meta.append(BehaviorNode(target_id, event_id, entity_ids, target_chain))
        edge_id = None
        if source_tx is not None:
            edge_id = len(edges)
            edges.append((source_tx, target_id))
            edge_meta.append(
                BehaviorEdge(
                    edge_id,
                    source_tx,
                    target_id,
                    event_id,
                    target_chain,
                    action.edge_kind,
                    action_id,
                    amount_minor,
                    step,
                    bridge_id,
                )
            )
        event = BehaviorEvent(
            event_id=event_id,
            instance_id=instance_id,
            step=step,
            actor_entity_id=actor.entity_id,
            counterparty_entity_ids=(counterparty.entity_id,),
            action_id=action_id,
            profile_id=profile_id,
            source_tx_ids=() if source_tx is None else (source_tx,),
            target_tx_id=target_id,
            target_entity_id=target.entity_id,
            amount_minor=amount_minor,
            chain_id=target_chain,
            bridge_id=bridge_id,
            source_entity_ids=() if source is None else (source.entity_id,),
        )
        events.append(event)
        tx_events[target_id] = event_id
        target.last_tx_id = target_id
        mark(actor.entity_id, step)
        mark(counterparty.entity_id, step)
        mark(target.entity_id, step)
        if source is not None:
            mark(source.entity_id, step)
    if len(nodes) != config.n_txs:
        raise BehaviorDriftError("behavior scheduler did not emit exact n_txs")
    components: dict[int, list[BehaviorEvent]] = {}
    for event in events:
        components.setdefault(find_event(event.event_id), []).append(event)
    ordered_roots = sorted(
        components, key=lambda root: min(event.event_id for event in components[root])
    )
    root_to_instance = {root: index for index, root in enumerate(ordered_roots)}
    events = [
        replace(event, instance_id=root_to_instance[find_event(event.event_id)]) for event in events
    ]
    edge_by_event = {edge.event_id: edge for edge in edge_meta}
    node_meta_by_tx = {meta.tx_id: meta for meta in node_meta}
    instances = []
    for instance_id, root in enumerate(ordered_roots):
        component_events = sorted(components[root], key=lambda event: event.event_id)
        event_ids = tuple(event.event_id for event in component_events)
        node_ids = tuple(sorted(event.target_tx_id for event in component_events))
        edge_ids = tuple(
            sorted(
                edge_by_event[event_id].edge_id
                for event_id in event_ids
                if event_id in edge_by_event
            )
        )
        if any(
            source not in node_ids or target not in node_ids
            for edge_id in edge_ids
            for source, target in (edges[edge_id],)
        ):
            raise BehaviorDriftError("behavior instance has an external edge endpoint")
        entity_ids = tuple(
            sorted(
                {
                    entity_id
                    for event in component_events
                    for entity_id in node_meta_by_tx[event.target_tx_id].entity_ids
                }
            )
        )
        profile_ids = tuple(sorted({event.profile_id for event in component_events}))
        chain_path = tuple(dict.fromkeys(event.chain_id for event in component_events))
        steps_for_instance = [nodes[tx_id].step for tx_id in node_ids]
        instances.append(
            BehaviorInstance(
                instance_id,
                component_events[0].profile_id,
                entity_ids,
                event_ids,
                node_ids,
                edge_ids,
                (min(steps_for_instance), max(steps_for_instance)),
                chain_path,
                profile_ids,
            )
        )
    records = [
        EntityRecord(
            state.entity_id,
            state.profile_id,
            get_profile(state.profile_id).version,
            get_profile(state.profile_id).entity_type,
            state.chain_id,
            _native_entity_id(config.seed, state.entity_id, state.chain_id),
            entity_steps[state.entity_id][0],
            entity_steps[state.entity_id][1],
        )
        for state in entities
    ]
    return BehaviorGraph(
        GeneratedGraph(nodes=nodes, edges=edges),
        node_meta,
        edge_meta,
        records,
        events,
        instances,
    )


def build_behavior_edge_attributes(
    graph: BehaviorGraph, timestamp_scale: int = 1200
) -> list[dict[str, int | float]]:
    """Build row-aligned amount and timestamp attributes for behavior edges.

    Parameters
    ----------
    graph : BehaviorGraph
        Behavior graph whose graph edges must align with ``edge_meta``.
    timestamp_scale : int, default 1200
        Number of timestamp units per elapsed time step.

    Returns
    -------
    list[dict[str, int | float]]
        One attribute record per graph edge, in the same order as ``graph.edges``.

    Raises
    ------
    ValueError
        If edge metadata count, IDs, or endpoints do not match the graph edges.
    """
    if len(graph.edge_meta) != len(graph.edges):
        raise ValueError("behavior edge metadata is not row-aligned")
    step_by_tx = {node.tx_id: node.step for node in graph.nodes}
    attrs: list[dict[str, int | float]] = []
    for edge_id, edge in enumerate(graph.edge_meta):
        if edge.edge_id != edge_id or graph.edges[edge_id] != (edge.txId1, edge.txId2):
            raise ValueError("behavior edge metadata is not row-aligned")
        attrs.append(
            {
                "txId1": edge.txId1,
                "txId2": edge.txId2,
                "amount": round(edge.amount_minor / 100.0, 2),
                "timestamp": timestamp_scale
                * (max(step_by_tx[edge.txId1], step_by_tx[edge.txId2]) - 1),
            }
        )
    return attrs


def project_behavior_instances(
    graph: BehaviorGraph, config: GeneratorConfig | None = None
) -> list[Any]:
    """Project behavior components into the common retrieval-instance shape.

    Parameters
    ----------
    graph : BehaviorGraph
        Graph containing behavior instances and their node and edge metadata.
    config : GeneratorConfig or None, optional
        Configuration controlling whether each instance uses a central anchor.
        When omitted, central anchors are used.

    Returns
    -------
    list[PatternInstance]
        Retrieval instances with anchors, distances, entity type, and edge recipes.
    """
    from .anchors import PatternInstance, _articulation_points, _central_anchor, _eccentricity

    projected: list[PatternInstance] = []
    for instance in graph.instances:
        node_ids = list(instance.node_ids)
        edge_ids = list(instance.edge_ids)
        anchor = (
            _central_anchor(node_ids, graph, edge_ids)
            if config is None or config.anchors.prefer_central_anchors
            else node_ids[0]
        )
        projected.append(
            PatternInstance(
                instance_id=instance.instance_id,
                pattern_type=instance.profile_id,
                node_ids=node_ids,
                edge_ids=edge_ids,
                anchor_node_id=anchor,
                anchor_role=graph.nodes[anchor].role,
                max_anchor_distance=_eccentricity(graph, node_ids, edge_ids, anchor),
                temporal_window=instance.temporal_window,
                edge_attr_recipe={edge_id: graph.edge_meta[edge_id].role for edge_id in edge_ids},
                entity_type=get_profile(instance.profile_id).entity_type,
                articulation_points=_articulation_points(graph, node_ids, edge_ids),
            )
        )
    return projected


__all__ = [
    "ActionDefinition",
    "BehaviorEdge",
    "BehaviorEvent",
    "BehaviorGraph",
    "BehaviorInstance",
    "BehaviorNode",
    "BehaviorProfile",
    "EntityRecord",
    "EntityState",
    "get_action",
    "get_chain",
    "get_profile",
    "build_behavior_graph",
    "build_behavior_edge_attributes",
    "derive_rng",
    "project_behavior_instances",
    "validate_registry",
]
