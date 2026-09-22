"""Publication-style figures of the synthetic dataset for the README."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kyt_engine.synth.anchors import attr_rng, build_edge_attributes, build_instances
from kyt_engine.synth.config import GeneratorConfig
from kyt_engine.synth.graph import build_graph
from kyt_engine.synth.stats import EllipticStats

CLASS_COLORS = {"illicit": "#d62828", "licit": "#2a9d43", "unknown": "#8d99ae"}
CLASS_LABELS = {"illicit": "illicit (1)", "licit": "licit (2)", "unknown": "unknown"}
LAYER_COLORS = ["#ece2f0", "#d4b9da", "#c994c7", "#df65b0", "#dd1c77"]  # 0..3+ hops

sn_schemes = ["mixer", "peel_chain", "fanout", "hub_spoke", "wash", "structuring"]

FIGDIR = Path("artifacts/figures")


def _setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.05)
    matplotlib.rcParams.update(
        {
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#444444",
            "grid.alpha": 0.35,
        }
    )


def _load_run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    feat = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    n_feat = feat.shape[1] - 2
    feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, 2 + n_feat)]
    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    classes["class"] = classes["class"].replace({"1": "illicit", "2": "licit", "unknown": "unknown"})
    edgelist = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    attrs = pd.read_csv(root / "elliptic_txs_edge_attributes.csv")
    manifest = json.loads((root / "manifest.json").read_text())
    return feat, classes, edgelist, attrs, manifest


# ---------------------------------------------------------------- figure 1
def fig_temporal_classes(feat: pd.DataFrame, classes: pd.DataFrame, out: Path) -> None:
    df = classes.merge(feat[["txId", "time_step"]], on="txId")
    count = (
        df.groupby(["time_step", "class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(range(1, 50), fill_value=0)
    )
    order = ["illicit", "licit", "unknown"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    count[order].plot.bar(
        ax=ax,
        stacked=True,
        width=0.9,
        color=[CLASS_COLORS[c] for c in order],
        legend=False,
        linewidth=0,
    )
    for s, tup in (("train", (0.5, 30.5)), ("valid", (30.5, 40.5)), ("test", (40.5, 49.5))):
        ax.axvspan(*tup, color="#1f77b4" if s == "train" else "#ff7f0e", alpha=0.06)
        ax.text(np.mean(tup), ax.get_ylim()[1] * 1.02, s, ha="center", va="bottom", fontsize=8, color="#555555")
    for x in (30.5, 40.5):
        ax.axvline(x, color="k", ls="--", lw=0.7, alpha=0.35)
    ax.legend(
        handles=[Line2D([0], [0], color=CLASS_COLORS[c], label=CLASS_LABELS[c], lw=6) for c in order],
        loc="upper left",
        frameon=True,
        ncol=3,
    )
    ax.set_xticks(range(1, 50, 2))
    ax.set_xlabel("time step")
    ax.set_ylabel("transactions")
    ax.set_title("(a) Class volume over the 49 time steps (train/valid/test split shaded)")
    fig.savefig(out / "fig_temporal_classes.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def _build_instances_for_figure() -> tuple[list, dict[int, dict], dict, int]:
    cfg = GeneratorConfig(
        seed=11,
        n_txs=6000,
        labeled_ratio=0.3,
        illicit_ratio_in_labeled=0.2,
        stats_dir=Path("data/elliptic_stats"),
        out_dir=Path("/tmp/kyt_figs_out"),
        schemes={
            "mixer": {"n_instances": 3, "fan_in_range": (5, 8), "fan_out_range": (5, 8)},
            "peel_chain": {"n_instances": 3, "length_range": (5, 8)},
            "fanout": {"n_instances": 3, "victims_range": (5, 10)},
            "hub_spoke": {"n_instances": 2, "spokes_range": (6, 10)},
            "wash": {"n_instances": 2, "cycle_len_range": (4, 6)},
            "structuring": {"n_instances": 2, "deposits_range": (5, 9)},
        },
        p2p_edges_per_tx=1.2,
    )
    cfg.drift.enabled = True
    cfg.drift.shutdown_step = 40
    cfg.drift.novel_schemes = ["wash"]
    cfg.drift.novel_step = 45
    stats = EllipticStats(cfg.stats_dir)
    graph = build_graph(cfg, stats)
    instances = build_instances(graph, cfg)
    edge_attrs = build_edge_attributes(graph, instances, cfg, attr_rng(cfg.seed))
    return graph, edge_attrs, instances, cfg.seed


def _scheme_layout_ordered(scheme: str, n: int) -> list[np.ndarray]:
    pos: list[tuple[float, float]] = [(0.0, 0.0)]
    if scheme in ("fanout", "hub_spoke", "structuring"):
        for i in range(1, n):
            ang = 2 * np.pi * (i - 1) / max(n - 1, 1)
            pos.append((1.4 * np.cos(ang), 1.4 * np.sin(ang)))
    elif scheme in ("peel_chain", "bridge_hopping", "amm_swap_chain"):
        for i in range(1, n):
            pos.append((0.0, float(-i)))
    elif scheme in ("wash", "cycle_round_trip", "cycle"):
        for i in range(1, n):
            ang = 2 * np.pi * i / n
            pos.append((np.cos(ang), np.sin(ang)))
    elif scheme == "mixer":
        n_in = max((n - 1) // 2, 1)
        n_out = max(n - 1 - n_in, 1)
        for i in range(1, 1 + n_in):
            t = 0.5 + 0.5 * (i - 1) / (n_in - 1 if n_in > 1 else 1)
            pos.append((-1.6, 2.6 * t - 1.3))
        for i in range(1 + n_in, n):
            t = 0.5 + 0.5 * (i - 1 - n_in) / (n_out - 1 if n_out > 1 else 1)
            pos.append((1.6, 2.6 * t - 1.3))
    return [np.array(p) for p in pos]


def fig_scheme_topologies(out: Path) -> None:
    graph, edge_attrs, instances, seed = _build_instances_for_figure()
    by_pattern: dict[str, object] = {}
    for inst in instances:
        by_pattern.setdefault(inst.pattern_type, inst)
    classes = {nd.tx_id: nd.cls for nd in graph.nodes}
    roles = {nd.tx_id: nd.role for nd in graph.nodes}

    cols = 3
    rows = (len(sn_schemes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9.5, 4.6 * len(sn_schemes) // cols + 0.4))
    axes = np.array(axes).reshape(-1)
    for k, scheme in enumerate(sn_schemes):
        ax = axes[k]
        inst = by_pattern[scheme]
        node_ids = inst.node_ids
        if scheme == "mixer":
            node_ids = [node_ids[0]] + sorted(node_ids[1:])
        local = {tid: i for i, tid in enumerate(node_ids)}
        edges = [(local[u], local[v]) for u, v in (graph.edges[e] for e in inst.edge_ids)]
        pos = {tid: p for tid, p in zip(node_ids, _scheme_layout_ordered(scheme, len(node_ids)))}
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        G.add_edges_from((node_ids[u], node_ids[v]) for u, v in edges)
        node_colors = {tid: CLASS_COLORS[classes[tid]] for tid in node_ids}
        amts = np.array([edge_attrs[e]["amount"] for e in inst.edge_ids], dtype=float)
        amts = np.maximum(amts, 1.0)
        widths = 0.6 + 2.4 * np.log1p(amts) / np.log1p(amts.max())
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=widths, edge_color="#5a5a5a", arrows=True, arrowsize=10, alpha=0.85
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=[node_colors[t] for t in node_ids],
            node_size=180, linewidths=0.7, edgecolors="white",
        )
        anchor = inst.anchor_node_id
        if anchor in local:
            ax.scatter(*pos[anchor], marker="*", s=420, c=CLASS_COLORS[classes[anchor]],
                       edgecolors="k", linewidths=0.8, zorder=5)
        core = node_ids[0]
        if core in pos and roles[core]:
            ax.text(pos[core][0], pos[core][1] + 0.35, roles[core], ha="center",
                    fontsize=7, color="#333333")
        ax.set_title(f"(b-{k + 1}) {scheme}", fontsize=9.5)
        ax.set_axis_off()
    for k in range(len(sn_schemes), len(axes)):
        axes[k].set_axis_off()
    fig.suptitle("Planted laundering / licit subgraphs (nodes colored by class, * = retrieval anchor, edge width ~ amount)",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out / "fig_scheme_topologies.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_amounts_by_scheme(edgelist: pd.DataFrame, attrs: pd.DataFrame, manifest: dict, out: Path) -> None:
    edge_scheme = {}
    for entry in manifest["edge_attribute_ground_truth"]:
        edge_scheme[entry["edge_id"]] = entry["pattern_type"]
    df = pd.DataFrame(
        {
            "scheme": [
                edge_scheme.get(i, "p2p (background)") for i in range(len(edgelist))
            ],
            "amount": attrs["amount"],
        }
    )
    order = sorted(df["scheme"].unique(), key=lambda s: -df.loc[df["scheme"] == s, "amount"].median()
                   if s != "p2p (background)" else -1e9)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    pal = sns.color_palette("husl", n_colors=len(order))
    sns.boxenplot(data=df, x="scheme", y="amount", order=order, hue="scheme",
                  palette=pal, linewidth=0.5, legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("edge pattern")
    ax.set_ylabel("edge amount (USD, log scale)")
    ax.set_title("(c) Edge amount distributions across planted patterns and background")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out / "fig_amounts_by_scheme.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_pca_classes(feat: pd.DataFrame, classes: pd.DataFrame, out: Path) -> None:
    X = feat.iloc[:, 2:].to_numpy(dtype=float)
    y = feat["txId"].map(classes.set_index("txId")["class"])
    scl = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=0).fit_transform(scl)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for cls in ("illicit", "licit", "unknown"):
        m = y == cls
        ax.scatter(pcs[m, 0], pcs[m, 1], s=7, alpha=0.45, c=CLASS_COLORS[cls],
                   label=CLASS_LABELS[cls], edgecolors="none")
    ax.legend(loc="best", frameon=True)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(d) Feature space (cdf mode), PCA to 2 dims")
    fig.savefig(out / "fig_pca_classes.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_anchor_neighbourhood(
    feat: pd.DataFrame, edgelist: pd.DataFrame, manifest: dict, out: Path
) -> None:
    anchors = manifest["anchor_registry"]["anchors"]
    anchor = next(a for a in anchors if a["pattern_type"] == "mixer")
    aid = anchor["anchor_node_id"]
    txids = set(feat["txId"])
    adj: dict[int, list[int]] = {t: [] for t in txids}
    for u, v in zip(edgelist["txId1"], edgelist["txId2"]):
        if u in txids and v in txids:
            adj[u].append(v)
    for lst in adj.values():
        lst.sort()

    dist: dict[int, int] = {aid: 0}
    queue = [aid]
    for u in queue:
        if dist[u] >= 3:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    seen = set(dist)
    for u in queue:
        for v in adj[u]:
            if v not in seen:
                seen.add(v)

    sub = sorted(seen)
    local = {tid: i for i, tid in enumerate(sub)}
    G = nx.DiGraph()
    G.add_nodes_from(range(len(sub)))
    edges = [(u, v) for u in sub for v in adj[u] if v in seen]
    G.add_edges_from((local[u], local[v]) for u, v in edges)

    pos = nx.spring_layout(G, seed=3, k=0.9)
    inst_id = {gt["txId"]: gt["instance_id"] for gt in manifest["ground_truth"]}
    anchor_inst = inst_id[aid]
    inst_nodes = {t for t, i in inst_id.items() if i == anchor_inst}

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    node_colors = [LAYER_COLORS[dist[t]] if t in dist else "#dddddd" for t in sub]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa0a6", width=0.7, alpha=0.8, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=90, linewidths=0.6,
        edgecolors="#666666" if len(sub) else "none",
    )
    members = [i for i, t in enumerate(sub) if t in inst_nodes]
    if members:
        xs = [pos[i][0] for i in members]
        ys = [pos[i][1] for i in members]
        ax.scatter(xs, ys, s=130, facecolors="none", edgecolors="#1a1a1a", linewidths=1.4, zorder=4)
    ax.scatter(*pos[local[aid]], marker="*", s=520, c="#7b2d8b", edgecolors="k", linewidths=0.9, zorder=6)
    handles = [
        Line2D([0], [0], marker="o", ls="", color="#df65b0", label="anchor (mixer core)"),
    ]
    handles += [
        Line2D([0], [0], marker="o", ls="", color=LAYER_COLORS[d],
               label=f"distance {d} hop{'s' if d != 1 else ''}" if d else "anchor") for d in sorted(set(dist.values()))
    ]
    handles.append(Line2D([0], [0], marker="o", ls="", markerfacecolor="none", markeredgecolor="#1a1a1a",
                          label="mixer instance members"))
    ax.legend(handles=handles, loc="best", frameon=True)
    ax.set_axis_off()
    ax.set_title("(e) Retrieval query: 3-hop neighbourhood of a mixer anchor (full graph)")
    fig.savefig(out / "fig_anchor_neighbourhood.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/synthetic/run")
    ap.add_argument("--out", default=str(FIGDIR))
    args = ap.parse_args()

    root = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _setup_style()
    feat, classes, edgelist, attrs, manifest = _load_run(root)

    fig_temporal_classes(feat, classes, out)
    fig_scheme_topologies(out)
    fig_amounts_by_scheme(edgelist, attrs, manifest, out)
    fig_pca_classes(feat, classes, out)
    fig_anchor_neighbourhood(feat, edgelist, manifest, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
