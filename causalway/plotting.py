"""Edge-labeled graph drawing and the constraint heatmap."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .constraints import EdgeConstraint
from .result import OntologicalCausalGraph


def plot_graph(ocg: OntologicalCausalGraph, ax=None, seed: int = 42,
               node_size: int = 2600, font_size: int = 8, **kwargs):
    """Draw an :class:`OntologicalCausalGraph` with relation labels on edges."""
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))

    g = ocg.to_networkx()
    pos = nx.spring_layout(g, seed=seed, k=1.0 / max(1, ocg.n) ** 0.5)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size,
                           node_color="#cfe8f5", edgecolors="#1f77b4", linewidths=1.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=font_size)

    for (u, v, key, data) in g.edges(keys=True, data=True):
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=[(u, v)],
            connectionstyle="arc3,rad=0.15", arrowsize=16,
            arrowstyle="->", edge_color="#444444",
        )
    nx.draw_networkx_edge_labels(
        g, pos, ax=ax,
        edge_labels={(u, v): data["label"] for (u, v, key, data) in g.edges(keys=True, data=True)},
        font_size=max(5, font_size - 2), label_pos=0.55,
        connectionstyle="arc3,rad=0.15",
    )
    ax.set_axis_off()
    ax.set_title("Ontological causal graph")
    return ax


def plot_constraint_heatmap(constraint: EdgeConstraint, ax=None, cmap="RdBu",
                            title: str = "Allowed edges (Assumption 1)"):
    """Heatmap of ``allowed`` with relation labels annotated on allowed cells."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 9))

    n = constraint.n
    mask = constraint.allowed.astype(int)
    im = ax.imshow(mask, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(constraint.names, rotation=90, fontsize=7)
    ax.set_yticklabels(constraint.names, fontsize=7)

    for i in range(n):
        for j in range(n):
            if i != j and constraint.allowed[i, j]:
                ax.text(j, i, constraint.label_str(i, j), ha="center", va="center",
                        fontsize=5, color="#333333")

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="allowed (1) / forbidden (0)")
    return ax
