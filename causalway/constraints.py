"""Assumption 1 (topological causal assumption) as a hard edge constraint.

The :class:`EdgeConstraint` encodes, for every ordered pair of property nodes,
whether ``i -> j`` is *topologically valid* under Assumption 1, and stores the
relation label (``epsilon`` for intra-class, or the object property traversed
forward/inverse for inter-class).  Thin adapter methods convert the matrix into
the prior-knowledge object each discovery library expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdflib import URIRef

from ._util import local_name
from .nodes import PropertyNode
from .ontology import OntologySchema

EPSILON = URIRef("http://sdm-causalway.org/epsilon")


def _relations_between(schema: OntologySchema, from_cls, to_cls,
                       relation_direction: str, max_hops: int) -> list:
    """Return the relation labels connecting ``from_cls`` to ``to_cls``.

    For ``max_hops == 1`` each label is a ``(relation, direction)`` tuple.  For
    ``max_hops > 1`` each label is a tuple of such hops (a path of length <=
    ``max_hops``).
    """
    if from_cls == to_cls:
        return []

    if max_hops <= 1:
        labels = []
        for p, drs in schema.object_properties.items():
            for (d, r) in drs:
                if d == from_cls and r == to_cls:
                    labels.append((p, "forward"))
                elif relation_direction == "both" and d == to_cls and r == from_cls:
                    labels.append((p, "inverse"))
        return labels

    # Multi-hop: BFS over the class graph, building labelled paths.
    from collections import deque
    paths = []
    queue = deque([(from_cls, [])])
    seen = {from_cls}
    while queue:
        node, hops = queue.popleft()
        if len(hops) == max_hops:
            continue
        for _, nxt, data in schema.class_graph.edges(node, data=True):
            p = data["property"]
            direction = "forward"
            if nxt == node:  # reflexive edge, skip
                continue
            if nxt in seen and nxt != to_cls:
                continue
            new_hops = hops + [(p, direction)]
            if nxt == to_cls:
                paths.append(tuple(new_hops))
            else:
                queue.append((nxt, new_hops))
                seen.add(nxt)
        # inverse traversal when relation_direction == "both"
        if relation_direction == "both":
            for pred in schema.class_graph.predecessors(node):
                for data in schema.class_graph.get_edge_data(pred, node).values():
                    p = data["property"]
                    new_hops = hops + [(p, "inverse")]
                    if pred == to_cls:
                        paths.append(tuple(new_hops))
                    elif pred not in seen:
                        queue.append((pred, new_hops))
                        seen.add(pred)
    return paths


@dataclass
class EdgeConstraint:
    """Hard constraint matrix + relation labels for a node set.

    Attributes
    ----------
    names: column names aligned with ``allowed``.
    nodes: the :class:`PropertyNode` objects, one per column.
    allowed: ``(n, n)`` boolean matrix, ``allowed[i, j]`` means ``i -> j`` is
        topologically valid.
    labels: ``(i, j) -> [label, ...]`` where each label is either a single
        ``(relation, "forward"|"inverse")`` tuple, ``(EPSILON, "epsilon")``, or
        (for multi-hop relaxations) a tuple of such hop tuples.
    """

    names: list
    nodes: list
    allowed: np.ndarray
    labels: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_schema(cls, schema: OntologySchema, nodes: list[PropertyNode],
                    relation_direction: str = "both",
                    allow_epsilon: bool = True,
                    max_hops: int = 1) -> "EdgeConstraint":
        n = len(nodes)
        allowed = np.zeros((n, n), dtype=bool)
        labels: dict = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Di, Dj = nodes[i].domain, nodes[j].domain
                rels = []
                if allow_epsilon and Di == Dj:
                    rels.append((EPSILON, "epsilon"))
                rels.extend(_relations_between(
                    schema, Di, Dj, relation_direction, max_hops
                ))
                if rels:
                    allowed[i, j] = True
                    labels[(i, j)] = rels

        return cls(
            names=[nd.name for nd in nodes],
            nodes=list(nodes),
            allowed=allowed,
            labels=labels,
        )

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.names)

    def subset(self, indices) -> "EdgeConstraint":
        """Return a constraint restricted to the given node indices (in order)."""
        idx = list(indices)
        names = [self.names[i] for i in idx]
        nodes = [self.nodes[i] for i in idx]
        allowed = self.allowed[np.ix_(idx, idx)]
        labels = {
            (ni, nj): self.labels[(i, j)]
            for ni, i in enumerate(idx)
            for nj, j in enumerate(idx)
            if (i, j) in self.labels
        }
        return EdgeConstraint(names=names, nodes=nodes, allowed=allowed, labels=labels)

    def skeleton_allowed(self) -> np.ndarray:
        """Undirected skeleton view: ``allowed | allowed.T``."""
        return self.allowed | self.allowed.T

    def mask(self) -> np.ndarray:
        """``(n, n)`` float 0/1 matrix, 1 where the edge is allowed."""
        return self.allowed.astype(float)

    def stats(self) -> dict:
        total = self.n * (self.n - 1)
        n_allowed = int(self.allowed.sum())
        n_forbidden = total - n_allowed
        return {
            "n_nodes": self.n,
            "n_allowed": n_allowed,
            "n_forbidden": n_forbidden,
            "pruning_rate": n_forbidden / total if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Adapters
    # ------------------------------------------------------------------ #
    def forbidden_edges(self) -> list:
        """Directed forbidden edges as ``(name_i, name_j)`` for pgmpy."""
        n = self.n
        return [
            (self.names[i], self.names[j])
            for i in range(n) for j in range(n)
            if i != j and not self.allowed[i, j]
        ]

    def search_space(self) -> list:
        """Allowed directed edges (white list) as ``(name_i, name_j)``."""
        n = self.n
        return [
            (self.names[i], self.names[j])
            for i in range(n) for j in range(n)
            if self.allowed[i, j]
        ]

    def to_expert_knowledge(self):
        """pgmpy :class:`ExpertKnowledge` white-list for HillClimbSearch.

        ``search_space`` is the set of allowed directed edges; pgmpy derives the
        complementary forbidden edges from it, which is an exact hard constraint
        for both add and flip operations.
        """
        from pgmpy.causal_discovery import ExpertKnowledge
        return ExpertKnowledge(search_space=self.search_space())

    def to_skeleton_expert_knowledge(self):
        """pgmpy :class:`ExpertKnowledge` for the PC skeleton.

        Forbids both directions of a pair only when the pair is fully forbidden
        (neither direction allowed), so PC prunes exactly the skeleton edges that
        Assumption 1 rules out.
        """
        from pgmpy.causal_discovery import ExpertKnowledge
        skel = self.skeleton_allowed()
        forbidden = []
        n = self.n
        for i in range(n):
            for j in range(i + 1, n):
                if not skel[i, j]:
                    forbidden.append((self.names[i], self.names[j]))
                    forbidden.append((self.names[j], self.names[i]))
        return ExpertKnowledge(forbidden_edges=forbidden)

    def to_castle_priori(self):
        """gcastle :class:`PrioriKnowledge` with all forbidden directed edges."""
        from castle.common.priori_knowledge import PrioriKnowledge
        n = self.n
        priori = PrioriKnowledge(n_nodes=n)
        for i in range(n):
            for j in range(n):
                if i != j and not self.allowed[i, j]:
                    priori.add_forbidden_edge(i, j)
        return priori

    def to_lingam_prior(self) -> np.ndarray:
        """``(n, n)`` prior for gcastle DirectLiNGAM.

        Emits ``0`` at ``[j, i]`` to forbid ``i -> j`` (gcastle's matrix is
        transposed relative to an adjacency matrix), ``-1`` (unknown) elsewhere.
        """
        n = self.n
        mat = np.full((n, n), -1, dtype=float)
        np.fill_diagonal(mat, 0.0)
        for i in range(n):
            for j in range(n):
                if i != j and not self.allowed[i, j]:
                    mat[j, i] = 0.0
        return mat

    def exclude_edges(self) -> list:
        """``[(i, j), ...]`` of forbidden directed edges, for DAGMA."""
        n = self.n
        return [
            (i, j) for i in range(n) for j in range(n)
            if i != j and not self.allowed[i, j]
        ]

    def soft_prior(self, eps: float = 1e-4) -> np.ndarray:
        """A ``B`` prior matrix with ``eps`` on forbidden entries, 0.5 elsewhere.

        Intended for composition with the LLM directional prior, not as the
        enforcement mechanism (the hard constraint is applied independently).
        """
        n = self.n
        B = np.full((n, n), 0.5)
        B[~self.allowed] = eps
        np.fill_diagonal(B, 0.5)
        return B

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def label_str(self, i: int, j: int) -> str:
        """Human-readable relation label(s) for edge ``i -> j``."""
        rels = self.labels.get((i, j), [])
        if not rels:
            return ""
        return "; ".join(render_label(r) for r in rels)

    def relation_iris(self, i: int, j: int) -> list:
        """Object-property IRIs traversed by edge ``i -> j`` (empty for epsilon)."""
        iris = []
        for label in self.labels.get((i, j), []):
            if _is_hop(label):
                r, _ = label
                if r != EPSILON:
                    iris.append(r)
            else:
                for hop in label:
                    r, _ = hop
                    if r != EPSILON:
                        iris.append(r)
        return iris


def _is_hop(label) -> bool:
    return isinstance(label, tuple) and len(label) == 2 and isinstance(label[0], URIRef)


def render_label(label) -> str:
    if _is_hop(label):
        r, direction = label
        if r == EPSILON:
            return "ε"
        return f"{local_name(r)} ({direction})"
    return " → ".join(render_label(hop) for hop in label)
