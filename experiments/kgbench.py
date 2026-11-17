"""KGBench — snowflake-schema knowledge graphs with a known ontological causal graph.

Why a generator, and why this shape
-----------------------------------
The bundled ``synthetic_clinic`` KG has three classes, so Assumption 1 forbids
16 % of its variable pairs and every experiment on it is one draw of one schema.
Whether a schema-derived constraint is worth anything is a question about how
its effect *scales with the schema*, and that needs schemas of controlled size
with a ground truth. This module generates them.

A benchmark KG is a snowflake: one *fact* class and a random tree of *dimension*
classes below it, every tree edge an N:1 object property pointing away from the
fact (``Therapy -> Patient -> Hospital -> Region``). That shape is chosen for
three reasons, each a property the experiments rely on:

* the flat join has exactly one row per fact entity, so "sample size" means one
  thing and is controlled directly (a tree with two N-side branches under one
  node would multiply rows instead);
* dimension attributes are replicated across rows by construction, so the
  i.i.d. violation the paper documents is present at a known multiplicity
  rather than assumed away;
* with ``k = 1`` the share of forbidden pairs is a function of the tree alone
  and grows towards 1 with the number of classes, which is the axis the
  constraint's benefit is plotted against.

Ground truth: a DAG over the property nodes ``(class, attribute)`` in which an
edge is either intra-class or goes from a dimension to the class that links to
it (the 1 side causes the N side: a hospital's attribute shapes its patients').
Every true edge therefore satisfies Assumption 1 — the benchmark measures what
the constraint buys when it is *correct*, and says nothing about a KG whose
schema is causally wrong. Mechanisms are additive-logit categoricals, so each
parent has a marginal effect and the truth is recoverable by a greedy learner
in the limit; an XOR-style mechanism would make every method fail for a reason
that has nothing to do with knowledge graphs.

Because the mechanism is kept separate from the sample, ``Mechanism.sample``
also draws from an *intervened* model. That is what lets
``e4_interventional.py`` score an interventional answer against the truth
instead of against another estimate.

Nothing here imports ``causalway``: the generator writes plain RDF (T-Box and
A-Box) and the experiments push that file through the public pipeline, so they
test the path a user's own KG would take.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS
from rdflib.namespace import XSD

__all__ = ["Mechanism", "Sample", "make_mechanism"]

NS = Namespace("http://example.org/kgbench/")


@dataclass
class Mechanism:
    """A schema, a ground-truth DAG over its property nodes, and the mechanisms."""

    n_classes: int
    n_attrs: int
    parent_class: list            # parent_class[c] = the class c links to (None for... see below)
    depth: list                   # depth[c]; the fact class is class 0 at depth 0
    levels: dict                  # node -> number of categorical levels
    parents: dict                 # node -> list of parent nodes (ground truth)
    weights: dict                 # node -> {parent: (levels[parent], levels[node]) array}
    bias: dict                    # node -> (levels[node],) array
    multiplicity: float = 12.0    # fact entities per depth-1 entity, and so on down

    # ``links_to[c]`` are the classes c has an N:1 object property to. The tree
    # is stored child -> parent in ``parent_class`` with the *fact* as root, so
    # the object property runs root-ward-to-leaf-ward: parent_class[d] = c means
    # c --rel--> d.
    @property
    def nodes(self) -> list:
        return [(c, i) for c in range(self.n_classes) for i in range(self.n_attrs)]

    @staticmethod
    def name(node) -> str:
        c, i = node
        return f"C{c}.a{c}_{i}"

    def links_to(self, c: int) -> list:
        return [d for d in range(self.n_classes) if self.parent_class[d] == c]

    def class_order(self) -> list:
        """Causes first: deepest dimension classes, the fact class last."""
        return sorted(range(self.n_classes), key=lambda c: -self.depth[c])

    def truth_edges(self) -> list:
        return [(self.name(p), self.name(n)) for n in self.nodes for p in self.parents[n]]

    def truth_adjacency(self, names: list) -> np.ndarray:
        idx = {name: k for k, name in enumerate(names)}
        adj = np.zeros((len(names), len(names)), dtype=int)
        for a, b in self.truth_edges():
            if a in idx and b in idx:
                adj[idx[a], idx[b]] = 1
        return adj

    def allowed_pairs_share(self) -> float:
        """Share of ordered node pairs Assumption 1 (k = 1) forbids, from the tree alone."""
        n = self.n_classes * self.n_attrs
        adjacent = sum(1 for d in range(1, self.n_classes))          # tree edges
        allowed = self.n_classes * self.n_attrs * (self.n_attrs - 1) + 2 * adjacent * self.n_attrs ** 2
        return 1.0 - allowed / (n * (n - 1))

    # ------------------------------------------------------------- sampling --
    def entity_counts(self, n_fact: int) -> list:
        counts = []
        for c in range(self.n_classes):
            size = n_fact / (self.multiplicity ** min(self.depth[c], 1)) / (4 ** max(self.depth[c] - 1, 0))
            counts.append(n_fact if c == 0 else max(6, int(round(size))))
        return counts

    def sample(self, n_fact: int, seed: int, do: Optional[dict] = None) -> "Sample":
        """Draw a KG instance; ``do`` maps a node to the level it is clamped to."""
        rng = np.random.default_rng(seed)
        do = do or {}
        counts = self.entity_counts(n_fact)
        link = {}                                   # link[d] = for each entity of parent_class[d], its d entity
        for d in range(1, self.n_classes):
            link[d] = rng.integers(0, counts[d], size=counts[self.parent_class[d]])

        def reach(c: int, d: int) -> np.ndarray:
            """For every class-c entity, the index of the class-d entity it reaches (d below c)."""
            chain = []
            while d != c:
                chain.append(d)
                d = self.parent_class[d]
            idx = np.arange(counts[c])
            for step in reversed(chain):
                idx = link[step][idx]
            return idx

        values = {}
        for c in self.class_order():
            for i in range(self.n_attrs):
                node = (c, i)
                k = self.levels[node]
                if node in do:
                    values[node] = np.full(counts[c], int(do[node]))
                    continue
                logits = np.tile(self.bias[node], (counts[c], 1))
                for p in self.parents[node]:
                    logits += self.weights[node][p][values[p][reach(c, p[0])]]
                prob = np.exp(logits - logits.max(axis=1, keepdims=True))
                prob /= prob.sum(axis=1, keepdims=True)
                u = rng.random(counts[c])
                values[node] = (u[:, None] > np.cumsum(prob, axis=1)).sum(axis=1).clip(0, k - 1)
        return Sample(self, counts, link, values)


@dataclass
class Sample:
    mechanism: Mechanism
    counts: list
    link: dict
    values: dict
    _index: dict = field(default_factory=dict)

    def entity_index(self, c: int) -> np.ndarray:
        """For every fact row, the index of the class-c entity it reaches."""
        if c not in self._index:
            m = self.mechanism
            if c == 0:
                self._index[c] = np.arange(self.counts[0])
            else:
                self._index[c] = self.link[c][self.entity_index(m.parent_class[c])]
        return self._index[c]

    def flat(self) -> pd.DataFrame:
        """The table the flat join must produce: one row per fact entity."""
        m = self.mechanism
        return pd.DataFrame({
            m.name(node): [f"L{v}" for v in self.values[node][self.entity_index(node[0])]]
            for node in m.nodes
        })

    def to_rdf(self) -> Graph:
        m = self.mechanism
        g = Graph()
        g.bind("kb", NS)
        for c in range(m.n_classes):
            g.add((NS[f"C{c}"], RDF.type, OWL.Class))
            g.add((NS[f"C{c}"], RDFS.label, Literal(f"class {c}")))
            for i in range(m.n_attrs):
                p = NS[f"a{c}_{i}"]
                g.add((p, RDF.type, OWL.DatatypeProperty))
                g.add((p, RDFS.domain, NS[f"C{c}"]))
                g.add((p, RDFS.range, XSD.string))
        for d in range(1, m.n_classes):
            r = NS[f"rel{d}"]
            g.add((r, RDF.type, OWL.ObjectProperty))
            g.add((r, RDFS.domain, NS[f"C{m.parent_class[d]}"]))
            g.add((r, RDFS.range, NS[f"C{d}"]))
        for c in range(m.n_classes):
            for e in range(self.counts[c]):
                iri = NS[f"c{c}_e{e}"]
                g.add((iri, RDF.type, NS[f"C{c}"]))
                for i in range(m.n_attrs):
                    g.add((iri, NS[f"a{c}_{i}"], Literal(f"L{self.values[(c, i)][e]}")))
        for d in range(1, m.n_classes):
            pc = m.parent_class[d]
            for e in range(self.counts[pc]):
                g.add((NS[f"c{pc}_e{e}"], NS[f"rel{d}"], NS[f"c{d}_e{self.link[d][e]}"]))
        return g

    def write_ttl(self, path: str) -> None:
        self.to_rdf().serialize(destination=path, format="turtle")


def make_mechanism(n_classes: int, n_attrs: int = 3, seed: int = 0, *,
                   p_intra: float = 0.45, p_inter: float = 0.25, max_parents: int = 3,
                   effect: float = 1.6, multiplicity: float = 12.0,
                   p_skip: float = 0.0) -> Mechanism:
    """Draw a schema tree, a ground-truth DAG that respects it, and the mechanisms.

    ``effect`` is the standard deviation of the additive logit weights. At 1.6 a
    parent moves the child's distribution by tens of percentage points, which
    keeps the truth learnable at a few hundred rows; the experiments vary the
    sample size, not this.

    ``p_skip`` > 0 makes Assumption 1 (k = 1) *false* on purpose: an attribute
    may then also be caused directly by an attribute of a class two relations
    away (a region's attribute acting on a patient with no hospital attribute
    in between). ``e7_violation.py`` uses it to measure what the constraint
    costs when the schema is not the causal structure. At the default 0 no
    extra random number is drawn, so every other experiment's KGs are unchanged.
    """
    rng = np.random.default_rng(seed)
    parent_class = [None] + [int(rng.integers(0, d)) for d in range(1, n_classes)]
    depth = [0] * n_classes
    for d in range(1, n_classes):
        depth[d] = depth[parent_class[d]] + 1

    mech = Mechanism(n_classes, n_attrs, parent_class, depth, {}, {}, {}, {}, multiplicity)
    for node in mech.nodes:
        mech.levels[node] = int(rng.integers(2, 4))

    for c in range(n_classes):
        for i in range(n_attrs):
            node = (c, i)
            candidates = [((c, j), p_intra) for j in range(i)]
            candidates += [((d, j), p_inter) for d in mech.links_to(c) for j in range(n_attrs)]
            if p_skip > 0:
                candidates += [((e, j), p_skip) for d in mech.links_to(c)
                               for e in mech.links_to(d) for j in range(n_attrs)]
            chosen = [p for p, prob in candidates if rng.random() < prob]
            if len(chosen) > max_parents:
                chosen = [chosen[k] for k in sorted(rng.choice(len(chosen), max_parents, replace=False))]
            mech.parents[node] = chosen
            mech.bias[node] = rng.normal(0, 0.3, size=mech.levels[node])
            mech.weights[node] = {
                p: rng.normal(0, effect, size=(mech.levels[p], mech.levels[node])) for p in chosen
            }
    return mech
