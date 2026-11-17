"""The discovered :class:`OntologicalCausalGraph`, its export, and metrics.

The RDF export is **not** written by hand here.  :meth:`~OntologicalCausalGraph.to_json`
renders the instance as a plain JSON document, and :meth:`~OntologicalCausalGraph.to_rdf`
hands that document plus ``ocg_mapping.rml.ttl`` to `SDM-RDFizer
<https://github.com/SDM-TIB/SDM-RDFizer>`_, which does the mapping.  The point
is that the JSON-to-triples contract lives in one declarative, standard RML
file that a reviewer can read (and that a non-Python consumer can run against
the same JSON) instead of being spread over a hundred ``g.add(...)`` calls.

What the export says (terms in :mod:`causalway.vocab`):

* ``graph cw:discoveredBy run``; the run carries ``cw:sourceKG``,
  ``cw:byMethod``, ``cw:discoveredTime`` and one ``cw:Parameter`` per setting —
  including ``maxhop`` and ``topologicalConstraint``, which are parameters of
  the run like any other rather than terms of their own.
* One ``cw:CausalNode`` per node, **scoped to this graph** (``cw:inGraph``).  Two
  graphs over one property have two node resources; compare them on
  ``(cw:domain, cw:property, cw:range)``.  Per-graph nodes are what make a
  graph's acyclicity checkable in SHACL (``causalway/shapes.ttl``): a property
  path over shared nodes would follow other graphs' edges.
* One ``cw:CausalEdge`` per edge, with ``cw:curated``, a ``cw:weight`` unless
  an expert drew the edge by hand, and one ``cw:viaRelationalPath`` per
  relational path — a SPARQL 1.1 property path from the cause's domain to the
  effect's.  An intra-class (ε) edge has no path: its cause and effect are
  properties of one entity.

Two properties of SDM-RDFizer 4.7.5 are visible from Python:

* **No collections.**  RML's collection extension is parsed but produces
  nothing, so a relational path is a property-path *literal*
  (:func:`render_property_path` / :func:`parse_property_path`), not an
  ``rdf:List`` — which also keeps each hop's direction (``^``), something a
  list of property IRIs would lose.
* **Quotes are corrupted.**  Both ``"`` and ``'`` are rewritten to ``\\'`` in
  any literal the engine emits, without a way back.  :meth:`to_rdf` raises on
  a quoted literal rather than write a silently corrupted one.

**Only a DAG is exported.**  PC and GES return a CPDAG, whose undirected
edges arrive here as a pair of opposite directed edges — a 2-cycle, which the
cw: model (and its acyclicity shape) does not admit.  :meth:`to_rdf` therefore
refuses a cyclic graph rather than publish one, and
:meth:`~OntologicalCausalGraph.to_dag` with ``strategy="extension"`` turns a CPDAG
into one DAG of its Markov equivalence class (:func:`extend_to_dag`).  The raw
adjacency is left alone everywhere else, so discovery metrics keep scoring what
the method actually returned.

Node order is not exported: a graph's nodes are a set.  :meth:`from_rdf`
restores the canonical order :func:`~causalway.nodes.build_nodes` produces —
sorted by ``(domain, property, range)`` — so ``adj`` round-trips index for
index for every graph this project builds.

The cost of the engine is that ``to_rdf`` spawns a subprocess (~1 s, see
:mod:`causalway.rdfizer`).  It is an export path, called once per result;
``to_json`` is free and is the right thing to call in a loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from ._util import local_name
from .constraints import EPSILON, EdgeConstraint, _is_hop, render_label
from .nodes import PropertyNode, _with_name
from .rdfizer import check_quote_free, materialise
from .vocab import (CW, GRAPH_STEM, PARAMETER_COMMENTS, RUN_STEM, kind_from_range,
                    vocabulary)

__all__ = [
    "OntologicalCausalGraph", "write_bundle", "vocabulary", "RML_MAPPING_PATH",
    "render_property_path", "parse_property_path", "parameter_rows", "extend_to_dag",
    "CW", "GRAPH_STEM", "RUN_STEM",
]

#: The RML mapping that defines the JSON -> RDF contract.  Ships next to this
#: module; it is a data file, not importable Python, and the repo has no
#: ``pip install -e .``, so it is located relative to ``__file__``.
RML_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ocg_mapping.rml.ttl")

#: Run parameters that are fields of :class:`OntologicalCausalGraph` rather
#: than entries of its ``params`` dict, and the field each one fills.
_FIELD_PARAMETERS = {"topologicalConstraint": "constrained", "maxhop": "max_hops"}


def _slug(text: str) -> str:
    """IRI-safe fragment."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(text)) or "x"


def _param_literal(value) -> Optional[tuple]:
    """``(text, datatype-or-None)`` for one parameter value, or ``None`` to write nothing.

    A value is exported as the typed literal it is — ``42`` as ``xsd:integer``,
    ``0.05`` as ``xsd:double``, ``True`` as ``xsd:boolean`` — and a string as a
    plain literal (which in RDF 1.1 *is* ``xsd:string``).  A null parameter
    writes no ``cw:Parameter`` at all: "not set" is the absence of a setting.
    A list or dict becomes comma-joined text, because JSON is all quotes and
    SDM-RDFizer rewrites quotes irreversibly.  numpy scalars are unwrapped
    first — ``np.bool_`` is not a ``bool`` — and ``bool`` is tested before
    ``int``, because in Python ``True`` *is* an ``int``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, bool):
        return ("true" if value else "false"), str(XSD.boolean)
    if isinstance(value, int):
        return str(value), str(XSD.integer)
    if isinstance(value, float):
        return repr(value), str(XSD.double)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value), None
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items())), None
    text = str(value)
    return (text, None) if text else None


def parameter_rows(owner_key: str, stem: str, params: dict) -> tuple:
    """``(plain, typed, links)`` rows for a run's parameters — shared by graph and model export.

    ``owner_key`` fills the run's subject template and ``stem`` is the run IRI
    the parameter IRIs hang off (``{stem}/parameter/{name}``).  Typed and plain
    values travel in separate arrays because an ``rml:datatypeMap`` row with no
    datatype takes its whole triples map down (``ocg_mapping.rml.ttl``, fact 4).
    """
    plain, typed, links = [], [], []
    for name in sorted(params or {}):
        rendered = _param_literal(params[name])
        if rendered is None:
            continue
        text, datatype = rendered
        key = _slug(name)
        iri = f"{stem}/parameter/{key}"
        row = {"gid": owner_key, "run_key": owner_key, "param_key": key,
               "parameter_iri": iri, "name": name, "value": text}
        comment = PARAMETER_COMMENTS.get(name)
        if comment:
            row["comment"] = comment
        if datatype:
            typed.append({**row, "datatype": datatype})
        else:
            plain.append(row)
        links.append({"gid": owner_key, "run_key": owner_key, "parameter_iri": iri})
    return plain, typed, links


#: Fields of :meth:`OntologicalCausalGraph.to_json` that carry an IRI, and are
#: therefore exempt from the engine's literal-quote rewriting.
_IRI_FIELDS = {
    "run_iri", "parameter_iri", "node_iri", "graph_iri", "edge_iri",
    "cause_iri", "effect_iri", "domain", "property", "range", "datatype",
}


def _hops_of(label) -> list:
    """Normalise one ``edge_labels`` entry to a list of ``(relation, direction)``."""
    return [label] if _is_hop(label) else list(label)


def render_property_path(hops) -> str:
    """Render ordered hops as one **SPARQL 1.1 property path**.

    ``[(treatedAt, "inverse"), (receives, "forward")]`` becomes
    ``^<http://.../treatedAt>/<http://.../receives>`` — ``^`` is the standard
    inverse-path operator and ``/`` the sequence operator, so the literal is
    both the exact machine-readable label *and* something that can be pasted
    into a SPARQL query as-is.  Contains no quote, so SDM-RDFizer carries it
    intact.
    """
    return "/".join(
        ("^" if direction == "inverse" else "") + f"<{rel}>" for rel, direction in hops
    )


#: One step of a property path: an optional inverse operator and an IRI in
#: angle brackets.  The brackets are what make the scan below possible — the
#: sequence operator ``/`` also occurs *inside* every http IRI, so the path
#: cannot simply be split on it.
_PATH_STEP = re.compile(r"\s*(\^?)\s*<([^<>]*)>\s*")


def parse_property_path(text: str) -> list:
    """Inverse of :func:`render_property_path` — exact, since the grammar is tiny.

    Only the two operators this project emits are recognised (``^`` and ``/``);
    a path written by hand with grouping, alternation or ``*`` is not something
    :meth:`OntologicalCausalGraph.to_rdf` can produce and is rejected rather
    than half-understood.
    """
    text = str(text)
    hops: list = []
    pos = 0
    while pos < len(text):
        if hops:
            if text[pos] != "/":
                break
            pos += 1
        match = _PATH_STEP.match(text, pos)
        if match is None:
            break
        hops.append((URIRef(match.group(2)),
                     "inverse" if match.group(1) else "forward"))
        pos = match.end()
    if not hops or pos != len(text):
        raise ValueError(
            f"Not a cw:viaRelationalPath this project writes: {text!r}. Expected steps "
            "of the form '<iri>' or '^<iri>' joined by '/', with no grouping, "
            "alternation or modifiers."
        )
    return hops


def _labels_from_edge(graph: Graph, edge: URIRef, same_domain: bool) -> list:
    """Rebuild an edge's ``edge_labels`` entry from its RDF form.

    Every ``cw:viaRelationalPath`` is one label.  The intra-class label ε is not
    written — it has no path — and is restored for every edge whose cause and
    effect share a domain class, first, as :meth:`EdgeConstraint.from_schema`
    orders it.  The flat join binds one variable per class, so two properties of
    one class are always read off the same entity and ε is always the operative
    relation between them.  (The one case this does not reproduce exactly is a
    same-class edge from a constraint built with ``allow_epsilon=False``.)
    """
    labels: list = [(EPSILON, "epsilon")] if same_domain else []
    for path in sorted(str(p) for p in graph.objects(edge, CW.viaRelationalPath)):
        hops = parse_property_path(path)
        labels.append(hops[0] if len(hops) == 1 else tuple(hops))
    return labels


# ---------------------------------------------------------------------- #
# CPDAG -> DAG.  A consistent extension (Dor & Tarsi, 1992): same skeleton,
# same directed edges, same v-structures, no cycle.
# ---------------------------------------------------------------------- #
def _is_undirected(adj: np.ndarray, i: int, j: int) -> bool:
    return bool(adj[i, j]) and bool(adj[j, i])


def _v_structures(adj: np.ndarray) -> set:
    """``{(a, b, c)}`` with ``a < c``, ``a -> b <- c`` both *directed*, ``a`` and ``c`` non-adjacent."""
    n = adj.shape[0]
    directed = (adj != 0) & (adj.T == 0)
    adjacent = (adj != 0) | (adj.T != 0)
    out = set()
    for b in range(n):
        parents = np.flatnonzero(directed[:, b])
        for x in range(len(parents)):
            for y in range(x + 1, len(parents)):
                a, c = int(parents[x]), int(parents[y])
                if not adjacent[a, c]:
                    out.add((a, b, c))
    return out


def _consistent_extension(pdag: np.ndarray, rng=None) -> Optional[np.ndarray]:
    """Dor–Tarsi: a DAG extending ``pdag`` with no new v-structure, or ``None`` if none exists.

    Repeatedly pick a node ``x`` that (a) has no outgoing directed edge among the
    remaining nodes and (b) whose every undirected neighbour is adjacent to all of
    ``x``'s other neighbours; orient every undirected edge at ``x`` *into* ``x``
    and remove ``x``.  (a) keeps the result acyclic, (b) is what forbids a new
    v-structure at ``x``.  ``rng`` picks among the eligible nodes, which is where
    the choice among Markov-equivalent DAGs is made; ``None`` picks the lowest
    index, deterministically.
    """
    a = (np.asarray(pdag) != 0).astype(int)
    n = a.shape[0]
    dag = ((a != 0) & (a.T == 0)).astype(int)          # the directed part, kept as is
    remaining = set(range(n))
    while remaining:
        eligible = []
        for x in sorted(remaining):
            if any(a[x, y] and not a[y, x] for y in remaining if y != x):
                continue                                  # (a) x still has a child
            neighbours = [y for y in remaining if y != x and (a[x, y] or a[y, x])]
            undirected = [y for y in neighbours if _is_undirected(a, x, y)]
            if all(a[y, z] or a[z, y] for y in undirected for z in neighbours if z != y):
                eligible.append(x)                        # (b)
        if not eligible:
            return None
        x = eligible[int(rng.integers(len(eligible)))] if rng is not None else eligible[0]
        for y in remaining:
            if y != x and _is_undirected(a, x, y):
                dag[y, x] = 1                             # y -> x
        remaining.discard(x)
    return dag


def _is_acyclic(adj: np.ndarray) -> bool:
    return nx.is_directed_acyclic_graph(nx.DiGraph((np.asarray(adj) != 0).astype(int)))


def extend_to_dag(adj: np.ndarray, admissible: Optional[np.ndarray] = None,
                  seed: Optional[int] = 0) -> tuple:
    """Orient a CPDAG's undirected edges into one DAG of its equivalence class.

    Returns ``(dag, oriented)`` where ``oriented`` is the set of ``(i, j)`` edges
    whose direction this function chose.  Guarantees, when it succeeds: the same
    skeleton, every directed edge of ``adj`` kept, **no new v-structure** and no
    cycle — so the DAG encodes exactly the independences the CPDAG does.

    Knowledge first, chance second.  An undirected edge that ``admissible``
    (Assumption 1's feasibility matrix, or anything shaped like it) allows in one
    direction only is oriented that way — but only if the graph still has an
    extension with the CPDAG's v-structures, since background knowledge that
    contradicts the data is not applied.  Every remaining choice is made by a
    seeded random order over the eligible nodes of :func:`_consistent_extension`:
    both orientations of such an edge score identically, so there is nothing in
    the data to prefer one, and the seed makes the arbitrary choice reproducible.

    ``adj`` with no undirected edge is returned unchanged.  If it has no
    consistent extension at all (its *directed* part already holds a cycle, as
    a non-CPDAG method can return), it is also returned unchanged, with an
    empty ``oriented`` — breaking such a cycle is a different decision
    (``to_dag(strategy="weight")``) and is not taken here.
    """
    adj = (np.asarray(adj) != 0).astype(int)
    n = adj.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if _is_undirected(adj, i, j)]
    if not pairs:
        return adj.copy(), set()
    target = _v_structures(adj)

    def valid(pdag) -> bool:
        ext = _consistent_extension(pdag)
        return ext is not None and _v_structures(ext) == target

    pdag = adj.copy()
    if admissible is not None:
        allowed = np.asarray(admissible, dtype=bool)
        for i, j in pairs:
            if allowed[i, j] == allowed[j, i]:
                continue
            drop = (j, i) if allowed[i, j] else (i, j)
            trial = pdag.copy()
            trial[drop] = 0
            if valid(trial):
                pdag = trial

    dag = _consistent_extension(pdag, rng=np.random.default_rng(seed))
    if dag is None or not _is_acyclic(dag) or _v_structures(dag) != target:
        return adj.copy(), set()
    oriented = {(i, j) if dag[i, j] else (j, i) for i, j in pairs}
    return dag, oriented


def _node_sort_key(node: PropertyNode) -> tuple:
    """The order :func:`~causalway.nodes.build_nodes` gives nodes — restored by :meth:`from_rdf`."""
    return (str(node.domain), str(node.prop), str(node.range_))


@dataclass
class OntologicalCausalGraph:
    """An edge-labeled directed graph over property nodes.

    Besides the structure itself the instance carries the *provenance* needed to
    store it in a knowledge graph: which method produced it, with which
    parameters, whether the topological constraint was enforced and at what
    path length, and the per-edge weight (score / coefficient) the method
    assigned.
    """

    nodes: list
    adj: np.ndarray
    edge_labels: dict = field(default_factory=dict)  # (i, j) -> [label, ...]

    # --- provenance / annotation -------------------------------------- #
    weights: Optional[np.ndarray] = None   # (n, n) float; 1.0 assumed if None
    method: Optional[str] = None           # "GES", "NOTEARS", ..., "curated"
    params: dict = field(default_factory=dict)   # {"alpha": 0.05, ...}
    constrained: Optional[bool] = None     # was Assumption 1 enforced? -> topologicalConstraint
    source: Optional[str] = None           # path/IRI of the input KG
    graph_id: Optional[str] = None         # explicit id; else content hash
    created_at: Optional[str] = None       # ISO-8601, filled on first export
    max_hops: Optional[int] = None         # longest relational path allowed -> maxhop

    #: True when an expert reviewed this graph's edges and kept them — every
    #: edge is then exported ``cw:curated true``.  A raw discovery run is not.
    curated: bool = False

    #: ``{(i, j), ...}`` — edges an expert added by hand rather than a method
    #: proposing them.  No method assigned them a strength, so they are
    #: exported without a ``cw:weight``; that absence is how RDF tells them apart.
    manual_edges: set = field(default_factory=set)

    #: ``{(i, j), ...}`` — edges whose direction :meth:`to_dag` chose by
    #: ``strategy="extension"`` rather than the method identifying it.  Their
    #: count and the seed become the run parameters ``orientedEdges`` /
    #: ``orientationSeed``, and their ``rdfs:label`` says so.
    oriented_edges: set = field(default_factory=set)

    @property
    def n(self) -> int:
        return len(self.nodes)

    @classmethod
    def from_discovery(cls, adj: np.ndarray, constraint: EdgeConstraint,
                       weights: Optional[np.ndarray] = None,
                       method: Optional[str] = None,
                       params: Optional[dict] = None,
                       constrained: Optional[bool] = None,
                       source: Optional[str] = None,
                       graph_id: Optional[str] = None,
                       manual_edges: Optional[set] = None,
                       curated: bool = False) -> "OntologicalCausalGraph":
        """Build an OCG from a 0/1 adjacency matrix and its constraint.

        ``weights`` is the method's ``(n, n)`` weight matrix (LiNGAM/NOTEARS/
        DAGMA/DAG-GNN coefficients; 1.0 for the constraint- and score-based
        methods).  The remaining arguments are provenance and end up on the
        ``cw:CausalDiscoveryRun`` in the Turtle export; the constraint's
        ``max_hops`` becomes its ``maxhop`` parameter.

        ``curated`` and ``manual_edges`` describe an expert-assembled graph:
        every edge was reviewed and kept, and ``manual_edges`` are the ``(i, j)``
        pairs the expert drew by hand.
        """
        n = adj.shape[0]
        edge_labels = {}
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j] == 1:
                    edge_labels[(i, j)] = constraint.labels.get((i, j), [])
        return cls(nodes=constraint.nodes, adj=np.asarray(adj, dtype=int),
                   edge_labels=edge_labels,
                   weights=None if weights is None else np.asarray(weights, dtype=float),
                   method=method, params=dict(params or {}),
                   constrained=constrained, source=source, graph_id=graph_id,
                   max_hops=getattr(constraint, "max_hops", None),
                   curated=curated,
                   manual_edges={tuple(e) for e in (manual_edges or ())})

    # ------------------------------------------------------------------ #
    # RDF import — inverts :meth:`to_rdf`
    # ------------------------------------------------------------------ #
    @classmethod
    def from_rdf(cls, graph: Graph, graph_id: Optional[str] = None,
                 method: Optional[str] = None) -> "OntologicalCausalGraph":
        """Rebuild an :class:`OntologicalCausalGraph` from an RDF graph written by :meth:`to_rdf`.

        Nodes are the ``cw:CausalNode``s with ``cw:inGraph`` this graph, put in
        the canonical ``(domain, property, range)`` order and re-labelled from
        ``rdfs:label`` so a uniquified name (``foo_2``) survives.  A node's kind
        is read off its range (:func:`~causalway.vocab.kind_from_range`).  Edges,
        weights and provenance come from the ``cw:CausalEdge`` and
        ``cw:CausalDiscoveryRun`` resources; the ``maxhop`` and
        ``topologicalConstraint`` parameters fill :attr:`max_hops` and
        :attr:`constrained`.

        When the graph carries several ``cw:OntologicalCausalGraph``
        instances (e.g. a bundle written by :func:`write_bundle`), pass
        ``graph_id=`` or ``method=`` to disambiguate; with exactly one
        candidate neither is required.
        """
        candidates = list(graph.subjects(RDF.type, CW.OntologicalCausalGraph))
        if graph_id is not None:
            graph_iri = URIRef(GRAPH_STEM + _slug(graph_id))
            if graph_iri not in candidates:
                raise ValueError(
                    f"No cw:OntologicalCausalGraph {graph_iri} in the given graph. "
                    f"Candidates: {candidates}"
                )
        elif method is not None:
            def _method_of(gi):
                run = graph.value(gi, CW.discoveredBy)
                return graph.value(run, CW.byMethod) if run is not None else None

            matches = [g for g in candidates if _method_of(g) == Literal(method)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one cw:OntologicalCausalGraph with method={method!r}, "
                    f"found {len(matches)}. Candidates: {candidates}"
                )
            graph_iri = matches[0]
        elif len(candidates) == 1:
            graph_iri = candidates[0]
        else:
            raise ValueError(
                "Ambiguous or missing cw:OntologicalCausalGraph in the given graph: "
                f"pass graph_id= or method= to disambiguate. Candidates: {candidates}"
            )

        read = []
        for node_iri in graph.subjects(CW.inGraph, graph_iri):
            if (node_iri, RDF.type, CW.CausalNode) not in graph:
                continue
            domain = graph.value(node_iri, CW.domain)
            prop = graph.value(node_iri, CW["property"])
            range_ = graph.value(node_iri, CW["range"])
            node = PropertyNode(domain=domain, prop=prop, range_=range_,
                                kind=kind_from_range(range_))
            label = graph.value(node_iri, RDFS.label)
            if label is not None and str(label) != node.name:
                node = _with_name(node, str(label))
            read.append((node, node_iri))
        read.sort(key=lambda pair: _node_sort_key(pair[0]))
        nodes = [node for node, _ in read]
        index_of = {iri: k for k, (_, iri) in enumerate(read)}

        n = len(nodes)
        adj = np.zeros((n, n), dtype=int)
        weights = np.zeros((n, n), dtype=float)
        has_weights = False
        edge_labels: dict = {}
        manual_edges: set = set()
        curated = False

        for edge in graph.subjects(CW.inGraph, graph_iri):
            if (edge, RDF.type, CW.CausalEdge) not in graph:
                continue
            cause = graph.value(edge, CW.cause)
            effect = graph.value(edge, CW.effect)
            if cause not in index_of or effect not in index_of:
                continue
            i, j = index_of[cause], index_of[effect]
            adj[i, j] = 1
            w = graph.value(edge, CW.weight)
            if w is not None:
                weights[i, j] = float(w)
                has_weights = True
            else:
                manual_edges.add((i, j))
            edge_labels[(i, j)] = _labels_from_edge(
                graph, edge, nodes[i].domain == nodes[j].domain)
            if bool(graph.value(edge, CW.curated)):
                curated = True
        # A raw run's edges all carry a weight; only a curated graph can hold an
        # edge without one, so outside a curated graph "no weight" means nothing.
        if not curated:
            manual_edges = set()

        run_iri = graph.value(graph_iri, CW.discoveredBy)
        run_method = source = created_at = None
        params: dict = {}
        if run_iri is not None:
            run_method = graph.value(run_iri, CW.byMethod)
            src = graph.value(run_iri, CW.sourceKG)
            source = str(src) if src is not None else None
            ended = graph.value(run_iri, CW.discoveredTime)
            created_at = str(ended) if ended is not None else None
            for param in graph.objects(run_iri, CW.hasParameter):
                name = graph.value(param, CW.parameterName)
                value = graph.value(param, CW.parameterValue)
                if name is not None and value is not None:
                    params[str(name)] = value.toPython()
        fields = {attr: params.pop(name, None) for name, attr in _FIELD_PARAMETERS.items()}

        return cls(
            nodes=nodes, adj=adj, edge_labels=edge_labels,
            weights=weights if has_weights else None,
            method=str(run_method) if run_method is not None else None,
            params=params,
            constrained=(bool(fields["constrained"])
                         if fields["constrained"] is not None else None),
            source=source, graph_id=str(graph_iri)[len(GRAPH_STEM):],
            created_at=created_at,
            max_hops=(int(fields["max_hops"]) if fields["max_hops"] is not None else None),
            curated=curated, manual_edges=manual_edges,
        )

    # ------------------------------------------------------------------ #
    # Cycle handling — the discovery output is not guaranteed acyclic
    # ------------------------------------------------------------------ #
    def to_dag(self, strategy: str = "strict",
              constraint: Optional[EdgeConstraint] = None,
              seed: Optional[int] = None) -> "OntologicalCausalGraph":
        """Return an acyclic version of this graph (``gcm.validate_causal_dag`` rejects cycles).

        ``strategy``:

        * ``"strict"`` (default) — raise, listing the offending cycle(s).
          Silently orienting an edge changes every downstream answer, so it
          must be a deliberate act.
        * ``"weight"`` — repeatedly break the lowest-``cw:weight`` edge of
          each remaining cycle until the graph is acyclic.
        * ``"constraint"`` — requires ``constraint=``; keep only directions
          with ``constraint.allowed[i, j]``, flipping an edge into its
          allowed reverse direction when only one direction is admissible and
          the reverse slot is free, else dropping it. Raises if cycles remain
          afterwards.
        * ``"extension"`` — for a CPDAG (PC, GES): orient each undirected edge
          (stored as two opposite edges) by :func:`extend_to_dag`, preferring the
          one direction Assumption 1 admits when there is one, and otherwise
          choosing one DAG of the equivalence class with ``seed`` (default: the
          run's own ``seed`` parameter, else 0). No new v-structure, no cycle,
          no edge dropped. The choice is recorded (``oriented_edges``, and the
          run parameters ``orientedEdges`` / ``orientationSeed``). A graph whose
          *directed* part is already cyclic is returned unchanged — that is a
          ``"weight"`` or ``"constraint"`` decision, not an orientation.
        """
        if strategy == "extension":
            return self._extend(seed)

        def cycle_of(a: np.ndarray):
            g = nx.DiGraph()
            g.add_nodes_from(range(self.n))
            for i in range(self.n):
                for j in range(self.n):
                    if a[i, j] == 1:
                        g.add_edge(i, j)
            try:
                return list(nx.find_cycle(g, orientation="original"))
            except nx.NetworkXNoCycle:
                return []

        if strategy == "strict":
            cyc = cycle_of(self.adj)
            if cyc:
                raise ValueError(
                    "OntologicalCausalGraph is not acyclic: found cycle "
                    f"{[(self.nodes[u].name, self.nodes[v].name) for u, v, _ in cyc]}. "
                    "gcm.validate_causal_dag will reject this graph; pass "
                    "to_dag(strategy='weight'|'constraint', ...) to resolve it deliberately."
                )
            return self

        adj = self.adj.copy()
        weights = None if self.weights is None else self.weights.copy()

        if strategy == "weight":
            while True:
                cyc = cycle_of(adj)
                if not cyc:
                    break
                edges_in_cycle = [(u, v) for u, v, _ in cyc]
                w = (lambda e: weights[e[0], e[1]]) if weights is not None else (lambda e: 1.0)
                drop = min(edges_in_cycle, key=w)
                adj[drop[0], drop[1]] = 0
                if weights is not None:
                    weights[drop[0], drop[1]] = 0.0
        elif strategy == "constraint":
            if constraint is None:
                raise ValueError("to_dag(strategy='constraint') requires constraint=")
            for i in range(self.n):
                for j in range(self.n):
                    if adj[i, j] == 1 and not constraint.allowed[i, j]:
                        if constraint.allowed[j, i] and adj[j, i] == 0:
                            adj[i, j], adj[j, i] = 0, 1
                            if weights is not None:
                                weights[j, i], weights[i, j] = weights[i, j], 0.0
                        else:
                            adj[i, j] = 0
                            if weights is not None:
                                weights[i, j] = 0.0
            remaining = cycle_of(adj)
            if remaining:
                raise ValueError(
                    "to_dag(strategy='constraint') could not resolve all cycles: "
                    f"{[(self.nodes[u].name, self.nodes[v].name) for u, v, _ in remaining]}"
                )
        else:
            raise ValueError(f"Unknown to_dag strategy: {strategy!r}")

        edge_labels = {
            (i, j): self.edge_labels.get((i, j), [])
            for i in range(self.n) for j in range(self.n)
            if i != j and adj[i, j] == 1
        }
        return OntologicalCausalGraph(
            nodes=self.nodes, adj=adj, edge_labels=edge_labels, weights=weights,
            method=self.method, params=dict(self.params), constrained=self.constrained,
            source=self.source, graph_id=self.graph_id, max_hops=self.max_hops,
            curated=self.curated,
            manual_edges={e for e in self.manual_edges if adj[e[0], e[1]] == 1},
        )

    def _extend(self, seed: Optional[int]) -> "OntologicalCausalGraph":
        """``to_dag(strategy="extension")`` — see there."""
        if seed is None:
            run_seed = (self.params or {}).get("seed")
            seed = int(run_seed) if isinstance(run_seed, (int, np.integer)) else 0
        # An edge label exists only for a direction Assumption 1 admits, so the
        # labels *are* the admissibility matrix restricted to this graph's edges.
        admissible = np.zeros(self.adj.shape, dtype=bool)
        for (i, j), labels in self.edge_labels.items():
            admissible[i, j] = bool(labels)
        dag, oriented = extend_to_dag(self.adj, admissible=admissible, seed=seed)
        if not oriented:
            return self
        weights = None if self.weights is None else np.where(dag != 0, self.weights, 0.0)
        params = dict(self.params)
        params["orientedEdges"] = len(oriented)
        params["orientationSeed"] = int(seed)
        return OntologicalCausalGraph(
            nodes=self.nodes, adj=dag,
            edge_labels={e: labels for e, labels in self.edge_labels.items() if dag[e] != 0},
            weights=weights, method=self.method, params=params,
            constrained=self.constrained, source=self.source, graph_id=self.graph_id,
            created_at=self.created_at, max_hops=self.max_hops, curated=self.curated,
            manual_edges={e for e in self.manual_edges if dag[e] != 0},
            oriented_edges=set(oriented),
        )

    def edges(self) -> list:
        """Directed edge list ``[(i, j), ...]``."""
        out = []
        for i in range(self.n):
            for j in range(self.n):
                if self.adj[i, j] == 1:
                    out.append((i, j))
        return out

    def weight(self, i: int, j: int) -> float:
        """Weight of edge ``i -> j`` (1.0 when the method reports no weight)."""
        if self.weights is None:
            return 1.0
        return float(self.weights[i, j])

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    def params_json(self) -> str:
        """Canonical JSON of the run's hyper-parameters ("{}" when there are none)."""
        return json.dumps(self.params or {}, sort_keys=True, default=str)

    def fingerprint(self) -> str:
        """Content hash over (method, params, node names, adjacency).

        Stable across re-exports of one result, and distinct for two runs that
        differ only in their hyper-parameters even when the graph came out the
        same — so the parameters must be part of the hash.
        """
        h = hashlib.sha1()
        h.update((self.method or "").encode())
        h.update(self.params_json().encode())
        h.update("|".join(nd.name for nd in self.nodes).encode())
        h.update(np.asarray(self.adj, dtype=np.int8).tobytes())
        return h.hexdigest()[:12]

    @property
    def gid(self) -> str:
        """Identifier used in the graph / run / edge IRIs."""
        if self.graph_id:
            return _slug(self.graph_id)
        return f"{_slug(self.method or 'ocg')}-{self.fingerprint()}"

    @property
    def iri(self) -> URIRef:
        """IRI of this graph instance."""
        return URIRef(GRAPH_STEM + self.gid)

    @property
    def run_iri(self) -> URIRef:
        """IRI of the discovery run that produced this graph."""
        return URIRef(RUN_STEM + self.gid)

    def node_key(self, k: int) -> str:
        """Slug-safe last segment of :meth:`node_iri`, for the RML subject template.

        SDM-RDFizer percent-encodes a *subject* ``rr:template``'s field values,
        so the mapping can only interpolate fragments like this one; the IRI
        stem is spelled out literally in ``ocg_mapping.rml.ttl``.  The digest of
        ``(D_p, p, R_p)`` keeps two nodes apart whose names slug alike.
        """
        node = self.nodes[k]
        key = f"{node.domain}|{node.prop}|{node.range_}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:8]
        return f"{_slug(node.name)}-{digest}"

    def node_iri(self, k: int) -> URIRef:
        """IRI of causal node ``k`` *in this graph* — nodes are not shared across graphs."""
        return URIRef(f"{GRAPH_STEM}{self.gid}/node/{self.node_key(k)}")

    def edge_key(self, i: int, j: int) -> str:
        """``{causeKey}__{effectKey}`` — named by its endpoints, not by matrix position."""
        return f"{self.node_key(i)}__{self.node_key(j)}"

    def edge_iri(self, i: int, j: int) -> URIRef:
        """IRI of the edge ``i -> j`` inside this graph."""
        return URIRef(f"{GRAPH_STEM}{self.gid}/edge/{self.edge_key(i, j)}")

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def to_networkx(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for node in self.nodes:
            g.add_node(node.name, node=node)
        for (i, j) in self.edges():
            label = "; ".join(render_label(r) for r in self.edge_labels.get((i, j), [])) or "?"
            g.add_edge(self.nodes[i].name, self.nodes[j].name, key=(i, j),
                       label=label, weight=self.weight(i, j))
        return g

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (i, j) in self.edges():
            rows.append({
                "method": self.method,
                "cause": self.nodes[i].name,
                "effect": self.nodes[j].name,
                "cause_domain": local_name(self.nodes[i].domain),
                "effect_domain": local_name(self.nodes[j].domain),
                "relation": "; ".join(
                    render_label(r) for r in self.edge_labels.get((i, j), [])
                ) or "?",
                "weight": self.weight(i, j),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # JSON export — the source document of the RML mapping
    # ------------------------------------------------------------------ #
    def run_parameters(self) -> dict:
        """The run's parameters as exported: :attr:`params` plus ``maxhop`` and ``topologicalConstraint``."""
        out = dict(self.params or {})
        for name, attr in _FIELD_PARAMETERS.items():
            value = getattr(self, attr)
            if value is not None:
                out[name] = value
        return out

    def to_json(self) -> dict:
        """Render the whole OCG instance as a plain JSON-serializable ``dict``.

        This is the document :meth:`to_rdf` feeds to SDM-RDFizer, so its shape
        is dictated by ``ocg_mapping.rml.ttl`` rather than by what would be
        prettiest to read: it is a set of flat arrays, one per triples map, and
        it is **denormalised** — every record repeats the keys its own subject
        template and object maps need, because a join onto a blank-node parent
        is broken in that engine and every join is therefore done here instead.

        Two conventions carry meaning:

        * **An absent key means "emit no triple".**  An optional value
          (``weight`` of a hand-drawn edge, a parameter's ``comment``) is
          *omitted*, never set to ``None`` — RML skips an unmatched reference,
          but would happily write the string ``"None"`` for a null.
        * **Booleans are pre-rendered as ``"true"`` / ``"false"``** so the
          mapping can stamp ``xsd:boolean`` on them without a cast.

        Calling this fills :attr:`created_at` if it is still unset, so the same
        instance exported twice keeps one timestamp.
        """
        if not _is_acyclic(self.adj):
            cycle = nx.find_cycle(nx.DiGraph((np.asarray(self.adj) != 0).astype(int)))
            raise ValueError(
                "Not exported: the graph has a directed cycle "
                f"{[(self.nodes[u].name, self.nodes[v].name) for u, v in cycle]}, and a "
                "cw:OntologicalCausalGraph is a DAG. A pair of opposite edges is an "
                "undirected CPDAG edge: to_dag(strategy='extension') orients it without a "
                "new v-structure. A longer cycle needs to_dag(strategy='weight') or "
                "'constraint' — a deliberate choice, not one an export makes silently."
            )
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        gid = self.gid
        graph_iri = str(self.iri)
        run_iri = str(self.run_iri)
        node_iris = [str(self.node_iri(k)) for k in range(self.n)]

        plain, typed, links = parameter_rows(gid, run_iri, self.run_parameters())

        nodes = [{
            "gid": gid,
            "node_key": self.node_key(k),
            "graph_iri": graph_iri,
            "label": node.name,
            "domain": str(node.domain),
            "property": str(node.prop),
            "range": str(node.range_),
        } for k, node in enumerate(self.nodes)]

        edges, edge_paths = [], []
        for (i, j) in self.edges():
            ni, nj = self.nodes[i], self.nodes[j]
            edge_key = self.edge_key(i, j)
            labels = self.edge_labels.get((i, j), [])
            label = "; ".join(render_label(r) for r in labels) or "?"
            record = {
                "gid": gid,
                "edge_key": edge_key,
                "graph_iri": graph_iri,
                "label": f"{ni.name} --[{label}]--> {nj.name}" + (
                    " (direction chosen within the Markov equivalence class)"
                    if (i, j) in self.oriented_edges else ""),
                "cause_iri": node_iris[i],
                "effect_iri": node_iris[j],
                "curated": "true" if self.curated else "false",
            }
            # A hand-drawn edge has no method-assigned strength; the missing
            # cw:weight is how the RDF says the expert, not a method, put it there.
            if not (self.curated and (i, j) in self.manual_edges):
                record["weight"] = self.weight(i, j)
            edges.append(record)

            # One property-path literal per relational path.  ε has no path —
            # cause and effect are one entity — so it writes nothing, and
            # `_labels_from_edge` restores it from the shared domain class.
            for label_value in labels:
                steps = _hops_of(label_value)
                if len(steps) == 1 and steps[0][0] == EPSILON:
                    continue
                edge_paths.append({"gid": gid, "edge_key": edge_key,
                                   "path": render_property_path(steps)})

        return {
            "graph": [{"gid": gid, "run_iri": run_iri}],
            "run": [{
                "gid": gid,
                "method": self.method or "unspecified",
                "source": str(self.source) if self.source else "unknown",
                "discovered_time": self.created_at,
            }],
            "parameters": plain,
            "parameters_typed": typed,
            "run_parameters": links,
            "nodes": nodes,
            "edges": edges,
            "edge_paths": edge_paths,
        }

    # ------------------------------------------------------------------ #
    # RDF export — SDM-RDFizer applies ocg_mapping.rml.ttl to to_json()
    # ------------------------------------------------------------------ #
    def to_rdf(self, graph: Optional[Graph] = None,
               with_vocabulary: bool = False,
               workdir: Optional[str] = None) -> Graph:
        """Serialize the whole OCG instance into an rdflib :class:`Graph`.

        The triples are produced by `SDM-RDFizer
        <https://github.com/SDM-TIB/SDM-RDFizer>`_ from :meth:`to_json`'s
        document and the RML mapping shipped as ``causalway/ocg_mapping.rml.ttl``
        — no triple is written by this method.  The output is one
        ``cw:OntologicalCausalGraph``, the ``cw:CausalDiscoveryRun`` that
        discovered it with its ``cw:Parameter``s, one ``cw:CausalNode`` per node
        and one ``cw:CausalEdge`` per edge.

        Pass an existing ``graph`` to accumulate several graphs into one store.
        Pass ``workdir`` to keep the generated ``ocg.json``, config and
        N-Triples output on disk for inspection instead of in a directory that
        is deleted on return — the only way to see what the engine actually
        received when a mapping change misbehaves.

        Raises ``RuntimeError`` if the engine fails, and ``ValueError`` if any
        literal contains a quote (see this module's docstring).
        """
        g = Graph() if graph is None else graph
        g.bind("cw", CW)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)
        if with_vocabulary:
            vocabulary(g)

        payload = self.to_json()
        check_quote_free(payload, _IRI_FIELDS)
        materialise(payload, RML_MAPPING_PATH, source_name="ocg.json",
                    graph=g, workdir=workdir)
        return g

    def to_turtle(self, path: str, with_vocabulary: bool = False,
                  format: str = "turtle") -> None:
        """Write the full OCG instance (graph + run + nodes + edges) to ``path``."""
        self.to_rdf(with_vocabulary=with_vocabulary).serialize(
            destination=path, format=format)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def metrics(self, truth: "OntologicalCausalGraph") -> dict:
        """SHD, directed precision/recall/F1, skeleton F1, topological validity."""
        truth_adj = truth.adj
        adj = self.adj
        n = self.n

        # SHD on directed adjacency.
        shd = int(np.sum(adj != truth_adj))

        def tp_fn_fp(A, B):
            tp = int(np.sum((A == 1) & (B == 1)))
            fn = int(np.sum((A == 1) & (B == 0)))
            fp = int(np.sum((A == 0) & (B == 1)))
            return tp, fn, fp

        tp, fn, fp = tp_fn_fp(truth_adj, adj)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        skel_a = ((adj + adj.T) > 0).astype(int)
        skel_t = ((truth_adj + truth_adj.T) > 0).astype(int)
        tp_s, fn_s, fp_s = tp_fn_fp(skel_t, skel_a)
        skel_prec = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
        skel_rec = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
        skel_f1 = (
            2 * skel_prec * skel_rec / (skel_prec + skel_rec)
            if (skel_prec + skel_rec) else 0.0
        )

        return {
            "shd": shd,
            "n_edges": int(adj.sum()),
            "n_truth_edges": int(truth_adj.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "skeleton_precision": skel_prec,
            "skeleton_recall": skel_rec,
            "skeleton_f1": skel_f1,
        }

    def topological_validity(self, constraint: EdgeConstraint) -> float:
        """Fraction of returned edges satisfying Assumption 1 (1.0 if no edges)."""
        edges = self.edges()
        if not edges:
            return 1.0
        valid = sum(1 for (i, j) in edges if constraint.allowed[i, j])
        return valid / len(edges)


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def write_bundle(ocgs, path: str, with_vocabulary: bool = True,
                 format: str = "turtle") -> Graph:
    """Serialize several OCGs (e.g. one per method) into a single KG file.

    Causal nodes are scoped to their graph, so the per-method graphs do not
    share node resources; compare them in SPARQL by joining on
    ``(cw:domain, cw:property, cw:range)``.

    ``ocgs`` is an iterable of :class:`OntologicalCausalGraph`, or a
    ``{method: ocg}`` mapping (the key is used as the method when the OCG does
    not carry one).
    """
    items = ocgs.items() if isinstance(ocgs, dict) else [(None, o) for o in ocgs]
    g = Graph()
    g.bind("cw", CW)
    if with_vocabulary:
        vocabulary(g)
    for key, ocg in items:
        if ocg.method is None and key is not None:
            ocg.method = str(key)
        ocg.to_rdf(graph=g)
    g.serialize(destination=path, format=format)
    return g


