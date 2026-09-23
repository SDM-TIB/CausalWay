"""The discovered :class:`OntologicalCausalGraph`, its export, and metrics.

The RDF export is **not** written by hand here.  :meth:`~OntologicalCausalGraph.to_json`
renders the instance as a plain JSON document, and :meth:`~OntologicalCausalGraph.to_rdf`
hands that document plus ``ocg_mapping.rml.ttl`` to `SDM-RDFizer
<https://github.com/SDM-TIB/SDM-RDFizer>`_, which does the mapping.  The point
is that the JSON-to-triples contract lives in one declarative, standard RML
file that a reviewer can read (and that a non-Python consumer can run against
the same JSON) instead of being spread over a hundred ``g.add(...)`` calls.

Three properties of SDM-RDFizer 4.7.5 shaped both the JSON and the cw: terms
it produces; the mapping file documents them at length, and two of them are
visible from Python:

* **The ordered node layout is no longer an ``rdf:List``.**  RML's collection
  extension is parsed but produces nothing, so ``cw:nodeList`` is replaced by
  ``cw:NodeSlot`` resources carrying an explicit ``cw:slotIndex``.
  :meth:`~OntologicalCausalGraph.from_rdf` still reads the legacy list, so
  Turtle written before this change (everything under ``results/``) keeps
  loading.
* **The run's hyper-parameters are resources, not a JSON literal.**  Both ``"``
  and ``'`` are rewritten to ``\\'`` in any literal the engine emits, without a
  way back — and a JSON object is nothing but double quotes.  ``cw:parameters``
  is therefore superseded by ``cw:hasParameter`` → ``cw:parameterName`` /
  ``cw:parameterValue`` / ``cw:parameterKind``.  Any *other* literal holding a
  quote makes :meth:`~OntologicalCausalGraph.to_rdf` raise rather than write a
  silently corrupted one; no property or class local name in this project's
  schemas contains one.

A third shaped the vocabulary rather than the JSON: **there is no ordered
collection**, so an edge's multi-hop relation label is one ``cw:viaPath``
literal in SPARQL 1.1 property-path syntax (``^<a>/<b>``) rather than a
``cw:RelationPath`` of reified hops.  ``from_rdf`` reads both.

The cost is that ``to_rdf`` spawns a subprocess (~1 s, see
:mod:`causalway.rdfizer`) and touches a temporary directory, where it used to be
pure in-memory rdflib.  It is an
export path, called once per result, so that is paid gladly; ``to_json`` is
free and is the right thing to call in a loop.
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
from rdflib.collection import Collection

from ._util import local_name
from .constraints import EPSILON, EdgeConstraint, _is_hop, render_label
from .nodes import PropertyNode, _with_name
from .rdfizer import check_quote_free, materialise
from .vocab import CW, GRAPH_STEM, NODE_STEM, PROV, RUN_STEM, vocabulary

__all__ = [
    "OntologicalCausalGraph", "write_bundle", "vocabulary", "RML_MAPPING_PATH",
    "render_property_path", "parse_property_path",
    "CW", "PROV", "NODE_STEM", "GRAPH_STEM", "RUN_STEM",
]

#: The RML mapping that defines the JSON -> RDF contract.  Ships next to this
#: module; it is a data file, not importable Python, and the repo has no
#: ``pip install -e .``, so it is located relative to ``__file__``.
RML_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ocg_mapping.rml.ttl")


def _slug(text: str) -> str:
    """IRI-safe fragment."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(text)) or "x"


def _param_entry(name: str, value) -> tuple:
    """Render one hyper-parameter as the ``(kind, text)`` pair the RDF carries.

    Typing is explicit because RML fixes ``rr:datatype`` per predicate-object
    map, not per record: one ``cw:parameterValue`` predicate has to serve an
    ``alpha=0.05`` and a ``constrained=True`` alike, so the type travels
    alongside as ``cw:parameterKind`` and :meth:`_params_from_rdf` inverts it.
    ``bool`` is tested before ``int`` — in Python ``True`` *is* an ``int``.
    numpy scalars are unwrapped first: ``np.bool_`` is not a ``bool`` and
    ``np.int64`` not an ``int``, so a flag read back off a pandas index would
    otherwise fall through to the JSON branch as the quoted text ``"True"``,
    which :func:`~causalway.rdfizer.check_quote_free` then refuses.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "null", ""
    if isinstance(value, bool):
        return "bool", "true" if value else "false"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "float", repr(value)
    if isinstance(value, str):
        return "str", value
    return "json", json.dumps(value, sort_keys=True, default=str)


def _param_value(kind: str, text: str):
    """Inverse of :func:`_param_entry`."""
    if kind == "null":
        return None
    if kind == "bool":
        return text == "true"
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


#: Fields of :meth:`OntologicalCausalGraph.to_json` that carry an IRI, and are
#: therefore exempt from the engine's literal-quote rewriting.
_IRI_FIELDS = {
    "run_iri", "parameter_iri", "node_iri", "slot_iri", "graph_iri",
    "edge_iri", "cause_iri", "effect_iri", "derived_from",
    "domain", "property", "range", "relation",
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
    into a SPARQL query as-is.  This replaces the ``cw:RelationPath`` /
    ``cw:RelationHop`` resources and their five properties: those existed only
    to keep hop order and direction, which the operators already encode.
    Contains no quote, so SDM-RDFizer carries it intact.
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
            f"Not a cw:viaPath this project writes: {text!r}. Expected steps of the "
            "form '<iri>' or '^<iri>' joined by '/', with no grouping, alternation "
            "or modifiers."
        )
    return hops


def _labels_from_edge(graph: Graph, edge: URIRef) -> list:
    """Rebuild an edge's ``edge_labels`` entry from its reified RDF form.

    ``cw:viaPath`` carries the ordered, directed label exactly; ``cw:viaRelation``
    contributes any relation no path covers, which by construction is a plain
    forward hop (or ``cw:epsilon``).  Legacy Turtle whose paths are reified as
    ``cw:RelationPath``/``cw:RelationHop`` resources is read too, so everything
    under ``results/`` keeps loading.
    """
    labels: list = []
    covered_rels = set()

    for path in graph.objects(edge, CW.viaPath):
        if isinstance(path, Literal):
            hops = parse_property_path(str(path))
        else:                                   # legacy reified cw:RelationPath
            hops = [
                (graph.value(hop, CW.hopRelation),
                 str(graph.value(hop, CW.hopDirection)))
                for hop in sorted(graph.objects(path, CW.hop),
                                  key=lambda h: int(graph.value(h, CW.hopIndex)))
            ]
        if not hops:
            continue
        covered_rels.update(rel for rel, _ in hops)
        labels.append(hops[0] if len(hops) == 1 else tuple(hops))

    # Legacy only: a single-hop inverse label used to be a flat cw:viaRelation
    # plus a separate cw:relationDirection. Current exports put it in viaPath.
    directions = list(graph.objects(edge, CW.relationDirection))
    fallback_direction = str(directions[0]) if directions else "forward"
    for rel in graph.objects(edge, CW.viaRelation):
        if rel in covered_rels:
            continue
        labels.append((EPSILON, "epsilon") if rel == EPSILON
                      else (rel, fallback_direction))
    return labels


@dataclass
class OntologicalCausalGraph:
    """An edge-labeled directed graph over property nodes.

    Besides the structure itself the instance carries the *provenance* needed to
    store it in a knowledge graph: which method produced it, with which
    parameters, whether the topological constraint was enforced, and the
    per-edge weight (score / coefficient) the method assigned.
    """

    nodes: list
    adj: np.ndarray
    edge_labels: dict = field(default_factory=dict)  # (i, j) -> [label, ...]

    # --- provenance / annotation -------------------------------------- #
    weights: Optional[np.ndarray] = None   # (n, n) float; 1.0 assumed if None
    method: Optional[str] = None           # "GES", "NOTEARS", ...
    params: dict = field(default_factory=dict)   # {"alpha": 0.05, ...}
    constrained: Optional[bool] = None     # was Assumption 1 enforced?
    source: Optional[str] = None           # path/IRI of the input KG
    graph_id: Optional[str] = None         # explicit id; else content hash
    created_at: Optional[str] = None       # ISO-8601, filled on first export

    #: ``{(i, j), ...}`` — edges a human added during curation rather than a
    #: method proposing them.  A curated graph that cannot say which edges came
    #: from the analyst is not reproducible provenance, so this rides along and
    #: becomes ``cw:manuallyAdded`` on the edge.
    manual_edges: set = field(default_factory=set)

    #: IRIs (or ids) this graph was derived from — the contributing
    #: ``cw:DiscoveryRun``s of a curated selection.  Exported as
    #: ``prov:wasDerivedFrom`` on the graph.
    derived_from: list = field(default_factory=list)

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
                       derived_from: Optional[list] = None) -> "OntologicalCausalGraph":
        """Build an OCG from a 0/1 adjacency matrix and its constraint.

        ``weights`` is the method's ``(n, n)`` weight matrix (LiNGAM/NOTEARS/
        DAGMA/DAG-GNN coefficients; 1.0 for the constraint- and score-based
        methods).  The remaining arguments are provenance and end up on the
        ``cw:DiscoveryRun`` in the Turtle export.

        ``manual_edges`` and ``derived_from`` describe a *curated* graph: which
        ``(i, j)`` pairs the analyst added by hand, and which runs the selection
        was assembled from.
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
                   manual_edges={tuple(e) for e in (manual_edges or ())},
                   derived_from=list(derived_from or []))

    # ------------------------------------------------------------------ #
    # RDF import — inverts :meth:`to_rdf`
    # ------------------------------------------------------------------ #
    @classmethod
    def from_rdf(cls, graph: Graph, graph_id: Optional[str] = None,
                 method: Optional[str] = None) -> "OntologicalCausalGraph":
        """Rebuild an :class:`OntologicalCausalGraph` from an RDF graph written by :meth:`to_rdf`.

        Nodes come from the ``cw:slotIndex``-ordered ``cw:NodeSlot``
        resources (``cw:hasNode`` is unordered and would scramble the
        adjacency), falling back to the legacy ``cw:nodeList`` ``rdf:List``
        for Turtle written before the RML export — everything under
        ``results/`` is of that older vintage.  Each node is rebuilt as a
        :class:`~causalway.nodes.PropertyNode` and re-labelled from
        ``rdfs:label`` so a uniquified name (``foo_2``) survives.  Edges,
        weights and provenance are read off the reified ``cw:CausalEdge`` /
        ``cw:DiscoveryRun`` resources; hyper-parameters come from
        ``cw:hasParameter``, or from a legacy ``cw:parameters`` JSON literal.

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
            # cw:method lives on the DiscoveryRun (prov:wasGeneratedBy); only
            # legacy Turtle put it on the graph itself — same lookup as below.
            def _method_of(gi):
                run = graph.value(gi, PROV.wasGeneratedBy)
                found = graph.value(run, CW.method) if run is not None else None
                return found if found is not None else graph.value(gi, CW.method)

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

        slots = []
        for slot in graph.objects(graph_iri, CW.hasNodeSlot):
            index = graph.value(slot, CW.slotIndex)
            node = graph.value(slot, CW.slotNode)
            if index is not None and node is not None:
                slots.append((int(index), node))
        if slots:
            node_iris = [node for _, node in sorted(slots, key=lambda s: s[0])]
        else:
            node_list_bnode = graph.value(graph_iri, CW.nodeList)
            if node_list_bnode is None:
                raise ValueError(
                    f"{graph_iri} has neither cw:hasNodeSlot nor a legacy cw:nodeList"
                )
            node_iris = list(Collection(graph, node_list_bnode))

        nodes: list = []
        for node_iri in node_iris:
            domain = graph.value(node_iri, CW.domain)
            if domain is None:                       # legacy cw:domainClass
                domain = graph.value(node_iri, CW.domainClass)
            prop = graph.value(node_iri, CW["property"])
            kind = str(graph.value(node_iri, CW.nodeKind))
            range_ = graph.value(node_iri, CW["range"])
            if range_ is None:                       # legacy split range terms
                range_ = graph.value(
                    node_iri, CW.rangeClass if kind == "object" else CW.rangeDatatype
                )
            node = PropertyNode(domain=domain, prop=prop, range_=range_, kind=kind)
            label = graph.value(node_iri, RDFS.label)
            if label is not None and str(label) != node.name:
                node = _with_name(node, str(label))
            nodes.append(node)

        index_of = {iri: k for k, iri in enumerate(node_iris)}
        n = len(nodes)
        adj = np.zeros((n, n), dtype=int)
        weights = np.zeros((n, n), dtype=float)
        has_weights = False
        edge_labels: dict = {}
        manual_edges: set = set()

        for edge in graph.objects(graph_iri, CW.hasEdge):
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
            edge_labels[(i, j)] = _labels_from_edge(graph, edge)
            if bool(graph.value(edge, CW.manuallyAdded)):
                manual_edges.add((i, j))

        # `cw:method` and the constraint flag live on the cw:DiscoveryRun; the
        # graph reaches them through prov:wasGeneratedBy. Older Turtle duplicated
        # both onto the graph itself, so that copy is still read as a fallback.
        run_iri = graph.value(graph_iri, PROV.wasGeneratedBy)
        run_method = None
        constrained_lit = None
        derived_from = [str(d) for d in graph.objects(graph_iri, PROV.wasDerivedFrom)]
        source = None
        created_at = None
        params: dict = {}
        if run_iri is not None:
            run_method = graph.value(run_iri, CW.method)
            constrained_lit = graph.value(run_iri, CW.usedTopologicalConstraint)
            src = graph.value(run_iri, CW.sourceKG)
            source = str(src) if src is not None else None
            ended = graph.value(run_iri, PROV.endedAtTime)
            created_at = str(ended) if ended is not None else None
            for param in graph.objects(run_iri, CW.hasParameter):
                name = graph.value(param, CW.parameterName)
                if name is None:
                    continue
                kind = graph.value(param, CW.parameterKind)
                text = graph.value(param, CW.parameterValue)
                params[str(name)] = _param_value(
                    str(kind) if kind is not None else "str",
                    str(text) if text is not None else "",
                )
            params_lit = graph.value(run_iri, CW.parameters)
            if not params and params_lit is not None:   # legacy JSON literal
                try:
                    params = json.loads(str(params_lit))
                except json.JSONDecodeError:
                    params = {}

        if run_method is None:
            run_method = graph.value(graph_iri, CW.method)          # legacy
        if constrained_lit is None:
            constrained_lit = graph.value(graph_iri, CW.constrained)  # legacy
        recovered_gid = str(graph_iri)[len(GRAPH_STEM):]

        return cls(
            nodes=nodes, adj=adj, edge_labels=edge_labels,
            weights=weights if has_weights else None,
            method=str(run_method) if run_method is not None else None,
            params=params,
            constrained=bool(constrained_lit) if constrained_lit is not None else None,
            source=source, graph_id=recovered_gid, created_at=created_at,
            manual_edges=manual_edges, derived_from=derived_from,
        )

    # ------------------------------------------------------------------ #
    # Cycle handling — the discovery output is not guaranteed acyclic
    # ------------------------------------------------------------------ #
    def to_dag(self, strategy: str = "strict",
              constraint: Optional[EdgeConstraint] = None) -> "OntologicalCausalGraph":
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
        """
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
            source=self.source, graph_id=self.graph_id,
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
        stem is spelled out literally in ``ocg_mapping.rml.ttl``.
        """
        node = self.nodes[k]
        key = f"{node.domain}|{node.prop}|{node.range_}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:8]
        return f"{_slug(node.name)}-{digest}"

    def node_iri(self, k: int) -> URIRef:
        """Global IRI of property node ``k`` — the triple ``(D_p, p, R_p)``."""
        return URIRef(NODE_STEM + self.node_key(k))

    def edge_iri(self, i: int, j: int) -> URIRef:
        """IRI of the reified edge ``i -> j`` inside this graph."""
        return URIRef(f"{GRAPH_STEM}{self.gid}/edge/{i}_{j}")

    def slot_iri(self, k: int) -> URIRef:
        """IRI of the ``cw:NodeSlot`` holding column ``k`` of ``adj``."""
        return URIRef(f"{GRAPH_STEM}{self.gid}/slot/{k}")

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
    def to_json(self) -> dict:
        """Render the whole OCG instance as a plain JSON-serializable ``dict``.

        This is the document :meth:`to_rdf` feeds to SDM-RDFizer, so its shape
        is dictated by ``ocg_mapping.rml.ttl`` rather than by what would be
        prettiest to read: it is a set of flat arrays, one per triples map, and
        it is **denormalised** — every record repeats the keys its own subject
        template and object maps need, because a join onto a blank-node parent
        is broken in that engine and every join is therefore done here instead.

        Two conventions carry meaning:

        * **An absent key means "emit no triple".**  Optional provenance
          (``method``, ``constrained``, ``source``, ``manual``) is *omitted*,
          never set to ``None`` — RML skips an unmatched reference, but would
          happily write the string ``"None"`` for a null.
        * **Booleans are pre-rendered as ``"true"`` / ``"false"``** so the
          mapping can stamp ``xsd:boolean`` on them without a cast.

        Calling this fills :attr:`created_at` if it is still unset, exactly as
        :meth:`to_rdf` used to, so the same instance exported twice keeps one
        timestamp.
        """
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        gid = self.gid
        graph_iri = str(self.iri)
        edge_list = self.edges()
        node_iris = [str(self.node_iri(k)) for k in range(self.n)]
        label_of = f"({self.method or 'unspecified method'})"

        # `method` and the constraint flag describe the *run*, not the graph, and
        # the graph reaches the run through prov:wasGeneratedBy — so they are
        # written once, on the run, instead of being duplicated onto both.
        graph_record = {
            "gid": gid,
            "iri": graph_iri,
            "run_iri": str(self.run_iri),
            "label": f"Ontological causal graph {label_of}",
            "node_count": self.n,
            "edge_count": len(edge_list),
        }
        run_record = {
            "gid": gid,
            "iri": str(self.run_iri),
            "label": f"Causal discovery run: {self.method or 'unspecified method'}",
            "created_at": self.created_at,
        }
        if self.method:
            run_record["method"] = self.method
        if self.constrained is not None:
            run_record["constrained"] = "true" if self.constrained else "false"
        if self.source:
            run_record["source"] = str(self.source)

        derivations = [
            {"gid": gid, "graph_iri": graph_iri, "derived_from": str(d)}
            for d in self.derived_from
        ]

        parameters = []
        for name in sorted(self.params or {}):
            kind, text = _param_entry(name, self.params[name])
            param_key = _slug(name)
            record = {
                "gid": gid,
                "param_key": param_key,
                "parameter_iri": f"{RUN_STEM}{gid}/parameter/{param_key}",
                "name": name,
                "kind": kind,
            }
            # An *empty* reference is not an absent one to SDM-RDFizer: it
            # writes the literal "None". A null parameter (and the empty
            # string) therefore carries no cw:parameterValue at all, and
            # cw:parameterKind alone tells the two apart on the way back.
            if text != "":
                record["value"] = text
            parameters.append(record)

        nodes, slots = [], []
        for k, node in enumerate(self.nodes):
            record = {
                "gid": gid,
                "node_key": self.node_key(k),
                "node_iri": node_iris[k],
                "label": node.name,
                "domain": str(node.domain),
                "property": str(node.prop),
                "kind": node.kind,
                # One range term for a class and for a datatype alike, exactly as
                # rdfs:range is one term; cw:nodeKind already says which it is.
                "range": str(node.range_),
            }
            nodes.append(record)
            slots.append({
                "gid": gid,
                "graph_iri": graph_iri,
                "index": k,
                "slot_iri": str(self.slot_iri(k)),
                "node_iri": node_iris[k],
            })

        edges, edge_relations, edge_paths = [], [], []
        for (i, j) in edge_list:
            ni, nj = self.nodes[i], self.nodes[j]
            edge_key = f"{i}_{j}"
            edge_iri = str(self.edge_iri(i, j))
            labels = self.edge_labels.get((i, j), [])
            label = "; ".join(render_label(r) for r in labels) or "?"
            record = {
                "gid": gid,
                "edge_key": edge_key,
                "edge_iri": edge_iri,
                "graph_iri": graph_iri,
                "label": f"{ni.name} --[{label}]--> {nj.name}",
                "cause_iri": node_iris[i],
                "effect_iri": node_iris[j],
                "weight": self.weight(i, j),
            }
            if (i, j) in self.manual_edges:
                record["manual"] = "true"
            edges.append(record)

            # Every relation the edge's join touches is listed flat under
            # cw:viaRelation, so "which edges traverse this property" needs no
            # path parsing.  The *ordered, directed* form goes into one
            # cw:viaPath property-path literal — and only when it says
            # something viaRelation does not, i.e. never for a plain forward hop
            # and never for cw:epsilon, which has no direction to record.
            for label_value in labels:
                steps = _hops_of(label_value)
                for rel, _direction in steps:
                    edge_relations.append({"gid": gid, "edge_key": edge_key,
                                           "relation": str(rel)})
                if len(steps) == 1 and (steps[0][0] == EPSILON
                                        or steps[0][1] != "inverse"):
                    continue
                edge_paths.append({
                    "gid": gid, "edge_key": edge_key, "edge_iri": edge_iri,
                    "path": render_property_path(steps),
                })

        return {
            "graph": [graph_record],
            "run": [run_record],
            "derivations": derivations,
            "parameters": parameters,
            "nodes": nodes,
            "slots": slots,
            "edges": edges,
            "edge_relations": edge_relations,
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
        — no triple is written by this method.  The output is a complete,
        self-describing model:

        * one ``cw:OntologicalCausalGraph`` instance (this result),
        * the ``cw:DiscoveryRun`` that produced it (method, constraint flag,
          source KG, timestamp) and its ``cw:Parameter`` resources,
        * one ``cw:PropertyNode`` per node ``(D_p, p, R_p)``, plus one
          ``cw:NodeSlot`` fixing that node's column in ``adj``,
        * one reified ``cw:CausalEdge`` per edge, carrying ``cw:cause``,
          ``cw:effect``, ``cw:weight``, the relations it traverses
          (``cw:viaRelation``, ``cw:epsilon`` for intra-class edges) and,
          when order and direction matter, one ``cw:viaPath`` SPARQL
          property-path literal.

        Pass an existing ``graph`` to accumulate several methods into one store.
        Pass ``workdir`` to keep the generated ``ocg.json``, config and
        N-Triples output on disk for inspection instead of in a directory that
        is deleted on return — the only way to see what the engine actually
        received when a mapping change misbehaves.

        Raises ``RuntimeError`` if the engine fails, and ``ValueError`` if any
        literal contains a quote (see this module's docstring).
        """
        g = Graph() if graph is None else graph
        g.bind("cw", CW)
        g.bind("prov", PROV)
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

    Property nodes carry global IRIs, so the per-method graphs join on the same
    ``cw:PropertyNode`` resources and can be compared with one SPARQL query.

    ``ocgs`` is an iterable of :class:`OntologicalCausalGraph`, or a
    ``{method: ocg}`` mapping (the key is used as the method when the OCG does
    not carry one).
    """
    items = ocgs.items() if isinstance(ocgs, dict) else [(None, o) for o in ocgs]
    g = Graph()
    g.bind("cw", CW)
    g.bind("prov", PROV)
    if with_vocabulary:
        vocabulary(g)
    for key, ocg in items:
        if ocg.method is None and key is not None:
            ocg.method = str(key)
        ocg.to_rdf(graph=g)
    g.serialize(destination=path, format=format)
    return g


