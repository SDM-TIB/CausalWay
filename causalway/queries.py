"""``Query`` / ``Answer`` dataclasses, IRI minting, RDF round-trip, and reach validation.

The cw: model this module serves (terms in :mod:`causalway.vocab`):

* A **query is posed on a causal graph**, not on a model.  It is one of
  ``P(V | Z=z)`` (conditional), ``P(V | do(X=x), Z=z)`` (interventional) or
  ``P(V_{do(X=x)} | e)`` (counterfactual), and it carries exactly X (its
  interventions), Z (its conditions) and — for a counterfactual about a named
  unit — the entity e.  Its IRI is minted from that content and the graph, so
  the same question asked twice is one resource.
* Its **answers are evaluated by a model**, one per target node.  The target set
  V is not stored on the query: it is the set of nodes its answers are about,
  which is exactly what the service's boards compute (every node not in X ∪ Z).
  One query can therefore carry the answers of several models.
* **No reference arm.**  An average causal effect ``E[Y | do(x)] - E[Y | do(x')]``
  is two interventional queries; the subtraction needs no term of its own.
  :func:`causalway.inference.intervene` still computes a contrast when asked —
  that is an engine feature, and it does not reach the RDF.

Disjointness is part of the model and enforced here, at construction, as well
as in ``causalway/shapes.ttl``: an interventional or conditional query must
have X ∩ Z = ∅, and the answers of any query must not be about a node in X.  A
counterfactual is the exception on purpose — its evidence may name the very
node it intervenes on ("she smoked; had she not…"), which is what makes it a
counterfactual rather than an intervention.

``Answer`` stays the rich object the inference engines return (backend, ESS,
per-row diagnostics); only a slice of it — predicted value, its probability or
standard deviation, the factual value — becomes RDF.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd
from rdflib import Graph, RDF, RDFS, URIRef

from .bgp import Materialization
from .entities import rows_for_entity
from .result import OntologicalCausalGraph
from .vocab import ANSWER_STEM, CONDITION_STEM, CW, GRAPH_STEM, INTERVENTION_STEM, QUERY_STEM

__all__ = [
    "Intervention", "Condition", "Query", "Answer", "KINDS",
    "query_iri", "intervention_iri", "condition_iri", "answer_iri", "answers_of",
    "validate_query", "store_query", "load_queries", "load_answers", "replay",
]

#: The three query kinds, in rung order.
KINDS = ("conditional", "interventional", "counterfactual")


# ---------------------------------------------------------------------- #
# Dataclasses
# ---------------------------------------------------------------------- #
@dataclass
class Intervention:
    """``do(node := value)``.

    ``entity`` is engine-level scoping, not part of the cw: model: it names
    *which* entity's ``node`` a cross-entity counterfactual sets (a patient's own
    hospital), so :func:`validate_query` can check that the two are joined in
    the source KG.  It is not exported — in the cw: model an intervention in a
    counterfactual acts on the unit and the entities the flat join relates to it.
    """

    node: str                          # CausalNode name / SCM column
    value: Any = None
    entity: Optional[str] = None       # engine scoping only; never exported


@dataclass
class Condition:
    """``Z = z``: an observed value, on a sub-population or on a hand-described unit."""

    node: str
    value: Any


@dataclass
class Query:
    """A causal query on one causal graph. ``kind`` entails the rung — never declared separately.

    ``graph_id`` is the ``gid`` of the :class:`OntologicalCausalGraph` the query
    is posed on; left ``None``, the exporter fills it in from the model's graph.
    """

    kind: str                                            # conditional | interventional | counterfactual
    interventions: list = field(default_factory=list)    # X: list[Intervention]
    conditions: list = field(default_factory=list)       # Z: list[Condition]
    entity: Optional[str] = None                         # e: counterfactual about a named unit
    graph_id: Optional[str] = None
    label: Optional[str] = None                          # the estimand, as text
    qid: Optional[str] = None                            # override the content id

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Query(kind={self.kind!r}): expected one of {KINDS}")
        x = [iv.node for iv in self.interventions]
        z = [c.node for c in self.conditions]
        for what, nodes in (("intervened on", x), ("conditioned on", z)):
            twice = sorted({n for n in nodes if nodes.count(n) > 1})
            if twice:
                raise ValueError(f"Query: {', '.join(twice)} is {what} twice.")
        if self.kind == "conditional" and x:
            raise ValueError("Query(kind='conditional') has no interventions; "
                             "a do() makes it interventional.")
        if self.kind != "conditional" and not x:
            raise ValueError(f"Query(kind={self.kind!r}) needs at least one intervention.")
        if self.kind != "counterfactual":
            if self.entity:
                raise ValueError(
                    f"Query(kind={self.kind!r}) is about a (sub-)population: it cannot "
                    "carry entity=. A question about one unit is a counterfactual.")
            overlap = sorted(set(x) & set(z))
            if overlap:
                raise ValueError(
                    f"Query: {', '.join(overlap)} is both intervened on and conditioned "
                    "on (X ∩ Z must be empty outside a counterfactual).")
        elif bool(self.entity) == bool(z):
            raise ValueError(
                "Query(kind='counterfactual') needs exactly one of entity= (a named unit, "
                "whose evidence is in the source KG) or conditions= (the observed state "
                "of a hand-described unit).")

    @property
    def intervened(self) -> set:
        return {iv.node for iv in self.interventions}

    @property
    def conditioned(self) -> set:
        return {c.node for c in self.conditions}

    def excluded_targets(self) -> set:
        """Nodes this query's answers may not be about: X always, and Z outside a counterfactual."""
        if self.kind == "counterfactual":
            return self.intervened
        return self.intervened | self.conditioned


@dataclass
class Answer:
    """What an inference engine returns for one target. Rich in Python; only a slice becomes RDF."""

    kind: str
    target: str                          # the node this answer is about
    entity: Optional[str] = None         # set iff kind == "counterfactual" about a named unit
    predicted: Any = None
    distribution: Optional[dict] = None  # {level: probability} for categorical
    mean: Optional[float] = None
    std: Optional[float] = None
    ci: Optional[tuple] = None
    factual: Optional[Any] = None
    effect: Any = None                   # contrast vs a reference arm (engine only, not exported)
    backend: Optional[str] = None
    n_samples: Optional[int] = None
    ess: Optional[float] = None
    low_confidence: bool = False
    per_row: Optional[pd.DataFrame] = None
    coupling: Optional[str] = None       # counterfactual only: "gumbel-max"
    evaluated_time: Optional[str] = None # ISO-8601; the exporter stamps "now" when unset

    @property
    def probability(self) -> Optional[float]:
        """Mass of the reported level, when the answer carries a distribution."""
        if not self.distribution or self.predicted is None:
            return None
        mass = self.distribution.get(self.predicted,
                                     self.distribution.get(str(self.predicted)))
        return None if mass is None else float(mass)


def answers_of(answers) -> list:
    """Normalise ``Answer | list[Answer] | {node: Answer} | None`` to a list."""
    if answers is None:
        return []
    if isinstance(answers, Answer):
        return [answers]
    if isinstance(answers, dict):
        return list(answers.values())
    return list(answers)


# ---------------------------------------------------------------------- #
# Deterministic IRIs — minted from content, so the same treatment on the same
# graph is one cw:Intervention however many queries use it.
# ---------------------------------------------------------------------- #
def _hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:n]


def intervention_iri(graph_id: str, iv: Intervention) -> URIRef:
    return URIRef(INTERVENTION_STEM + _hash(f"{graph_id}|{iv.node}|{iv.value}"))


def condition_iri(graph_id: str, c: Condition) -> URIRef:
    return URIRef(CONDITION_STEM + _hash(f"{graph_id}|{c.node}|{c.value}"))


def _query_content_id(query: Query) -> str:
    payload = json.dumps({
        "graph": query.graph_id, "kind": query.kind,
        "interventions": sorted(f"{iv.node}={iv.value}" for iv in query.interventions),
        "conditions": sorted(f"{c.node}={c.value}" for c in query.conditions),
        "entity": query.entity,
    }, sort_keys=True)
    return _hash(payload)


def query_key(query: Query) -> str:
    return query.qid or _query_content_id(query)


def query_iri(query: Query) -> URIRef:
    """The query IRI — which is also the hypothetical world's IRI; there is no separate world."""
    return URIRef(QUERY_STEM + query_key(query))


def answer_iri(query: Query, model_id: str, node_key: str) -> URIRef:
    """``answer/{query}/{model}/{node}`` — one answer per query, model and target node."""
    return URIRef(f"{ANSWER_STEM}{query_key(query)}/{model_id}/{node_key}")


# ---------------------------------------------------------------------- #
# Validating an intervention's reach
# ---------------------------------------------------------------------- #
def _node_index(ocg: OntologicalCausalGraph, name: str) -> int:
    for i, n in enumerate(ocg.nodes):
        if n.name == name:
            return i
    raise KeyError(f"Node {name!r} not found in the causal graph (known: "
                   f"{[n.name for n in ocg.nodes]})")


def _has_path(ocg: OntologicalCausalGraph, i: int, j: int) -> bool:
    if i == j:
        return True
    seen = {i}
    stack = [i]
    while stack:
        u = stack.pop()
        for v in range(ocg.n):
            if ocg.adj[u, v] == 1 and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def validate_query(query: Query, ocg: OntologicalCausalGraph,
                   mat: Optional[Materialization] = None, *,
                   targets: Iterable[str] = ()) -> None:
    """Reject a query whose interventions cannot reach its targets — both checks.

    1. **Causal reach**: a directed path in ``ocg`` from every intervened node
       to every target in ``targets``.  An intervention that reaches no target
       changes nothing, and answering it would report a number as if it did.
    2. **Relational reach** (only when an intervention names an entity other
       than the query's own): the two entities must co-occur in at least one
       materialised row, i.e. be joined in the source KG — needs ``mat=``.

    Also re-checks the disjointness rule for ``targets``: no target may be in X,
    nor in Z outside a counterfactual.  Raises ``ValueError``; stores nothing —
    both checks are re-derivable from the graph and the KG.
    """
    targets = list(targets)
    bad = sorted(set(targets) & query.excluded_targets())
    if bad:
        raise ValueError(
            f"validate_query: {', '.join(bad)} cannot be a target — it is "
            + ("intervened on." if query.kind == "counterfactual"
               else "intervened on or conditioned on."))
    for t in targets:
        t_idx = _node_index(ocg, t)
        for iv in query.interventions:
            i_idx = _node_index(ocg, iv.node)
            if not _has_path(ocg, i_idx, t_idx):
                raise ValueError(
                    f"validate_query: no causal path from {iv.node!r} to target {t!r}; "
                    "this intervention cannot reach the target (causal reach)."
                )
    for iv in query.interventions:
        if iv.entity and query.entity and str(iv.entity) != str(query.entity):
            if mat is None:
                raise ValueError(
                    "validate_query: cross-entity intervention on "
                    f"{iv.entity!r} needs the source KG (mat=) to check relational reach.")
            if not (set(rows_for_entity(mat, query.entity))
                    & set(rows_for_entity(mat, iv.entity))):
                raise ValueError(
                    f"validate_query: {iv.entity} is not joined to {query.entity} in the "
                    "source KG along any relation (relational reach).")


# ---------------------------------------------------------------------- #
# RDF export
# ---------------------------------------------------------------------- #
_QUERY_CLASS = {
    "conditional": CW.ConditionalQuery,
    "interventional": CW.InterventionalQuery,
    "counterfactual": CW.CounterfactualQuery,
}
_ANSWER_CLASS = {
    "conditional": CW.ConditionalAnswer,
    "interventional": CW.InterventionalAnswer,
    "counterfactual": CW.CounterfactualAnswer,
}


def store_query(answer, query: Query, model, *,
                graph: Optional[Graph] = None, merge: str = "annotation") -> Graph:
    """Serialise ``query`` and its answer(s) into ``graph``.

    ``answer`` is one :class:`Answer`, a list of them, or the ``{node: Answer}``
    dict a multi-target engine call returns.  ``model`` is the fitted
    :class:`~causalway.model.CausalModel` that evaluated them.

    ``merge``:

    * ``"annotation"`` (default) — the query, its interventions and conditions,
      and its answers as plain triples.  Asserts no ``entity property value``,
      so merging into the source KG contradicts nothing.
    * ``"named-graph"`` — additionally projects each counterfactual answer as
      ``GRAPH <queryIri> { entity property value }`` (needs a quad store).

    A thin wrapper over :func:`causalway.query_export.queries_to_rdf`; the
    named-graph projection stays here because it deliberately asserts what the
    export contract never does.
    """
    if merge not in ("annotation", "named-graph"):
        raise ValueError(f"Unknown merge mode: {merge!r}")
    if graph is not None:
        g = graph
    elif merge == "named-graph":
        from rdflib import Dataset
        g = Dataset()
    else:
        g = Graph()
    if merge == "named-graph" and not hasattr(g, "get_context"):
        raise ValueError(
            "store_query(merge='named-graph') needs a quad-capable graph "
            "(rdflib.Dataset / ConjunctiveGraph), not a plain rdflib.Graph."
        )

    from .query_export import queries_to_rdf   # imports this module; one-way at import time

    queries_to_rdf([(query, answer)], model, graph=g)
    if merge == "named-graph":
        _project_named_graph(g, query, answers_of(answer), model)
    return g


def _project_named_graph(g: Graph, query: Query, answers: list, model) -> None:
    """Materialise each counterfactual answer inside ``GRAPH <queryIri>``.

    Only a counterfactual about a named entity has a subject to project onto;
    a population answer has no entity whose property could take the value.
    """
    if query.kind != "counterfactual" or not query.entity:
        return
    from rdflib import Literal
    ocg = model.spec.ocg
    ctx = g.get_context(query_iri(query))
    for ans in answers:
        prop = ocg.nodes[_node_index(ocg, ans.target)].prop
        value = ans.predicted
        term = (URIRef(value) if isinstance(value, str) and value.startswith(("http://", "https://"))
                else Literal(value))
        ctx.add((URIRef(query.entity), prop, term))


# ---------------------------------------------------------------------- #
# Round trip: RDF -> executable Query
# ---------------------------------------------------------------------- #
def load_queries(source, ocg: Optional[OntologicalCausalGraph] = None) -> list:
    """Reconstruct :class:`Query` objects from a KG.

    ``source`` is anything :func:`causalway.sources.resolve_graph` accepts.
    Node names are resolved through ``ocg`` when given (a query export does not
    repeat its graph), else through the nodes' ``rdfs:label`` when the graph is
    in ``source``.  ``qid`` is the IRI's tail, so :func:`query_iri` of a loaded
    query is the IRI it was loaded from.
    """
    from .sources import resolve_graph
    g = source if isinstance(source, Graph) else resolve_graph(source)
    names = _names_of(ocg)
    _node_name = lambda g_, iri: names.get(iri) or _label_of(g_, iri)  # noqa: E731

    kind_of = {v: k for k, v in _QUERY_CLASS.items()}
    out = []
    for q_iri, cls in g.subject_objects(RDF.type):
        if cls not in kind_of:
            continue
        on_graph = g.value(q_iri, CW.onGraph)
        entity = g.value(q_iri, CW.aboutEntity)
        label = g.value(q_iri, RDFS.label)
        out.append(Query(
            kind=kind_of[cls],
            interventions=[Intervention(node=_node_name(g, g.value(i, CW.onNode)),
                                        value=_py(g.value(i, CW.setValue)))
                           for i in g.objects(q_iri, CW.hasIntervention)],
            conditions=[Condition(node=_node_name(g, g.value(c, CW.onNode)),
                                  value=_py(g.value(c, CW.observedValue)))
                        for c in g.objects(q_iri, CW.hasCondition)],
            entity=str(entity) if entity is not None else None,
            graph_id=(str(on_graph)[len(GRAPH_STEM):] if on_graph is not None else None),
            label=str(label) if label is not None else None,
            qid=str(q_iri)[len(QUERY_STEM):],
        ))
    return out


def load_answers(source, ocg: Optional[OntologicalCausalGraph] = None) -> pd.DataFrame:
    """One row per stored answer: ``query, model, node, predicted, probability, std, factual``.

    Node names are resolved as in :func:`load_queries`.
    """
    from .sources import resolve_graph
    g = source if isinstance(source, Graph) else resolve_graph(source)
    names = _names_of(ocg)
    _node_name = lambda g_, iri: names.get(iri) or _label_of(g_, iri)  # noqa: E731
    rows = []
    for cls in _ANSWER_CLASS.values():
        for a in g.subjects(RDF.type, cls):
            rows.append({
                "query": str(g.value(a, CW.forQuery)),
                "model": str(g.value(a, CW.evaluatedBy)),
                "node": _node_name(g, g.value(a, CW.onNode)),
                "predicted": _py(g.value(a, CW.predictedValue)),
                "probability": _py(g.value(a, CW.probability)),
                "std": _py(g.value(a, CW.std)),
                "factual": _py(g.value(a, CW.factualValue)),
            })
    return pd.DataFrame(rows, columns=["query", "model", "node", "predicted",
                                       "probability", "std", "factual"])


def _py(term):
    return None if term is None else term.toPython()


def _names_of(ocg: Optional[OntologicalCausalGraph]) -> dict:
    """``{node IRI: node name}`` for a graph, or ``{}``."""
    if ocg is None:
        return {}
    return {ocg.node_iri(k): node.name for k, node in enumerate(ocg.nodes)}


def _label_of(g: Graph, node_iri) -> str:
    label = g.value(node_iri, RDFS.label) if node_iri is not None else None
    return str(label) if label is not None else str(node_iri)


def replay(source, model) -> pd.DataFrame:
    """Re-evaluate every stored answer of ``model`` and compare it with what was stored.

    The targets of each query are the nodes its stored answers are about (V is
    not stored on the query).  Returns ``query, kind, node, stored, recomputed,
    match``.  A counterfactual about a hand-described unit is not replayed here —
    the engine abducts from the source KG's rows — and reports ``recomputed``
    as ``None``.
    """
    from .inference import condition as run_conditional
    from .inference import counterfactual as run_counterfactual
    from .inference import intervene as run_interventional
    from .sources import resolve_graph

    g = source if isinstance(source, Graph) else resolve_graph(source)
    stored = load_answers(g, ocg=model.spec.ocg)
    stored = stored[stored["model"] == str(model.iri)]
    rows = []
    for q in load_queries(g, ocg=model.spec.ocg):
        mine = stored[stored["query"] == str(query_iri(q))]
        if mine.empty:
            continue
        targets = list(mine["node"])
        z = {c.node: c.value for c in q.conditions}
        x = {iv.node: iv.value for iv in q.interventions}
        if q.kind == "conditional":
            out = run_conditional(model, targets, z)
        elif q.kind == "interventional":
            out = run_interventional(model, x, target=targets, conditions=z or None)
        elif q.entity:
            out = run_counterfactual(model, {(None, k): v for k, v in x.items()},
                                     entity=q.entity, target=targets)
        else:
            out = None
        for _, row in mine.iterrows():
            recomputed = None if out is None else out[row["node"]].predicted
            rows.append({
                "query": row["query"], "kind": q.kind, "node": row["node"],
                "stored": row["predicted"], "recomputed": recomputed,
                "match": (recomputed is not None
                          and str(row["predicted"]) == str(recomputed)),
            })
    return pd.DataFrame(rows, columns=["query", "kind", "node", "stored",
                                       "recomputed", "match"])
