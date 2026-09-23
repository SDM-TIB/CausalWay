"""``Query`` / ``Answer`` dataclasses, RDF (de)serialisation, and reach validation.

Implements Plan 2 §5.4 and §6: the query IRI *is* the hypothetical world
(§6.0 principle 1), so ``store_query`` never mints a separate world resource;
``Answer`` stays a rich Python object (backend/ESS/per_row diagnostics) while
its RDF projection stays minimal (§6.0 principle 4) — one
``cw:predictedValue`` term for both modalities (§6.0 principle 3).

**Known inconsistency: this module still writes untyped literals.**
:mod:`causalway.cf_export` types every exported value — a continuous prediction
comes out as ``"111.017"^^xsd:double``, an object-valued categorical as an
``rr:IRI`` term — using :func:`causalway.vocab.value_datatype` and the *fitted*
dtype of the column. ``store_query`` below does not: every ``cw:predictedValue``
and ``cw:hasValue`` it writes is a plain literal, so ``211.24`` exported from
here is a string and cannot be compared, aggregated or filtered numerically by
any consumer of the RDF.

This is recorded rather than fixed because the fix is not local. ``store_query``
takes an :class:`Answer`, which carries no dtype map — the one authority on
whether a column is continuous, discrete or a discretised set of labels is the
fitted model's manifest, and the query log deliberately does not depend on a
model (``load_queries``/``replay`` work against a graph alone). Threading dtypes
in would either add a required argument to a function several callers already
use, or have this module guess a datatype from the value's *text*, which is the
one thing :func:`~causalway.vocab.value_datatype` is careful not to do for
categorical columns: a discretised ``age`` whose levels are ``"1"``/``"2"``/``"3"``
would silently become ``xsd:integer`` and acquire an ordering the model does not
have.

So the split stands: counterfactual worlds — the artefact that describes named
entities of the source KG and is meant to be merged back into it — are typed,
and the query log is not. A reader comparing the two exports will see the
difference; this paragraph is why.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from .bgp import Materialization
from .entities import rows_for_entity
from .result import OntologicalCausalGraph
from .vocab import ANSWER_STEM, CW, CONDITION_STEM, INTERVENTION_STEM, PROV, QUERY_STEM

__all__ = [
    "Intervention", "Condition", "Query", "Answer",
    "validate_query", "store_query", "load_queries", "replay",
]


# ---------------------------------------------------------------------- #
# Dataclasses
# ---------------------------------------------------------------------- #
@dataclass
class Intervention:
    """``do(node := value)`` (atomic) or ``do(node := expression)`` (soft; informational only)."""

    node: str                          # PropertyNode.name / SCM column
    value: Any = None
    expression: Optional[str] = None   # source text of a soft-intervention callable, if any
    entity: Optional[str] = None       # onEntity (str(URIRef)); None => population-level

    def as_callable(self):
        if self.expression is not None:
            raise ValueError(
                f"Intervention on {self.node!r} carries a soft-intervention expression "
                f"({self.expression!r}) that was not reconstructed into a callable; "
                "pass fn= explicitly to the inference call instead of relying on replay."
            )
        v = self.value
        return lambda _: v


@dataclass
class Condition:
    """An observed-value selector: evidence (conditional) or a sub-population filter."""

    node: str
    value: Any


@dataclass
class Query:
    """A causal query. ``kind`` entails the level (Plan 2 §5.2) — never declared separately."""

    kind: str                                    # "conditional" | "interventional" | "counterfactual"
    model_id: str
    target: list = field(default_factory=list)   # 1..n node names
    interventions: list = field(default_factory=list)   # list[Intervention]
    reference: list = field(default_factory=list)        # list[Intervention]; interventional only
    evidence: list = field(default_factory=list)         # list[Condition]; conditional only
    condition_on: list = field(default_factory=list)     # list[Condition]; sub-population (CATE)
    entity: Optional[str] = None                  # aboutEntity; counterfactual only
    qid: Optional[str] = None                     # override the deterministic id
    label: Optional[str] = None

    def __post_init__(self):
        if self.kind == "counterfactual" and not self.entity:
            raise ValueError("Query(kind='counterfactual') requires entity=")
        if self.kind != "counterfactual" and self.entity:
            raise ValueError(f"Query(kind={self.kind!r}) must not set entity= (population level)")


@dataclass
class Answer:
    """The answer to a :class:`Query`. Rich in Python; only a slice becomes RDF (§6.0 principle 4)."""

    kind: str
    target: str                          # node name (single-target convenience; see targets for 1..n)
    entity: Optional[str] = None         # set iff kind == "counterfactual"
    predicted: Any = None
    distribution: Optional[dict] = None  # {level: probability} for categorical
    mean: Optional[float] = None
    std: Optional[float] = None
    ci: Optional[tuple] = None
    factual: Optional[Any] = None
    effect: Any = None                   # ACE, or per-level contrast dict
    backend: Optional[str] = None
    n_samples: Optional[int] = None
    ess: Optional[float] = None
    low_confidence: bool = False
    per_row: Optional[pd.DataFrame] = None
    coupling: Optional[str] = None       # counterfactual only: "gumbel-max"


# ---------------------------------------------------------------------- #
# Deterministic IRIs (§6.3: interventions are IRIs, minted from their content,
# so two queries applying the same treatment share one resource)
# ---------------------------------------------------------------------- #
def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def intervention_iri(iv: Intervention) -> URIRef:
    payload = f"{iv.entity or ''}|{iv.node}|{iv.expression or iv.value}"
    return URIRef(INTERVENTION_STEM + _short_hash(payload))


def condition_iri(c: Condition) -> URIRef:
    return URIRef(CONDITION_STEM + _short_hash(f"{c.node}|{c.value}"))


def _query_content_id(query: Query) -> str:
    payload = json.dumps({
        "model": query.model_id, "kind": query.kind, "target": sorted(query.target),
        "interventions": sorted(str(intervention_iri(iv)) for iv in query.interventions),
        "reference": sorted(str(intervention_iri(iv)) for iv in query.reference),
        "evidence": sorted(f"{c.node}={c.value}" for c in query.evidence),
        "condition_on": sorted(f"{c.node}={c.value}" for c in query.condition_on),
        "entity": query.entity,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def query_iri(query: Query) -> URIRef:
    """The query IRI — which, per §6.0 principle 1, *is* the hypothetical world's IRI."""
    return URIRef(QUERY_STEM + (query.qid or _query_content_id(query)))


def answer_iri(query: Query) -> URIRef:
    return URIRef(ANSWER_STEM + (query.qid or _query_content_id(query)))


# ---------------------------------------------------------------------- #
# §6.5 — validating an intervention's reach
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
                   mat: Optional[Materialization] = None) -> None:
    """§6.5: reject an intervention that cannot reach its target — both checks.

    1. **Causal reach**: a directed path in ``ocg`` from the intervened node
       to every target node (skipped when they coincide).
    2. **Relational reach** (only when an intervention names a *different*
       entity than the query's ``aboutEntity``): the two entities must
       actually co-occur in at least one materialised row, i.e. be joined in
       the source KG — needs ``mat=``.

    Raises ``ValueError`` naming the offending intervention; does not modify
    ``query`` or store anything (neither condition is persisted — both are
    re-derivable from the OCG and the KG, per §6.5).
    """
    for t in query.target:
        t_idx = _node_index(ocg, t)
        for iv in list(query.interventions) + list(query.reference):
            i_idx = _node_index(ocg, iv.node)
            if i_idx != t_idx and not _has_path(ocg, i_idx, t_idx):
                raise ValueError(
                    f"validate_query: no causal path from {iv.node!r} to target {t!r}; "
                    "this intervention cannot reach the target (§6.5 causal reach)."
                )
            if iv.entity and query.entity and str(iv.entity) != str(query.entity):
                if mat is None:
                    raise ValueError(
                        "validate_query: cross-entity intervention on "
                        f"{iv.entity!r} needs the source KG (mat=) to check relational reach."
                    )
                rows_a = set(rows_for_entity(mat, query.entity))
                rows_b = set(rows_for_entity(mat, iv.entity))
                if not (rows_a & rows_b):
                    raise ValueError(
                        f"validate_query: {iv.entity} is not joined to {query.entity} in the "
                        "source KG along any relation (§6.5 relational reach)."
                    )


# ---------------------------------------------------------------------- #
# RDF export (§6.3-6.4, worked examples in §6.7)
# ---------------------------------------------------------------------- #
_QUERY_CLASS = {
    "conditional": CW.ConditionalQuery,
    "interventional": CW.InterventionalQuery,
    "counterfactual": CW.CounterfactualQuery,
}
_ESTIMATE_CLASS = {
    "conditional": CW.ConditionalEstimate,
    "interventional": CW.InterventionalEstimate,
    "counterfactual": CW.CounterfactualEstimate,
}


def _node_iri_and_prop(ocg: OntologicalCausalGraph, name: str):
    idx = _node_index(ocg, name)
    return ocg.node_iri(idx), ocg.nodes[idx].prop, ocg.nodes[idx].range_


# The three helpers that used to live here — `_describe_node`,
# `_add_intervention`, `_add_condition` — wrote the query's triples by hand.
# They are gone: `query_mapping.rml.ttl` writes those triples now, and keeping a
# second implementation of the same shape is exactly how the two drift apart.
# `tests/test_query_rml_export.py` pins the equivalence that retired them.


def store_query(answer: Answer, query: Query, model, *,
                graph: Optional[Graph] = None, merge: str = "annotation") -> Graph:
    """Serialise ``query`` (Layer B) and ``answer`` (Layer C) into ``graph``.

    ``model`` is the fitted :class:`~causalway.model.CausalModel` that
    answered the query (used for ``cw:usedModel`` and to resolve node IRIs
    via ``model.spec.ocg``).

    ``merge``:

    * ``"annotation"`` (default) — the query/intervention/condition/estimate
      resources as plain triples, plus ``entity cw:hasQuery queryIri`` for
      entity-first navigation. Adds no ``entity predicate value`` triple, so
      merging into the source KG contradicts nothing (§6.1).
    * ``"named-graph"`` — additionally projects the target node (and every
      descendant of an intervened node) as materialised triples inside
      ``GRAPH <queryIri> { ... }`` (§6.8; needs a quad store, not plain
      ``.ttl``).

    The annotation triples are **not** written here any more: this function is
    now a thin wrapper over :func:`causalway.query_export.queries_to_rdf`, which
    hands ``query_mapping.rml.ttl`` to SDM-RDFizer like every other export in
    the project.  What stays behind is the one thing that is not an export
    shape — the ``"named-graph"`` projection, which deliberately asserts
    ``entity property value`` quads and so cannot travel through a mapping
    whose whole contract is that it never does.
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
    if model.spec.ocg is None:
        raise ValueError("store_query: model.spec.ocg is unavailable (a loaded model needs "
                         "its OntologicalCausalGraph passed separately — see CausalModel.load).")

    # Imported here, not at module scope: `query_export` imports this module for
    # the dataclasses and the IRI minting, so the dependency only runs one way
    # at import time.
    from .query_export import queries_to_rdf

    queries_to_rdf([(query, answer)], model, graph=g)

    if merge == "named-graph":
        _project_named_graph(g, query_iri(query), answer, query, model)

    return g


def _predicted_literal(value: Any, range_: URIRef) -> Literal:
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Literal(value, datatype=XSD.decimal)
    return Literal(str(value))


def _project_named_graph(g: Graph, q_iri: URIRef, answer: Answer, query: Query, model) -> None:
    """§6.8 named-graph mode: materialise the target's hypothetical value inside ``GRAPH <queryIri>``.

    Open question Q2 (not resolved by the plan) is whether this should
    project the *whole* hypothetical world — every descendant of an
    intervened node, not just the target — which would need a value for each
    of them. ``Answer`` only carries a value for the target node, so this
    implementation projects the target only; extending it to every
    descendant needs the engines in :mod:`causalway.inference` to return a
    full-row prediction, not just the target's.
    """
    if not query.entity:
        return  # needs a subject entity; population-level named graphs aren't meaningful here
    ocg = model.spec.ocg
    _, prop, range_ = _node_iri_and_prop(ocg, query.target[0])
    ctx = g.get_context(q_iri)
    ctx.add((URIRef(query.entity), prop, _predicted_literal(answer.predicted, range_)))


# ---------------------------------------------------------------------- #
# Round trip: RDF -> executable Query
# ---------------------------------------------------------------------- #
def load_queries(source, model=None) -> list:
    """Reconstruct executable :class:`Query` objects from a KG.

    ``source`` is anything :func:`causalway.sources.resolve_graph` accepts.
    """
    from .sources import resolve_graph
    g = resolve_graph(source)

    kind_of = {v: k for k, v in _QUERY_CLASS.items()}
    out = []
    for q_iri, cls in g.subject_objects(RDF.type):
        if cls not in kind_of:
            continue
        kind = kind_of[cls]
        used_model = g.value(q_iri, CW.usedModel)
        model_id = str(used_model).rsplit("/", 1)[-1] if used_model else None
        entity = g.value(q_iri, CW.aboutEntity)
        target = [_local_node_name(g, n) for n in g.objects(q_iri, CW.targetNode)]
        interventions = [_read_intervention(g, iv) for iv in g.objects(q_iri, CW.hasIntervention)]
        reference = [_read_intervention(g, iv) for iv in g.objects(q_iri, CW.referenceIntervention)]
        evidence = [_read_condition(g, c) for c in g.objects(q_iri, CW.hasEvidence)]
        condition_on = [_read_condition(g, c) for c in g.objects(q_iri, CW.hasCondition)]
        qid = str(q_iri).rsplit("/", 1)[-1]
        out.append(Query(
            kind=kind, model_id=model_id, target=target,
            interventions=interventions, reference=reference,
            evidence=evidence, condition_on=condition_on,
            entity=str(entity) if entity else None, qid=qid,
        ))
    return out


def _local_node_name(g: Graph, node_iri: URIRef) -> str:
    label = g.value(node_iri, RDFS.label)
    return str(label) if label is not None else str(node_iri)


def _read_intervention(g: Graph, iri: URIRef) -> Intervention:
    node = _local_node_name(g, g.value(iri, CW.onNode))
    entity = g.value(iri, CW.onEntity)
    expr = g.value(iri, CW.setExpression)
    val = g.value(iri, CW.setValue)
    return Intervention(
        node=node, value=None if val is None else val.toPython(),
        expression=str(expr) if expr is not None else None,
        entity=str(entity) if entity is not None else None,
    )


def _read_condition(g: Graph, iri: URIRef) -> Condition:
    node = _local_node_name(g, g.value(iri, CW.onNode))
    val = g.value(iri, CW.observedValue)
    return Condition(node=node, value=None if val is None else val.toPython())


def replay(source, model) -> pd.DataFrame:
    """Re-execute every :class:`Query` loaded from ``source`` and compare against the stored answer.

    Returns one row per query: ``qid, kind, target, stored, recomputed, match``.
    """
    from .inference import condition as run_conditional
    from .inference import counterfactual as run_counterfactual
    from .inference import intervene as run_interventional

    g_graph = _resolve_for_replay(source)
    queries = load_queries(g_graph, model=model)
    rows = []
    for q in queries:
        q_iri = query_iri(q)
        a_iri = answer_iri(q)
        stored = g_graph.value(a_iri, CW.predictedValue)
        stored_val = stored.toPython() if stored is not None else None

        if q.kind == "conditional":
            evidence = {c.node: c.value for c in q.evidence}
            ans = run_conditional(model, q.target[0], evidence,
                                  conditions={c.node: c.value for c in q.condition_on})
        elif q.kind == "interventional":
            interventions = {iv.node: iv.value for iv in q.interventions}
            ans = run_interventional(model, interventions, target=q.target[0],
                                     reference={iv.node: iv.value for iv in q.reference} or None,
                                     conditions={c.node: c.value for c in q.condition_on})
        else:
            interventions = {(iv.entity or q.entity, iv.node): iv.value for iv in q.interventions}
            ans = run_counterfactual(model, interventions, entity=q.entity, target=q.target[0])

        rows.append({
            "qid": q.qid, "kind": q.kind, "target": q.target[0],
            "stored": stored_val, "recomputed": ans.predicted,
            "match": str(stored_val) == str(ans.predicted),
        })
    return pd.DataFrame(rows, columns=["qid", "kind", "target", "stored", "recomputed", "match"])


def _resolve_for_replay(source):
    if isinstance(source, Graph):
        return source
    from .sources import resolve_graph
    return resolve_graph(source)
