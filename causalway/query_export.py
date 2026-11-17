"""Causal queries and their answers → JSON → RDF, through SDM-RDFizer.

One exporter for all three rungs.  In the cw: model a conditional, an
interventional and a counterfactual query have one shape — a graph, a set of
interventions X, a set of conditions Z, and for a counterfactual about a named
unit its entity — and so do their answers: one per target node, each naming
the model that evaluated it.  :func:`queries_to_json` writes the document,
``query_mapping.rml.ttl`` writes the triples, and nothing here adds one.  (The
counterfactual *world* the service's board produces is a counterfactual query
plus one answer per node; :mod:`causalway.cf_export` adapts it onto this
module rather than keeping a mapping of its own.)

Three shape decisions are forced by the engine — see ``ocg_mapping.rml.ttl``:

* The document is **denormalised**; no triples map joins onto another.
* A value that is an IRI travels in its own ``*_object`` array, and a value
  that carries a datatype in its own ``*_typed`` array, because the engine
  cannot switch term type or datatype per row.
* An absent field **drops its key** rather than holding an empty string: an
  empty ``rml:reference`` produces the literal ``"None"``, not "no triple".

Typing
------
Values are typed through :func:`causalway.vocab.value_datatype`, whose source of
truth is the *fitted* dtype rather than the ontology's ``R_p`` — a property
declared ``xsd:double`` but discretised into bins holds ``"(10, 20]"``, and
typing that as a double produces a literal no validator accepts.  A node whose
range is a class carries an IRI value as a resource.  An answer's predicted and
factual values are the same variable in two worlds, so they share one datatype.

Uncertainty
-----------
A categorical answer carries ``cw:probability`` — the mass of the level it
reports; a continuous one carries ``cw:std``.  Never both.  For a
counterfactual, a categorical node's spread is a genuine Gumbel-max draw from
the noise posterior, while a continuous node's residual is abducted exactly and
varies only through a stochastic categorical ancestor — so its ``cw:std`` is
often 0, and when the flat join gave the unit several rows it also folds in the
spread across those rows.  That last ambiguity is a known limitation of the
compact model, which no longer records the row count.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from rdflib import Graph, RDFS, XSD

from .queries import (
    _ANSWER_CLASS,
    _QUERY_CLASS,
    answers_of,
    condition_iri,
    intervention_iri,
    query_key,
)
from .rdfizer import check_quote_free, materialise
from .result import OntologicalCausalGraph
from .vocab import (
    CONDITION_STEM,
    CW,
    GRAPH_STEM,
    INTERVENTION_STEM,
    MODEL_STEM,
    QUERY_STEM,
    value_datatype,
    vocabulary,
)

__all__ = ["queries_to_json", "queries_to_rdf", "QUERY_MAPPING_PATH"]

#: The RML mapping that defines the JSON -> RDF contract for queries and answers.
QUERY_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "query_mapping.rml.ttl"
)

#: JSON fields carrying an IRI, exempt from the engine's quote rewriting.
_IRI_FIELDS = {
    "graph_iri", "entity", "intervention_iri", "condition_iri", "node_iri",
    "property", "value_iri", "factual_iri", "class_iri", "query_iri", "model_iri",
    "datatype",
}

#: Model dtypes whose answers are a level with a probability, not a mean with an sd.
_CATEGORICAL = ("categorical", "ordinal")


def _text(value) -> Optional[str]:
    """A value as the text the RDF carries, or ``None`` to emit no triple at all.

    An *empty* reference is not treated as absent by SDM-RDFizer — it writes the
    literal ``"None"`` — so an absent value has to drop the key entirely.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text or None


def _is_iri(text: Optional[str]) -> bool:
    """Whether ``text`` is an absolute IRI this export can emit as a resource."""
    if not text:
        return False
    return text.startswith(("http://", "https://", "urn:")) and " " not in text


def _as_object(node, value_text: Optional[str]) -> bool:
    """Whether this node's value travels as an IRI rather than a literal.

    Both conditions are required: an object-property node, *and* a value that
    really is an IRI.  A bare word under ``rr:termType rr:IRI`` would mint a
    resource out of nothing; a literal loses nothing that was there.
    """
    return getattr(node, "kind", None) == "object" and _is_iri(value_text)


def _shared_datatype(raw_values: list, dtype: Optional[str]) -> Optional[str]:
    """One datatype for every value of one node, or ``None`` to stay plain.

    Integers widen to ``xsd:double`` as soon as any value of the node is
    fractional; a disagreement widening cannot reconcile (``"N/A"`` beside
    ``3.5``) drops the row to a plain literal rather than stamping a type onto a
    value that does not have it.
    """
    present = [v for v in raw_values if v is not None]
    if not present:
        return None
    datatypes = {value_datatype(v, dtype) for v in present}
    if None in datatypes:
        return None
    if datatypes == {str(XSD.integer), str(XSD.double)}:
        return str(XSD.double)
    return datatypes.pop() if len(datatypes) == 1 else None


def _split(doc: dict, bucket: str, base: dict, node, raw_values: dict, dtype) -> None:
    """Route one row into ``bucket`` / ``bucket_typed`` / ``bucket_object`` by its values.

    ``raw_values`` maps a JSON field (``value``, ``factual``) to its Python value;
    the row goes to the object array when any of them is an IRI of an object
    node, to the typed array when they share an XSD datatype, else to the plain one.
    """
    texts = {f: _text(v) for f, v in raw_values.items()}
    if any(_as_object(node, t) for t in texts.values()):
        row = dict(base)
        for f, t in texts.items():
            if t is not None:
                row[f"{f}_iri"] = t
        doc[f"{bucket}_object"].append(row)
        return
    row = dict(base)
    for f, t in texts.items():
        if t is not None:
            row[f] = t
    datatype = _shared_datatype(list(raw_values.values()), dtype)
    if datatype:
        row["datatype"] = datatype
        doc[f"{bucket}_typed"].append(row)
    else:
        doc[bucket].append(row)


def _uncertainty(answer, dtype: Optional[str]) -> dict:
    """``{"probability": p}`` for a categorical answer, ``{"std": s}`` for a continuous one."""
    probability, std = answer.probability, answer.std
    categorical = dtype in _CATEGORICAL if dtype else (probability is not None)
    if categorical and probability is not None:
        return {"probability": float(probability)}
    if not categorical and std is not None:
        return {"std": float(std)}
    return {}


def queries_to_json(items: Iterable[tuple], model, *,
                    dtypes: Optional[dict] = None,
                    evaluated_time: Optional[str] = None) -> dict:
    """Render ``(query, answers)`` pairs as the flat document the mapping reads.

    ``answers`` is one :class:`~causalway.queries.Answer`, a list, a
    ``{node: Answer}`` dict, or ``None`` for a query exported without answers.
    ``model`` is the model that evaluated the answers (``cw:evaluatedBy``); its
    graph (``model.spec.ocg``) is the graph the queries are posed on and resolves
    node names to node IRIs.  ``dtypes`` is the fitted
    ``{column: "categorical" | ...}`` record; it defaults to the model's own.

    A node absent from the graph is skipped rather than guessed at, and an
    answer about a node the query intervened on (or, outside a counterfactual,
    conditioned on) raises: V, X and Z are disjoint by definition.
    """
    ocg: Optional[OntologicalCausalGraph] = getattr(getattr(model, "spec", None), "ocg", None)
    if ocg is None:
        raise ValueError(
            "queries_to_json: model.spec.ocg is unavailable (a loaded model needs "
            "its OntologicalCausalGraph — see CausalModel.load)."
        )
    if dtypes is None:
        dtypes = (getattr(model, "manifest", None) or {}).get("dtypes") or {}
    index = {node.name: k for k, node in enumerate(ocg.nodes)}
    gid = ocg.gid
    graph_iri = f"{GRAPH_STEM}{gid}"
    model_key = getattr(model, "model_id", None) or "unfitted"
    model_iri = f"{MODEL_STEM}{model_key}"
    evaluated_time = evaluated_time or datetime.now(timezone.utc).isoformat(timespec="seconds")

    doc: dict = {k: [] for k in (
        "queries", "query_interventions", "query_conditions",
        "interventions", "interventions_typed", "interventions_object",
        "conditions", "conditions_typed", "conditions_object",
        "answers", "answers_typed", "answers_object",
    )}
    seen: set = set()

    # One answer per (query, model, node): asking the same question twice is one
    # query, and its IRI carries one answer per node — the latest one.  Merging
    # here, rather than trusting every caller to, keeps a repeated question from
    # putting two predicted values under one cw:*Answer.
    merged: dict = {}
    for query, answers in items:
        if query.graph_id is None:
            query.graph_id = gid
        elif query.graph_id != gid:
            raise ValueError(
                f"Query {query.label or ''} is posed on graph {query.graph_id!r}, but "
                f"the model evaluating it was fitted on {gid!r}.")
        entry = merged.setdefault(query_key(query), (query, {}))
        for answer in answers_of(answers):
            entry[1][answer.target] = answer

    for query, by_node in merged.values():
        answers = list(by_node.values())
        q_key = query_key(query)
        q_iri = f"{QUERY_STEM}{q_key}"
        if q_iri not in seen:
            seen.add(q_iri)
            row = {"query_key": q_key, "class_iri": str(_QUERY_CLASS[query.kind]),
                   "graph_iri": graph_iri}
            if query.label:
                row["label"] = query.label
            if query.entity:
                row["entity"] = str(query.entity)
            doc["queries"].append(row)

            for acts, link, bucket, mint, stem, key_field, iri_field in (
                (query.interventions, "query_interventions", "interventions",
                 intervention_iri, INTERVENTION_STEM, "intervention_key", "intervention_iri"),
                (query.conditions, "query_conditions", "conditions",
                 condition_iri, CONDITION_STEM, "condition_key", "condition_iri"),
            ):
                for act in acts:
                    if act.node not in index or _text(act.value) is None:
                        continue
                    iri = str(mint(gid, act))
                    doc[link].append({"query_key": q_key, iri_field: iri})
                    if iri in seen:
                        continue
                    seen.add(iri)
                    k = index[act.node]
                    base = {key_field: iri[len(stem):],
                            "node_iri": str(ocg.node_iri(k)),
                            "property": str(ocg.nodes[k].prop)}
                    _split(doc, bucket, base, ocg.nodes[k], {"value": act.value},
                           dtypes.get(act.node))

        excluded = query.excluded_targets()
        for answer in answers_of(answers):
            name = answer.target
            if name in excluded:
                raise ValueError(
                    f"An answer about {name!r} cannot belong to this {query.kind} query: "
                    "the node is " + ("intervened on." if query.kind == "counterfactual"
                                      else "intervened on or conditioned on."))
            if name not in index or _text(answer.predicted) is None:
                continue
            k = index[name]
            node_key = ocg.node_key(k)
            base = {
                "query_key": q_key, "model_key": model_key, "node_key": node_key,
                "class_iri": str(_ANSWER_CLASS[query.kind]),
                "query_iri": q_iri, "model_iri": model_iri,
                "evaluated_time": getattr(answer, "evaluated_time", None) or evaluated_time,
                "node_iri": str(ocg.node_iri(k)),
                "property": str(ocg.nodes[k].prop),
                **_uncertainty(answer, dtypes.get(name)),
            }
            values: dict[str, Any] = {"value": answer.predicted}
            if query.kind == "counterfactual" and answer.factual is not None:
                values["factual"] = answer.factual
            _split(doc, "answers", base, ocg.nodes[k], values, dtypes.get(name))

    return doc


def queries_to_rdf(items: Iterable[tuple], model, *,
                   dtypes: Optional[dict] = None,
                   graph: Optional[Graph] = None,
                   with_vocabulary: bool = False,
                   workdir: Optional[str] = None) -> Graph:
    """Serialise ``(query, answers)`` pairs to RDF through SDM-RDFizer.

    No triple is written by this function.  The result never asserts
    ``entity property value``, so it merges into the source KG without
    contradicting it.  It describes the queries and answers only — export the
    model (:meth:`~causalway.model.CausalModel.to_rdf`) and its graph beside it
    for a self-contained trace.
    """
    g = Graph() if graph is None else graph
    g.bind("cw", CW)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    if with_vocabulary:
        vocabulary(g)

    payload = queries_to_json(items, model, dtypes=dtypes)
    check_quote_free(payload, _IRI_FIELDS)
    materialise(payload, QUERY_MAPPING_PATH, source_name="query.json",
                graph=g, workdir=workdir)
    return g

