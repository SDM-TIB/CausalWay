"""Causal queries and their answers → JSON → RDF, through SDM-RDFizer.

Why this module exists
----------------------
``queries.store_query`` built the same triples with thirty-odd ``g.add(...)``
calls.  That predated the RML export and was the last place in the project
where the *shape* of an exported graph lived in Python rather than in a
mapping file, which meant "change the export, change the mapping" was true of
the OCG and of counterfactual worlds but not of queries.  This module closes
that gap: :func:`queries_to_json` writes the document, ``query_mapping.rml.ttl``
writes the triples, and nothing here adds one.

Three shape decisions are forced by the engine and are the same ones
``cf_mapping.rml.ttl`` had to make — see that file's header and
``ocg_mapping.rml.ttl``'s for the evidence:

* The document is **denormalised**; no triples map joins onto another, because
  SDM-RDFizer 4.7.5 mishandles ``rr:parentTriplesMap`` onto a blank-node
  parent.
* A value that is an IRI travels in its own ``*_object`` array, and a value
  that carries a datatype in its own ``*_typed`` array, because the engine
  cannot switch term type or datatype per row.
* An absent field **drops its key** rather than holding an empty string: an
  empty ``rml:reference`` produces the literal ``"None"``, not "no triple".

One decision is *not* inherited.  ``rdf:type`` is emitted through an object map
(``rr:template "{class_iri}"`` + ``rr:termType rr:IRI``) rather than ``rr:class``,
because a query's class is one of three and an estimate's is one of three, and
``rr:class`` is static per triples map.  Carrying the class as data collapses
what would otherwise be six near-identical triples maps — multiplied again by
the literal/typed/object split — into two.

Known limitation, inherited from :class:`~causalway.queries.Answer`
-------------------------------------------------------------------
``Answer.effect`` is either a scalar ACE or a per-level contrast dict.  Only a
scalar reaches RDF, and for a dict the component belonging to the *reported*
``cw:predictedValue`` level is the one exported.  This is the same subtraction
that retired ``cw:CategoricalDistribution`` (see ``vocab.py``): the whole
per-level table is 1 + k extra subjects per answer to express something the
reader asked one question about.  It is also what keeps the document free of
the JSON literals SDM-RDFizer would corrupt — it rewrites both quote characters
through ``rml:reference``, irreversibly.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from rdflib import Graph

# These four encode SDM-RDFizer's observed behaviour (absent-vs-empty, what may
# become an IRI). They are single-sourced in cf_export rather than restated here
# so the two exports cannot drift into disagreeing about what an absent value is.
from .cf_export import _as_object, _is_iri, _node_index, _text
from .queries import (
    _ESTIMATE_CLASS,
    _QUERY_CLASS,
    Answer,
    Query,
    answer_iri,
    condition_iri,
    intervention_iri,
    query_iri,
)
from .rdfizer import check_quote_free, materialise
from .result import OntologicalCausalGraph
from .vocab import (
    ANSWER_STEM,
    CONDITION_STEM,
    CW,
    INTERVENTION_STEM,
    PROV,
    QUERY_STEM,
    value_datatype,
    vocabulary,
)

__all__ = ["queries_to_json", "queries_to_rdf", "QUERY_MAPPING_PATH"]

#: The RML mapping that defines the JSON -> RDF contract for a query + answer.
QUERY_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "query_mapping.rml.ttl"
)

#: JSON fields carrying an IRI, exempt from the engine's quote rewriting.
_IRI_FIELDS = {
    "query_iri", "estimate_iri", "model_iri", "entity", "node_iri", "property",
    "intervention_iri", "condition_iri", "value_iri", "class_iri", "datatype",
    "domain",
}


def _key(iri: str, stem: str) -> str:
    """The slug-safe tail of a deterministically minted IRI.

    Subject templates spell their stem out literally and interpolate only this,
    because a template subject is percent-encoded and re-based against
    ``http://example.com/base/`` when it is handed a whole IRI.
    """
    return iri[len(stem):] if iri.startswith(stem) else iri


def _node_row(ocg: OntologicalCausalGraph, index: dict, name: str) -> Optional[dict]:
    """The ``cw:PropertyNode`` description a query's own triples hang off.

    ``store_query`` wrote these for the same reason: a query graph references
    node IRIs, and without at least a label it is not self-describing, so
    ``load_queries`` could not replay it from the Turtle alone.
    """
    if name not in index:
        return None
    node = ocg.nodes[index[name]]
    row = {
        "node_iri": str(ocg.node_iri(index[name])),
        "property": str(node.prop),
        "node_kind": node.kind,
        "label": node.name,
    }
    domain = _text(getattr(node, "domain", None))
    if domain and _is_iri(domain):
        row["domain"] = domain
    return row


def _value_rows(base: dict, node, value: Any, dtypes: dict, field: str) -> tuple:
    """Split one value into (plain, typed, object) — the engine cannot switch per row.

    Returns a 3-tuple of optional rows; exactly one is non-None, or all three
    are when the value is absent, in which case no ``cw:setValue`` /
    ``cw:observedValue`` triple should exist at all.
    """
    text = _text(value)
    if text is None:
        return None, None, None
    if _as_object(node, text):
        return None, None, {**base, "value_iri": text}
    datatype = value_datatype(value, dtypes.get(getattr(node, "name", "")))
    if datatype:
        return None, {**base, field: text, "datatype": str(datatype)}, None
    return {**base, field: text}, None, None


def _effect_scalar(answer: Answer) -> Optional[float]:
    """The one contrast that reaches RDF — see this module's docstring."""
    effect = answer.effect
    if effect is None:
        return None
    if isinstance(effect, dict):
        effect = effect.get(str(answer.predicted), effect.get(answer.predicted))
        if effect is None:
            return None
    try:
        return float(effect)
    except (TypeError, ValueError):
        return None


def queries_to_json(pairs: Iterable[tuple], model, *,
                    dtypes: Optional[dict] = None) -> dict:
    """Render ``(query, answer)`` pairs as the flat document the mapping reads.

    ``model`` supplies ``cw:usedModel`` and, through ``model.spec.ocg``, the
    ``cw:PropertyNode`` resources the query's terms point at.  ``dtypes`` is the
    fitted model's ``{column: "categorical" | "ordinal" | ...}`` record; without
    it a discretised integer types itself as a number, which is wrong in exactly
    the way ``cf_export`` documents.
    """
    ocg = getattr(getattr(model, "spec", None), "ocg", None)
    if ocg is None:
        raise ValueError(
            "queries_to_json: model.spec.ocg is unavailable (a loaded model needs "
            "its OntologicalCausalGraph passed separately — see CausalModel.load)."
        )
    dtypes = dtypes or {}
    index = _node_index(ocg)

    doc: dict = {k: [] for k in (
        "queries", "queries_named", "targets", "nodes",
        "interventions", "interventions_typed", "interventions_object",
        "interventions_expr",
        "query_has_intervention", "query_has_reference",
        "conditions", "conditions_typed", "conditions_object",
        "query_has_evidence", "query_has_condition",
        "estimates", "estimates_typed", "estimates_object",
        "estimate_effects",
    )}
    seen_nodes: set = set()

    def node_of(name: str):
        return ocg.nodes[index[name]] if name in index else None

    def describe(name: str) -> Optional[dict]:
        row = _node_row(ocg, index, name)
        if row and row["node_iri"] not in seen_nodes:
            seen_nodes.add(row["node_iri"])
            doc["nodes"].append(row)
        return row

    for query, answer in pairs:
        q_iri = str(query_iri(query))
        q_key = _key(q_iri, QUERY_STEM)
        computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        row = {
            "query_key": q_key,
            "query_iri": q_iri,
            "class_iri": str(_QUERY_CLASS[query.kind]),
            "model_iri": str(model.iri),
            "computed_at": computed_at,
        }
        if query.label:
            row["label"] = query.label
        if query.entity:
            row["entity"] = str(query.entity)
            # A reference subject whose key is missing crashes the engine rather
            # than yielding no triple, so the entity-subject rows are selected
            # here in Python — the same guard cf_export needs.
            doc["queries_named"].append(
                {"entity": str(query.entity), "query_iri": q_iri}
            )
        doc["queries"].append(row)

        for name in query.target:
            described = describe(name)
            if described:
                doc["targets"].append({
                    "query_key": q_key,
                    "node_iri": described["node_iri"],
                    "property": described["property"],
                })

        for acts, link in ((query.interventions, "query_has_intervention"),
                           (query.reference, "query_has_reference")):
            for act in acts:
                described = describe(act.node)
                if not described:
                    continue
                iv_iri = str(intervention_iri(act))
                base = {
                    "intervention_key": _key(iv_iri, INTERVENTION_STEM),
                    "intervention_iri": iv_iri,
                    "node_iri": described["node_iri"],
                    "property": described["property"],
                }
                if act.entity:
                    base["entity"] = str(act.entity)
                doc[link].append({"query_key": q_key, "intervention_iri": iv_iri})
                if act.expression is not None:
                    doc["interventions_expr"].append(
                        {**base, "expression": str(act.expression)}
                    )
                    continue
                plain, typed, obj = _value_rows(
                    base, node_of(act.node), act.value, dtypes, "value"
                )
                for bucket, r in (("interventions", plain),
                                  ("interventions_typed", typed),
                                  ("interventions_object", obj)):
                    if r:
                        doc[bucket].append(r)

        for conds, link in ((query.evidence, "query_has_evidence"),
                            (query.condition_on, "query_has_condition")):
            for cond in conds:
                described = describe(cond.node)
                if not described:
                    continue
                c_iri = str(condition_iri(cond))
                base = {
                    "condition_key": _key(c_iri, CONDITION_STEM),
                    "condition_iri": c_iri,
                    "node_iri": described["node_iri"],
                    "property": described["property"],
                }
                doc[link].append({"query_key": q_key, "condition_iri": c_iri})
                plain, typed, obj = _value_rows(
                    base, node_of(cond.node), cond.value, dtypes, "observed_value"
                )
                for bucket, r in (("conditions", plain),
                                  ("conditions_typed", typed),
                                  ("conditions_object", obj)):
                    if r:
                        doc[bucket].append(r)

        # The estimate answers the query's *first* target; a multi-target query
        # produces one Answer per target upstream (decision 17), each arriving
        # here as its own pair.
        if not query.target:
            continue
        described = describe(query.target[0])
        if not described:
            continue
        a_iri = str(answer_iri(query))
        a_key = _key(a_iri, ANSWER_STEM)
        base = {
            "estimate_key": a_key,
            "estimate_iri": a_iri,
            "class_iri": str(_ESTIMATE_CLASS[query.kind]),
            "query_iri": q_iri,
            "node_iri": described["node_iri"],
            "property": described["property"],
        }
        if query.entity:
            base["entity"] = str(query.entity)
        plain, typed, obj = _value_rows(
            base, node_of(query.target[0]), answer.predicted, dtypes, "value"
        )
        for bucket, r in (("estimates", plain),
                          ("estimates_typed", typed),
                          ("estimates_object", obj)):
            if r:
                doc[bucket].append(r)
        effect = _effect_scalar(answer)
        if effect is not None:
            doc["estimate_effects"].append(
                {"estimate_key": a_key, "effect": repr(effect)}
            )

    return doc


def queries_to_rdf(pairs: Iterable[tuple], model, *,
                   dtypes: Optional[dict] = None,
                   graph: Optional[Graph] = None,
                   with_vocabulary: bool = False,
                   workdir: Optional[str] = None) -> Graph:
    """Serialise ``(query, answer)`` pairs to RDF through SDM-RDFizer.

    No triple is written by this function.  The result is in the §6.1
    *annotation* form — it never asserts ``entity property value``, so merging
    it into the source KG contradicts nothing there.  The §6.8 named-graph
    projection, which deliberately *does* assert that, is not an export shape
    and stays in :func:`causalway.queries.store_query`.
    """
    g = Graph() if graph is None else graph
    g.bind("cw", CW)
    g.bind("prov", PROV)
    if with_vocabulary:
        vocabulary(g)

    payload = queries_to_json(pairs, model, dtypes=dtypes)
    check_quote_free(payload, _IRI_FIELDS)
    materialise(payload, QUERY_MAPPING_PATH, source_name="query.json",
                graph=g, workdir=workdir)
    return g
