"""Counterfactual *worlds* as JSON, and as RDF via ``cf_mapping.rml.ttl``.

A world is one unit, one joint intervention over one or more causal variables,
and — for every non-intervened node of the model — the value that node would
most likely have taken.  :mod:`causalway.queries` already serialises a single
targeted query and its single answer; this module is the whole-board form,
which is what the counterfactual module of the web service computes and the
only artifact of this project that can describe *named resources of the source
KG*.  That is why it is exported on its own rather than inside the query log: a
population-level conditional or interventional answer is about the model and
cannot be merged back into the KG, while this can.

What a world says, and nothing more
-----------------------------------
The question a counterfactual is actually asked is: *for this unit, if I change
these properties, what would the others most likely be?*  Everything in the
export answers exactly that.  An earlier version also wrote the full outcome
distribution as a ``cw:CategoricalDistribution`` resource with one
``cw:Outcome`` child per level — for a mostly-categorical KG that is ``1 + k``
extra subjects *per node per world*, and it dominated the document while
answering a question nobody asked.  It is gone (see ``_RETIRED_*`` in
:mod:`causalway.vocab`).  What replaced it is one triple: ``cw:probability``,
the mass of the arg-max level that ``cw:predictedValue`` actually reports.

Three things stayed that a naive diet would have cut:

* **``cw:factualValue``.**  "dosage would have been 40" means nothing until
  "dosage was 25" is beside it.  One triple per node, and the single most
  load-bearing one in the document.  The contrast between them is deliberately
  not minted as a term: it is a subtraction.
* **Every non-intervened node**, not just nominated outcomes — a world is
  interesting precisely because it says what *else* would have been different.
* **``cw:rowCount``.**  See below; without it ``cw:standardDeviation`` is
  uninterpretable.

Uncertainty, honestly
---------------------
A categorical node's counterfactual is a genuine draw: observing class ``k``
constrains the Gumbel vector but does not determine it, so
:mod:`causalway.mechanisms` samples the noise *posterior* and the resulting
spread is real.  Those nodes get ``cw:probability``.

A continuous node is the opposite case, and this is the subtlety worth
spelling out.  Its mechanism ``X = f(PA) + N`` is invertible, so abduction from
a fully observed row *solves* for ``N`` exactly — one number, not a posterior.
Its counterfactual therefore varies only when something upstream varies, i.e.
when it has a stochastic (categorical) ancestor whose own counterfactual moves
between draws.  So ``cw:standardDeviation`` is genuine uncertainty for such a
node and identically zero for one without such an ancestor — and a reader
cannot tell which from the number alone.  Worse, when the flat join gave the
unit several rows, the spread *across those rows* is mixed inseparably into the
same figure.  ``cw:rowCount`` is exported beside it so the ambiguity is at
least visible: ``1`` means the spread is purely counterfactual, ``> 1`` means
row duplication is folded in.  Separating those two axes properly is not
implemented.

Datatypes and IRIs
------------------
Values are typed through :func:`causalway.vocab.value_datatype`, whose source of
truth is the *fitted* dtype rather than the ontology's ``R_p`` — a property
declared ``xsd:double`` but discretised into bins holds ``"(10, 20]"``, and
typing that as a double produces a literal no validator accepts.  A node whose
``kind`` is ``"object"`` carries its value as an IRI instead of a literal.
SDM-RDFizer has no ``rml:termTypeMap``, so IRI-valued and literal-valued rows
cannot share one predicate-object map; they travel in separate JSON arrays
(``*_object``) read by separate triples maps.  ``rml:datatypeMap`` *is*
supported, which is what makes per-row literal typing possible at all.

Units without a name
--------------------
A world is normally about a named entity of the source KG, and then it also
writes ``entity cw:hasQuery world`` for entity-first navigation.  The
counterfactual module can also pose a *hypothetical* unit — observed values
supplied by hand, no entity behind them — which is what makes an imported model
usable with no training data attached.  Such a world omits ``cw:aboutEntity``
and ``cw:hasQuery`` entirely (absence is the statement: there is no resource to
point at) and folds its observed values into its own IRI, so two hypothetical
units differing only in what was observed do not collide on one world resource.

Intervention and query IRIs for *named* units are minted by
:mod:`causalway.queries`, not re-derived here, so a world exported through this
module and the same intervention stored through ``store_query`` are the *same*
resources.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from rdflib import Graph

from .queries import Intervention, Query, intervention_iri, query_iri
from .rdfizer import check_quote_free, materialise
from .result import OntologicalCausalGraph
from .vocab import (ANSWER_STEM, CW, MODEL_STEM, PROV, QUERY_STEM, XSD,
                    value_datatype, vocabulary)

__all__ = ["CounterfactualWorld", "worlds_to_json", "worlds_to_rdf",
           "CF_MAPPING_PATH"]

#: The RML mapping that defines the JSON -> RDF contract for a world.
CF_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "cf_mapping.rml.ttl")

#: JSON fields carrying an IRI, exempt from the engine's quote rewriting.
_IRI_FIELDS = {
    "world_iri", "model_iri", "entity", "node_iri", "property",
    "intervention_iri", "estimate_iri", "value_iri", "factual_iri",
    "datatype", "factual_datatype",
}


@dataclass
class CounterfactualWorld:
    """One unit under one joint intervention, with every node's two values.

    ``interventions``, ``factual``, ``counterfactual``, ``distribution`` and
    ``std`` are all keyed by **node name** (the SCM column /
    ``PropertyNode.name``), which :func:`worlds_to_json` resolves against the
    OCG to node IRIs.

    ``entity`` is ``None`` for a hypothetical unit whose observed values were
    supplied by hand rather than read off a named resource.  ``distribution``
    is accepted in full (``{node: {level: probability}}``) but only the mass of
    the reported arg-max level survives into RDF.
    """

    interventions: dict                              # node name -> value set by do()
    counterfactual: dict                             # node name -> value it would have had
    entity: Optional[str] = None                     # None => hypothetical unit
    factual: dict = field(default_factory=dict)      # node name -> value it had
    distribution: dict = field(default_factory=dict) # node name -> {level: probability}
    std: dict = field(default_factory=dict)          # node name -> sd across draws
    row_count: Optional[int] = None                  # flat-join rows the unit occupied
    random_seed: Optional[int] = None                # seed behind the stochastic draws
    model_id: Optional[str] = None
    computed_at: Optional[str] = None                # ISO-8601; filled on export
    coupling: Optional[str] = None                   # "gumbel-max" for a categorical draw
    label: Optional[str] = None

    def as_query(self) -> Query:
        """The :class:`~causalway.queries.Query` whose IRI is this world's IRI.

        Named units only.  ``Query(kind="counterfactual")`` requires an entity
        by construction, which is the right rule for the query log — a
        hypothetical unit is not a query *about* anything in the KG — so a
        world without one mints its IRI through :func:`_hypothetical_qid`
        instead of through this method.
        """
        if not self.entity:
            raise ValueError(
                "CounterfactualWorld.as_query() is for a world about a named entity; "
                "this world is a hypothetical unit (entity=None). Its world IRI comes "
                "from _hypothetical_qid()."
            )
        return Query(
            kind="counterfactual", model_id=self.model_id or "",
            target=sorted(self.counterfactual),
            interventions=[Intervention(node=node, value=value, entity=self.entity)
                           for node, value in sorted(self.interventions.items())],
            entity=self.entity, label=self.label,
        )

    def acts(self) -> list:
        """The world's interventions as :class:`~causalway.queries.Intervention` objects."""
        return [Intervention(node=node, value=value, entity=self.entity)
                for node, value in sorted(self.interventions.items())]


def _hypothetical_qid(world: CounterfactualWorld) -> str:
    """A content id for a world with no named entity.

    The observed values are part of the identity, not just the interventions.
    Two hypothetical units given the same ``do()`` but different observations
    are different worlds with different answers, and hashing only the treatment
    would collapse them onto one resource — silently, and in a way that makes
    the second export overwrite the first's meaning.
    """
    payload = json.dumps({
        "model": world.model_id or "",
        "interventions": {k: str(v) for k, v in sorted(world.interventions.items())},
        "observed": {k: str(v) for k, v in sorted(world.factual.items())},
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _world_iri(world: CounterfactualWorld) -> str:
    if world.entity:
        return str(query_iri(world.as_query()))
    return QUERY_STEM + _hypothetical_qid(world)


def _node_index(ocg: OntologicalCausalGraph) -> dict:
    return {node.name: k for k, node in enumerate(ocg.nodes)}


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

    Both conditions are required, not just the node's kind.  An object-valued
    column only holds a full IRI once the materialisation stops shortening it
    to a local name; until then — and for any value that simply is not an IRI —
    emitting ``rr:termType rr:IRI`` would mint a resource out of a bare word.
    Falling back to a literal keeps the document well-formed and loses nothing
    that was there to begin with.
    """
    return getattr(node, "kind", None) == "object" and _is_iri(value_text)


def _row_datatype(raw_values: list, dtype: Optional[str]) -> Optional[str]:
    """One datatype for every value of a single node, or ``None`` to stay plain.

    A node's counterfactual and factual values are the same variable observed in
    two worlds, so they must carry the *same* datatype; typing one ``xsd:integer``
    and the other ``xsd:double`` would say they are different kinds of quantity.
    Integers are therefore widened to ``xsd:double`` as soon as any value of the
    node is fractional, and any disagreement that widening cannot reconcile
    (``"N/A"`` beside ``3.5``) drops the whole row to a plain literal rather than
    stamping a type onto a value that does not have it.

    This is also why the split is per *row* and not per field: SDM-RDFizer drops
    a value whose ``rml:datatypeMap`` resolves to nothing and drops the entire
    triples map when the key is absent, so a typed map may only ever see rows
    that really are typed.
    """
    present = [v for v in raw_values if v is not None]
    if not present:
        return None
    datatypes = {value_datatype(v, dtype) for v in present}
    if None in datatypes:
        return None
    if datatypes == {str(XSD.integer), str(XSD.double)}:
        return str(XSD.double)
    if len(datatypes) != 1:
        return None
    return datatypes.pop()


def _probability_of(world: CounterfactualWorld, name: str, predicted) -> Optional[float]:
    """The mass the reported arg-max level actually carries, if it is known."""
    levels = world.distribution.get(name) or {}
    if not levels or predicted is None:
        return None
    mass = levels.get(predicted, levels.get(str(predicted)))
    return None if mass is None else float(mass)


def worlds_to_json(worlds, ocg: OntologicalCausalGraph,
                   dtypes: Optional[dict] = None) -> dict:
    """Render counterfactual worlds as the flat document ``cf_mapping.rml.ttl`` reads.

    Denormalised in the same way and for the same reason as
    :meth:`OntologicalCausalGraph.to_json`: every record carries the fields its
    own subject template and object maps need, so no triples map has to join.
    The ``*_object`` arrays carry the rows whose value is an IRI, because the
    engine cannot switch term type per row.

    ``dtypes`` is the fitted model's ``{column: "categorical" | "ordinal" |
    ...}`` record, used to type literals.  Omitted, the datatype is inferred
    from each Python value alone, which is right for a genuine number and wrong
    for a discretised one — pass it whenever a model is at hand.

    Nodes named in a world but absent from ``ocg`` are skipped rather than
    guessed at — a value with no ``cw:PropertyNode`` to hang off has no place
    in the graph.
    """
    index = _node_index(ocg)
    dtypes = dtypes or {}
    world_rows: list = []
    named_world_rows: list = []
    intervention_rows: list = []
    intervention_typed_rows: list = []
    intervention_object_rows: list = []
    estimate_rows: list = []
    estimate_typed_rows: list = []
    estimate_object_rows: list = []

    for world in worlds:
        if world.computed_at is None:
            world.computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        world_iri = _world_iri(world)
        world_key = world_iri[len(QUERY_STEM):]
        treatment = ", ".join(f"do({node} = {value})"
                              for node, value in sorted(world.interventions.items()))

        row = {
            "world_key": world_key,
            "world_iri": world_iri,
            "computed_at": world.computed_at,
            "label": world.label or (
                f"Counterfactual world: {treatment} for "
                + (world.entity if world.entity else "a hypothetical unit")
            ),
        }
        # Absent, not empty: a hypothetical unit has no entity resource, and the
        # missing cw:aboutEntity *is* how the document says so.
        if world.entity:
            row["entity"] = str(world.entity)
        if world.model_id:
            row["model_iri"] = MODEL_STEM + str(world.model_id)
        if world.coupling:
            row["coupling"] = world.coupling
        if world.row_count is not None:
            row["row_count"] = int(world.row_count)
        if world.random_seed is not None:
            row["random_seed"] = int(world.random_seed)
        world_rows.append(row)
        # `<#EntityHasQueryMap>` puts the entity in *subject* position, and a
        # reference subject whose key is missing does not yield "no triple" the
        # way a missing object does — it crashes SDM-RDFizer outright
        # (`subject_value[1:-1]` on None, with no guard, unlike every template
        # branch beside it). So the rows that have an entity are selected here,
        # in Python, rather than left for the engine to skip.
        if world.entity:
            named_world_rows.append(row)

        for act in world.acts():
            if act.node not in index:
                continue
            node = ocg.nodes[index[act.node]]
            value = _text(act.value)
            if value is None:
                continue          # do() with no value is not an act, it is a bug
            iri = str(intervention_iri(act))
            record = {
                "world_key": world_key,
                "world_iri": world_iri,
                "intervention_key": iri.rsplit("/", 1)[-1],
                "intervention_iri": iri,
                "node_iri": str(ocg.node_iri(index[act.node])),
                "property": str(node.prop),
            }
            if world.entity:
                record["entity"] = str(world.entity)
            if _as_object(node, value):
                record["value_iri"] = value
                intervention_object_rows.append(record)
                continue
            record["value"] = value
            datatype = _row_datatype([act.value], dtypes.get(act.node))
            if datatype:
                record["datatype"] = datatype
                intervention_typed_rows.append(record)
            else:
                intervention_rows.append(record)

        for name in sorted(world.counterfactual):
            # An intervened node has no counterfactual to report: its value is
            # the treatment, already stated as cw:setValue on the Intervention.
            if name in world.interventions or name not in index:
                continue
            k = index[name]
            node = ocg.nodes[k]
            node_key = ocg.node_key(k)
            predicted_raw = world.counterfactual.get(name)
            predicted = _text(predicted_raw)
            factual_raw = world.factual.get(name)
            factual = _text(factual_raw)
            record = {
                "world_key": world_key,
                "world_iri": world_iri,
                "node_key": node_key,
                "estimate_iri": f"{ANSWER_STEM}{world_key}/{node_key}",
                "node_iri": str(ocg.node_iri(k)),
                "property": str(node.prop),
            }
            if world.entity:
                record["entity"] = str(world.entity)

            probability = _probability_of(world, name, predicted_raw)
            if probability is not None:
                record["probability"] = float(probability)
            # Continuous only, and only alongside row_count on the world —
            # see this module's docstring for why the pair is inseparable.
            sd = world.std.get(name)
            if sd is not None and dtypes.get(name) not in ("categorical", "ordinal"):
                record["std"] = float(sd)

            if _as_object(node, predicted) or _as_object(node, factual):
                if predicted is not None:
                    record["value_iri"] = predicted
                if factual is not None:
                    record["factual_iri"] = factual
                estimate_object_rows.append(record)
                continue

            if predicted is not None:
                record["value"] = predicted
            if factual is not None:
                record["factual"] = factual
            datatype = _row_datatype([predicted_raw, factual_raw], dtypes.get(name))
            if datatype:
                record["datatype"] = datatype
                estimate_typed_rows.append(record)
            else:
                estimate_rows.append(record)

    # Three arrays per kind, because the engine cannot switch either term type or
    # "typed vs plain" per row: `*_object` values are IRIs, `*_typed` values carry
    # an xsd datatype, the bare array is plain literals. Every row of a `*_typed`
    # array is guaranteed to have a `datatype` key — an absent one takes the whole
    # triples map down with it.
    return {
        "worlds": world_rows,
        "worlds_named": named_world_rows,
        "interventions": intervention_rows,
        "interventions_typed": intervention_typed_rows,
        "interventions_object": intervention_object_rows,
        "estimates": estimate_rows,
        "estimates_typed": estimate_typed_rows,
        "estimates_object": estimate_object_rows,
    }


def worlds_to_rdf(worlds, ocg: OntologicalCausalGraph, *,
                  dtypes: Optional[dict] = None,
                  graph: Optional[Graph] = None,
                  with_vocabulary: bool = False,
                  workdir: Optional[str] = None) -> Graph:
    """Serialise counterfactual worlds to RDF through SDM-RDFizer.

    No triple is written by this function: :func:`worlds_to_json` produces the
    document and ``cf_mapping.rml.ttl`` produces the triples.  The result is in
    the §6.1 *annotation* form — it never asserts ``entity property value``, so
    merging it into the source KG contradicts nothing there.
    """
    g = Graph() if graph is None else graph
    g.bind("cw", CW)
    g.bind("prov", PROV)
    if with_vocabulary:
        vocabulary(g)

    payload = worlds_to_json(worlds, ocg, dtypes=dtypes)
    check_quote_free(payload, _IRI_FIELDS)
    materialise(payload, CF_MAPPING_PATH, source_name="cf.json",
                graph=g, workdir=workdir)
    return g
