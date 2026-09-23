"""The ``cw:`` vocabulary (``http://sdm-causalway.org/``) shared by Plan 1 and Plan 2.

Plan 1 minted a small vocabulary for the discovered ontological causal graph
(``cw:OntologicalCausalGraph``, ``cw:PropertyNode``, ``cw:CausalEdge``, ...)
inline in :mod:`causalway.result`.  Plan 2 roughly triples it with the terms
needed to describe a fitted SCM (Layer A), a hypothetical query (Layer B) and
its answer (Layer C) — see ``Plan2 (causal model & prediction).md`` §6.  This
module is the single source of truth for all of it; :mod:`causalway.result`
re-exports the Plan 1 subset so existing imports keep working.

**The namespace does not dereference.**  ``http://sdm-causalway.org/`` is not a
domain this project controls, so the usual linked-data contract — resolve the
term IRI, get the schema — cannot be honoured, and the foreign alignments below
would otherwise be links into nothing.  Two things stand in for it, both
generated from the dictionaries in this module so neither can drift from the
terms it describes: ``causalway/ontology.ttl`` (a committed snapshot, written by
``python -m causalway.vocab``) and the service's ``GET /vocab``.  Point the
domain at the latter and the contract is honoured with no code change.
"""

from __future__ import annotations

import numbers
import os
from typing import Optional

from rdflib import Namespace, RDF, RDFS, Literal, Graph

CW = Namespace("http://sdm-causalway.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")

#: ML Schema — the one foreign vocabulary this project aligns *to*.  The rule is
#: deliberately narrow: align only where the semantics correspond exactly, never
#: to gain a citation.  Two axioms survive that test (see :func:`vocabulary`),
#: both on Layer A, because "this resource is a fitted machine-learning model"
#: and "this resource is one training run" are exactly what ``mls:Model`` and
#: ``mls:Run`` mean.  Everything else was considered and rejected:
#:
#: * ``cw:performance`` -> ``mls:hasQuality`` / ``mls:EvaluationMeasure``: mls
#:   reifies a measure as a resource, cw carries a bare value plus
#:   ``cw:performanceMetric``.  Not the same shape; an alignment would be a lie.
#: * ``cw:usedModel`` -> ``prov:wasInfluencedBy``: prov's term is so broad it
#:   asserts almost nothing, and the derivation-flavoured alternatives
#:   (``prov:wasDerivedFrom``) claim the estimate is a *transformation of* the
#:   model, which it is not.  An estimate reaches its model in one hop through
#:   ``cw:forQuery``/``cw:usedModel``; that is enough.
#: * ``cw:library`` / ``cw:libraryVersion`` / ``cw:trainingRows``: no exact
#:   counterpart in mls.  Left unaligned rather than forced.
MLS = Namespace("http://www.w3.org/ns/mls#")

# ---------------------------------------------------------------------- #
# IRI stems.  Node IRIs are *global* (Plan 1); model/query/answer/
# intervention/condition IRIs are minted deterministically (Plan 2) so that
# two queries applying the same treatment share one intervention resource,
# and so that ``store_query`` / ``load_queries`` round-trip.
# ---------------------------------------------------------------------- #
NODE_STEM = "http://sdm-causalway.org/node/"
GRAPH_STEM = "http://sdm-causalway.org/graph/"
RUN_STEM = "http://sdm-causalway.org/run/"                  # cw:DiscoveryRun (Plan 1)
MODEL_STEM = "http://sdm-causalway.org/model/"
MODEL_RUN_STEM = "http://sdm-causalway.org/modelrun/"       # cw:ModelFitRun (Plan 2)
QUERY_STEM = "http://sdm-causalway.org/query/"
ANSWER_STEM = "http://sdm-causalway.org/answer/"
INTERVENTION_STEM = "http://sdm-causalway.org/intervention/"
CONDITION_STEM = "http://sdm-causalway.org/condition/"

# ---------------------------------------------------------------------- #
# Plan 1 — the discovered ontological causal graph
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES = {
    "OntologicalCausalGraph": "A discovered ontological causal graph G = (N, E_N).",
    "DiscoveryRun": "One execution of a causal-discovery method over one KG.",
    "PropertyNode": "A causal node (D_p, p, R_p): domain class, property, range.",
    "CausalEdge": "A reified causal edge (v1, r, v2) with a method-assigned weight.",
    "NodeSlot": "One position of the column layout of adj: a (graph, index, node) triple.",
    "Parameter": "One named hyper-parameter of a cw:DiscoveryRun.",
}

_VOCAB_PROPERTIES = {
    "hasNode": "Property node belonging to this causal graph "
               "(the cw:hasNodeSlot/cw:slotNode shortcut).",
    "hasNodeSlot": "A cw:NodeSlot of this graph: one column of adj.",
    "slotIndex": "0-based column position of a cw:NodeSlot in adj.",
    "slotNode": "The cw:PropertyNode occupying a cw:NodeSlot.",
    "hasParameter": "A cw:Parameter of this discovery run.",
    "parameterName": "Name of a hyper-parameter, e.g. 'alpha' or 'seed'.",
    "parameterValue": "Value of a hyper-parameter, as text; read with cw:parameterKind.",
    "parameterKind": "'str' | 'int' | 'float' | 'bool' | 'null' | 'json' — how to read cw:parameterValue.",
    "hasEdge": "Causal edge belonging to this causal graph.",
    "inGraph": "The causal graph this edge belongs to.",
    "method": "Name of the causal-discovery method (GES, NOTEARS, ...). On the "
              "cw:DiscoveryRun; reach it from a graph via prov:wasGeneratedBy.",
    "usedTopologicalConstraint": "Whether the run enforced Assumption 1.",
    "sourceKG": "The knowledge graph the data was materialized from.",
    "nodeCount": "Number of property nodes in the graph.",
    "edgeCount": "Number of causal edges in the graph.",
    "domain": "Domain class D_p of a property node.",
    "property": "The property p of a property node.",
    "range": "Range R_p of a property node: a class when cw:nodeKind is 'object', "
             "a datatype when it is 'data'. One term for both, as rdfs:range is — "
             "but deliberately *not* rdfs:range, whose RDFS entailment types its "
             "subject as an rdf:Property and intersects multiple domains globally, "
             "while (D_p, p, R_p) is a local, per-node scoping of p.",
    "nodeKind": "'data' or 'object' — selects the variable type downstream.",
    "cause": "The cw:PropertyNode acting as cause.",
    "effect": "The cw:PropertyNode acting as effect.",
    "weight": "Method-assigned strength of the edge (coefficient or score).",
    "manuallyAdded": "True iff a human added this edge during curation rather than "
                     "a discovery method proposing it.",
    "viaRelation": "Object property traversed by the edge; cw:epsilon intra-class. "
                   "Unordered: the set of properties the edge's join touches.",
    "viaPath": "The edge's relation label as one SPARQL 1.1 property path, e.g. "
               "'^<.../treatedAt>/<.../receives>' — '^' is an inverse traversal and "
               "'/' a sequence. Present only when the ordered, directed form carries "
               "information cw:viaRelation does not: a multi-hop join, or a single "
               "inverse hop. Absent for a plain forward hop and for cw:epsilon.",
}

# ---------------------------------------------------------------------- #
# Legacy Plan 1 terms.  ``OntologicalCausalGraph.from_rdf`` still reads all of
# these so Turtle written before the RML export keeps loading (everything under
# ``results/``), but nothing emits them any more and ``vocabulary()`` does not
# describe them — a schema should document what this version writes.
# ---------------------------------------------------------------------- #
_LEGACY_PROPERTIES = {
    "nodeList": "Ordered rdf:List of the property nodes. Superseded by cw:hasNodeSlot: "
                "RML's collection extension parses but emits nothing.",
    "parameters": "Hyper-parameters of the run as one JSON literal. Superseded by "
                  "cw:hasParameter — SDM-RDFizer cannot carry a quote through a "
                  "literal, and a JSON object is all quotes.",
    "domainClass": "Superseded by cw:domain.",
    "rangeClass": "Superseded by cw:range.",
    "rangeDatatype": "Superseded by cw:range (cw:nodeKind already carries the distinction).",
    "constrained": "Duplicated cw:usedTopologicalConstraint onto the graph. "
                   "Superseded by that term on the cw:DiscoveryRun alone.",
    "relationLabel": "Human-readable rendering of the edge's relation. Superseded by "
                     "rdfs:label (for humans) and cw:viaPath (for machines).",
    "relationDirection": "'forward' | 'inverse'. Superseded by the '^' operator inside cw:viaPath.",
    "pathLength": "Number of hops. Derivable from cw:viaPath.",
    "hop": "A step of a path. Superseded by cw:viaPath.",
    "hopIndex": "1-based position of a hop. Superseded by the order inside cw:viaPath.",
    "hopRelation": "Object property traversed by a hop. Superseded by cw:viaPath.",
    "hopDirection": "'forward' | 'inverse' traversal of a hop. Superseded by cw:viaPath.",
}
_LEGACY_CLASSES = {
    "RelationPath": "An ordered multi-hop relation label, reified. Superseded by the "
                    "cw:viaPath property-path literal.",
    "RelationHop": "One step inside a cw:RelationPath. Superseded by cw:viaPath.",
}

# ---------------------------------------------------------------------- #
# Plan 2, Layer A — the fitted model
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES.update({
    "CausalModel": "A fitted structural causal model (dowhy.gcm SCM) over an OntologicalCausalGraph.",
    "ModelFitRun": "One execution of CausalModel.fit that produced a cw:CausalModel.",
    "MechanismAssignment": "The causal mechanism (model class, invertibility, performance) assigned to one node.",
})
_VOCAB_PROPERTIES.update({
    "overGraph": "The cw:OntologicalCausalGraph this model was fitted on.",
    "trainedOn": "Source knowledge graph the model's data was materialized from.",
    "trainingRows": "Number of flat-join rows used to fit the model.",
    "library": "Modelling library, e.g. 'dowhy.gcm'.",
    "libraryVersion": "Version of the modelling library used to fit the model.",
    "artifact": "Filesystem path of the persisted model artifact (scm.pkl).",
    "hasMechanism": "A cw:MechanismAssignment belonging to this model, one per node.",
    "forNode": "The cw:PropertyNode a mechanism assignment describes.",
    "mechanismType": "Class name of the assigned causal mechanism (e.g. InvertibleClassifierFCM).",
    "predictionModel": "Class name of the underlying prediction model gcm.auto selected.",
    "invertible": "Whether the assigned mechanism is an InvertibleFunctionalCausalModel.",
    "counterfactualCoupling": "Noise coupling used for counterfactuals ('gumbel-max' or 'ordinal').",
    "performance": "Held-out performance value of the mechanism (from gcm.evaluate_causal_model).",
    "performanceMetric": "Name of the metric cw:performance is measured in (F1, R2, ...).",
})

# ---------------------------------------------------------------------- #
# Plan 2, Layer B — the scenario (query IRI == hypothetical world)
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES.update({
    "CausalQuery": "Abstract: a query against a cw:CausalModel. The query IRI is also the world IRI.",
    "ConditionalQuery": "P(target | evidence[, condition]) — (sub-)population.",
    "InterventionalQuery": "P(target | do(intervention)[, condition]) — (sub-)population.",
    "CounterfactualQuery": "Entity-level counterfactual: what target would have been for aboutEntity.",
    "Intervention": "A do(node := value) or do(node := f(.)) act, optionally bound to onEntity.",
    "Condition": "An observed-value selector: evidence (conditional) or sub-population filter (hasCondition).",
})
_VOCAB_PROPERTIES.update({
    "usedModel": "The cw:CausalModel that answered this query.",
    "aboutEntity": "Counterfactual only: the single entity the hypothetical world is about.",
    "targetNode": "The cw:PropertyNode targeted as the outcome (1..n).",
    "targetProperty": "Derived shortcut: the property IRI of cw:targetNode.",
    "hasIntervention": "A cw:Intervention applied in this query (1..n; several = joint intervention).",
    "referenceIntervention": "Interventional only: the baseline arm of a contrast.",
    "hasEvidence": "Conditional only: a cw:Condition read as observed evidence.",
    "hasCondition": "The sub-population selector (CATE), e.g. E[Y | do(t), stage = III].",
    "computedAt": "Timestamp the estimate was computed — the one provenance triple that survives (§6.6).",
    "onNode": "The cw:PropertyNode an Intervention/Estimate targets.",
    "onProperty": "Derived shortcut: the property IRI of cw:onNode.",
    "setValue": "do(): the atomic value an Intervention sets its node to.",
    "setExpression": "do(): a soft-intervention expression (source text) an Intervention applies.",
    "onEntity": "The entity an Intervention (or a CounterfactualEstimate) targets. Present iff entity-level.",
    "observedValue": "The value a Condition observes.",
    "hasQuery": "Entity-first navigation: a cw:CausalQuery this entity is the subject of.",
})

# ---------------------------------------------------------------------- #
# Plan 2, Layer C — the answer
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES.update({
    "CausalEstimate": "Abstract: the answer to a cw:CausalQuery.",
    "ConditionalEstimate": "Answer to a cw:ConditionalQuery.",
    "InterventionalEstimate": "Answer to a cw:InterventionalQuery.",
    "CounterfactualEstimate": "Answer to a cw:CounterfactualQuery.",
    "IntervalEstimate": "Uncertainty hook for a continuous predictedValue: a credible interval.",
})
_VOCAB_PROPERTIES.update({
    "forQuery": "The cw:CausalQuery (== hypothetical world) this estimate answers.",
    "predictedValue": "The answer: one modality-polymorphic value term for both categorical and continuous targets.",
    "factualValue": "Counterfactual only: what cw:onNode actually *was* for cw:onEntity in "
                    "the source KG — the baseline cw:predictedValue is a counterfactual to. "
                    "Aggregated over the entity's rows when the flat join gave it several. "
                    "The contrast is deliberately not minted: it is predictedValue - factualValue.",
    "causalEffect": "Interventional only: contrast against cw:referenceIntervention.",
    "diagnostics": "Optional hook (0..1): a JSON literal of backend/samples/ESS/coupling — run diagnostics, not world facts.",
    "lowerBound": "Lower bound of a cw:IntervalEstimate.",
    "upperBound": "Upper bound of a cw:IntervalEstimate.",
    "credibleLevel": "Credible level (e.g. 0.95) of a cw:IntervalEstimate.",
    "probability": "Probability of the cw:predictedValue actually reported — i.e. of the "
                   "arg-max level, not of some level named elsewhere. Categorical and "
                   "ordinal nodes only: a Gumbel-max counterfactual is a genuine draw from "
                   "the noise posterior, so this number is real. A continuous node's "
                   "counterfactual is a deterministic inversion and carries no probability.",
    "standardDeviation": "Spread of a continuous cw:predictedValue across the draws it was "
                         "summarised from. Read it together with cw:rowCount, which says "
                         "whether that spread can be trusted as uncertainty: see the note on "
                         "cw:rowCount.",
    "rowCount": "How many flat-join rows the unit of this world occupied. 1 means any "
                "cw:standardDeviation is purely counterfactual uncertainty (propagated from "
                "a stochastic categorical ancestor, or 0 when the node has none). Greater "
                "than 1 means the flat join duplicated the unit, and that duplication is "
                "mixed inseparably into the spread — the number is then a lower-confidence "
                "quantity, not a clean posterior sd.",
    "randomSeed": "The seed that produced this world's stochastic draws. Present iff the "
                  "model was fitted with a random_state, which is what makes a Gumbel-max "
                  "counterfactual reproducible rather than merely coupled.",
})

# ---------------------------------------------------------------------- #
# Retired Plan 2 terms.  These were the "uncertainty hook" on a counterfactual
# estimate: a cw:CategoricalDistribution resource with one cw:Outcome child
# per level.  For a KG whose nodes are mostly categorical that is 1 + k extra
# subjects *per node per world*, which dominated the export — while the
# question a counterfactual is actually asked ("what would this entity's other
# properties most likely be?") needs exactly one value and one probability.
# Those now sit directly on the cw:CounterfactualEstimate as cw:predictedValue
# and cw:probability, so the whole distribution sub-tree is gone.
#
# Nothing emits these any more and vocabulary() does not describe them.  They
# are listed for the same reason the Plan 1 legacy terms are: Turtle written by
# an earlier version is still out there, and a reader who meets one of these
# terms deserves to find out what replaced it.
# ---------------------------------------------------------------------- #
_RETIRED_CLASSES = {
    "CategoricalDistribution": "Full outcome distribution of a categorical predictedValue. "
                               "Superseded by cw:probability on the estimate itself, which "
                               "carries the arg-max level's mass and nothing else.",
    "Outcome": "One (value, probability) pair inside a cw:CategoricalDistribution. "
               "Superseded along with it.",
}
_RETIRED_PROPERTIES = {
    "uncertainty": "Hook from an estimate to a cw:IntervalEstimate or "
                   "cw:CategoricalDistribution. The categorical half is superseded by "
                   "cw:probability; no estimate ever carried the interval half.",
    "outcome": "A cw:Outcome belonging to a cw:CategoricalDistribution. Superseded.",
    "outcomeValue": "The class label of a cw:Outcome. Superseded by cw:predictedValue, "
                    "which now names the arg-max level directly.",
}

CW.epsilon  # noqa: B018  (touch the term so it's bound below)


# ---------------------------------------------------------------------- #
# Value typing.  One function, so the two export paths that write a causal
# *value* cannot disagree about how it is typed.
# ---------------------------------------------------------------------- #
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


def value_datatype(value, dtype: Optional[str] = None) -> Optional[str]:
    """The XSD datatype IRI ``value`` should carry, or ``None`` for a plain literal.

    The source of truth is ``dtype`` — the fitted model's own record of what the
    column *became* (``manifest["dtypes"]``), not the ontology's ``R_p``.  The
    two disagree exactly when it matters: a property declared ``xsd:double`` in
    the T-Box but discretised into bins holds ``"(10, 20]"`` at prediction time,
    and stamping that with ``^^xsd:double`` would produce an ill-typed literal
    that any validating consumer rejects.  So a column the model treats as
    categorical or ordinal is a plain literal whatever the T-Box claimed, and a
    transformed value whose original type is no longer recoverable stays a plain
    literal too.

    ``xsd:string`` is never returned.  In RDF 1.1 a plain literal *is* an
    ``xsd:string``, so writing it out is pure noise.
    """
    if dtype in ("categorical", "ordinal"):
        return None
    if isinstance(value, bool):
        return str(XSD.boolean)
    if isinstance(value, numbers.Integral):
        return str(XSD.integer)
    if isinstance(value, numbers.Real):
        return str(XSD.double)
    # A value that arrived as text (SPARQL results, a JSON round-trip) still has
    # a type if it parses as one — but only when the model did not already say
    # the column is categorical, which the guard above has ruled out.
    text = str(value).strip()
    if not text:
        return None
    try:
        int(text)
    except ValueError:
        pass
    else:
        return str(XSD.integer)
    try:
        float(text)
    except ValueError:
        return None
    return str(XSD.double)


#: Alignments to foreign vocabularies, as ``(cw term, predicate, foreign term)``.
#: Kept as data rather than inline triples so the list can be read — and argued
#: with — without reading :func:`vocabulary`.  See the note on :data:`MLS` for
#: why it is this short.
_ALIGNMENTS = [
    (CW.CausalModel, RDFS.subClassOf, MLS.Model),
    (CW.ModelFitRun, RDFS.subClassOf, MLS.Run),
]


def vocabulary(graph: Optional[Graph] = None) -> Graph:
    """The full ``cw:`` schema (Plan 1 + Plan 2) describing every term used above.

    Includes the foreign alignments in :data:`_ALIGNMENTS`, so a consumer that
    already speaks ML Schema can classify a ``cw:CausalModel`` without reading
    a word of this project's documentation.  Retired terms
    (:data:`_RETIRED_CLASSES` / :data:`_RETIRED_PROPERTIES`) are *not* described:
    a schema should document what this version writes.
    """
    g = Graph() if graph is None else graph
    g.bind("cw", CW)
    g.bind("rdfs", RDFS)
    g.bind("mls", MLS)
    for name, comment in _VOCAB_CLASSES.items():
        term = CW[name]
        g.add((term, RDF.type, RDFS.Class))
        g.add((term, RDFS.label, Literal(name)))
        g.add((term, RDFS.comment, Literal(comment)))
    for name, comment in _VOCAB_PROPERTIES.items():
        term = CW[name]
        g.add((term, RDF.type, RDF.Property))
        g.add((term, RDFS.label, Literal(name)))
        g.add((term, RDFS.comment, Literal(comment)))
    for subject, predicate, obj in _ALIGNMENTS:
        g.add((subject, predicate, obj))
    g.add((CW.epsilon, RDFS.label, Literal("ε")))
    g.add((CW.epsilon, RDFS.comment, Literal(
        "Identity relation: an intra-entity (intra-class) causal transition.")))
    return g


#: The committed snapshot of :func:`vocabulary`, beside this module.
ONTOLOGY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology.ttl")


def write_ontology(path: Optional[str] = None) -> str:
    """Write :func:`vocabulary` to ``path`` (default :data:`ONTOLOGY_PATH`).

    The snapshot exists only because the namespace does not dereference; it is
    *generated*, never hand-edited, for the same reason the RML mappings own the
    export shape — a second hand-maintained description of the same vocabulary
    drifts from the first, silently, and then dies.  (That is precisely how the
    project's old ``shapes.ttl`` ended up describing nothing anyone emitted.)
    """
    target = path or ONTOLOGY_PATH
    vocabulary().serialize(destination=target, format="turtle")
    return target


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    print(write_ontology())
