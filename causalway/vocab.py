"""The ``cw:`` vocabulary (``http://sdm-causalway.org/``): causal graph, model, queries, answers.

This module is the single source of truth for every term the project writes.
The model has four groups, and each group maps onto exactly one RML mapping:

* **Causal graph** (``ocg_mapping.rml.ttl``) — a ``cw:OntologicalCausalGraph``
  discovered by one ``cw:CausalDiscoveryRun``; its ``cw:CausalNode``s are the
  triples ``(D_p, p, R_p)`` and its ``cw:CausalEdge``s carry the relational path
  a cause travels to reach its effect.
* **Causal model** (``model_mapping.rml.ttl``) — a ``cw:CausalModel`` fitted on
  a graph by one ``cw:ModelFittingRun``, with one ``cw:CausalMechanism`` per node.
* **Queries and answers** (``query_mapping.rml.ttl``) — a conditional,
  interventional or counterfactual query is posed **on a graph**
  (``cw:onGraph``); each of its answers is **evaluated by a model**
  (``cw:evaluatedBy``), one answer per target node.  The query says nothing
  about any model, so one query can carry the answers of several.

Design rules, each of which removed terms an earlier version had:

* **Subtractive.**  Nothing that can be derived from other triples is minted:
  the target set V of a query is the ``cw:onNode`` of its answers; a node's
  kind (data vs object property) is whether ``cw:range`` is a datatype; an
  intra-class (ε) edge is an edge with no ``cw:viaRelationalPath``; a graph's
  nodes are ``?n cw:inGraph ?g``; an average causal effect is two queries.
* **One naming pattern.**  Provenance reads ``<verb>By`` / ``<verb>Time`` on the
  thing produced — ``cw:discoveredBy``/``cw:discoveredTime`` for a graph's run,
  ``cw:fittedBy``/``cw:fittedTime`` for a model's run,
  ``cw:evaluatedBy``/``cw:evaluatedTime`` on an answer.  Values read
  ``<role>Value``: ``cw:setValue`` (do), ``cw:observedValue`` (see),
  ``cw:predictedValue``, ``cw:factualValue``.
* **No foreign alignment.**  PROV-O and ML Schema were dropped on purpose: they
  bought a citation, not a capability anyone used, and every alignment is a
  claim that has to stay true.

**The namespace does not dereference.**  ``http://sdm-causalway.org/`` is not a
domain this project controls, so two artefacts stand in for "resolve the term,
get its definition", both generated from the dictionaries below so neither can
drift: ``causalway/ontology.ttl`` (``python -m causalway.vocab``) and the
service's ``GET /api/vocab``.  The *shape* of valid data — cardinalities,
disjointness, acyclicity — is not here but in ``causalway/shapes.ttl``, which
``tests/test_shapes.py`` runs against real exports.
"""

from __future__ import annotations

import numbers
import os
from typing import Optional

from rdflib import Namespace, RDF, RDFS, Literal, Graph

CW = Namespace("http://sdm-causalway.org/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

__all__ = [
    "CW", "XSD",
    "GRAPH_STEM", "RUN_STEM", "MODEL_STEM", "MODEL_RUN_STEM",
    "QUERY_STEM", "ANSWER_STEM", "INTERVENTION_STEM", "CONDITION_STEM",
    "PARAMETER_COMMENTS", "value_datatype", "kind_from_range",
    "vocabulary", "write_ontology", "ONTOLOGY_PATH",
]

# ---------------------------------------------------------------------- #
# IRI stems.  Everything is minted deterministically from content, so the
# same treatment on the same node is one cw:Intervention resource however many
# queries use it, and an export read back and written again is byte-stable.
#
#   graph/{gid}                          cw:OntologicalCausalGraph
#   graph/{gid}/node/{nodeKey}           cw:CausalNode   (per graph, not global)
#   graph/{gid}/edge/{causeKey}__{effectKey}   cw:CausalEdge
#   run/{gid}  +  run/{gid}/parameter/{name}   cw:CausalDiscoveryRun, cw:Parameter
#   model/{modelId}  +  model/{modelId}/mechanism/{nodeKey}
#   modelrun/{modelId}  +  modelrun/{modelId}/parameter/{name}
#   query/{queryKey}, intervention/{hash}, condition/{hash}
#   answer/{queryKey}/{modelId}/{nodeKey}
# ---------------------------------------------------------------------- #
GRAPH_STEM = "http://sdm-causalway.org/graph/"
RUN_STEM = "http://sdm-causalway.org/run/"
MODEL_STEM = "http://sdm-causalway.org/model/"
MODEL_RUN_STEM = "http://sdm-causalway.org/modelrun/"
QUERY_STEM = "http://sdm-causalway.org/query/"
ANSWER_STEM = "http://sdm-causalway.org/answer/"
INTERVENTION_STEM = "http://sdm-causalway.org/intervention/"
CONDITION_STEM = "http://sdm-causalway.org/condition/"

# ---------------------------------------------------------------------- #
# Causal graph
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES = {
    "OntologicalCausalGraph": "A discovered ontological causal graph G = (N, E_N): a DAG "
                              "over causal nodes (D_p, p, R_p).",
    "CausalDiscoveryRun": "One execution of a causal-discovery method (or of expert "
                          "curation) over one knowledge graph.",
    "Parameter": "One named setting of a run: a seed, a significance level, maxhop, "
                 "topologicalConstraint, and so on.",
    "CausalNode": "A causal node (D_p, p, R_p) of one graph: property p, scoped to the "
                  "domain class D_p and the range R_p.",
    "CausalEdge": "A directed causal edge of one graph, with the relational path along "
                  "which the cause reaches the effect.",
}
_VOCAB_PROPERTIES = {
    "discoveredBy": "The cw:CausalDiscoveryRun that produced this graph.",
    "sourceKG": "The knowledge graph a run read: a file name or a SPARQL endpoint.",
    "byMethod": "The method a run executed: GES, PC, NOTEARS, ... for discovery, "
                "'curated' for an expert-assembled graph, dowhy.gcm or pgmpy for fitting.",
    "discoveredTime": "When the discovery run finished.",
    "hasParameter": "A cw:Parameter of this run.",
    "parameterName": "Name of a parameter, e.g. 'seed' or 'maxhop'.",
    "parameterValue": "Value of a parameter, as a typed literal (xsd:integer, xsd:double, "
                      "xsd:boolean or a plain string).",
    "inGraph": "The graph a node or an edge belongs to (exactly one).",
    "domain": "Domain class D_p of a causal node. Deliberately not rdfs:domain, whose "
              "entailment would make the node an rdf:Property and intersect domains globally.",
    "property": "The property p of a causal node.",
    "range": "Range R_p of a causal node: a class for an object property, a datatype for a "
             "data property. Deliberately not rdfs:range, for the reason given at cw:domain.",
    "cause": "The causal node an edge starts from.",
    "effect": "The causal node an edge points to.",
    "viaRelationalPath": "The relational path from the cause's domain to the effect's "
                         "domain, as a SPARQL 1.1 property path: '<p>' a forward hop, "
                         "'^<p>' an inverse hop, '/' a sequence. Absent on an intra-class "
                         "(epsilon) edge, whose cause and effect belong to one entity.",
    "curated": "True iff an expert reviewed the edge and kept it (or added it).",
    "weight": "Method-assigned strength of the edge (a coefficient or a score). In a "
              "curated graph, the fraction of discovery runs that proposed the edge. "
              "Absent on an edge an expert added by hand.",
}

# ---------------------------------------------------------------------- #
# Causal model
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES.update({
    "CausalModel": "A causal model fitted on one causal graph: a structural causal model "
                   "(one mechanism per node) or a causal Bayesian network (one CPT per node).",
    "ModelFittingRun": "One execution of model fitting that produced a cw:CausalModel.",
    "CausalMechanism": "The mechanism a structural causal model assigned to one node.",
})
_VOCAB_PROPERTIES.update({
    "onGraph": "The causal graph a query is posed on, or a model is fitted on.",
    "fittedBy": "The cw:ModelFittingRun that produced this model.",
    "fittedTime": "When the fitting run finished.",
    "artifactChecksum": "sha256 of the saved model file (scm.pkl), so a reader can check "
                        "that the artifact at hand is the one that produced an answer.",
    "hasMechanism": "A cw:CausalMechanism of this model, one per node.",
    "mechanismType": "Class name of the mechanism, e.g. InvertibleClassifierFCM.",
    "invertible": "Whether the mechanism's noise can be abducted from an observation — "
                  "i.e. whether a counterfactual on this node is defined.",
})

# ---------------------------------------------------------------------- #
# Queries and answers
# ---------------------------------------------------------------------- #
_VOCAB_CLASSES.update({
    "ConditionalQuery": "P(V | Z = z): what the targets are observed to be in the "
                        "(sub-)population selected by the conditions.",
    "InterventionalQuery": "P(V | do(X = x), Z = z): what the targets would be in the "
                           "(sub-)population if X were set to x.",
    "CounterfactualQuery": "P(V_{do(X = x)} | e): what the targets would have been for one "
                           "unit e — a named entity or a hand-described unit — had X been x.",
    "Intervention": "do(X = x): one node set to one value.",
    "Condition": "Z = z: one node observed at one value.",
    "ConditionalAnswer": "The answer to a cw:ConditionalQuery for one target node.",
    "InterventionalAnswer": "The answer to a cw:InterventionalQuery for one target node.",
    "CounterfactualAnswer": "The answer to a cw:CounterfactualQuery for one target node.",
})
_VOCAB_PROPERTIES.update({
    "hasIntervention": "An intervention of the query (the set X; several = a joint do()).",
    "hasCondition": "A condition of the query (the set Z). On a counterfactual, the "
                    "observed state of a hand-described unit.",
    "aboutEntity": "Counterfactual only: the individual of the source KG the query is about.",
    "onNode": "The causal node an intervention, condition, answer or mechanism is about.",
    "onProperty": "Shortcut: the cw:property of cw:onNode.",
    "setValue": "The value an intervention sets its node to.",
    "observedValue": "The value a condition observes.",
    "forQuery": "The query this answer answers.",
    "evaluatedBy": "The cw:CausalModel that computed this answer.",
    "evaluatedTime": "When this answer was computed.",
    "predictedValue": "The answer: the most probable value of a categorical node, the "
                      "mean of a continuous one.",
    "probability": "Categorical nodes only: the probability of cw:predictedValue.",
    "std": "Continuous nodes only: the standard deviation around cw:predictedValue.",
    "factualValue": "Counterfactual only: what the node actually was for the unit.",
})

#: What each run parameter the project writes means — exported as
#: ``rdfs:comment`` on the ``cw:Parameter``, so a reader of the RDF does not
#: need this repository to know what ``maxhop`` does.  Quote-free on purpose:
#: SDM-RDFizer rewrites quote characters in literals irreversibly.
PARAMETER_COMMENTS = {
    "seed": "Random seed of the run; fixes tie-breaking, initialisation and sampling.",
    "alpha": "Significance level of the conditional-independence tests.",
    "maxhop": "Maximum number of object-property hops a relational path may take for "
              "an effect to pass between two classes.",
    "topologicalConstraint": "Whether Assumption 1 was enforced: an edge is allowed only "
                             "between nodes linked by a relational path of at most maxhop hops.",
    "priors": "Source of the LLM edge priors, if any were used.",
    "library": "Library that implements the discovery method.",
    "nRuns": "Number of discovery runs the curated graph was assembled from.",
    "nManual": "Number of edges an expert added by hand.",
    "orientedEdges": "Number of undirected (CPDAG) edges whose direction was chosen within "
                     "the Markov equivalence class, adding no v-structure and no cycle.",
    "orientationSeed": "Seed of the choice among Markov-equivalent orientations.",
    "trainingRows": "Number of flat-join rows the model was fitted on.",
    "libraryVersion": "Version of the modelling library.",
    "quality": "dowhy.gcm model-selection quality preset.",
    "categoricalCounterfactual": "How categorical counterfactuals are coupled to the "
                                 "factual world: gumbel-max, or ordinal.",
}


# ---------------------------------------------------------------------- #
# Value typing.  One function, so every export that writes a causal *value*
# types it the same way.
# ---------------------------------------------------------------------- #
def value_datatype(value, dtype: Optional[str] = None) -> Optional[str]:
    """The XSD datatype IRI ``value`` should carry, or ``None`` for a plain literal.

    The source of truth is ``dtype`` — the fitted model's own record of what the
    column *became* (``manifest["dtypes"]``), not the ontology's ``R_p``.  The
    two disagree exactly when it matters: a property declared ``xsd:double`` in
    the T-Box but discretised into bins holds ``"(10, 20]"`` at prediction time,
    and stamping that with ``^^xsd:double`` would produce an ill-typed literal
    that any validating consumer rejects.  So a column the model treats as
    categorical or ordinal is a plain literal whatever the T-Box claimed.

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


#: Datatype IRIs outside the XSD namespace that still mark a data property.
_RDF_DATATYPES = {
    str(RDFS.Literal),
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#HTML",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON",
}


def kind_from_range(range_) -> str:
    """``"data"`` if ``R_p`` is a datatype, else ``"object"`` — the rule that retired ``cw:nodeKind``.

    Known limitation: a *custom* datatype outside the XSD and RDF namespaces
    reads as a class.  Neither sample KG, nor any endpoint tried so far, declares
    one; if one appears, the fix is here and nowhere else.
    """
    text = str(range_)
    return "data" if text.startswith(str(XSD)) or text in _RDF_DATATYPES else "object"


def vocabulary(graph: Optional[Graph] = None) -> Graph:
    """The full ``cw:`` schema — one ``rdfs:Class`` / ``rdf:Property`` per term, labelled and commented.

    Deliberately carries no ``rdfs:domain``/``rdfs:range`` axioms: ``cw:onNode``
    is used by four classes and ``cw:hasParameter`` by two, and RDFS reads
    several domains as their *intersection*.  What each class may carry is
    stated in ``causalway/shapes.ttl`` instead, as SHACL, where it is checked
    rather than entailed.
    """
    g = Graph() if graph is None else graph
    g.bind("cw", CW)
    g.bind("rdfs", RDFS)
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
    return g


#: The committed snapshot of :func:`vocabulary`, beside this module.
ONTOLOGY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology.ttl")


def write_ontology(path: Optional[str] = None) -> str:
    """Write :func:`vocabulary` to ``path`` (default :data:`ONTOLOGY_PATH`).

    The snapshot exists only because the namespace does not dereference; it is
    *generated*, never hand-edited, so it cannot drift from this module.
    """
    target = path or ONTOLOGY_PATH
    vocabulary().serialize(destination=target, format="turtle")
    return target


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    print(write_ontology())
