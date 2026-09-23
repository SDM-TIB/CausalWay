"""Counterfactual worlds: ``worlds_to_json`` -> ``cf_mapping.rml.ttl`` -> RDF.

The counterfactual export is the one artifact of this project that describes
*named resources of the source KG*, so what is checked here is not only that
the triples come out, but that they come out in a form that could be merged
into that KG without contradicting it (Plan 2 §6.1): the entity IRIs must be
verbatim, and nothing may assert ``entity property value``.

Note the SDM-RDFizer subject-map trap this covers: for a *subject*, a foreign
IRI has to travel as ``rml:reference`` + ``rr:termType rr:IRI``, because
``rr:template "{entity}"`` percent-encodes it and re-bases it under
``http://example.com/base/``. In an *object* map the two idioms swap. Nothing
but an assertion catches that regressing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from rdflib import Literal, RDF, URIRef, XSD

from causalway.cf_export import CounterfactualWorld, worlds_to_json, worlds_to_rdf
from causalway.constraints import EdgeConstraint
from causalway.nodes import build_nodes
from causalway.ontology import OntologySchema
from causalway.result import OntologicalCausalGraph
from causalway.vocab import ANSWER_STEM, CW, QUERY_STEM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC = os.path.join(HERE, "kgs", "ttls", "synthetic_clinic.ttl")

ENTITY = "http://causalkg.example.org/synthetic/hospital_0"


@pytest.fixture(scope="module")
def ocg() -> OntologicalCausalGraph:
    schema = OntologySchema.from_source(CLINIC)
    nodes = build_nodes(schema, include_object_properties=True)
    constraint = EdgeConstraint.from_schema(schema, nodes, max_hops=1)
    n = len(nodes)
    adj = np.zeros((n, n), dtype=int)
    for k, (i, j) in enumerate(sorted(constraint.labels)):
        if k >= 6:
            break
        adj[i, j] = 1
    return OntologicalCausalGraph.from_discovery(
        adj, constraint, method="curated", source=CLINIC, graph_id="cf-test")


def _names(ocg: OntologicalCausalGraph) -> list:
    return [node.name for node in ocg.nodes]


def _world(ocg: OntologicalCausalGraph) -> CounterfactualWorld:
    """One entity under a *joint* intervention over two variables.

    ``a`` and ``b`` are intervened, so they get no estimate — their value is the
    treatment, already stated once as ``cw:setValue``.  ``c`` and ``d`` are the
    outcomes: one categorical (with a distribution behind it) and one continuous
    (with a standard deviation), which is exactly the pair the export types
    differently.
    """
    a, b, c, d = _names(ocg)[:4]
    return CounterfactualWorld(
        entity=ENTITY,
        interventions={a: "High", b: "Medium"},
        counterfactual={a: "High", b: "Medium", c: "East", d: 111.017},
        factual={a: "Low", b: "High", c: "North", d: 99.5},
        distribution={c: {"East": 0.7, "North": 0.0, "West": 0.3}},
        std={d: 3.25}, row_count=1, random_seed=7,
        model_id="m-123", coupling="gumbel-max",
    )


def _dtypes(ocg: OntologicalCausalGraph) -> dict:
    """The fitted dtypes for :func:`_world`: everything categorical but ``d``."""
    names = _names(ocg)
    return {name: "categorical" for name in names} | {names[3]: "continuous"}


# ---------------------------------------------------------------------- #
# The JSON document
# ---------------------------------------------------------------------- #
def test_worlds_to_json_is_denormalised_and_serializable(ocg):
    import json

    payload = worlds_to_json([_world(ocg)], ocg, dtypes=_dtypes(ocg))
    json.dumps(payload)

    assert len(payload["worlds"]) == 1
    assert len(payload["worlds_named"]) == 1        # this world has an entity
    assert len(payload["interventions"]) == 2       # the joint intervention
    # Two outcomes, split by value form: the categorical one is a plain literal,
    # the continuous one carries an xsd datatype, and the engine cannot switch
    # between those per row.
    assert len(payload["estimates"]) == 1
    assert len(payload["estimates_typed"]) == 1
    assert "outcomes" not in payload                # the distribution sub-tree is gone

    key = payload["worlds"][0]["world_key"]
    for section in ("interventions", "estimates", "estimates_typed"):
        for record in payload[section]:
            assert record["world_key"] == key       # no triples map has to join


def test_every_typed_row_carries_a_datatype(ocg):
    """An absent datatype key takes the whole triples map down with it (4.7.5)."""
    payload = worlds_to_json([_world(ocg)], ocg, dtypes=_dtypes(ocg))
    for section in ("interventions_typed", "estimates_typed"):
        for record in payload[section]:
            assert record.get("datatype"), f"{section} row without a datatype: {record}"


def test_an_intervened_node_gets_no_estimate(ocg):
    """Its value is the treatment, not a prediction; restating it would dress an
    assumption up as a result."""
    world = _world(ocg)
    payload = worlds_to_json([world], ocg, dtypes=_dtypes(ocg))
    estimated = {r["node_key"] for r in payload["estimates"] + payload["estimates_typed"]}
    intervened = {r["node_key"] for r in payload["interventions"]
                  if "node_key" in r}
    assert not (estimated & intervened)
    assert len(estimated) == 2                      # c and d, not a and b


def test_world_iri_is_the_query_iri(ocg):
    """Plan 2 §6.0 principle 1: the query identifies the world, so no World class."""
    from causalway.queries import query_iri

    world = _world(ocg)
    payload = worlds_to_json([world], ocg)
    assert payload["worlds"][0]["world_iri"] == str(query_iri(world.as_query()))
    assert payload["worlds"][0]["world_iri"].startswith(QUERY_STEM)


def test_the_same_do_on_the_same_entity_shares_one_intervention_resource(ocg):
    """§6.3: intervention IRIs are minted from content, so they coalesce."""
    a = ocg.nodes[0].name
    one = CounterfactualWorld(entity=ENTITY, interventions={a: "High"},
                              counterfactual={a: "High"}, factual={a: "Low"})
    two = CounterfactualWorld(entity=ENTITY, interventions={a: "High"},
                              counterfactual={a: "High"}, factual={a: "Low"},
                              model_id="other")
    payload = worlds_to_json([one, two], ocg)
    iris = {r["intervention_iri"] for r in payload["interventions"]}
    assert len(iris) == 1


def test_estimate_iris_are_per_node_not_per_query(ocg):
    """A world has n answers, so one answer IRI per query would collide."""
    payload = worlds_to_json([_world(ocg)], ocg, dtypes=_dtypes(ocg))
    records = payload["estimates"] + payload["estimates_typed"]
    iris = {r["estimate_iri"] for r in records}
    assert len(iris) == len(records)
    for record in records:
        assert record["estimate_iri"].startswith(ANSWER_STEM)
        assert record["estimate_iri"].endswith(record["node_key"])


def test_a_node_absent_from_the_graph_is_skipped_not_guessed(ocg):
    world = _world(ocg)
    world.counterfactual["NotANode.atAll"] = "x"
    payload = worlds_to_json([world], ocg, dtypes=_dtypes(ocg))
    records = payload["estimates"] + payload["estimates_typed"]
    assert all("NotANode" not in r["node_key"] for r in records)


# ---------------------------------------------------------------------- #
# The mapping, as applied by SDM-RDFizer
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cf_graph(ocg):
    return worlds_to_rdf([_world(ocg)], ocg, dtypes=_dtypes(ocg))


def test_the_world_is_one_counterfactual_query(cf_graph, ocg):
    from causalway.queries import query_iri

    world_iri = query_iri(_world(ocg).as_query())
    assert (world_iri, RDF.type, CW.CounterfactualQuery) in cf_graph
    assert cf_graph.value(world_iri, CW.aboutEntity) == URIRef(ENTITY)
    assert cf_graph.value(world_iri, CW.computedAt).datatype == XSD.dateTime
    assert str(cf_graph.value(world_iri, CW.counterfactualCoupling)) == "gumbel-max"
    # A joint intervention is several cw:hasIntervention on one world.
    acts = list(cf_graph.objects(world_iri, CW.hasIntervention))
    assert len(acts) == 2
    for act in acts:
        assert (act, RDF.type, CW.Intervention) in cf_graph
        assert cf_graph.value(act, CW.onEntity) == URIRef(ENTITY)
        assert cf_graph.value(act, CW.setValue) is not None


def test_the_entity_iri_is_verbatim_in_subject_position(cf_graph, ocg):
    """The subject-template trap: a re-based IRI would break the merge silently."""
    from causalway.queries import query_iri

    world_iri = query_iri(_world(ocg).as_query())
    assert (URIRef(ENTITY), CW.hasQuery, world_iri) in cf_graph
    assert not [s for s in cf_graph.subjects()
                if str(s).startswith("http://example.com/base/")]


def test_every_non_intervened_node_gets_an_estimate_with_both_values(cf_graph, ocg):
    estimates = list(cf_graph.subjects(RDF.type, CW.CounterfactualEstimate))
    assert len(estimates) == 2                      # c and d; a and b are intervened
    pairs = {}
    for estimate in estimates:
        node = cf_graph.value(estimate, CW.onNode)
        assert cf_graph.value(estimate, CW.onEntity) == URIRef(ENTITY)
        assert cf_graph.value(estimate, CW.onProperty) is not None
        pairs[str(node)] = (
            str(cf_graph.value(estimate, CW.factualValue)),
            str(cf_graph.value(estimate, CW.predictedValue)),
        )
    # A counterfactual with no baseline is unreadable; cw:factualValue exists
    # for exactly this, and is the only term this export had to mint.
    assert all(factual and predicted for factual, predicted in pairs.values())
    assert any(factual != predicted for factual, predicted in pairs.values())


def test_the_distribution_sub_tree_is_gone(cf_graph):
    """1 + k extra subjects per node per world, answering a question nobody asked."""
    assert not list(cf_graph.subjects(RDF.type, CW.CategoricalDistribution))
    assert not list(cf_graph.subjects(RDF.type, CW.Outcome))
    assert not list(cf_graph.subject_objects(CW.uncertainty))
    assert not list(cf_graph.subject_objects(CW.outcomeValue))


def test_a_categorical_estimate_carries_the_arg_max_probability(cf_graph, ocg):
    """What replaced the distribution: the mass of the value actually reported."""
    c = _names(ocg)[2]
    node_iri = ocg.node_iri(_names(ocg).index(c))
    estimate = next(s for s in cf_graph.subjects(RDF.type, CW.CounterfactualEstimate)
                    if cf_graph.value(s, CW.onNode) == node_iri)
    assert str(cf_graph.value(estimate, CW.predictedValue)) == "East"
    probability = cf_graph.value(estimate, CW.probability)
    assert probability.datatype == XSD.double
    assert float(probability) == pytest.approx(0.7)   # P(East), not P(some other level)
    # A categorical counterfactual is a draw, not an inversion: no sd.
    assert cf_graph.value(estimate, CW.standardDeviation) is None


def test_a_continuous_estimate_carries_sd_and_the_world_carries_row_count(cf_graph, ocg):
    """sd alone is uninterpretable — row_count says whether row duplication is in it."""
    from causalway.queries import query_iri

    d = _names(ocg)[3]
    node_iri = ocg.node_iri(_names(ocg).index(d))
    estimate = next(s for s in cf_graph.subjects(RDF.type, CW.CounterfactualEstimate)
                    if cf_graph.value(s, CW.onNode) == node_iri)
    sd = cf_graph.value(estimate, CW.standardDeviation)
    assert sd.datatype == XSD.double and float(sd) == pytest.approx(3.25)
    # A continuous counterfactual is a deterministic inversion: no probability.
    assert cf_graph.value(estimate, CW.probability) is None

    world_iri = query_iri(_world(ocg).as_query())
    assert int(cf_graph.value(world_iri, CW.rowCount)) == 1
    assert int(cf_graph.value(world_iri, CW.randomSeed)) == 7


def test_a_continuous_value_is_typed_and_a_categorical_one_is_not(cf_graph, ocg):
    """The whole point of the datatypeMap work: 111.017 is a double, "East" is not
    stamped with a type it does not have."""
    names = _names(ocg)
    continuous = next(s for s in cf_graph.subjects(RDF.type, CW.CounterfactualEstimate)
                      if cf_graph.value(s, CW.onNode) == ocg.node_iri(3))
    for term in (CW.predictedValue, CW.factualValue):
        assert cf_graph.value(continuous, term).datatype == XSD.double

    categorical = next(s for s in cf_graph.subjects(RDF.type, CW.CounterfactualEstimate)
                       if cf_graph.value(s, CW.onNode) == ocg.node_iri(2))
    for term in (CW.predictedValue, CW.factualValue):
        # RDF 1.1: a plain literal already *is* an xsd:string, so writing the
        # type out would be noise — and a discretised bin is not a double.
        assert cf_graph.value(categorical, term).datatype is None
    assert names  # the fixture's node order is what the indices above rely on


def test_annotation_form_asserts_no_entity_property_value_triple(cf_graph, ocg):
    """§6.1: merging this into the source KG must contradict nothing in it."""
    properties = {node.prop for node in ocg.nodes}
    for _s, predicate, _o in cf_graph.triples((URIRef(ENTITY), None, None)):
        assert predicate not in properties, (
            f"{predicate} would overwrite the entity's actual value on merge"
        )


def test_no_diagnostics_json_literal_is_written(cf_graph):
    """§6.6, and a hard engine limit: a JSON literal cannot survive the quote rewrite."""
    assert not list(cf_graph.subject_objects(CW.diagnostics))


def test_worlds_to_rdf_accumulates_into_a_given_graph(ocg):
    from rdflib import Graph

    store = Graph()
    worlds_to_rdf([_world(ocg)], ocg, graph=store)
    first = len(store)
    other = _world(ocg)
    other.model_id = "m-999"
    worlds_to_rdf([other], ocg, graph=store)
    assert len(store) > first
    assert len(list(store.subjects(RDF.type, CW.CounterfactualQuery))) == 2


# ---------------------------------------------------------------------- #
# A unit with no name
# ---------------------------------------------------------------------- #
def _hypothetical(ocg: OntologicalCausalGraph, **observed) -> CounterfactualWorld:
    a, _b, c, d = _names(ocg)[:4]
    factual = {c: "North", d: 99.5} | observed
    return CounterfactualWorld(
        interventions={a: "High"},
        counterfactual={c: "East", d: 111.017},
        factual=factual, model_id="m-123",
    )


def test_a_hypothetical_unit_has_no_entity_and_no_entity_navigation(ocg):
    """Absence is the statement: there is no resource to point cw:aboutEntity at."""
    graph = worlds_to_rdf([_hypothetical(ocg)], ocg, dtypes=_dtypes(ocg))
    worlds = list(graph.subjects(RDF.type, CW.CounterfactualQuery))
    assert len(worlds) == 1
    assert graph.value(worlds[0], CW.aboutEntity) is None
    assert not list(graph.subject_objects(CW.hasQuery))
    # ...but it is still a complete world: the estimates are all there.
    assert len(list(graph.subjects(RDF.type, CW.CounterfactualEstimate))) == 2


def test_two_hypothetical_units_differing_only_in_observations_do_not_collide(ocg):
    """The observed values are part of the unit's identity, so they are hashed in.

    Without this the two worlds share one IRI and the second export silently
    redefines the first — the same do() over a different unit is a different
    question with a different answer.
    """
    d = _names(ocg)[3]
    one = _hypothetical(ocg)
    two = _hypothetical(ocg, **{d: 12.5})
    payload = worlds_to_json([one, two], ocg, dtypes=_dtypes(ocg))
    iris = {row["world_iri"] for row in payload["worlds"]}
    assert len(iris) == 2
    assert payload["worlds_named"] == []


def test_a_named_and_a_hypothetical_world_export_together(ocg):
    """The mixed case is the one that crashed the engine before `worlds_named`."""
    graph = worlds_to_rdf([_world(ocg), _hypothetical(ocg)], ocg, dtypes=_dtypes(ocg))
    assert len(list(graph.subjects(RDF.type, CW.CounterfactualQuery))) == 2
    assert len(list(graph.subject_objects(CW.hasQuery))) == 1   # only the named one


# ---------------------------------------------------------------------- #
# Round trip.  This is what stands in for the SHACL shapes that were deleted:
# a hand-written second description of the vocabulary drifts from the first and
# then dies unnoticed, while this fails in CI the moment the contract moves.
# ---------------------------------------------------------------------- #
def test_the_rdf_reproduces_the_world_it_was_built_from(ocg):
    world = _world(ocg)
    graph = worlds_to_rdf([world], ocg, dtypes=_dtypes(ocg))

    rows = graph.query("""
        PREFIX cw: <http://sdm-causalway.org/>
        SELECT ?node ?predicted ?factual WHERE {
            ?e a cw:CounterfactualEstimate ;
               cw:forQuery ?w ; cw:onNode ?node ;
               cw:predictedValue ?predicted ; cw:factualValue ?factual .
        }
    """)
    by_node = {str(node): (predicted, factual) for node, predicted, factual in rows}
    names = _names(ocg)
    for name, predicted in world.counterfactual.items():
        if name in world.interventions:
            continue
        node_iri = str(ocg.node_iri(names.index(name)))
        got_predicted, got_factual = by_node[node_iri]
        # `.toPython()` so a typed literal comes back as the number it was, not
        # as its lexical form — that round trip is the point of the datatypes.
        assert got_predicted.toPython() == predicted
        assert got_factual.toPython() == world.factual[name]

    acts = graph.query("""
        PREFIX cw: <http://sdm-causalway.org/>
        SELECT ?node ?value WHERE {
            ?w cw:hasIntervention ?i . ?i cw:onNode ?node ; cw:setValue ?value .
        }
    """)
    got = {str(node): value.toPython() for node, value in acts}
    assert got == {str(ocg.node_iri(names.index(name))): value
                   for name, value in world.interventions.items()}


def test_a_quote_in_a_value_is_refused_not_corrupted(ocg):
    world = _world(ocg)
    world.counterfactual[ocg.nodes[2].name] = 'he said "no"'
    with pytest.raises(ValueError, match="SDM-RDFizer"):
        worlds_to_rdf([world], ocg)
