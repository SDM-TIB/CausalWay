"""Counterfactual worlds are cw: counterfactual queries: ``world_to_query`` -> ``query_mapping``.

The world export is the one artifact of this project that describes *named
resources of the source KG*, so beyond "the triples come out" it is checked that
they could be merged into that KG without contradicting it: the entity IRI is
verbatim, and nothing asserts ``entity property value``.  A hand-described unit
has no entity; its observed state becomes the query's conditions.
"""

from __future__ import annotations

import pytest
from rdflib import Literal, RDF, URIRef, XSD

from causalway.cf_export import CounterfactualWorld, world_to_query, worlds_to_json, worlds_to_rdf
from causalway.queries import query_iri
from causalway.vocab import CW, QUERY_STEM

from _fixtures import KG, PATIENT_0, FakeModel, clinic_ocg

GID = clinic_ocg().gid


def _world(**overrides) -> CounterfactualWorld:
    """One entity under a joint do() on two nodes; one categorical and one continuous outcome."""
    kwargs = dict(
        entity=PATIENT_0,
        interventions={"Patient.smoking": "No", "Hospital.careQuality": "High"},
        counterfactual={"Patient.smoking": "No", "Hospital.careQuality": "High",
                        "Patient.survival": "Long", "Therapy.dosage": 111.017},
        factual={"Patient.smoking": "Yes", "Hospital.careQuality": "Low",
                 "Patient.survival": "Short", "Therapy.dosage": 99.5},
        distribution={"Patient.survival": {"Long": 0.7, "Short": 0.3}},
        std={"Therapy.dosage": 3.25},
    )
    kwargs.update(overrides)
    return CounterfactualWorld(**kwargs)


@pytest.fixture(scope="module")
def model():
    ocg = clinic_ocg()
    dtypes = {n.name: "categorical" for n in ocg.nodes}
    dtypes["Therapy.dosage"] = "continuous"
    return FakeModel(ocg, dtypes=dtypes)


@pytest.fixture(scope="module")
def cf_graph(model):
    return worlds_to_rdf([_world()], model)


def test_a_world_is_a_counterfactual_query_with_one_answer_per_free_node():
    query, answers = world_to_query(_world(), GID)
    assert query.kind == "counterfactual" and query.entity == PATIENT_0
    assert query.conditions == []                  # a named unit's evidence is the KG
    assert {iv.node for iv in query.interventions} == {"Patient.smoking", "Hospital.careQuality"}
    assert {a.target for a in answers} == {"Patient.survival", "Therapy.dosage"}


def test_the_world_iri_is_the_query_iri(cf_graph):
    query, _ = world_to_query(_world(), GID)
    assert (query_iri(query), RDF.type, CW.CounterfactualQuery) in cf_graph
    assert str(query_iri(query)).startswith(QUERY_STEM)


def test_the_entity_iri_is_verbatim(cf_graph):
    query, _ = world_to_query(_world(), GID)
    assert (query_iri(query), CW.aboutEntity, URIRef(PATIENT_0)) in cf_graph


def test_every_answer_has_both_values(cf_graph):
    query, _ = world_to_query(_world(), GID)
    answers = {str(cf_graph.value(a, CW.onProperty)): a
               for a in cf_graph.subjects(CW.forQuery, query_iri(query))}
    assert set(answers) == {KG + "survival", KG + "dosage"}
    survival = answers[KG + "survival"]
    assert cf_graph.value(survival, CW.predictedValue) == Literal("Long")
    assert cf_graph.value(survival, CW.factualValue) == Literal("Short")


def test_categorical_carries_probability_continuous_carries_a_typed_value_and_std(cf_graph):
    query, _ = world_to_query(_world(), GID)
    answers = {str(cf_graph.value(a, CW.onProperty)): a
               for a in cf_graph.subjects(CW.forQuery, query_iri(query))}
    survival, dosage = answers[KG + "survival"], answers[KG + "dosage"]
    assert cf_graph.value(survival, CW.probability) == Literal(0.7, datatype=XSD.double)
    assert cf_graph.value(survival, CW.std) is None
    assert cf_graph.value(dosage, CW.predictedValue) == Literal(111.017, datatype=XSD.double)
    assert cf_graph.value(dosage, CW.factualValue) == Literal(99.5, datatype=XSD.double)
    assert cf_graph.value(dosage, CW.std) == Literal(3.25, datatype=XSD.double)
    assert cf_graph.value(dosage, CW.probability) is None


def test_nothing_is_asserted_about_the_entity(cf_graph):
    assert (URIRef(PATIENT_0), None, None) not in cf_graph


def test_a_hypothetical_unit_has_no_entity_and_its_observations_are_conditions(model):
    world = _world(entity=None)
    query, answers = world_to_query(world, GID)
    assert query.entity is None
    assert {(c.node, c.value) for c in query.conditions} == set(world.factual.items())
    g = worlds_to_rdf([world], model)
    q = query_iri(query)
    assert (q, CW.aboutEntity, None) not in g
    assert len(list(g.objects(q, CW.hasCondition))) == len(world.factual)


def test_two_hypothetical_units_differing_only_in_observations_do_not_collide():
    a = world_to_query(_world(entity=None))[0]
    b = world_to_query(_world(entity=None, factual={"Patient.smoking": "Yes",
                                                     "Hospital.careQuality": "High",
                                                     "Patient.survival": "Long",
                                                     "Therapy.dosage": 80.0}))[0]
    assert query_iri(a) != query_iri(b)


def test_every_typed_row_carries_a_datatype(model):
    doc = worlds_to_json([_world(), _world(entity=None)], model)
    for bucket in ("answers_typed", "interventions_typed", "conditions_typed"):
        assert all(r.get("datatype") for r in doc[bucket])


def test_a_quote_in_a_value_is_refused_not_corrupted(model):
    world = _world(counterfactual={**_world().counterfactual, "Patient.survival": 'say "no"'})
    with pytest.raises(ValueError):
        worlds_to_rdf([world], model)
