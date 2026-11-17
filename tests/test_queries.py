"""The cw: query model in Python: kinds, disjointness, IRIs, reach, and the RDF round trip."""

import pytest
from rdflib import Literal, RDF, URIRef

from causalway.queries import (
    Answer, Condition, Intervention, Query, answer_iri, condition_iri,
    intervention_iri, load_answers, load_queries, query_iri, store_query,
    validate_query,
)
from causalway.vocab import CW

from _fixtures import KG, PATIENT_0, FakeModel, clinic_ocg


# ---------------------------------------------------------------------- #
# What a query may be
# ---------------------------------------------------------------------- #
def test_a_conditional_query_has_no_intervention():
    Query(kind="conditional", conditions=[Condition("Patient.smoking", "Yes")])
    Query(kind="conditional")                       # P(V): no condition is fine
    with pytest.raises(ValueError, match="interventional"):
        Query(kind="conditional", interventions=[Intervention("Patient.smoking", "No")])


def test_interventional_and_counterfactual_queries_need_an_intervention():
    for kind in ("interventional", "counterfactual"):
        with pytest.raises(ValueError, match="at least one intervention"):
            Query(kind=kind, entity=PATIENT_0 if kind == "counterfactual" else None)


def test_x_and_z_are_disjoint_outside_a_counterfactual():
    with pytest.raises(ValueError, match="X ∩ Z"):
        Query(kind="interventional", interventions=[Intervention("Patient.smoking", "No")],
              conditions=[Condition("Patient.smoking", "Yes")])
    # A counterfactual's evidence may name the very node it intervenes on.
    Query(kind="counterfactual", interventions=[Intervention("Patient.smoking", "No")],
          conditions=[Condition("Patient.smoking", "Yes")])


def test_a_node_is_intervened_on_or_conditioned_on_at_most_once():
    with pytest.raises(ValueError, match="twice"):
        Query(kind="interventional", interventions=[Intervention("Patient.smoking", "No"),
                                                    Intervention("Patient.smoking", "Yes")])


def test_only_a_counterfactual_is_about_one_entity():
    with pytest.raises(ValueError, match="counterfactual"):
        Query(kind="interventional", interventions=[Intervention("Patient.smoking", "No")],
              entity=PATIENT_0)


def test_a_counterfactual_has_exactly_one_kind_of_evidence():
    x = [Intervention("Patient.smoking", "No")]
    Query(kind="counterfactual", interventions=x, entity=PATIENT_0)
    Query(kind="counterfactual", interventions=x, conditions=[Condition("Patient.age", "Old")])
    with pytest.raises(ValueError, match="exactly one"):
        Query(kind="counterfactual", interventions=x)
    with pytest.raises(ValueError, match="exactly one"):
        Query(kind="counterfactual", interventions=x, entity=PATIENT_0,
              conditions=[Condition("Patient.age", "Old")])


# ---------------------------------------------------------------------- #
# IRIs
# ---------------------------------------------------------------------- #
def test_iris_are_minted_from_content_and_graph():
    q1 = Query(kind="interventional", graph_id="g",
               interventions=[Intervention("Patient.smoking", "No")],
               conditions=[Condition("Patient.age", "Old")])
    q2 = Query(kind="interventional", graph_id="g",
               conditions=[Condition("Patient.age", "Old")],
               interventions=[Intervention("Patient.smoking", "No")])
    q3 = Query(kind="interventional", graph_id="other",
               interventions=[Intervention("Patient.smoking", "No")],
               conditions=[Condition("Patient.age", "Old")])
    assert query_iri(q1) == query_iri(q2) != query_iri(q3)
    iv = Intervention("Patient.smoking", "No")
    assert intervention_iri("g", iv) == intervention_iri("g", Intervention("Patient.smoking", "No"))
    assert intervention_iri("g", iv) != intervention_iri("other", iv)
    assert condition_iri("g", Condition("a", 1)) != condition_iri("g", Condition("a", 2))
    assert str(answer_iri(q1, "m1", "node-key")).endswith(f"{str(query_iri(q1)).rsplit('/', 1)[1]}/m1/node-key")


# ---------------------------------------------------------------------- #
# Reach
# ---------------------------------------------------------------------- #
def test_validate_query_checks_causal_reach():
    ocg = clinic_ocg()
    q = Query(kind="interventional", interventions=[Intervention("Patient.smoking", "No")])
    validate_query(q, ocg, targets=["Patient.survival"])     # smoking -> stage -> survival
    with pytest.raises(ValueError, match="no causal path"):
        validate_query(q, ocg, targets=["Hospital.careQuality"])


def test_validate_query_refuses_a_target_in_x_or_z():
    ocg = clinic_ocg()
    q = Query(kind="interventional", interventions=[Intervention("Patient.smoking", "No")],
              conditions=[Condition("Patient.tumorStage", "III")])
    for target in ("Patient.smoking", "Patient.tumorStage"):
        with pytest.raises(ValueError, match="cannot be a target"):
            validate_query(q, ocg, targets=[target])


def test_validate_query_needs_the_kg_for_a_cross_entity_intervention():
    ocg = clinic_ocg()
    q = Query(kind="counterfactual", entity=PATIENT_0,
              interventions=[Intervention("Hospital.careQuality", "High",
                                          entity=KG + "hospital_0")])
    with pytest.raises(ValueError, match="relational reach"):
        validate_query(q, ocg, targets=["Patient.survival"])


# ---------------------------------------------------------------------- #
# The RDF round trip
# ---------------------------------------------------------------------- #
def test_store_and_load_round_trip():
    ocg = clinic_ocg()
    model = FakeModel(ocg)
    query = Query(kind="interventional",
                  interventions=[Intervention("Hospital.careQuality", "High")],
                  conditions=[Condition("Patient.smoking", "Yes")])
    answers = {
        "Patient.survival": Answer(kind="interventional", target="Patient.survival",
                                   predicted="Long", distribution={"Long": 0.58}),
        "Patient.tumorStage": Answer(kind="interventional", target="Patient.tumorStage",
                                     predicted="III", distribution={"III": 0.41}),
    }
    g = store_query(answers, query, model)

    [loaded] = load_queries(g, ocg=ocg)
    assert loaded.kind == "interventional" and loaded.graph_id == ocg.gid
    assert [(iv.node, iv.value) for iv in loaded.interventions] == [("Hospital.careQuality", "High")]
    assert [(c.node, c.value) for c in loaded.conditions] == [("Patient.smoking", "Yes")]
    assert query_iri(loaded) == query_iri(query)

    stored = load_answers(g, ocg=ocg).set_index("node")
    assert set(stored.index) == {"Patient.survival", "Patient.tumorStage"}   # V
    assert stored.loc["Patient.survival", "predicted"] == "Long"
    assert stored.loc["Patient.survival", "probability"] == pytest.approx(0.58)
    assert set(stored["model"]) == {str(model.iri)}


def test_the_annotation_form_asserts_nothing_about_the_entity():
    ocg = clinic_ocg()
    model = FakeModel(ocg)
    query = Query(kind="counterfactual", entity=PATIENT_0,
                  interventions=[Intervention("Patient.smoking", "No")])
    answer = Answer(kind="counterfactual", target="Patient.survival", entity=PATIENT_0,
                    predicted="Long", distribution={"Long": 0.6}, factual="Short")
    g = store_query(answer, query, model)
    assert (URIRef(PATIENT_0), None, None) not in g
    assert (query_iri(query), CW.aboutEntity, URIRef(PATIENT_0)) in g
    assert (URIRef(PATIENT_0), URIRef(KG + "survival"), Literal("Long")) not in g


def test_the_named_graph_form_projects_the_counterfactual_onto_the_entity():
    from rdflib import Dataset

    ocg = clinic_ocg()
    model = FakeModel(ocg)
    query = Query(kind="counterfactual", entity=PATIENT_0,
                  interventions=[Intervention("Patient.smoking", "No")])
    answer = Answer(kind="counterfactual", target="Patient.survival", predicted="Long")
    ds = store_query(answer, query, model, graph=Dataset(), merge="named-graph")
    ctx = ds.get_context(query_iri(query))
    assert (URIRef(PATIENT_0), URIRef(KG + "survival"), Literal("Long")) in ctx


def test_an_answer_about_an_intervened_node_is_refused():
    ocg = clinic_ocg()
    query = Query(kind="interventional", interventions=[Intervention("Patient.smoking", "No")])
    bad = Answer(kind="interventional", target="Patient.smoking", predicted="No")
    with pytest.raises(ValueError, match="intervened on"):
        store_query(bad, query, FakeModel(ocg))


def test_one_query_carries_the_answers_of_two_models():
    ocg = clinic_ocg()
    query = Query(kind="interventional", interventions=[Intervention("Hospital.careQuality", "High")])
    g = store_query(Answer(kind="interventional", target="Patient.survival", predicted="Long",
                           distribution={"Long": 0.6}), query, FakeModel(ocg, model_id="m1"))
    store_query(Answer(kind="interventional", target="Patient.survival", predicted="Short",
                       distribution={"Short": 0.55}), query, FakeModel(ocg, model_id="m2"),
                graph=g)
    assert len(list(g.subjects(RDF.type, CW.InterventionalQuery))) == 1
    answers = list(g.subjects(CW.forQuery, query_iri(query)))
    assert len(answers) == 2
    assert {str(g.value(a, CW.evaluatedBy)).rsplit("/", 1)[1] for a in answers} == {"m1", "m2"}
