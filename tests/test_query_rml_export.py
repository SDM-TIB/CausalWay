"""Queries and answers go through ``query_mapping.rml.ttl`` and come out in the cw: shape.

What is pinned here:

* a query points at its **graph** (``cw:onGraph``) and names no model; each
  answer points at its **model** (``cw:evaluatedBy``) and at its query;
* one answer per target node, typed by the model's fitted dtype, with
  ``cw:probability`` for a categorical node and ``cw:std`` for a continuous one;
* Z is one set — ``cw:hasCondition`` on every kind of query;
* the service turns its flat answer log into these queries, one answer per board
  node, and bundles the model and its graph so every IRI resolves in the file.

These tests need SDM-RDFizer on PATH, like ``test_result_rml_export.py``.
"""

import pytest
from rdflib import Graph, Literal, RDF, URIRef, XSD

from causalway.queries import Answer, Condition, Intervention, Query, query_iri
from causalway.query_export import queries_to_json, queries_to_rdf
from causalway.vocab import CW

from _fixtures import PATIENT_0, FakeModel, clinic_ocg


def _answer(node, predicted, probability=None, std=None, **kw):
    return Answer(kind="x", target=node, predicted=predicted,
                  distribution=({predicted: probability} if probability is not None else None),
                  std=std, **kw)


@pytest.fixture(scope="module")
def setup():
    ocg = clinic_ocg()
    dtypes = {n.name: "categorical" for n in ocg.nodes}
    dtypes["Patient.age"] = "continuous"
    model = FakeModel(ocg, dtypes=dtypes)
    items = [
        (Query(kind="conditional", conditions=[Condition("Patient.smoking", "Yes")]),
         [_answer("Patient.survival", "Short", 0.6), _answer("Patient.age", 61.5, std=4.2)]),
        (Query(kind="interventional", label="P(V | do(careQuality=High), smoking=Yes)",
               interventions=[Intervention("Hospital.careQuality", "High")],
               conditions=[Condition("Patient.smoking", "Yes")]),
         [_answer("Patient.survival", "Long", 0.58)]),
        (Query(kind="counterfactual", entity=PATIENT_0,
               interventions=[Intervention("Patient.smoking", "No")]),
         [_answer("Patient.survival", "Long", 0.64, factual="Short")]),
    ]
    return ocg, model, items, queries_to_rdf(items, model)


def test_each_query_is_posed_on_the_graph_and_names_no_model(setup):
    ocg, model, items, g = setup
    for query, _ in items:
        q = query_iri(query)
        assert (q, CW.onGraph, ocg.iri) in g
        assert (q, CW.evaluatedBy, None) not in g
        assert (q, CW.usedModel, None) not in g


def test_the_class_follows_the_kind(setup):
    _, _, items, g = setup
    expected = {"conditional": (CW.ConditionalQuery, CW.ConditionalAnswer),
                "interventional": (CW.InterventionalQuery, CW.InterventionalAnswer),
                "counterfactual": (CW.CounterfactualQuery, CW.CounterfactualAnswer)}
    for query, _ in items:
        q_cls, a_cls = expected[query.kind]
        assert (query_iri(query), RDF.type, q_cls) in g
        for a in g.subjects(CW.forQuery, query_iri(query)):
            assert (a, RDF.type, a_cls) in g


def test_every_answer_names_its_model_node_and_time(setup):
    ocg, model, items, g = setup
    for query, answers in items:
        found = list(g.subjects(CW.forQuery, query_iri(query)))
        assert len(found) == len(answers)
        for a in found:
            assert g.value(a, CW.evaluatedBy) == model.iri
            assert g.value(a, CW.evaluatedTime).datatype == XSD.dateTime
            node = g.value(a, CW.onNode)
            assert str(node).startswith(str(ocg.iri) + "/node/")
            assert g.value(a, CW.onProperty) is not None


def test_z_is_has_condition_on_every_kind(setup):
    _, _, items, g = setup
    cond, interv, _ = items
    assert (query_iri(cond[0]), CW.hasCondition, None) in g
    assert (query_iri(interv[0]), CW.hasCondition, None) in g
    assert (query_iri(interv[0]), CW.hasIntervention, None) in g
    for retired in ("hasEvidence", "referenceIntervention", "targetNode", "causalEffect"):
        assert (None, CW[retired], None) not in g


def test_categorical_answers_carry_probability_and_continuous_ones_std(setup):
    _, _, items, g = setup
    cond = query_iri(items[0][0])
    by_value = {g.value(a, CW.predictedValue): a for a in g.subjects(CW.forQuery, cond)}
    survival = by_value[Literal("Short")]
    age = by_value[Literal(61.5, datatype=XSD.double)]
    assert g.value(survival, CW.probability) == Literal(0.6, datatype=XSD.double)
    assert g.value(survival, CW.std) is None
    assert g.value(age, CW.std) == Literal(4.2, datatype=XSD.double)
    assert g.value(age, CW.probability) is None


def test_a_counterfactual_answer_carries_the_factual_value(setup):
    _, _, items, g = setup
    q = query_iri(items[2][0])
    [a] = list(g.subjects(CW.forQuery, q))
    assert g.value(a, CW.factualValue) == Literal("Short")
    assert (q, CW.aboutEntity, URIRef(PATIENT_0)) in g
    # Only the counterfactual answer has a factual value.
    for query, _ in items[:2]:
        for other in g.subjects(CW.forQuery, query_iri(query)):
            assert g.value(other, CW.factualValue) is None


def test_the_same_intervention_on_the_same_graph_is_one_resource():
    ocg = clinic_ocg()
    model = FakeModel(ocg)
    x = [Intervention("Hospital.careQuality", "High")]
    doc = queries_to_json([
        (Query(kind="interventional", interventions=x), None),
        (Query(kind="interventional", interventions=x,
               conditions=[Condition("Patient.smoking", "Yes")]), None),
    ], model)
    links = {r["intervention_iri"] for r in doc["query_interventions"]}
    assert len(links) == 1
    assert len(doc["interventions"]) == 1


def test_a_query_on_another_graph_is_refused():
    ocg = clinic_ocg()
    q = Query(kind="conditional", graph_id="some-other-graph")
    with pytest.raises(ValueError, match="fitted on"):
        queries_to_json([(q, None)], FakeModel(ocg))


# ---------------------------------------------------------------------- #
# The service's answer log -> cw: queries
# ---------------------------------------------------------------------- #
def _project(model):
    return type("_P", (), {"model": model})()


def test_the_service_turns_a_board_record_into_one_answer_per_free_node():
    pytest.importorskip("fastapi")
    from service.api.main import _answers_turtle

    ocg = clinic_ocg()
    model = FakeModel(ocg)
    records = [
        {"answer_id": 1, "at": 1_790_000_000, "kind": "interventional",
         "estimand": "P(V | do(Hospital.careQuality = High), Patient.smoking = Yes)",
         "target": "V (all nodes)", "entity": None,
         "interventions": {"Hospital.careQuality": "High"},
         "evidence": {"Patient.smoking": "Yes"}, "conditions": {},
         "board": {"Patient.survival": {"predicted": "Long", "probability": 0.58},
                   "Patient.tumorStage": {"predicted": "III", "probability": 0.41},
                   # pinned nodes may sit on the board; they are not answers
                   "Patient.smoking": {"predicted": "Yes", "probability": 1.0}}},
        {"answer_id": 2, "target": "Patient.survival"},       # no kind: skipped
    ]
    payload, media_type, suffix = _answers_turtle(_project(model), records)
    assert (media_type, suffix) == ("text/turtle", "ttl")
    g = Graph().parse(data=payload.decode(), format="turtle")

    [q] = list(g.subjects(RDF.type, CW.InterventionalQuery))
    answers = list(g.subjects(CW.forQuery, q))
    assert len(answers) == 2
    assert (q, CW.hasCondition, None) in g
    # The model and its graph travel with the answers, so every IRI resolves.
    assert (model.iri, RDF.type, CW.CausalModel) in g
    assert (ocg.iri, RDF.type, CW.OntologicalCausalGraph) in g


def test_the_service_logs_a_hypothetical_counterfactual_unit_as_conditions():
    pytest.importorskip("fastapi")
    from service.api.main import _log_queries

    ocg = clinic_ocg()
    [(query, answers)] = _log_queries(_project(FakeModel(ocg)), [{
        "answer_id": 1, "kind": "counterfactual", "entity": None,
        "estimand": "P(V_{Patient.smoking=No} | a hypothetical unit)",
        "interventions": {"Patient.smoking": "No"},
        "observed": {"Patient.smoking": "Yes", "Patient.survival": "Short"},
        "board": {"Patient.survival": {"predicted": "Long", "probability": 0.6,
                                       "factual": "Short"}},
    }])
    assert query.entity is None
    assert {(c.node, c.value) for c in query.conditions} == {
        ("Patient.smoking", "Yes"), ("Patient.survival", "Short")}
    assert [a.factual for a in answers] == ["Short"]


def test_the_service_drops_what_the_cw_model_cannot_say():
    pytest.importorskip("fastapi")
    from service.api.main import _log_queries

    ocg = clinic_ocg()
    pairs = _log_queries(_project(FakeModel(ocg)), [
        # an interventional question about one entity: that is a counterfactual
        {"answer_id": 1, "kind": "interventional", "entity": PATIENT_0,
         "target": "Patient.survival", "predicted": "Long",
         "interventions": {"Hospital.careQuality": "High"}},
        # the treatment arm of a contrast still exports; the reference arm is not a term
        {"answer_id": 2, "kind": "interventional", "entity": None,
         "target": "Patient.survival", "predicted": "Long", "effect": 0.17,
         "distribution": {"Long": 0.6, "Short": 0.4},
         "interventions": {"Hospital.careQuality": "High"},
         "reference": {"Hospital.careQuality": "Low"}},
    ])
    [(query, answers)] = pairs
    assert [(iv.node, iv.value) for iv in query.interventions] == [("Hospital.careQuality", "High")]
    assert answers[0].probability == pytest.approx(0.6)


def test_asking_twice_is_one_query_and_the_latest_answer_wins():
    pytest.importorskip("fastapi")
    from service.api.main import _log_queries

    ocg = clinic_ocg()
    rec = {"kind": "conditional", "entity": None, "evidence": {"Patient.smoking": "Yes"},
           "interventions": {}, "target": "Patient.survival"}
    pairs = _log_queries(_project(FakeModel(ocg)), [
        {**rec, "answer_id": 1, "predicted": "Short"},
        {**rec, "answer_id": 2, "predicted": "Long"},
    ])
    [(_, answers)] = pairs
    assert [a.predicted for a in answers] == ["Long"]
