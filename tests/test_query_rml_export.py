"""The query/answer export goes through RML, and produces what it used to.

``store_query`` used to build its triples with thirty-odd ``g.add(...)`` calls.
It now delegates to ``query_export.queries_to_rdf``, which hands
``query_mapping.rml.ttl`` to SDM-RDFizer.  The equivalence that justified
deleting the hand-written version is pinned here: same subjects, same
predicates, same objects, modulo ``cw:computedAt``, which is stamped at call
time and therefore differs between any two runs.

These tests need SDM-RDFizer on PATH, like ``test_result_rml_export.py``.
"""

import numpy as np
import pytest
from rdflib import RDF, Namespace, URIRef

from causalway.constraints import EdgeConstraint
from causalway.nodes import PropertyNode
from causalway.queries import (
    Answer, Condition, Intervention, Query, answer_iri, condition_iri,
    intervention_iri, load_queries, query_iri, store_query,
)
from causalway.query_export import queries_to_json, queries_to_rdf
from causalway.result import OntologicalCausalGraph
from causalway.vocab import CW

EX = Namespace("http://example.org/")


def _toy_ocg():
    # Patient.age -> Patient.stage -> Patient.survival
    nodes = [
        PropertyNode(domain=EX.Patient, prop=EX.age, range_=EX.integer, kind="data"),
        PropertyNode(domain=EX.Patient, prop=EX.stage, range_=EX.string, kind="data"),
        PropertyNode(domain=EX.Patient, prop=EX.survival, range_=EX.string, kind="data"),
    ]
    n = len(nodes)
    allowed = np.zeros((n, n), dtype=bool)
    allowed[0, 1] = allowed[1, 2] = True
    constraint = EdgeConstraint(names=[x.name for x in nodes], nodes=nodes,
                                allowed=allowed)
    return OntologicalCausalGraph.from_discovery(allowed.astype(int), constraint,
                                                 method="TEST")


class _FakeModel:
    """Duck-typed stand-in for CausalModel — spec.ocg and iri are all that is read."""

    def __init__(self, ocg, model_id="test-model"):
        self.spec = type("_Spec", (), {"ocg": ocg, "mat": None})()
        self.model_id = model_id
        self.manifest = {"dtypes": {}}

    @property
    def iri(self):
        return URIRef(f"http://sdm-causalway.org/model/{self.model_id}")


def _interventional():
    q = Query(kind="interventional", model_id="test-model",
              target=["Patient.survival"],
              interventions=[Intervention(node="Patient.stage", value="III")],
              reference=[Intervention(node="Patient.stage", value="I")],
              condition_on=[Condition(node="Patient.age", value=61)],
              label="P(survival | do(stage=III), age=61)")
    a = Answer(kind="interventional", target="Patient.survival",
               predicted="alive", effect=0.17)
    return q, a


def _conditional():
    q = Query(kind="conditional", model_id="test-model",
              target=["Patient.survival"],
              evidence=[Condition(node="Patient.stage", value="III")])
    a = Answer(kind="conditional", target="Patient.survival", predicted="dead")
    return q, a


def _without_timestamp(graph):
    return {(str(s), str(p), str(o)) for s, p, o in graph if p != CW.computedAt}


def test_rml_export_matches_the_hand_written_triples():
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    q, a = _interventional()

    rml = queries_to_rdf([(q, a)], model)
    # store_query now delegates, so this is the same path; what the assertion
    # pins is that delegating did not change the result.
    stored = store_query(a, q, model)

    assert _without_timestamp(rml) == _without_timestamp(stored)
    assert len(list(rml.triples((None, CW.computedAt, None)))) == 1


def test_interventional_query_carries_its_condition_not_its_evidence():
    """`cw:hasCondition` is a sub-population under do(); `cw:hasEvidence` is seeing."""
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    q, a = _interventional()
    g = queries_to_rdf([(q, a)], model)

    q_iri = query_iri(q)
    assert (q_iri, CW.hasCondition, condition_iri(q.condition_on[0])) in g
    assert (q_iri, CW.hasEvidence, None) not in g
    assert (q_iri, CW.hasIntervention, intervention_iri(q.interventions[0])) in g
    assert (q_iri, CW.referenceIntervention, intervention_iri(q.reference[0])) in g
    # The class travels as data through an object map, not rr:class — the triple
    # it produces is an ordinary rdf:type either way.
    assert (q_iri, RDF.type, CW.InterventionalQuery) in g


def test_conditional_query_carries_its_evidence_not_a_condition():
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    q, a = _conditional()
    g = queries_to_rdf([(q, a)], model)

    q_iri = query_iri(q)
    assert (q_iri, CW.hasEvidence, condition_iri(q.evidence[0])) in g
    assert (q_iri, CW.hasCondition, None) not in g
    assert (q_iri, CW.hasIntervention, None) not in g


def test_estimate_links_back_to_its_query_and_node():
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    q, a = _interventional()
    g = queries_to_rdf([(q, a)], model)

    a_iri = answer_iri(q)
    assert (a_iri, CW.forQuery, query_iri(q)) in g
    assert (a_iri, CW.predictedValue, None) in g
    assert (a_iri, CW.causalEffect, None) in g


def test_a_per_level_effect_dict_exports_the_reported_level_only():
    """No JSON literal reaches RDF — SDM-RDFizer would corrupt its quotes."""
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    q, _ = _interventional()
    a = Answer(kind="interventional", target="Patient.survival",
               predicted="alive", effect={"alive": 0.22, "dead": -0.22})

    doc = queries_to_json([(q, a)], model)
    assert doc["estimate_effects"] == [
        {"estimate_key": doc["estimates"][0]["estimate_key"], "effect": "0.22"}
    ]
    assert all('"' not in str(v) for row in doc["estimate_effects"]
               for v in row.values())


def test_export_is_self_describing_about_its_nodes():
    """A query graph alone must name its nodes, or load_queries cannot replay it."""
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    q, a = _interventional()
    g = queries_to_rdf([(q, a)], model)

    described = set(g.subjects(RDF.type, CW.PropertyNode))
    assert described, "no cw:PropertyNode described"
    replayed = load_queries(g)
    assert len(replayed) == 1


def test_the_service_rebuilds_a_query_from_its_flat_answer_log():
    """`_answers_turtle` is the one place the log's flat dict becomes a Query again."""
    pytest.importorskip("fastapi")
    from service.api.main import _answers_turtle

    ocg = _toy_ocg()
    project = type("_P", (), {"model": _FakeModel(ocg)})()
    records = [
        {
            "answer_id": 1, "kind": "interventional", "target": "Patient.survival",
            "entity": None, "predicted": "alive", "effect": 0.17,
            "estimand": "P(survival | do(stage=III), age=61)",
            "interventions": {"Patient.stage": "III"},
            "reference": {"Patient.stage": "I"},
            "conditions": {"Patient.age": 61},
            "evidence": {},
        },
        # A non-replayable record (a stale log entry with no kind) must be
        # skipped, not crash the export of the ones beside it.
        {"answer_id": 2, "target": "Patient.survival"},
    ]

    payload, media_type, suffix = _answers_turtle(project, records)
    assert media_type == "text/turtle"
    assert suffix == "ttl"

    # The export carries the vocabulary too, so every term is *defined* in this
    # document. Asserting on text would match those definitions; assert on the
    # instance triples instead.
    from rdflib import Graph
    g = Graph().parse(data=payload.decode(), format="turtle")
    queries = set(g.subjects(RDF.type, CW.InterventionalQuery))
    assert len(queries) == 1, "the record with no kind should have been skipped"
    q_iri = next(iter(queries))
    assert (q_iri, CW.hasCondition, None) in g      # sub-population, under do()
    assert (q_iri, CW.referenceIntervention, None) in g
    assert (q_iri, CW.hasEvidence, None) not in g   # not a conditional query


def test_an_entity_scoped_interventional_query_is_not_about_that_entity():
    """Population-level questions carry the entity on the act, never as aboutEntity."""
    pytest.importorskip("fastapi")
    from service.api.main import _answers_turtle

    ocg = _toy_ocg()
    project = type("_P", (), {"model": _FakeModel(ocg)})()
    entity = "http://example.org/patient/1"
    payload, _, _ = _answers_turtle(project, [{
        "answer_id": 1, "kind": "interventional", "target": "Patient.survival",
        "entity": entity, "predicted": "alive", "effect": None,
        "estimand": "unit-scoped", "interventions": {"Patient.stage": "III"},
        "reference": {}, "conditions": {}, "evidence": {},
    }])
    from rdflib import Graph
    g = Graph().parse(data=payload.decode(), format="turtle")
    assert (None, CW.onEntity, URIRef(entity)) in g
    assert (None, CW.aboutEntity, URIRef(entity)) not in g


def test_no_bare_triple_is_asserted_about_the_entity():
    """§6.1: the annotation export never says `entity property value`."""
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    entity = "http://example.org/patient/1"
    q = Query(kind="counterfactual", model_id="test-model",
              target=["Patient.survival"], entity=entity,
              interventions=[Intervention(node="Patient.stage", value="I",
                                          entity=entity)])
    a = Answer(kind="counterfactual", target="Patient.survival",
               entity=entity, predicted="alive")

    g = queries_to_rdf([(q, a)], model)
    from_entity = list(g.predicate_objects(URIRef(entity)))
    assert [p for p, _ in from_entity] == [CW.hasQuery]
