"""Tests for Query/Answer RDF (de)serialisation and §6.5 reach validation."""

import numpy as np
import pytest
from rdflib import Namespace, URIRef

from causalway.constraints import EdgeConstraint
from causalway.nodes import PropertyNode
from causalway.queries import (
    Answer, Condition, Intervention, Query, answer_iri, intervention_iri,
    load_queries, query_iri, store_query, validate_query,
)
from causalway.result import OntologicalCausalGraph

EX = Namespace("http://example.org/")


def _toy_ocg():
    # Patient.age -> Patient.stage -> Patient.survival ; Hospital.quality (isolated)
    age = PropertyNode(domain=EX.Patient, prop=EX.age, range_=EX.integer, kind="data")
    stage = PropertyNode(domain=EX.Patient, prop=EX.stage, range_=EX.string, kind="data")
    survival = PropertyNode(domain=EX.Patient, prop=EX.survival, range_=EX.string, kind="data")
    quality = PropertyNode(domain=EX.Hospital, prop=EX.quality, range_=EX.string, kind="data")
    nodes = [age, stage, survival, quality]
    n = len(nodes)
    allowed = np.zeros((n, n), dtype=bool)
    allowed[0, 1] = allowed[1, 2] = True
    constraint = EdgeConstraint(names=[x.name for x in nodes], nodes=nodes, allowed=allowed)
    adj = allowed.astype(int)
    return OntologicalCausalGraph.from_discovery(adj, constraint, method="TEST")


class _FakeSpec:
    def __init__(self, ocg, mat=None):
        self.ocg = ocg
        self.mat = mat


class _FakeModel:
    """Duck-typed stand-in for CausalModel — enough for store_query/validate_query."""

    def __init__(self, ocg, mat=None, model_id="test-model"):
        self.spec = _FakeSpec(ocg, mat)
        self._model_id = model_id

    @property
    def model_id(self):
        return self._model_id

    @property
    def iri(self):
        return URIRef(f"http://sdm-causalway.org/model/{self._model_id}")


def test_validate_query_accepts_direct_parent():
    ocg = _toy_ocg()
    q = Query(kind="interventional", model_id="m", target=["Patient.stage"],
             interventions=[Intervention(node="Patient.age", value=1)])
    validate_query(q, ocg)  # should not raise


def test_validate_query_rejects_unreachable_node():
    ocg = _toy_ocg()
    q = Query(kind="interventional", model_id="m", target=["Patient.stage"],
             interventions=[Intervention(node="Hospital.quality", value="high")])
    with pytest.raises(ValueError, match="causal reach"):
        validate_query(q, ocg)


def test_validate_query_accepts_multi_hop_path():
    ocg = _toy_ocg()
    q = Query(kind="interventional", model_id="m", target=["Patient.survival"],
             interventions=[Intervention(node="Patient.age", value=1)])
    validate_query(q, ocg)


def test_query_counterfactual_requires_entity():
    with pytest.raises(ValueError):
        Query(kind="counterfactual", model_id="m", target=["Patient.survival"])


def test_query_non_counterfactual_forbids_entity():
    with pytest.raises(ValueError):
        Query(kind="conditional", model_id="m", target=["Patient.survival"],
             entity="http://example.org/patient1")


def test_intervention_iri_deterministic_and_shared():
    iv1 = Intervention(node="Patient.age", value=5)
    iv2 = Intervention(node="Patient.age", value=5)
    iv3 = Intervention(node="Patient.age", value=6)
    assert intervention_iri(iv1) == intervention_iri(iv2)
    assert intervention_iri(iv1) != intervention_iri(iv3)


def test_query_iri_stable_for_same_content():
    q1 = Query(kind="conditional", model_id="m", target=["Patient.stage"],
              evidence=[Condition(node="Patient.age", value=1)])
    q2 = Query(kind="conditional", model_id="m", target=["Patient.stage"],
              evidence=[Condition(node="Patient.age", value=1)])
    assert query_iri(q1) == query_iri(q2)
    assert answer_iri(q1) == answer_iri(q2)


def test_store_and_load_query_roundtrip():
    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    query = Query(kind="interventional", model_id=model.model_id, target=["Patient.stage"],
                 interventions=[Intervention(node="Patient.age", value="old")])
    answer = Answer(kind="interventional", target="Patient.stage", predicted="III")

    g = store_query(answer, query, model)
    loaded = load_queries(g)
    assert len(loaded) == 1
    lq = loaded[0]
    assert lq.kind == "interventional"
    assert lq.target == ["Patient.stage"]
    assert len(lq.interventions) == 1
    assert lq.interventions[0].node == "Patient.age"
    assert lq.interventions[0].value == "old"


def test_store_query_adds_no_bare_triple_about_entity():
    """§6.1: merging must never add a plain `entity predicate value` triple."""
    from rdflib import Literal

    ocg = _toy_ocg()
    model = _FakeModel(ocg)
    patient = "http://example.org/patient1"
    query = Query(kind="counterfactual", model_id=model.model_id, target=["Patient.survival"],
                 interventions=[Intervention(node="Patient.age", value=70, entity=patient)],
                 entity=patient)
    answer = Answer(kind="counterfactual", target="Patient.survival", entity=patient,
                    predicted="long")
    g = store_query(answer, query, model, merge="annotation")

    assert (URIRef(patient), EX.survival, Literal("long")) not in g
