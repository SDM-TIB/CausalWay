"""``causalway/shapes.ttl`` against real exports, and against data broken on purpose.

The shapes are the constraints of the cw: model — a DAG, one answer type per
query type, V/X/Z disjointness, Assumption 1, and the rest.  A shape file that
nothing runs drifts from what the mappings emit and then describes nothing
(the fate of the previous ``shapes.ttl``), so both directions are pinned here:

* every export the project writes — a discovered graph, a curated graph with a
  hand-drawn edge, a fitted model, all three kinds of query with their answers,
  a counterfactual about a hand-described unit, and every graph under
  ``results/`` — **conforms**;
* each constraint, violated on its own, is **reported** under its own message.

These tests need SDM-RDFizer on PATH and pyshacl installed.
"""

from __future__ import annotations

import glob
import os

import pytest
from rdflib import Graph, Literal, RDFS, URIRef, XSD
from rdflib.namespace import SH

from causalway.cf_export import CounterfactualWorld, worlds_to_rdf
from causalway.queries import Answer, Condition, Intervention, Query
from causalway.query_export import queries_to_rdf
from causalway.vocab import CW

from _fixtures import CLINIC, HERE, KG, PATIENT_0, FakeModel, clinic_ocg

pyshacl = pytest.importorskip("pyshacl")
SHAPES = Graph().parse(os.path.join(HERE, "causalway", "shapes.ttl"))


def _report(data: Graph) -> set:
    """``{(severity, message)}`` of every result the shapes report on ``data``."""
    _, results, _ = pyshacl.validate(data, shacl_graph=SHAPES)
    out = set()
    for r in results.subjects(SH.resultSeverity, None):
        severity = str(results.value(r, SH.resultSeverity)).rsplit("#", 1)[1]
        out.add((severity, str(results.value(r, SH.resultMessage))))
    return out


def _tbox() -> Graph:
    """The clinic KG's rdfs:domain / rdfs:range, so the single-hop Assumption 1 check runs."""
    kg = Graph().parse(CLINIC)
    g = Graph()
    for pred in (RDFS.domain, RDFS.range):
        for s, o in kg.subject_objects(pred):
            g.add((s, pred, o))
    return g


@pytest.fixture(scope="module")
def valid() -> Graph:
    ocg = clinic_ocg(max_hops=2, edges=[
        ("Patient.smoking", "Patient.tumorStage"), ("Patient.tumorStage", "Patient.survival"),
        ("Hospital.careQuality", "Patient.survival"), ("Therapy.toxicity", "Patient.survival"),
        ("Hospital.region", "Therapy.dosage")])            # a two-hop path
    dtypes = {n.name: "categorical" for n in ocg.nodes}
    dtypes["Therapy.dosage"] = "continuous"
    model = FakeModel(ocg, dtypes=dtypes)

    curated = clinic_ocg(graph_id="clinic-curated")
    curated.method, curated.curated = "curated", True
    curated.manual_edges = {curated.edges()[0]}

    g = _tbox()
    ocg.to_rdf(graph=g)
    curated.to_rdf(graph=g)
    model.to_rdf(graph=g)
    queries_to_rdf([
        (Query(kind="conditional", conditions=[Condition("Patient.smoking", "Yes")]),
         [Answer(kind="c", target="Patient.survival", predicted="Short", distribution={"Short": 0.6}),
          Answer(kind="c", target="Therapy.dosage", predicted=42.5, std=3.1)]),
        (Query(kind="interventional", interventions=[Intervention("Hospital.careQuality", "High")],
               conditions=[Condition("Patient.smoking", "Yes")]),
         [Answer(kind="i", target="Patient.survival", predicted="Long", distribution={"Long": 0.58})]),
        (Query(kind="counterfactual", entity=PATIENT_0,
               interventions=[Intervention("Patient.smoking", "No")]),
         [Answer(kind="x", target="Patient.survival", predicted="Long",
                 distribution={"Long": 0.64}, factual="Short")]),
    ], model, graph=g)
    worlds_to_rdf([CounterfactualWorld(
        interventions={"Patient.smoking": "No"},
        counterfactual={"Patient.survival": "Long", "Therapy.dosage": 40.0},
        factual={"Patient.smoking": "Yes", "Patient.survival": "Short", "Therapy.dosage": 44.0},
        distribution={"Patient.survival": {"Long": 0.7}}, std={"Therapy.dosage": 0.0},
    )], model, graph=g)
    return g


def test_every_export_conforms(valid):
    report = _report(valid)
    assert not {r for r in report if r[0] == "Violation"}, report
    # The one warning is the one the shapes exist to raise: the clinic T-Box
    # declares Therapy.dosage an xsd:string, the model fitted it as a number, and
    # an observed 44.0 is therefore outside the declared range. Legal, flagged.
    assert report == {("Warning",
                       "Value 44.0 does not match the node's range "
                       "http://www.w3.org/2001/XMLSchema#string.")}


@pytest.mark.parametrize("path", sorted(
    p for p in glob.glob(os.path.join(HERE, "results", "**", "*.ttl"), recursive=True)
    if os.path.getsize(p) > 1))
def test_every_result_graph_conforms(path):
    # Including best_GES_constrained.ttl, whose two undirected (CPDAG) edges are
    # oriented within the Markov equivalence class before export.
    assert _report(Graph().parse(path)) == set()


# ---------------------------------------------------------------------- #
# Each constraint, violated on its own
# ---------------------------------------------------------------------- #
G = "http://sdm-causalway.org/graph/clinic-test"


def _node(g: Graph, label: str, graph: str = G) -> URIRef:
    return next(n for n in g.subjects(RDFS.label, Literal(label))
                if (n, CW.inGraph, URIRef(graph)) in g)


def _edge(g: Graph, iri: str, cause: str, effect: str, *, path=None, curated=False,
          weight=0.5, graph: str = G) -> None:
    e = URIRef(iri)
    g.add((e, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), CW.CausalEdge))
    g.add((e, RDFS.label, Literal("added by the test")))
    g.add((e, CW.inGraph, URIRef(graph)))
    g.add((e, CW.cause, _node(g, cause, graph)))
    g.add((e, CW.effect, _node(g, effect, graph)))
    g.add((e, CW.curated, Literal(curated)))
    if weight is not None:
        g.add((e, CW.weight, Literal(weight, datatype=XSD.double)))
    if path:
        g.add((e, CW.viaRelationalPath, Literal(path)))


def _q(g: Graph, cls) -> URIRef:
    """The first resource of class ``cls`` — a query, an answer or an intervention."""
    return next(g.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), cls))


_a = _q


def _cycle(g):
    _edge(g, "urn:e:cycle", "Patient.survival", "Patient.smoking")


def _self_loop(g):
    _edge(g, "urn:e:self", "Patient.age", "Patient.age")


def _duplicate_edge(g):
    _edge(g, "urn:e:dup", "Patient.smoking", "Patient.tumorStage")


def _epsilon_across_classes(g):
    _edge(g, "urn:e:eps", "Hospital.careQuality", "Patient.age")


def _wrong_direction(g):
    # treatedAt runs Patient -> Hospital; Hospital -> Patient needs '^'.
    _edge(g, "urn:e:dir", "Hospital.region", "Patient.age", path=f"<{KG}treatedAt>")


def _weightless_raw_edge(g):
    _edge(g, "urn:e:noweight", "Patient.geneticRisk", "Patient.survival", weight=None)


def _duplicate_node(g):
    n = _node(g, "Patient.age")
    twin = URIRef(str(n) + "-twin")
    for p, o in list(g.predicate_objects(n)):
        g.add((twin, p, o))


def _conditional_with_do(g):
    q = _q(g, CW.ConditionalQuery)
    g.add((q, CW.hasIntervention, _q(g, CW.Intervention)))


def _interventional_about_an_entity(g):
    g.add((_q(g, CW.InterventionalQuery), CW.aboutEntity, URIRef(PATIENT_0)))


def _x_meets_z(g):
    q = _q(g, CW.InterventionalQuery)
    iv = URIRef("urn:iv:smoking")
    g.add((iv, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), CW.Intervention))
    g.add((iv, CW.onNode, _node(g, "Patient.smoking")))
    g.add((iv, CW.onProperty, URIRef(KG + "smoking")))
    g.add((iv, CW.setValue, Literal("No")))
    g.add((q, CW.hasIntervention, iv))


def _intervened_twice(g):
    q = _q(g, CW.InterventionalQuery)
    iv = URIRef("urn:iv:quality-low")
    g.add((iv, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), CW.Intervention))
    g.add((iv, CW.onNode, _node(g, "Hospital.careQuality")))
    g.add((iv, CW.onProperty, URIRef(KG + "careQuality")))
    g.add((iv, CW.setValue, Literal("Low")))
    g.add((q, CW.hasIntervention, iv))


def _counterfactual_with_both_kinds_of_evidence(g):
    q = next(q for q in g.subjects(CW.aboutEntity, None))
    g.add((q, CW.hasCondition, next(g.subjects(CW.observedValue, None))))


def _wrong_answer_type(g):
    a = _a(g, CW.InterventionalAnswer)
    g.set((a, CW.forQuery, _q(g, CW.ConditionalQuery)))


def _answer_about_x(g):
    a = _a(g, CW.InterventionalAnswer)
    g.set((a, CW.onNode, _node(g, "Hospital.careQuality")))
    g.set((a, CW.onProperty, URIRef(KG + "careQuality")))


def _answer_about_z(g):
    a = _a(g, CW.InterventionalAnswer)
    g.set((a, CW.onNode, _node(g, "Patient.smoking")))
    g.set((a, CW.onProperty, URIRef(KG + "smoking")))


def _property_mismatch(g):
    g.set((_a(g, CW.InterventionalAnswer), CW.onProperty, URIRef(KG + "age")))


def _probability_above_one(g):
    g.set((_a(g, CW.InterventionalAnswer), CW.probability, Literal(1.3, datatype=XSD.double)))


def _probability_and_std(g):
    g.add((_a(g, CW.InterventionalAnswer), CW.std, Literal(0.2, datatype=XSD.double)))


def _factual_on_an_interventional_answer(g):
    g.add((_a(g, CW.InterventionalAnswer), CW.factualValue, Literal("Short")))


def _model_on_another_graph(g):
    m = next(g.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), CW.CausalModel))
    g.set((m, CW.onGraph, URIRef("http://sdm-causalway.org/graph/clinic-curated")))


def _duplicate_answer(g):
    a = _a(g, CW.InterventionalAnswer)
    twin = URIRef(str(a) + "-twin")
    for p, o in list(g.predicate_objects(a)):
        g.add((twin, p, o))


def _no_uncertainty(g):
    g.remove((_a(g, CW.InterventionalAnswer), CW.probability, None))


def _non_invertible_counterfactual(g):
    m = next(g.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), CW.CausalModel))
    for mech in g.objects(m, CW.hasMechanism):
        if g.value(mech, CW.onNode) == _node(g, "Patient.survival"):
            g.set((mech, CW.invertible, Literal(False)))


BROKEN = [
    (_cycle, "Violation", "directed cycle"),
    (_self_loop, "Violation", "no self-loop"),
    (_duplicate_edge, "Violation", "Duplicate edge"),
    (_epsilon_across_classes, "Violation", "Assumption 1: an edge with no relational path"),
    (_wrong_direction, "Violation", "does not lead from"),
    (_weightless_raw_edge, "Violation", "must be curated"),
    (_duplicate_node, "Violation", "Node duplicates"),
    (_conditional_with_do, "Violation", "a do() makes it interventional"),
    (_interventional_about_an_entity, "Violation", "one entity is a counterfactual"),
    (_x_meets_z, "Violation", "X ∩ Z"),
    (_intervened_twice, "Violation", "intervened on twice"),
    (_counterfactual_with_both_kinds_of_evidence, "Violation", "exactly one of cw:aboutEntity"),
    (_wrong_answer_type, "Violation", "An InterventionalAnswer answers an InterventionalQuery"),
    (_answer_about_x, "Violation", "V ∩ X"),
    (_answer_about_z, "Violation", "V ∩ Z"),
    (_property_mismatch, "Violation", "cw:onProperty is not the cw:property"),
    (_probability_above_one, "Violation", None),
    (_probability_and_std, "Violation", None),
    (_factual_on_an_interventional_answer, "Violation", None),
    (_model_on_another_graph, "Violation", "was not fitted on the query's graph"),
    (_duplicate_answer, "Violation", "Duplicate answer"),
    (_no_uncertainty, "Warning", "neither cw:probability nor cw:std"),
    (_non_invertible_counterfactual, "Warning", "not invertible"),
]


@pytest.mark.parametrize("break_it, severity, message", BROKEN,
                         ids=[b[0].__name__.strip("_") for b in BROKEN])
def test_each_constraint_is_reported(valid, break_it, severity, message):
    g = Graph()
    for t in valid:
        g.add(t)
    break_it(g)
    report = _report(g)
    assert report, f"{break_it.__name__} went unreported"
    assert any(s == severity and (message is None or message in m) for s, m in report), report
