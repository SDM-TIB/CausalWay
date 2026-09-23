"""Tests for the edge-constraint matrix and its per-library adapters."""

import numpy as np
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL
from rdflib.namespace import XSD

from causalway.ontology import OntologySchema
from causalway.nodes import build_nodes
from causalway.constraints import EdgeConstraint

TOY = Namespace("http://toy/")


def _toy_schema() -> OntologySchema:
    g = Graph()
    for name in ("A", "B", "C"):
        g.add((TOY[name], RDF.type, OWL.Class))
    for cls, prop in [("A", "a1"), ("A", "a2"), ("B", "b1"), ("C", "c1")]:
        p = TOY[prop]
        g.add((p, RDF.type, OWL.DatatypeProperty))
        g.add((p, RDFS.domain, TOY[cls]))
        g.add((p, RDFS.range, XSD.string))
    r = TOY["r"]
    g.add((r, RDF.type, OWL.ObjectProperty))
    g.add((r, RDFS.domain, TOY["A"]))
    g.add((r, RDFS.range, TOY["B"]))
    return OntologySchema.from_graph(g)


def _toy_constraint(direction="both"):
    schema = _toy_schema()
    nodes = build_nodes(schema, include_object_properties=False)
    return schema, nodes, EdgeConstraint.from_schema(
        schema, nodes, relation_direction=direction
    )


def test_node_order():
    _, nodes, _ = _toy_constraint()
    assert [n.name for n in nodes] == ["A.a1", "A.a2", "B.b1", "C.c1"]


def test_allowed_matrix_both():
    _, nodes, c = _toy_constraint(direction="both")
    # C.c1 (index 3) is isolated: no allowed edges in or out.
    assert not c.allowed[3, :].any()
    assert not c.allowed[:, 3].any()
    # intra-class A.a1 <-> A.a2 via epsilon
    assert c.allowed[0, 1] and c.allowed[1, 0]
    # inter-class A -> B and (both) B -> A via r
    assert c.allowed[0, 2] and c.allowed[2, 0]
    assert c.allowed[1, 2] and c.allowed[2, 1]


def test_allowed_matrix_forward():
    _, nodes, c = _toy_constraint(direction="forward")
    assert c.allowed[0, 2] and c.allowed[1, 2]  # A -> B forward
    assert not c.allowed[2, 0] and not c.allowed[2, 1]  # B -> A forbidden
    assert c.allowed[0, 1] and c.allowed[1, 0]  # epsilon still symmetric


def test_stats_pruning():
    _, _, c = _toy_constraint()
    s = c.stats()
    assert s["n_nodes"] == 4
    assert s["n_allowed"] == 6
    assert s["n_forbidden"] == 6
    assert abs(s["pruning_rate"] - 0.5) < 1e-9


def test_lingam_prior_transpose():
    _, nodes, c = _toy_constraint()
    prior = c.to_lingam_prior()
    assert prior.shape == (4, 4)
    # forbid C.c1 (3) -> A.a1 (0): Aknw[0, 3] must be 0 (transposed).
    assert not c.allowed[3, 0]
    assert prior[0, 3] == 0.0
    # allowed A.a1 -> B.b1 (0 -> 2): not forbidden, so prior[2, 0] != 0.
    assert prior[2, 0] == -1.0


def test_adapters_counts():
    _, nodes, c = _toy_constraint()
    assert len(c.forbidden_edges()) == 6
    assert len(c.search_space()) == 6
    assert len(c.exclude_edges()) == 6
    assert c.mask().shape == (4, 4)
    assert int(c.mask().sum()) == 6
    skel = c.skeleton_allowed()
    assert skel.shape == (4, 4)
    # skeleton view is symmetric
    assert (skel == skel.T).all()


def test_castle_priori():
    _, nodes, c = _toy_constraint()
    priori = c.to_castle_priori()
    assert len(priori.forbidden_edges) == 6
    assert priori.matrix.shape == (4, 4)


def test_expert_knowledge():
    _, nodes, c = _toy_constraint()
    ek = c.to_expert_knowledge()
    assert len(ek.search_space) == 6
    skel_ek = c.to_skeleton_expert_knowledge()
    # C.c1 is isolated -> both (C.c1, x) and (x, C.c1) are forbidden.
    assert ("C.c1", "A.a1") in skel_ek.forbidden_edges
    assert ("A.a1", "C.c1") in skel_ek.forbidden_edges


def test_soft_prior():
    _, nodes, c = _toy_constraint()
    B = c.soft_prior(eps=1e-4)
    assert B.shape == (4, 4)
    assert B[0, 2] == 0.5  # allowed -> neutral
    assert B[3, 0] == 1e-4  # forbidden -> eps


def test_subset():
    _, nodes, c = _toy_constraint()
    sub = c.subset([0, 1])
    assert sub.names == ["A.a1", "A.a2"]
    assert sub.allowed.shape == (2, 2)
    assert sub.allowed[0, 1] and sub.allowed[1, 0]
