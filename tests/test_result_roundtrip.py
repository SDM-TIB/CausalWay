"""``from_rdf(to_rdf(ocg))`` must reproduce ``adj``, ``weights``, node order and ``gid`` bit-for-bit.

Uses the Turtle files Plan 1's notebook 01 already committed under
``results/sclc/`` — no discovery run needed, and no dowhy/pgmpy dependency,
so this file exercises the RDF (de)serialisation in isolation.
"""

import os

import numpy as np
import pytest

from causalway.result import OntologicalCausalGraph
from causalway.sources import resolve_graph

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_FILES = [
    os.path.join(_ROOT, "results", "sclc", "GES.ttl"),
    os.path.join(_ROOT, "results", "sclc", "PC.ttl"),
    os.path.join(_ROOT, "results", "sclc", "NOTEARS.ttl"),
]


@pytest.mark.parametrize("path", RESULT_FILES)
def test_double_roundtrip_reproduces_adjacency(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")

    ocg1 = OntologicalCausalGraph.from_rdf(resolve_graph(path))
    g2 = ocg1.to_rdf()
    ocg2 = OntologicalCausalGraph.from_rdf(g2)

    assert ocg1.gid == ocg2.gid
    assert [n.name for n in ocg1.nodes] == [n.name for n in ocg2.nodes]
    assert np.array_equal(ocg1.adj, ocg2.adj)
    # `.weight(i, j)` is what every downstream consumer reads (defaults to 1.0
    # for an edge when no explicit weight matrix is set) — compare via that,
    # not raw `.weights`, since to_rdf always emits an explicit weight literal.
    for i, j in ocg1.edges():
        assert ocg1.weight(i, j) == pytest.approx(ocg2.weight(i, j))
    assert ocg1.method == ocg2.method
    assert ocg1.constrained == ocg2.constrained


def test_from_rdf_recovers_uniquified_names():
    # Build an OCG whose nodes collide on the base name, forcing the `_2`
    # suffix nodes.py._with_name uses, and check the label survives to_rdf/from_rdf.
    from rdflib import Namespace, RDF, RDFS

    from causalway.constraints import EdgeConstraint
    from causalway.nodes import PropertyNode

    EX = Namespace("http://example.org/")
    n1 = PropertyNode(domain=EX.Patient, prop=EX.stage, range_=EX.string, kind="data")
    n2 = PropertyNode(domain=EX.Patient, prop=EX.stage, range_=EX.integer, kind="data")
    from causalway.nodes import _with_name
    n2 = _with_name(n2, "Patient.stage_2")

    adj = np.array([[0, 1], [0, 0]])
    constraint = EdgeConstraint(names=[n1.name, n2.name], nodes=[n1, n2], allowed=~np.eye(2, dtype=bool))
    ocg = OntologicalCausalGraph.from_discovery(adj, constraint, method="TEST")

    g = ocg.to_rdf()
    ocg2 = OntologicalCausalGraph.from_rdf(g)
    assert [n.name for n in ocg2.nodes] == ["Patient.stage", "Patient.stage_2"]
    assert np.array_equal(ocg.adj, ocg2.adj)


def test_ambiguous_graph_requires_graph_id_or_method():
    bundle = os.path.join(_ROOT, "results", "sclc", "all_methods.ttl")
    if not os.path.exists(bundle):
        pytest.skip("all_methods.ttl not present")
    g = resolve_graph(bundle)
    with pytest.raises(ValueError):
        OntologicalCausalGraph.from_rdf(g)
    # Disambiguated by method works.
    ocg = OntologicalCausalGraph.from_rdf(g, method="GES")
    assert ocg.method == "GES"


def test_to_dag_strict_raises_on_cycle():
    from causalway.constraints import EdgeConstraint
    from causalway.nodes import PropertyNode
    from rdflib import Namespace

    EX = Namespace("http://example.org/")
    a = PropertyNode(domain=EX.A, prop=EX.a, range_=EX.string, kind="data")
    b = PropertyNode(domain=EX.B, prop=EX.b, range_=EX.string, kind="data")
    adj = np.array([[0, 1], [1, 0]])  # a <-> b cycle
    constraint = EdgeConstraint(names=[a.name, b.name], nodes=[a, b],
                                allowed=~np.eye(2, dtype=bool))
    ocg = OntologicalCausalGraph.from_discovery(adj, constraint, method="TEST")
    with pytest.raises(ValueError, match="not acyclic"):
        ocg.to_dag(strategy="strict")


def test_to_dag_weight_breaks_lowest_weight_edge():
    from causalway.constraints import EdgeConstraint
    from causalway.nodes import PropertyNode
    from rdflib import Namespace

    EX = Namespace("http://example.org/")
    a = PropertyNode(domain=EX.A, prop=EX.a, range_=EX.string, kind="data")
    b = PropertyNode(domain=EX.B, prop=EX.b, range_=EX.string, kind="data")
    adj = np.array([[0, 1], [1, 0]])
    weights = np.array([[0.0, 0.9], [0.1, 0.0]])
    constraint = EdgeConstraint(names=[a.name, b.name], nodes=[a, b],
                                allowed=~np.eye(2, dtype=bool))
    ocg = OntologicalCausalGraph.from_discovery(adj, constraint, weights=weights, method="TEST")
    dag = ocg.to_dag(strategy="weight")
    # The b->a edge (weight 0.1) is the weaker one and should be dropped.
    assert dag.adj[1, 0] == 0
    assert dag.adj[0, 1] == 1
