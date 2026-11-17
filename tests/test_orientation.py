"""CPDAG -> DAG: ``extend_to_dag`` and ``to_dag(strategy="extension")``.

PC and GES return a CPDAG, whose undirected edges arrive as pairs of opposite
edges; an exported ``cw:OntologicalCausalGraph`` is a DAG.  The orientation must
pick *one member of the Markov equivalence class* — same skeleton, same
v-structures, no cycle — preferring the direction Assumption 1 admits when it
admits only one, and it must never be silent: the export refuses a cyclic graph.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from causalway.result import (
    OntologicalCausalGraph, _is_acyclic, _v_structures, extend_to_dag,
)

from _fixtures import clinic_ocg


def _pdag(n, directed=(), undirected=()):
    adj = np.zeros((n, n), dtype=int)
    for i, j in directed:
        adj[i, j] = 1
    for i, j in undirected:
        adj[i, j] = adj[j, i] = 1
    return adj


def _skeleton(adj):
    return ((adj + adj.T) > 0).astype(int)


def _assert_extension(cpdag, dag):
    assert _is_acyclic(dag)
    assert np.array_equal(_skeleton(cpdag), _skeleton(dag))
    assert _v_structures(dag) == _v_structures(cpdag)
    directed = (cpdag != 0) & (cpdag.T == 0)
    assert np.all(dag[directed] == 1), "a directed edge of the CPDAG was flipped"


def test_an_undirected_chain_never_becomes_a_collider():
    # 0 - 1 - 2, 0 and 2 non-adjacent: 0 -> 1 <- 2 would be a new v-structure.
    cpdag = _pdag(3, undirected=[(0, 1), (1, 2)])
    for seed in range(20):
        dag, oriented = extend_to_dag(cpdag, seed=seed)
        _assert_extension(cpdag, dag)
        assert not (dag[0, 1] and dag[2, 1])
        assert len(oriented) == 2


def test_an_existing_v_structure_forces_the_orientation_below_it():
    # 0 -> 2 <- 1 is a v-structure; 2 - 3 must become 2 -> 3, since 3 -> 2
    # would make 0 -> 2 <- 3 and 1 -> 2 <- 3, two new ones.
    cpdag = _pdag(4, directed=[(0, 2), (1, 2)], undirected=[(2, 3)])
    for seed in range(10):
        dag, oriented = extend_to_dag(cpdag, seed=seed)
        _assert_extension(cpdag, dag)
        assert dag[2, 3] == 1 and oriented == {(2, 3)}


def test_the_seed_chooses_among_equivalent_dags_and_is_reproducible():
    # A 4-node undirected clique has 24 equivalent DAGs (every topological order).
    cpdag = _pdag(4, undirected=list(itertools.combinations(range(4), 2)))
    seen = set()
    for seed in range(30):
        dag, _ = extend_to_dag(cpdag, seed=seed)
        _assert_extension(cpdag, dag)
        seen.add(dag.tobytes())
        again, _ = extend_to_dag(cpdag, seed=seed)
        assert np.array_equal(dag, again)
    assert len(seen) > 1


def test_assumption_one_decides_when_it_admits_one_direction():
    cpdag = _pdag(3, undirected=[(0, 1), (1, 2)])
    admissible = np.ones((3, 3), dtype=bool)
    admissible[1, 0] = False                        # only 0 -> 1 is admissible
    for seed in range(10):
        dag, _ = extend_to_dag(cpdag, admissible=admissible, seed=seed)
        _assert_extension(cpdag, dag)
        assert dag[0, 1] == 1
        assert dag[1, 2] == 1                       # 2 -> 1 would make 0 -> 1 <- 2


def test_knowledge_that_would_add_a_v_structure_is_not_applied():
    cpdag = _pdag(3, undirected=[(0, 1), (1, 2)])
    admissible = np.ones((3, 3), dtype=bool)
    admissible[1, 0] = admissible[1, 2] = False     # asks for 0 -> 1 <- 2: a new collider
    dag, _ = extend_to_dag(cpdag, admissible=admissible, seed=0)
    _assert_extension(cpdag, dag)


def test_a_directed_cycle_is_not_an_orientation_problem():
    cyclic = _pdag(3, directed=[(0, 1), (1, 2), (2, 0)])
    dag, oriented = extend_to_dag(cyclic)
    assert np.array_equal(dag, cyclic) and oriented == set()


def test_to_dag_extension_records_the_choice():
    ocg = clinic_ocg(edges=[("Patient.smoking", "Patient.tumorStage"),
                            ("Patient.tumorStage", "Patient.smoking")])
    dag = ocg.to_dag(strategy="extension", seed=3)
    assert _is_acyclic(dag.adj) and len(dag.edges()) == 1
    assert dag.params["orientedEdges"] == 1 and dag.params["orientationSeed"] == 3
    assert dag.oriented_edges == set(dag.edges())
    [edge] = dag.to_json()["edges"]
    assert "Markov equivalence class" in edge["label"]
    # A DAG is left alone, and records nothing.
    plain = clinic_ocg()
    assert plain.to_dag(strategy="extension") is plain


def test_the_export_refuses_a_cyclic_graph():
    ocg = clinic_ocg(edges=[("Patient.smoking", "Patient.tumorStage"),
                            ("Patient.tumorStage", "Patient.smoking")])
    with pytest.raises(ValueError, match="to_dag"):
        ocg.to_json()
    OntologicalCausalGraph.from_rdf(ocg.to_dag(strategy="extension").to_rdf())
