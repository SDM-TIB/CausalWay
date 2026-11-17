"""The GES / GES-Prior learner in ``algs/ges_prior``.

Until the WWW'27 resource revision the learner had no tests at all, which is how
a backward phase that could not delete an edge survived eighteen sessions: every
existing test exercised what happens to a graph *after* discovery. These tests
pin the four things that were wrong, each against the smallest input that shows
it, plus the priors contract the web service relies on.

``legacy=True`` arms are asserted too. They document the defect the fix removes;
if someone "repairs" the legacy path, ``experiments/e1_ges_correctness.py`` would
silently start comparing the fix with itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algs.ges_prior.learn_bn import _normalise_priors, discover_structure
from algs.ges_prior.operators.deleter import delete
from algs.ges_prior.operators.ges_search import bes
from algs.ges_prior.operators.graph import (
    semi_directed_path,
    semi_directed_reachable,
    _to_dag,
    dag_to_cpdag,
    enforce_directions,
    is_dag,
    orient_by_B,
)
from algs.ges_prior.entity_check import entity_level_prune
from algs.ges_prior.scores.prior_score import PriorScore, prior_weights


def _adj(n: int, edges) -> np.ndarray:
    G = np.zeros((n, n), dtype=int)
    for i, j in edges:
        G[i, j] = 1
    return G


def _und(n: int, pairs) -> np.ndarray:
    return _adj(n, [(i, j) for i, j in pairs] + [(j, i) for i, j in pairs])


class _SparsityScore:
    """A score that prefers fewer parents, so every deletion is an improvement."""

    def __init__(self, n: int):
        self.variables = list(range(n))

    def local_score(self, variable, parents) -> float:
        return -float(len(parents))


# ------------------------------------------------------------ DAG -> CPDAG --

def test_cpdag_of_a_chain_is_undirected():
    assert np.array_equal(dag_to_cpdag(_adj(3, [(0, 1), (1, 2)])), _und(3, [(0, 1), (1, 2)]))


def test_cpdag_keeps_a_v_structure():
    G = _adj(3, [(0, 2), (1, 2)])
    assert np.array_equal(dag_to_cpdag(G), G)


def test_cpdag_without_v_structure_is_fully_undirected():
    # 0->1, 0->2, 0->3, 2->3 has no unshielded collider (0-2-3 is a triangle),
    # so every edge is reversible. gCastle's routine keeps 0->3 compelled.
    G = _adj(4, [(0, 1), (0, 2), (0, 3), (2, 3)])
    assert np.array_equal(dag_to_cpdag(G), _und(4, [(0, 1), (0, 2), (0, 3), (2, 3)]))
    legacy = dag_to_cpdag(G, legacy=True)
    assert legacy[0, 3] == 1 and legacy[3, 0] == 0


def test_cpdag_applies_meek_r1():
    # 0 -> 2 <- 1 is a v-structure; 2 - 3 must then point away from 2.
    G = _adj(4, [(0, 2), (1, 2), (2, 3)])
    assert np.array_equal(dag_to_cpdag(G), G)


# ------------------------------------------------------------------- BES ---

def test_backward_phase_deletes_edges():
    C = _und(3, [(0, 1), (1, 2), (0, 2)])
    out = bes(C.copy(), _SparsityScore(3))
    assert out.sum() == 0


def test_legacy_backward_phase_is_a_no_op():
    C = _und(3, [(0, 1), (1, 2), (0, 2)])
    assert np.array_equal(bes(C.copy(), _SparsityScore(3), legacy=True), C)


def test_delete_orients_x_to_h():
    # x - y, y - h, x - h. Deleting x - y with H = {h} makes h a child of y and
    # of x (Chickering 2002, Definition 13).
    x, y, h = 0, 1, 2
    C = delete(x, y, {h}, _und(3, [(x, y), (y, h), (x, h)]))
    assert C[x, y] == 0 and C[y, x] == 0
    assert C[y, h] == 1 and C[h, y] == 0
    assert C[x, h] == 1 and C[h, x] == 0


def test_reachability_agrees_with_path_enumeration():
    # The forward phase's validity test used to enumerate every semi-directed
    # path (exponential); it now asks a reachability question. Same answers.
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 8))
        order = rng.permutation(n)
        dag = np.zeros((n, n), dtype=int)
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < 0.4:
                    dag[order[a], order[b]] = 1
        C = dag_to_cpdag(dag)
        x, y = (int(v) for v in rng.choice(n, 2, replace=False))
        blocked = {int(v) for v in rng.choice(n, int(rng.integers(0, n)), replace=False)} - {x, y}
        by_paths = any(not (set(path) & blocked) for path in semi_directed_path(x, y, C))
        assert semi_directed_reachable(x, y, C, blocked) == by_paths


# ----------------------------------------------------------- orientation ---

def test_to_dag_is_a_member_of_the_class():
    cpdag = _und(4, [(0, 1), (0, 2), (0, 3), (2, 3)])
    dag = _to_dag(cpdag)
    assert is_dag(dag)
    assert np.array_equal(dag_to_cpdag(dag), cpdag)


def test_orient_by_B_cannot_close_a_cycle():
    # An undirected triangle with a prior that prefers 0->1, 1->2, 2->0. Orienting
    # each edge on its own (the old ``oriente_via_B``) returns exactly that cycle.
    cpdag = _und(3, [(0, 1), (1, 2), (0, 2)])
    B = np.full((3, 3), 0.1)
    B[0, 1], B[1, 2], B[2, 0] = 0.9, 0.8, 0.7
    dag = orient_by_B(cpdag, B)
    assert is_dag(dag)
    assert np.array_equal(dag_to_cpdag(dag), cpdag)
    assert dag[0, 1] == 1 and dag[1, 2] == 1          # the two strongest preferences win


def test_orient_by_B_respects_the_constraint_over_the_prior():
    cpdag = _und(3, [(0, 1), (1, 2)])
    B = np.full((3, 3), 0.5)
    B[1, 0], B[0, 1] = 0.99, 0.01                     # the prior wants 1 -> 0 ...
    allowed = np.ones((3, 3), dtype=bool)
    allowed[1, 0] = False                             # ... the schema forbids it
    dag = orient_by_B(cpdag, B, allowed=allowed)
    assert is_dag(dag)
    assert dag[0, 1] == 1 and dag[1, 0] == 0
    assert not (dag.astype(bool) & ~allowed).any()


def test_enforce_directions_reverses_or_drops():
    P = _adj(3, [(0, 1), (1, 2)])
    allowed = np.ones((3, 3), dtype=bool)
    allowed[0, 1] = False                             # reversible
    allowed[1, 2] = allowed[2, 1] = False             # no direction left
    out = enforce_directions(P, allowed)
    assert out[1, 0] == 1 and out[0, 1] == 0
    assert out[1, 2] == 0 and out[2, 1] == 0


# ------------------------------------------------------------ prior score ---

def test_bic_m_b_does_not_punish_an_unoriented_but_likely_edge():
    M = np.array([[0.5, 0.9], [0.9, 0.5]])
    D = np.array([[0.5, 0.5], [0.5, 0.5]])
    has = ~np.eye(2, dtype=bool)
    assert prior_weights("BIC+M+B", M, D, has)[0, 1] > 0     # log(.9/.1) + log(.5)
    assert prior_weights("BIC+B", M, D, has)[0, 1] < 0       # log(.45/.55): the old Phase II


def test_a_pair_without_a_prior_contributes_nothing():
    M = np.array([[0.5, 0.01], [0.01, 0.5]])
    D = np.array([[0.5, 0.99], [0.01, 0.5]])
    none = np.zeros((2, 2), dtype=bool)
    for name in ("BIC+M", "BIC+B", "BIC+M+B"):
        assert not prior_weights(name, M, D, none).any()


def test_prior_weight_is_bounded():
    M = np.array([[0.5, 1.0], [1.0, 0.5]])
    D = np.array([[0.5, 1.0], [0.0, 0.5]])
    W = prior_weights("BIC+M+B", M, D, ~np.eye(2, dtype=bool), eps=1e-3)
    assert np.all(np.abs(W) <= 2 * np.log(999) + 1e-9)


def test_prior_score_adds_lambda_times_weight():
    df = pd.DataFrame({"a": list("0101"), "b": list("0011")})
    W = np.array([[0.0, 2.0], [0.0, 0.0]])
    base = PriorScore(df, "bic-d")
    with_prior = PriorScore(df, "bic-d", W=W, lam=0.5)
    assert with_prior.local_score("b", ("a",)) == pytest.approx(base.local_score("b", ("a",)) + 1.0)
    assert with_prior.local_score("a", ("b",)) == pytest.approx(base.local_score("a", ("b",)))
    assert with_prior.local_score("b", ()) == pytest.approx(base.local_score("b", ()))


# ---------------------------------------------------------- priors contract --

def test_two_matrix_priors_are_read_as_before():
    # The form the web service stores: 0.5 = no answer; B = M * D.
    M = np.array([[0.5, 0.8, 0.5], [0.8, 0.5, 0.5], [0.5, 0.5, 0.5]])
    B = np.array([[0.5, 0.6, 0.5], [0.2, 0.5, 0.5], [0.5, 0.5, 0.5]])
    pri = _normalise_priors({"M": M, "B": B}, 3)
    assert pri["has"][0, 1] and pri["has"][1, 0]
    assert not pri["has"][0, 2] and not pri["has"][1, 2]
    assert pri["D"][0, 1] == pytest.approx(0.75) and pri["D"][1, 0] == pytest.approx(0.25)


def test_old_direction_unknown_encoding_means_one_half():
    # Before the revision an informed pair with no direction stored B = M both ways.
    M = np.array([[0.5, 0.9], [0.9, 0.5]])
    pri = _normalise_priors({"M": M, "B": M.copy()}, 2)
    assert pri["has"][0, 1]
    assert pri["D"][0, 1] == pytest.approx(0.5)


# ------------------------------------------------------------- end to end ---

def _chain_collider_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """a -> c <- b, c -> d; binary, strong and *additive* dependences.

    Additive on purpose: with c = a XOR b each parent is marginally independent
    of c, and no greedy search adds either edge first. That is unfaithfulness,
    not a property of this learner.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    c = (rng.random(n) < 0.1 + 0.4 * a + 0.4 * b).astype(int)
    d = (c ^ (rng.random(n) < 0.1)).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def test_ges_recovers_a_collider_and_its_descendant():
    df = _chain_collider_frame()
    adj = discover_structure(df, algorithm="GES")
    truth = _adj(4, [(0, 2), (1, 2), (2, 3)])
    assert np.array_equal(adj, truth)


def test_ges_prior_without_priors_is_ges_plus_hill_climbing():
    df = _chain_collider_frame()
    adj = discover_structure(df, algorithm="GES-M-B-HL")
    assert is_dag(adj)
    assert np.array_equal(adj, _adj(4, [(0, 2), (1, 2), (2, 3)]))


def test_ges_prior_orients_what_the_data_cannot():
    # x -> y alone: the data cannot orient it, the prior can.
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 2000)
    y = (x ^ (rng.random(2000) < 0.2)).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    M = np.array([[0.5, 0.9], [0.9, 0.5]])
    for d_xy, expected in ((0.9, (0, 1)), (0.1, (1, 0))):
        D = np.array([[0.5, d_xy], [1 - d_xy, 0.5]])
        priors = {"M": M, "B": M * D, "D": D, "has": ~np.eye(2, dtype=bool)}
        for algorithm in ("GES-M-B", "GES-M-B-HL"):
            adj = discover_structure(df, algorithm=algorithm, priors=priors)
            assert adj[expected] == 1 and adj[expected[::-1]] == 0, algorithm


def test_constrained_ges_prior_never_returns_a_forbidden_edge():
    class Constraint:
        def __init__(self, allowed):
            self.allowed = allowed

    df = _chain_collider_frame()
    allowed = np.ones((4, 4), dtype=bool)
    allowed[2, 3] = False                              # c -> d forbidden, d -> c permitted
    allowed[0, 1] = allowed[1, 0] = False
    for algorithm in ("GES", "GES-M-B", "GES-M-B-HL"):
        adj = discover_structure(df, algorithm=algorithm, constraint=Constraint(allowed))
        assert not (adj.astype(bool) & ~allowed).any(), algorithm


# ----------------------------------------------------------- entity score ---

def _replicated_frame(per_dim: int = 60, seed: int = 3):
    """A dimension class with two unrelated attributes, replicated per fact row.

    The twelve entities are written out rather than drawn: x and y agree on 8 of
    12, which is what two fair coins do by chance one time in four, and is the
    situation the entity-level score exists for.
    """
    rng = np.random.default_rng(seed)
    x = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    n_dim = len(x)
    dim = np.repeat(np.arange(n_dim), per_dim)
    fact = np.arange(n_dim * per_dim)
    f = (rng.random(len(fact)) < 0.2 + 0.6 * x[dim]).astype(int)
    df = pd.DataFrame({"D.x": x[dim], "D.y": y[dim], "F.f": f}).astype(str)
    ids = pd.DataFrame({"D": dim, "F": fact})
    owner = {"D.x": "D", "D.y": "D", "F.f": "F"}
    return df, ids, owner


def test_entity_check_scores_each_edge_on_its_owning_entities():
    df, ids, owner = _replicated_frame()
    adj = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])           # D.x -> D.y, D.x -> F.f
    out, report = entity_level_prune(adj, df, owner, ids, return_report=True)
    by_child = {r["child"]: r for r in report}
    assert by_child["D.y"]["n_entities"] == 12 and not by_child["D.y"]["kept"]
    assert "F.f" not in by_child                                # one row per fact already
    assert out.tolist() == [[0, 0, 1], [0, 0, 0], [0, 0, 0]]


def test_entity_check_removes_the_replication_edge_and_keeps_the_real_one():
    # Twelve entities whose two attributes agree by chance, sixty copies of each:
    # on the join the association looks certain.
    df, ids, owner = _replicated_frame()
    on_join = discover_structure(df, algorithm="GES")
    assert on_join[0, 1] or on_join[1, 0]                       # the spurious D.x - D.y edge
    checked = discover_structure(df, algorithm="GES", entity_ids=ids, owner_of=owner)
    assert not checked[0, 1] and not checked[1, 0]
    assert checked[0, 2] or checked[2, 0]                       # the real D.x - F.f edge survives


def test_entity_check_leaves_an_n_side_parent_alone():
    # F.f -> D.x: the child's entity reaches many parent entities, so the family
    # has no one-row-per-entity form and the check must not touch it.
    df, ids, owner = _replicated_frame()
    adj = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    out, report = entity_level_prune(adj, df, owner, ids, return_report=True)
    assert np.array_equal(out, adj) and report == []


def test_entity_check_is_a_no_op_on_a_single_class():
    df = _chain_collider_frame(500)
    ids = pd.DataFrame({"E": np.arange(len(df))})
    owner = {c: "E" for c in df.columns}
    assert np.array_equal(discover_structure(df, algorithm="GES"),
                          discover_structure(df, algorithm="GES", entity_ids=ids, owner_of=owner))
