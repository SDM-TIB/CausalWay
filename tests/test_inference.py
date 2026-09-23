"""Integration tests for CausalModel.fit + the condition/intervene/counterfactual engines.

Heavier than the rest of the suite (fits a real SCM), so it's scoped to a
small ``limit=`` slice of SCLC and to one module-level fixture shared by
every test. Requires ``dowhy``/``pgmpy``/``scikit-learn`` (the ``rdfenv``
environment); skipped cleanly if they're unavailable.
"""

import os

import pytest

pytest.importorskip("dowhy")
pytest.importorskip("pgmpy")

from causalway.model import CausalModel  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCLC_TTL = os.path.join(_ROOT, "kgs", "ttls", "SCLC_patients.ttl")
GES_TTL = os.path.join(_ROOT, "results", "sclc", "GES.ttl")


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(GES_TTL):
        pytest.skip(f"{GES_TTL} not present — run notebook 01 / run_kg_discovery first")
    # Fit on the full SCLC KG (4279 rows): every column's rarest class has
    # >= 217 examples, which gcm.auto's internal (non-stratified) k-fold CV
    # needs to avoid a fold with a class unseen in training. A `limit=`
    # slice (e.g. 300, taken in SPARQL result order, not a random sample)
    # undersamples the rarer classes of `episodeType`/`stage` badly enough
    # to make that CV fail outright.
    return CausalModel.fit(GES_TTL, SCLC_TTL, random_state=0)


def test_fit_produces_invertible_non_root_mechanisms(model):
    table = model.mechanism_table()
    non_root = table[~table["is_root"]]
    assert len(non_root) > 0, "GES on SCLC should produce at least one non-root node"
    assert non_root["invertible"].all(), (
        "every non-root mechanism must be invertible after CausalModel.fit's E1 override"
    )
    gumbel = non_root[non_root["dtype"] == "categorical"]
    assert (gumbel["mechanism_type"] == "InvertibleClassifierFCM").all()


def test_all_columns_categorical_selects_pgmpy_exact_tier(model):
    # SCLC is entirely categorical, so backend="auto" should reach tier 1.
    from causalway.inference import _all_discrete
    assert _all_discrete(model)


def test_tier0_and_tier1_and_tier2_agree_on_a_direct_parent_query(model):
    """Regression test for the backend ladder: three tiers, one answer (to MC error)."""
    target = model.spec.columns[-1]
    parents = list(model.scm.graph.predecessors(target))
    if not parents:
        pytest.skip(f"{target} is a root node in this discovered graph; pick another target")
    parent = parents[0]
    example_value = str(model.spec.data[parent].iloc[0])
    evidence = {parent: example_value}

    a0 = model.condition(target, evidence, backend="mechanism")
    a1 = model.condition(target, evidence, backend="pgmpy-exact")
    a2 = model.condition(target, evidence, backend="likelihood-weighting", num_samples=20_000)

    assert a0.predicted == a1.predicted == a2.predicted or _close_distributions(a0, a1, a2)


def _close_distributions(*answers, tol=0.1):
    dists = [a.distribution for a in answers if a.distribution]
    if len(dists) < 2:
        return False
    keys = set(dists[0])
    for d in dists[1:]:
        keys &= set(d)
    return all(
        abs(dists[0][k] - d[k]) < tol for d in dists[1:] for k in keys
    )


def test_intervene_returns_a_distribution_for_categorical_target(model):
    target = model.spec.columns[0]
    do_node = next((c for c in model.spec.columns if c != target), None)
    value = str(model.spec.data[do_node].iloc[0])
    answer = model.intervene({do_node: value}, target=target, num_samples=500)
    assert answer.kind == "interventional"
    if model.spec.dtypes.get(target) in ("categorical", "ordinal"):
        assert answer.distribution
        assert abs(sum(answer.distribution.values()) - 1.0) < 1e-6


def test_counterfactual_is_entity_level_and_stochastic_for_categorical(model):
    entity = str(model.spec.mat.entity_ids.iloc[0, 0])
    target = model.spec.columns[0]
    do_node = next((c for c in model.spec.columns if c != target), model.spec.columns[0])
    value = str(model.spec.data[do_node].iloc[1])
    answer = model.counterfactual({(entity, do_node): value}, entity=entity, target=target,
                                 num_samples=30)
    assert answer.kind == "counterfactual"
    assert answer.entity == entity
    if model.spec.dtypes.get(target) in ("categorical", "ordinal"):
        assert answer.coupling == "gumbel-max"
