"""`counterfactual()` draws in one batched abduction pass, not one pass per draw.

Abduction and evaluation are row-wise and independent across rows, so "n draws
for this unit" and "one draw for n tiled copies of this unit" are the same
computation. That identity is what makes the batching in
``causalway.inference.counterfactual`` legal, and it is what these tests pin:
tiling must not change the distribution, must not correlate the draws, and must
not quietly change what ``n_samples`` counts or what ``per_row`` is about.

``CF_BATCH_ROWS`` is monkeypatched down to 1 to reproduce the old serial loop
exactly (one draw per batch), which gives a reference to compare against
without keeping a second copy of the implementation around to rot.

Categorical counterfactuals are a Gumbel-max *draw*, not an inversion, so the
two paths are identically distributed and **not** bit-identical — the
comparisons below are therefore against Monte-Carlo error, not equality. The
intervention is chosen to leave the target genuinely stochastic; a do() on a
childless root, or one the coupling is sticky under, collapses the target to a
point mass and would make any of these assertions pass for the wrong reason.
"""

import os

import pytest

pytest.importorskip("dowhy")
pytest.importorskip("pgmpy")

import numpy as np  # noqa: E402

from causalway import inference  # noqa: E402
from causalway.model import CausalModel  # noqa: E402
from causalway.nodes import build_nodes  # noqa: E402
from causalway.result import OntologicalCausalGraph  # noqa: E402
from causalway.sources import resolve_schema  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_TTL = os.path.join(_ROOT, "kgs", "ttls", "synthetic_clinic.ttl")

SUBGRAPH = [
    ("Patient.tumorStage", "Patient.survival"),
    ("Patient.tumorStage", "Therapy.dosage"),
    ("Therapy.dosage", "Therapy.toxicity"),
    ("Therapy.toxicity", "Patient.survival"),
]
NODE_NAMES = ["Patient.survival", "Patient.tumorStage", "Therapy.dosage", "Therapy.toxicity"]
TARGET = "Patient.survival"


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(CLINIC_TTL):
        pytest.skip(f"{CLINIC_TTL} not present")
    schema = resolve_schema(CLINIC_TTL)
    by_name = {n.name: n for n in build_nodes(schema, include_object_properties=False)}
    nodes = [by_name[n] for n in NODE_NAMES]
    index = {n: i for i, n in enumerate(NODE_NAMES)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=int)
    for cause, effect in SUBGRAPH:
        adj[index[cause], index[effect]] = 1
    ocg = OntologicalCausalGraph(nodes=nodes, adj=adj)
    return CausalModel.fit(ocg, CLINIC_TTL, on_missing="drop", random_state=0)


@pytest.fixture(scope="module")
def entity(model):
    var = model.spec.mat.entity_ids.columns[0]
    return str(model.spec.mat.entity_ids[var].iloc[0])


@pytest.fixture(scope="module")
def do(model, entity):
    """An intervention that leaves `TARGET` genuinely stochastic."""
    stage = sorted(set(model.spec.data["Patient.tumorStage"].astype(str)))[0]
    return {(entity, "Patient.tumorStage"): stage}


def _run(model, do, entity, num_samples):
    return model.counterfactual(dict(do), entity=entity, target=TARGET,
                                num_samples=num_samples)


def test_the_target_is_actually_stochastic(model, do, entity):
    """Guard for the other tests: a point mass would make them vacuous."""
    answer = _run(model, do, entity, 200)
    assert answer.distribution is not None
    assert max(answer.distribution.values()) < 0.99, (
        f"{TARGET} collapsed to a point mass under this intervention "
        f"({answer.distribution}); the equivalence tests below would pass trivially."
    )


def test_n_samples_counts_draws_times_rows_however_it_is_chunked(
        model, do, entity, monkeypatch):
    rows = len(inference.population_rows(model.spec.mat, entity))
    for cap in (1, 7, 20_000):
        monkeypatch.setattr(inference, "CF_BATCH_ROWS", cap)
        answer = _run(model, do, entity, 40)
        assert answer.n_samples == 40 * rows, f"cap={cap}"


def test_per_row_is_one_draw_over_this_entitys_own_rows(model, do, entity, monkeypatch):
    """Batching tiles the frame; `per_row` must still be the unit, not the tiling."""
    rows = inference.population_rows(model.spec.mat, entity)
    for cap in (1, 20_000):
        monkeypatch.setattr(inference, "CF_BATCH_ROWS", cap)
        answer = _run(model, do, entity, 40)
        assert list(answer.per_row.columns) == [TARGET], f"cap={cap}"
        assert len(answer.per_row) == len(rows), f"cap={cap}"
        assert list(answer.per_row.index) == list(rows), f"cap={cap}"


def test_batched_and_serial_agree_within_monte_carlo_error(model, do, entity, monkeypatch):
    n = 400
    monkeypatch.setattr(inference, "CF_BATCH_ROWS", 1)       # the old serial loop
    serial = _run(model, do, entity, n)
    monkeypatch.setattr(inference, "CF_BATCH_ROWS", 20_000)  # one tiled pass
    batched = _run(model, do, entity, n)

    assert set(serial.distribution) == set(batched.distribution)
    # Two independent estimates of the same proportion: sd of the difference is
    # sqrt(2) * sqrt(p(1-p)/N). Three of those is a ~99.7% band, per level.
    N = min(serial.n_samples, batched.n_samples)
    for level, p_serial in serial.distribution.items():
        p_batched = batched.distribution[level]
        se = np.sqrt(2 * max(p_serial * (1 - p_serial), 1e-6) / N)
        assert abs(p_serial - p_batched) < 3 * se, (
            f"{level}: serial {p_serial:.4f} vs batched {p_batched:.4f}, "
            f"3 SE = {3 * se:.4f}"
        )


def test_the_draws_in_one_batch_are_not_correlated(model, do, entity, monkeypatch):
    """Tiling must not make `n_samples` overstate the precision it claims.

    If the tiled rows shared noise, repeated batched runs would scatter *more*
    than binomial — the estimate would carry the precision of far fewer than
    `n_samples` independent draws. Measured against the binomial SE rather than
    asserted from the vectorisation argument alone.
    """
    n, repeats = 200, 8
    monkeypatch.setattr(inference, "CF_BATCH_ROWS", 20_000)
    runs = [_run(model, do, entity, n) for _ in range(repeats)]
    level = max(runs[0].distribution, key=runs[0].distribution.get)

    ps = np.array([r.distribution.get(level, 0.0) for r in runs])
    N = runs[0].n_samples
    binomial_se = np.sqrt(max(ps.mean() * (1 - ps.mean()), 1e-6) / N)
    observed_sd = ps.std(ddof=1)
    # Over-dispersion is the failure; with 8 repeats the sd estimate is itself
    # noisy, so the bar is deliberately loose — it catches shared noise (which
    # would inflate this several-fold), not a 20% wobble.
    assert observed_sd < 2.5 * binomial_se, (
        f"observed sd {observed_sd:.5f} vs binomial SE {binomial_se:.5f} "
        f"— the batched draws look correlated"
    )


def test_batching_is_not_slower_than_the_loop_it_replaced(model, do, entity, monkeypatch):
    """The whole point of the change, asserted rather than only measured once.

    The threshold is generous (a 5x floor against a measured ~150x) because CI
    timing is noisy and this test exists to catch the batching being *undone*,
    not to police a performance number.
    """
    import time

    n = 100
    monkeypatch.setattr(inference, "CF_BATCH_ROWS", 1)
    t0 = time.perf_counter()
    _run(model, do, entity, n)
    serial = time.perf_counter() - t0

    monkeypatch.setattr(inference, "CF_BATCH_ROWS", 20_000)
    t0 = time.perf_counter()
    _run(model, do, entity, n)
    batched = time.perf_counter() - t0

    assert batched * 5 < serial, f"serial {serial:.2f}s vs batched {batched:.2f}s"
