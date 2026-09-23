"""A query has 1..n targets — the shape you ask in is the shape you get back.

Companion to ``test_inference.py``, which cannot run at all right now: its
fixture goes through ``load_ocg(results/sclc/GES.ttl)``, and reading an OCG
back from Turtle is broken independently of anything here (the same
pre-existing failure as ``test_result_roundtrip.py`` / ``test_sources.py``).
So this module builds its causal graph **in memory** from
``causalway.synthetic``'s ground truth instead, which keeps the multi-target
engines covered without waiting on that bug.

Deliberately small: four nodes out of the eleven, so a real ``gcm`` fit stays
a few seconds rather than half a minute.
"""

import os

import pytest

pytest.importorskip("dowhy")
pytest.importorskip("pgmpy")

import numpy as np  # noqa: E402

from causalway.model import CausalModel  # noqa: E402
from causalway.nodes import build_nodes  # noqa: E402
from causalway.result import OntologicalCausalGraph  # noqa: E402
from causalway.sources import resolve_schema  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_TTL = os.path.join(_ROOT, "kgs", "ttls", "synthetic_clinic.ttl")

# A connected four-node slice of causalway.synthetic's ground truth.
SUBGRAPH = [
    ("Patient.tumorStage", "Patient.survival"),
    ("Patient.tumorStage", "Therapy.dosage"),
    ("Therapy.dosage", "Therapy.toxicity"),
    ("Therapy.toxicity", "Patient.survival"),
]
NODE_NAMES = ["Patient.survival", "Patient.tumorStage", "Therapy.dosage", "Therapy.toxicity"]


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
def evidence(model):
    """One observed value for a node that is nobody's target below."""
    return {"Patient.tumorStage": str(model.spec.data["Patient.tumorStage"].iloc[0])}


def test_a_string_target_answers_one_answer(model, evidence):
    answer = model.condition("Patient.survival", evidence)
    assert answer.target == "Patient.survival"
    assert answer.distribution


def test_a_collection_target_answers_one_answer_per_target(model, evidence):
    answers = model.condition(["Patient.survival", "Therapy.toxicity"], evidence)
    assert set(answers) == {"Patient.survival", "Therapy.toxicity"}
    assert [a.target for a in answers.values()] == ["Patient.survival", "Therapy.toxicity"]


def test_a_one_element_collection_is_still_a_collection(model, evidence):
    """The request shape decides the response shape, so a client that always sends a
    list never has to branch on how many targets it happened to ask about."""
    answers = model.condition(["Patient.survival"], evidence)
    assert isinstance(answers, dict) and set(answers) == {"Patient.survival"}


def test_a_set_target_is_answered_in_a_deterministic_order(model, evidence):
    answers = model.condition({"Therapy.toxicity", "Patient.survival"}, evidence)
    assert list(answers) == sorted(["Therapy.toxicity", "Patient.survival"])


def test_batched_answers_match_the_single_target_answers(model, evidence):
    """Same query, asked one at a time or together: this model is all-categorical, so
    both go through the exact pgmpy tier and must agree exactly, not approximately."""
    batch = model.condition(["Patient.survival", "Therapy.toxicity"], evidence)
    for node, answer in batch.items():
        alone = model.condition(node, evidence)
        assert answer.backend == alone.backend == "pgmpy-exact"
        assert answer.distribution == pytest.approx(alone.distribution)


def test_batched_answers_share_one_computation(model, evidence):
    """One weighted sample set for the whole batch — equal `ess` is the observable
    consequence, and disagreeing cards are what it prevents (Plan 3 W23)."""
    batch = model.condition(["Patient.survival", "Therapy.toxicity"], evidence,
                            backend="likelihood-weighting", num_samples=2000)
    esses = {a.ess for a in batch.values()}
    assert len(esses) == 1


def test_a_root_target_is_answered_by_its_marginal_not_tier_0(model):
    """A root's mechanism is a plain distribution with no noise to draw; tier 0 used
    to call draw_noise_samples on it and crash (notebook 05's multi-target cell)."""
    alone = model.condition("Patient.tumorStage", {})
    assert alone.backend != "mechanism"
    assert alone.distribution and sum(alone.distribution.values()) == pytest.approx(1.0)
    batch = model.condition(["Patient.tumorStage", "Therapy.toxicity"], {})
    assert set(batch) == {"Patient.tumorStage", "Therapy.toxicity"}


def test_an_unknown_target_is_named(model, evidence):
    with pytest.raises(ValueError, match="Therapy.nope"):
        model.condition(["Patient.survival", "Therapy.nope"], evidence)


def test_no_target_is_refused(model, evidence):
    with pytest.raises(ValueError, match="at least one target"):
        model.condition([], evidence)


def test_intervene_batches_targets_off_one_draw(model):
    value = str(model.spec.data["Therapy.dosage"].iloc[0])
    answers = model.intervene({"Therapy.dosage": value},
                              target=["Therapy.toxicity", "Patient.survival"],
                              num_samples=2000)
    assert set(answers) == {"Therapy.toxicity", "Patient.survival"}
    # do(dosage) pins the intervened node, so every answer was drawn from the same
    # mutilated model — the sample count is the shared draw, not a per-target one.
    assert len({a.n_samples for a in answers.values()}) == 1
    assert answers["Therapy.toxicity"].distribution


def test_intervene_contrast_is_reported_per_target(model):
    values = model.spec.data["Therapy.dosage"].astype(str).unique().tolist()
    if len(values) < 2:
        pytest.skip("Therapy.dosage has a single level in this KG")
    answers = model.intervene({"Therapy.dosage": values[0]},
                              target=["Therapy.toxicity", "Patient.survival"],
                              reference={"Therapy.dosage": values[1]}, num_samples=2000)
    for answer in answers.values():
        assert isinstance(answer.effect, dict)
        # A per-level contrast is a difference of two distributions: it sums to zero.
        assert sum(answer.effect.values()) == pytest.approx(0.0, abs=1e-9)


def test_counterfactual_batches_targets_off_one_abduction(model):
    entity_var = model.spec.mat.entity_ids.columns[0]
    entity = str(model.spec.mat.entity_ids[entity_var].iloc[0])
    value = str(model.spec.data["Therapy.dosage"].iloc[0])
    answers = model.counterfactual({(entity, "Therapy.dosage"): value},
                                   entity=entity,
                                   target=["Therapy.toxicity", "Patient.survival"],
                                   num_samples=25)
    assert set(answers) == {"Therapy.toxicity", "Patient.survival"}
    for node, answer in answers.items():
        assert answer.entity == entity
        assert answer.target == node
        # `per_row` is this entity's own rows — the provenance that makes the answer
        # about a unit rather than a population.
        assert list(answer.per_row.columns) == [node]
