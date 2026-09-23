"""Tests for InvertibleClassifierFCM (Gumbel-max coupling).

The single most important property (per Plan 2 §3.4) is exact consistency:
``evaluate(pa, estimate_noise(y, pa)) == y`` for every row. Order-invariance
and observational-distribution match are the other two asserted properties.
"""

import numpy as np
import pytest
from scipy.stats import chisquare

from causalway.mechanisms import InvertibleClassifierFCM


class _StubClassifier:
    """A minimal ``ClassificationModel`` stand-in with a fixed probability table."""

    def __init__(self, classes, probs):
        self._classes = list(classes)
        self._probs = np.asarray(probs)

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return np.array(self._classes)[np.argmax(self._probs, axis=1)]

    def predict_probabilities(self, X):
        return self._probs

    @property
    def classes(self):
        return self._classes

    def clone(self):
        return _StubClassifier(self._classes, self._probs)


def _mechanism(seed=0, n=500, k=3):
    rng = np.random.RandomState(seed)
    probs = rng.dirichlet(alpha=[2.0] * k, size=n)
    classes = [f"class_{i}" for i in range(k)]
    mech = InvertibleClassifierFCM(classifier_model=_StubClassifier(classes, probs))
    X = np.zeros((n, 1))
    return mech, X


def test_consistency_exact_inverse():
    mech, X = _mechanism(seed=1)
    n = X.shape[0]
    y = mech.evaluate(X, mech.draw_noise_samples(n)).reshape(-1)
    noise = mech.estimate_noise(y, X)
    y_again = mech.evaluate(X, noise).reshape(-1)
    assert np.array_equal(y, y_again), (
        "estimate_noise -> evaluate must exactly reconstruct the observed class"
    )


def test_consistency_holds_for_every_class_not_just_the_mode():
    # Force the observed class to the *least* likely one for at least some
    # rows, so the test doesn't only exercise the argmax branch.
    mech, X = _mechanism(seed=2)
    probs = mech._classifier_model._probs
    classes = np.array(mech._classifier_model._classes)
    least_likely = classes[np.argmin(probs, axis=1)]
    noise = mech.estimate_noise(least_likely, X)
    recovered = mech.evaluate(X, noise).reshape(-1)
    assert np.array_equal(recovered, least_likely)


def test_order_invariance_of_abduction():
    """Permuting the class order must not change which label abduction recovers."""
    mech, X = _mechanism(seed=3)
    n = X.shape[0]
    y = mech.evaluate(X, mech.draw_noise_samples(n)).reshape(-1)

    classes = mech._classifier_model._classes
    probs = mech._classifier_model._probs
    perm = list(reversed(range(len(classes))))
    permuted = InvertibleClassifierFCM(classifier_model=_StubClassifier(
        [classes[i] for i in perm], probs[:, perm]))

    noise_a = mech.estimate_noise(y, X)
    y_a = mech.evaluate(X, noise_a).reshape(-1)
    noise_b = permuted.estimate_noise(y, X)
    y_b = permuted.evaluate(X, noise_b).reshape(-1)

    assert np.array_equal(y_a, y)
    assert np.array_equal(y_b, y)
    assert np.array_equal(y_a, y_b)


def test_observationally_matches_classifier_fcm_sampling():
    """The Gumbel-max coupling must reproduce ClassifierFCM's sampling distribution."""
    from dowhy.gcm.causal_mechanisms import ClassifierFCM

    classes = ["a", "b", "c"]
    probs_row = np.array([[0.6, 0.3, 0.1]])
    stub = _StubClassifier(classes, probs_row)

    base = ClassifierFCM(classifier_model=stub)
    inv = InvertibleClassifierFCM(classifier_model=stub)

    n = 5000
    pa = np.zeros((n, 1))
    samples_base = base.evaluate(pa, base.draw_noise_samples(n)).reshape(-1)
    samples_inv = inv.evaluate(pa, inv.draw_noise_samples(n)).reshape(-1)

    counts_base = [int(np.sum(samples_base == c)) for c in classes]
    counts_inv = [int(np.sum(samples_inv == c)) for c in classes]
    stat, p = chisquare(counts_inv, f_exp=[max(c, 1) for c in counts_base])
    assert p > 0.001, f"Gumbel-max sampling distribution diverges from ClassifierFCM's (p={p})"


def test_estimate_noise_rejects_unknown_class():
    mech, X = _mechanism(seed=4)
    with pytest.raises(ValueError):
        mech.estimate_noise(np.array(["not_a_class"] * X.shape[0]), X)


def test_clone_returns_invertible_classifier_fcm():
    mech, _ = _mechanism(seed=5)
    cloned = mech.clone()
    assert isinstance(cloned, InvertibleClassifierFCM)
