"""``InvertibleClassifierFCM`` — a categorical FCM with a Gumbel-max noise coupling.

``dowhy.gcm``'s ``ClassifierFCM`` subclasses ``FunctionalCausalModel``, not
``InvertibleFunctionalCausalModel``, and has no ``estimate_noise``
(``dowhy/gcm/causal_mechanisms.py``): ``gcm.counterfactual_samples`` cannot
abduct noise for a categorical node fitted with it. Since a lung-cancer-style
KG is entirely categorical, every non-root node needs an invertible
mechanism before counterfactual queries are possible at all (Plan 2 §1.2,
E1).

Inverting ``ClassifierFCM``'s own coupling (``argmax(cumsum(p) >= u)``,
``u ~ U[0,1]``) is not the right choice: it depends on ``LabelEncoder``'s
alphabetical class order, which is meaningless for a nominal property such as
``drugClass`` or ``tumorStage`` — permuting the class labels would silently
change every counterfactual answer. The Gumbel-max coupling is order-invariant
by construction, so this module implements that instead.

    Y = argmax_k ( log p_k(PA) + G_k ),   G_k ~ Gumbel(0, 1) i.i.d.

Marginally this reproduces exactly ``ClassifierFCM``'s sampling distribution
(the Gumbel-max trick is just categorical sampling in disguise); what it adds
is a closed-form abduction step, so the *observational* distribution is
unchanged and only the counterfactual behaviour differs from the base class.

**Stated limitation.** A categorical counterfactual is *not identified* by the
observational distribution: infinitely many SCMs induce the same
``p(Y | PA)`` and disagree on counterfactuals. Gumbel-max fixes one
particular coupling (Oberst & Sontag, "Counterfactual Off-Policy Evaluation
with Gumbel-Max Structural Causal Models", ICML 2019, building on the
abduction identity in Maddison, Tarlow & Minka, "A* Sampling", NeurIPS 2014).
Every answer computed with this mechanism should carry
``cw:counterfactualCoupling "gumbel-max"`` so the choice is visible rather
than implied.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dowhy.gcm.causal_mechanisms import ClassifierFCM, InvertibleFunctionalCausalModel
from dowhy.gcm.ml import ClassificationModel
from dowhy.gcm.util.general import shape_into_2d

__all__ = ["InvertibleClassifierFCM"]

_LOG_EPS = 1e-300  # clip probabilities before log(); argmax is unaffected by the clip.


class _NoiseArray(np.ndarray):
    """An ``(n,)`` object array of per-row Gumbel vectors whose ``.squeeze()`` is a no-op.

    ``dowhy.gcm`` calls ``.squeeze()`` unconditionally on a mechanism's noise
    output before storing it in a per-node ``pd.DataFrame`` column
    (``_noise.compute_noise_from_data``). For a genuine ``(n,)`` array that
    collapses an ``n == 1`` batch to a 0-d array — which pandas then refuses
    to assign to a column at all (``AssertionError: Shape of new values must
    be compatible with manager shape``; a real pandas limitation specific to
    0-d *object* arrays, verified interactively, not a design necessity).
    Overriding ``squeeze`` to a no-op keeps the row axis intact regardless of
    ``n``; it is a true no-op for ``n > 1`` too, since there is no size-1 axis
    to remove there in the first place.
    """

    def squeeze(self, axis=None):
        return self


def _pack_noise(rows: np.ndarray) -> np.ndarray:
    """``(n, d)`` float array -> ``(n,)`` object array, one Gumbel row-vector per element.

    See :meth:`InvertibleClassifierFCM.draw_noise_samples` for why this
    packing exists: gcm's per-node noise ``DataFrame`` column has room for
    exactly one value per row.
    """
    n = rows.shape[0]
    packed = np.empty(n, dtype=object)
    for i in range(n):
        packed[i] = rows[i]
    return packed.view(_NoiseArray)


def _unpack_noise(noise_samples, n_classes: int) -> np.ndarray:
    """Inverse of :func:`_pack_noise`; also accepts an already-``(n, d)`` array unchanged."""
    arr = np.asarray(noise_samples)
    if arr.dtype == object:
        rows = [np.asarray(v, dtype=float).reshape(-1) for v in arr.reshape(-1)]
        return np.stack(rows, axis=0)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        # A single row whose leading (n=1) axis was squeezed away upstream
        # (dowhy.gcm does this routinely, e.g. compute_noise_from_data).
        return arr.reshape(1, n_classes)
    return arr


class InvertibleClassifierFCM(ClassifierFCM, InvertibleFunctionalCausalModel):
    """Categorical FCM ``Y = f(PA, N)`` with an invertible, order-invariant noise coupling.

    Reuses ``ClassifierFCM``'s ``fit`` / ``estimate_probabilities`` /
    ``get_class_names`` / ``classifier_model`` unchanged — only the noise
    representation (``draw_noise_samples`` / ``evaluate``) and the new
    ``estimate_noise`` (abduction) differ.

    **Reproducibility.**  Both Gumbel draws come from this mechanism's *own*
    :class:`numpy.random.Generator`, not from the global ``np.random`` state.
    That is what makes a counterfactual answer reproducible rather than merely
    documented: a stochastic result published as an RDF assertion that nobody
    can recompute is the worst of both worlds.  Each node gets its own
    generator, spawned from the model's ``random_state`` by node name in
    :meth:`causalway.model.CausalModel.fit`, so the draws do not depend on the
    order gcm happens to visit nodes in — sharing one stream across nodes would
    make every answer a function of that traversal order.

    ``random_state=None`` keeps the previous behaviour (a fresh, unseeded
    generator), so an unseeded fit is exactly as reproducible as it was: not.
    """

    def __init__(self, classifier_model: Optional[ClassificationModel] = None,
                 random_state: Optional[int] = None):
        super().__init__(classifier_model=classifier_model)
        self._random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @property
    def random_state(self) -> Optional[int]:
        """The seed this mechanism's noise is drawn from, if it has one."""
        return getattr(self, "_random_state", None)

    @property
    def _generator(self) -> np.random.Generator:
        """This mechanism's generator, rebuilt on demand.

        A model pickled before mechanisms carried their own generator unpickles
        without one; rebuilding it here means such a model still runs (unseeded,
        as it always was) instead of raising on the first counterfactual.
        """
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = np.random.default_rng(self.random_state)
            self._rng = rng
        return rng

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        """``(num_samples,)`` object array; each element is a length-``n_classes`` Gumbel(0, 1) vector.

        Packed one-vector-per-row rather than returned as a plain
        ``(num_samples, n_classes)`` array because ``dowhy.gcm`` stores a
        node's noise as *one column of a per-node ``pd.DataFrame``*
        (``compute_noise_from_data``, ``noise_samples_of_ancestors``) — i.e.
        it assumes a single scalar per row. A Gumbel-max noise term is
        instead a length-``n_classes`` vector per row, so it has to travel as
        one Python object per row to survive that column; :meth:`evaluate`
        unpacks it (and also accepts a plain 2D array, for direct calls that
        never go through gcm's DataFrame machinery).
        """
        n_classes = len(self._classifier_model.classes)
        raw = self._generator.gumbel(loc=0.0, scale=1.0, size=(num_samples, n_classes))
        return _pack_noise(raw)

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        """``Y = argmax_k (log p_k(PA) + G_k)``."""
        log_p = self._log_probabilities(parent_samples)
        noise = _unpack_noise(noise_samples, log_p.shape[1])
        indices = np.argmax(log_p + noise, axis=1)
        return shape_into_2d(np.array(self.get_class_names(indices)))

    def estimate_noise(self, target_samples: np.ndarray, parent_samples: np.ndarray) -> np.ndarray:
        """Closed-form abduction: posterior-sample the Gumbel vector given the observed class.

        Standard "top-down" Gumbel sampling (Maddison et al. 2014 §B; the same
        construction used by Oberst & Sontag 2019 for counterfactual SCMs).
        For each row with observed class ``k``:

        * the winning coordinate is exact: sample the row's max value
          ``T ~ Gumbel(0, 1)`` (location 0, since ``log Σ_j exp(log p_j) = 0``
          for a normalised distribution) and set ``G_k = T - log p_k``;
        * every other coordinate ``j != k`` is a Gumbel(``log p_j``) draw
          truncated to be ``<= T``, sampled via the numerically stable
          identity ``truncated = -logaddexp(-g'_j, -T)`` applied to an
          unconstrained draw ``g'_j ~ Gumbel(log p_j)``.

        This guarantees ``evaluate(pa, estimate_noise(y, pa)) == y`` exactly
        for every row (the coordinate for ``k`` equals ``T``; every other
        coordinate is strictly below ``T`` whenever its unconstrained draw is
        finite), which is the consistency property counterfactual abduction
        depends on.
        """
        target_samples, parent_samples = shape_into_2d(target_samples, parent_samples)
        log_p = self._log_probabilities(parent_samples)
        n, d = log_p.shape

        class_names = [str(c) for c in self._classifier_model.classes]
        index_of = {name: i for i, name in enumerate(class_names)}
        try:
            k = np.array([index_of[str(y)] for y in target_samples.reshape(-1)])
        except KeyError as e:
            raise ValueError(
                f"estimate_noise: observed class {e} is not among the classes "
                f"this mechanism was fitted on: {class_names}"
            ) from e

        top_gumbel = self._generator.gumbel(size=n)              # T ~ Gumbel(0, 1)
        g_prime = self._generator.gumbel(size=(n, d)) + log_p    # unconstrained Gumbel(log p_j)
        truncated = -np.logaddexp(-g_prime, -top_gumbel[:, None])

        noise = truncated - log_p
        rows = np.arange(n)
        noise[rows, k] = top_gumbel - log_p[rows, k]
        return _pack_noise(noise)

    def _log_probabilities(self, parent_samples: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self.estimate_probabilities(parent_samples), _LOG_EPS, None))

    def clone(self) -> "InvertibleClassifierFCM":
        return InvertibleClassifierFCM(classifier_model=self._classifier_model.clone(),
                                      random_state=self.random_state)

    def __repr__(self) -> str:
        return "Invertible (Gumbel-max) classifier FCM based on %s" % self.classifier_model

    @classmethod
    def from_classifier_fcm(cls, mechanism: ClassifierFCM,
                            random_state: Optional[int] = None) -> "InvertibleClassifierFCM":
        """Wrap an already-selected (but unfitted) ``ClassifierFCM``'s classifier.

        Used by ``CausalModel.fit`` (Plan 2 §3.3 step 7) to override every
        non-root ``ClassifierFCM`` that ``gcm.auto.assign_causal_mechanisms``
        picked, while keeping the same (cloned, unfitted) classifier class it
        selected via log-loss.
        """
        classifier: Optional[ClassificationModel] = mechanism.classifier_model
        return cls(classifier_model=None if classifier is None else classifier.clone(),
                   random_state=random_state)
