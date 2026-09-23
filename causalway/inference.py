"""Conditional / interventional / counterfactual prediction engines (Plan 2 §4-5).

``dowhy.gcm`` has no ``P(Y | X = x)`` primitive, so :func:`condition` is a
tiered backend ladder, selected automatically and always reported on the
returned :class:`~causalway.queries.Answer`:

0. **mechanism** — evidence covers ``target``'s parents (and no evidence key
   is a descendant of ``target``, so it is screened off by the parents alone,
   Pearl's local Markov property): evaluate the fitted mechanism directly.
1. **pgmpy-exact** — every node is categorical/ordinal: an exact
   ``DiscreteBayesianNetwork`` + ``CausalInference.query``.
2. **likelihood-weighting** — ancestral sampling with evidence nodes clamped
   and the sample weight multiplied by ``p(value | parents)`` at each one
   (categorical: ``estimate_probabilities``; continuous: a Gaussian KDE over
   the fitted ANM's noise samples — never rejects).
3. **rejection** — a safety-net fallback for a mechanism with no usable
   density (unreachable for the mechanism families :mod:`causalway.model`
   assigns, kept for custom/future mechanisms).

Tier 1b (exact linear-Gaussian via ``pgmpy.LinearGaussianBayesianNetwork``,
Plan 2 §4) is **not implemented** — an all-continuous-linear-Gaussian KG falls
through to tier 2 (likelihood weighting) instead of the exact closed form.
This is a scoped-out gap, not a silent approximation: the tier is reported
truthfully as ``"likelihood-weighting"`` and ``ess`` still measures how much
that costs.

``intervene`` and ``counterfactual`` wrap ``gcm.interventional_samples`` /
``gcm.counterfactual_samples`` directly (§5.3).

**A query has 1..n targets.** ``Query.target`` was always a list (§6.3), and
all three engines here accept either one node name or a collection of them.
A bare ``str`` returns one :class:`~causalway.queries.Answer` (unchanged); a
list/tuple/set returns ``{node: Answer}``, one per target, even for a
one-element collection — the *shape you asked in* decides the shape you get
back, so a caller never has to guess.

Multi-target is not a loop over the single-target path, and that is the
point: every target is read off **one** shared computation — one pgmpy
network, one weighted sample set, one interventional draw, one abduction.
Answering them separately would let the cards disagree with each other
(the same argument as Plan 3 W23), and would pay for the expensive part
once per target.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm._noise import compute_noise_from_data
from dowhy.graph import is_root_node

from .entities import aggregate, population_rows
from .queries import Answer

__all__ = ["condition", "intervene", "counterfactual", "CF_BATCH_ROWS"]

#: Ceiling on the rows in one abduction batch, i.e. `draws x rows_per_draw`
#: in :func:`counterfactual`. Large enough that scikit-learn's per-call
#: overhead is amortised away, small enough that the packed Gumbel noise —
#: one small object array per row per categorical node — stays in the tens of
#: megabytes even for an entity with hundreds of join rows.
#: `service/api/main.py` keeps its own copy of this number, because importing
#: this module at its top level would undo the Plan 1 / Plan 2 lazy-import
#: split (see CLAUDE.md §2); the two are documented to move together.
CF_BATCH_ROWS = 20_000

# One node name, or several. `str` is deliberately not `Iterable[str]` here —
# a string is iterable, and treating one as a collection of characters is the
# classic way this kind of API goes wrong.
Targets = Union[str, Iterable[str]]


def _normalise_targets(model, target: Targets) -> tuple[list, bool]:
    """``target`` -> ``(targets, single)``.

    ``single`` is True only for a bare ``str``; it is what decides whether the
    caller gets one :class:`Answer` or a ``{node: Answer}`` dict. A *set* is
    sorted, because iteration order over a set is arbitrary and the answer
    dict's insertion order should not be; a list/tuple keeps the caller's
    order, duplicates removed.
    """
    if isinstance(target, str):
        targets = [target]
        single = True
    else:
        try:
            raw = sorted(target) if isinstance(target, (set, frozenset)) else list(target)
        except TypeError as exc:
            raise TypeError(
                f"target must be a node name or a collection of node names, got "
                f"{type(target).__name__}"
            ) from exc
        targets = list(dict.fromkeys(raw))
        single = False
    if not targets:
        raise ValueError("target: at least one target node is required")
    known = set(model.spec.columns)
    unknown = [t for t in targets if t not in known]
    if unknown:
        raise ValueError(
            f"target: {unknown} not in the fitted model's nodes {sorted(known)}"
        )
    return targets, single


# ---------------------------------------------------------------------- #
# condition() — the backend ladder
# ---------------------------------------------------------------------- #
def condition(model, target: Targets, evidence: dict, *, conditions: Optional[dict] = None,
             backend: str = "auto", num_samples: int = 10_000,
             ess_warn: float = 100.0):
    """``P(target | evidence, conditions)`` — (sub-)population conditional prediction.

    ``target`` is one node name (one :class:`Answer` back) or a collection of
    them (``{node: Answer}`` back). With several targets the ladder is walked
    **once**: whichever targets tier 0 can answer from their own parents are
    taken there, and the rest share a single pgmpy network or a single
    weighted sample set, so their answers are mutually consistent and the
    expensive step is paid once rather than per target.

    ``conditions`` is the sub-population selector (CATE, ``cw:hasCondition``)
    and is merged with ``evidence`` for computation — both narrow the same
    conditioning set; the RDF layer keeps them distinct (§6.3) because they
    mean different things (observed evidence vs. a population filter), not
    because they are computed differently.
    """
    targets, single = _normalise_targets(model, target)
    all_evidence = dict(evidence)
    all_evidence.update(conditions or {})
    clash = [t for t in targets if t in all_evidence]
    if clash:
        raise ValueError(
            f"condition: target {clash if len(clash) > 1 else clash[0]!r} cannot also be "
            "in evidence/conditions"
        )
    all_evidence = _encode_ordinal(model, all_evidence)

    answers = _condition_many(model, targets, all_evidence, backend, num_samples)

    for t in targets:
        answer = answers[t]
        if answer.ess is not None and answer.ess < ess_warn:
            answer.low_confidence = True
        answer.predicted = _decode_ordinal_value(model, t, answer.predicted)
        answer.distribution = _decode_ordinal_distribution(model, t, answer.distribution)
    return answers[targets[0]] if single else answers


# Each tier answers *as many of* `targets` as it can and returns
# `{node: Answer}` — an empty dict, or a dict missing some targets, means
# "does not apply to those", which is what makes the ladder below a filter
# rather than a chain of Nones.
_TIERS = {}


def _condition_many(model, targets: list, all_evidence: dict, backend: str,
                    num_samples: int) -> dict:
    if backend != "auto":
        if backend not in _TIERS:
            raise ValueError(
                f"condition: unknown backend {backend!r}; choose from {list(_TIERS)}")
        answers = _TIERS[backend](model, targets, all_evidence, num_samples)
        missing = [t for t in targets if t not in answers]
        if missing:
            raise ValueError(
                f"condition: backend={backend!r} does not apply to "
                f"{missing if len(missing) > 1 else 'this query'}"
            )
        return answers

    answers = _tier0_mechanism_many(model, targets, all_evidence, num_samples)
    pending = [t for t in targets if t not in answers]
    if pending and _all_discrete(model):
        answers.update(_tier1_pgmpy_exact(model, pending, all_evidence, num_samples))
        pending = [t for t in targets if t not in answers]
    if pending:
        answers.update(_tier2_likelihood_weighting(model, pending, all_evidence, num_samples))
        pending = [t for t in targets if t not in answers]
    if pending:
        answers.update(_tier3_rejection(model, pending, all_evidence, num_samples))
    return answers


def _encode_ordinal(model, values: dict) -> dict:
    """Translate an original ordinal label into its fitted integer code, if needed.

    Ordinal columns (``ordinal=`` at fit time) are stored as
    ``pd.factorize`` integer codes in ``model.spec.data``; a caller who
    passes the original label (recorded in ``model.ordinal_maps``) has it
    silently translated here so evidence/query values can be given either way.
    """
    ordinal_maps = getattr(model, "ordinal_maps", {}) or {}
    out = dict(values)
    for node, val in list(out.items()):
        code_map = ordinal_maps.get(node)
        if not code_map or val in code_map:
            continue
        inverse = {label: code for code, label in code_map.items()}
        if val in inverse:
            out[node] = inverse[val]
    return out


def _decode_ordinal_value(model, node: str, value):
    """Inverse of :func:`_encode_ordinal` for a single predicted value, applied on the way out."""
    if value is None:
        return value
    code_map = (getattr(model, "ordinal_maps", {}) or {}).get(node)
    if not code_map:
        return value
    try:
        return code_map.get(int(value), value)
    except (TypeError, ValueError):
        return value


def _decode_ordinal_distribution(model, node: str, distribution: Optional[dict]) -> Optional[dict]:
    """Inverse of :func:`_encode_ordinal` for a ``{code: probability}`` distribution."""
    if not distribution:
        return distribution
    code_map = (getattr(model, "ordinal_maps", {}) or {}).get(node)
    if not code_map:
        return distribution
    out: dict = {}
    for k, v in distribution.items():
        try:
            label = code_map.get(int(k), k)
        except (TypeError, ValueError):
            label = k
        out[str(label)] = out.get(str(label), 0.0) + v
    return out


def _all_discrete(model) -> bool:
    return all(dt in ("categorical", "ordinal") for dt in model.spec.dtypes.values())


def _tier0_mechanism_many(model, targets: list, all_evidence: dict,
                          num_samples: int = 0) -> dict:
    """Tier 0 is genuinely per-target — whether a node is screened off by its own
    parents is a property of that node, so there is nothing to share here."""
    out = {}
    for t in targets:
        answer = _tier0_mechanism(model, t, all_evidence)
        if answer is not None:
            out[t] = answer
    return out


def _tier0_mechanism(model, target: str, all_evidence: dict) -> Optional[Answer]:
    # A root has no parents to plug in: its mechanism is a plain distribution
    # (e.g. EmpiricalDistribution), with no noise to draw and nothing to evaluate.
    # Its answer is the marginal, which the next tiers compute.
    if is_root_node(model.scm.graph, target):
        return None
    parents = set(model.scm.graph.predecessors(target))
    if not parents.issubset(all_evidence.keys()):
        return None
    descendants = nx.descendants(model.scm.graph, target)
    if descendants & set(all_evidence.keys()):
        return None  # not screened off by parents alone; needs the sampling tiers

    pa_order = list(model.scm.graph.predecessors(target))
    pa_values = np.array([[all_evidence[p] for p in pa_order]], dtype=object)
    mech = model.scm.causal_mechanism(target)
    dtype = model.spec.dtypes.get(target)

    if hasattr(mech, "estimate_probabilities"):
        probs = mech.estimate_probabilities(pa_values)[0]
        classes = list(mech.get_class_names(np.arange(len(probs))))
        distribution = {str(c): float(p) for c, p in zip(classes, probs)}
        predicted = classes[int(np.argmax(probs))]
        return Answer(kind="conditional", target=target, predicted=predicted,
                     distribution=distribution, backend="mechanism", n_samples=1, ess=1.0)

    n = 4000
    noise = mech.draw_noise_samples(n)
    pa_rep = np.repeat(pa_values.astype(float), n, axis=0)
    vals = np.asarray(mech.evaluate(pa_rep, noise)).reshape(-1).astype(float)
    predicted = float(np.mean(vals))
    return Answer(kind="conditional", target=target, predicted=predicted, mean=predicted,
                 std=float(np.std(vals)), backend="mechanism", n_samples=n, ess=float(n))


def _tier1_pgmpy_exact(model, targets: list, all_evidence: dict,
                       num_samples: int = 0) -> dict:
    """One network, one CPT estimation, one marginal query per target.

    The network and its BDeu parameters do not depend on the target, and
    estimating them is the expensive half — so they are built once for the
    whole batch. Each target is then queried as its own marginal rather than
    as a joint over all of them: a joint of n discrete variables is
    exponential in n and the caller asked for n marginals, not their table.
    """
    if not _all_discrete(model):
        return {}
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import CausalInference
    from pgmpy.models import DiscreteBayesianNetwork

    columns = model.spec.columns
    edges = [(model.spec.ocg.nodes[i].name, model.spec.ocg.nodes[j].name)
             for i, j in model.spec.ocg.edges()]
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(columns)
    bn.add_edges_from(edges)

    df_str = model.spec.data[columns].astype(str)
    cpds = BayesianEstimator(bn, df_str).get_parameters(
        prior_type="BDeu", equivalent_sample_size=5)
    bn.add_cpds(*cpds)

    ev = {k: str(v) for k, v in all_evidence.items()}
    engine = CausalInference(bn)
    out = {}
    for target in targets:
        factor = engine.query(variables=[target], evidence=ev, show_progress=False)
        state_names = factor.state_names[target]
        values = np.asarray(factor.values).reshape(-1)
        distribution = {s: float(v) for s, v in zip(state_names, values)}
        predicted = state_names[int(np.argmax(values))]
        out[target] = Answer(kind="conditional", target=target, predicted=predicted,
                             distribution=distribution, backend="pgmpy-exact",
                             n_samples=len(df_str), ess=float(len(df_str)))
    return out


def _tier2_likelihood_weighting(model, targets: list, all_evidence: dict,
                                num_samples: int) -> dict:
    """One ancestral pass, one weight vector, every target read off it.

    Sharing is not just an optimisation here: two targets weighted by two
    independently drawn sample sets can report marginals that no single joint
    distribution has, which is the disagreement Plan 3 W23 exists to prevent.
    """
    g = model.scm.graph
    order = list(nx.topological_sort(g))
    samples: dict = {}
    weights = np.ones(num_samples)

    for node in order:
        parents = list(g.predecessors(node))
        pa = (np.column_stack([samples[p] for p in parents])
              if parents else np.zeros((num_samples, 0)))
        if node in all_evidence:
            val = all_evidence[node]
            samples[node] = np.array([val] * num_samples, dtype=object)
            weights *= _density_at(model, node, pa, val, num_samples)
        else:
            mech = model.scm.causal_mechanism(node)
            if is_root_node(g, node):
                samples[node] = np.asarray(mech.draw_samples(num_samples)).reshape(-1)
            else:
                samples[node] = np.asarray(mech.draw_samples(pa)).reshape(-1)

    if weights.sum() <= 0:
        raise ValueError(
            f"condition: likelihood weighting collapsed to zero total weight for "
            f"evidence={all_evidence} — every drawn sample had zero density under it."
        )
    weights = weights / weights.sum()
    ess = float(1.0 / np.sum(weights ** 2))

    out = {}
    for target in targets:
        target_vals = samples[target]
        dtype = model.spec.dtypes.get(target)
        if dtype in ("categorical", "ordinal"):
            s = pd.Series(target_vals)
            distribution = s.groupby(s).apply(lambda idx: float(weights[idx.index].sum()))
            distribution = {str(k): float(v) for k, v in distribution.items()}
            predicted = max(distribution, key=distribution.get)
            out[target] = Answer(kind="conditional", target=target, predicted=predicted,
                                 distribution=distribution, backend="likelihood-weighting",
                                 n_samples=num_samples, ess=ess)
            continue

        vals = target_vals.astype(float)
        mean = float(np.sum(weights * vals))
        std = float(np.sqrt(max(np.sum(weights * (vals - mean) ** 2), 0.0)))
        out[target] = Answer(kind="conditional", target=target, predicted=mean, mean=mean,
                             std=std, backend="likelihood-weighting",
                             n_samples=num_samples, ess=ess)
    return out


def _density_at(model, node: str, pa: np.ndarray, val: Any, n: int) -> np.ndarray:
    """``p(node = val | pa)`` (or the marginal, for a root) as an ``(n,)`` weight vector.

    Routed by dtype, not by ``hasattr(mech, "estimate_probabilities")``: every root
    node is fitted as ``EmpiricalDistribution`` regardless of dtype, which never has
    ``estimate_probabilities``, so that check alone would silently send a categorical
    *root* into the continuous/KDE branch below (and fail outright, since a KDE over
    non-numeric string values has nothing to fit).
    """
    from scipy.stats import gaussian_kde

    dtype = model.spec.dtypes.get(node)
    is_root = is_root_node(model.scm.graph, node)

    if is_root:
        if dtype in ("categorical", "ordinal"):
            counts = model.spec.data[node].astype(str).value_counts(normalize=True)
            p = float(counts.get(str(val), 0.0))
            return np.full(n, p)
        ref = pd.to_numeric(model.spec.data[node], errors="coerce").dropna().to_numpy()
        kde = gaussian_kde(ref)
        return np.full(n, float(kde([float(val)])[0]))

    mech = model.scm.causal_mechanism(node)
    if dtype == "categorical" and hasattr(mech, "estimate_probabilities"):
        probs = mech.estimate_probabilities(pa)
        classes = [str(c) for c in mech.get_class_names(np.arange(probs.shape[1]))]
        if str(val) not in classes:
            return np.zeros(n)
        return probs[:, classes.index(str(val))]

    # ordinal / discrete / continuous, non-root: ANM family (has estimate_noise /
    # draw_noise_samples regardless of dtype) — density via the noise-residual KDE.
    residual = mech.estimate_noise(np.full((n, 1), float(val)), pa).reshape(-1)
    ref = np.asarray(mech.draw_noise_samples(5000)).reshape(-1)
    kde = gaussian_kde(ref)
    return kde(residual)


def _tier3_rejection(model, targets: list, all_evidence: dict, num_samples: int) -> dict:
    """Draw joint samples ignoring evidence, then exact/kernel-weight-match it post hoc.

    Safety-net fallback for a mechanism with no usable density (unreachable
    given the mechanism families :mod:`causalway.model` assigns). One draw and
    one weight vector for every target, same as tier 2.
    """
    samples = gcm.interventional_samples(model.scm, {}, num_samples_to_draw=num_samples)
    weights = np.ones(len(samples))
    for node, val in all_evidence.items():
        dtype = model.spec.dtypes.get(node)
        if dtype in ("categorical", "ordinal"):
            weights *= (samples[node].astype(str) == str(val)).to_numpy(dtype=float)
        else:
            from scipy.stats import gaussian_kde
            ref = pd.to_numeric(samples[node], errors="coerce").dropna().to_numpy()
            kde = gaussian_kde(ref)
            weights *= kde(pd.to_numeric(samples[node], errors="coerce").to_numpy() * 0 + float(val))

    if weights.sum() <= 0:
        raise ValueError(f"condition: rejection sampling found no matches for evidence={all_evidence}")
    weights = weights / weights.sum()
    ess = float(1.0 / np.sum(weights ** 2))

    out = {}
    for target in targets:
        dtype = model.spec.dtypes.get(target)
        if dtype in ("categorical", "ordinal"):
            s = samples[target]
            distribution: dict = {}
            for level in s.unique():
                distribution[str(level)] = float(weights[(s == level).to_numpy()].sum())
            predicted = max(distribution, key=distribution.get)
            out[target] = Answer(kind="conditional", target=target, predicted=predicted,
                                 distribution=distribution, backend="rejection",
                                 n_samples=num_samples, ess=ess)
            continue
        vals = samples[target].astype(float).to_numpy()
        mean = float(np.sum(weights * vals))
        std = float(np.sqrt(max(np.sum(weights * (vals - mean) ** 2), 0.0)))
        out[target] = Answer(kind="conditional", target=target, predicted=mean, mean=mean,
                             std=std, backend="rejection", n_samples=num_samples, ess=ess)
    return out


_TIERS.update({
    "mechanism": _tier0_mechanism_many,
    "pgmpy-exact": _tier1_pgmpy_exact,
    "likelihood-weighting": _tier2_likelihood_weighting,
    "rejection": _tier3_rejection,
})


# ---------------------------------------------------------------------- #
# intervene() — interventional prediction + contrast
# ---------------------------------------------------------------------- #
def _as_fn(value) -> Callable[[np.ndarray], Any]:
    if callable(value):
        return value
    return lambda _: value


def _observed_data_for(model, *, conditions: Optional[dict], entity: Optional[str]):
    if entity is not None:
        rows = population_rows(model.spec.mat, entity)
        if len(rows) == 0:
            raise ValueError(f"intervene: entity {entity} has no rows in the materialised KG")
        return model.spec.data.loc[rows, model.spec.columns]
    if conditions:
        df = model.spec.data
        mask = pd.Series(True, index=df.index)
        for node, value in conditions.items():
            mask &= (df[node].astype(str) == str(value))
        sub = df.loc[mask, model.spec.columns]
        if len(sub) == 0:
            raise ValueError(f"intervene: no rows match conditions={conditions}")
        return sub
    return None


def intervene(model, interventions: dict, *, target: Targets,
             reference: Optional[dict] = None, conditions: Optional[dict] = None,
             entity: Optional[str] = None, num_samples: int = 10_000):
    """``P(target | do(interventions)[, conditions])`` — (sub-)population interventional prediction.

    ``target`` is one node name (one :class:`Answer` back) or a collection of
    them (``{node: Answer}`` back). The mutilated model is sampled **once**
    for the whole batch — a do() draw is a full joint row, so every target is
    already in it, and re-drawing per target would only add noise between
    answers that came from the same intervention.
    """
    targets, single = _normalise_targets(model, target)
    interventions = {node: (value if callable(value) else _encode_ordinal(model, {node: value})[node])
                     for node, value in interventions.items()}
    fns = {node: _as_fn(value) for node, value in interventions.items()}
    observed = _observed_data_for(model, conditions=_encode_ordinal(model, conditions or {}) or None,
                                  entity=entity)
    kwargs = dict(observed_data=observed) if observed is not None else dict(
        num_samples_to_draw=num_samples)
    samples = gcm.interventional_samples(model.scm, fns, **kwargs)

    ref_samples = None
    if reference is not None:
        reference = {node: (value if callable(value) else _encode_ordinal(model, {node: value})[node])
                    for node, value in reference.items()}
        ref_fns = {node: _as_fn(value) for node, value in reference.items()}
        ref_samples = gcm.interventional_samples(model.scm, ref_fns, **kwargs)

    answers = {}
    for t in targets:
        dtype = model.spec.dtypes.get(t)
        predicted_col = samples[t]
        predicted = aggregate(predicted_col, dtype)
        distribution = None
        mean = std = None
        if dtype in ("categorical", "ordinal"):
            distribution = {str(k): float(v) for k, v in
                            predicted_col.value_counts(normalize=True).items()}
        else:
            vals = predicted_col.astype(float)
            mean = float(vals.mean())
            std = float(vals.std()) if len(vals) > 1 else 0.0

        effect = None if ref_samples is None else _contrast(samples, ref_samples, t, dtype)

        predicted = _decode_ordinal_value(model, t, predicted)
        distribution = _decode_ordinal_distribution(model, t, distribution)
        effect = _decode_ordinal_distribution(model, t, effect) if isinstance(effect, dict) else effect
        answers[t] = Answer(kind="interventional", target=t, predicted=predicted,
                            distribution=distribution, mean=mean, std=std, effect=effect,
                            backend="gcm.interventional_samples", n_samples=len(samples))
    return answers[targets[0]] if single else answers


def _contrast(alt: pd.DataFrame, ref: pd.DataFrame, target: str, dtype: str):
    """The alternative-minus-reference contrast, read off two already-drawn sample sets.

    Both frames come from the *same* call site, one draw each, shared across
    every target — so a batch of targets is contrasted against one reference
    world rather than one per target. The continuous branch is the estimator
    ``gcm.average_causal_effect`` computes (a difference of sample means);
    it is spelled out here so it reads the same draws the distribution above
    was reported from, instead of quietly drawing its own.
    """
    if dtype in ("categorical", "ordinal"):
        p_alt = alt[target].value_counts(normalize=True)
        p_ref = ref[target].value_counts(normalize=True)
        levels = sorted(set(p_alt.index) | set(p_ref.index), key=str)
        return {str(lvl): float(p_alt.get(lvl, 0.0) - p_ref.get(lvl, 0.0)) for lvl in levels}
    return float(alt[target].astype(float).mean() - ref[target].astype(float).mean())


# ---------------------------------------------------------------------- #
# counterfactual() — entity-level, abduct once, evaluate per hypothetical
# ---------------------------------------------------------------------- #
def counterfactual(model, interventions: dict, *, entity: str, target: Targets,
                   num_samples: int = 200):
    """Entity-level counterfactual: abduct noise from ``entity``'s row(s), then evaluate.

    ``interventions`` is keyed by ``(entity_iri_or_None, node_name)`` per
    Plan 2 §5.3 — the entity component is bookkeeping for §6.5 reach
    validation and RDF provenance; at the flat-join SCM level every
    intervention is just ``do(node_name := value)`` regardless of which
    entity's property it names (Plan 2 §10 risk #2).

    Which rows *are* this entity's is read from ``model.spec.mat.entity_ids``
    (via :func:`causalway.entities.population_rows`), the flat join's record of
    which entity owns which cell — so a model fitted on a different join than
    the one the entity was chosen from would abduct from the wrong rows. See
    :func:`causalway.model._materialize_kg`.

    ``target`` is one node name (one :class:`Answer` back) or a collection of
    them (``{node: Answer}`` back). All targets are read off **one** sequence
    of abductions: the noise is this entity's, so drawing it again per target
    would answer "what would have happened to this unit" from a different
    unit each time.
    """
    targets, single = _normalise_targets(model, target)
    rows = population_rows(model.spec.mat, entity)
    if len(rows) == 0:
        raise ValueError(f"counterfactual: entity {entity} has no rows in the materialised KG")
    observed = model.spec.data.loc[rows, model.spec.columns]

    interventions = {
        key: (value if callable(value) else _encode_ordinal(model, {key[1]: value})[key[1]])
        for key, value in interventions.items()
    }
    fns = {node: _as_fn(value) for (_iv_entity, node), value in interventions.items()}
    # Only true categorical nodes use the (stochastic) Gumbel-max coupling; ordinal
    # nodes use DiscreteAdditiveNoiseModel, whose abduction is an exact residual —
    # redrawing it would just repeat the same value num_samples times for nothing.
    # With several targets one categorical one is enough to need the draws; the
    # exactly-abducted targets in the same batch simply repeat, which changes their
    # distribution not at all.
    n_draws = num_samples if any(
        model.spec.dtypes.get(t) == "categorical" for t in targets) else 1

    # The draws are taken in **one** abduction pass over `n_draws` tiled copies
    # of this entity's rows, not `n_draws` passes over one copy. Abduction and
    # evaluation are both row-wise and independent across rows — every
    # `estimate_noise`/`evaluate` in `mechanisms.py` is vectorised over the
    # frame it is handed — so "n draws for this unit" and "one draw for n
    # copies of this unit" are the same computation, drawn from the same
    # distribution. What differs is cost: the loop paid scikit-learn's
    # per-call setup (plus an OpenMP fork/join) once per node per draw on a
    # frame of **one row**, which is fixed overhead, so the wall clock was
    # linear in `num_samples` and dominated by everything except the
    # arithmetic. Batching is not a micro-optimisation here; it is the
    # difference between seconds and milliseconds at the default 200 draws.
    #
    # Deliberately *not* a thread or process pool. Each mechanism owns a
    # private `np.random.Generator` seeded per node (`model.py`), and several
    # threads drawing from one of them would be neither thread-safe nor
    # reproducible; a process pool would have to ship the fitted SCM to every
    # worker and re-import dowhy there, which costs more than the query does.
    #
    # The batch is chunked because what has to stay bounded is `n_draws x
    # rows_per_draw`, not `n_draws`: the packed Gumbel noise is one small
    # object array per row per categorical node, and an entity with hundreds
    # of join rows would otherwise allocate hundreds of times what a
    # single-row entity does for the same `num_samples`.
    rows_per_draw = len(observed)
    draws_per_batch = max(1, CF_BATCH_ROWS // rows_per_draw)
    draws = []
    remaining = n_draws
    while remaining > 0:
        k = min(draws_per_batch, remaining)
        batch = observed if k == 1 else pd.concat([observed] * k, ignore_index=True)
        noise_data = compute_noise_from_data(model.scm, batch)
        samples = gcm.counterfactual_samples(model.scm, fns, noise_data=noise_data)
        draws.append(samples[targets])
        remaining -= k

    # `per_row` is one draw over this entity's own rows — the provenance that
    # makes the answer about a unit and not a population — so it is the first
    # tile of the first batch, carrying `observed`'s row labels back with it.
    first_tile = draws[0].iloc[:rows_per_draw].copy()
    first_tile.index = observed.index

    answers = {}
    for t in targets:
        dtype = model.spec.dtypes.get(t)
        is_gumbel = dtype == "categorical"
        discrete_like = dtype in ("categorical", "ordinal")
        all_vals = pd.concat([d[t] for d in draws], ignore_index=True)

        predicted = aggregate(all_vals, dtype)
        distribution = None
        mean = std = None
        if discrete_like:
            distribution = {str(k): float(v) for k, v in
                            all_vals.value_counts(normalize=True).items()}
        else:
            vals = all_vals.astype(float)
            mean = float(vals.mean())
            std = float(vals.std()) if len(vals) > 1 else 0.0

        predicted = _decode_ordinal_value(model, t, predicted)
        distribution = _decode_ordinal_distribution(model, t, distribution)
        answers[t] = Answer(
            kind="counterfactual", target=t, entity=str(entity), predicted=predicted,
            distribution=distribution, mean=mean, std=std,
            backend="gumbel-max-abduction" if is_gumbel else "anm-abduction",
            n_samples=len(all_vals), per_row=first_tile[[t]],
            coupling="gumbel-max" if is_gumbel else None)
    return answers[targets[0]] if single else answers
