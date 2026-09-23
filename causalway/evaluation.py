"""Model + graph evaluation and falsification (Plan 2 §7). Run before trusting any answer:

1. :func:`evaluate_model` — wraps ``gcm.evaluate_causal_model``: per-mechanism
   performance, the invertibility assumption check, graph falsification.
2. :func:`falsify` — wraps ``gcm.falsify.falsify_graph``: does the graph
   survive its own conditional-independence implications?
3. :func:`held_out_cv` — k-fold CV of conditional prediction from each node's
   parents, against a marginal baseline; a node that doesn't beat the
   baseline makes interventional answers about it noise.
4. :func:`counterfactual_agreement` — agreement rate against a known
   ground-truth SEM (synthetic KGs only, notebook 04).
5. :func:`row_weights` / :func:`resample_by_weight` — the flat-join i.i.d.
   caveat (Plan 2 §10 risk #4): ``gcm.fit`` has no sample-weight hook, so
   this is a resampling *approximation* of ``row_weight = 1/multiplicity``,
   not a principled weighted fit — stated, not hidden.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm import falsify as _falsify
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold

__all__ = [
    "evaluate_model", "falsify", "held_out_cv", "counterfactual_agreement",
    "row_weights", "resample_by_weight",
]


def evaluate_model(model, *, evaluate_causal_mechanisms: bool = True,
                   evaluate_invertibility_assumptions: bool = True,
                   evaluate_causal_structure: bool = True, max_num_samples: int = -1):
    """``gcm.evaluate_causal_model`` over the fitted model's own training data.

    The invertibility check is reimplemented rather than delegated to
    ``gcm``'s own ``evaluate_invertibility_assumptions=True`` path — see
    :func:`_invertibility_p_values` for why gcm's version crashes on
    ``InvertibleClassifierFCM`` nodes. ``result.pnl_assumptions`` keeps gcm's
    own shape (``{node: (corrected_p_value, rejected, significance_level)}``,
    ``rejected`` after Bonferroni FDR correction), so ``str(result)`` and
    other gcm-side consumers keep working unchanged.
    """
    result = gcm.evaluate_causal_model(
        model.scm, model.spec.data, max_num_samples=max_num_samples,
        evaluate_causal_mechanisms=evaluate_causal_mechanisms,
        evaluate_invertibility_assumptions=False,
        evaluate_causal_structure=evaluate_causal_structure,
    )
    if evaluate_invertibility_assumptions:
        result.pnl_assumptions = _invertibility_p_values(model)
    return result


def _invertibility_p_values(model, max_num_samples: int = 2000,
                            significance_level: float = 0.05,
                            fdr_control_method: str = "bonferroni") -> dict:
    """Independence test between each non-root node's recovered noise and its parents.

    gcm's own ``evaluate_causal_model(evaluate_invertibility_assumptions=True)``
    calls ``mechanism.estimate_noise(...)`` directly and feeds the raw return
    value to a kernel independence test. ``InvertibleClassifierFCM``'s
    Gumbel-max noise is a length-``n_classes`` vector per row, packed into an
    object array so it survives ``dowhy.gcm``'s per-node noise ``DataFrame``
    column (``_noise.compute_noise_from_data``, needed for
    ``counterfactual_samples`` — see ``causalway.mechanisms``); that packed
    form crashes gcm's kernel test (``np.unique`` on an array of arrays). This
    unpacks it first, then reuses the same test and the same Bonferroni FDR
    correction gcm's own version applies.
    """
    from dowhy.gcm.independence_test import kernel_based
    from dowhy.graph import get_ordered_predecessors, is_root_node
    from statsmodels.stats.multitest import multipletests

    from .mechanisms import InvertibleClassifierFCM, _unpack_noise

    data = model.spec.data
    n = min(max_num_samples, len(data))
    idx = (np.random.choice(len(data), n, replace=False) if n < len(data)
          else np.arange(len(data)))
    p_values, nodes = [], []
    for node in model.spec.columns:
        if is_root_node(model.scm.graph, node):
            continue
        parents = get_ordered_predecessors(model.scm.graph, node)
        if not parents:
            continue
        mech = model.scm.causal_mechanism(node)
        pa = data[parents].to_numpy()[idx]
        y = data[node].to_numpy()[idx]
        raw_noise = mech.estimate_noise(y, pa)
        if isinstance(mech, InvertibleClassifierFCM):
            noise = _unpack_noise(raw_noise, len(mech.classifier_model.classes))
        else:
            noise = np.asarray(raw_noise)
        p_values.append(float(kernel_based(noise, pa)))
        nodes.append(node)

    if not nodes:
        return {}
    rejected, corrected, _, _ = multipletests(p_values, significance_level, method=fdr_control_method)
    return {node: (float(p), bool(r), significance_level)
           for node, p, r in zip(nodes, corrected, rejected)}


def falsify(model, **kwargs):
    """``gcm.falsify.falsify_graph`` — whether the graph survives its own CI implications."""
    return _falsify.falsify_graph(model.scm.graph, model.spec.data, **kwargs)


def held_out_cv(model, *, n_splits: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
    """k-fold CV: predict each non-root node from its parents, vs. a marginal baseline.

    Accuracy + macro-F1 for categorical/ordinal targets, R2 for
    discrete/continuous ones. Uses the fitted mechanism directly (no
    refitting per fold) — this measures the *mechanism's* generalisation,
    not a from-scratch retraining protocol.
    """
    data = model.spec.data
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for node in model.spec.columns:
        dtype = model.spec.dtypes.get(node)
        parents = list(model.scm.graph.predecessors(node))
        if not parents:
            continue
        preds, baseline_preds, truths = [], [], []
        for train_idx, test_idx in kf.split(data):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            baseline = (train[node].mode().iloc[0] if dtype in ("categorical", "ordinal")
                       else float(train[node].astype(float).mean()))
            mech = model.scm.causal_mechanism(node)
            pa = test[parents].to_numpy()
            if hasattr(mech, "estimate_probabilities"):
                probs = mech.estimate_probabilities(pa)
                classes = list(mech.get_class_names(np.arange(probs.shape[1])))
                pred = [classes[i] for i in np.argmax(probs, axis=1)]
            else:
                noise = mech.draw_noise_samples(len(test))
                pred = np.asarray(mech.evaluate(pa, noise)).reshape(-1)
            preds.extend(pred)
            baseline_preds.extend([baseline] * len(test))
            truths.extend(test[node].tolist())

        if dtype in ("categorical", "ordinal"):
            acc = float(np.mean([p == t for p, t in zip(preds, truths)]))
            base_acc = float(np.mean([b == t for b, t in zip(baseline_preds, truths)]))
            rows.append({
                "node": node, "metric": "accuracy", "model": acc, "baseline": base_acc,
                "beats_baseline": acc > base_acc,
                "macro_f1": float(f1_score(truths, preds, average="macro", zero_division=0)),
            })
        else:
            r2 = float(r2_score(truths, preds))
            base_r2 = float(r2_score(truths, baseline_preds))
            rows.append({"node": node, "metric": "r2", "model": r2, "baseline": base_r2,
                        "beats_baseline": r2 > base_r2, "macro_f1": None})
    return pd.DataFrame(rows)


def counterfactual_agreement(model, entities: list, interventions: dict, target: str,
                             ground_truth: Callable[[str], object],
                             num_samples: int = 200) -> pd.DataFrame:
    """Agreement rate between the fitted model's counterfactual and a known ground-truth SEM.

    ``ground_truth(entity) -> value`` computes the *true* counterfactual for
    one entity under ``interventions`` — only possible when the generating
    SEM is known (a synthetic KG; notebook 04). Also useful to compare the
    Gumbel-max coupling against the ordinal one for the same entities, by
    calling this twice with models fitted under each.
    """
    rows = []
    for entity in entities:
        answer = model.counterfactual(interventions, entity=entity, target=target,
                                     num_samples=num_samples)
        truth = ground_truth(entity)
        rows.append({
            "entity": str(entity), "predicted": answer.predicted, "truth": truth,
            "match": str(answer.predicted) == str(truth),
        })
    return pd.DataFrame(rows)


def row_weights(mat, class_var: str) -> pd.Series:
    """``1 / multiplicity`` per row, down-weighting rows duplicated by a 1:N join on ``class_var``."""
    counts = mat.entity_ids[class_var].value_counts()
    return mat.entity_ids[class_var].map(lambda e: 1.0 / counts[e])


def resample_by_weight(data: pd.DataFrame, weights: pd.Series,
                       random_state: Optional[int] = None) -> pd.DataFrame:
    """Resample ``data`` with probability proportional to ``weights`` (see module docstring, point 5)."""
    w = weights.reindex(data.index).fillna(1.0)
    probs = (w / w.sum()).to_numpy()
    idx = np.random.RandomState(random_state).choice(data.index, size=len(data), replace=True, p=probs)
    return data.loc[idx].reset_index(drop=True)
