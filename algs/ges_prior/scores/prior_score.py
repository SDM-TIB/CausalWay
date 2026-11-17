"""
PriorScore: a decomposable structure score with an additive edge prior

    S(G) = S_base(G) + lambda * sum_{i->j in G} W[i, j]

and the edge weights W of the four GES-Prior scores, built from the LLM priors

    M[i, j] = P(i and j are directly causally related)      (symmetric, M = A * C)
    D[i, j] = P(i -> j | i and j are causally related)      (D[i, j] + D[j, i] = 1)
    B[i, j] = M[i, j] * D[i, j]                             (P(i -> j))

    "BIC"      W = 0
    "BIC+M"    W = log(M / (1 - M))
    "BIC+B"    W = log(B / (1 - B))
    "BIC+M+B"  W = log(B / (1 - M)) = log(M / (1 - M)) + log(D)

Pairs without a prior (the LLM was not asked, or its answer did not parse) get
W = 0, i.e. their score is the base score.

Why this replaces ``StructureScoreWithPrior`` in the search
-----------------------------------------------------------
The earlier Phase II scored an edge by ``log(B / (1 - B))``. That is the log
odds of *this direction being right*, which confuses two questions. Take a pair
the LLM is sure is causally related (M = 0.9) but cannot orient (D = 0.5): then
B = 0.45, the log odds are negative, and the search is told to *remove* an edge
the prior believes in — in both directions. ``BIC+M+B`` keeps the two questions
apart: ``log(M / (1 - M))`` rewards or penalises the adjacency, ``log(D)`` only
ever costs, and costs less for the likelier direction.

The second difference is the treatment of missing knowledge. The old matrices
stored 0.5 for "no prior", which is neutral for M but was copied into B as
``B = M`` when only the direction was unknown — i.e. "direction certain", both
ways. Here missing knowledge is an explicit mask (``has``) and contributes
exactly zero, so an uninformed pair is scored by the data alone and GES-Prior
degrades to GES when the LLM knows nothing.

``StructureScoreWithPrior`` is kept for callers that hold a ready-made B matrix
(the constraint-as-soft-prior baseline); the GES-Prior search no longer uses it.

Written against pgmpy >= 1.0 (``BaseStructureScore`` caches ``_local_score``;
``get_scoring_method`` returns the score object).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from pgmpy.structure_score import BaseStructureScore, get_scoring_method

__all__ = ["SCORES", "prior_weights", "PriorScore"]

SCORES = ("BIC", "BIC+M", "BIC+B", "BIC+M+B")


def prior_weights(score: str, M: np.ndarray, D: np.ndarray, has: np.ndarray,
                  eps: float = 1e-3) -> np.ndarray:
    """
    Edge weights W of `score` (one of SCORES).

    Parameters
    ----------
    M, D : np.ndarray
        (d, d) prior matrices as described in the module docstring; NaN where unknown
    has : np.ndarray
        (d, d) bool, True where the pair has a prior
    eps : float
        probabilities are clipped to [eps, 1 - eps], so each edge contributes at
        most log((1 - eps) / eps) (6.9 for eps = 1e-3) per unit of lambda. The
        bound is what lets the data overrule a confident but wrong prior: a BIC
        difference grows with the sample size, the prior term does not.
    """
    if score not in SCORES:
        raise ValueError(f"Unknown score '{score}'. Choose from {SCORES}")
    M = np.asarray(M, dtype=float)
    D = np.asarray(D, dtype=float)
    d = M.shape[0]
    if score == "BIC":
        return np.zeros((d, d))

    Mc = np.clip(np.nan_to_num(M, nan=0.5), eps, 1 - eps)
    Dc = np.clip(np.nan_to_num(D, nan=0.5), eps, 1 - eps)
    if score == "BIC+M":
        W = np.log(Mc / (1 - Mc))
    elif score == "BIC+B":
        Bc = np.clip(Mc * Dc, eps, 1 - eps)
        W = np.log(Bc / (1 - Bc))
    else:  # "BIC+M+B"
        W = np.log(Mc / (1 - Mc)) + np.log(Dc)

    W = np.where(np.asarray(has, dtype=bool), W, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


class PriorScore(BaseStructureScore):
    """
    Decomposable score  S_base(G) + lambda * sum_{i->j in G} W[i, j].

    Parameters
    ----------
    data : pandas.DataFrame
    base_score : str or BaseStructureScore
        base score, e.g. 'bic-d' or 'bic-g' (see pgmpy's `get_scoring_method`)
    W : np.ndarray, optional
        (d, d) edge weights, rows / columns in the order of `data.columns`;
        W[i, j] is added when i is a parent of j. None means no prior.
    lam : float
        weight of the prior term
    """

    def __init__(
        self,
        data: pd.DataFrame,
        base_score: Optional[Union[str, BaseStructureScore]] = None,
        W: Optional[np.ndarray] = None,
        lam: float = 1.0,
        state_names=None,
    ):
        super().__init__(data, state_names=state_names)
        self.base_estimator = get_scoring_method(base_score, data)
        self.var_to_idx = {var: idx for idx, var in enumerate(data.columns)}

        d = data.shape[1]
        W = np.zeros((d, d)) if W is None else np.array(W, dtype=float)
        if W.shape != (d, d):
            raise ValueError(f"W must be of shape ({d}, {d}), got {W.shape}")
        if not np.all(np.isfinite(W)):
            raise ValueError("W must be finite")
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.lam = lam

    def _local_score(self, variable, parents) -> float:
        parents = tuple(parents)
        score = self.base_estimator.local_score(variable, parents)
        if self.lam and parents:
            j = self.var_to_idx[variable]
            score += self.lam * sum(self.W[self.var_to_idx[p], j] for p in parents)
        return score
