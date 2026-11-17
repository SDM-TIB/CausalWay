"""GES / GES-Prior structure learning with optional hard edge constraints.

This module is the controlled implementation of GES (and its LLM-prior
extensions) that the rest of the project re-routes constrained GES to.  The
``constraint`` argument is duck-typed: it only needs to expose ``allowed`` (an
``(n, n)`` boolean matrix) and, optionally, ``to_expert_knowledge()``.

The priors contract
-------------------
``estimate_priors`` returns, and ``discover_structure`` accepts, a dict with

    M    (n, n) symmetric,  P(i and j are directly causally related)
    B    (n, n)             P(i -> j) = M * D
    D    (n, n)             P(i -> j | related), D[i, j] + D[j, i] = 1
    has  (n, n) bool        True where the LLM actually answered

``M`` and ``B`` are the published contract (the web service stores exactly
those two, with 0.5 meaning "no answer"); ``D`` and ``has`` were added in the
WWW'27 resource revision and are *derived* when a caller hands over the older
two-matrix form (``_normalise_priors``), so every priors file written before
the revision still loads and means the same thing.

Which score each algorithm uses (``scores/prior_score.py`` has the formulas):

    algorithm        Phase I            Phase II
    GES              GES, BIC           CPDAG (oriented at the publication boundary)
    GES-M            GES, BIC+M         CPDAG
    GES-B            GES, BIC+B         CPDAG
    GES-M-B          GES, BIC+M         orient by B inside the equivalence class
    GES-M-B-HL       GES, BIC+M         hill climbing, BIC+M+B, from an extension
    GES-M-B-Greedy   GES, BIC+M         per-edge orientation by B
    GES-HL           -                  hill climbing, BIC
    GES-M-HL         -                  hill climbing, BIC+M
    GES-B-HL         -                  hill climbing, BIC+B

"GES-Prior" in the runners and the UI is GES-M-B-HL. Hill climbing breaks score
ties in set-iteration order, so set ``PYTHONHASHSEED`` for bit-identical reruns.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

from .llm_prior import causal_discovery_prompt, extract_probabilities
from .entity_check import entity_level_prune
from .scores.prior_score import PriorScore, prior_weights
from .ges_prior import run_ges_prior, run_ges_hl
from .llm_call import LLMClient


def _call_llm(client, prompt: str) -> str:
    if hasattr(client, "query_response"):
        return client.query_response(prompt=prompt)
    elif hasattr(client, "text_generate"):
        return client.text_generate(prompt=prompt)
    elif hasattr(client, "ask"):
        return client.ask(question=prompt)
    else:
        raise ValueError(
            f"Unsupported LLM client type: {type(client)}. "
            "Must have query_response(), text_generate(), or ask() method."
        )


def _estimate_priors(
    columns: list,
    var_meta: Optional[dict],
    pair_meta: Optional[dict],
    domain: str,
    llm_client,
    max_workers: int = 16,
    allowed: Optional[np.ndarray] = None,
) -> Optional[dict]:
    if var_meta is None and pair_meta is None:
        return None

    n = len(columns)
    A = np.full((n, n), np.nan)
    C = np.full((n, n), np.nan)
    D = np.full((n, n), np.nan)

    pairs = []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            # Skip LLM queries for topologically forbidden pairs; they stay NaN
            # and the masking below writes the neutral 0.5 prior.
            if allowed is not None and not (allowed[i_idx, j_idx] or allowed[j_idx, i_idx]):
                continue
            pairs.append((i_idx, j_idx))

    def _query_pair(i_idx, j_idx):
        var_a = columns[i_idx]
        var_b = columns[j_idx]

        context_a = var_meta.get(var_a, var_a) if var_meta else var_a
        context_b = var_meta.get(var_b, var_b) if var_meta else var_b

        key_forward = (var_a, var_b)
        key_reverse = (var_b, var_a)
        causal_knowledge = ""
        if pair_meta:
            if key_forward in pair_meta:
                causal_knowledge = pair_meta[key_forward]
            elif key_reverse in pair_meta:
                causal_knowledge = pair_meta[key_reverse]

        prompt = causal_discovery_prompt(
            domain=domain,
            var_a=var_a,
            var_b=var_b,
            context_a=context_a,
            context_b=context_b,
            causal_knowledge=causal_knowledge,
            repeat=True,
        )

        answer = _call_llm(llm_client, prompt)
        probs = extract_probabilities(answer)
        return (i_idx, j_idx, probs)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i_idx, j_idx in pairs:
            fut = executor.submit(_query_pair, i_idx, j_idx)
            futures[fut] = (i_idx, j_idx)

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="LLM queries", unit="pair"
        ):
            i_idx, j_idx = futures[fut]
            try:
                _, _, probs = fut.result()
            except Exception as e:
                var_a, var_b = columns[i_idx], columns[j_idx]
                print(f"\nWarning: LLM query failed for ({var_a}, {var_b}): {e}")
                continue

            if probs is None:
                continue

            try:
                if "P(A - B)" in probs:
                    val = float(probs["P(A - B)"])
                    A[i_idx, j_idx] = val
                    A[j_idx, i_idx] = val

                if "P(A <-> B | A - B)" in probs:
                    val = float(probs["P(A <-> B | A - B)"])
                    C[i_idx, j_idx] = val
                    C[j_idx, i_idx] = val

                if "P(A -> B | A <-> B)" in probs:
                    val = float(probs["P(A -> B | A <-> B)"])
                    D[i_idx, j_idx] = val
                    D[j_idx, i_idx] = 1.0 - val
            except (ValueError, TypeError) as e:
                var_a, var_b = columns[i_idx], columns[j_idx]
                print(f"\nWarning: invalid probability for ({var_a}, {var_b}): {e}")

    if np.all(np.isnan(A)):
        return None

    # A pair has a prior when the LLM gave both the association and the causal
    # probability. A missing direction alone is not "no prior": the adjacency
    # is still informed, and D = 0.5 says exactly "cannot orient".
    M = A * C
    has = np.isfinite(M) & np.isfinite(M.T)
    np.fill_diagonal(has, False)
    D = np.where(np.isfinite(D), D, 0.5)

    M = np.where(has, (M + M.T) / 2, 0.5)
    B = np.where(has, M * D, 0.5)
    D = np.where(has, D, 0.5)

    return {"M": M, "B": B, "D": D, "has": has}


def _normalise_priors(priors: dict, n: int) -> dict:
    """Return ``{"M", "B", "D", "has"}`` from either priors form (see module docstring).

    The two-matrix form marks an unanswered pair by ``M = B = 0.5`` in both
    directions. An answered pair can never look like that: its two B entries
    sum to M, so they cannot both equal M = 0.5. The direction is recovered as
    ``B[i, j] / (B[i, j] + B[j, i])``, which also reads the old "direction
    unknown" encoding (``B = M`` both ways) as the 0.5 it meant.
    """
    M = np.asarray(priors["M"], dtype=float)
    B = np.asarray(priors["B"], dtype=float)
    if M.shape != (n, n) or B.shape != (n, n):
        raise ValueError(f"priors must be ({n}, {n}) matrices, got M {M.shape} and B {B.shape}")
    M = np.nan_to_num(M, nan=0.5)
    B = np.nan_to_num(B, nan=0.5)

    if priors.get("has") is not None:
        has = np.asarray(priors["has"], dtype=bool)
    else:
        neutral = np.isclose(M, 0.5) & np.isclose(B, 0.5) & np.isclose(B.T, 0.5)
        has = ~neutral
    has = has & has.T
    np.fill_diagonal(has, False)

    if priors.get("D") is not None:
        D = np.nan_to_num(np.asarray(priors["D"], dtype=float), nan=0.5)
    else:
        total = B + B.T
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(total > 0, B / total, 0.5)
    D = np.where(has, D, 0.5)
    return {"M": M, "B": np.where(has, M * D, 0.5), "D": D, "has": has}


def _adj_to_edges(adj: np.ndarray, columns: list) -> list:
    edges = []
    n = len(columns)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1 and adj[j, i] == 0:
                edges.append((columns[i], columns[j]))
            elif adj[j, i] == 1 and adj[i, j] == 0:
                edges.append((columns[j], columns[i]))
            elif adj[i, j] == 1 and adj[j, i] == 1:
                edges.append((columns[i], columns[j]))
    return edges


# algorithm -> (Phase I score, orient strategy, Phase II score)
_ALGORITHMS = {
    "GES":            ("BIC",   0, "BIC"),
    "GES-M":          ("BIC+M", 0, "BIC+M"),
    "GES-B":          ("BIC+B", 0, "BIC+B"),
    "GES-M-B":        ("BIC+M", 1, "BIC+M+B"),
    "GES-M-B-HL":     ("BIC+M", 2, "BIC+M+B"),
    "GES-M-B-Greedy": ("BIC+M", 3, "BIC+M+B"),
}

# algorithm -> the score hill climbing runs under, from the empty graph
_ALGORITHMS_HL = {"GES-HL": "BIC", "GES-M-HL": "BIC+M", "GES-B-HL": "BIC+B"}


def estimate_priors(
    df: pd.DataFrame,
    var_meta: Optional[dict] = None,
    pair_meta: Optional[dict] = None,
    domain: str = "general domain",
    llm_client=None,
    max_workers: int = 16,
    cache_path: Optional[str] = None,
    allowed: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Estimate ``M`` and ``B`` prior matrices from LLM queries.

    Forbidden pairs (per ``allowed``) are skipped and left at the neutral 0.5.
    Results are cached to ``cache_path`` if provided.
    """
    columns = df.columns.tolist()

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached priors from {cache_path}")
        return _load_priors(cache_path, columns)

    priors = _estimate_priors(
        columns, var_meta, pair_meta, domain, llm_client, max_workers, allowed=allowed
    )

    if cache_path and priors is not None:
        _save_priors(priors, columns, cache_path)
        print(f"Saved priors to {cache_path}")

    return priors


def _save_priors(priors, columns, path):
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "columns": columns,
        "M": [
            [None if np.isnan(x) else x for x in row] for row in priors["M"].tolist()
        ],
        "B": [
            [None if np.isnan(x) else x for x in row] for row in priors["B"].tolist()
        ],
    }
    if priors.get("D") is not None:
        data["D"] = np.asarray(priors["D"], dtype=float).tolist()
    if priors.get("has") is not None:
        data["has"] = np.asarray(priors["has"], dtype=bool).tolist()
    with open(path, "w") as f:
        json.dump(data, f)


def _load_priors(path, columns):
    import json

    with open(path, "r") as f:
        data = json.load(f)
    out = {"M": np.array(data["M"], dtype=float), "B": np.array(data["B"], dtype=float)}
    if data.get("D") is not None:
        out["D"] = np.array(data["D"], dtype=float)
    if data.get("has") is not None:
        out["has"] = np.array(data["has"], dtype=bool)
    return out


def discover_structure(
    df: pd.DataFrame,
    algorithm: str = "GES",
    var_meta: Optional[dict] = None,
    pair_meta: Optional[dict] = None,
    domain: str = "general domain",
    llm_client=None,
    llm_model: str = "deepseek-v4-flash-vision-exp",
    max_workers: int = 12,
    priors: Optional[dict] = None,
    lambda_prior: float = 1.0,
    base_score: str = "bic-d",
    constraint=None,
    prior_eps: float = 1e-3,
    legacy_ges: bool = False,
    entity_ids: Optional[pd.DataFrame] = None,
    owner_of: Optional[dict] = None,
    hc_start: str = "prior",
) -> np.ndarray:
    """Discover a structure and return its adjacency matrix.

    ``constraint`` is duck-typed (needs ``allowed``, optionally
    ``to_expert_knowledge()``).  When priors are needed and no ``llm_client`` is
    given, an :class:`LLMClient` is built with ``llm_model``.

    ``prior_eps`` clips the prior probabilities, bounding what one edge can gain
    or lose from the prior at ``lambda_prior * log((1 - eps) / eps)``.
    ``legacy_ges`` reproduces the search as it was before the backward phase
    and the CPDAG conversion were fixed; it exists for the correctness
    experiment and nothing else should pass it.

    ``entity_ids`` + ``owner_of`` add the entity-level check after the search
    (:func:`entity_level_prune`): an edge whose family can be read off one row
    per owning entity is kept only if those entities support it, not merely the
    repeated rows of the join (``entity_ids`` is ``Materialization.entity_ids``,
    ``owner_of`` maps each column to the class variable that holds it). Off
    unless both are given.

    ``hc_start`` is where GES-M-B-HL's hill climbing starts: ``"prior"`` (the
    member of the equivalence class the direction prior prefers) or
    ``"extension"`` (an arbitrary consistent extension).
    """
    columns = df.columns.tolist()
    data = df.astype(str)

    allowed = getattr(constraint, "allowed", None) if constraint is not None else None
    if constraint is not None and hasattr(constraint, "to_expert_knowledge"):
        expert_knowledge = constraint.to_expert_knowledge()
    elif allowed is not None:
        # A constraint that only exposes ``allowed`` must still bind hill
        # climbing, or Phase II could flip an edge back the forbidden way.
        from pgmpy.causal_discovery import ExpertKnowledge

        n = len(columns)
        expert_knowledge = ExpertKnowledge(search_space=[
            (columns[i], columns[j])
            for i in range(n) for j in range(n) if i != j and allowed[i, j]
        ])
    else:
        expert_knowledge = None

    if priors is None and (var_meta is not None or pair_meta is not None):
        if llm_client is None:
            llm_client = LLMClient(model=llm_model)
        priors = _estimate_priors(
            columns, var_meta, pair_meta, domain, llm_client, max_workers,
            allowed=allowed,
        )

    pri = _normalise_priors(priors, len(columns)) if priors is not None else None

    def make_score(name: str):
        # Without priors every algorithm runs under the base score: GES-Prior
        # with nothing to add *is* GES, which is what the UI promises.
        if pri is None or name == "BIC" or not lambda_prior:
            return base_score
        W = prior_weights(name, pri["M"], pri["D"], pri["has"], eps=prior_eps)
        return PriorScore(data, base_score=base_score, W=W, lam=lambda_prior)

    if algorithm in _ALGORITHMS_HL:
        adj = run_ges_hl(
            data=data,
            scoring_method=make_score(_ALGORITHMS_HL[algorithm]),
            expert_knowledge=expert_knowledge,
        )
    elif algorithm in _ALGORITHMS:
        phase1, orient_strategy, phase2 = _ALGORITHMS[algorithm]
        adj = run_ges_prior(
            data=data,
            scoring_method_i=make_score(phase1),
            scoring_method_ii=make_score(phase2),
            orient_strategy=orient_strategy,
            allowed=allowed,
            expert_knowledge=expert_knowledge,
            B=pri["B"] if pri is not None else None,
            legacy_ges=legacy_ges,
            hc_start=hc_start,
        )
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: "
            f"{list(_ALGORITHMS) + list(_ALGORITHMS_HL)}"
        )

    if entity_ids is not None and owner_of is not None:
        adj = entity_level_prune(np.asarray(adj, dtype=int), data, owner_of, entity_ids,
                                 base_score=base_score)

    return adj


def learn_bayesian_network(
    df: pd.DataFrame,
    var_meta: Optional[dict] = None,
    pair_meta: Optional[dict] = None,
    algorithm: str = "GES-M-B-HL",
    lambda_prior: float = 1.0,
    base_score: str = "bic-d",
    domain: str = "general domain",
    llm_client=None,
    llm_model: str = "deepseek-v4-flash-vision-exp",
    max_workers: int = 12,
    priors: Optional[dict] = None,
    constraint=None,
) -> DiscreteBayesianNetwork:
    """Estimate a causal graph and learn the fitted :class:`DiscreteBayesianNetwork`.

    See :func:`discover_structure` for the ``constraint`` and ``llm_model``
    behaviour; all other arguments are unchanged from the original signature.
    """
    columns = df.columns.tolist()
    data = df.astype(str)

    adj = discover_structure(
        df=df,
        algorithm=algorithm,
        var_meta=var_meta,
        pair_meta=pair_meta,
        domain=domain,
        llm_client=llm_client,
        llm_model=llm_model,
        max_workers=max_workers,
        priors=priors,
        lambda_prior=lambda_prior,
        base_score=base_score,
        constraint=constraint,
    )

    edges = _adj_to_edges(adj, columns)

    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(columns)

    isolated = [c for c in columns if c not in model.nodes()]
    if isolated:
        model.add_nodes_from(isolated)

    model.fit(data, estimator=MaximumLikelihoodEstimator)

    return model
