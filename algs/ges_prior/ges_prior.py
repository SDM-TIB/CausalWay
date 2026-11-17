"""GES / GES-Prior drivers: Phase I (equivalence-class search) + Phase II (a DAG).

``orient_strategy`` picks Phase II:

    -1  no Phase I; hill climbing from the empty graph under ``scoring_method_i``
     0  return the CPDAG, closed under Meek's rules (and under the constraint's
        forced orientations). The caller orients it at the publication boundary
        (``causalway.result.extend_to_dag``), not here.
     1  orient the CPDAG by the direction prior B, inside the equivalence class
        when unconstrained (``operators.graph.orient_by_B``)
     2  hill climbing under ``scoring_method_ii`` from a consistent extension of
        the CPDAG (Dor & Tarsi) — the GES-Prior default, "GES-M-B-HL"
     3  greedy per-edge orientation by B ("imperfect expert"); kept for
        comparison, it gives no acyclicity guarantee on an undirected triangle
     4  score-greedy orientation restricted to the equivalence class

Strategies 1 and 2 changed in the WWW'27 resource revision; see
``scores/prior_score.py`` (what Phase II scores) and ``operators/graph.py``
(what Phase I searches over) for the reasons.
"""

from .ges_phase_I import ges_phase_i
from .ges_phase_II import ges_phase_ii
from .operators.graph import _apply_meek_rules, _to_dag, enforce_directions, orient_by_B
from typing import Union, Optional
import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.structure_score import BaseStructureScore as StructureScore
from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.base import DAG, PDAG


def _extract_B(scoring_method, n: int) -> np.ndarray:
    """Safely extract the directional prior ``B`` from a scoring method.

    Returns a neutral 0.5 matrix when ``scoring_method`` is a plain string
    (i.e. no priors were supplied) rather than a :class:`StructureScoreWithPrior`.
    """
    B = getattr(scoring_method, "B", None)
    if B is None:
        return np.full((n, n), 0.5)
    return np.asarray(B, dtype=float)


def run_ges_prior(data: pd.DataFrame, use_cache: bool = True,
    scoring_method_i: Optional[Union[str, StructureScore]] = 'bic-g',
    scoring_method_ii: Optional[Union[str, StructureScore]] = 'bic-g',
    orient_strategy=0,
    allowed: Optional[np.ndarray] = None,
    expert_knowledge=None,
    B: Optional[np.ndarray] = None,
    legacy_ges: bool = False,
    hc_start: str = "prior",
):
    """Run GES (Phase I) and turn its CPDAG into the result ``orient_strategy`` asks for.

    ``B`` is the direction prior ``P(i -> j)`` read by strategies 1 and 3; when
    omitted it is taken from ``scoring_method_ii.B`` if that exists and is
    neutral (0.5) otherwise. ``legacy_ges`` reproduces the pre-fix search and is
    for the correctness experiment only. ``hc_start`` chooses the DAG strategy 2
    starts hill climbing from (see the comment there).
    """
    if orient_strategy == -1:
        varnames = data.columns.tolist()
        hc = HillClimbSearch(
            scoring_method=scoring_method_i,
            expert_knowledge=expert_knowledge,
            return_type="dag",
            show_progress=False,
        )
        hc.fit(data)
        best_model = hc.causal_graph_

        dag = nx.to_numpy_array(
            best_model,
            nodelist=varnames,
            dtype=int,
            weight=None
        )
        return dag

    cpdag = ges_phase_i(data, use_cache, scoring_method_i, allowed, legacy_ges=legacy_ges)
    if allowed is not None:
        # Forward search checks the skeleton only, so a *compelled* edge (one a
        # v-structure or a Meek rule fixes) can point the forbidden way. Every
        # strategy below starts from a CPDAG whose directed edges are permitted.
        cpdag = enforce_directions(cpdag, np.asarray(allowed, dtype=bool))
    if B is None:
        B = _extract_B(scoring_method_ii, cpdag.shape[0])

    if orient_strategy == 0:    # meek's rules
        dag = _apply_meek_rules(cpdag, allowed)
    elif orient_strategy == 1:  # orient by B, inside the equivalence class
        dag = orient_by_B(cpdag, B, allowed=allowed)

    elif orient_strategy == 2:  # hill climbing under score II from an extension
        varnames = data.columns.tolist()

        # Where hill climbing starts decides where it ends: it is greedy, and a
        # move that leaves the equivalence class costs BIC that a bounded prior
        # term often cannot pay back. Two starts:
        #
        #   "prior"      the member of the class the direction prior prefers
        #                (``orient_by_B``). Inside a class every member has the
        #                same BIC, so this is the best-scoring member under
        #                BIC+M+B that greedy orientation finds, and hill
        #                climbing only has to decide what lies outside the class.
        #   "extension"  an arbitrary consistent extension (Dor & Tarsi). Kept
        #                because it is the textbook choice and the experiment
        #                compares the two.
        #
        # Either way the start must be a DAG the search is allowed to be in:
        # hill climbing refuses to *add* a forbidden edge but would keep one it
        # was handed, so with a constraint the constraint orients first.
        if hc_start not in ("prior", "extension"):
            raise ValueError(f"hc_start must be 'prior' or 'extension', got {hc_start!r}")
        if hc_start == "prior":
            start = orient_by_B(cpdag, B, allowed=allowed)
        elif allowed is None:
            start = _to_dag(cpdag)
        else:
            start = orient_by_B(cpdag, np.full(cpdag.shape, 0.5), allowed=allowed)

        start_dag = DAG()
        start_dag.add_nodes_from(varnames)
        start_dag.add_edges_from(
            (varnames[i], varnames[j]) for i, j in np.argwhere(start)
        )

        hc = HillClimbSearch(
            scoring_method=scoring_method_ii,
            start_dag=start_dag,
            expert_knowledge=expert_knowledge,
            return_type="dag",
            show_progress=False,
        )
        hc.fit(data)
        best_model = hc.causal_graph_

        dag = nx.to_numpy_array(
            best_model,
            nodelist=varnames,
            dtype=int,
            weight=None
        )
    elif orient_strategy == 3:
        # 'GES-M-B-Greedy', the unperfect expert
        dag = orient_via_imperfect_expert_greedy(cpdag, B=B, allowed=allowed)
    elif orient_strategy == 4:
        # MEC-restricted greedy orientation of the CPDAG.
        dag = ges_phase_ii(cpdag, data, use_cache, scoring_method_ii, allowed=allowed)
    else:
        raise ValueError(f"Unknown orient_strategy {orient_strategy}")
    return dag


def run_ges_hl(data: pd.DataFrame,
               use_cache: bool = True,
               scoring_method: Optional[Union[str, StructureScore]] = 'bic-g',
               expert_knowledge=None):
    varnames = data.columns.tolist()

    # Initialize Hill Climbing search
    hc = HillClimbSearch(
        scoring_method=scoring_method,
        expert_knowledge=expert_knowledge,
        return_type="dag",
        show_progress=False,
    )

    # Estimate the best DAG structure
    hc.fit(data)
    best_model = hc.causal_graph_

    # Add any isolated nodes that might not be in the learned structure
    best_model.add_nodes_from(list(set(varnames).difference(set(best_model.nodes()))))

    # Convert to numpy adjacency matrix with consistent node ordering
    dag = nx.to_numpy_array(
        best_model,
        nodelist=varnames,
        dtype=int,
        weight=None
    )

    return dag


"""
score type

1: GES with normal score
2: GES with prior_i (M = A * C)
3. GES with prior_i (M) and prior_ii (B = M * D)

4. GES with prior_i (B)
"""


def oriente_via_B(cpdag, B, allowed=None):
    n = cpdag.shape[0]
    dag = cpdag.copy()

    # Find undirected edges (where both directions exist)
    # Only iterate over upper triangle to avoid processing each edge twice
    for i in range(n):
        for j in range(i + 1, n):
            # Check if this is an undirected edge
            if cpdag[i, j] == 1 and cpdag[j, i] == 1:
                # Single permitted direction forces that orientation, regardless of B.
                if allowed is not None:
                    if allowed[i, j] and not allowed[j, i]:
                        dag[i, j] = 1
                        dag[j, i] = 0
                        continue
                    if allowed[j, i] and not allowed[i, j]:
                        dag[i, j] = 0
                        dag[j, i] = 1
                        continue
                    if not allowed[i, j] and not allowed[j, i]:
                        dag[i, j] = 0
                        dag[j, i] = 0
                        continue
                # Orient based on B matrix
                if B[i, j] > B[j, i]:
                    # Orient as i -> j
                    dag[i, j] = 1
                    dag[j, i] = 0
                elif B[j, i] > B[i, j]:
                    # Orient as j -> i
                    dag[i, j] = 0
                    dag[j, i] = 1
                else:
                    # Tie-breaking: use lexicographic order (i -> j since i < j)
                    dag[i, j] = 1
                    dag[j, i] = 0

    return dag


def orient_via_imperfect_expert_greedy(cpdag, B, eta=1.0, allowed=None):
    """
    Greedy orientation: iteratively orient the safest edge first.
    """

    dag = cpdag.copy()
    n = cpdag.shape[0]

    # Pre-orient edges whose direction is fixed by the constraint.
    if allowed is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if dag[i, j] == 1 and dag[j, i] == 1:
                    if allowed[i, j] and not allowed[j, i]:
                        dag[i, j] = 1
                        dag[j, i] = 0
                    elif allowed[j, i] and not allowed[i, j]:
                        dag[i, j] = 0
                        dag[j, i] = 1
                    elif not allowed[i, j] and not allowed[j, i]:
                        dag[i, j] = 0
                        dag[j, i] = 0

    while True:
        candidates = []

        for i in range(n):
            for j in range(i + 1, n):
                if dag[i, j] == 1 and dag[j, i] == 1:
                    pij, pji = B[i, j], B[j, i]
                    p_max = max(pij, pji)

                    if p_max >= 1.0 - eta:
                        candidates.append((p_max, i, j))

        # No safe orientation left
        if not candidates:
            break

        # Pick the safest edge
        _, i, j = max(candidates)

        if B[i, j] > B[j, i]:
            dag[i, j] = 1
            dag[j, i] = 0
        else:
            dag[i, j] = 0
            dag[j, i] = 1

        # (Optional) apply Meek rules here if you want propagation

    return dag
