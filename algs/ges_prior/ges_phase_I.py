"""
Correct GES (Greedy Equivalence Search) Implementation

A clean implementation following Chickering (2002) using pgmpy's infrastructure.

References:
    Chickering, D. M. (2002). Optimal structure identification with greedy search.
    Journal of machine learning research, 3(Nov), 507-554.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd

from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from .operators.ges_search import fes, bes
#  ,StructureEstimator
# from pgmpy.estimators.ScoreCache import ScoreCache

def ges_phase_i(data: pd.DataFrame,
                use_cache: bool = True,
                scoring_method: Optional[Union[str, BaseStructureScore]] = 'bic-g',
                allowed: Optional[np.ndarray] = None,
                legacy_ges: bool = False
) -> np.ndarray:
    """
    Estimate the CPDAG structure from data.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore
        Scoring method: 'bic-g', 'bic-d', 'k2', 'bdeu', etc.
    use_cache : bool
        Unused; local scores are cached internally by pgmpy's scoring classes.
    allowed : np.ndarray, optional
        (d, d) boolean matrix; ``allowed[i, j]`` means ``i -> j`` is valid.
        Enforced at the skeleton level during forward search.
    legacy_ges : bool
        If True, reproduce gCastle's original GES: its BES never deletes an
        edge and its DAG -> CPDAG conversion is wrong for some DAGs. Only the
        correctness experiment passes it.

    Returns
    -------
    CPDAG
        Estimated adjacency matrix representing CPDAG
    """
    criterion = get_scoring_method(scoring_method, data)

    d = len(data.columns)
    cpdag = np.zeros((d, d), dtype=int)

    # Phase 1: Forward search
    cpdag = fes(cpdag, criterion, allowed, legacy=legacy_ges)

    # Phase 2: Backward search
    cpdag = bes(cpdag, criterion, allowed, legacy=legacy_ges)

    return cpdag

