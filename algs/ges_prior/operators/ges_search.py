# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .graph import *
from .utils import subset_generator
from .inserter import insert, insert_validity
from .deleter import delete, delete_validity
from .utils import apply_score



def fes(C, criterion, allowed=None, legacy=False):
    """
    Forward Equivalence Search

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str or DecomposableScore
        scoring, one of ['bic', 'bdeu'].
    allowed: np.ndarray, optional
        (d, d) boolean matrix; ``allowed[i, j]`` means ``i -> j`` is a valid
        direction.  Only edges whose skeleton is allowed are inserted.
    legacy: bool
        if True, use gCastle's original (wrong) `dag_to_cpdag`

    Returns
    -------
    out: np.array
        cpdag
    """

    while True:
        edge, t = forward_search(C, criterion, allowed)
        if edge is None:
            break
        x, y = edge
        C = insert(x, y, t, C)
        C = pdag_to_cpdag(C, legacy=legacy)

    return C


def forward_search(C, criterion, allowed=None):
    """
    forward search

    starts with an empty (i.e., no-edge) CPDAG and greedily applies GES
    insert operators until no operator has a positive score.

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str or DecomposableScore
        scoring, one of ['bic', 'bdeu'].
    allowed: np.ndarray, optional
        (d, d) boolean matrix.  ``insert(x, y, T)`` is skipped when the skeleton
        of ``{x, y}`` or of ``{t, y}`` for any ``t`` in ``T`` is forbidden.

    Returns
    -------
    out: tuple
        ((X, Y), T), the edge (X, Y) denotes X->Y is valid and T is a subset of
        the neighbors of Y that are not adjacent to X,
    """

    d = C.shape[0]
    edge = None
    subset = {}
    best = 0
    skel_allowed = (allowed | allowed.T) if allowed is not None else None
    V = np.arange(d)
    for x in V:
        Vy = connect(x, C, relation=None)
        for y in Vy:
            if skel_allowed is not None and not skel_allowed[x, y]:
                continue
            T0 = subset_generator(neighbors(y, C) - adjacent(x, C))
            for T in T0:
                if skel_allowed is not None and any(not skel_allowed[t, y] for t in T):
                    continue
                if not insert_validity(x, y, T, C):
                    continue
                # det = f (Y, PaPC (Y) ∪ {X} ∪ T ∪ NAY,X ) − f (Y, PaPC (Y) ∪ T ∪ NAY,X ).
                na_yx = neighbors(y, C) & adjacent(x, C)
                pa_y = parent(y, C)
                pa1 = pa_y | {x} | T | na_yx
                pa2 = pa_y | T | na_yx
                try:
                    det = (apply_score(criterion, y, pa1)
                           - apply_score(criterion, y, pa2))
                except AttributeError:
                    raise AttributeError(f"The  has no attribute named "
                                         f"`local_score`, you can create a class inherit"
                                         f"`DecomposableScore` and implement `local_score`"
                                         f" method.")

                if det > best:
                    best = det
                    edge = (x, y)
                    subset = T
    return edge, subset


def bes(C, criterion, allowed=None, legacy=False):
    """
    Backward Equivalence Search

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str or DecomposableScore
        scoring, one of ['bic', 'bdeu'].
    allowed: np.ndarray, optional
        accepted for signature symmetry (deletion never violates a constraint).

    Returns
    -------
    out: np.array
        cpdag
    """

    while True:
        edge, h = backward_search(C, criterion, allowed, legacy=legacy)
        if edge is None:
            break
        x, y = edge
        C = delete(x, y, h, C, legacy=legacy)
        C = pdag_to_cpdag(C, legacy=legacy)

    return C


def backward_search(C, criterion, allowed=None, legacy=False):
    """
    backward search

    starts with a CPDAG and greedily applies GES delete operators until no
    operator has a positive score.

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str or DecomposableScore
        scoring, one of ['bic', 'bdeu'].
    allowed: np.ndarray, optional
        accepted for signature symmetry (deletion never violates a constraint).
    legacy: bool
        if True, use gCastle's original search, which (1) draws H from the
        neighbors of Y that are NOT adjacent to X and (2) leaves X in both
        parent sets of the score difference, so `det` is always 0 and no edge
        is ever deleted.

    Returns
    -------
    out: tuple
        ((X, Y), H), the edge (X, Y) denotes X->Y is valid and H is a subset of
        the neighbors of Y that are adjacent to X,
    """

    if not legacy:
        return _backward_search(C, criterion)

    d = C.shape[0] # .d
    edge = None
    subset = {}
    best = 0
    V = np.arange(d)
    for x in V:
        Vy = adjacent(x, C)
        for y in Vy:
            H0 = subset_generator(neighbors(y, C) - adjacent(x, C))
            for H in H0:
                if not delete_validity(x, y, H, C):
                    continue
                # det = f (Y, PaPC (Y) ∪ {NAY,X \ H} \ X) − f (Y, PaPC (Y) ∪ {NAY,X \ H}).
                na_yx = neighbors(y, C) & adjacent(x, C)
                pa_y = parent(y, C)
                pa1 = pa_y | ((na_yx - H) - {x})
                pa2 = pa_y | (na_yx - H)
                det = (apply_score(criterion, y, pa1)
                       - apply_score(criterion, y, pa2))
                if det > best:
                    best = det
                    edge = (x, y)
                    subset = H

    return edge, subset


def _backward_search(C, criterion):
    """
    backward search following Chickering (2002), Theorem 17.

    For every edge X -> Y or X - Y and every H subset of NA_{Y,X} (neighbors of
    Y adjacent to X) such that NA_{Y,X} \\ H is a clique,
        det = f(Y, Pa(Y) U (NA_{Y,X} \\ H) \\ {X}) - f(Y, Pa(Y) U (NA_{Y,X} \\ H) U {X}).

    gCastle's version (kept as ``legacy=True``) writes both parent sets without
    X when X - Y is undirected and with X when X -> Y is directed, so the two
    scores are the same number and the backward phase is a no-op. With it, what
    the project called GES was forward search alone: every edge the forward
    phase added to explain away a dependence stayed in the graph.
    """

    d = C.shape[0]
    edge = None
    subset = set()
    best = 0
    for x in range(d):
        for y in np.flatnonzero(C[x, :]):           # X -> Y or X - Y
            y = int(y)
            na_yx = neighbors(y, C) & adjacent(x, C)
            for H in subset_generator(na_yx):
                if not delete_validity(x, y, H, C):
                    continue
                pa = parent(y, C) | (na_yx - H)
                det = (apply_score(criterion, y, pa - {x})
                       - apply_score(criterion, y, pa | {x}))
                if det > best:
                    best = det
                    edge = (x, y)
                    subset = H

    return edge, subset
