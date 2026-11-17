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

import warnings
from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# What changed against gCastle's GES, and why (WWW'27 resource revision)
#
# This file started as a copy of gCastle's ``ges/functional/graph.py``. Two of
# its routines are wrong, and both are on the path every GES step takes:
#
# * ``dag_to_cpdag`` labels edges compelled / reversible with Chickering's
#   algorithm, but tests ``lowest_edge in np.argwhere(...)``, which is an
#   element-wise membership test rather than a row-wise one. For some DAGs
#   (e.g. 0->1, 0->2, 0->3, 2->3) it keeps a reversible edge directed, so the
#   search walks over graphs that are not CPDAGs. The default below builds the
#   CPDAG the textbook way instead: keep the v-structures, close under Meek's
#   rules R1-R3.
# * the backward phase (``ges_search.backward_search``) never deleted an edge.
#
# ``legacy=True`` keeps the original behaviour on every affected function. It
# exists for one purpose: ``experiments/e1_ges_correctness.py`` measures what
# the fix is worth, and that needs both arms. Nothing else should pass it.
# ---------------------------------------------------------------------------


def _apply_meek_rules(cpdag: np.ndarray, allowed: np.ndarray = None) -> np.ndarray:
    """
    Apply Meek orientation rules to orient edges in a CPDAG.

    Rules:
    R1: Orient i-j into i->j if there's a directed edge k->i and k and j are not adjacent
    R2: Orient i-j into i->j if there's a chain i->k->j
    R3: Orient i-j into i->j if there are two chains i-k->j and i-l->j with k and l not adjacent
    R4: Orient i-j into i->j if i-k->l->j with i and l not adjacent

    Parameters
    ----------
    cpdag : np.ndarray
        CPDAG adjacency matrix (directed + undirected edges)
    allowed : np.ndarray, optional
        (n, n) boolean matrix; ``allowed[i, j]`` means ``i -> j`` is a valid
        direction.  Undirected edges whose reverse is forbidden are pre-oriented,
        and no rule is allowed to orient into a forbidden direction.

    Returns
    -------
    np.ndarray
        DAG adjacency matrix
    """
    n = cpdag.shape[0]
    dag = cpdag.copy()

    def _orient(i, j):
        """Orient i -> j if permitted; else j -> i if that is permitted."""
        if allowed is None or allowed[i, j]:
            if dag[j, i] == 1:
                dag[j, i] = 0
                return True
        elif allowed[j, i]:
            if dag[i, j] == 1:
                dag[i, j] = 0
                return True
        return False

    # Pre-orient every undirected edge whose reverse direction is forbidden.
    if allowed is not None:
        for i in range(n):
            for j in range(n):
                if dag[i, j] == 1 and dag[j, i] == 1:
                    if allowed[i, j] and not allowed[j, i]:
                        dag[j, i] = 0
                    elif allowed[j, i] and not allowed[i, j]:
                        dag[i, j] = 0

    changed = True
    max_iter = 100
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        # Rule 1: Orient i-j into i->j if k->i and k,j not adjacent
        for i in range(n):
            for j in range(n):
                if dag[i, j] == 1 and dag[j, i] == 1:  # Undirected edge
                    for k in range(n):
                        if k != i and k != j:
                            # Check if k->i and k,j not adjacent
                            if dag[k, i] == 1 and dag[i, k] == 0 and dag[k, j] == 0 and dag[j, k] == 0:
                                if _orient(i, j):
                                    changed = True
                                    break

        # Rule 2: Orient i-j into i->j if i->k->j
        for i in range(n):
            for j in range(n):
                if dag[i, j] == 1 and dag[j, i] == 1:  # Undirected edge
                    for k in range(n):
                        if k != i and k != j:
                            # Check if i->k->j
                            if dag[i, k] == 1 and dag[k, i] == 0 and dag[k, j] == 1 and dag[j, k] == 0:
                                if _orient(i, j):
                                    changed = True
                                    break

        # Rule 3: Orient i-j into i->j if i-k->j and i-l->j with k,l not adjacent
        for i in range(n):
            for j in range(n):
                if dag[i, j] == 1 and dag[j, i] == 1:  # Undirected edge
                    for k in range(n):
                        for l in range(n):
                            if k != i and k != j and l != i and l != j and k != l:
                                cond1 = dag[i, k] == 1 and dag[k, i] == 1  # i-k
                                cond2 = dag[k, j] == 1 and dag[j, k] == 0  # k->j
                                cond3 = dag[i, l] == 1 and dag[l, i] == 1  # i-l
                                cond4 = dag[l, j] == 1 and dag[j, l] == 0  # l->j
                                cond5 = dag[k, l] == 0 and dag[l, k] == 0  # k,l not adjacent

                                if cond1 and cond2 and cond3 and cond4 and cond5:
                                    if _orient(i, j):
                                        changed = True
                                        break
                            if dag[i, j] != 1 or dag[j, i] != 1:
                                break

        # Rule 4: Orient i-j into i->j if i-k->l->j with i,l not adjacent
        for i in range(n):
            for j in range(n):
                if dag[i, j] == 1 and dag[j, i] == 1:  # Undirected edge
                    for k in range(n):
                        if k == i or k == j or not (dag[i, k] == 1 and dag[k, i] == 1):
                            continue
                        for l in range(n):
                            if l == i or l == j or l == k:
                                continue
                            # k->l directed, l->j directed, i,l not adjacent
                            if (dag[k, l] == 1 and dag[l, k] == 0
                                    and dag[l, j] == 1 and dag[j, l] == 0
                                    and dag[i, l] == 0 and dag[l, i] == 0):
                                if _orient(i, j):
                                    changed = True
                                    break
                        if dag[i, j] != 1 or dag[j, i] != 1:
                            break

    return dag

def connect(i, G, relation=None) -> set:
    """
    return set of nodes for node i in G with `relation`

    Parameters
    ----------
    i: int
        node
    G: np.ndarray
        a graph matrix
    relation: None or str
        if None, denotes not adjacent

    Returns
    -------
    out: Set
        a set of node or ∅
    """

    from_i = G[i, :] != 0
    to_i = G[:, i] != 0
    if relation is None:
        out = np.where(np.logical_and(~from_i, ~to_i))[0]
    elif relation.lower() == 'neighbors':
        out = np.where(np.logical_and(from_i, to_i))[0]
    elif relation.lower() == 'adjacent':
        out = np.where(np.logical_or(from_i, to_i))[0]
    elif relation.lower() == 'parent':
        out = np.where(np.logical_and(~from_i, to_i))[0]
    elif relation.lower() == 'child':
        out = np.where(np.logical_and(from_i,  ~to_i))[0]
    else:
        raise ValueError(f"The excepted value of parameter `connection` is "
                         f"one of ['neighbors', 'adjacent', None], but got "
                         f"{relation}.")

    return set(out) - {i}


def neighbors(i, G) -> set:
    """return neighbors of node i in graph G"""

    return connect(i, G, relation='neighbors')


def adjacent(i, G) -> set:
    """return adjacent of node i in graph G"""

    return connect(i, G, relation='adjacent')


def parent(i, G) -> set:
    """return parent nodes of i in G"""

    return connect(i, G, relation='parent')


def child(i, G) -> set:
    """return child nodes of i in G"""

    return connect(i, G, relation='child')


def semi_directed_path(x, y, G) -> list:
    """
    Return all paths from x to y in G.
    A semi-directed path between two nodes x and y is a path including both
    directed and undirected edges, but in the case of directed edges,
    they must point to y.

    Parameters
    ----------
    x: int
        the index of the starting node
    y: int
        the index of the target node
    G: np.ndarray
        the adjacency matrix of the graph, where G[i, j] != 0 denotes i -> j.

    Returns
    -------
    semi_paths: list
        all semi-directed paths between x and y
    """

    semi_paths = []
    visitable = {i: child(i, G) | neighbors(i, G) for i in range(G.shape[0])}
    cache = [[x]]
    while len(cache) > 0:
        current_path = cache.pop(0)
        next = list(visitable[current_path[-1]] - set(current_path))
        for next_node in next:
            new_path = current_path.copy()
            new_path.append(next_node)
            if next_node == y:
                semi_paths.append(new_path)
            else:
                cache.append(new_path)

    return semi_paths


def semi_directed_reachable(x, y, G, blocked=()) -> bool:
    """
    Is there a semi-directed path from x to y in G that avoids ``blocked``?

    A semi-directed path follows undirected edges either way and directed edges
    forwards. ``blocked`` nodes may not appear on the path (x and y themselves
    are never treated as blocked). Breadth-first, O(V + E).
    """
    G = np.asarray(G)
    blocked = set(blocked) - {x, y}
    seen = {x}
    frontier = [x]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(G[u, :] != 0):      # u -> v or u - v
                v = int(v)
                if v == y:
                    return True
                if v in seen or v in blocked:
                    continue
                seen.add(v)
                nxt.append(v)
        frontier = nxt
    return False


def is_clique(sub_nodes, C) -> bool:
    """
    check whether the graph corresponding to the sub-nodes is a complete
    subgraph of graph C

    A subgraph over X is complete if every two nodes in X are connected by some
    edge. The set X is often called a clique;

    Parameters
    ----------
    sub_nodes: set
        sub nodes
    C: np.ndarray
        a graph matrix

    Returns
    -------
    out: bool
        whether sub_nodes is clique
    """

    sub_nodes = list(sub_nodes)
    n = len(sub_nodes)
    subgraph = C[sub_nodes, :][:, sub_nodes]
    sub_skeleton = subgraph + subgraph.T
    edges_num = np.sum((sub_skeleton != 0) * 1)

    return edges_num == n * (n - 1)


def is_dag(G) -> bool:
    """check whether a graph G is DAG"""

    p = np.eye(G.shape[0])
    for _ in range(G.shape[0]):
        p = G @ p
        if np.trace(p) != 0:
            return False

    return True


def pdag_to_cpdag(P, legacy=False) -> np.ndarray:
    """
    transform PDAG to CPDAG

    Parameters
    ----------
    P: np.array
        PDAG
    legacy: bool
        if True, use gCastle's original (wrong) `dag_to_cpdag`

    Returns
    -------
    out: np.array
        CPDAG
    """

    G = pdag_to_dag(P)
    C = dag_to_cpdag(G, legacy=legacy)

    return C


def pdag_to_dag(P) -> np.ndarray:
    """
    Return an consistent extension of Partially directed acyclic graph (PDAG)

    References
    ----------
    https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

    Parameters
    ----------
    P: np.array
        PDAG

    Returns
    -------
    out: np.array
        DAG
    """

    G = only_directed_graph(P)
    all_nodes = list(np.arange(len(P)))
    while (len(P) > 0):
        found = False
        i = 0
        while not found and i < len(P):
            # condition1
            cond1 = (len(child(i, P)) == 0)
            # condition2
            n_i = neighbors(i, P)
            adj_i = adjacent(i, P)
            cond2 = np.all([adj_i - {y} <= adjacent(y, P) for y in n_i])
            if cond1 and cond2:
                found = True
                x = all_nodes[i]
                n_x = [all_nodes[j] for j in n_i]
                G[n_x, x] = 1

                but_x = list(set(range(len(P))) - {i})
                P = P[but_x, :][:, but_x]
                all_nodes.remove(x)
            else:
                i += 1
        if not found:
            raise ValueError("The PDAG does not admit any extension.")

    return G


def dag_to_cpdag(G, legacy=False) -> np.ndarray:
    """
    Return the completed partially directed acyclic graph (CPDAG) that
    represents the Markov equivalence class of a given DAG.

    The default keeps the edges of the v-structures directed and closes the
    graph under Meek's rules R1-R3 (Meek, 1995).

    Parameters
    ----------
    G: np.array
        DAG
    legacy: bool
        if True, use gCastle's original `label_edges`-based routine. Its test
        `lowest_edge in np.argwhere(...)` is element-wise rather than row-wise,
        so it returns a wrong CPDAG for some DAGs (e.g. 0->1, 0->2, 0->3, 2->3,
        where it keeps 0->3 directed).

    Returns
    -------
    out: np.array
        CPDAG
    """

    if not legacy:
        G = (np.asarray(G) != 0).astype(int)
        adj = (G + G.T) > 0
        cpdag = adj.astype(int)
        for c in range(G.shape[0]):
            pa = np.flatnonzero(G[:, c])
            for a, b in combinations(pa, 2):
                if not adj[a, b]:       # v-structure a -> c <- b
                    cpdag[c, a] = 0
                    cpdag[c, b] = 0
        return meek_closure(cpdag, apply_r4=False)

    labeled_g = label_edges(G)
    cpdag = np.zeros_like(labeled_g)

    compelled = np.argwhere(labeled_g == 1)
    for x, y in compelled:
        cpdag[x, y] = 1
        cpdag[y, x] = 0
    reversible = np.argwhere(labeled_g == 2)
    for x, y in reversible:
        cpdag[x, y], cpdag[y, x] = 1, 1

    return cpdag


def topological_sort(G) -> list:
    """
    return a topological sort of a graph

    Parameters
    ----------
    G: np.ndarray
        must be a DAG

    Returns
    -------
    out: List
        An ordering of the nodes
    """

    if not is_dag(G):
        raise ValueError(f"The input G is not a DAG.")

    ordering = []
    G = G.copy()
    unmarked = list(np.where(G.sum(axis=0) == 0)[0])
    while len(unmarked) > 0:
        x = unmarked.pop()
        ordering.append(x)
        ch_x = child(x, G)
        for y in ch_x:
            G[x, y] = 0   # delete parent
            if len(parent(y, G)) == 0:
                unmarked.append(y)

    return ordering


def order_edges(G) -> tuple:
    """
    produce a total ordering over the edges in a DAG.

    Parameters
    ----------
    G: np.array
        DAG

    Returns
    -------
    out: tuple
        element 0 denotes order_edges list;
        element 1 denotes ordered DAG
    """

    if not is_dag(G):
        raise ValueError(f"The input G is not a DAG.")

    ordered_notes = topological_sort(G)
    ordered_g = (G != 0) * -1
    y_idx = -1
    i = 1
    ordered_edges = []
    while (ordered_g == -1).any():
        y = ordered_notes[y_idx]
        pa_y = parent(y, G)
        for xi in ordered_notes:
            if xi in pa_y:
                ordered_g[xi, y] = i
                ordered_edges.append((xi, y))
                i += 1
        y_idx -= 1

    return ordered_edges, ordered_g


def label_edges(G) -> np.ndarray:
    """
    label edges with 'compelled' or 'reversible'

    Parameters
    ----------
    G: np.array
        DAG

    Returns
    -------
    out: np.array
        DAG with each edge labeled either "compelled" or "reversible"
    """

    ordered_edges, ordered_g = order_edges(G)

    # define: -1: unknown, 1: compelled, 2: reversible,
    labeled_g = (ordered_g != 0) * -1
    while (labeled_g == -1).any():
        lowest_edge = ordered_edges.pop(-1)
        if lowest_edge in np.argwhere(labeled_g == -1):
            x, y = lowest_edge
            goto = False
            w = np.where(labeled_g[:, x] == 1)[0]
            for each_w in w:
                pa_y = parent(y, labeled_g)
                if each_w not in pa_y:
                    labeled_g[x, y] = 1
                    labeled_g[np.where(labeled_g[:, y] == 1)[0], y] = 1
                    goto = True
                    break
                else:
                    labeled_g[each_w, y] = 1
            if not goto:
                pa_x = parent(x, labeled_g)
                z = parent(y, labeled_g)
                if len(z - {x} - pa_x) > 0:
                    labeled_g[x, y] = 1
                    labeled_g[np.where(labeled_g[:, y] == -1)[0], y] = 1
                else:
                    labeled_g[x, y] = 2
                    labeled_g[np.where(labeled_g[:, y] == -1)[0], y] = 2

    return labeled_g


def only_directed_graph(P) -> np.ndarray:
    """
    return a graph contains all of the directed edges from P and no other edges
    """

    G = P.copy()
    G[(G + G.T) == 2] = 0

    return G


def _has_directed_path(G: np.ndarray, src: int, dst: int) -> bool:
    """Return True if there is a directed path src -> ... -> dst in G.

    Only strictly directed edges count (``G[u, v] != 0 and G[v, u] == 0``), so
    the function is safe to call on a PDAG.
    """
    stack = [src]
    visited = {src}
    while stack:
        u = stack.pop()
        if u == dst:
            return True
        for v in np.flatnonzero((G[u] != 0) & (G[:, u] == 0)):
            v = int(v)
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return False


def _to_dag(cpdag: np.ndarray) -> np.ndarray:
    """
    Return one DAG in the equivalence class represented by a PDAG/CPDAG.

    A numpy port of ``pgmpy.base.PDAG.to_dag`` (Dor & Tarsi, 1992):
    repeatedly pick a node X with (1) no directed children and (2) every
    undirected neighbour of X adjacent to all other neighbours of X; orient
    all undirected edges into X and remove X. If no such node exists the PDAG
    has no consistent extension, and the remaining undirected edges are
    oriented arbitrarily (skipping those that would create a cycle), as pgmpy does.

    Parameters
    ----------
    cpdag : np.ndarray
        PDAG adjacency matrix; cpdag[i, j] = 1 and cpdag[j, i] = 0 denotes i -> j,
        cpdag[i, j] = cpdag[j, i] = 1 denotes i - j.

    Returns
    -------
    np.ndarray
        DAG adjacency matrix
    """
    P = (np.asarray(cpdag) != 0).astype(int)
    n = P.shape[0]
    undirected = (P == 1) & (P.T == 1)
    dag = np.where(undirected, 0, P)          # keep directed edges only

    remaining = list(range(n))
    while remaining:
        found = False
        for x in remaining:
            sub = [v for v in remaining if v != x]
            und_nbrs = [y for y in sub if undirected[x, y]]
            children = [y for y in sub if P[x, y] and not P[y, x]]
            if children:
                continue
            all_nbrs = [z for z in sub if P[x, z] or P[z, x]]
            nbrs_adjacent = all(
                P[y, z] or P[z, y]
                for z in all_nbrs for y in und_nbrs if y != z
            )
            if not und_nbrs or nbrs_adjacent:
                found = True
                for y in und_nbrs:
                    dag[y, x] = 1
                remaining.remove(x)
                break

        if not found:
            warnings.warn(
                "PDAG has no faithful extension (= no oriented DAG with the "
                "same v-structures as PDAG). Remaining undirected PDAG edges "
                "oriented arbitrarily."
            )
            for i in remaining:
                for j in remaining:
                    if P[i, j] and not dag[j, i] and not dag[i, j]:
                        if not _has_directed_path(dag, j, i):
                            dag[i, j] = 1
            break

    return dag


def meek_closure(P: np.ndarray, apply_r4: bool = True, allowed: np.ndarray = None) -> np.ndarray:
    """
    Orient the undirected edges of a PDAG with Meek's rules until no rule applies.

    R1: c -> a - b, c and b not adjacent                        => a -> b
    R2: a -> c -> b and a - b                                   => a -> b
    R3: a - c -> b, a - d -> b, c and d not adjacent, a - b     => a -> b
    R4: a - d, d -> c -> b, a adjacent to c, b and d not adjacent, a - b => a -> b

    Parameters
    ----------
    P : np.ndarray
        PDAG adjacency matrix; P[i, j] = 1 and P[j, i] = 0 denotes i -> j,
        P[i, j] = P[j, i] = 1 denotes i - j.
    apply_r4 : bool
        R4 is only needed when the PDAG holds orientations beyond its v-structures.
    allowed : np.ndarray, optional
        (d, d) boolean matrix. A rule never orients ``a -> b`` when
        ``allowed[a, b]`` is False; that edge is left undirected for the caller,
        which knows what the constraint wants (see ``orient_by_B``).

    Returns
    -------
    np.ndarray
        The maximally oriented PDAG
    """
    P = (np.asarray(P) != 0).astype(int)
    d = P.shape[0]
    changed = True
    while changed:
        changed = False
        for a in range(d):
            for b in range(d):
                und = (P == 1) & (P.T == 1)
                if not und[a, b]:
                    continue
                if allowed is not None and not allowed[a, b]:
                    continue
                dirc = (P == 1) & (P.T == 0)
                adj = (P + P.T) > 0
                r1 = np.any(dirc[:, a] & ~adj[:, b] & (np.arange(d) != b))
                r2 = np.any(dirc[a, :] & dirc[:, b])
                zs = np.flatnonzero(und[a, :] & dirc[:, b])
                r3 = any(not adj[z1, z2] for z1, z2 in combinations(zs, 2))
                r4 = apply_r4 and any(
                    und[a, dd] and not adj[b, dd] and dd != b
                    for c in np.flatnonzero(dirc[:, b] & adj[a, :])
                    for dd in np.flatnonzero(dirc[:, c])
                )
                if r1 or r2 or r3 or r4:
                    P[b, a] = 0
                    changed = True
    return P


def enforce_directions(P: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """
    Make the *directed* edges of a PDAG respect a directional constraint.

    Forward search only checks the skeleton (an equivalence class has no edge
    directions to check), so a compelled edge can point the forbidden way. Such
    an edge is reversed when the reverse is permitted and closes no directed
    cycle, and removed otherwise. Undirected edges are left alone.
    """
    P = (np.asarray(P) != 0).astype(int)
    d = P.shape[0]
    for i in range(d):
        for j in range(d):
            if P[i, j] and not P[j, i] and not allowed[i, j]:
                P[i, j] = 0
                if allowed[j, i] and not _has_directed_path(P, i, j):
                    P[j, i] = 1
    return P


def orient_by_B(cpdag: np.ndarray, B: np.ndarray, allowed: np.ndarray = None) -> np.ndarray:
    """
    Orient the undirected edges of a CPDAG with an asymmetric prior B.

    Without a constraint the result stays inside the Markov equivalence class:
    repeatedly take the undirected edge i - j with the largest
    |B[i, j] - B[j, i]| (ties: lowest (i, j)), orient it towards the larger of
    B[i, j] and B[j, i] (i -> j on a tie), and propagate with Meek's rules
    R1-R4. After the propagation every remaining undirected edge can still be
    oriented either way (Meek, 1995), so no cycle or new v-structure is ever
    created. This replaces the earlier per-edge orientation, which looked at
    each edge in isolation and could close a directed cycle on any undirected
    triangle.

    With a constraint (``allowed``), the constraint outranks both the prior and
    the equivalence class: an edge with one permitted direction is oriented
    first and that way, Meek's rules only fire into permitted directions, and
    an orientation that would close a cycle is reversed if the reverse is
    permitted and dropped otherwise. The result is then a DAG inside the
    constraint, but not necessarily inside the class — Assumption 1 is
    knowledge the class does not have.

    Parameters
    ----------
    cpdag : np.ndarray
        CPDAG adjacency matrix
    B : np.ndarray
        B[i, j] = prior probability of i -> j; NaN is treated as no preference
    allowed : np.ndarray, optional
        (d, d) boolean matrix; ``allowed[i, j]`` means ``i -> j`` is permitted

    Returns
    -------
    np.ndarray
        DAG adjacency matrix
    """
    P = (np.asarray(cpdag) != 0).astype(int)
    B = np.nan_to_num(np.asarray(B, dtype=float), nan=0.5)

    if allowed is None:
        while True:
            und = np.argwhere(np.triu((P == 1) & (P.T == 1), 1))
            if len(und) == 0:
                break
            k = int(np.argmax([abs(B[i, j] - B[j, i]) for i, j in und]))
            i, j = und[k]
            u, v = (j, i) if B[j, i] > B[i, j] else (i, j)
            P[v, u] = 0
            P = meek_closure(P)
        if not (is_dag(P) and np.array_equal(dag_to_cpdag(P), dag_to_cpdag(_to_dag(cpdag)))):
            raise AssertionError("Bug: B orientation left the equivalence class of the CPDAG.")
        return P

    allowed = np.asarray(allowed, dtype=bool)
    P = enforce_directions(P, allowed)
    while True:
        und = [(int(i), int(j)) for i, j in np.argwhere(np.triu((P == 1) & (P.T == 1), 1))]
        if not und:
            break

        def priority(e):
            i, j = e
            forced = allowed[i, j] != allowed[j, i]
            return (forced, abs(B[i, j] - B[j, i]))

        i, j = max(und, key=priority)       # max() keeps the first of equals: lowest (i, j)
        if allowed[i, j] and not allowed[j, i]:
            order = [(i, j)]
        elif allowed[j, i] and not allowed[i, j]:
            order = [(j, i)]
        elif not allowed[i, j] and not allowed[j, i]:
            order = []
        else:
            order = [(j, i), (i, j)] if B[j, i] > B[i, j] else [(i, j), (j, i)]

        P[i, j] = P[j, i] = 0
        for u, v in order:
            if not _has_directed_path(P, v, u):
                P[u, v] = 1
                break
        P = meek_closure(P, allowed=allowed)

    if not is_dag(P):
        raise AssertionError("Bug: constrained B orientation produced a cycle.")
    return P
