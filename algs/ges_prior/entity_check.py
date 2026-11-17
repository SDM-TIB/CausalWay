"""Entity-level edge check: is an edge supported by entities, or only by repeated rows?

The problem
-----------
The flat join repeats an entity's attributes on every row the entity appears
in. Twenty hospitals joined to 1,240 therapy rows give 1,240 observations of
``Hospital.region`` and ``Hospital.careQuality`` — of which twenty are
independent. A score computed on the join reads a chance association among
twenty hospitals as an association among 1,240 samples: the likelihood term
grows 62-fold, the penalty does not, and the search adds an edge between two
attributes that have nothing to do with each other. In the benchmark of
``experiments/e2_constraint.py`` edges *inside a replicated class* are the
largest single source of false positives, and Assumption 1 cannot remove them —
they are intra-class, hence admissible.

What this module does
---------------------
In a knowledge graph an attribute is generated once per entity, so the evidence
for ``a -> b`` is the evidence among the entities that own ``b``. After the
search, every edge ``a -> b`` whose family can be read off one row per
``b``-entity is re-scored there, with the delete criterion GES itself uses:

    keep  iff  S(b | Pa(b)) - S(b | Pa(b) \ {a}) > 0      on owner(b)'s entities

A family can be read off one row per ``b``-entity when each of those entities
reaches exactly one entity of every parent's class — parents in the same class,
or across an N:1 relation seen from the N side (a patient has one hospital).
``Hospital.region -> Hospital.careQuality`` is then scored on 20 rows and
``Hospital.careQuality -> Patient.survival`` on 500, not on 1,240. A family with
a parent on the N side (a patient's survival and the toxicity of *each* of
their therapies) has no one-row-per-entity form without an aggregate; it is
left as the search returned it, and that case stays a documented limitation.

Why a check after the search and not a score inside it
-------------------------------------------------------
The obvious design — make the *search* score every family on its own entities —
was implemented first and withdrawn. GES moves between equivalence classes and
relies on Markov-equivalent DAGs scoring the same: an undirected ``x - f`` may be
tested for deletion through either orientation. Once the two orientations are
scored on different numbers of rows (``f | x`` on 720 fact rows, ``x | f`` on 12
dimension entities) they disagree, and the backward phase deletes a true edge
because its weaker reading says so; on the three-variable case in
``tests/test_ges_prior.py`` it returned the empty graph. The check below
compares a child's score with and without one parent *on the same rows*, so it
never mixes sample sizes, and it can only remove edges.

It is therefore conservative in one direction only: it removes edges the
entities do not support, and with few entities it will also remove a true edge
whose effect twenty observations cannot show. That is the honest reading of
twenty observations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pgmpy.structure_score import get_scoring_method

__all__ = ["entity_level_prune"]


def entity_level_prune(adj: np.ndarray, data: pd.DataFrame, owner_of: dict,
                       entity_ids: pd.DataFrame, base_score: str = "bic-d",
                       return_report: bool = False):
    """Remove the edges of ``adj`` that the owning entities do not support.

    Parameters
    ----------
    adj : np.ndarray
        (n, n) adjacency over ``data.columns``; ``adj[i, j] = adj[j, i] = 1`` is
        an undirected edge (a CPDAG is accepted).
    data : pandas.DataFrame
        The flat join the graph was learned on.
    owner_of : dict
        ``{column: owner}`` — the class variable whose entity holds the column.
    entity_ids : pandas.DataFrame
        One column per owner, row-aligned with ``data`` (positionally).
        ``Materialization.entity_ids`` is exactly this.
    return_report : bool
        Also return one record per examined edge (rows it was scored on, score
        difference, kept or not).

    Returns
    -------
    np.ndarray, or (np.ndarray, list) with ``return_report=True``
    """
    if len(entity_ids) != len(data):
        raise ValueError(f"entity_ids has {len(entity_ids)} rows, the data has {len(data)}")
    columns = list(data.columns)
    missing = [c for c in columns if c not in owner_of]
    if missing:
        raise ValueError(f"no owner given for columns {missing}")

    frame = data.reset_index(drop=True)
    codes = {o: pd.factorize(entity_ids[o].to_numpy())[0] for o in entity_ids.columns}
    n_rows = len(frame)

    determines: dict = {}

    def functionally_determines(child_owner: str, parent_owner: str) -> bool:
        """Does every ``child_owner`` entity reach exactly one ``parent_owner`` entity?"""
        key = (child_owner, parent_owner)
        if key not in determines:
            pairs = pd.DataFrame({"c": codes[child_owner], "p": codes[parent_owner]}).drop_duplicates()
            determines[key] = bool(not pairs["c"].duplicated().any())
        return determines[key]

    scorers: dict = {}

    def scorer(owner: str):
        if owner not in scorers:
            keep = ~pd.Series(codes[owner]).duplicated().to_numpy()
            scorers[owner] = (get_scoring_method(base_score, frame.loc[keep]), int(keep.sum()))
        return scorers[owner]

    out = (np.asarray(adj) != 0).astype(int)
    n = len(columns)
    report = []

    def readable(child: int, parents: list) -> bool:
        owner = owner_of[columns[child]]
        return all(functionally_determines(owner, owner_of[columns[p]]) for p in parents)

    for i in range(n):
        for j in range(n):
            if not out[i, j]:
                continue
            undirected = bool(out[j, i])
            if undirected and i > j:
                continue
            # Candidate readings of the edge: parent -> child, with the child's
            # other (directed) parents. An undirected edge has two.
            readings = [(i, j)] + ([(j, i)] if undirected else [])
            for parent, child in readings:
                others = [p for p in range(n) if out[p, child] and not out[child, p] and p != parent]
                family = others + [parent]
                if not readable(child, family):
                    continue
                score, n_entities = scorer(owner_of[columns[child]])
                if n_entities == n_rows:
                    break                     # nothing is replicated; the search's verdict stands
                name = columns[child]
                with_edge = score.local_score(name, tuple(columns[p] for p in family))
                without = score.local_score(name, tuple(columns[p] for p in others))
                keep = with_edge - without > 0
                report.append({"parent": columns[parent], "child": name, "n_entities": n_entities,
                               "n_rows": n_rows, "delta": float(with_edge - without), "kept": bool(keep)})
                if not keep:
                    out[i, j] = out[j, i] = 0
                break
    return (out, report) if return_report else out
