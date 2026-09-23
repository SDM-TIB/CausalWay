"""Row <-> entity resolution and scope aggregation (Plan 2 §5.1).

``Materialization.entity_ids`` already holds one column per class variable,
aligned by row index (``causalway/bgp.py``), so everything here is lookup, not
re-querying: an entity resolves to the set of flat-join rows it appears in,
and an entity-level answer is the (mode / mean) aggregate of a per-row
prediction over that row set.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from rdflib import URIRef

from .bgp import Materialization

__all__ = ["rows_for_entity", "entity_of", "aggregate", "population_rows"]


def rows_for_entity(mat: Materialization, entity, var: Optional[str] = None) -> pd.Index:
    """Row indices where ``entity`` is bound to a class variable.

    ``var`` restricts the search to one class variable (a
    ``ColumnSpec.subject_var`` / ``mat.entity_ids`` column, e.g. ``"Patient"``
    for a node whose ``PropertyNode.var == "Patient"``). With ``var=None``
    every class-variable column is searched — the entity may appear as more
    than one class in principle, though in practice each IRI is one type.
    """
    entity = str(entity)
    columns = [var] if var is not None else list(mat.entity_ids.columns)
    mask = pd.Series(False, index=mat.entity_ids.index)
    for c in columns:
        if c not in mat.entity_ids.columns:
            raise KeyError(
                f"rows_for_entity: unknown class variable {c!r}; known: "
                f"{list(mat.entity_ids.columns)}"
            )
        mask = mask | (mat.entity_ids[c].astype(str) == entity)
    return mat.entity_ids.index[mask]


def entity_of(mat: Materialization, row, var: str) -> URIRef:
    """The entity bound to class variable ``var`` in ``row`` (equivalent to ``mat.owner``, by var not column)."""
    return URIRef(str(mat.entity_ids.loc[row, var]))


def aggregate(values: pd.Series, dtype: str) -> Any:
    """Aggregate several per-row predictions for one entity into one answer.

    Mode for categorical/ordinal targets, mean for discrete/continuous ones.
    Returns ``None`` for an all-missing series (an entity whose rows were all
    dropped upstream, e.g. by a required-property join).
    """
    values = values.dropna()
    if len(values) == 0:
        return None
    if dtype in ("categorical", "ordinal"):
        return values.mode(dropna=True).iloc[0]
    return float(values.mean())


def population_rows(mat: Materialization, entity) -> pd.Index:
    """Every row ``entity`` touches, across every class variable it is bound to."""
    return rows_for_entity(mat, entity, var=None)
