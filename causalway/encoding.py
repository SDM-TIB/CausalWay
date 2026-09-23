"""Literal typing, discretisation and object-property encoding for the flat join.

These are the KG-side data preparations the discovery algorithms need: turning
object-property IRIs into categoricals, discretising continuous columns for
discrete scores, and dropping the constant columns that break BIC.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ._util import local_name


def localize_object_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Replace full-IRI strings in the given columns with their local names."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).map(_localize)
    return df


def _localize(value):
    text = str(value)
    if text.startswith("http") and ("#" in text or "/" in text):
        return local_name(text)
    return value


def discretize(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    method: str = "quantile",
    bins: int = 3,
) -> pd.DataFrame:
    """Discretise numeric columns into integer-coded categories.

    Non-numeric columns are left untouched.  ``method`` is ``"quantile"``
    (``pd.qcut``) or ``"uniform"`` (``pd.cut``).
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        if series.nunique() <= bins:
            # Already low-cardinality; integer-code it deterministically.
            df[col] = series.rank(method="dense").astype(int)
            continue
        try:
            if method == "quantile":
                codes = pd.qcut(series, bins, labels=False, duplicates="drop")
            else:
                codes = pd.cut(series, bins, labels=False)
        except ValueError:
            continue
        df[col] = codes.astype(float)
    return df


def drop_constant_columns(df: pd.DataFrame, verbose: bool = False):
    """Return ``(df, dropped)`` with constant columns removed."""
    dropped = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if verbose and dropped:
        print(f"Dropped constant columns: {dropped}")
    return df.drop(columns=dropped), dropped


def high_cardinality_warnings(
    df: pd.DataFrame, max_levels: int = 20, verbose: bool = False
) -> list:
    """Warn about columns whose distinct-value count exceeds ``max_levels``."""
    warnings_list = []
    for c in df.columns:
        n = df[c].nunique(dropna=True)
        if n > max_levels:
            msg = f"Column '{c}' has {n} distinct values (> {max_levels})"
            warnings_list.append(msg)
            if verbose:
                print(f"Warning: {msg}")
    return warnings_list


def to_discrete_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All columns as string categories (for ``bic-d``/``bdeu``/``k2``)."""
    return df.astype(str)


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-code categorical columns, keep numeric (for continuous methods).

    Mirrors the label-encoding used inside ``algs.discovery_alg`` so the KG
    runner can produce the frame the continuous-only methods expect.
    """
    from sklearn.preprocessing import LabelEncoder

    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col].dtype) and not pd.api.types.is_bool_dtype(
            out[col].dtype
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            continue
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
    return out
