"""Tests for row <-> entity resolution (causalway.entities), over the SCLC materialisation."""

import os

import pandas as pd
import pytest

from causalway.bgp import materialize
from causalway.entities import aggregate, entity_of, population_rows, rows_for_entity
from causalway.nodes import build_nodes
from causalway.ontology import OntologySchema

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCLC_TTL = os.path.join(_ROOT, "kgs", "ttls", "SCLC_patients.ttl")


@pytest.fixture(scope="module")
def mat():
    schema = OntologySchema.from_file(SCLC_TTL)
    nodes = build_nodes(schema)
    m = materialize(schema, nodes, limit=200)
    assert not isinstance(m, list)
    return m


def test_rows_for_entity_single_row_per_patient(mat):
    entity = mat.entity_ids.iloc[0, 0]
    rows = rows_for_entity(mat, entity, var="SCLCPatient")
    assert list(rows) == [0]


def test_rows_for_entity_unknown_var_raises(mat):
    with pytest.raises(KeyError):
        rows_for_entity(mat, mat.entity_ids.iloc[0, 0], var="NotAClass")


def test_entity_of_matches_owner(mat):
    e1 = entity_of(mat, 0, "SCLCPatient")
    e2 = mat.owner("SCLCPatient.ageGroup", 0)
    assert str(e1) == str(e2)


def test_population_rows_equals_rows_for_entity_no_var(mat):
    entity = mat.entity_ids.iloc[5, 0]
    assert list(population_rows(mat, entity)) == list(rows_for_entity(mat, entity))


def test_aggregate_categorical_uses_mode():
    values = pd.Series(["a", "a", "b"])
    assert aggregate(values, "categorical") == "a"


def test_aggregate_continuous_uses_mean():
    values = pd.Series([1.0, 2.0, 3.0])
    assert aggregate(values, "continuous") == pytest.approx(2.0)


def test_aggregate_all_missing_returns_none():
    values = pd.Series([None, None], dtype=object)
    assert aggregate(values, "categorical") is None
