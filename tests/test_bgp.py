"""Tests for BGP construction / flat-join materialisation over SCLC."""

import os

import pandas as pd
import pytest

from causalway.ontology import OntologySchema
from causalway.nodes import build_nodes
from causalway.constraints import EdgeConstraint
from causalway.bgp import (
    DisconnectedJoinError,
    build_query,
    join_role_options,
    materialize,
    object_property_role,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCLC_TTL = os.path.join(_ROOT, "kgs", "ttls", "SCLC_patients.ttl")
SCLC_CSV = os.path.join(_ROOT, "kgs", "csvs", "SCLC_patients.csv")
CLINIC_TTL = os.path.join(_ROOT, "kgs", "ttls", "synthetic_clinic.ttl")

_MAPPING = {
    "SCLCPatient.ageGroup": "Age",
    "SCLCPatient.gender": "Gender",
    "SCLCPatient.smokerType": "SmokerType",
    "SCLCPatient.familyCancer": "FamilyCancer",
    "SCLCPatient.biomarker": "Biomarker",
    "SCLCPatient.episodeType": "EpisodeType",
    "SCLCPatient.relapseStatus": "Relapse",
    "SCLCPatient.locatedIn": "LocatedIn",
    "SCLCPatient.stage": "Stage",
    "SCLCPatient.familyGender": "FamilyGender",
}


def _materialize():
    schema = OntologySchema.from_file(SCLC_TTL)
    nodes = build_nodes(schema)
    constraint = EdgeConstraint.from_schema(schema, nodes)
    mat = materialize(schema, nodes)
    assert not isinstance(mat, list), "SCLC should be a single connected component"
    return schema, nodes, constraint, mat


def test_shape():
    _, nodes, _, mat = _materialize()
    assert mat.df.shape == (4279, 10)
    assert mat.df.columns.tolist() == [n.name for n in nodes]


def test_entity_ids():
    _, _, _, mat = _materialize()
    assert mat.entity_ids.shape == (4279, 1)
    assert mat.entity_ids.columns.tolist() == ["SCLCPatient"]
    # deterministic natural ordering by entity IRI
    assert mat.entity_ids.iloc[0, 0].endswith("patient_0")
    assert mat.entity_ids.iloc[-1, 0].endswith("patient_4278")


def test_multiplicity_is_one():
    _, _, _, mat = _materialize()
    # one row per patient (no object properties)
    assert mat.multiplicity["SCLCPatient"] == 1.0


def test_matches_csv():
    _, _, _, mat = _materialize()
    csv = pd.read_csv(SCLC_CSV)
    assert len(csv) == 4279

    value_cols = list(_MAPPING.values())
    renamed = mat.df.rename(columns=_MAPPING)
    actual = renamed[value_cols].reset_index(drop=True).astype(str)
    expected = csv[value_cols].reset_index(drop=True).astype(str)
    assert actual.equals(expected), "flat join does not reproduce the source CSV"


def test_owner_roundtrip():
    _, _, _, mat = _materialize()
    owner = mat.owner("SCLCPatient.ageGroup", 0)
    assert str(owner).endswith("patient_0")


def test_disconnected_join_refused_unlimited():
    """A curated-away join leaves classes disconnected; an unlimited run over that
    would silently compute their full cross product, so it is refused outright
    rather than left to actually run (Preprocess revision 3 performance fix)."""
    schema = OntologySchema.from_file(CLINIC_TTL)
    nodes = build_nodes(schema)
    excluded_joins = {n.name for n in nodes if n.kind == "object"}
    with pytest.raises(DisconnectedJoinError):
        materialize(schema, nodes, excluded_joins=excluded_joins, limit=None)


def test_disconnected_join_allowed_with_limit():
    schema = OntologySchema.from_file(CLINIC_TTL)
    nodes = build_nodes(schema)
    excluded_joins = {n.name for n in nodes if n.kind == "object"}
    mat = materialize(schema, nodes, excluded_joins=excluded_joins, limit=25)
    assert mat.df.shape[0] == 25


def test_build_query_matches_materialize_without_running():
    """`build_query` is the live preview path — it must emit the exact text
    `materialize` runs, without needing a limit or executing anything, even for
    curation that `materialize` itself would refuse to run unlimited."""
    schema = OntologySchema.from_file(CLINIC_TTL)
    nodes = build_nodes(schema)
    excluded_joins = {n.name for n in nodes if n.kind == "object"}

    query, warnings = build_query(schema, nodes, excluded_joins=excluded_joins, limit=None)
    assert "SELECT" in query
    assert any("cross product" in w for w in warnings)

    limited = materialize(schema, nodes, excluded_joins=excluded_joins, limit=10)
    query_limited, _ = build_query(schema, nodes, excluded_joins=excluded_joins, limit=10)
    assert limited.query == query_limited


# --- object-property roles (relationship XOR causal variable) ----------------


def _clinic():
    schema = OntologySchema.from_file(CLINIC_TTL)
    return schema, build_nodes(schema, include_object_properties=True)


def test_an_object_property_defaults_to_relationship_not_variable():
    """Uncurated, both sets empty: every object property is a join and none is a
    column. It used to be silently both, which put a column in the frame that was
    a deterministic function of the join that produced the row."""
    schema, nodes = _clinic()
    objs = [n for n in nodes if n.kind == "object"]
    assert objs, "the clinic schema should have object properties to curate"

    for n in objs:
        assert object_property_role(n.name, set(), set()) == "relationship"

    mat = materialize(schema, nodes, excluded=set(), excluded_joins=set())
    assert not any(n.name in mat.df.columns for n in objs)
    # the join itself is untouched — same rows as before the roles were split
    assert mat.df.shape[0] == 1240


def test_the_two_roles_are_mutually_exclusive():
    assert object_property_role("p", set(), set()) == "relationship"
    # keeping the column but dropping the join is the only way to get a variable
    assert object_property_role("p", set(), {"p"}) == "variable"
    assert object_property_role("p", {"p"}, {"p"}) == "dropped"
    # excluding the column alone leaves the join, hence still a relationship
    assert object_property_role("p", {"p"}, set()) == "relationship"


def test_a_variable_object_property_keeps_the_full_iri():
    """`cf_export` needs the IRI to emit an rr:IRI term; shortening to a local
    name here destroyed information nothing downstream could recover."""
    schema, nodes = _clinic()
    therapy = {n.name for n in nodes if n.name.startswith("Therapy.")}
    mat = materialize(schema, nodes,
                      excluded=therapy,
                      excluded_joins={"Patient.receives"})
    value = str(mat.df["Patient.receives"].iloc[0])
    assert value.startswith("http://"), value


def test_a_variable_object_property_does_not_pull_its_range_class_in():
    """The value comes off a private pattern, so choosing "variable" really does
    keep the range class's own properties out of the row."""
    schema, nodes = _clinic()
    therapy = {n.name for n in nodes if n.name.startswith("Therapy.")}
    mat = materialize(schema, nodes,
                      excluded=therapy,
                      excluded_joins={"Patient.receives"})
    assert not any(str(c).endswith("Therapy") for c in mat.classes)
    assert not any(c.startswith("Therapy.") for c in mat.df.columns)
    assert mat.df.shape[0] == 1240


def test_the_variable_role_is_refused_when_it_would_split_the_query():
    """Dropping the last join between two classes leaves SPARQL computing their
    cross product; module 1 disables the choice rather than letting the user find
    out at materialise time."""
    schema, nodes = _clinic()
    opts = join_role_options(schema, nodes)
    assert opts["Patient.receives"]["role"] == "relationship"
    assert opts["Patient.receives"]["can_be_variable"] is False
    assert "cross product" in opts["Patient.receives"]["reason"]


def test_the_variable_role_opens_up_once_the_range_class_is_not_needed():
    """Availability is decided by building the pattern the choice would produce,
    not by reasoning about the join graph: a dropped join that also drops its
    range class leaves a perfectly connected query."""
    schema, nodes = _clinic()
    therapy = {n.name for n in nodes if n.name.startswith("Therapy.")}
    opts = join_role_options(schema, nodes, excluded=therapy)
    assert opts["Patient.receives"]["can_be_variable"] is True
    assert opts["Patient.receives"]["reason"] is None
    # Hospital's nodes are still retained, so its join is still load-bearing
    assert opts["Patient.treatedAt"]["can_be_variable"] is False


def test_a_conflict_that_already_exists_is_reported_not_only_prevented():
    """The same test runs whatever the current role is, so module 1 can flag a
    curation that is already split rather than only blocking a new choice."""
    schema, nodes = _clinic()
    opts = join_role_options(schema, nodes, excluded=set(),
                             excluded_joins={"Patient.receives"})
    assert opts["Patient.receives"]["role"] == "variable"
    assert opts["Patient.receives"]["can_be_variable"] is False


def test_an_unrelated_dropped_join_does_not_make_every_role_unavailable():
    """Availability is relative — a role is refused only when it leaves strictly
    more groups than the other one would. An absolute "is this query
    disconnected?" test would blame every object property for one unrelated
    canvas edit, leaving the user no way to tell which choice was the problem."""
    schema, nodes = _clinic()
    therapy = {n.name for n in nodes if n.name.startswith("Therapy.")}
    hospital = {n.name for n in nodes if n.name.startswith("Hospital.")}
    # Hospital is already split off by a dropped join, yet `receives` is not to
    # blame for it and can still legitimately become a variable.
    opts = join_role_options(
        schema, nodes,
        excluded=therapy | hospital,
        excluded_joins={"Patient.treatedAt"},
    )
    assert opts["Patient.receives"]["can_be_variable"] is True
