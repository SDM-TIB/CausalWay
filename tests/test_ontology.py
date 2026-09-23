"""Tests for ontology parsing over the SCLC KG."""

import os

from causalway.ontology import OntologySchema
from causalway._util import local_name

SCLC_TTL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kgs", "ttls", "SCLC_patients.ttl",
)


def test_sclc_schema_shape():
    schema = OntologySchema.from_file(SCLC_TTL)
    s = schema.summary()
    assert s["n_classes"] == 1
    assert s["n_data_properties"] == 10
    assert s["n_object_properties"] == 0


def test_sclc_property_names():
    schema = OntologySchema.from_file(SCLC_TTL)
    prop_names = sorted(local_name(p) for p in schema.data_properties)
    assert prop_names == [
        "ageGroup", "biomarker", "episodeType", "familyCancer", "familyGender",
        "gender", "locatedIn", "relapseStatus", "smokerType", "stage",
    ]


def test_sclc_labels_and_comments_present():
    schema = OntologySchema.from_file(SCLC_TTL)
    # Every property carries a label and comment (the LLM-meta source).
    for p in schema.data_properties:
        assert p in schema.labels, f"missing label for {p}"
        assert p in schema.comments, f"missing comment for {p}"


def test_vocab_terms_filtered():
    schema = OntologySchema.from_file(SCLC_TTL)
    # No RDF/RDFS/OWL/RML vocabulary terms should leak into the property set.
    for p in schema.data_properties:
        assert str(p).startswith("http://causalkg.example.org/sclc/")
