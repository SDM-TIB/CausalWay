"""Tests for polymorphic input resolution (causalway.sources)."""

import os

import rdflib

from causalway.ontology import OntologySchema
from causalway.result import OntologicalCausalGraph
from causalway.sources import is_endpoint, list_ocgs, load_ocg, resolve_graph, resolve_schema

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCLC_TTL = os.path.join(_ROOT, "kgs", "ttls", "SCLC_patients.ttl")
GES_TTL = os.path.join(_ROOT, "results", "sclc", "GES.ttl")
ALL_METHODS_TTL = os.path.join(_ROOT, "results", "sclc", "all_methods.ttl")


def test_is_endpoint():
    assert is_endpoint("http://localhost:3030/ds/sparql")
    assert is_endpoint("https://example.org/sparql")
    assert not is_endpoint(SCLC_TTL)
    assert not is_endpoint(rdflib.Graph())


def test_resolve_graph_passthrough_for_graph_instance():
    g = rdflib.Graph()
    g.parse(SCLC_TTL)
    assert resolve_graph(g) is g


def test_resolve_graph_file():
    g = resolve_graph(SCLC_TTL)
    assert len(g) > 0


def test_resolve_schema_file_infers_missing_by_default():
    schema = resolve_schema(SCLC_TTL)
    assert isinstance(schema, OntologySchema)
    assert schema.summary()["n_data_properties"] == 10


def test_resolve_schema_endpoint_defaults_infer_missing_false(monkeypatch, capsys):
    # Avoid a real network call: patch resolve_graph to return a local graph
    # while still exercising the endpoint-detection branch of resolve_schema.
    import causalway.sources as sources_mod

    g = rdflib.Graph()
    g.parse(SCLC_TTL)
    monkeypatch.setattr(sources_mod, "resolve_graph", lambda *a, **k: g)

    schema = resolve_schema("http://example.org/sparql")
    captured = capsys.readouterr()
    assert "infer_missing defaults to False for endpoints" in captured.out
    # infer_missing=False means undeclared A-Box-only properties are not picked up;
    # SCLC's properties are declared, so this alone doesn't change the count here,
    # but the printed note above is the behavioural contract being tested.
    assert isinstance(schema, OntologySchema)


def test_load_ocg_passthrough_for_instance():
    ocg = OntologicalCausalGraph.from_rdf(resolve_graph(GES_TTL))
    assert load_ocg(ocg) is ocg


def test_load_ocg_from_file_single_graph():
    ocg = load_ocg(GES_TTL)
    assert ocg.n == 10
    assert ocg.method == "GES"


def test_list_ocgs_and_disambiguation():
    df = list_ocgs(ALL_METHODS_TTL)
    assert len(df) >= 2
    assert {"gid", "method", "n_nodes", "n_edges"}.issubset(df.columns)

    method = df.iloc[0]["method"]
    ocg = load_ocg(ALL_METHODS_TTL, method=method)
    assert ocg.method == method
