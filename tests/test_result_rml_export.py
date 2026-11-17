"""``to_json`` -> ``ocg_mapping.rml.ttl`` -> SDM-RDFizer -> ``from_rdf`` must round-trip.

Three things can break independently and are checked separately:

* the JSON document (:meth:`OntologicalCausalGraph.to_json`) — pure Python,
  fast, no engine involved;
* the mapping — that the engine really produces the cw: terms the vocabulary
  promises (``cw:discoveredBy``, per-graph ``cw:CausalNode``s, typed
  ``cw:parameterValue``s, ``cw:viaRelationalPath`` literals), with the right
  IRIs and datatypes;
* the inverse (:meth:`OntologicalCausalGraph.from_rdf`).

Every OCG here is built from ``kgs/ttls/synthetic_clinic.ttl``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from causalway.constraints import EPSILON, EdgeConstraint
from causalway.nodes import build_nodes
from causalway.ontology import OntologySchema
from causalway.result import (
    GRAPH_STEM, RUN_STEM, CW,
    OntologicalCausalGraph, parse_property_path, render_property_path,
    write_bundle,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC = os.path.join(HERE, "kgs", "ttls", "synthetic_clinic.ttl")

PARAMS = {"alpha": 0.05, "seed": 7, "flag": True,
          "priors": None, "note": "GES over the clinic"}


def _flat_hops(label) -> tuple:
    """A label as a comparable ``((relation, direction), ...)`` sequence."""
    steps = ([label]
             if isinstance(label, tuple) and len(label) == 2
             and not isinstance(label[0], tuple)
             else list(label))
    return tuple((str(rel), direction) for rel, direction in steps)


def _ocg(max_hops: int = 1, **kwargs) -> OntologicalCausalGraph:
    """A real OCG over the synthetic clinic, with a deterministic acyclic adjacency.

    Edges are taken from the constraint's own admissible set, forward in node
    order only (so the graph is a DAG), so that every edge carries a genuine
    relation label — epsilon (intra-class) and cross-class ones and, at
    ``max_hops=2``, multi-hop paths.
    """
    schema = OntologySchema.from_source(CLINIC)
    nodes = build_nodes(schema, include_object_properties=True)
    constraint = EdgeConstraint.from_schema(schema, nodes, max_hops=max_hops)

    n = len(nodes)
    adj = np.zeros((n, n), dtype=int)
    weights = np.zeros((n, n), dtype=float)
    for k, (i, j) in enumerate(p for p in sorted(constraint.labels) if p[0] < p[1]):
        if k >= 12:
            break
        adj[i, j] = 1
        weights[i, j] = round(0.1 * (k + 1), 3)

    return OntologicalCausalGraph.from_discovery(
        adj, constraint, weights=weights, method="GES", params=dict(PARAMS),
        constrained=True, source=CLINIC, graph_id="test-ocg", **kwargs,
    )


# ---------------------------------------------------------------------- #
# The JSON document
# ---------------------------------------------------------------------- #
def test_to_json_is_serializable_and_denormalised():
    import json

    ocg = _ocg()
    payload = ocg.to_json()
    json.dumps(payload)   # must not raise: no numpy scalars, no URIRefs

    assert len(payload["graph"]) == 1
    assert len(payload["run"]) == 1
    assert len(payload["nodes"]) == ocg.n
    assert len(payload["edges"]) == len(ocg.edges())
    # Every record carries the keys its own subject template needs, so no
    # triples map has to join (see ocg_mapping.rml.ttl).
    for section in ("nodes", "edges", "edge_paths", "parameters", "parameters_typed"):
        for record in payload[section]:
            assert record["gid"] == ocg.gid
    # No node order, no counts, no slots: a graph's nodes are a set.
    assert "slots" not in payload
    assert "node_count" not in payload["graph"][0]


def test_maxhop_and_the_constraint_flag_are_parameters_of_the_run():
    params = {r["name"]: r for r in _ocg(max_hops=2).to_json()["parameters_typed"]}
    assert params["maxhop"]["value"] == "2"
    assert params["maxhop"]["datatype"] == str(XSD.integer)
    assert params["topologicalConstraint"]["value"] == "true"
    assert params["topologicalConstraint"]["datatype"] == str(XSD.boolean)
    # A parameter the project documents carries its meaning with it.
    assert "hops" in params["maxhop"]["comment"]


def test_parameter_values_are_typed_literals():
    payload = _ocg().to_json()
    typed = {r["name"]: r["datatype"] for r in payload["parameters_typed"]}
    plain = {r["name"] for r in payload["parameters"]}
    assert typed["alpha"] == str(XSD.double)
    assert typed["seed"] == str(XSD.integer)
    assert typed["flag"] == str(XSD.boolean)
    assert "note" in plain
    # A null parameter is not a setting; it writes no cw:Parameter at all.
    assert "priors" not in plain and "priors" not in typed
    # Every typed row has a datatype: an absent one takes the triples map down.
    assert all(r.get("datatype") for r in payload["parameters_typed"])


def test_numpy_scalar_parameters_type_like_their_python_values():
    ocg = _ocg()
    ocg.params = {"flag": np.bool_(True), "k": np.int64(3), "x": np.float64(0.5)}
    typed = {r["name"]: (r["value"], r["datatype"]) for r in ocg.to_json()["parameters_typed"]}
    assert typed["flag"] == ("true", str(XSD.boolean))
    assert typed["k"] == ("3", str(XSD.integer))
    assert typed["x"] == ("0.5", str(XSD.double))


def test_to_json_fills_created_at_once():
    ocg = _ocg()
    first = ocg.to_json()["run"][0]["discovered_time"]
    second = ocg.to_json()["run"][0]["discovered_time"]
    assert first == second == ocg.created_at


def test_a_run_without_method_or_source_says_so_rather_than_omitting_them():
    ocg = _ocg()
    ocg.method = None
    ocg.source = None
    run = ocg.to_json()["run"][0]
    assert run["method"] == "unspecified" and run["source"] == "unknown"


# ---------------------------------------------------------------------- #
# The mapping, as applied by SDM-RDFizer
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def exported():
    ocg = _ocg(max_hops=2)
    return ocg, ocg.to_rdf()


def test_to_rdf_mints_the_iris_the_python_helpers_promise(exported):
    ocg, g = exported
    assert (ocg.iri, RDF.type, CW.OntologicalCausalGraph) in g
    assert (ocg.run_iri, RDF.type, CW.CausalDiscoveryRun) in g
    assert (ocg.iri, CW.discoveredBy, ocg.run_iri) in g
    for k in range(ocg.n):
        assert (ocg.node_iri(k), RDF.type, CW.CausalNode) in g
        assert (ocg.node_iri(k), CW.inGraph, ocg.iri) in g
    for i, j in ocg.edges():
        assert (ocg.edge_iri(i, j), RDF.type, CW.CausalEdge) in g
        assert (ocg.edge_iri(i, j), CW.inGraph, ocg.iri) in g
    assert str(ocg.iri) == GRAPH_STEM + "test-ocg"
    assert str(ocg.run_iri).startswith(RUN_STEM)
    # Nodes and edges live under their graph; an edge is named by its endpoints.
    assert str(ocg.node_iri(0)).startswith(f"{GRAPH_STEM}test-ocg/node/")
    i, j = ocg.edges()[0]
    assert str(ocg.edge_iri(i, j)).endswith(f"{ocg.node_key(i)}__{ocg.node_key(j)}")


def test_to_rdf_writes_nothing_the_model_retired(exported):
    _, g = exported
    retired = {"hasNode", "hasEdge", "hasNodeSlot", "NodeSlot", "nodeCount", "edgeCount",
               "nodeKind", "viaRelation", "viaPath", "method", "manuallyAdded",
               "usedTopologicalConstraint", "parameterKind", "PropertyNode", "DiscoveryRun"}
    used = {str(t)[len(str(CW)):] for triple in g for t in triple
            if isinstance(t, URIRef) and str(t).startswith(str(CW))
            and "/" not in str(t)[len(str(CW)):]}
    assert not used & retired
    assert not any(str(p).startswith("http://www.w3.org/ns/prov#") for _, p, _ in g)


def test_to_rdf_run_provenance_and_datatypes(exported):
    ocg, g = exported
    assert str(g.value(ocg.run_iri, CW.byMethod)) == "GES"
    assert str(g.value(ocg.run_iri, CW.sourceKG)) == CLINIC
    assert g.value(ocg.run_iri, CW.discoveredTime).datatype == XSD.dateTime
    values = {str(g.value(p, CW.parameterName)): g.value(p, CW.parameterValue)
              for p in g.objects(ocg.run_iri, CW.hasParameter)}
    assert values["alpha"] == Literal(0.05, datatype=XSD.double)
    assert values["seed"] == Literal(7, datatype=XSD.integer)
    assert values["maxhop"] == Literal(2, datatype=XSD.integer)
    assert values["topologicalConstraint"] == Literal(True, datatype=XSD.boolean)
    assert values["note"] == Literal("GES over the clinic")


def test_to_rdf_nodes_carry_domain_property_range(exported):
    ocg, g = exported
    for k, node in enumerate(ocg.nodes):
        iri = ocg.node_iri(k)
        assert g.value(iri, CW.domain) == node.domain
        assert g.value(iri, CW["property"]) == node.prop
        assert g.value(iri, CW["range"]) == node.range_
        assert str(g.value(iri, RDFS.label)) == node.name


def test_an_intra_class_edge_has_no_path_and_a_cross_class_edge_has_one(exported):
    ocg, g = exported
    seen_eps = seen_path = False
    for i, j in ocg.edges():
        paths = list(g.objects(ocg.edge_iri(i, j), CW.viaRelationalPath))
        non_eps = [lab for lab in ocg.edge_labels[(i, j)]
                   if _flat_hops(lab)[0][0] != str(EPSILON)]
        assert len(paths) == len(non_eps)
        if ocg.nodes[i].domain == ocg.nodes[j].domain and not non_eps:
            seen_eps = True
        if non_eps:
            seen_path = True
            for p in paths:
                assert p.datatype is None or p.datatype == XSD.string
    assert seen_eps and seen_path


def test_a_multi_hop_path_is_one_sparql_property_path_literal(exported):
    ocg, g = exported
    paths = [str(p) for i, j in ocg.edges()
             for p in g.objects(ocg.edge_iri(i, j), CW.viaRelationalPath)]
    multi = [p for p in paths if len(parse_property_path(p)) > 1]
    assert multi, "max_hops=2 produced no multi-hop path"
    assert all("<" in p and "/" in p for p in multi)


def test_every_edge_of_a_raw_run_is_weighted_and_not_curated(exported):
    ocg, g = exported
    for i, j in ocg.edges():
        edge = ocg.edge_iri(i, j)
        assert g.value(edge, CW.curated) == Literal(False, datatype=XSD.boolean)
        assert g.value(edge, CW.weight) is not None


def test_a_hand_drawn_edge_in_a_curated_graph_has_no_weight():
    ocg = _ocg()
    hand = ocg.edges()[0]
    ocg.curated = True
    ocg.manual_edges = {hand}
    g = ocg.to_rdf()
    for i, j in ocg.edges():
        edge = ocg.edge_iri(i, j)
        assert g.value(edge, CW.curated) == Literal(True, datatype=XSD.boolean)
        assert (g.value(edge, CW.weight) is None) == ((i, j) == hand)
    back = OntologicalCausalGraph.from_rdf(g)
    assert back.curated is True and back.manual_edges == {hand}


def test_to_rdf_accumulates_into_a_given_graph():
    a, b = _ocg(), _ocg()
    b.graph_id, b.method = "other-ocg", "PC"
    g = Graph()
    a.to_rdf(graph=g)
    b.to_rdf(graph=g)
    assert len(list(g.subjects(RDF.type, CW.OntologicalCausalGraph))) == 2
    # Nodes are per graph: the same (D_p, p, R_p) is two node resources.
    assert a.node_iri(0) != b.node_iri(0)
    assert (a.node_iri(0), CW.inGraph, a.iri) in g and (b.node_iri(0), CW.inGraph, b.iri) in g


def test_to_rdf_keeps_its_workdir_when_asked(tmp_path):
    _ocg().to_rdf(workdir=str(tmp_path))
    assert (tmp_path / "ocg.json").exists()


def test_to_rdf_refuses_a_literal_the_engine_would_corrupt():
    ocg = _ocg()
    ocg.params = {"note": 'it said "hello"'}
    with pytest.raises(ValueError):
        ocg.to_rdf()


def test_write_bundle_compares_methods_on_node_identity_not_iri(tmp_path):
    a, b = _ocg(), _ocg()
    b.graph_id, b.method = "other-ocg", "PC"
    g = write_bundle([a, b], str(tmp_path / "bundle.ttl"))
    rows = g.query("""
        PREFIX cw: <http://sdm-causalway.org/>
        SELECT (COUNT(*) AS ?n) WHERE {
            ?x cw:inGraph ?g1 ; cw:domain ?d ; cw:property ?p ; cw:range ?r .
            ?y cw:inGraph ?g2 ; cw:domain ?d ; cw:property ?p ; cw:range ?r .
            FILTER (?g1 != ?g2 && STR(?g1) < STR(?g2))
        }""")
    assert int(next(iter(rows))[0]) == a.n


# ---------------------------------------------------------------------- #
# The inverse
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("max_hops", [1, 2])
def test_from_rdf_round_trips_the_rml_output(max_hops):
    ocg = _ocg(max_hops=max_hops)
    back = OntologicalCausalGraph.from_rdf(ocg.to_rdf())

    assert [n.name for n in back.nodes] == [n.name for n in ocg.nodes]
    assert [n.kind for n in back.nodes] == [n.kind for n in ocg.nodes]
    assert (back.adj == ocg.adj).all()
    assert np.allclose(back.weights, ocg.weights)
    assert back.method == ocg.method
    assert back.params == {k: v for k, v in ocg.params.items() if v is not None}
    assert back.constrained is True and back.max_hops == max_hops
    assert back.source == CLINIC and back.graph_id == "test-ocg"
    assert back.created_at == ocg.created_at


def test_from_rdf_round_trips_relation_labels():
    ocg = _ocg(max_hops=2)
    back = OntologicalCausalGraph.from_rdf(ocg.to_rdf())
    for edge, labels in ocg.edge_labels.items():
        assert sorted(_flat_hops(x) for x in back.edge_labels[edge]) == \
            sorted(_flat_hops(x) for x in labels)


def test_from_rdf_by_method_picks_one_of_several():
    a, b = _ocg(), _ocg()
    b.graph_id, b.method = "other-ocg", "PC"
    g = Graph()
    a.to_rdf(graph=g)
    b.to_rdf(graph=g)
    assert OntologicalCausalGraph.from_rdf(g, method="PC").graph_id == "other-ocg"
    with pytest.raises(ValueError):
        OntologicalCausalGraph.from_rdf(g)


def test_property_path_round_trips():
    hops = [(URIRef("http://x.org/a/treatedAt"), "inverse"),
            (URIRef("http://x.org/a/receives"), "forward")]
    text = render_property_path(hops)
    assert text == "^<http://x.org/a/treatedAt>/<http://x.org/a/receives>"
    assert parse_property_path(text) == hops
    with pytest.raises(ValueError):
        parse_property_path("<a>|<b>")
