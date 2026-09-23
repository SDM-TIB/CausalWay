"""``to_json`` -> ``ocg_mapping.rml.ttl`` -> SDM-RDFizer -> ``from_rdf`` must round-trip.

The export stopped being hand-written rdflib and became a declarative RML
mapping applied by SDM-RDFizer, which means three separate things can now
break independently and are checked separately here:

* the JSON document (:meth:`OntologicalCausalGraph.to_json`) — pure Python,
  fast, no engine involved;
* the mapping — that the engine really produces the cw: terms the vocabulary
  promises, with the right IRIs, datatypes and cardinalities;
* the inverse (:meth:`OntologicalCausalGraph.from_rdf`) — including its
  fallback onto the *legacy* ``cw:nodeList`` / ``cw:parameters`` shapes,
  since nothing regenerated the Turtle already sitting in ``results/``.

Note that those ``results/*.ttl`` files cannot be used as fixtures: they were
written when the vocabulary namespace was ``urn:causalway:``, so ``from_rdf``
finds no ``cw:OntologicalCausalGraph`` in them at all. That is the
pre-existing failure in ``test_result_roundtrip.py`` / ``test_sources.py``,
and it predates this module. Everything below builds its OCG from
``kgs/ttls/synthetic_clinic.ttl`` instead.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD
from rdflib.collection import Collection

from causalway.constraints import EPSILON, EdgeConstraint
from causalway.nodes import build_nodes
from causalway.ontology import OntologySchema
from causalway.result import (
    GRAPH_STEM, NODE_STEM, RUN_STEM, CW, PROV,
    OntologicalCausalGraph, parse_property_path, render_property_path,
    write_bundle,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC = os.path.join(HERE, "kgs", "ttls", "synthetic_clinic.ttl")

PARAMS = {"alpha": 0.05, "constrained": True, "n_runs": 3,
          "priors": None, "note": "GES over the clinic"}


def _flat_hops(label) -> tuple:
    """A label as a comparable ``((relation, direction), ...)`` sequence."""
    steps = ([label]
             if isinstance(label, tuple) and len(label) == 2
             and not isinstance(label[0], tuple)
             else list(label))
    return tuple((str(rel), direction) for rel, direction in steps)


def _ocg(max_hops: int = 1) -> OntologicalCausalGraph:
    """A real OCG over the synthetic clinic, with a deterministic adjacency.

    Edges are taken from the constraint's own admissible set so that every
    edge carries a genuine relation label — an epsilon (intra-class) one and,
    at ``max_hops=2``, a ``cw:RelationPath``.
    """
    schema = OntologySchema.from_source(CLINIC)
    nodes = build_nodes(schema, include_object_properties=True)
    constraint = EdgeConstraint.from_schema(schema, nodes, max_hops=max_hops)

    n = len(nodes)
    adj = np.zeros((n, n), dtype=int)
    weights = np.zeros((n, n), dtype=float)
    for k, (i, j) in enumerate(sorted(constraint.labels)):
        if k >= 12:
            break
        adj[i, j] = 1
        weights[i, j] = round(0.1 * (k + 1), 3)

    return OntologicalCausalGraph.from_discovery(
        adj, constraint, weights=weights, method="GES", params=dict(PARAMS),
        constrained=True, source=CLINIC, graph_id="test-ocg",
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
    assert len(payload["slots"]) == ocg.n
    assert len(payload["edges"]) == len(ocg.edges())
    assert len(payload["parameters"]) == len(PARAMS)

    # Every record carries the keys its own subject template needs, so no
    # triples map has to join (see ocg_mapping.rml.ttl).
    for section in ("nodes", "slots", "edges", "edge_relations", "parameters"):
        for record in payload[section]:
            assert record["gid"] == ocg.gid
    for record in payload["edge_relations"] + payload["edge_paths"]:
        assert "edge_key" in record
    assert "paths" not in payload and "hops" not in payload


def test_to_json_omits_absent_fields_rather_than_nulling_them():
    ocg = _ocg()
    ocg.method = None
    ocg.constrained = None
    ocg.source = None
    payload = ocg.to_json()

    for key in ("method", "constrained"):
        assert key not in payload["run"][0]
    assert "source" not in payload["run"][0]

    # `method` and the constraint flag belong to the run, never to the graph.
    assert "method" not in payload["graph"][0]
    assert "constrained" not in payload["graph"][0]

    # One range term whatever the node kind — cw:nodeKind carries the distinction.
    for record in payload["nodes"]:
        assert "range" in record
        assert "range_class" not in record and "range_datatype" not in record

    # A discovered edge is not marked manual.
    assert all("manual" not in e for e in payload["edges"])


def test_to_json_types_every_parameter_kind():
    payload = _ocg().to_json()
    kinds = {p["name"]: (p["kind"], p.get("value")) for p in payload["parameters"]}
    assert kinds["alpha"] == ("float", "0.05")
    assert kinds["constrained"] == ("bool", "true")   # bool before int
    assert kinds["n_runs"] == ("int", "3")
    assert kinds["priors"] == ("null", None)   # no value key at all
    assert kinds["note"] == ("str", "GES over the clinic")


def test_numpy_scalar_parameters_type_like_their_python_values():
    """A flag read back off a pandas index is np.bool_, not bool (notebook 05)."""
    ocg = _ocg()
    ocg.params = {"constrained": np.bool_(True), "n_runs": np.int64(3),
                  "alpha": np.float64(0.05)}
    kinds = {p["name"]: (p["kind"], p.get("value"))
             for p in ocg.to_json()["parameters"]}
    assert kinds == {"constrained": ("bool", "true"), "n_runs": ("int", "3"),
                     "alpha": ("float", "0.05")}
    ocg.to_rdf()   # used to raise: the JSON branch wrote the quoted text "True"


def test_to_json_fills_created_at_once():
    ocg = _ocg()
    assert ocg.created_at is None
    ocg.to_json()
    stamp = ocg.created_at
    assert stamp is not None
    ocg.to_json()
    assert ocg.created_at == stamp


# ---------------------------------------------------------------------- #
# The mapping, as applied by SDM-RDFizer
# ---------------------------------------------------------------------- #
def test_to_rdf_mints_the_iris_the_python_helpers_promise():
    """The mapping hardcodes its IRI stems; this is what catches them drifting."""
    ocg = _ocg()
    g = ocg.to_rdf()

    assert (ocg.iri, RDF.type, CW.OntologicalCausalGraph) in g
    assert (ocg.iri, RDF.type, PROV.Entity) in g
    assert (ocg.run_iri, RDF.type, CW.DiscoveryRun) in g
    assert (ocg.iri, PROV.wasGeneratedBy, ocg.run_iri) in g
    for k in range(ocg.n):
        assert (ocg.node_iri(k), RDF.type, CW.PropertyNode) in g
        assert (ocg.iri, CW.hasNode, ocg.node_iri(k)) in g
        assert (ocg.slot_iri(k), CW.slotNode, ocg.node_iri(k)) in g
    for (i, j) in ocg.edges():
        assert (ocg.iri, CW.hasEdge, ocg.edge_iri(i, j)) in g
        assert (ocg.edge_iri(i, j), CW.inGraph, ocg.iri) in g

    assert str(ocg.iri).startswith(GRAPH_STEM)
    assert str(ocg.run_iri).startswith(RUN_STEM)
    assert str(ocg.node_iri(0)).startswith(NODE_STEM)


def test_to_rdf_datatypes_and_provenance():
    ocg = _ocg()
    g = ocg.to_rdf()

    assert g.value(ocg.iri, CW.nodeCount) == Literal(ocg.n, datatype=XSD.integer)
    assert g.value(ocg.iri, CW.edgeCount) == Literal(len(ocg.edges()),
                                                      datatype=XSD.integer)
    # The run holds the method and the constraint flag; the graph does not
    # duplicate them, it reaches them through prov:wasGeneratedBy.
    assert g.value(ocg.iri, CW.method) is None
    assert g.value(ocg.iri, CW.constrained) is None
    assert str(g.value(ocg.run_iri, CW.method)) == "GES"
    assert g.value(ocg.run_iri, CW.usedTopologicalConstraint) == Literal(
        True, datatype=XSD.boolean)
    assert str(g.value(ocg.run_iri, CW.sourceKG)) == CLINIC
    assert str(g.value(ocg.run_iri, PROV.used)) == CLINIC
    assert g.value(ocg.run_iri, PROV.endedAtTime).datatype == XSD.dateTime

    for (i, j) in ocg.edges():
        weight = g.value(ocg.edge_iri(i, j), CW.weight)
        assert weight.datatype == XSD.double
        assert float(weight) == pytest.approx(ocg.weight(i, j))


def test_to_rdf_writes_parameters_as_resources_not_a_json_literal():
    ocg = _ocg()
    g = ocg.to_rdf()

    assert g.value(ocg.run_iri, CW.parameters) is None    # the legacy shape is gone
    found = {}
    for param in g.objects(ocg.run_iri, CW.hasParameter):
        assert (param, RDF.type, CW.Parameter) in g
        value = g.value(param, CW.parameterValue)
        found[str(g.value(param, CW.parameterName))] = (
            str(g.value(param, CW.parameterKind)),
            None if value is None else str(value),
        )
    assert found["alpha"] == ("float", "0.05")
    assert found["priors"] == ("null", None)
    assert set(found) == set(PARAMS)


def test_to_rdf_orders_nodes_by_slot_index():
    ocg = _ocg()
    g = ocg.to_rdf()

    slots = sorted(
        (int(g.value(s, CW.slotIndex)), g.value(s, CW.slotNode))
        for s in g.objects(ocg.iri, CW.hasNodeSlot)
    )
    assert [k for k, _ in slots] == list(range(ocg.n))
    assert [iri for _, iri in slots] == [ocg.node_iri(k) for k in range(ocg.n)]
    assert g.value(ocg.iri, CW.nodeList) is None   # no rdf:List any more


def test_to_rdf_labels_epsilon_and_relation_edges():
    ocg = _ocg(max_hops=1)
    g = ocg.to_rdf()

    saw_epsilon = saw_relation = False
    for (i, j) in ocg.edges():
        edge = ocg.edge_iri(i, j)
        relations = set(g.objects(edge, CW.viaRelation))
        assert relations, f"edge {i}->{j} carries no cw:viaRelation"
        if EPSILON in relations:
            saw_epsilon = True
        if relations - {EPSILON}:
            saw_relation = True
    assert saw_epsilon and saw_relation

    # cw:epsilon has no direction, so it never gets a path; the reified path
    # resources and their five properties are gone entirely.
    for term in (CW.RelationPath, CW.RelationHop):
        assert not list(g.subjects(RDF.type, term))
    for term in (CW.relationDirection, CW.relationLabel, CW.pathLength,
                 CW.hop, CW.hopIndex, CW.hopRelation, CW.hopDirection):
        assert not list(g.subject_objects(term)), f"{term} should no longer be written"


def test_to_rdf_writes_a_multi_hop_label_as_a_sparql_property_path():
    ocg = _ocg(max_hops=2)
    g = ocg.to_rdf()

    paths = [(s, str(o)) for s, o in g.subject_objects(CW.viaPath)]
    assert paths, "max_hops=2 produced no cw:viaPath"
    saw_multi = False
    for edge, text in paths:
        assert isinstance(g.value(edge, CW.viaPath), Literal)
        hops = parse_property_path(text)
        saw_multi = saw_multi or len(hops) > 1
        for rel, direction in hops:
            assert direction in {"forward", "inverse"}
            # every step's relation is also listed flat on the edge, so a
            # "which edges use this property" query needs no path parsing
            assert (edge, CW.viaRelation, rel) in g
        assert render_property_path(hops) == text   # the syntax round-trips
    assert saw_multi, "no multi-hop path in the fixture"


def test_to_rdf_records_curation_provenance():
    """A curated graph must say which edges the human added and what it came from.

    Without this the curated Turtle is indistinguishable from a discovered one,
    and "which edges did the analyst put there" — the question the curation step
    exists to answer — cannot be asked of the export at all.
    """
    ocg = _ocg()
    hand_drawn = ocg.edges()[0]
    ocg.manual_edges = {hand_drawn}
    ocg.derived_from = [f"{RUN_STEM}run-1", f"{RUN_STEM}run-2"]
    g = ocg.to_rdf()

    flag = Literal(True, datatype=XSD.boolean)
    assert (ocg.edge_iri(*hand_drawn), CW.manuallyAdded, flag) in g
    # A discovered edge gets no triple at all, not `false`.
    for (i, j) in ocg.edges():
        if (i, j) != hand_drawn:
            assert g.value(ocg.edge_iri(i, j), CW.manuallyAdded) is None
    assert set(g.objects(ocg.iri, PROV.wasDerivedFrom)) == {
        URIRef(f"{RUN_STEM}run-1"), URIRef(f"{RUN_STEM}run-2")}

    back = OntologicalCausalGraph.from_rdf(g)
    assert back.manual_edges == {hand_drawn}
    assert sorted(back.derived_from) == sorted(ocg.derived_from)


def test_to_rdf_accumulates_into_a_given_graph():
    ocg = _ocg()
    g = Graph()
    ocg.to_rdf(graph=g)
    n_first = len(g)
    other = _ocg()
    other.method = "PC"
    other.graph_id = "test-ocg-2"
    other.to_rdf(graph=g)

    assert len(g) > n_first
    assert (ocg.iri, RDF.type, CW.OntologicalCausalGraph) in g
    assert (other.iri, RDF.type, CW.OntologicalCausalGraph) in g
    # Node IRIs are global, so the two graphs join on the same node resources.
    assert ocg.node_iri(0) == other.node_iri(0)


def test_to_rdf_keeps_its_workdir_when_asked(tmp_path):
    work = tmp_path / "rdfizer"
    _ocg().to_rdf(workdir=str(work))
    assert (work / "ocg.json").is_file()
    assert (work / "ocg_mapping.rml.ttl").is_file()
    assert (work / "config.ini").is_file()
    assert list((work / "out").iterdir())


def test_to_rdf_refuses_a_literal_the_engine_would_corrupt():
    ocg = _ocg()
    ocg.params = {"note": 'he said "no"'}
    with pytest.raises(ValueError, match="SDM-RDFizer"):
        ocg.to_rdf()


def test_write_bundle_still_joins_methods_on_shared_nodes(tmp_path):
    a, b = _ocg(), _ocg()
    a.graph_id, b.graph_id = "bundle-a", "bundle-b"
    path = tmp_path / "bundle.ttl"
    g = write_bundle({"GES": a, "PC": b}, str(path))

    assert path.is_file()
    assert len(list(g.subjects(RDF.type, CW.OntologicalCausalGraph))) == 2
    assert (a.node_iri(0), RDF.type, CW.PropertyNode) in g
    assert g.value(CW.OntologicalCausalGraph, RDFS.comment) is not None  # vocabulary


# ---------------------------------------------------------------------- #
# The inverse
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("max_hops", [1, 2])
def test_from_rdf_round_trips_the_rml_output(max_hops):
    ocg = _ocg(max_hops=max_hops)
    back = OntologicalCausalGraph.from_rdf(ocg.to_rdf())

    assert back.gid == ocg.gid
    assert back.n == ocg.n
    assert [nd.name for nd in back.nodes] == [nd.name for nd in ocg.nodes]
    assert [nd.kind for nd in back.nodes] == [nd.kind for nd in ocg.nodes]
    assert [nd.range_ for nd in back.nodes] == [nd.range_ for nd in ocg.nodes]
    np.testing.assert_array_equal(back.adj, ocg.adj)
    for (i, j) in ocg.edges():
        assert back.weight(i, j) == pytest.approx(ocg.weight(i, j))
    assert back.method == ocg.method
    assert back.constrained == ocg.constrained
    assert back.source == ocg.source
    assert back.created_at == ocg.created_at
    assert back.params == ocg.params           # kinds survive the trip
    assert back.fingerprint() == ocg.fingerprint()


def test_from_rdf_round_trips_relation_labels():
    """Up to the one normalisation the RDF shape has always imposed.

    A label is either a ``(relation, direction)`` hop or a tuple of them, and
    a *one-element* tuple is written as one step — so ``((r, 'inverse'),)``
    comes back as ``(r, 'inverse')``. That is not new: the hand-written
    exporter took the same single-hop branch on ``len(hops) == 1``, and
    ``max_hops=2`` produces plenty of those wrappers. Comparing flattened hop
    sequences is therefore the honest assertion; comparing the raw label
    objects would only be asserting the wrapper.
    """
    ocg = _ocg(max_hops=2)
    back = OntologicalCausalGraph.from_rdf(ocg.to_rdf())

    hops = _flat_hops

    assert set(back.edge_labels) == set(ocg.edge_labels)
    assert any(len(hops(l)) > 1                      # the fixture really has paths
               for labels in ocg.edge_labels.values() for l in labels)
    for key, labels in ocg.edge_labels.items():
        assert (sorted(map(hops, back.edge_labels[key]))
                == sorted(map(hops, labels)))


def test_from_rdf_still_reads_every_legacy_shape():
    """Turtle written before this vocabulary must keep loading — nothing regenerated it.

    Exercises all six superseded shapes at once: the ``rdf:List`` node order,
    the ``cw:parameters`` JSON literal, the split ``cw:rangeClass`` /
    ``cw:rangeDatatype``, ``cw:domainClass``, ``cw:method`` and
    ``cw:constrained`` duplicated onto the graph, and reified
    ``cw:RelationPath`` / ``cw:RelationHop`` resources.
    """
    ocg = _ocg(max_hops=2)
    g = ocg.to_rdf()

    # --- node order: cw:hasNodeSlot -> rdf:List ------------------------- #
    for slot in list(g.objects(ocg.iri, CW.hasNodeSlot)):
        g.remove((ocg.iri, CW.hasNodeSlot, slot))
        g.remove((slot, None, None))
    node_list = Collection(g, None, [ocg.node_iri(k) for k in range(ocg.n)])
    g.add((ocg.iri, CW.nodeList, node_list.uri))

    # --- parameters: resources -> one JSON literal ----------------------- #
    for param in list(g.objects(ocg.run_iri, CW.hasParameter)):
        g.remove((ocg.run_iri, CW.hasParameter, param))
        g.remove((param, None, None))
    g.add((ocg.run_iri, CW.parameters,
           Literal('{"alpha": 0.05, "constrained": true}')))

    # --- node terms: cw:domain/cw:range -> the old split terms --------- #
    for k in range(ocg.n):
        node_iri = ocg.node_iri(k)
        domain = g.value(node_iri, CW.domain)
        range_ = g.value(node_iri, CW["range"])
        kind = str(g.value(node_iri, CW.nodeKind))
        g.remove((node_iri, CW.domain, None))
        g.remove((node_iri, CW["range"], None))
        g.add((node_iri, CW.domainClass, domain))
        g.add((node_iri, CW.rangeClass if kind == "object" else CW.rangeDatatype,
               range_))

    # --- run provenance: duplicated back onto the graph ------------------ #
    g.remove((ocg.run_iri, CW.method, None))
    g.remove((ocg.run_iri, CW.usedTopologicalConstraint, None))
    g.add((ocg.iri, CW.method, Literal("GES")))
    g.add((ocg.iri, CW.constrained, Literal(True, datatype=XSD.boolean)))

    # --- relation paths: viaPath literal -> reified resources ------------ #
    downgraded = 0
    for edge, text in list(g.subject_objects(CW.viaPath)):
        g.remove((edge, CW.viaPath, text))
        path = URIRef(f"{edge}/path/1")
        g.add((edge, CW.viaPath, path))
        g.add((path, RDF.type, CW.RelationPath))
        for index, (rel, direction) in enumerate(parse_property_path(str(text)), 1):
            hop = URIRef(f"{path}/hop/{index}")
            g.add((path, CW.hop, hop))
            g.add((hop, RDF.type, CW.RelationHop))
            g.add((hop, CW.hopIndex, Literal(index, datatype=XSD.integer)))
            g.add((hop, CW.hopRelation, rel))
            g.add((hop, CW.hopDirection, Literal(direction)))
        downgraded += 1
    assert downgraded, "fixture produced no path to downgrade"

    back = OntologicalCausalGraph.from_rdf(g)
    assert [nd.name for nd in back.nodes] == [nd.name for nd in ocg.nodes]
    assert [nd.range_ for nd in back.nodes] == [nd.range_ for nd in ocg.nodes]
    np.testing.assert_array_equal(back.adj, ocg.adj)
    assert back.params == {"alpha": 0.05, "constrained": True}
    assert back.method == "GES"
    assert back.constrained is True
    # Same single-hop-wrapper normalisation as the current shape imposes; see
    # test_from_rdf_round_trips_relation_labels.
    assert set(back.edge_labels) == set(ocg.edge_labels)
    for key, labels in ocg.edge_labels.items():
        assert (sorted(map(_flat_hops, back.edge_labels[key]))
                == sorted(map(_flat_hops, labels)))


def test_from_rdf_raises_when_neither_node_ordering_is_present():
    ocg = _ocg()
    g = ocg.to_rdf()
    for slot in list(g.objects(ocg.iri, CW.hasNodeSlot)):
        g.remove((ocg.iri, CW.hasNodeSlot, slot))
        g.remove((slot, None, None))
    with pytest.raises(ValueError, match="cw:hasNodeSlot"):
        OntologicalCausalGraph.from_rdf(g)


# ---------------------------------------------------------------------- #
# The pruning
# ---------------------------------------------------------------------- #
def test_removed_helpers_are_gone():
    assert not hasattr(OntologicalCausalGraph, "from_source")
    assert not hasattr(OntologicalCausalGraph, "_rel_iris")
    assert not hasattr(OntologicalCausalGraph, "_add_relation_labels")
