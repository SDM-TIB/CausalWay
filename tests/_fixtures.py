"""Shared test fixtures: a small causal graph over the clinic KG and a stand-in model.

The stand-in duck-types exactly the part of :class:`causalway.model.CausalModel`
the RDF exports read — ``spec.ocg``, ``model_id``, ``iri``, ``manifest`` and
``mechanism_table()`` — so query, counterfactual and SHACL tests run in
seconds instead of fitting a dowhy model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rdflib import URIRef

from causalway.constraints import EdgeConstraint
from causalway.nodes import build_nodes
from causalway.ontology import OntologySchema
from causalway.result import OntologicalCausalGraph

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC = os.path.join(HERE, "kgs", "ttls", "synthetic_clinic.ttl")
KG = "http://causalkg.example.org/synthetic/"
PATIENT_0 = KG + "patient_0"

#: smoking -> tumorStage -> survival <- careQuality ; toxicity -> survival
EDGES = [("Patient.smoking", "Patient.tumorStage"),
         ("Patient.tumorStage", "Patient.survival"),
         ("Hospital.careQuality", "Patient.survival"),
         ("Therapy.toxicity", "Patient.survival")]


def clinic_ocg(graph_id: str = "clinic-test", max_hops: int = 1,
               edges=EDGES) -> OntologicalCausalGraph:
    schema = OntologySchema.from_source(CLINIC)
    nodes = build_nodes(schema, include_object_properties=True)
    constraint = EdgeConstraint.from_schema(schema, nodes, max_hops=max_hops)
    index = {n.name: i for i, n in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=int)
    weights = np.zeros(adj.shape, dtype=float)
    for k, (a, b) in enumerate(edges):
        adj[index[a], index[b]] = 1
        weights[index[a], index[b]] = 0.1 * (k + 1)
    return OntologicalCausalGraph.from_discovery(
        adj, constraint, weights=weights, method="GES", params={"seed": 1},
        constrained=True, source=CLINIC, graph_id=graph_id)


class FakeSpec:
    def __init__(self, ocg, mat=None):
        self.ocg = ocg
        self.mat = mat


class FakeModel:
    """Enough of a CausalModel for every RDF export and for ``validate_query``."""

    def __init__(self, ocg, model_id: str = "m1", dtypes=None, kind: str = "scm",
                 invertible: bool = True, checksum: str = "0" * 64):
        self.spec = FakeSpec(ocg)
        dtypes = dtypes or {n.name: "categorical" for n in ocg.nodes}
        self.manifest = {
            "model_id": model_id, "model_kind": kind, "dtypes": dtypes,
            "ocg_gid": ocg.gid, "random_state": 7, "training_rows": 1240,
            "versions": {"dowhy": "0.12", "pgmpy": "1.0"}, "quality": "good",
            "categorical_counterfactual": "gumbel-max",
            "fitted_at": "2026-09-23T10:00:00+00:00",
            "source_kg": {"path": CLINIC}, "artifact_sha256": checksum,
        }
        self._invertible = invertible

    @property
    def model_id(self):
        return self.manifest["model_id"]

    @property
    def iri(self):
        return URIRef(f"http://sdm-causalway.org/model/{self.model_id}")

    def mechanism_table(self):
        return pd.DataFrame([{"node": n.name, "mechanism_type": "InvertibleClassifierFCM",
                              "invertible": self._invertible} for n in self.spec.ocg.nodes])

    def to_rdf(self, graph=None, *, with_vocabulary=False, workdir=None):
        from causalway.model_export import model_to_rdf
        return model_to_rdf(self, graph=graph, with_vocabulary=with_vocabulary,
                            workdir=workdir)
