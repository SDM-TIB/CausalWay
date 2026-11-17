"""A fitted causal model → JSON → RDF, through SDM-RDFizer.

Why a model is in the knowledge graph at all
--------------------------------------------
A query is posed on a *graph*; its answers are evaluated by a *model*
(``cw:evaluatedBy``).  Without a description of that model an answer cannot
be traced: two models fitted on one graph — different seeds, different data,
an SCM and a CBN — answer the same query differently.  So a model is exported
as a ``cw:CausalModel`` that names its graph, the ``cw:ModelFittingRun`` that
produced it (source KG, library, time, seed, training rows), the mechanism of
every node, and the sha256 of its saved artifact.

The fitted parameters themselves — sklearn estimators inside ``scm.pkl`` — are
deliberately *not* in RDF.  Tracing needs to identify the model; reproducing
an answer needs the artifact, and ``cw:artifactChecksum`` is what ties the two
together.

Of everything a mechanism could say, one fact is kept besides its class:
``cw:invertible``, because whether a node's noise can be abducted is whether a
counterfactual about it is defined.  Held-out performance is an evaluation
result, not a property of the model's identity, and is not exported here.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from rdflib import Graph, RDFS, XSD

from .rdfizer import check_quote_free, materialise
from .result import parameter_rows
from .vocab import CW, GRAPH_STEM, MODEL_RUN_STEM, MODEL_STEM, vocabulary

__all__ = ["model_to_json", "model_to_rdf", "MODEL_MAPPING_PATH", "model_kind"]

#: The RML mapping that defines the JSON -> RDF contract for a fitted model.
MODEL_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_mapping.rml.ttl"
)

#: JSON fields carrying an IRI, exempt from the engine's quote rewriting.
_IRI_FIELDS = {"graph_iri", "run_iri", "parameter_iri", "node_iri", "mechanism_iri",
               "datatype"}

#: ``cw:byMethod`` of the fitting run, per model kind.
_FIT_METHOD = {"scm": "dowhy.gcm", "cbn": "pgmpy"}


def model_kind(model) -> str:
    """``"scm"`` or ``"cbn"`` — recorded in the manifest by whoever fitted the model."""
    return (getattr(model, "manifest", None) or {}).get("model_kind", "scm")


def _graph_iri(model) -> Optional[str]:
    ocg = getattr(getattr(model, "spec", None), "ocg", None)
    if ocg is not None:
        return str(ocg.iri)
    gid = (model.manifest or {}).get("ocg_gid")
    return f"{GRAPH_STEM}{gid}" if gid else None


def model_to_json(model) -> dict:
    """Render a fitted model as the flat document ``model_mapping.rml.ttl`` reads.

    Denormalised for the same reason every other export here is: SDM-RDFizer
    4.7.5 mishandles ``rr:parentTriplesMap`` onto a blank-node parent, so each
    record carries every field its own maps need.
    """
    manifest = model.manifest or {}
    model_key = model.model_id or "unfitted"
    kind = model_kind(model)
    run_iri = f"{MODEL_RUN_STEM}{model_key}"

    row = {"model_key": model_key, "run_iri": run_iri}
    graph_iri = _graph_iri(model)
    if graph_iri:
        row["graph_iri"] = graph_iri
    if manifest.get("artifact_sha256"):
        row["checksum"] = str(manifest["artifact_sha256"])

    source = manifest.get("source_kg") or {}
    versions = manifest.get("versions", {}) or {}
    run_row = {
        "run_key": model_key,
        "method": _FIT_METHOD.get(kind, kind),
        # A literal, not an IRI: the source may be a filesystem path, and the
        # export says where the data came from rather than minting a resource.
        "source": str(source.get("path") or source.get("url") or "unknown"),
        "fitted_time": str(manifest.get("fitted_at")
                           or datetime.now(timezone.utc).isoformat(timespec="seconds")),
    }

    params = {
        "seed": manifest.get("random_state"),
        "trainingRows": int(manifest.get("training_rows", 0) or 0),
        "libraryVersion": versions.get("pgmpy" if kind == "cbn" else "dowhy"),
    }
    if kind == "scm":
        params["quality"] = manifest.get("quality")
        params["categoricalCounterfactual"] = manifest.get("categorical_counterfactual")
    plain, typed, links = parameter_rows(model_key, run_iri, params)

    mechanisms = []
    ocg = getattr(getattr(model, "spec", None), "ocg", None)
    if kind == "scm" and ocg is not None and manifest.get("mechanisms_fitted", True):
        index = {node.name: k for k, node in enumerate(ocg.nodes)}
        for _, mech in model.mechanism_table().iterrows():
            name = str(mech["node"])
            if name not in index:
                continue
            node_key = ocg.node_key(index[name])
            mechanisms.append({
                "model_key": model_key,
                "node_key": node_key,
                "node_iri": str(ocg.node_iri(index[name])),
                "mechanism_iri": f"{MODEL_STEM}{model_key}/mechanism/{node_key}",
                "mechanism_type": str(mech["mechanism_type"]),
                "invertible": "true" if bool(mech["invertible"]) else "false",
            })

    return {
        "models": [row],
        "runs": [run_row],
        "parameters": plain,
        "parameters_typed": typed,
        "run_parameters": links,
        "mechanisms": mechanisms,
    }


def model_to_rdf(model, *, graph: Optional[Graph] = None,
                 with_vocabulary: bool = False,
                 workdir: Optional[str] = None) -> Graph:
    """Serialise a fitted model to RDF through SDM-RDFizer.

    No triple is written by this function: :func:`model_to_json` produces the
    document and ``model_mapping.rml.ttl`` produces the triples.
    """
    g = Graph() if graph is None else graph
    g.bind("cw", CW)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    if with_vocabulary:
        vocabulary(g)

    payload = model_to_json(model)
    check_quote_free(payload, _IRI_FIELDS)
    materialise(payload, MODEL_MAPPING_PATH, source_name="model.json",
                graph=g, workdir=workdir)
    return g
