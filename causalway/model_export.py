"""A fitted causal model → JSON → RDF, through SDM-RDFizer.

Why this module exists
----------------------
``CausalModel.to_rdf`` built the Layer A triples by hand, which left it as the
last export in the project whose shape lived in Python.  It also still emitted
``cw:parameters`` — a *legacy* term (see ``vocab.py``), holding a JSON literal,
retired everywhere else precisely because SDM-RDFizer rewrites both quote
characters through ``rml:reference`` and a JSON object is all quotes.  Moving
this export onto a mapping fixes both at once: the fit run's hyper-parameters
become ``cw:Parameter`` resources with a name, a value and a kind, exactly as
a discovery run's already are.

No new vocabulary is minted for that.  ``cw:hasParameter`` / ``cw:Parameter``
were described for ``cw:DiscoveryRun``; a ``cw:ModelFitRun`` is the same sort of
activity and reuses them, which is what the subtractive rule in
``CLAUDE.md`` asks for.
"""

from __future__ import annotations

import os
from typing import Optional

from rdflib import Graph

from .rdfizer import check_quote_free, materialise
from .result import _param_entry, _slug
from .vocab import (
    CW,
    GRAPH_STEM,
    MODEL_RUN_STEM,
    MODEL_STEM,
    PROV,
    vocabulary,
)

__all__ = ["model_to_json", "model_to_rdf", "MODEL_MAPPING_PATH"]

#: The RML mapping that defines the JSON -> RDF contract for a fitted model.
MODEL_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_mapping.rml.ttl"
)

#: JSON fields carrying an IRI, exempt from the engine's quote rewriting.
_IRI_FIELDS = {"model_iri", "run_iri", "graph_iri"}


def model_to_json(model) -> dict:
    """Render a fitted model as the flat document ``model_mapping.rml.ttl`` reads.

    Denormalised for the same reason every other export here is: SDM-RDFizer
    4.7.5 mishandles ``rr:parentTriplesMap`` onto a blank-node parent, so each
    record carries every field its own maps need.
    """
    manifest = model.manifest or {}
    model_key = model.model_id or "unfitted"

    row = {
        "model_key": model_key,
        "model_iri": str(model.iri),
        "run_iri": str(model.run_iri),
        "label": f"Causal model {model_key}",
        "library": "dowhy.gcm",
        "library_version": manifest.get("versions", {}).get("dowhy", "unknown"),
        "training_rows": str(int(manifest.get("training_rows", 0) or 0)),
    }
    gid = manifest.get("ocg_gid")
    if gid:
        row["graph_iri"] = f"{GRAPH_STEM}{gid}"
    source = manifest.get("source_kg") or {}
    # A literal, not an IRI: the source may be a filesystem path, and the
    # export says where the data came from rather than minting a resource for it.
    trained_on = source.get("path") or source.get("url")
    if trained_on:
        row["trained_on"] = str(trained_on)

    run_row = {"run_key": model_key}
    fitted_at = manifest.get("fitted_at")
    if fitted_at:
        run_row["ended_at"] = str(fitted_at)

    parameters = []
    for name in ("quality", "categorical_counterfactual", "random_state"):
        kind, text = _param_entry(name, manifest.get(name))
        record = {
            "run_key": model_key,
            "param_key": _slug(name),
            "name": name,
            "kind": kind,
        }
        # An *empty* reference is not an absent one to SDM-RDFizer — it writes
        # the literal "None" — so a null parameter carries no cw:parameterValue
        # at all and cw:parameterKind alone tells the two apart on the way back.
        if text != "":
            record["value"] = text
        parameters.append(record)

    mechanisms = []
    for _, mech in model.mechanism_table().iterrows():
        record = {
            "model_key": model_key,
            "node_slug": _slug(str(mech["node"])),
            "mechanism_type": str(mech["mechanism_type"]),
            "invertible": "true" if bool(mech["invertible"]) else "false",
        }
        if str(mech["mechanism_type"]) == "InvertibleClassifierFCM":
            record["coupling"] = "gumbel-max"
        mechanisms.append(record)

    return {
        "models": [row],
        "runs": [run_row],
        "run_parameters": parameters,
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
    g.bind("prov", PROV)
    if with_vocabulary:
        vocabulary(g)

    payload = model_to_json(model)
    check_quote_free(payload, _IRI_FIELDS)
    materialise(payload, MODEL_MAPPING_PATH, source_name="model.json",
                graph=g, workdir=workdir)
    return g
