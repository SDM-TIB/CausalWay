"""Shared small helpers for the ``causalway`` package."""

from __future__ import annotations

import re
from typing import Union

from rdflib import URIRef

# RDF/RDFS/OWL and the RML/R2RML mapping vocabulary left over in these TTL files.
_VOCAB_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/ns/r2rml#",
    "http://semweb.mmlab.be/ns/rml#",
    "http://semweb.mmlab.be/ns/ql#",
)


def local_name(uri: Union[str, URIRef]) -> str:
    """Return the local part of a URI (after the last ``#`` or ``/``)."""
    text = str(uri)
    for sep in ("#", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[1]
    return text or str(uri)


def is_vocab(uri: Union[str, URIRef]) -> bool:
    """True for RDF/RDFS/OWL and RML mapping-vocabulary terms to be filtered out."""
    return str(uri).startswith(_VOCAB_NAMESPACES)


def natural_key(text: str):
    """Sort key that orders embedded integers numerically (patient_2 < patient_10)."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]
