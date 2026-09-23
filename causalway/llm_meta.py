"""Derive LLM ``var_meta`` / ``pair_meta`` / ``domain`` strings from the schema.

The ``rdfs:label`` and ``rdfs:comment`` annotations already present in the TTL
are turned into the natural-language context the GES-Prior LLM prompt needs, so
no hand-written metadata is required.
"""

from __future__ import annotations

from ._util import local_name
from .constraints import EdgeConstraint, EPSILON
from .nodes import PropertyNode
from .ontology import OntologySchema


def _label(schema: OntologySchema, uri) -> str:
    return schema.labels.get(uri, local_name(uri))


def _comment(schema: OntologySchema, uri) -> str:
    return schema.comments.get(uri, "")


def build_var_meta(schema: OntologySchema, nodes: list) -> dict:
    """``{column: "label(p) (attribute of label(D_p)): comment(p)"}``."""
    meta = {}
    for node in nodes:
        p_label = _label(schema, node.prop)
        d_label = _label(schema, node.domain)
        p_comment = _comment(schema, node.prop)
        text = f"{p_label} (attribute of {d_label})"
        if p_comment:
            text += f": {p_comment}"
        meta[node.name] = text
    return meta


def build_pair_meta(schema: OntologySchema, nodes: list,
                    constraint: EdgeConstraint) -> dict:
    """``{(col_i, col_j): relationship description}`` for topologically valid pairs."""
    meta = {}
    names = [n.name for n in nodes]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j or not constraint.allowed[i, j]:
                continue
            Di, Dj = nodes[i].domain, nodes[j].domain
            if Di == Dj:
                text = f"Both are attributes of the same {_label(schema, Di)} entity."
            else:
                rel_texts = []
                for label in constraint.labels.get((i, j), []):
                    rel_texts.append(_render_relation(schema, label))
                text = (
                    f"{_label(schema, Di)} is connected to {_label(schema, Dj)} "
                    f"by the relation {'; '.join(rel_texts)}."
                )
            meta[(names[i], names[j])] = text
    return meta


def _render_relation(schema, label) -> str:
    from .constraints import _is_hop

    if _is_hop(label):
        r, direction = label
        if r == EPSILON:
            return "identity (same entity)"
        base = f"{_label(schema, r)} ({direction})"
        c = _comment(schema, r)
        return f"{base}: {c}" if c else base
    return " → ".join(_render_relation(schema, hop) for hop in label)


def build_domain_str(schema: OntologySchema) -> str:
    """One-line domain description assembled from the class comments."""
    parts = []
    for c in sorted(schema.classes, key=str):
        label = _label(schema, c)
        comment = _comment(schema, c)
        if comment:
            parts.append(f"{label}: {comment}")
        else:
            parts.append(label)
    return " ".join(parts) if parts else "general domain"
