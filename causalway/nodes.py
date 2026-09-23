"""Property nodes: the causal nodes ``(D_p, p, R_p)`` of the ontological causal graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rdflib import URIRef

from ._util import local_name
from .ontology import OntologySchema


@dataclass(frozen=True)
class PropertyNode:
    """A single causal node ``(D_p, p, R_p)``.

    ``domain`` is the class owning the property, ``prop`` the property URI,
    ``range_`` the range class (object property) or datatype (data property),
    and ``kind`` distinguishes the two.
    """

    domain: URIRef
    prop: URIRef
    range_: URIRef
    kind: Literal["data", "object"]

    @property
    def name(self) -> str:
        """DataFrame column name, e.g. ``"SCLCPatient.ageGroup"``."""
        return f"{local_name(self.domain)}.{local_name(self.prop)}"

    @property
    def var(self) -> str:
        """Name of the BGP variable bound to the entity of type ``domain``."""
        return local_name(self.domain)


def build_nodes(schema: OntologySchema,
                include_object_properties: bool = True) -> list[PropertyNode]:
    """Build the ordered, collision-free list of property nodes.

    A property declared with *k* domains/ranges yields *k* nodes.  Names are
    made unique by suffixing ``_2``, ``_3``, ... on collision, and the final
    list is stably sorted by ``(domain, prop, range)``.
    """
    nodes: list[PropertyNode] = []

    for p, drs in schema.data_properties.items():
        for (d, datatype) in drs:
            nodes.append(PropertyNode(domain=d, prop=p, range_=datatype, kind="data"))

    if include_object_properties:
        for p, drs in schema.object_properties.items():
            for (d, r) in drs:
                nodes.append(PropertyNode(domain=d, prop=p, range_=r, kind="object"))

    # Stable sort so every matrix in the pipeline shares one node order.
    nodes.sort(key=lambda n: (str(n.domain), str(n.prop), str(n.range_)))

    # Uniquify names on collision.
    counts: dict = {}
    unique: list[PropertyNode] = []
    for n in nodes:
        base = n.name
        if base not in counts:
            counts[base] = 0
            unique.append(n)
        else:
            counts[base] += 1
            unique.append(_with_name(n, f"{base}_{counts[base] + 1}"))
    return unique


@dataclass(frozen=True)
class _NamedPropertyNode(PropertyNode):
    _name: str = ""

    @property
    def name(self) -> str:
        return self._name


def _with_name(node: PropertyNode, name: str) -> PropertyNode:
    """Return a shallow clone carrying a forced column name."""
    return _NamedPropertyNode(
        domain=node.domain, prop=node.prop, range_=node.range_,
        kind=node.kind, _name=name,
    )
