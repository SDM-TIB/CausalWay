"""``causalway`` — bridge from a knowledge graph to causal discovery, model fitting and prediction."""

from .ontology import OntologySchema
from .nodes import PropertyNode, build_nodes
from .constraints import EdgeConstraint, EPSILON
from .bgp import (
    ColumnSpec,
    DisconnectedJoinError,
    Materialization,
    join_role_options,
    materialize,
    object_property_role,
)
from .result import OntologicalCausalGraph
from .sources import resolve_graph, resolve_schema, load_ocg, list_ocgs

__all__ = [
    "OntologySchema",
    "PropertyNode",
    "build_nodes",
    "EdgeConstraint",
    "EPSILON",
    "ColumnSpec",
    "DisconnectedJoinError",
    "Materialization",
    "materialize",
    "join_role_options",
    "object_property_role",
    "OntologicalCausalGraph",
    "resolve_graph",
    "resolve_schema",
    "load_ocg",
    "list_ocgs",
]

# Plan 2 modules (causalway.model, .mechanisms, .entities, .inference, .queries,
# .evaluation) depend on dowhy.gcm / pgmpy and are intentionally *not*
# imported here — import them explicitly, e.g. ``from causalway.model import
# CausalModel``, so plain Plan 1 discovery use stays free of that dependency.
