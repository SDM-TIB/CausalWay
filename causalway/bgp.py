"""BGP construction and SPARQL materialisation (the "flat join", decision D1).

One SPARQL basic graph pattern per connected component of the class-level
schema graph.  Each BGP solution is one row; 1:N relations expand into multiple
rows.  Every column carries provenance back to its causal node ``(D_p, p, R_p)``
and to the BGP variable holding the entity of type ``D_p``.

An object property can play one of two roles, and **only one at a time**: it
is either a *relationship* (a join edge, letting its range class's own
properties ride into the same row) or a *causal variable* (a column whose
value is the identity of the related entity). The two roles are mutually
exclusive, and the default is *relationship*:

- ``excluded_joins`` (object properties only) decides whether the
  relationship is joined. This is what the ontology canvas's edge
  represents; excluding one drops that join from the query, and the class it
  used to connect only stays in the query if some other retained node still
  needs it (closes G1).
- ``excluded`` hides a *column*. For a data property this never touches the
  query's shape — its required pattern keeps filtering rows whether or not it
  is excluded, so which data properties are candidate variables never
  silently changes the row count.
- For an object property the two combine into one of three roles, resolved by
  :func:`object_property_role`: joined => ``"relationship"``; not joined but
  retained => ``"variable"``; neither => ``"dropped"``.

An object property used to be allowed to be *both* at once, and that was a
mistake rather than a feature. As a join it says "these two entities are the
same row"; as a variable it says "which entity this points at is a cause".
Holding both at once puts a column in the frame whose value is, by
construction, a deterministic function of the join that produced the row —
the discovery algorithms then see a variable that perfectly predicts (and is
perfectly predicted by) every property of the range class, which is an
artefact of the materialisation, not a causal fact about the KG. Worse, the
old both-roles column read its value off the *joined* class variable, so
excluding the join silently changed what the column meant. Forcing the
choice removes both problems, and the value-only branch reads the object off
a private pattern that never activates the range class, so choosing
``"variable"`` really does keep the range class's other properties out.

Choosing ``"variable"`` therefore *drops a join*, which can disconnect the
active classes. That is refused rather than absorbed — see
:func:`join_role_options`, which module 1 calls to disable the choice before
the user makes it, and ``DisconnectedJoinError`` for the same refusal at
materialisation time.

``include_object_properties`` / ``object_value`` / ``key_properties``
(Plan 1 D2) is the older, independent, *global* decision of whether every
retained object property becomes a value column, and how — module 3 still
uses this path when it re-materialises from source (Plan 2 §3.3). It is a
different code path with different semantics (the value is the range *class*
or a key property, not the entity), it predates the role split, and the
mutual-exclusivity rule above deliberately does not apply to it: the web
service passes ``include_object_properties=False`` to get the per-node role
behaviour instead.

Object-valued columns carry the **full IRI** of the related entity, not its
local name. Two entities in different namespaces can share a local name, and
``causalway.cf_export`` needs the IRI to emit the value as an ``rr:IRI`` term
rather than a string. Readability is the front end's job (``shortIri()``),
not the materialiser's — shortening here destroyed information that could
not be recovered downstream.

Query construction (assembling the SPARQL text from curated nodes) is kept
separate from query execution (``schema.graph.query(...)``), so a caller that
only wants to show the pattern — the Preprocess module's live "Graph Pattern"
preview — can get it via :func:`build_query` without paying for the join
itself (Plan 3 §Preprocess revision 3).
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import networkx as nx
import pandas as pd
from rdflib import Literal, URIRef

from ._util import local_name
from .nodes import PropertyNode
from .ontology import OntologySchema


def _sparql_var(name: str) -> str:
    """Sanitise a name into a legal SPARQL variable name."""
    v = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not v or v[0].isdigit():
        v = "v_" + v
    return v


def _to_py(o):
    if isinstance(o, Literal):
        return o.toPython()
    if isinstance(o, URIRef):
        return str(o)
    return str(o)


class DisconnectedJoinError(ValueError):
    """Raised instead of running an unlimited query over a curation-induced cross
    product (see :func:`_warn_disconnected`)."""


#: The three roles an object property can hold. See the module docstring.
OBJECT_ROLES = ("relationship", "variable", "dropped")


def object_property_role(name: str, excluded: set, excluded_joins: set) -> str:
    """Resolve one object property's role from the two curation axes.

    The join wins when both axes say "keep", which is what makes
    *relationship* the default: an uncurated project has both sets empty, so
    every object property starts as a relationship and none of them start as
    a variable. A caller that wants the variable role has to say so by
    dropping the join — that is the whole point of the exclusivity, and it is
    resolved in exactly one place so the service, the front end and the
    materialiser cannot drift apart on it.
    """
    if name not in excluded_joins:
        return "relationship"
    if name not in excluded:
        return "variable"
    return "dropped"


@dataclass
class ColumnSpec:
    """Provenance of one DataFrame column back to a causal node."""
    name: str            # DataFrame column name
    node: PropertyNode   # the causal node (D_p, p, R_p) this column realises
    subject_var: str     # BGP variable bound to the entity of type D_p


@dataclass
class Materialization:
    """Rows = BGP solutions, columns = ``ColumnSpec.name``.

    ``entity_ids`` holds one column per class variable, aligned by row index,
    so the owning entity of any cell is ``entity_ids.loc[row, subject_var]``.
    """
    df: pd.DataFrame
    columns: list
    entity_ids: pd.DataFrame
    classes: list
    query: str
    multiplicity: dict = field(default_factory=dict)
    skipped_patterns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def shape(self):
        return self.df.shape

    def _colspec(self, column_name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == column_name:
                return c
        raise KeyError(f"Unknown column '{column_name}'")

    def owner(self, column_name: str, row: int) -> URIRef:
        """The entity (of type ``D_p``) that owns cell ``(column_name, row)``."""
        spec = self._colspec(column_name)
        return URIRef(self.entity_ids.loc[row, spec.subject_var])

    def attribute_predictions(self, preds: pd.Series, column_name: str) -> dict:
        """Map an inferred column back onto entities: ``{entity_IRI: value}``."""
        spec = self._colspec(column_name)
        out = {}
        for row in preds.index:
            out[URIRef(self.entity_ids.loc[row, spec.subject_var])] = preds.loc[row]
        return out


def _component_map(schema: OntologySchema) -> dict:
    mapping = {}
    ug = schema.class_graph.to_undirected()
    for idx, comp in enumerate(nx.connected_components(ug)):
        for c in comp:
            mapping[c] = idx
    return mapping


def _unique_vars(classes) -> dict:
    """Assign a unique SPARQL variable per class, sanitising local names."""
    out = {}
    used = set()
    for c in sorted(classes, key=str):
        base = _sparql_var(local_name(c))
        name = base
        k = 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        out[c] = name
    return out


@dataclass
class _Pattern:
    """The assembled query text plus everything the execution/post-processing
    steps need, without having run anything yet."""
    query: str
    var_to_col: dict
    warnings: list
    skipped: list
    active_classes: list
    class_vars: dict
    join_active: dict
    object_roles: dict
    object_value_columns: list
    n_disconnected_groups: int = 1


def materialize(
    schema: OntologySchema,
    nodes: list,
    excluded: Optional[set] = None,
    excluded_joins: Optional[set] = None,
    include_object_properties: bool = False,
    optional_data_properties: bool = False,
    object_value: str = "range_class",
    key_properties: Optional[dict] = None,
    limit: Optional[int] = None,
) -> Union["Materialization", list]:
    """Materialise the flat join for ``nodes``.

    ``nodes`` should carry every node of interest, retained *and* excluded —
    exclusion is passed separately via ``excluded`` (a set of
    ``PropertyNode.name``) so the join can tell "not a candidate variable"
    apart from "not part of the graph at all".

    ``excluded_joins`` is the independent object-property-only axis (see the
    module docstring). When not given it defaults to ``excluded`` itself,
    which reproduces the old, single-axis behaviour for callers that only
    know about ``excluded`` (module 3, the runners) — excluding an object
    node there drops both its value and its join together, exactly as
    before.

    Returns a single :class:`Materialization` when all nodes share one
    connected component (the normal case), otherwise a list of one
    :class:`Materialization` per component.
    """
    excluded, excluded_joins, key_properties = _defaults(excluded, excluded_joins, key_properties)

    mats = []
    for comp_idx, comp_nodes, classes in _grouped_by_component(schema, nodes):
        mats.append(_materialize_component(
            schema, comp_nodes, classes, excluded, excluded_joins,
            include_object_properties, optional_data_properties, object_value,
            key_properties, limit,
        ))

    if len(mats) == 1:
        return mats[0]
    return mats


def build_query(
    schema: OntologySchema,
    nodes: list,
    excluded: Optional[set] = None,
    excluded_joins: Optional[set] = None,
    include_object_properties: bool = False,
    optional_data_properties: bool = False,
    object_value: str = "range_class",
    key_properties: Optional[dict] = None,
    limit: Optional[int] = None,
) -> tuple[str, list]:
    """Build the SPARQL query text for ``nodes`` without executing it.

    Same curation-to-pattern construction as :func:`materialize`, minus the
    query execution and DataFrame assembly — cheap enough to call on every
    curation change so a live "Graph Pattern" preview can stay in sync with
    the query :func:`materialize` would actually run, without paying for the
    join itself. Returns ``(query_text, warnings)``; one component's nodes in
    produce one query, but a mixed set (multiple components) concatenates
    each component's query, since the preview shows "what will run", not a
    per-component breakdown.
    """
    excluded, excluded_joins, key_properties = _defaults(excluded, excluded_joins, key_properties)

    queries = []
    warnings: list = []
    for comp_idx, comp_nodes, classes in _grouped_by_component(schema, nodes):
        pattern = _build_pattern(
            schema, comp_nodes, classes, excluded, excluded_joins,
            include_object_properties, optional_data_properties, object_value,
            key_properties, limit,
        )
        queries.append(pattern.query)
        warnings.extend(pattern.warnings)
    return "\n\n".join(queries), warnings


def join_role_options(
    schema: OntologySchema,
    nodes: list,
    excluded: Optional[set] = None,
    excluded_joins: Optional[set] = None,
) -> dict:
    """Per object property: its current role, and whether each role is available.

    Choosing ``"variable"`` drops a join, and a dropped join can leave the
    active classes in two groups with nothing connecting them — at which
    point SPARQL returns their cross product rather than a join. That is
    refused at materialisation time (:class:`DisconnectedJoinError`), but a
    refusal that only arrives after the user has committed a curation and
    pressed Materialise is a bad way to learn it: module 1 calls this instead
    and disables the choice up front, with the reason attached.

    Availability is decided by *actually building the pattern* each role would
    produce and counting its connected groups, not by reasoning about the join
    graph directly. The two can differ: dropping a join may also deactivate
    the range class entirely (when no other retained node needs it), and a
    query that lost a class is still perfectly connected. Query construction
    is pure string assembly over the schema — no KG access — so the extra
    passes cost nothing worth optimising.

    The comparison is *relative*: a role is unavailable only when it leaves
    strictly more disconnected groups than the other role would. An absolute
    "is this disconnected?" test would blame every object property in the
    project for one unrelated dropped join, and the user would have no way to
    tell which choice was actually the problem.

    The same test is applied whatever the node's current role is, so a node
    already sitting at ``"variable"`` reports ``can_be_variable=False`` when
    that is what split the query — which is how module 1 flags a conflict that
    already exists rather than only preventing a new one.

    Returns ``{node_name: {"role", "can_be_relationship", "can_be_variable",
    "reason"}}``; ``reason`` is non-``None`` only when a role is unavailable.
    """
    excluded, excluded_joins, _ = _defaults(excluded, excluded_joins, None)

    out: dict = {}
    for node in nodes:
        if node.kind != "object":
            continue
        as_variable = _disconnected_groups(
            schema, nodes,
            excluded=excluded - {node.name},
            excluded_joins=excluded_joins | {node.name},
        )
        as_relationship = _disconnected_groups(
            schema, nodes,
            excluded=excluded | {node.name},
            excluded_joins=excluded_joins - {node.name},
        )
        can_var = as_variable <= as_relationship
        out[node.name] = {
            "role": object_property_role(node.name, excluded, excluded_joins),
            "can_be_relationship": True,
            "can_be_variable": can_var,
            "reason": None if can_var else (
                f"{local_name(node.prop)} is the only relationship holding "
                f"{local_name(node.domain)} and {local_name(node.range_)} in the "
                "same row. As a causal variable it stops joining them, and the query "
                "computes their cross product instead of a join. Exclude "
                f"{local_name(node.range_)}'s own nodes first, or keep this a "
                "relationship."
            ),
        }
    return out


def _disconnected_groups(schema, nodes, excluded, excluded_joins) -> int:
    """How many unjoined groups the active classes would fall into, summed over
    schema components (each component contributes at least 1, so a schema that
    is disconnected to begin with is not mistaken for a curation error)."""
    total = 0
    for _, comp_nodes, classes in _grouped_by_component(schema, nodes):
        pattern = _build_pattern(
            schema, comp_nodes, classes, excluded, excluded_joins,
            False, False, "range_class", {}, None,
        )
        total += pattern.n_disconnected_groups - 1
    return total + 1


def _defaults(excluded, excluded_joins, key_properties):
    if excluded is None:
        excluded = set()
    if excluded_joins is None:
        excluded_joins = set(excluded)
    if key_properties is None:
        key_properties = {}
    return excluded, excluded_joins, key_properties


def _grouped_by_component(schema: OntologySchema, nodes: list):
    """Yield ``(comp_idx, comp_nodes, classes)`` per connected component touched
    by ``nodes``, in the same grouping :func:`materialize` and :func:`build_query`
    both rely on."""
    comp_of_class = _component_map(schema)
    groups: dict = defaultdict(list)
    for n in nodes:
        groups[comp_of_class[n.domain]].append(n)
    for comp_idx, comp_nodes in sorted(groups.items()):
        classes = sorted(
            (c for c in schema.classes if comp_of_class.get(c) == comp_idx),
            key=str,
        )
        yield comp_idx, comp_nodes, classes


def _build_pattern(
    schema: OntologySchema,
    comp_nodes: list,
    classes: list,
    excluded: set,
    excluded_joins: set,
    include_object_properties: bool,
    optional_data_properties: bool,
    object_value: str,
    key_properties: dict,
    limit: Optional[int],
) -> _Pattern:
    # A class only enters the join if curation kept at least one node that
    # needs it. A data property's domain needs it if the property is a
    # candidate variable. An object property's domain needs it if the
    # property is a candidate variable *or* its relationship is joined —
    # either one binds the domain's own entity variable; its range needs it
    # only when the relationship is actually joined, since a variable-only
    # object property reads its value off a private pattern that never
    # touches the range class.
    active = set()
    for n in comp_nodes:
        var_kept = n.name not in excluded
        if n.kind == "object":
            join_kept = n.name not in excluded_joins
            if var_kept or join_kept:
                active.add(n.domain)
            if join_kept:
                active.add(n.range_)
        elif var_kept:
            active.add(n.domain)

    active_classes = [c for c in classes if c in active]
    class_vars = _unique_vars(active_classes)
    var_to_col: dict = {}
    skipped: list = []
    warnings: list = []

    # Patterns are no longer accumulated into one flat list. rdflib's BGP
    # join executor (`sparql/algebra.py:reorderTriples`) sorts triples by how
    # many terms are already bound, evaluated once per query — every
    # `?X a <Class>` triple has 2 bound terms (the predicate and the class
    # IRI) versus 1 for a join or data-property triple, so *all* of them are
    # scheduled as the first block and run before the joins that actually
    # connect the classes ever apply. For N active classes with n1..nN
    # instances each, that block alone materialises their full cross product
    # (empirically confirmed: the 500/1240/20-instance default join on
    # `synthetic_clinic` — already fully connected, not the curation-induced
    # disconnection `DisconnectedJoinError` guards against — took ~60s this
    # way, all of it inside `evalBGP`, evaluating almost exactly
    # 500*1240*20 = 12.4M intermediate bindings before the join patterns ever
    # filtered them down to the true 1240-row result). Listing the type
    # triple before or after its class's join in the flat text makes no
    # difference — reordering happens globally regardless of input order.
    #
    # The fix is structural, not textual: nest each non-root class's type
    # triple and properties inside a `{ }` group scoped under the join
    # pattern that binds it (a spanning tree over the join graph, rooted
    # arbitrarily per connected component). A nested group is a separate
    # algebra node, so `reorderTriples` only ever sees one class's join
    # immediately followed by that same class's own type/property triples —
    # nothing to hoist ahead of. This dropped the same query to ~3-4s. See
    # `_render_class_scope` below.
    type_triple: dict = {}
    class_patterns: dict = defaultdict(list)

    # 1. Type assertions — exactly one variable per active class, scoped to
    # each class rather than appended to a shared flat list (see above).
    for c in active_classes:
        type_triple[c] = f"?{class_vars[c]} a <{c}> ."
        var_to_col[class_vars[c]] = class_vars[c]

    # 2. Object-property JOIN patterns — one per node whose relationship edge
    # is kept, independent of whether the node is also a candidate variable
    # (that is curated separately, in the node panel; §Preprocess revision).
    join_edges: list = []   # (domain, range, triple_text), in encounter order
    join_active: dict = {}
    for n in comp_nodes:
        if n.kind != "object":
            continue
        kept = n.name not in excluded_joins
        join_active[n.name] = False
        if not kept:
            continue
        d, r = n.domain, n.range_
        if d not in class_vars or r not in class_vars:
            continue
        if d == r:
            skipped.append(f"reflexive object property {local_name(n.prop)}")
            continue
        join_edges.append((d, r, f"?{class_vars[d]} <{n.prop}> ?{class_vars[r]} ."))
        join_active[n.name] = True

    joined_pairs = [(d, r) for d, r, _ in join_edges]
    _warn_parallel(joined_pairs, warnings)
    n_disconnected_groups = _warn_disconnected(active_classes, joined_pairs, warnings)

    # 3. Data-property patterns. Every data node of an active class keeps its
    # pattern regardless of curation, so which properties are candidate
    # variables never changes the row population; only a *retained* node's
    # variable is projected into an output column.
    for node in comp_nodes:
        if node.kind != "data" or node.domain not in class_vars:
            continue
        dv = class_vars[node.domain]
        pv = f"{dv}__{_sparql_var(local_name(node.prop))}"
        if optional_data_properties:
            class_patterns[node.domain].append(f"OPTIONAL {{ ?{dv} <{node.prop}> ?{pv} . }}")
        else:
            class_patterns[node.domain].append(f"?{dv} <{node.prop}> ?{pv} .")
        if node.name not in excluded:
            var_to_col[pv] = node.name

    # 3b. Object-property VALUE columns — only for a node whose role resolved
    # to "variable", i.e. one whose join the caller dropped on purpose (see
    # the module docstring on why the roles are exclusive). The value is read
    # off a *private* pattern rather than off the joined class variable: that
    # binds just enough to read the related entity's IRI without activating
    # the range class, so its own properties cannot ride along through this
    # relation, and the column's meaning no longer depends on a join that is
    # not there.
    object_roles: dict = {}
    object_value_columns: list = []
    if not include_object_properties:
        for node in comp_nodes:
            if node.kind != "object":
                continue
            role = object_property_role(node.name, excluded, excluded_joins)
            object_roles[node.name] = role
            if role != "variable" or node.domain not in class_vars:
                continue
            dv = class_vars[node.domain]
            pv = f"{dv}__{_sparql_var(local_name(node.prop))}__objval"
            class_patterns[node.domain].append(f"?{dv} <{node.prop}> ?{pv} .")
            var_to_col[pv] = node.name
            object_value_columns.append(node.name)

    # 4. Object-property value columns (decision D2) — the older, independent
    # *global* toggle used by non-web callers (Plan 1 D2 / model.py).
    if include_object_properties:
        for node in comp_nodes:
            if node.kind != "object" or node.name in excluded:
                continue
            if not join_active.get(node.name):
                continue
            dv = class_vars[node.domain]
            rv = class_vars.get(node.range_)
            if rv is None:
                continue
            valvar = f"{dv}__{_sparql_var(local_name(node.prop))}_val"
            var_to_col[valvar] = node.name
            target = class_patterns[node.domain]
            if object_value == "key_property":
                kprop = key_properties.get(node.range_)
                if kprop is None:
                    warnings.append(
                        f"object node {node.name}: no key_properties for "
                        f"{local_name(node.range_)}; falling back to range_class"
                    )
                    _bind_range_class(target, rv, valvar, active_classes)
                else:
                    target.append(f"?{rv} <{kprop}> ?{valvar} .")
            else:
                _bind_range_class(target, rv, valvar, active_classes)

    # 5. Assemble query: a spanning tree per connected component of the join
    # graph, each non-root class nested under the join that binds it (see the
    # note above step 1). Any join edge left over once the tree is built —
    # a parallel object property between the same two classes, or one that
    # closes a cycle — still has to hold, so it is added back as a flat
    # top-level triple; both its endpoints are already bound by then, so it
    # is a cheap filter rather than a fresh cross join.
    tree = nx.Graph()
    tree.add_nodes_from(active_classes)
    edge_text: dict = {}
    extra_edges: list = []
    for d, r, text in join_edges:
        pair = frozenset((d, r))
        if pair in edge_text:
            extra_edges.append(text)
        else:
            edge_text[pair] = text
            tree.add_edge(d, r)

    children: dict = defaultdict(list)
    visited: set = set()
    roots: list = []
    for c in active_classes:
        if c in visited:
            continue
        roots.append(c)
        visited.add(c)
        queue = [c]
        while queue:
            cur = queue.pop(0)
            for nbr in tree.neighbors(cur):
                if nbr in visited:
                    continue
                visited.add(nbr)
                children[cur].append((nbr, edge_text[frozenset((cur, nbr))]))
                queue.append(nbr)

    def _render_class_scope(c: str) -> str:
        lines = [type_triple[c], *class_patterns.get(c, [])]
        for child, text in children.get(c, []):
            nested = "\n".join([text, _render_class_scope(child)])
            lines.append("{\n" + textwrap.indent(nested, "    ") + "\n}")
        return "\n".join(lines)

    top_level = [_render_class_scope(root) for root in roots] + extra_edges
    select = " ".join(f"?{v}" for v in var_to_col)
    body = textwrap.indent("\n".join(top_level), "    ")
    query = f"SELECT {select}\nWHERE {{\n{body}\n}}"
    if limit is not None:
        query += f"\nLIMIT {int(limit)}"

    return _Pattern(
        query=query,
        var_to_col=var_to_col,
        warnings=warnings,
        skipped=skipped,
        active_classes=active_classes,
        class_vars=class_vars,
        join_active=join_active,
        object_roles=object_roles,
        object_value_columns=object_value_columns,
        n_disconnected_groups=n_disconnected_groups,
    )


def _materialize_component(
    schema: OntologySchema,
    comp_nodes: list,
    classes: list,
    excluded: set,
    excluded_joins: set,
    include_object_properties: bool,
    optional_data_properties: bool,
    object_value: str,
    key_properties: dict,
    limit: Optional[int],
) -> Materialization:
    p = _build_pattern(
        schema, comp_nodes, classes, excluded, excluded_joins,
        include_object_properties, optional_data_properties, object_value,
        key_properties, limit,
    )
    query = p.query
    var_to_col = p.var_to_col
    active_classes = p.active_classes
    class_vars = p.class_vars

    # A disconnected join is (almost always) a dropped join, not a dropped class — SPARQL
    # still has to compute its full cross product, and that unbounded nested-loop product
    # is what actually produces the "materialise never finishes" reports, not the engine
    # itself (confirmed empirically: 1240x1240 disconnected instances alone reach 1.5M+
    # rows on this project's own sample data). Refusing it outright, rather than running it
    # and reporting how slow it was, is the fix — a row limit makes the intent explicit and
    # keeps the query fast regardless of how large the cross product would have been.
    if p.n_disconnected_groups > 1 and limit is None:
        raise DisconnectedJoinError(
            f"The active classes fall into {p.n_disconnected_groups} groups with no retained "
            "relationship joining them, so an unlimited run would compute their full cross "
            "product rather than a join — almost certainly far larger than intended, and slow "
            "in proportion. Set a row limit to run it anyway, or restore the relationship "
            "connecting them (see the Graph Pattern warning)."
        )

    rows = []
    for row in schema.graph.query(query):
        rows.append({col: _to_py(row[var]) for var, col in var_to_col.items()})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        cols = list(dict.fromkeys(var_to_col.values()))
        df = pd.DataFrame({c: pd.Series(dtype=object) for c in cols})

    # An object-node value column holds the related entity's **full IRI**,
    # unshortened — see the module docstring's last paragraph.

    # Split property columns from entity-id columns.
    entity_cols = [class_vars[c] for c in active_classes]
    prop_cols = [col for var, col in var_to_col.items() if var not in set(entity_cols)]

    # Deterministic row order: natural sort by entity identifiers.
    if entity_cols:
        keys = [
            tuple(_nat_key(str(x)) for x in row)
            for row in df[entity_cols].itertuples(index=False, name=None)
        ]
        order = sorted(range(len(df)), key=lambda i: keys[i])
        df = df.iloc[order].reset_index(drop=True)

    entity_ids = df[entity_cols].copy() if entity_cols else pd.DataFrame(index=df.index)
    data_df = df[prop_cols].copy() if prop_cols else pd.DataFrame(index=df.index)

    prop_col_set = set(prop_cols)
    columns = [
        ColumnSpec(
            name=node.name,
            node=node,
            subject_var=class_vars[node.domain],
        )
        for node in comp_nodes
        if node.name in prop_col_set
    ]

    multiplicity = {}
    if entity_cols:
        for cvar in entity_cols:
            if cvar in df.columns:
                counts = df[cvar].value_counts()
                multiplicity[cvar] = float(counts.mean()) if len(counts) else 0.0

    return Materialization(
        df=data_df,
        columns=columns,
        entity_ids=entity_ids,
        classes=active_classes,
        query=query,
        multiplicity=multiplicity,
        skipped_patterns=p.skipped,
        warnings=p.warnings,
    )


def _bind_range_class(patterns, rv, valvar, classes):
    patterns.append(f"?{rv} a ?{valvar} .")
    cls_list = ", ".join(f"<{c}>" for c in classes)
    patterns.append(f"FILTER(?{valvar} IN ({cls_list}))")


def _warn_parallel(joined_pairs: list, warnings: list) -> None:
    for (d, r), k in Counter(joined_pairs).items():
        if k > 1:
            warnings.append(
                f"{k} parallel object properties between {local_name(d)} and "
                f"{local_name(r)} collapse onto the same variable pair"
            )


def _warn_disconnected(active_classes: list, joined_pairs: list, warnings: list) -> int:
    """Active classes with no retained path between them still share a query,
    so SPARQL computes their cross product rather than leaving one out.

    Returns the number of disconnected groups — 1 for an ordinary, fully joined
    query. The caller uses this to refuse an *unlimited* run of a disconnected
    query outright (see :func:`_materialize_component`): a cross product this
    shape is essentially always accidental curation (a dropped join, not a
    dropped class), and running it to completion is what actually produced the
    "materialise takes forever" reports — the fix is to stop it before it
    starts, not to make the underlying join faster.
    """
    if len(active_classes) <= 1:
        return 1
    ug = nx.Graph()
    ug.add_nodes_from(active_classes)
    ug.add_edges_from(joined_pairs)
    n_comp = nx.number_connected_components(ug)
    if n_comp > 1:
        warnings.append(
            f"The {len(active_classes)} active classes fall into {n_comp} groups with no "
            "retained relationship joining them — the query returns their cross product, "
            "not a join. Restore a relationship, or exclude every node on one side."
        )
    return n_comp


def _nat_key(s: str):
    from ._util import natural_key
    return natural_key(s)
