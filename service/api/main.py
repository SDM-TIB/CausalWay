"""CausalWay web service — Plan 3, modules 1-5.

Ingest -> induced schema -> candidate node graph (curate) -> materialise ->
discover -> curate a DAG -> fit an SCM -> condition/intervene -> counterfactual.
In-memory project store, synchronous processing (no Celery/Postgres/Redis) —
deliberately minimal infra so the whole pipeline works end-to-end in a browser.
See `Plan3 (web service).md` for the full target architecture this is a slice of.

The heavy modelling imports (`causalway.model`, `causalway.inference`,
`causalway.evaluation`) pull in dowhy/pgmpy/sklearn and cost seconds of import
time, so they are imported lazily inside the module 3-5 handlers: a user who
only runs discovery never pays for them.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from service.api import bundle  # noqa: E402

from causalway import bgp  # noqa: E402
from causalway.constraints import EdgeConstraint, render_label  # noqa: E402
from causalway.encoding import (  # noqa: E402
    drop_constant_columns, to_discrete_frame, to_numeric_frame,
)
from causalway.llm_meta import build_domain_str, build_pair_meta, build_var_meta  # noqa: E402
from causalway.nodes import PropertyNode, build_nodes  # noqa: E402
from causalway.ontology import OntologySchema  # noqa: E402
from causalway.result import OntologicalCausalGraph, vocabulary  # noqa: E402
from causalway.sources import resolve_schema  # noqa: E402
from runners.run_kg_discovery import (  # noqa: E402
    ALLOWED_METHODS, CONTINUOUS, DiscoveryContext, run_algorithm, to_ocg,
)

SAMPLE_DIR = REPO_ROOT / "kgs" / "ttls"
SAMPLES = {
    "synthetic_clinic": SAMPLE_DIR / "synthetic_clinic.ttl",
    "sclc_patients": SAMPLE_DIR / "SCLC_patients.ttl",
}

app = FastAPI(title="CausalWay web service (Module 1 slice)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# In-memory project state
# --------------------------------------------------------------------------- #
class ProjectState:
    def __init__(self, name: str):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.created_at = time.time()
        self.source: Optional[dict] = None
        # The path/URL `CausalModel.fit` re-reads in module 3. It re-materialises the KG
        # itself (Plan 2 §3.3 step 2), so it needs the source, not our frame.
        self.source_ref: Optional[str] = None
        self.schema: Optional[OntologySchema] = None
        self.nodes: list[PropertyNode] = []
        self.excluded: set[str] = set()
        # Object properties only: whether the relationship edge is joined,
        # independent of `excluded` (candidate-variable status). See the note
        # above `_range_hint`.
        self.excluded_joins: set[str] = set()
        self.constraint_options = {
            "relation_direction": "both",
            "allow_epsilon": True,
            "max_hops": 1,
        }
        # W20: where a numeric column stops being a set of labels and becomes a number.
        # See TYPE_OPTIONS_NOTE.
        self.type_options: dict = {"max_levels": DEFAULT_MAX_LEVELS,
                                   "float_as_continuous": True}
        # W21: materialisation is a job, not a request. See MATERIALISE_JOB_NOTE.
        self.mat_job: Optional[dict] = None
        self.last_materialization = None
        self.last_component = None
        self.mat_key: Optional[tuple] = None
        self.n_rows = 0
        self.distinct_counts: dict[str, int] = {}
        self.discovery_context: Optional[DiscoveryContext] = None
        self.discovery_context_key: Optional[tuple] = None
        self.discovery_runs: list[dict] = []
        self.next_run_id = 1
        # W12/W13: canvas positions and the curated causal graph are project state, not
        # browser state, so a reload restores the board and an export can carry it.
        self.layouts: dict[str, dict[str, dict]] = {
            "ontology": {}, "causal": {}, "model": {}, "inference": {}, "counterfactual": {},
        }
        self.manual_edges: list[tuple[str, str]] = []   # hand-authored, frequency 0
        self.selected_edges: list[tuple[str, str]] = [] # the curated causal graph
        # W15: GES-Prior metadata and the estimated prior matrices.
        self.prior_meta: dict = {
            "domain": "", "var_text": {}, "entity_map": {}, "endpoint": "", "llm_model": "",
        }
        self.priors: Optional[dict] = None       # {"columns": [...], "M": [[...]], "B": [[...]]}
        self.priors_source: Optional[str] = None # "llm" | "imported"
        # W16: modules 3-5. The fitted model is a live Python object; nothing about it is
        # persisted, which is exactly why /export has to be able to serialise it.
        self.model = None
        self.model_info: Optional[dict] = None
        self.model_key: Optional[tuple] = None
        # W22: "causal Bayesian network" and "structural causal model" are two objects,
        # not two labels for one. See MODEL_KINDS_NOTE.
        self.model_kind: str = "scm"
        self.cbn = None            # pgmpy DiscreteBayesianNetwork, when model_kind == "cbn"
        self.cbn_infer = None      # pgmpy CausalInference over self.cbn
        self.evaluation: Optional[dict] = None
        self.answers: list[dict] = []
        self.next_answer_id = 1
        # Counterfactual *worlds* are kept apart from the query log: each is one
        # entity under one joint intervention, with every node's factual and
        # counterfactual value — the only artifact here that describes named
        # resources of the source KG and can therefore be merged back into it.
        self.cf_worlds: list[dict] = []
        self.next_world_id = 1
        # W29: set only on a project restored from a bundle. `import_notes` is
        # what the panels show instead of looking broken; `imported_source`
        # carries the sha256 a re-attached KG is checked against.
        self.import_notes: list[str] = []
        self.imported_source: dict = {}


PROJECTS: dict[str, ProjectState] = {}


def _get_project(pid: str) -> ProjectState:
    p = PROJECTS.get(pid)
    if p is None:
        raise HTTPException(404, f"No project '{pid}'")
    return p


def _require_schema(p: ProjectState) -> OntologySchema:
    if p.schema is None:
        raise HTTPException(400, "Project has no source yet. POST a source first.")
    return p.schema


def _node_by_name(p: ProjectState, name: str) -> PropertyNode:
    for n in p.nodes:
        if n.name == name:
            return n
    raise HTTPException(404, f"No node '{name}'")


# --------------------------------------------------------------------------- #
# W26 — an object property is a relationship XOR a causal variable
#
# It used to be allowed to be both at once (Plan 3 §Preprocess revision 2), and
# that was a mistake rather than a feature. A both-roles column's value was
# read off the *joined* class variable, so it was by construction a
# deterministic function of the join that produced the row: discovery then saw
# a variable that perfectly predicts every property of the range class, which
# is an artefact of the materialisation, not a causal fact about the KG. It
# also meant toggling the canvas edge silently changed what the column meant.
#
# The three roles, resolved by `bgp.object_property_role` from the same two
# sets as before (no API break, and the canvas edge still writes
# `excluded_joins`):
#
#   relationship  joined; not a column. The default — an object property reads
#                 as a relationship, and opting it in as a variable is the
#                 user's call (§3.5).
#   variable      not joined; a column holding the related entity's full IRI,
#                 read off a private pattern that does not activate the range
#                 class.
#   dropped       neither.
#
# Choosing `variable` therefore drops a join, which can leave the query
# computing a cross product. `_reject_role_conflicts` refuses that at the
# moment of choice and names the property; `bgp.join_role_options` supplies
# the same verdict to the card so the choice is disabled before it is made.
#
# `PUT /nodes` accepts `object_roles` for this; `excluded`/`excluded_joins`
# remain the wire format underneath.
#
# All of it flows into `bgp.materialize` with `include_object_properties=False`
# — the older *global* toggle for the `object_value=`/`key_properties=`
# encoding (Plan 1 D2), which `model.py` still uses when it re-materialises
# from source (module 3), is a different code path with different semantics
# and the exclusivity rule deliberately does not apply to it.
#
# W30 narrows the *default* without touching the rule: an object property whose
# range class has no data properties starts at `variable`, because for that
# property the relationship role joins in a class with nothing to contribute.
# See `_seed_object_variables`.
# --------------------------------------------------------------------------- #


def _range_hint(range_str: str) -> str:
    low = range_str.lower()
    if any(t in low for t in ("int", "float", "double", "decimal", "long", "short")):
        return "numeric"
    if any(t in low for t in ("date", "time")):
        return "date"
    return "string"


def _set_schema(p: ProjectState, schema: OntologySchema, source_meta: dict,
                source_ref: str) -> None:
    p.schema = schema
    p.source = source_meta
    p.source_ref = source_ref
    # Object properties get PropertyNodes like data properties do, and start in the
    # `relationship` role (W26): excluded as a candidate *variable*, joined as a
    # relationship — which is both the natural reading of an object property and
    # what the ontology canvas draws by default. Opting one in as a variable is the
    # user's call (§3.5), and it costs the join.
    #
    # W30 is the one exception, and it is seeded here rather than left to the user:
    # an object property whose *range class carries no data properties* is an
    # attribute an ontologist chose to model as a class, and the relationship role
    # is empty for it. See `_seed_object_variables`.
    p.nodes = build_nodes(schema, include_object_properties=True)
    p.excluded = {n.name for n in p.nodes if n.kind == "object"}
    p.excluded_joins = set()
    _seed_object_variables(schema, p.nodes, p.excluded, p.excluded_joins)
    _invalidate_materialisation(p)
    p.layouts ={"ontology": {}, "causal": {}, "model": {}, "inference": {}, "counterfactual": {}}
    p.manual_edges = []
    p.selected_edges = []
    p.prior_meta = {"domain": "", "var_text": {}, "entity_map": {}, "endpoint": "", "llm_model": ""}
    p.priors = None
    p.priors_source = None
    # Pre-flag constant columns is data-dependent (needs materialisation); skipped in this slice.


def _invalidate_model(p: ProjectState) -> None:
    """A fitted model is only meaningful for the graph it was fitted to (§9 cascade)."""
    p.model = None
    p.model_info = None
    p.model_key = None
    p.cbn = None
    p.cbn_infer = None
    p.evaluation = None
    p.answers = []
    p.next_answer_id = 1
    p.cf_worlds = []
    p.next_world_id = 1


def _invalidate_materialisation(p: ProjectState) -> None:
    """Everything downstream of the frame: mat -> context -> runs -> ledger -> model."""
    p.last_materialization = None
    p.last_component = None
    p.mat_key = None
    p.n_rows = 0
    p.distinct_counts = {}
    p.discovery_context = None
    p.discovery_context_key = None
    p.discovery_runs = []
    p.next_run_id = 1
    _invalidate_model(p)


def _schema_components(schema: OntologySchema) -> list[set]:
    ug = schema.class_graph.to_undirected()
    ug.add_nodes_from(schema.classes)
    return [c for c in nx.connected_components(ug)]


# --------------------------------------------------------------------------- #
# W31 — the component is a choice, not a consequence
#
# One `materialize` call is one basic graph pattern over one connected component
# of the class graph, and that part is not a limitation: Assumption 1 forbids
# every edge between two classes with no relation path, so a cross-component
# edge is inadmissible by construction and joining two components could only
# manufacture a cross product.
#
# What *was* a limitation is that the component was never chosen. Every route
# has always accepted `component=`, and nothing ever sent one — so the fallback
# below ("whichever has the most retained nodes") was not a default, it was the
# whole behaviour. On the bundled samples, which have one component each, that
# is invisible, which is why it survived this long. On a real endpoint whose
# T-Box holds seventy of them it means sixty-nine are unreachable through the
# UI, with no way to even see what is in them.
#
# So the component list now carries enough to choose from — its classes, and
# how many of its nodes are retained right now — and every node says which
# component it is in, so a client can scope its curation list to the component
# whose query it is showing. `component=None` keeps its old meaning exactly:
# *auto*, which still follows the curation as the user changes it. Pinning one
# is the user overriding that, the same relationship a pinned row limit has to
# "no limit".
# --------------------------------------------------------------------------- #
def _auto_component_index(nodes: list[PropertyNode], excluded: set[str],
                          components: list[set]) -> Optional[int]:
    """What `component=None` resolves to: most retained nodes, ties to the lowest index.

    `None` when there is nothing to resolve — no components, or every node
    excluded — which is the same condition `_resolve_component` rejects.
    """
    retained = [n for n in nodes if n.name not in excluded]
    if not components or not retained:
        return None
    counts = [sum(1 for n in retained if n.domain in c) for c in components]
    return max(range(len(components)), key=lambda i: counts[i])


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class CreateProject(BaseModel):
    name: str = "Untitled project"


class SampleSource(BaseModel):
    sample_id: str


class EndpointSource(BaseModel):
    url: str
    infer_missing: Optional[bool] = None


class NodeUpdate(BaseModel):
    excluded: list[str]
    excluded_joins: list[str] = []
    # W26: the preferred way to curate an object property. `excluded` /
    # `excluded_joins` are two independent axes whose *four* combinations only
    # ever meant three things (see `bgp.object_property_role`); making the
    # client encode a role into two sets meant the client had to know the
    # resolution rule, and a client that got it wrong produced a curation the
    # server silently reinterpreted. Sent as `{node_name: role}`, it overrides
    # both sets for the object properties it names and leaves the rest alone.
    object_roles: Optional[dict[str, str]] = None
    relation_direction: str = "both"
    allow_epsilon: bool = True
    max_hops: int = 1
    max_levels: Optional[int] = None
    float_as_continuous: Optional[bool] = None


# --------------------------------------------------------------------------- #
# W20 — where a numeric column stops being labels and becomes a number
#
# `max_levels` only ever applies to *integer-valued* numeric columns. A string or
# IRI column is categorical however many levels it has, and a genuinely fractional
# column is continuous however few — binding 0.5/1.0/2.0 into three unrelated
# labels throws away ordering the numbers already carry, and gains nothing.
#
# For integer columns the threshold is a real modelling choice with a real cost
# on each side:
#   <= max_levels  categorical — a classifier / CPT per node. Levels are unordered,
#                  so the model cannot know 12 lies between 11 and 13. Keeps the
#                  network fully discrete, which is what makes exact inference
#                  (pgmpy variable elimination) available at all.
#   >  max_levels  discrete — a regressor. Ordering is respected and the model can
#                  interpolate, but ONE such column flips the whole network out of
#                  the exact backend and into sampling.
# Nothing is ever dropped: every column is modelled either way.
#
# 6 is the default because the integer columns a KG actually produces at that
# cardinality (stage, grade, ordinal scores, counts of a handful) are labels, and
# anything wider is almost always a measurement.
# --------------------------------------------------------------------------- #
DEFAULT_MAX_LEVELS = 6
TYPE_OPTIONS_NOTE = "max_levels applies to integer columns only; floats are always continuous."


# --------------------------------------------------------------------------- #
# W21 — materialisation is a job, not a request
#
# The flat join has no natural bound and no query timeout: on a 1:N-heavy schema
# rdflib's in-memory join runs for minutes. The old service capped `limit` at 5000
# to keep the request synchronous, which quietly made truncation the norm — every
# downstream number was computed on the first 500 solutions the store happened to
# return, in whatever order it returned them.
#
# So: the default is now *unlimited* and started automatically, on a worker thread,
# with elapsed time reported while it runs and a limit offered as the escape hatch
# once it gets slow.
#
# `cancel` is honest about what it can do. rdflib evaluates the SELECT inside a
# generator this service does not own, and a Python thread cannot be killed from
# outside, so cancelling marks the job abandoned and *discards its result* — the UI
# returns immediately, but the query keeps consuming CPU until it finishes on its
# own. Making that genuinely interruptible means a cancel check inside
# `causalway/bgp.py`'s row loop, which is an engine change and a separate decision.
# --------------------------------------------------------------------------- #
MATERIALISE_JOB_NOTE = (
    "Cancelling abandons the result; the underlying rdflib join keeps running to "
    "completion in the background."
)
SLOW_MATERIALISE_SECONDS = 180


class MaterializeRequest(BaseModel):
    component: Optional[int] = None
    optional_data_properties: bool = False
    # None == no SPARQL LIMIT at all. The whole join, however long it takes.
    limit: Optional[int] = None


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
@app.post("/api/projects")
def create_project(body: CreateProject):
    p = ProjectState(body.name)
    PROJECTS[p.id] = p
    return {"id": p.id, "name": p.name, "created_at": p.created_at}


@app.get("/api/projects")
def list_projects():
    return [
        {"id": p.id, "name": p.name, "created_at": p.created_at, "has_source": p.source is not None}
        for p in PROJECTS.values()
    ]


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = _get_project(pid)
    return {
        "id": p.id,
        "name": p.name,
        "created_at": p.created_at,
        "source": p.source,
        "n_nodes": len(p.nodes),
        "n_excluded": len(p.excluded),
        "constraint_options": p.constraint_options,
    }


@app.get("/api/samples")
def list_samples():
    return [
        {"sample_id": k, "exists": v.exists(), "path": str(v)}
        for k, v in SAMPLES.items()
    ]


# --------------------------------------------------------------------------- #
# Sources (Module 1 §3.1)
# --------------------------------------------------------------------------- #
@app.post("/api/projects/{pid}/sources/sample")
def add_sample_source(pid: str, body: SampleSource):
    p = _get_project(pid)
    path = SAMPLES.get(body.sample_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"Unknown sample '{body.sample_id}'")
    data = path.read_bytes()
    schema = resolve_schema(str(path))
    _set_schema(p, schema, {
        "kind": "sample", "name": path.name, "sha256": _sha256_bytes(data),
        "bytes": len(data),
    }, source_ref=str(path))
    return _schema_payload(p)


@app.post("/api/projects/{pid}/sources/upload")
async def upload_source(pid: str, file: UploadFile = File(...)):
    p = _get_project(pid)
    data = await file.read()
    suffix = Path(file.filename or "upload.ttl").suffix or ".ttl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        schema = resolve_schema(tmp_path)
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI verbatim
        raise HTTPException(400, f"Could not parse '{file.filename}': {exc}") from exc
    # The temp file is deliberately not deleted: module 3 re-reads the source to
    # materialise independently (Plan 2 §3.3), so the upload has to outlive this request.
    _set_schema(p, schema, {
        "kind": "file", "name": file.filename, "sha256": _sha256_bytes(data),
        "bytes": len(data),
    }, source_ref=tmp_path)
    return _schema_payload(p)


@app.post("/api/projects/{pid}/sources/endpoint")
def add_endpoint_source(pid: str, body: EndpointSource):
    p = _get_project(pid)
    try:
        schema = resolve_schema(body.url, infer_missing=body.infer_missing)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not resolve endpoint '{body.url}': {exc}") from exc
    _set_schema(p, schema, {"kind": "endpoint", "name": body.url, "sha256": None, "bytes": None},
                source_ref=body.url)
    return _schema_payload(p)


def _schema_payload(p: ProjectState) -> dict:
    # The guard belongs here, not only at the routes: a project imported from a
    # zip has no source KG (W29 — the KG is deliberately not in the bundle), so
    # `p.schema` is None on every artifact path that reaches this. Without it
    # the AttributeError escaped as a 500 through `_render(p, "schema", ...)`
    # and took the whole `what=project` zip with it, because that loop skips an
    # artifact only on HTTPException. An absent schema is a 400 like every other
    # not-produced-yet artifact.
    schema = _require_schema(p)
    summary = schema.summary()
    return {
        "source": p.source,
        "schema": {
            **summary,
            "classes": sorted(schema.labels.get(c, str(c).split("/")[-1].split("#")[-1]) for c in schema.classes),
            "conflicts": [str(c) for c in schema.conflicts],
        },
        "n_nodes": len(p.nodes),
    }


@app.get("/api/projects/{pid}/schema")
def get_schema(pid: str):
    p = _get_project(pid)
    _require_schema(p)
    return _schema_payload(p)


# --------------------------------------------------------------------------- #
# Candidate node graph + curation (Module 1 §3.4-3.5)
# --------------------------------------------------------------------------- #
@app.get("/api/projects/{pid}/nodes")
def get_nodes(pid: str):
    p = _get_project(pid)
    schema = _require_schema(p)
    opts = p.constraint_options
    ec = EdgeConstraint.from_schema(
        schema, p.nodes,
        relation_direction=opts["relation_direction"],
        allow_epsilon=opts["allow_epsilon"],
        max_hops=opts["max_hops"],
    )
    # W26: an object property is a relationship XOR a causal variable. The
    # options come from `bgp` rather than being recomputed here, so the card's
    # disabled state and the materialiser's refusal cannot disagree.
    role_options = bgp.join_role_options(schema, p.nodes, p.excluded, p.excluded_joins)
    # W30: which object properties point at a class that has no data properties.
    # Recomputed from the node set rather than remembered from `_set_schema`,
    # because it is a fact about the *schema* and never about the curation — it
    # stays true (and stays worth showing) after the user overrides the role.
    auto_variable = _auto_variable_names(p.nodes)
    # W31: which connected component each class — and so each node — belongs to.
    # A node's component is its *domain*'s, because `n.domain in components[i]`
    # is the exact test `_component_all_nodes` uses to draw a component's join
    # from: a node the client lists under component i is a node the query for
    # component i will contain, with no second rule to drift out of step.
    components = _schema_components(schema)
    class_component = {c: i for i, comp in enumerate(components) for c in comp}

    nodes_payload = []
    for n in p.nodes:
        range_str = n.range_.split("#")[-1].split("/")[-1] if n.kind == "data" else \
            schema.labels.get(n.range_, str(n.range_).split("/")[-1].split("#")[-1])
        nodes_payload.append({
            "name": n.name,
            "domain": schema.labels.get(n.domain, str(n.domain).split("/")[-1].split("#")[-1]),
            "prop": str(n.prop).split("/")[-1].split("#")[-1],
            "range": range_str,
            "kind": n.kind,
            "value_hint": _range_hint(range_str) if n.kind == "data" else "entity",
            "excluded": n.name in p.excluded,
            # Object properties only — whether the relationship is joined.
            "join_excluded": n.kind == "object" and n.name in p.excluded_joins,
            # W26: the resolved role, and which roles this curation allows.
            # `None` for a data property, which has no role to choose.
            "role": role_options.get(n.name, {}).get("role"),
            "role_options": role_options.get(n.name),
            # W30: the range class declares no data properties, so joining it
            # contributes no columns and the only thing it can say is which
            # instance was pointed at — an attribute modelled as a class. These
            # start in the `variable` role instead of `relationship`.
            "auto_variable": n.name in auto_variable,
            # W31: `_schema_components` seeds its graph with every class, so this
            # is a total map and the `None` branch is unreachable for a node this
            # schema produced. If it ever were `None` it would be the truth: a
            # node whose domain is in no component is one no join can reach.
            "component": class_component.get(n.domain),
            # W11: the card's second line reads "<range> · <n> distinct" once a materialisation
            # has supplied a count; None until then, and the card shows the range alone.
            "n_distinct": p.distinct_counts.get(n.name),
            "n_rows": p.n_rows or None,
            # W17: an object property whose value is near-unique per row is an entity
            # identifier wearing a variable's clothes — every level occurs once, so no
            # score can learn anything from it. Flagged, not hidden: whether it is a real
            # categorical or an id is a modelling judgement, and it is the user's.
            "near_unique": bool(
                p.n_rows and p.distinct_counts.get(n.name, 0) >= max(2, 0.9 * p.n_rows)
            ),
            # Constant means the column carries no information and will be dropped
            # before fitting.
            "constant": bool(p.n_rows and p.distinct_counts.get(n.name, 0) <= 1),
        })

    # Collapse symmetric (i, j)/(j, i) pairs into one undirected edge for display.
    edges = []
    seen = set()
    n = ec.n
    for i in range(n):
        for j in range(n):
            if i == j or (j, i) in seen:
                continue
            fwd = ec.allowed[i, j]
            bwd = ec.allowed[j, i]
            if not (fwd or bwd):
                continue
            labels = []
            if fwd:
                labels.append(ec.label_str(i, j))
            if bwd and (i, j) not in seen:
                pass
            seen.add((i, j))
            edges.append({
                "source": ec.names[i],
                "target": ec.names[j],
                "label": ec.label_str(i, j) if fwd else ec.label_str(j, i),
                "bidirectional": bool(fwd and bwd),
            })

    dtypes = _frame_dtypes(p)
    for row in nodes_payload:
        row["dtype"] = dtypes.get(row["name"])

    return {
        "nodes": nodes_payload,
        "edges": edges,
        "stats": ec.stats(),
        "constraint_options": p.constraint_options,
        "type_options": {
            **p.type_options,
            "default_max_levels": DEFAULT_MAX_LEVELS,
            "note": TYPE_OPTIONS_NOTE,
        },
        "dtypes": dtypes,
        # W31: enough per component to pick one. An index alone is not a choice a
        # human can make among seventy of them — the class names are what say
        # which part of the ontology this is, and `n_retained` is what says
        # whether `materialise` would accept it (zero is the one case it refuses
        # outright), so the picker can report that before it is chosen.
        "components": [
            {
                "index": i,
                "n_classes": len(c),
                "n_nodes": sum(1 for nd in p.nodes if nd.domain in c),
                "n_retained": sum(1 for nd in p.nodes
                                  if nd.domain in c and nd.name not in p.excluded),
                "classes": sorted(
                    schema.labels.get(cl, str(cl).split("/")[-1].split("#")[-1]) for cl in c
                ),
            }
            for i, c in enumerate(components)
        ],
        # What `component=None` means at this curation, so the client can show
        # the auto choice rather than re-deriving the rule and disagreeing.
        "auto_component": _auto_component_index(p.nodes, p.excluded, components),
    }


def _apply_object_roles(nodes: list[PropertyNode], excluded: set[str],
                        excluded_joins: set[str], roles: dict[str, str]) -> tuple[set, set]:
    """Fold `{node: role}` back into the two curation sets (W26).

    The inverse of `bgp.object_property_role`, and the only place the service
    performs it — a second encoding of the same rule somewhere else is how the
    client and the materialiser drift apart on what a curation means.
    """
    object_names = {n.name for n in nodes if n.kind == "object"}
    unknown = set(roles) - object_names
    if unknown:
        raise HTTPException(400, (
            f"object_roles names something that is not an object property: {sorted(unknown)}. "
            "Only object properties have a role; a data property is a candidate variable "
            "or not, via `excluded`."
        ))
    bad = {k: v for k, v in roles.items() if v not in bgp.OBJECT_ROLES}
    if bad:
        raise HTTPException(400, f"Unknown object-property roles {bad}; "
                                 f"expected one of {list(bgp.OBJECT_ROLES)}.")

    excluded, excluded_joins = set(excluded), set(excluded_joins)
    for name, role in roles.items():
        if role == "relationship":
            excluded.add(name)            # not a variable
            excluded_joins.discard(name)  # but joined
        elif role == "variable":
            excluded.discard(name)
            excluded_joins.add(name)      # a variable means the join is dropped
        else:  # "dropped"
            excluded.add(name)
            excluded_joins.add(name)
    return excluded, excluded_joins


def _value_only_ranges(nodes: list[PropertyNode]) -> set:
    """Class IRIs that no *data* property node is declared on.

    A class with no literal attributes of its own cannot contribute a single
    column to the flat join, so joining it buys nothing: the only thing such a
    class can tell the analysis is *which* of its instances an entity points at.
    Computed from `nodes` rather than from `schema.data_properties` so it uses
    exactly the candidate-node set the rest of module 1 curates — a class whose
    only data property was dropped during schema induction is value-only here
    too, which is the truthful answer for the join.
    """
    data_bearing = {n.domain for n in nodes if n.kind == "data"}
    return {n.range_ for n in nodes if n.kind == "object"} - data_bearing


def _auto_variable_names(nodes: list[PropertyNode]) -> set[str]:
    """Object properties that W30 seeds into the `variable` role (see below)."""
    value_only = _value_only_ranges(nodes)
    return {n.name for n in nodes if n.kind == "object" and n.range_ in value_only}


# --------------------------------------------------------------------------- #
# W30 — an object property pointing at a class that has no data properties
#       starts as a *causal variable*, not as a relationship.
#
# W26 made `relationship` the default for every object property, which is the
# right reading when the range class is a real entity with attributes of its
# own: joining Therapy into a Patient row brings Therapy's columns with it.
# It is the wrong reading — and silently loses the variable — when the range
# class is an *attribute modelled as a class*. Ontologists do this constantly:
# an age band, a tumour stage, a severity level become `:Age`, `:Stage`,
# `:Severity`, instances `:young`/`:middle`/`:old`, and `:p :hasAge :young`.
# Such a class declares no data properties at all, so the relationship role
# joins in a class that contributes no columns, and the one fact the KG was
# expressing — which band this patient is in — reaches the analysis table
# nowhere. The user then has to find every such property by hand and flip it,
# on a curation screen that gives no hint which ones they are.
#
# Seeded, not forced: this only sets the starting curation. The three-way role
# selector still moves any of them back, and `PUT /nodes` is unchanged.
#
# Applied one at a time, each verified against the accumulated state, because
# the variable role drops a join and several dropped joins can split the query
# into a cross product (W26 (b)). A flip that would do that is reverted and the
# property is left as a relationship — the same verdict `_reject_role_conflicts`
# would give, from the same function, so a seeded curation is never one the
# server would refuse if the user typed it in. Candidates are taken in name
# order so the seed is reproducible for a given schema.
# --------------------------------------------------------------------------- #
def _seed_object_variables(schema, nodes: list[PropertyNode], excluded: set[str],
                           excluded_joins: set[str]) -> list[str]:
    """Flip the W30 candidates into the `variable` role, in place. Returns them."""
    by_name = {n.name: n for n in nodes}
    flipped: list[str] = []
    for name in sorted(_auto_variable_names(nodes)):
        if name not in by_name:
            continue
        excluded.discard(name)
        excluded_joins.add(name)
        if _variable_role_conflicts(schema, nodes, excluded, excluded_joins):
            excluded.add(name)
            excluded_joins.discard(name)
            continue
        flipped.append(name)
    return flipped


def _variable_role_conflicts(schema, nodes: list[PropertyNode], excluded: set[str],
                             excluded_joins: set[str]) -> list[tuple[str, str]]:
    """`(name, reason)` for every node sitting at `variable` that cannot be one."""
    return [
        (name, opt["reason"])
        for name, opt in bgp.join_role_options(
            schema, nodes, excluded, excluded_joins).items()
        if opt["role"] == "variable" and not opt["can_be_variable"]
    ]


def _reject_role_conflicts(p: ProjectState, excluded: set[str],
                           excluded_joins: set[str]) -> None:
    """Refuse a curation whose *variable* roles are what split the query.

    A dropped join leaves SPARQL computing a cross product rather than a join,
    and `bgp.materialize` already refuses to run one unlimited. But finding out
    at Materialise — after the curation is committed and the job is queued — is
    a bad way to learn it, and the error there cannot say which of several
    choices caused it. This runs the same test at the moment the choice is
    made, and names the property.

    Only the *variable* role is refused. Dropping a relationship from the
    ontology canvas can disconnect the query too, and that has always been
    allowed (with a warning, and with a row limit at materialise time) — it is
    how a user narrows a join deliberately. Taking that away here would break a
    working feature to enforce a rule about a different one.
    """
    conflicts = _variable_role_conflicts(p.schema, p.nodes, excluded, excluded_joins)
    if conflicts:
        raise HTTPException(400, " ".join(reason for _, reason in conflicts))


@app.put("/api/projects/{pid}/nodes")
def update_nodes(pid: str, body: NodeUpdate):
    p = _get_project(pid)
    _require_schema(p)
    valid_names = {n.name for n in p.nodes}
    unknown = set(body.excluded) - valid_names
    if unknown:
        raise HTTPException(400, f"Unknown node names: {sorted(unknown)}")
    object_names = {n.name for n in p.nodes if n.kind == "object"}
    unknown_joins = set(body.excluded_joins) - object_names
    if unknown_joins:
        raise HTTPException(400, f"Unknown object-property node names: {sorted(unknown_joins)}")

    before = (frozenset(p.excluded), frozenset(p.excluded_joins), dict(p.constraint_options))

    new_excluded = set(body.excluded)
    new_joins = set(body.excluded_joins)
    if body.object_roles is not None:
        new_excluded, new_joins = _apply_object_roles(
            p.nodes, new_excluded, new_joins, body.object_roles)
    _reject_role_conflicts(p, new_excluded, new_joins)

    p.excluded = new_excluded
    p.excluded_joins = new_joins
    p.constraint_options = {
        "relation_direction": body.relation_direction,
        "allow_epsilon": body.allow_epsilon,
        "max_hops": body.max_hops,
    }

    type_changed = False
    if body.max_levels is not None:
        if body.max_levels < 2 or body.max_levels > 200:
            raise HTTPException(400, "max_levels must be between 2 and 200.")
        type_changed |= int(body.max_levels) != int(p.type_options["max_levels"])
        p.type_options["max_levels"] = int(body.max_levels)
    if body.float_as_continuous is not None:
        type_changed |= bool(body.float_as_continuous) != bool(
            p.type_options["float_as_continuous"])
        p.type_options["float_as_continuous"] = bool(body.float_as_continuous)

    after = (frozenset(p.excluded), frozenset(p.excluded_joins), dict(p.constraint_options))
    if before != after:
        _invalidate_materialisation(p)
    elif type_changed:
        # The type threshold does not change the join, only how its columns are read,
        # so it invalidates the *model* but not the materialisation (W20). Rerunning a
        # join that can take minutes to change one integer would be absurd.
        _invalidate_model(p)
    return get_nodes(pid)


# --------------------------------------------------------------------------- #
# Materialisation preview (Module 1 §3.8)
# --------------------------------------------------------------------------- #
def _clean_cell(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _frame_dtypes(p: ProjectState) -> dict[str, str]:
    """The dtype every materialised column *will* be given when a model is fitted.

    Computed here, from the same `_infer_dtype` the fit uses and the project's own
    `type_options`, so module 1 can show the consequence of the threshold before the
    30-second fit rather than after it — and so the UI knows whether a Bayesian
    network is even offerable (it is not, once one column is continuous) (W20).
    """
    if p.last_materialization is None:
        return {}
    from causalway.model import _infer_dtype  # noqa: PLC0415 — cheap, no dowhy import

    opts = p.type_options
    out: dict[str, str] = {}
    for col in p.last_materialization.df.columns:
        series = p.last_materialization.df[col]
        try:
            out[str(col)] = _infer_dtype(
                series, False, int(opts["max_levels"]),
                float_as_continuous=bool(opts["float_as_continuous"]),
            )
        except Exception:  # noqa: BLE001 — an unclassifiable column is just categorical
            out[str(col)] = "categorical"
    return out


def _resolve_component(p: ProjectState, schema: OntologySchema,
                       component: Optional[int]) -> tuple[int, list[PropertyNode]]:
    retained = [n for n in p.nodes if n.name not in p.excluded]
    if not retained:
        raise HTTPException(400, "All nodes are excluded; nothing to materialise.")
    components = _schema_components(schema)
    comp_idx = component
    if comp_idx is None:
        auto = _auto_component_index(p.nodes, p.excluded, components)
        comp_idx = 0 if auto is None else auto
    if comp_idx < 0 or comp_idx >= len(components):
        raise HTTPException(400, f"No component {comp_idx}")
    comp_nodes = [n for n in retained if n.domain in components[comp_idx]]
    if not comp_nodes:
        raise HTTPException(400, f"Component {comp_idx} has no retained nodes.")
    return comp_idx, comp_nodes


def _component_all_nodes(p: ProjectState, schema: OntologySchema, comp_idx: int) -> list[PropertyNode]:
    """Every node of the component, retained *and* excluded.

    The join needs both: an excluded data property still contributes its required
    pattern (§requirement 3), and only an excluded *object* property's join actually
    disappears from the query.
    """
    components = _schema_components(schema)
    return [n for n in p.nodes if n.domain in components[comp_idx]]


def _mat_key(p: ProjectState, comp_idx: int, limit: Optional[int], optional_dp: bool) -> tuple:
    return (comp_idx, limit, bool(optional_dp), frozenset(p.excluded), frozenset(p.excluded_joins))


def _run_join(p: ProjectState, schema: OntologySchema, comp_idx: int,
              limit: Optional[int], optional_dp: bool):
    """Run the join and return the frame — *without* writing it to the project.

    Separate from the commit because the worker thread may still be running long after
    the user cancelled it. A cancelled job that wrote its result here would overwrite a
    newer frame the user has since produced, and everything fitted to it. The commit is
    guarded; this is not (W21).

    A retained object-property node becomes a column here — the local name of the
    related entity — independent of whether its relationship is joined (see the
    note above `_range_hint`).
    """
    all_nodes = _component_all_nodes(p, schema, comp_idx)
    try:
        mat = bgp.materialize(
            schema, all_nodes, excluded=p.excluded, excluded_joins=p.excluded_joins,
            include_object_properties=False,
            optional_data_properties=optional_dp, limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim, a user-facing SPARQL error
        raise HTTPException(400, f"Materialisation failed: {exc}") from exc
    if isinstance(mat, list):
        # Shouldn't happen since comp_nodes is drawn from one component, but guard anyway.
        mat = mat[0]
    return mat


def _commit_materialisation(p: ProjectState, mat, key: tuple, comp_idx: int) -> None:
    p.last_materialization = mat
    p.mat_key = key
    p.last_component = comp_idx
    p.n_rows = int(mat.df.shape[0])
    # Distinct counts feed the ontology cards' second line (W11) and the near-unique
    # flag on object properties (W17). Computed here because this is where the frame is.
    p.distinct_counts = {str(c): int(mat.df[c].nunique(dropna=True)) for c in mat.df.columns}


def _materialise(p: ProjectState, schema: OntologySchema, comp_idx: int,
                 limit: Optional[int], optional_dp: bool):
    """The one flat join. Cached on the project so module 1's preview and module 2's
    discovery context share a single query rather than each running their own (W18).
    """
    key = _mat_key(p, comp_idx, limit, optional_dp)
    if p.last_materialization is not None and p.mat_key == key:
        return p.last_materialization
    mat = _run_join(p, schema, comp_idx, limit, optional_dp)
    _commit_materialisation(p, mat, key, comp_idx)
    return mat


def _mat_payload(p: ProjectState, schema: OntologySchema, comp_idx: int,
                 limit: Optional[int]) -> dict:
    mat = p.last_materialization
    components = _schema_components(schema)
    df = mat.df
    preview = [
        {k: _clean_cell(v) for k, v in row.items()}
        for row in df.head(20).to_dict(orient="records")
    ]
    return {
        "component": comp_idx,
        "n_components_total": len(components),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [c.name for c in mat.columns],
        "classes": [str(c) for c in mat.classes],
        "multiplicity": mat.multiplicity,
        "skipped_patterns": mat.skipped_patterns,
        "warnings": mat.warnings,
        "sparql": mat.query,
        "preview_rows": preview,
        "distinct_counts": p.distinct_counts,
        "dtypes": _frame_dtypes(p),
        "has_multi_relation": any(v > 1.0001 for v in mat.multiplicity.values()),
        # `limit` is a SPARQL LIMIT: the first N solutions in whatever order the store
        # returns them, not a random sample. Truncation is reported so nobody reads the
        # frame as an i.i.d. draw from the KG (W18).
        "truncated": limit is not None and int(df.shape[0]) >= limit,
        "limit": limit,
        "near_unique_columns": [
            c for c in df.columns
            if p.distinct_counts.get(str(c), 0) >= max(2, 0.9 * max(1, df.shape[0]))
        ],
    }


def _job_status(p: ProjectState, include_result: bool = True) -> dict:
    job = p.mat_job
    if job is None:
        return {"state": "idle", "elapsed": 0.0, "note": MATERIALISE_JOB_NOTE}
    elapsed = (job["finished"] or time.time()) - job["started"]
    out = {
        "job_id": job["id"],
        "state": job["state"],
        "elapsed": round(elapsed, 1),
        "limit": job["limit"],
        "component": job["component"],
        "error": job["error"],
        "slow": job["state"] == "running" and elapsed > SLOW_MATERIALISE_SECONDS,
        "slow_after": SLOW_MATERIALISE_SECONDS,
        "note": MATERIALISE_JOB_NOTE,
    }
    if include_result and job["state"] == "done":
        out["result"] = job["result"]
    return out


def _run_materialise_job(p: ProjectState, job: dict, schema: OntologySchema,
                         comp_idx: int) -> None:
    key = _mat_key(p, comp_idx, job["limit"], job["optional_dp"])
    try:
        mat = _run_join(p, schema, comp_idx, job["limit"], job["optional_dp"])
    except HTTPException as exc:
        if p.mat_job is job and job["state"] == "running":
            job["state"], job["error"] = "error", str(exc.detail)
            job["finished"] = time.time()
        return
    except Exception as exc:  # noqa: BLE001 — a SPARQL/join error is the user's to read
        if p.mat_job is job and job["state"] == "running":
            job["state"], job["error"] = "error", str(exc)
            job["finished"] = time.time()
        return

    job["finished"] = time.time()
    if job["state"] == "cancelled" or p.mat_job is not job:
        # Abandoned while it ran. Drop the frame on the floor and touch nothing else:
        # by now the user has very likely re-run at a limit and built a context, runs and
        # a model on top of it, and all of that is *newer* than this result, not staler.
        return
    _commit_materialisation(p, mat, key, comp_idx)
    # Result first, state last: a poll that lands between the two would otherwise see
    # "done" with nothing attached, and the client would show an empty table for a join
    # that had in fact just succeeded.
    job["result"] = _mat_payload(p, schema, comp_idx, job["limit"])
    job["state"] = "done"


@app.post("/api/projects/{pid}/materialise")
def materialize(pid: str, body: MaterializeRequest):
    """Start the flat join. Returns immediately; poll `/materialise/status` (W21)."""
    p = _get_project(pid)
    schema = _require_schema(p)
    if body.limit is not None and body.limit <= 0:
        raise HTTPException(400, "limit must be a positive row count, or null for no limit.")
    comp_idx, _comp_nodes = _resolve_component(p, schema, body.component)

    running = p.mat_job is not None and p.mat_job["state"] == "running"
    if running:
        raise HTTPException(409, "A materialisation is already running; cancel it first.")

    job = {
        "id": uuid.uuid4().hex[:8], "state": "running", "started": time.time(),
        "finished": None, "limit": body.limit, "component": comp_idx,
        "optional_dp": body.optional_data_properties, "result": None, "error": None,
    }
    p.mat_job = job
    threading.Thread(
        target=_run_materialise_job, args=(p, job, schema, comp_idx),
        daemon=True, name=f"materialise-{p.id}-{job['id']}",
    ).start()
    # A small join finishes inside the first tick, which saves the client a round trip
    # and makes the common case feel synchronous without being synchronous.
    time.sleep(0.35)
    return _job_status(p)


@app.get("/api/projects/{pid}/materialise/status")
def materialize_status(pid: str):
    return _job_status(_get_project(pid))


@app.get("/api/projects/{pid}/materialise/preview")
def materialize_preview(pid: str, component: Optional[int] = None, limit: Optional[int] = None):
    """The SPARQL text `materialise` would run, without running it.

    Query construction (`bgp.build_query`) is pure string assembly over the schema
    graph's class/property structure — no `schema.graph.query(...)` call, so this is
    cheap enough to hit on every curation change and back the Preprocess module's live
    "Graph Pattern" box, letting the user check the join's shape before paying for it.
    """
    p = _get_project(pid)
    schema = _require_schema(p)
    comp_idx, _comp_nodes = _resolve_component(p, schema, component)
    all_nodes = _component_all_nodes(p, schema, comp_idx)
    query, warnings = bgp.build_query(
        schema, all_nodes, excluded=p.excluded, excluded_joins=p.excluded_joins,
        include_object_properties=False, optional_data_properties=False, limit=limit,
    )
    return {"component": comp_idx, "sparql": query, "warnings": warnings}


@app.post("/api/projects/{pid}/materialise/cancel")
def materialize_cancel(pid: str):
    p = _get_project(pid)
    job = p.mat_job
    if job is None or job["state"] != "running":
        return _job_status(p)
    job["state"] = "cancelled"
    job["finished"] = time.time()
    return _job_status(p)


# --------------------------------------------------------------------------- #
# Causal discovery (Module 2 §4.1-4.3)
#
# All seven ALLOWED_METHODS run through runners.run_kg_discovery.run_algorithm.
# GES-Prior (an alias for GES-M-B-HL) is enabled: with prior matrices loaded it
# scores with StructureScoreWithPrior, and with none it degrades to the same
# search under the plain base score — a legitimate run that needs no LLM. See
# the /priors section for how M and B are obtained (W15).
#
# The discovery context (materialised frame + constraint over kept columns)
# is expensive to build (a flat join + drop-constant-columns pass), so it is
# cached per project and reused across runs as long as the node selection,
# constraint options, component and limit haven't changed. The join itself is
# shared with module 1's preview via _materialise (W18).
# Runs accumulate into an in-memory ledger (§4.3) that is recomputed fresh
# on every read rather than stored denormalised, so deleting a run can't
# leave it inconsistent. Any node/constraint change invalidates both the
# context and the ledger, since old runs would no longer align to the same
# column order.
# --------------------------------------------------------------------------- #
DISCOVERY_METHODS = list(ALLOWED_METHODS)
NEEDS_PRIORS = {"GES-Prior"}


class DiscoveryRunRequest(BaseModel):
    method: str
    constrained: bool = True
    alpha: Optional[float] = None
    component: Optional[int] = None
    limit: Optional[int] = None
    seed: Optional[int] = None


@app.get("/api/discovery/methods")
def list_discovery_methods():
    return [
        {
            "method": m,
            "continuous": m in CONTINUOUS,
            "disabled": False,
            "needs_priors": m in NEEDS_PRIORS,
            "reason": (
                "Scores with LLM-estimated association/direction priors when they are "
                "loaded; without them it runs the same search under the plain BIC score."
                if m in NEEDS_PRIORS else None
            ),
        }
        for m in DISCOVERY_METHODS
    ]


def _discovery_context_key(p: ProjectState, component: Optional[int], limit: int) -> tuple:
    opts = p.constraint_options
    return (
        component, limit, frozenset(p.excluded), frozenset(p.excluded_joins),
        opts["relation_direction"], opts["allow_epsilon"], opts["max_hops"],
    )


def _build_discovery_context(p: ProjectState, component: Optional[int], limit: Optional[int]) -> DiscoveryContext:
    schema = _require_schema(p)
    # None == the whole join, matching module 1's default (W21). Discovery reuses
    # whatever frame module 1 already materialised whenever the key still matches, so
    # this rarely runs a second query.
    lim = limit
    if lim is not None and lim <= 0:
        raise HTTPException(400, "limit must be a positive row count, or null for no limit.")

    key = _discovery_context_key(p, component, lim)
    if p.discovery_context is not None and p.discovery_context_key == key:
        return p.discovery_context

    comp_idx, comp_nodes = _resolve_component(p, schema, component)
    opts = p.constraint_options
    constraint = EdgeConstraint.from_schema(
        schema, comp_nodes,
        relation_direction=opts["relation_direction"],
        allow_epsilon=opts["allow_epsilon"],
        max_hops=opts["max_hops"],
    )
    # The same join module 1 previewed, if the selection and limit still match (W18).
    mat = _materialise(p, schema, comp_idx, lim, optional_dp=False)

    df = mat.df.reindex(columns=[n.name for n in comp_nodes])
    df_clean, dropped = drop_constant_columns(df, verbose=False)
    keep = [i for i, n in enumerate(comp_nodes) if n.name in df_clean.columns]
    if not keep:
        raise HTTPException(400, "Every retained column is constant after materialisation; nothing to run discovery on.")
    constraint_kept = constraint.subset(keep)
    nodes_kept = [comp_nodes[i] for i in keep]

    ctx = DiscoveryContext(
        schema=schema, nodes=comp_nodes, constraint=constraint, mat=mat,
        nodes_kept=nodes_kept, constraint_kept=constraint_kept,
        discrete_df=to_discrete_frame(df_clean), continuous_df=to_numeric_frame(df_clean),
        dropped_columns=dropped, source=(p.source or {}).get("name"),
    )
    p.discovery_context = ctx
    p.discovery_context_key = key
    p.discovery_runs = []  # a new context invalidates the ledger: columns no longer align
    p.next_run_id = 1
    # Hand-authored edges and the curated selection are keyed on the column *name*, not a
    # position, so they survive a reorder (Plan3 §4.3). Only edges whose endpoint no longer
    # exists are dropped, and the causal canvas layout is pruned to match.
    surviving = set(ctx.column_names)
    p.manual_edges = [(s, t) for s, t in p.manual_edges if s in surviving and t in surviving]
    p.selected_edges = [(s, t) for s, t in p.selected_edges if s in surviving and t in surviving]
    p.layouts["causal"] = {k: v for k, v in p.layouts["causal"].items() if k in surviving}
    _invalidate_model(p)
    return ctx


#: Which package actually implements each method, for the reproducibility
#: record. GES-Prior and the constrained NOTEARS are this repository's own
#: implementations under `algs/`, which has no version of its own — the repo
#: commit is the version, and the export cannot know it.
_METHOD_LIBRARY = {
    "GES": "castle", "PC": "castle", "NOTEARS": "castle",
    "DAGMA": "castle", "LiNGAM": "castle", "DAG-GNN": "castle",
}


def _algorithm_library(method: str) -> Optional[str]:
    """``'castle 1.0.4'``-style label for the package that ran ``method``."""
    package = _METHOD_LIBRARY.get(method)
    if package is None:
        return "algs (in-repo)"
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return f"{package} {version(package)}"
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return package


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001 — best-effort; torch is present in rdfenv but not load-bearing here
        pass


@app.get("/api/projects/{pid}/discovery/context")
def get_discovery_context(pid: str, component: Optional[int] = None, limit: Optional[int] = None):
    p = _get_project(pid)
    ctx = _build_discovery_context(p, component, limit)
    return {
        "columns": ctx.column_names,
        "dropped_columns": ctx.dropped_columns,
        "n_rows": int(ctx.discrete_df.shape[0]),
        "constraint_stats": ctx.constraint_kept.stats(),
    }


@app.post("/api/projects/{pid}/discovery/run")
def run_discovery(pid: str, body: DiscoveryRunRequest):
    p = _get_project(pid)
    _require_schema(p)
    if body.method not in DISCOVERY_METHODS:
        raise HTTPException(
            400,
            f"Unknown or unsupported method '{body.method}'. Available: {DISCOVERY_METHODS} "
            "(GES-Prior needs LLM priors, not implemented in this slice).",
        )
    ctx = _build_discovery_context(p, body.component, body.limit)
    seed = body.seed if body.seed is not None else (uuid.uuid4().int % (2**31))
    _seed_all(seed)
    # Priors are passed explicitly and var_meta/pair_meta are left None, so a run can
    # never trigger an LLM call as a side effect — estimation is its own explicit step.
    priors = _aligned_priors(p, ctx) if body.method in NEEDS_PRIORS else None
    try:
        adj, weights = run_algorithm(
            body.method, ctx, constrained=body.constrained, alpha=body.alpha,
            priors=priors, return_weights=True,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim, this is a user-facing algorithm error
        raise HTTPException(400, f"Discovery run failed: {exc}") from exc

    # The seed goes into the OCG's parameters, not only into the run record
    # below: a cw:DiscoveryRun is a prov:Activity, and PC/GES tie-breaking,
    # NOTEARS/DAGMA/DAG-GNN initialisation and any bootstrap are all stochastic,
    # so a run whose exported provenance omits the seed cannot be reproduced from
    # the KG. `library` is the other half of that — the same seed under a
    # different gcastle gives a different graph.
    ocg = to_ocg(
        adj, ctx, body.constrained, weights=weights, method=body.method,
        params={
            "alpha": body.alpha,
            "priors": p.priors_source if priors else None,
            "seed": seed,
            "library": _algorithm_library(body.method),
        },
        source=ctx.source,
    )
    run = {
        "run_id": p.next_run_id,
        "method": body.method,
        "constrained": body.constrained,
        "alpha": body.alpha,
        "priors": p.priors_source if priors else None,
        "seed": seed,
        "timestamp": time.time(),
        "columns": ctx.column_names,
        "adj": np.asarray(adj, dtype=int).tolist(),
        "weights": np.asarray(weights, dtype=float).tolist(),
        "n_edges": int(np.asarray(adj).sum()),
        "topological_validity": ocg.topological_validity(ctx.constraint_kept),
    }
    p.next_run_id += 1
    p.discovery_runs.append(run)
    return run


@app.get("/api/projects/{pid}/discovery/runs")
def list_discovery_runs(pid: str):
    p = _get_project(pid)
    return p.discovery_runs


@app.delete("/api/projects/{pid}/discovery/runs/{run_id}")
def delete_discovery_run(pid: str, run_id: int):
    p = _get_project(pid)
    before = len(p.discovery_runs)
    p.discovery_runs = [r for r in p.discovery_runs if r["run_id"] != run_id]
    if len(p.discovery_runs) == before:
        raise HTTPException(404, f"No run {run_id}")
    return {"ok": True}


@app.delete("/api/projects/{pid}/discovery/runs")
def clear_discovery_runs(pid: str):
    """Clear all runs (Plan3 W12).

    Every discovered edge leaves the canvas at once. Hand-authored edges are deliberately
    *not* touched: run output and user authorship are different kinds of thing and one
    button must not silently destroy the other. Selections that pointed at a now-vanished
    discovered edge are dropped, since selecting an edge nothing asserts is meaningless.
    """
    p = _get_project(pid)
    n = len(p.discovery_runs)
    p.discovery_runs = []
    p.next_run_id = 1
    manual = {tuple(e) for e in p.manual_edges}
    p.selected_edges = [e for e in p.selected_edges if tuple(e) in manual]
    _invalidate_model(p)
    return {"ok": True, "cleared": n}


# --------------------------------------------------------------------------- #
# The edge ledger and the curated causal graph (Module 2 §4.3-4.4, W12)
# --------------------------------------------------------------------------- #
def _ledger(p: ProjectState) -> dict:
    """Derived edge ledger — recomputed from `discovery_runs`, never stored denormalised.

    Directed, one entry per *oriented* edge rather than per undirected pair, because the
    canvas draws oriented edges with the frequency as the label and every one of them is
    independently selectable. The reverse orientation's count travels on the same record
    (`reverse_frequency` / `conflict`) so the Markov-equivalence case stays visible.
    """
    ctx = p.discovery_context
    if ctx is None:
        return {"columns": [], "n_runs": 0, "edges": []}
    names = list(ctx.column_names)
    index = {name: i for i, name in enumerate(names)}
    n = len(names)

    oriented: dict[tuple[int, int], int] = {}
    methods: dict[tuple[int, int], set] = {}
    for run in p.discovery_runs:
        adj = run["adj"]
        for i in range(n):
            for j in range(n):
                if i != j and adj[i][j]:
                    oriented[(i, j)] = oriented.get((i, j), 0) + 1
                    methods.setdefault((i, j), set()).add(run["method"])

    manual = {(index[s], index[t]) for s, t in p.manual_edges if s in index and t in index}
    selected = {(index[s], index[t]) for s, t in p.selected_edges if s in index and t in index}

    # The curated graph must stay a DAG at every intermediate state, not only when the
    # user finishes (§4.4, W19). An unselected edge s -> t would close a cycle exactly
    # when t already reaches s through the current selection, so that reachability is
    # computed once here and every edge carries the verdict.
    sel_graph = nx.DiGraph()
    sel_graph.add_nodes_from(names)
    sel_graph.add_edges_from((names[i], names[j]) for i, j in selected)

    def _closes_cycle(i: int, j: int) -> Optional[list[str]]:
        if (i, j) in selected:
            return None
        try:
            back = nx.shortest_path(sel_graph, names[j], names[i])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return [names[i]] + back

    edges = []
    for (i, j) in sorted(set(oriented) | manual | selected,
                         key=lambda k: (-oriented.get(k, 0), names[k[0]], names[k[1]])):
        freq = oriented.get((i, j), 0)
        rev = oriented.get((j, i), 0)
        cycle = _closes_cycle(i, j)
        edges.append({
            "source": names[i],
            "target": names[j],
            "frequency": freq,
            "reverse_frequency": rev,
            "adjacency_frequency": freq + rev,
            "conflict": freq > 0 and rev > 0,
            "methods": sorted(methods.get((i, j), ())),
            "relation_label": ctx.constraint_kept.label_str(i, j),
            "topologically_valid": bool(ctx.constraint_kept.allowed[i, j]),
            "manual": (i, j) in manual,
            "selected": (i, j) in selected,
            "blocked": cycle is not None,
            "blocked_path": cycle,
        })
    return {
        "columns": names,
        "n_runs": len(p.discovery_runs),
        "n_selected": len(selected),
        "edges": edges,
    }


@app.get("/api/projects/{pid}/discovery/ledger")
def get_discovery_ledger(pid: str):
    return _ledger(_get_project(pid))


def _require_columns(p: ProjectState) -> list[str]:
    if p.discovery_context is None:
        raise HTTPException(
            400,
            "No discovery context yet — run a method (or open the discovery module) first, "
            "so the column set an edge can refer to exists.",
        )
    return list(p.discovery_context.column_names)


def _find_cycle(edges) -> Optional[list[str]]:
    """Return a cycle as a node path, or None. Names the cycle for the 422 (Plan3 §4.4)."""
    g = nx.DiGraph()
    g.add_edges_from((s, t) for s, t in edges)
    try:
        cyc = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return None
    path = [u for u, _v, _k in cyc]
    return path + [path[0]]


class EdgeRef(BaseModel):
    source: str
    target: str


class SelectionUpdate(BaseModel):
    edges: list[list[str]]


@app.post("/api/projects/{pid}/graph/edges")
def add_manual_edge(pid: str, body: EdgeRef):
    """Add one hand-authored edge. It enters the ledger at frequency 0 (Plan3 W12)."""
    p = _get_project(pid)
    columns = _require_columns(p)
    for name in (body.source, body.target):
        if name not in columns:
            raise HTTPException(400, f"'{name}' is not a variable in the current column set.")
    if body.source == body.target:
        raise HTTPException(400, "An edge cannot start and end at the same variable.")
    pair = (body.source, body.target)
    if pair in {tuple(e) for e in p.manual_edges}:
        raise HTTPException(400, f"{body.source} → {body.target} is already hand-authored.")
    ledger = _ledger(p)
    for e in ledger["edges"]:
        if (e["source"], e["target"]) == pair and e["frequency"] > 0:
            raise HTTPException(
                400,
                f"{body.source} → {body.target} is already in the ledger at frequency "
                f"{e['frequency']} — select it rather than authoring it.",
            )
    p.manual_edges.append(pair)
    return _ledger(p)


@app.delete("/api/projects/{pid}/graph/edges")
def delete_manual_edge(pid: str, source: str, target: str):
    p = _get_project(pid)
    pair = (source, target)
    before = len(p.manual_edges)
    p.manual_edges = [e for e in p.manual_edges if tuple(e) != pair]
    if len(p.manual_edges) == before:
        raise HTTPException(404, f"No hand-authored edge {source} → {target}")
    p.selected_edges = [e for e in p.selected_edges if tuple(e) != pair]
    _invalidate_model(p)
    return _ledger(p)


@app.delete("/api/projects/{pid}/graph/edges/all")
def clear_manual_edges(pid: str):
    p = _get_project(pid)
    n = len(p.manual_edges)
    manual = {tuple(e) for e in p.manual_edges}
    p.manual_edges = []
    p.selected_edges = [e for e in p.selected_edges if tuple(e) not in manual]
    _invalidate_model(p)
    return {"ok": True, "cleared": n, "ledger": _ledger(p)}


@app.put("/api/projects/{pid}/graph/selection")
def set_selection(pid: str, body: SelectionUpdate):
    """Set the curated causal graph. 422 names the cycle rather than storing an unfittable graph."""
    p = _get_project(pid)
    columns = set(_require_columns(p))
    pairs: list[tuple[str, str]] = []
    for e in body.edges:
        if len(e) != 2:
            raise HTTPException(400, f"Malformed edge {e!r}; expected [source, target].")
        s, t = e
        if s not in columns or t not in columns:
            raise HTTPException(400, f"Edge {s} → {t} refers to a variable outside the column set.")
        if s == t:
            raise HTTPException(400, f"Self-loop on {s} is not a causal edge.")
        if (s, t) not in pairs:
            pairs.append((s, t))

    cycle = _find_cycle(pairs)
    if cycle is not None:
        raise HTTPException(422, detail={
            "error": "cycle",
            "path": cycle,
            "message": "Rejected: that selection closes a cycle: " + " → ".join(cycle),
        })
    if pairs != p.selected_edges:
        _invalidate_model(p)
    p.selected_edges = pairs
    return _ledger(p)


class AutoSelect(BaseModel):
    min_frequency: int = 1
    include_manual: bool = False
    allow_invalid: bool = False
    # Keep at most the k most-agreed-on edges. `k` counts edges *kept*, not edges
    # considered — a candidate skipped for closing a cycle does not spend a slot,
    # otherwise "top 10" would silently deliver 7.
    top_k: Optional[int] = None


@app.post("/api/projects/{pid}/graph/selection/auto")
def auto_select(pid: str, body: AutoSelect):
    """Select the strongest acyclic subset of the ledger (W19).

    The old behaviour was a *filter*: take every edge that is valid and was found at
    least once. That is wrong, and reliably so — each run returns a DAG, but a union of
    DAGs is not one. Three runs can contribute A→B, B→C and C→A, each individually
    unobjectionable, and the union is a cycle the selection endpoint then rejects
    wholesale, leaving the user with nothing selected and a message about an edge they
    never chose.

    So it is a *construction* instead: walk the candidates in descending oriented
    frequency and keep each one only if it does not close a cycle with what is already
    kept. Frequency order means the edges the most runs agree on win the conflicts, and
    a reciprocal pair resolves to its majority orientation for free — the minority
    direction is simply the second one seen, and it closes a 2-cycle. Every skip is
    reported, because "11 selected, 2 skipped" is information about method disagreement,
    not noise to swallow.
    """
    p = _get_project(pid)
    _require_columns(p)
    ledger = _ledger(p)
    candidates = [
        e for e in ledger["edges"]
        if (e["topologically_valid"] or body.allow_invalid)
        and (e["frequency"] >= body.min_frequency or (body.include_manual and e["manual"]))
    ]
    candidates.sort(key=lambda e: (-e["frequency"], -e["adjacency_frequency"],
                                   e["source"], e["target"]))

    if body.top_k is not None and body.top_k < 1:
        raise HTTPException(400, "top_k must be at least 1.")

    g = nx.DiGraph()
    kept: list[tuple[str, str]] = []
    skipped: list[dict] = []
    cut_off = 0
    for e in candidates:
        s, t = e["source"], e["target"]
        if s == t:
            continue
        if body.top_k is not None and len(kept) >= body.top_k:
            cut_off += 1
            continue
        if g.has_node(t) and g.has_node(s) and nx.has_path(g, t, s):
            skipped.append({
                "source": s, "target": t, "frequency": e["frequency"],
                "path": [s] + nx.shortest_path(g, t, s),
            })
            continue
        g.add_edge(s, t)
        kept.append((s, t))

    if kept != p.selected_edges:
        _invalidate_model(p)
    p.selected_edges = kept
    return {
        "ledger": _ledger(p), "n_selected": len(kept), "skipped": skipped,
        "n_candidates": len(candidates), "cut_off": cut_off,
        "min_kept_frequency": min((e["frequency"] for e in ledger["edges"]
                                   if (e["source"], e["target"]) in set(kept)), default=None),
    }


def _curated_graph(p: ProjectState) -> dict:
    ctx = p.discovery_context
    names = list(ctx.column_names) if ctx else []
    index = {name: i for i, name in enumerate(names)}
    pairs = [(s, t) for s, t in p.selected_edges if s in index and t in index]
    validity = None
    if pairs and ctx is not None:
        allowed = sum(1 for s, t in pairs if ctx.constraint_kept.allowed[index[s], index[t]])
        validity = allowed / len(pairs)
    used = sorted({n for pair in pairs for n in pair})
    return {
        "columns": names,
        "nodes": used,
        "edges": [
            {
                "source": s,
                "target": t,
                "manual": (s, t) in {tuple(e) for e in p.manual_edges},
                "relation_label": ctx.constraint_kept.label_str(index[s], index[t]) if ctx else "",
                "topologically_valid": bool(ctx.constraint_kept.allowed[index[s], index[t]]) if ctx else None,
            }
            for s, t in pairs
        ],
        "n_edges": len(pairs),
        "n_nodes": len(used),
        "acyclic": _find_cycle(pairs) is None,
        "topological_validity": validity,
    }


@app.get("/api/projects/{pid}/graph")
def get_curated_graph(pid: str):
    return _curated_graph(_get_project(pid))


# --------------------------------------------------------------------------- #
# GES-Prior metadata and priors (Module 2 §4.5, W15)
#
# GES-Prior scores with two matrices: M (how likely a pair is causally related at
# all) and B (which direction). They come from an LLM asked, per admissible pair,
# what it knows about those two variables — so the quality of the answer is the
# quality of the *description* each variable carries. Three sources of description,
# in increasing precedence:
#
#   1. the T-Box itself — rdfs:label / rdfs:comment, via causalway.llm_meta. Free,
#      always available, and usually the best thing about a well-annotated KG.
#   2. an external SPARQL endpoint (Wikidata by default) that some columns have been
#      mapped onto by QID: causalway.kg_endpoint_meta summarises each mapped entity's
#      1-hop neighbourhood and each mapped pair's relational path. Partial mapping is
#      the normal case — unmapped columns just keep their T-Box text.
#   3. text the user writes for a specific variable, which wins over both.
#
# The API key is *only* ever read from the server's environment. This service is
# sign-in free and holds no user data (W14), which makes it precisely the wrong place
# to accept somebody's API key through a form — so there is no field for one. When no
# key is configured the estimate step is unavailable and says so, GES-Prior still runs
# without priors, and a priors JSON estimated offline through runners/ can be uploaded.
# --------------------------------------------------------------------------- #
LLM_KEY_ENV = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_KEY",
    "KIMI_API_KEY", "KIMI_KEY", "GLM_API_KEY", "GLM_KEY", "LLM_API_KEY",
)
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
MAX_PRIOR_PAIRS = 300  # one LLM round-trip each; a runaway sweep is a real cost


def _llm_key_env() -> Optional[str]:
    """Which env var supplies the LLM key, if any. Never returns the value."""
    try:  # importing the client also loads any .env the repo ships, same as the CLI path
        import algs.ges_prior.llm_call  # noqa: F401
    except Exception:  # noqa: BLE001 — absence is an answer, not an error
        pass
    return next((k for k in LLM_KEY_ENV if os.environ.get(k)), None)


def _aligned_priors(p: ProjectState, ctx: DiscoveryContext) -> Optional[dict]:
    """Re-index stored priors onto the context's current column order.

    Priors are stored with the column list they were estimated for. A later curation
    change reorders or drops columns, and silently feeding a stale matrix to the score
    would mislabel every pair. Columns the priors don't cover fall back to the neutral
    0.5 the estimator itself uses for a pair it could not answer.
    """
    if not p.priors:
        return None
    names = list(ctx.column_names)
    old = list(p.priors["columns"])
    pos = {name: i for i, name in enumerate(old)}
    n = len(names)
    out = {}
    for key in ("M", "B"):
        src = np.asarray(p.priors[key], dtype=float)
        dst = np.full((n, n), 0.5, dtype=float)
        for a, na in enumerate(names):
            for b, nb in enumerate(names):
                ia, ib = pos.get(na), pos.get(nb)
                if ia is not None and ib is not None:
                    dst[a, b] = src[ia, ib]
        out[key] = dst
    return out


class PriorMetaUpdate(BaseModel):
    domain: Optional[str] = None
    var_text: Optional[dict[str, str]] = None
    entity_map: Optional[dict[str, str]] = None
    endpoint: Optional[str] = None
    llm_model: Optional[str] = None


class PriorEstimate(BaseModel):
    use_kg: bool = False           # enrich descriptions from the external endpoint
    llm_model: Optional[str] = None
    max_workers: int = 8


@app.get("/api/projects/{pid}/priors")
def get_priors(pid: str):
    """Per-variable metadata, the pair budget, and whether priors are loaded."""
    p = _get_project(pid)
    ctx = p.discovery_context
    names = list(ctx.column_names) if ctx else []
    schema_text: dict[str, str] = {}
    n_pairs = 0
    if ctx is not None:
        schema_text = build_var_meta(p.schema, ctx.nodes_kept)
        allowed = ctx.constraint_kept.allowed
        n_pairs = sum(
            1
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if allowed[i, j] or allowed[j, i]
        )
    key_env = _llm_key_env()
    return {
        "columns": names,
        "domain": p.prior_meta["domain"],
        "domain_default": build_domain_str(p.schema) if p.schema else "",
        "endpoint": p.prior_meta["endpoint"],
        "llm_model": p.prior_meta["llm_model"] or DEFAULT_LLM_MODEL,
        "variables": [
            {
                "column": name,
                "schema_text": schema_text.get(name, ""),
                "user_text": p.prior_meta["var_text"].get(name, ""),
                "entity_id": p.prior_meta["entity_map"].get(name, ""),
            }
            for name in names
        ],
        "n_pairs": n_pairs,
        "max_pairs": MAX_PRIOR_PAIRS,
        "llm_configured": key_env is not None,
        "llm_key_env": key_env,
        "loaded": p.priors is not None,
        "source": p.priors_source,
        "n_informative": _n_informative(p),
    }


def _n_informative(p: ProjectState) -> int:
    """Off-diagonal M entries the estimator actually answered (i.e. moved off 0.5)."""
    if not p.priors:
        return 0
    m = np.asarray(p.priors["M"], dtype=float)
    off = ~np.eye(m.shape[0], dtype=bool)
    return int(np.count_nonzero(np.abs(m[off] - 0.5) > 1e-9))


@app.put("/api/projects/{pid}/priors/meta")
def put_prior_meta(pid: str, body: PriorMetaUpdate):
    p = _get_project(pid)
    columns = set(_require_columns(p))
    if body.domain is not None:
        p.prior_meta["domain"] = body.domain
    if body.endpoint is not None:
        p.prior_meta["endpoint"] = body.endpoint
    if body.llm_model is not None:
        p.prior_meta["llm_model"] = body.llm_model
    for field, value in (("var_text", body.var_text), ("entity_map", body.entity_map)):
        if value is None:
            continue
        unknown = set(value) - columns
        if unknown:
            raise HTTPException(400, f"Not variables in the current column set: {sorted(unknown)}")
        p.prior_meta[field] = {k: v for k, v in value.items() if v.strip()}
    return get_priors(pid)


@app.post("/api/projects/{pid}/priors/estimate")
def estimate_project_priors(pid: str, body: PriorEstimate):
    """Ask the LLM for M and B over every admissible pair. Synchronous and slow."""
    p = _get_project(pid)
    ctx = _build_discovery_context(p, None, None) if p.discovery_context is None else p.discovery_context
    key_env = _llm_key_env()
    if key_env is None:
        raise HTTPException(400, {
            "error": "no_llm_key",
            "message": (
                "No LLM API key is configured on this server, and this service will not "
                "accept one through the browser — it is sign-in free and holds no user "
                "data, so a pasted key would be the only secret in the building. Set one "
                f"of {', '.join(LLM_KEY_ENV[:4])}… in the server's environment, or estimate "
                "priors offline with runners/run_kg_discovery.estimate_and_cache_priors and "
                "upload the JSON."
            ),
        })

    names = list(ctx.column_names)
    allowed = ctx.constraint_kept.allowed
    n_pairs = sum(1 for i in range(len(names)) for j in range(i + 1, len(names))
                  if allowed[i, j] or allowed[j, i])
    if n_pairs > MAX_PRIOR_PAIRS:
        raise HTTPException(400, (
            f"{n_pairs} admissible pairs is {n_pairs - MAX_PRIOR_PAIRS} over the "
            f"{MAX_PRIOR_PAIRS}-pair cap for a synchronous request — each pair is one LLM "
            "round-trip. Exclude variables in module 1, or tighten the constraint, or run "
            "the estimate offline through runners/."
        ))

    var_meta = build_var_meta(p.schema, ctx.nodes_kept)
    pair_meta = build_pair_meta(p.schema, ctx.nodes_kept, ctx.constraint_kept)
    model_name = body.llm_model or p.prior_meta["llm_model"] or DEFAULT_LLM_MODEL

    from algs.ges_prior.learn_bn import estimate_priors  # noqa: PLC0415 — heavy, and optional
    from algs.ges_prior.llm_call import LLMClient  # noqa: PLC0415

    client = LLMClient(model=model_name)
    kg_errors: list[str] = []
    entity_map = {k: v for k, v in p.prior_meta["entity_map"].items() if k in set(names)}
    if body.use_kg and entity_map:
        endpoint = p.prior_meta["endpoint"]
        from causalway.kg_endpoint_meta import (  # noqa: PLC0415
            WIKIDATA_ENDPOINT, build_pair_meta_from_kg, build_var_meta_from_kg,
        )
        endpoint = endpoint or WIKIDATA_ENDPOINT
        try:
            var_meta.update({
                k: v for k, v in build_var_meta_from_kg(
                    entity_map, endpoint=endpoint, llm_client=client).items() if v
            })
            mapped_pairs = [
                (names[i], names[j])
                for i in range(len(names)) for j in range(i + 1, len(names))
                if names[i] in entity_map and names[j] in entity_map
                and (allowed[i, j] or allowed[j, i])
            ]
            pair_meta.update({
                k: v for k, v in build_pair_meta_from_kg(
                    entity_map, endpoint=endpoint, llm_client=client,
                    pairs=mapped_pairs).items() if v
            })
        except Exception as exc:  # noqa: BLE001 — a flaky public endpoint must not lose the run
            kg_errors.append(f"{type(exc).__name__}: {exc}")

    # User text wins over both the T-Box and the endpoint: it is the most specific
    # statement of what the variable means, and it is the only one a human authored.
    var_meta.update({k: v for k, v in p.prior_meta["var_text"].items() if k in set(names)})
    domain = p.prior_meta["domain"] or build_domain_str(p.schema)

    started = time.time()
    try:
        priors = estimate_priors(
            ctx.discrete_df, var_meta=var_meta, pair_meta=pair_meta, domain=domain,
            llm_client=client, max_workers=max(1, min(16, body.max_workers)),
            allowed=allowed,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim
        raise HTTPException(400, f"Prior estimation failed: {exc}") from exc
    if priors is None:
        raise HTTPException(400, (
            "The LLM answered no pair usefully, so there are no priors to store. "
            "Check the model name and the server's key, then try again."
        ))

    p.priors = {
        "columns": names,
        "M": np.asarray(priors["M"], dtype=float).tolist(),
        "B": np.asarray(priors["B"], dtype=float).tolist(),
    }
    p.priors_source = "llm"
    out = get_priors(pid)
    out["elapsed"] = round(time.time() - started, 1)
    out["n_pairs_queried"] = n_pairs
    out["kg_errors"] = kg_errors
    out["model"] = model_name
    return out


@app.post("/api/projects/{pid}/priors/import")
async def import_priors(pid: str, file: UploadFile = File(...)):
    """Load a priors JSON estimated elsewhere — the offline path, and the export's inverse."""
    p = _get_project(pid)
    _require_columns(p)
    try:
        data = json.loads(await file.read())
        columns = [str(c) for c in data["columns"]]
        m = np.asarray(data["M"], dtype=float)
        b = np.asarray(data["B"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Not a priors JSON ({{columns, M, B}}): {exc}") from exc
    n = len(columns)
    if m.shape != (n, n) or b.shape != (n, n):
        raise HTTPException(400, f"M/B must both be {n}×{n} for {n} columns.")
    # An offline estimate stores NaN for pairs it skipped; the score wants a number.
    p.priors = {
        "columns": columns,
        "M": np.nan_to_num(m, nan=0.5).tolist(),
        "B": np.nan_to_num(b, nan=0.5).tolist(),
    }
    p.priors_source = "imported"
    overlap = len(set(columns) & set(_require_columns(p)))
    out = get_priors(pid)
    out["n_columns_matched"] = overlap
    return out


@app.delete("/api/projects/{pid}/priors")
def clear_priors(pid: str):
    p = _get_project(pid)
    p.priors = None
    p.priors_source = None
    return get_priors(pid)


# --------------------------------------------------------------------------- #
# Canvas layout (W13) — positions are project state, not browser state
# --------------------------------------------------------------------------- #
CANVASES = ("ontology", "causal", "model", "inference", "counterfactual")


class LayoutUpdate(BaseModel):
    canvas: str
    positions: dict[str, dict]


@app.get("/api/projects/{pid}/layout")
def get_layout(pid: str):
    return _get_project(pid).layouts


@app.put("/api/projects/{pid}/layout")
def put_layout(pid: str, body: LayoutUpdate):
    p = _get_project(pid)
    if body.canvas not in CANVASES:
        raise HTTPException(400, f"Unknown canvas '{body.canvas}'; expected one of {CANVASES}.")
    cleaned = {}
    for key, pos in body.positions.items():
        try:
            cleaned[key] = {"x": float(pos["x"]), "y": float(pos["y"])}
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"Bad position for '{key}': {pos!r}") from exc
    p.layouts[body.canvas] = cleaned
    return {"ok": True, "canvas": body.canvas, "n": len(cleaned)}


@app.delete("/api/projects/{pid}/layout/{canvas}")
def reset_layout(pid: str, canvas: str):
    p = _get_project(pid)
    if canvas not in CANVASES:
        raise HTTPException(400, f"Unknown canvas '{canvas}'; expected one of {CANVASES}.")
    p.layouts[canvas] = {}
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Module 3 — fit the causal model (Plan 3 §5, Plan 2 §3.3)
#
# `CausalModel.fit` re-materialises the KG from the *source* rather than taking our
# frame, because it needs `entity_ids` and the schema to align OCG nodes onto KG nodes
# by (domain, prop, range) identity. That is why the project keeps `source_ref`.
#
# The fit is keyed on everything that could change its meaning; re-posting an unchanged
# request returns the cached model rather than refitting, and any edit to the curated
# graph upstream drops it (`_invalidate_model`).
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# W22 — two model kinds, two fits
#
# "Causal Bayesian network" and "structural causal model" are different objects,
# not two names for one, and the difference decides what you may ask:
#
#   cbn  A DiscreteBayesianNetwork whose mechanism at each node is a conditional
#        probability table estimated from the frame. Exact inference (variable
#        elimination) for both P(y|x) and P(y|do(x)); the CPT is directly readable,
#        which is the point of choosing it. Every variable must be discrete — a CPT
#        over a continuous parent does not exist — so the choice is refused, not
#        silently discretised, when any column is continuous.
#        No counterfactuals: a CPT has no exogenous noise term to abduct, so
#        "what would have happened to *this* unit" is not defined over it.
#
#   scm  gcm's InvertibleStructuralCausalModel: a fitted function plus a noise
#        distribution per node. Handles continuous variables, and its invertible
#        mechanisms are exactly what module 4 abducts to answer counterfactuals.
#        Interventions are sampled from the mutilated model rather than solved.
#
# The CBN path still goes through `CausalModel.fit(fit_mechanisms=False)` because
# that is what re-materialises the KG and aligns the curated graph onto its columns
# — but it skips gcm's model selection and fit entirely, which is where the ~30 s
# goes. What answers a CBN query is the CPT, and nothing else.
# --------------------------------------------------------------------------- #
MODEL_KINDS = ("scm", "cbn")


class FitRequest(BaseModel):
    kind: str = "scm"
    ordinal: list[str] = []
    limit: Optional[int] = None
    on_missing: str = "drop"
    random_state: Optional[int] = 7


def _require_model(p: ProjectState):
    if p.model is None:
        raise HTTPException(400, (
            "No fitted model in this project. Select a causal graph in module 2, then "
            "fit it in module 3 — every answer below depends on the mechanisms."
        ))
    return p.model


def _py(v):
    """numpy/pandas scalar -> something json can carry."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.ndarray,)):
        return [_py(x) for x in v.tolist()]
    return v


def _cbn_mechanism_rows(p: ProjectState) -> list[dict]:
    """A CBN's 'mechanism' is its CPT; report its shape rather than a gcm class name."""
    graph = p.model.scm.graph
    rows = []
    for col in p.model.spec.columns:
        parents = sorted(graph.predecessors(col))
        try:
            cpd = p.cbn.get_cpds(col)
            n_params = int(np.prod(cpd.cardinality)) if cpd is not None else 0
        except Exception:  # noqa: BLE001
            n_params = 0
        rows.append({
            "node": col,
            "dtype": p.model.spec.dtypes.get(col),
            "is_root": not parents,
            "mechanism_type": (
                f"CPT · {n_params} parameters" if parents else
                f"marginal table · {n_params} parameters"
            ),
            # A CPT has no exogenous noise term, so there is nothing to abduct from it.
            "invertible": False,
        })
    return rows


def _model_payload(p: ProjectState) -> dict:
    model = p.model
    spec = model.spec
    graph = model.scm.graph
    data = spec.data
    mech = (_cbn_mechanism_rows(p) if p.model_kind == "cbn"
            else model.mechanism_table().to_dict(orient="records"))
    by_node = {r["node"]: r for r in mech}

    # W29: an imported model has mechanisms and no rows. Its manifest carries
    # `category_levels` / `numeric_ranges` precisely so the board can still be
    # drawn and the hypothetical-unit form still offer the levels the model was
    # fitted on — the summary statistics of a column travel with the model, the
    # column itself does not.
    levels: dict[str, list] = {}
    ranges: dict[str, dict] = {}
    manifest_levels = model.manifest.get("category_levels") or {}
    manifest_ranges = model.manifest.get("numeric_ranges") or {}
    for col in spec.columns:
        dtype = spec.dtypes.get(col)
        series = None if data is None else data[col]
        if dtype in ("categorical", "ordinal"):
            if series is not None:
                vals = sorted(str(v) for v in pd.unique(series.dropna()))
            else:
                vals = sorted(str(v) for v in manifest_levels.get(col, []))
            levels[col] = vals[:60]
        elif series is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            ranges[col] = {
                "min": _py(numeric.min()), "max": _py(numeric.max()),
                "mean": _py(numeric.mean()),
            }
        elif col in manifest_ranges:
            ranges[col] = {k: _py(v) for k, v in manifest_ranges[col].items()}

    from causalway.inference import _all_discrete  # noqa: PLC0415 — one private probe, deliberate

    return {
        "model_id": model.model_id,
        "kind": p.model_kind,
        "kind_label": ("Causal Bayesian network" if p.model_kind == "cbn"
                       else "Structural causal model"),
        # Only an SCM has an exogenous noise term per node, and abduction is what makes
        # module 4 a counterfactual rather than a second intervention (W22).
        "supports_counterfactual": p.model_kind == "scm",
        "fitted_at": time.time(),
        # The seed the mechanisms' own generators were spawned from. Read back
        # off the manifest rather than remembered separately, so it cannot drift
        # from what the model actually used, and `null` when the fit was
        # unseeded — a counterfactual from an unseeded model is not reproducible
        # and says so rather than implying otherwise.
        "random_state": model.manifest.get("random_state"),
        # Rows behind the *live* frame, which an imported model does not have.
        # `training_rows` is what it was fitted on and is recorded in the
        # manifest; reporting that as `n_rows` would claim rows are present.
        "n_rows": int(data.shape[0]) if data is not None else 0,
        "training_rows": model.manifest.get("training_rows"),
        "has_rows": data is not None,
        "columns": list(spec.columns),
        "dtypes": dict(spec.dtypes),
        "alignment": spec.alignment,
        "edges": [{"source": s, "target": t} for s, t in graph.edges()],
        # The family gate (§5.2): an all-categorical model can also be answered by an
        # exact DiscreteBayesianNetwork; anything continuous cannot, and falls to
        # sampling. Reported because it decides which backends inference can reach.
        "all_discrete": bool(_all_discrete(model)),
        "mechanisms": [
            {
                "node": r["node"],
                "dtype": r["dtype"],
                "is_root": bool(r["is_root"]),
                "mechanism_type": r["mechanism_type"],
                "invertible": bool(r["invertible"]),
                "parents": sorted(graph.predecessors(r["node"])),
                "n_levels": len(levels.get(r["node"], [])) or None,
            }
            for r in mech
        ],
        "roots": [r["node"] for r in mech if r["is_root"]],
        "non_invertible": [r["node"] for r in mech if not r["invertible"]],
        "levels": levels,
        "ranges": ranges,
        "by_node": {k: {"dtype": v["dtype"], "is_root": bool(v["is_root"])} for k, v in by_node.items()},
    }


@app.post("/api/projects/{pid}/model/fit")
def fit_model(pid: str, body: FitRequest):
    p = _get_project(pid)
    _require_schema(p)
    if not p.selected_edges:
        raise HTTPException(400, (
            "Nothing selected in module 2. The curated selection *is* the graph this "
            "module fits, so an empty selection has no mechanisms to learn."
        ))
    if not p.source_ref:
        raise HTTPException(400, "This project has no resolvable source to re-materialise.")
    graph = _curated_graph(p)
    if not graph["acyclic"]:
        raise HTTPException(422, detail={
            "error": "cycle", "path": _find_cycle(p.selected_edges) or [],
            "message": "The curated graph is cyclic; an SCM needs a DAG.",
        })

    if body.kind not in MODEL_KINDS:
        raise HTTPException(400, f"kind must be one of {MODEL_KINDS}")

    # The threshold that decides categorical-vs-numeric lives in module 1 now, because
    # it is a statement about the data, not about this fit (W20).
    max_levels = int(p.type_options["max_levels"])
    float_as_continuous = bool(p.type_options["float_as_continuous"])
    limit = body.limit if body.limit is not None else _last_limit(p)

    key = (tuple(sorted(p.selected_edges)), body.kind, tuple(sorted(body.ordinal)),
           limit, max_levels, float_as_continuous, body.on_missing, body.random_state,
           frozenset(p.excluded), frozenset(p.excluded_joins))
    if p.model is not None and p.model_key == key:
        return p.model_info

    from causalway.model import CausalModel  # noqa: PLC0415 — pulls dowhy/pgmpy, ~seconds

    ocg = _curated_ocg(p)
    started = time.time()
    try:
        model = CausalModel.fit(
            ocg, p.source_ref,
            quality="good",
            ordinal=body.ordinal or None,
            on_missing=body.on_missing,
            # The per-node value path (Plan 3 §Preprocess revision 2), not the legacy
            # global one: `include_object_properties=False` is what makes a *retained*
            # object property a column holding the related entity's identity.
            include_object_properties=False,
            # The whole point — module 1's curation, verbatim, so this fit runs the
            # *same* SPARQL the user watched materialise. Both axes, because they are
            # independent: `excluded` decides which properties are columns,
            # `excluded_joins` decides which relationships are joined, and it is the
            # joins that decide the rows. Without these the fit re-materialised the
            # full schema and could learn mechanisms from a row population the
            # discovered structure was never seen against — and resolve module 4's
            # "this entity's rows" against entity_ids from a join the user never ran.
            excluded=set(p.excluded),
            excluded_joins=set(p.excluded_joins),
            limit=limit,
            max_levels=max_levels,
            float_as_continuous=float_as_continuous,
            # A CBN estimates its own CPTs and never reads a gcm mechanism, so paying
            # for gcm.auto's per-node model selection would be ~30 s of pure waste (W22).
            fit_mechanisms=(body.kind == "scm"),
            random_state=body.random_state,
        )
    except Exception as exc:  # noqa: BLE001 — fit errors are the user's to read verbatim
        raise HTTPException(400, f"Fit failed: {exc}") from exc

    if body.kind == "cbn":
        continuous = [c for c in model.spec.columns
                      if model.spec.dtypes.get(c) not in ("categorical", "ordinal", "discrete")]
        if continuous:
            raise HTTPException(400, (
                "A causal Bayesian network needs a conditional probability table at every "
                f"node, and {', '.join(continuous)} " +
                ("is" if len(continuous) == 1 else "are") + " continuous — there is no CPT "
                "over a continuous parent. Either fit a structural causal model instead, or "
                f"raise 'max levels' in module 1 above the distinct-value count of "
                f"{'that column' if len(continuous) == 1 else 'those columns'} so "
                f"{'it is' if len(continuous) == 1 else 'they are'} read as labels."
            ))
        try:
            p.cbn, p.cbn_infer = _fit_cbn(model)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"CPT estimation failed: {exc}") from exc
    else:
        p.cbn = p.cbn_infer = None

    p.model = model
    p.model_kind = body.kind
    p.model_key = key
    p.evaluation = None
    p.answers = []
    p.next_answer_id = 1
    p.cf_worlds = []
    p.next_world_id = 1
    info = _model_payload(p)
    info["elapsed"] = round(time.time() - started, 1)
    p.model_info = info
    return info


def _last_limit(p: ProjectState) -> Optional[int]:
    """Whatever module 1's join was actually run with — the fit must see the same frame."""
    if p.mat_job is not None and p.mat_job["state"] == "done":
        return p.mat_job["limit"]
    return None


def _fit_cbn(model):
    """Estimate a CPT per node with a Bayesian (BDeu) estimator, and an exact engine.

    BDeu rather than plain MLE because a flat join routinely leaves some parent
    configurations with a handful of rows or none at all, and a maximum-likelihood CPT
    answers those cells with 0 or NaN — a zero probability that no amount of evidence
    can ever revise.
    """
    from pgmpy.estimators import BayesianEstimator  # noqa: PLC0415
    from pgmpy.inference import CausalInference  # noqa: PLC0415
    from pgmpy.models import DiscreteBayesianNetwork  # noqa: PLC0415

    frame = model.spec.data[model.spec.columns].astype(str)
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(list(model.spec.columns))
    bn.add_edges_from(list(model.scm.graph.edges()))
    bn.fit(frame, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
    return bn, CausalInference(bn)


@app.get("/api/projects/{pid}/model")
def get_model(pid: str):
    p = _get_project(pid)
    if p.model is None:
        return {"fitted": False}
    return {"fitted": True, **(p.model_info or _model_payload(p))}


@app.delete("/api/projects/{pid}/model")
def drop_model(pid: str):
    _invalidate_model(_get_project(pid))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# W25 — what "is this causal model any good?" can actually mean
#
# A fitted model always produces numbers, so "did it run" is not a quality signal.
# Four separable questions are, and they fail independently — a model can predict
# beautifully through a graph that is causally wrong, and a graph can survive every
# independence test while its mechanisms learn nothing:
#
#   structure     Does the DAG survive its own conditional-independence implications?
#                 `gcm.falsify.falsify_graph` permutes the graph and asks whether the
#                 real one violates fewer CIs than a random relabelling would. A graph
#                 that is *not* rejected is not thereby confirmed — it survived.
#   mechanisms    Held-out k-fold CV of each node from its parents, against predicting
#                 that node's own marginal. Below the baseline means the parents carry
#                 no information about the child, and every interventional answer
#                 about that node is noise wearing a decimal point.
#   calibration   How close the model's own generated joint is to the observed data
#                 (gcm's KL/crossvalidation report), and, for a CBN, the log-likelihood
#                 and BIC of the fitted CPTs — BIC because a CPT with more parameters
#                 than rows to estimate them will always fit better and mean less.
#   abduction     Per-node invertibility p-values. Only these decide whether module 4
#                 is entitled to run at all; a non-invertible node has no noise to hold
#                 fixed, so its "counterfactual" is an intervention with a nicer name.
#
# Each is reported separately with its own verdict rather than reduced to one score,
# because a single number would let a good structure hide bad mechanisms.
# --------------------------------------------------------------------------- #
class EvaluateRequest(BaseModel):
    cv_splits: int = 5
    random_state: Optional[int] = 7
    gcm: bool = False        # gcm.evaluate_causal_model is minutes, not seconds
    falsify: bool = False    # permutation test over the graph's CI implications


def _cpt_lookup(cpd) -> tuple[str, list[str], list[str], dict[tuple, np.ndarray]]:
    """A CPT as `{parent-configuration -> probability vector over the node's states}`.

    pgmpy lays `get_values()` out as (states x parent configurations), the columns in
    `itertools.product` order over the parents' state lists. Reading it directly is
    thousands of times cheaper than running variable elimination once per row, which is
    what `BayesianNetwork.predict` does.
    """
    import itertools  # noqa: PLC0415

    node = cpd.variables[0]
    parents = [str(v) for v in cpd.variables[1:]]
    states = [str(s) for s in cpd.state_names[node]]
    values = np.asarray(cpd.get_values(), dtype=float)
    parent_states = [[str(s) for s in cpd.state_names[v]] for v in parents]
    combos = list(itertools.product(*parent_states)) if parents else [()]
    table = {combo: values[:, j] for j, combo in enumerate(combos)}
    return node, parents, states, table


def _cbn_scores(p: ProjectState) -> Optional[dict]:
    """Log-likelihood, parameter count and BIC of the fitted CPTs, on their own frame.

    BIC and not log-likelihood alone: a CPT with more cells than rows to estimate them
    always fits better and always means less, and on a flat join that is the normal
    case rather than the pathological one.
    """
    if p.model_kind != "cbn" or p.cbn is None:
        return None
    try:
        frame = p.model.spec.data[p.model.spec.columns].astype(str)
        n_rows = int(len(frame))
        total_ll = 0.0
        n_params = 0
        for cpd in p.cbn.get_cpds():
            node, parents, states, table = _cpt_lookup(cpd)
            index = {s: i for i, s in enumerate(states)}
            card = np.asarray(cpd.cardinality, dtype=int)
            # Free parameters: (states - 1) per parent configuration; the last is implied.
            n_params += int((card[0] - 1) * max(1, int(card[1:].prod()) if len(card) > 1 else 1))
            keys = (list(zip(*[frame[q] for q in parents])) if parents
                    else [()] * n_rows)
            column = frame[node].to_numpy()
            for key, value in zip(keys, column):
                probs = table.get(tuple(key))
                if probs is None:
                    continue
                total_ll += math.log(max(float(probs[index[value]]), 1e-12))
        bic = total_ll - 0.5 * n_params * math.log(max(2, n_rows))
        return {
            "log_likelihood": float(total_ll),
            "n_parameters": int(n_params),
            "bic": float(bic),
            "per_row_log_likelihood": float(total_ll / max(1, n_rows)),
        }
    except Exception as exc:  # noqa: BLE001 — an optional score must never lose the CV
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/projects/{pid}/model/evaluate")
def evaluate(pid: str, body: EvaluateRequest):
    """Structure, mechanisms, calibration and abduction — scored separately (W25)."""
    p = _get_project(pid)
    model = _require_model(p)

    started = time.time()
    rows: list[dict] = []
    cv_error = None
    if p.model_kind == "scm":
        from causalway.evaluation import held_out_cv  # noqa: PLC0415
        try:
            cv = held_out_cv(model, n_splits=max(2, body.cv_splits),
                             random_state=body.random_state)
            rows = [{k: _py(v) for k, v in r.items()} for r in cv.to_dict(orient="records")]
        except Exception as exc:  # noqa: BLE001
            cv_error = f"{type(exc).__name__}: {exc}"
    else:
        rows, cv_error = _cbn_cv(p, body)

    n_scored = len(rows)
    n_failing = sum(1 for r in rows if r.get("beats_baseline") is False)
    lifts = [float(r["model"]) - float(r["baseline"]) for r in rows
             if r.get("model") is not None and r.get("baseline") is not None]

    graph = _curated_graph(p)
    structure = {
        "n_nodes": int(len(model.spec.columns)),
        "n_edges": int(model.scm.graph.number_of_edges()),
        "acyclic": bool(graph["acyclic"]),
        "topological_validity": graph["topological_validity"],
        "falsification": None,
    }
    if body.falsify:
        try:
            from causalway.evaluation import falsify  # noqa: PLC0415
            result = falsify(model)
            structure["falsification"] = {
                "summary": str(result),
                # falsify_graph rejects when the real graph violates no fewer CIs than a
                # random permutation of it does. Not rejected != correct.
                "rejected": bool(getattr(result, "falsifiable", False)
                                 and getattr(result, "falsified", False)),
            }
        except Exception as exc:  # noqa: BLE001
            structure["falsification"] = {"summary": f"{type(exc).__name__}: {exc}",
                                          "rejected": None}

    gcm_summary = None
    pnl = None
    if body.gcm and p.model_kind == "scm":
        from causalway.evaluation import evaluate_model  # noqa: PLC0415
        try:
            result = evaluate_model(model)
            gcm_summary = str(result)
            pnl = {
                str(k): [_py(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
                for k, v in (getattr(result, "pnl_assumptions", None) or {}).items()
            }
        except Exception as exc:  # noqa: BLE001 — an optional check must not lose the CV
            gcm_summary = f"gcm evaluation failed: {type(exc).__name__}: {exc}"

    non_invertible = (p.model_info or {}).get("non_invertible", [])
    payload = {
        "model_kind": p.model_kind,
        "cv": rows,
        "cv_error": cv_error,
        "n_failing": n_failing,
        "n_scored": n_scored,
        "summary": {
            "nodes_beating_baseline": n_scored - n_failing,
            "nodes_scored": n_scored,
            "mean_lift": (sum(lifts) / len(lifts)) if lifts else None,
            "worst_node": (min(rows, key=lambda r: float(r["model"]) - float(r["baseline"]))["node"]
                           if lifts else None),
        },
        "structure": structure,
        "calibration": _cbn_scores(p),
        "abduction": {
            "supported": p.model_kind == "scm",
            "non_invertible": non_invertible,
            "pnl_assumptions": pnl,
        },
        "gcm_summary": gcm_summary,
        "pnl_assumptions": pnl,
        "elapsed": round(time.time() - started, 1),
    }
    p.evaluation = payload
    return payload


def _cbn_cv(p: ProjectState, body: EvaluateRequest) -> tuple[list[dict], Optional[str]]:
    """k-fold CV of each node's CPT against its marginal mode, refitting per fold.

    Refitting per fold — unlike the SCM path, which scores the already-fitted
    mechanism — because a CPT estimated on the test rows would score itself.
    """
    try:
        from sklearn.metrics import f1_score  # noqa: PLC0415
        from sklearn.model_selection import KFold  # noqa: PLC0415

        from pgmpy.estimators import BayesianEstimator  # noqa: PLC0415
        from pgmpy.models import DiscreteBayesianNetwork  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"

    model = p.model
    frame = model.spec.data[model.spec.columns].astype(str).reset_index(drop=True)
    graph_edges = list(model.scm.graph.edges())
    scored = [c for c in model.spec.columns if list(model.scm.graph.predecessors(c))]
    if not scored:
        return [], None

    preds: dict[str, list] = {c: [] for c in scored}
    bases: dict[str, list] = {c: [] for c in scored}
    truths: dict[str, list] = {c: [] for c in scored}
    state_names = {c: sorted(frame[c].unique().tolist()) for c in model.spec.columns}
    kf = KFold(n_splits=max(2, body.cv_splits), shuffle=True, random_state=body.random_state)
    try:
        for train_idx, test_idx in kf.split(frame):
            train, test = frame.iloc[train_idx], frame.iloc[test_idx]
            bn = DiscreteBayesianNetwork()
            bn.add_nodes_from(list(model.spec.columns))
            bn.add_edges_from(graph_edges)
            bn.fit(train, estimator=BayesianEstimator, prior_type="BDeu",
                   equivalent_sample_size=5, state_names=state_names)
            for col in scored:
                # argmax of the node's own CPT column for each test row's parent
                # configuration. `bn.predict` would run variable elimination per row —
                # minutes for what is a dictionary lookup (see _cpt_lookup).
                _n, parents, states, table = _cpt_lookup(bn.get_cpds(col))
                fallback = train[col].mode().iloc[0]
                keys = (list(zip(*[test[q] for q in parents])) if parents
                        else [()] * len(test))
                for key in keys:
                    probs = table.get(tuple(key))
                    preds[col].append(fallback if probs is None
                                      else states[int(np.argmax(probs))])
                bases[col].extend([fallback] * len(test))
                truths[col].extend(test[col].tolist())
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"

    rows = []
    for col in scored:
        acc = float(np.mean([a == b for a, b in zip(preds[col], truths[col])]))
        base = float(np.mean([a == b for a, b in zip(bases[col], truths[col])]))
        rows.append({
            "node": col, "metric": "accuracy", "model": acc, "baseline": base,
            "beats_baseline": acc > base,
            "macro_f1": float(f1_score(truths[col], preds[col], average="macro",
                                       zero_division=0)),
        })
    return rows, None


@app.get("/api/projects/{pid}/model/evaluation")
def get_evaluation(pid: str):
    return _get_project(pid).evaluation or {"cv": [], "n_failing": 0, "gcm_summary": None}


# --------------------------------------------------------------------------- #
# Modules 4 and 5 — conditional, interventional and counterfactual answers
# (Plan 3 §6-§7, Plan 2 §4-§5)
# --------------------------------------------------------------------------- #
def _require_scm(p: ProjectState):
    """Anything routed through gcm mechanisms needs an SCM; a CBN has CPTs instead."""
    if p.model_kind != "scm":
        raise HTTPException(400, (
            "This project is fitted as a causal Bayesian network, whose mechanisms are "
            "conditional probability tables rather than gcm functions. Use the board's "
            "Predict (exact variable elimination), or refit as a structural causal model."
        ))
    return _require_model(p)


def _coerce(p: ProjectState, node: str, value: Any):
    """JSON carries strings; the fitted frame has dtypes. Meet in the middle."""
    dtype = (p.model_info or {}).get("dtypes", {}).get(node)
    if value is None or value == "":
        raise HTTPException(400, f"No value given for {node!r}.")
    if dtype in ("categorical", "ordinal"):
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"{node} is {dtype}; {value!r} is not a number.") from exc


def _check_nodes(p: ProjectState, names) -> None:
    columns = set((p.model_info or {}).get("columns", []))
    unknown = [n for n in names if n not in columns]
    if unknown:
        raise HTTPException(400, f"Not nodes in the fitted model: {unknown}")


def _check_reach(p: ProjectState, sources, target: str) -> None:
    """§6.5 causal reach. An unreachable intervention returns the factual value, which
    reads as a confident 'no effect' — the one failure mode worth a hard error."""
    graph = p.model.scm.graph
    for s in sources:
        if s != target and not nx.has_path(graph, s, target):
            raise HTTPException(422, detail={
                "error": "no_reach", "source": s, "target": target,
                "message": (
                    f"No directed path from {s} to {target} in the fitted graph, so "
                    f"do({s}) cannot move {target}. The answer would be the factual "
                    "value and would read as a confident 'no effect'."
                ),
            })


def _answer_payload(ans, kind: str, extra: dict) -> dict:
    dist = None
    if ans.distribution:
        dist = {str(k): _py(v) for k, v in ans.distribution.items()}
    effect = ans.effect
    if isinstance(effect, dict):
        effect = {str(k): _py(v) for k, v in effect.items()}
    else:
        effect = _py(effect)
    return {
        "kind": kind,
        "target": ans.target,
        "entity": ans.entity,
        "predicted": _py(ans.predicted),
        "distribution": dist,
        "mean": _py(ans.mean),
        "std": _py(ans.std),
        "ci": [_py(x) for x in ans.ci] if ans.ci else None,
        "factual": _py(ans.factual),
        "effect": effect,
        "backend": ans.backend,
        "n_samples": _py(ans.n_samples),
        "ess": _py(ans.ess),
        "low_confidence": bool(ans.low_confidence),
        "coupling": ans.coupling,
        **extra,
    }


def _log_answer(p: ProjectState, payload: dict) -> dict:
    payload = {"answer_id": p.next_answer_id, "at": time.time(), **payload}
    p.next_answer_id += 1
    p.answers.append(payload)
    del p.answers[:-50]  # a session log, not a database
    return payload


def _estimand(kind: str, target: str, evidence: dict, interventions: dict,
              conditions: dict, entity: Optional[str]) -> str:
    """The estimand, rendered. Kept on every answer so an associational number and a
    causal one can never be mistaken for each other (§6.2)."""
    given = []
    for k, v in (interventions or {}).items():
        given.append(f"do({k} = {v})")
    for k, v in (evidence or {}).items():
        given.append(f"{k} = {v}")
    for k, v in (conditions or {}).items():
        given.append(f"{k} = {v}")
    body = f"{target}" + (" | " + ", ".join(given) if given else "")
    if kind == "counterfactual":
        return f"P({target}_{{{', '.join(f'{k}={v}' for k, v in interventions.items())}}} | {entity})"
    return f"P({body})"


# --------------------------------------------------------------------------- #
# A query has 1..n targets (Plan 2 §6.3, `Query.target` was always a list).
#
# `target` on each of the three request bodies below takes either one node name
# or a list of them, and the *request* shape decides the response shape: a bare
# string answers exactly as it always did (one flat answer object, logged as one
# answer), a list answers `{"answers": [...]}` with one entry per target. A
# one-element list is still a list — a client that always sends a list can
# always read `answers`, without branching on how many targets it happened to ask
# about.
#
# The engine answers a batch off one shared computation (one pgmpy network, one
# weighted sample set, one interventional draw, one abduction), so the answers in
# a batch are mutually consistent — the same reason the board shares its sample
# set (W23). Each answer is logged separately, because each is a distinct
# estimand with its own backend and its own effective sample size.
# --------------------------------------------------------------------------- #
def _targets(target) -> tuple[list[str], bool]:
    """`(targets, single)` — `single` is True only for a bare string."""
    if isinstance(target, str):
        return [target], True
    targets = list(dict.fromkeys(target))
    if not targets:
        raise HTTPException(400, "target: at least one target node is required.")
    return targets, False


def _answers_response(p: ProjectState, answers, targets: list[str], single: bool,
                      kind: str, extra_for) -> dict:
    """Log every answer in the batch; return one payload or `{"answers": [...]}`."""
    if single:
        return _log_answer(p, _answer_payload(answers, kind, extra_for(targets[0])))
    logged = [_log_answer(p, _answer_payload(answers[t], kind, extra_for(t))) for t in targets]
    return {"answers": logged}


class ConditionRequest(BaseModel):
    target: Union[str, list[str]]
    evidence: dict[str, Any] = {}
    conditions: dict[str, Any] = {}
    backend: str = "auto"
    num_samples: int = 10_000


@app.post("/api/projects/{pid}/infer/condition")
def infer_condition(pid: str, body: ConditionRequest):
    p = _get_project(pid)
    model = _require_scm(p)
    targets, single = _targets(body.target)
    _check_nodes(p, [*targets, *body.evidence, *body.conditions])
    clash = [t for t in targets if t in body.evidence or t in body.conditions]
    if clash:
        raise HTTPException(400, f"{', '.join(clash)} is the target; it cannot also be evidence.")
    evidence = {k: _coerce(p, k, v) for k, v in body.evidence.items()}
    conditions = {k: _coerce(p, k, v) for k, v in body.conditions.items()}
    try:
        ans = model.condition(body.target, evidence, conditions=conditions or None,
                              backend=body.backend, num_samples=body.num_samples)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Conditional query failed: {exc}") from exc
    return _answers_response(p, ans, targets, single, "conditional", lambda t: {
        "estimand": _estimand("conditional", t, evidence, {}, conditions, None),
        "evidence": {k: _py(v) for k, v in evidence.items()},
        "conditions": {k: _py(v) for k, v in conditions.items()},
        "interventions": {},
    })


class InterveneRequest(BaseModel):
    target: Union[str, list[str]]
    interventions: dict[str, Any]
    reference: dict[str, Any] = {}
    conditions: dict[str, Any] = {}
    entity: Optional[str] = None
    num_samples: int = 10_000


@app.post("/api/projects/{pid}/infer/intervene")
def infer_intervene(pid: str, body: InterveneRequest):
    p = _get_project(pid)
    model = _require_scm(p)
    if not body.interventions:
        raise HTTPException(400, "An interventional query needs at least one do().")
    targets, single = _targets(body.target)
    _check_nodes(p, [*targets, *body.interventions, *body.reference, *body.conditions])
    fixed = [t for t in targets if t in body.interventions]
    if fixed:
        raise HTTPException(
            400, f"do({', '.join(fixed)}) fixes the target; there is nothing to predict.")
    # Reach is a per-target property (§6.5): an intervention that reaches one target
    # need not reach another, so every target is checked on its own.
    for t in targets:
        _check_reach(p, list(body.interventions), t)
    interventions = {k: _coerce(p, k, v) for k, v in body.interventions.items()}
    reference = {k: _coerce(p, k, v) for k, v in body.reference.items()}
    conditions = {k: _coerce(p, k, v) for k, v in body.conditions.items()}
    try:
        ans = model.intervene(
            interventions, target=body.target, reference=reference or None,
            conditions=conditions or None, entity=body.entity,
            num_samples=body.num_samples,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Interventional query failed: {exc}") from exc
    return _answers_response(p, ans, targets, single, "interventional", lambda t: {
        "estimand": _estimand("interventional", t, {}, interventions,
                              conditions, body.entity),
        "evidence": {},
        "conditions": {k: _py(v) for k, v in conditions.items()},
        "interventions": {k: _py(v) for k, v in interventions.items()},
        "reference": {k: _py(v) for k, v in reference.items()},
    })


class CounterfactualRequest(BaseModel):
    target: Union[str, list[str]]
    interventions: dict[str, Any]
    entity: str
    num_samples: int = 200


@app.post("/api/projects/{pid}/infer/counterfactual")
def infer_counterfactual(pid: str, body: CounterfactualRequest):
    p = _get_project(pid)
    model = _require_scm(p)
    if not body.interventions:
        raise HTTPException(400, "A counterfactual needs at least one do().")
    targets, single = _targets(body.target)
    _check_nodes(p, [*targets, *body.interventions])
    for t in targets:
        _check_reach(p, list(body.interventions), t)
    interventions = {k: _coerce(p, k, v) for k, v in body.interventions.items()}
    # `counterfactual` keys interventions by (entity_iri_or_None, node) — unlike
    # `intervene`, which takes bare node names (Plan 2 §5.3). None means "this entity".
    keyed = {(None, k): v for k, v in interventions.items()}
    try:
        ans = model.counterfactual(keyed, entity=body.entity, target=body.target,
                                   num_samples=body.num_samples)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Counterfactual query failed: {exc}") from exc

    # The engine leaves `factual` unset here, but "what actually happened to this
    # unit" is the whole point of the contrast — without it the counterfactual is a
    # number with nothing to be counter to. Read it off the entity's own rows, which
    # `spec.mat.entity_ids` is the record of (hence the fit materialising the same
    # join module 1 did).
    from causalway.entities import aggregate, population_rows  # noqa: PLC0415

    rows = population_rows(model.spec.mat, body.entity)
    for t in targets:
        answer = ans if single else ans[t]
        if answer.factual is None and len(rows):
            answer.factual = aggregate(model.spec.data.loc[rows, t],
                                       model.spec.dtypes.get(t))
    return _answers_response(p, ans, targets, single, "counterfactual", lambda t: {
        "estimand": _estimand("counterfactual", t, {}, interventions, {}, body.entity),
        "evidence": {},
        "conditions": {},
        "interventions": {k: _py(v) for k, v in interventions.items()},
    })


# --------------------------------------------------------------------------- #
# W23 — the board answers about every node, not about one target
#
# The old surface asked one question about one target and printed one number. That
# is not how anyone reads a causal graph: setting `smokerType = NonSmoker` and then
# looking at survival *alone* throws away the fact that the same evidence also moved
# biomarker, tumourStage and everything else downstream. So the board computes the
# whole posterior — every node's marginal under the current observations and
# interventions — and paints it back onto the node cards. A card is the answer.
#
# The cost of that is one shared sample set rather than one query per node. A single
# topological forward pass generates the joint under do() (intervened nodes replace
# their mechanism, so they contribute no likelihood factor) and weights it by the
# density of each observation (so evidence conditions rather than filters). Every
# node's marginal is then a weighted summary of the same draw, which also means the
# numbers on different cards are mutually consistent — computing them separately
# would let them disagree.
#
# A causal Bayesian network skips all of that: variable elimination answers each
# node exactly, so there is nothing to sample and no effective sample size to worry
# about.
# --------------------------------------------------------------------------- #
KDE_GRID = 64
LOW_ESS = 30.0


# --------------------------------------------------------------------------- #
# W26 — one axis per numeric node, for every card that ever draws it
#
# A continuous card is a density over a horizontal scale, and the scale is the
# half that carries the meaning: `bmi = 31.5` is only readable against the range
# bmi actually takes. Three ways that used to break, all of them the same bug —
# the axis was derived from whatever values that one card happened to hold:
#
#  - a *pinned* node (observed, do(), or a counterfactual's hypothetical) had no
#    values at all: `_pinned_card` returned `x: None`, the browser fell back to
#    its `[0, 1]` default, the curve vanished, and the next click on that card
#    read a value off a 0-to-1 ruler that had nothing to do with the variable.
#  - a *degenerate* answer (every counterfactual on an additive-noise node: the
#    abducted residual is exact, so the answer is one number, not a spread) was
#    drawn as a fabricated Gaussian bump on an invented +-5% axis. That is a
#    picture of uncertainty the answer does not contain, on a scale nothing else
#    on the board shares.
#  - module 4's dashed *factual* line is clamped into the axis it is given, so a
#    posterior axis narrower than the factual value silently parked the baseline
#    at the card's edge and claimed the unit's real value was there.
#
# So the axis is now a property of the node, not of the answer: the training
# marginal's range, padded, widened only when the current answer or a pinned
# value falls outside it. `/model/marginals` computes the same extent from the
# same data, so the ghost prior and the posterior it sits behind are on one
# scale and can actually be read against each other. A point mass is drawn as a
# spike *on that axis* and flagged `degenerate`, which is the honest picture: a
# counterfactual that did not move sits exactly on the factual line.
# --------------------------------------------------------------------------- #
def _is_point(lo: float, hi: float) -> bool:
    """Is this spread a point mass? Relative, because 200 identical draws of 183.6
    still differ in the last bit and an absolute epsilon would call that a distribution."""
    return (hi - lo) <= 1e-9 * max(1.0, abs(lo), abs(hi))


def _numeric_axis(p: ProjectState, node: str, *include: float) -> Optional[tuple[float, float]]:
    """The horizontal extent every card for `node` is drawn on: its training range,
    padded, widened to cover any value the caller must be able to point at."""
    series = pd.to_numeric(p.model.spec.data[node], errors="coerce").dropna()
    pts = [float(v) for v in include if v is not None and np.isfinite(float(v))]
    if series.empty and not pts:
        return None
    lo = float(series.min()) if not series.empty else min(pts)
    hi = float(series.max()) if not series.empty else max(pts)
    if pts:
        lo, hi = min(lo, *pts), max(hi, *pts)
    pad = max(abs(lo), abs(hi), 1.0) * 0.05 if _is_point(lo, hi) else (hi - lo) * 0.06
    return lo - pad, hi + pad


def _numeric_axes(p: ProjectState) -> dict[str, tuple[float, float]]:
    """`_numeric_axis` for every continuous column, computed once per request."""
    axes = {}
    for col in p.model.spec.columns:
        if p.model.spec.dtypes.get(col) in ("categorical", "ordinal"):
            continue
        axis = _numeric_axis(p, col)
        if axis is not None:
            axes[col] = axis
    return axes


def _kde_grid(values: np.ndarray, weights: Optional[np.ndarray] = None,
              n: int = KDE_GRID,
              axis: Optional[tuple[float, float]] = None,
              ) -> tuple[list[float], list[float]]:
    """A density curve for a continuous node, computed server-side (never in the browser).

    `axis` pins the horizontal extent to the node's own scale (see W26); the grid is
    still widened if the values themselves run past it, so nothing is ever cropped.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        if axis is None:
            return [0.0, 1.0], [0.0, 0.0]
        xs = np.linspace(axis[0], axis[1], n)
        return [float(x) for x in xs], [0.0] * n
    lo, hi = float(vals.min()), float(vals.max())
    if _is_point(lo, hi):
        # A point mass, not a density. Draw it as a spike on the node's own axis —
        # a Gaussian bump on an invented axis would be a picture of uncertainty this
        # answer does not have.
        if axis is None:
            pad = max(abs(lo) * 0.05, 0.5)
            glo, ghi = lo - pad, lo + pad
        else:
            glo, ghi = min(float(axis[0]), lo), max(float(axis[1]), lo)
        xs = np.linspace(glo, ghi, n)
        step = max(float(xs[1] - xs[0]) if n > 1 else 1.0, 1e-12)
        ys = np.clip(1.0 - np.abs(xs - lo) / (step * 1.2), 0.0, 1.0)
        return [float(x) for x in xs], [float(y) for y in ys]
    pad = (hi - lo) * 0.06
    glo, ghi = lo - pad, hi + pad
    if axis is not None:
        glo, ghi = min(glo, float(axis[0])), max(ghi, float(axis[1]))
    xs = np.linspace(glo, ghi, n)
    try:
        kde = gaussian_kde(vals, weights=weights)
        ys = kde(xs)
    except Exception:  # noqa: BLE001 — a KDE that will not fit falls back to a histogram
        counts, edges = np.histogram(vals, bins=min(20, max(4, vals.size // 5)),
                                     weights=weights, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ys = np.interp(xs, centres, counts, left=0.0, right=0.0)
    return [float(x) for x in xs], [float(max(0.0, y)) for y in ys]


def _weighted_dist(values, dtype: str, weights: Optional[np.ndarray],
                   levels: Optional[list[str]] = None,
                   axis: Optional[tuple[float, float]] = None) -> dict:
    """One node's marginal, in the shape its card knows how to draw."""
    n = len(values)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        w = np.ones(n)
        total = float(n)
    w = w / total

    if dtype in ("categorical", "ordinal"):
        agg: dict[str, float] = {}
        for value, weight in zip(values, w):
            key = str(value)
            agg[key] = agg.get(key, 0.0) + float(weight)
        if levels:
            agg = {lv: agg.get(lv, 0.0) for lv in levels} | {
                k: v for k, v in agg.items() if k not in set(levels)
            }
        predicted = max(agg, key=agg.get) if agg else None
        return {"kind": "categorical", "probs": agg, "predicted": predicted,
                "mean": None, "std": None, "ci": None, "x": None, "y": None}

    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(vals)
    vals, wv = vals[ok], w[ok]
    if vals.size == 0:
        return {"kind": "numeric", "probs": None, "predicted": None,
                "mean": None, "std": None, "ci": None, "x": [0.0, 1.0], "y": [0.0, 0.0],
                "degenerate": False}
    wv = wv / wv.sum() if wv.sum() > 0 else np.full(vals.size, 1.0 / vals.size)
    mean = float(np.sum(wv * vals))
    std = float(np.sqrt(max(float(np.sum(wv * (vals - mean) ** 2)), 0.0)))
    order = np.argsort(vals)
    cdf = np.cumsum(wv[order])
    ci = [float(np.interp(0.025, cdf, vals[order])), float(np.interp(0.975, cdf, vals[order]))]
    xs, ys = _kde_grid(vals, wv, axis=axis)
    return {"kind": "numeric", "probs": None, "predicted": mean, "mean": mean,
            "std": std, "ci": ci, "x": xs, "y": ys,
            "degenerate": _is_point(float(vals.min()), float(vals.max())),
            "min": float(vals.min()), "max": float(vals.max())}


def _joint_samples(model, evidence: dict, interventions: dict, num_samples: int):
    """One weighted draw from the joint under do(interventions), conditioned on evidence.

    Intervened nodes are *set* and contribute no likelihood factor — that is what
    makes it do() and not a very confident observation. Observed nodes are set too,
    but each multiplies the sample weight by its own conditional density, which is
    plain likelihood weighting. Both in one topological pass, so every node's marginal
    comes off the same sample set and they cannot disagree with each other.
    """
    from dowhy.graph import is_root_node  # noqa: PLC0415

    from causalway.inference import _density_at  # noqa: PLC0415

    g = model.scm.graph
    samples: dict[str, np.ndarray] = {}
    weights = np.ones(num_samples)
    for node in nx.topological_sort(g):
        parents = list(g.predecessors(node))
        pa = (np.column_stack([samples[q] for q in parents]) if parents
              else np.zeros((num_samples, 0)))
        if node in interventions:
            samples[node] = np.array([interventions[node]] * num_samples, dtype=object)
            continue
        if node in evidence:
            value = evidence[node]
            samples[node] = np.array([value] * num_samples, dtype=object)
            weights = weights * _density_at(model, node, pa, value, num_samples)
            continue
        mech = model.scm.causal_mechanism(node)
        drawn = (mech.draw_samples(num_samples) if is_root_node(g, node)
                 else mech.draw_samples(pa))
        samples[node] = np.asarray(drawn).reshape(-1)

    if evidence and float(weights.sum()) <= 0:
        raise HTTPException(400, (
            "Likelihood weighting collapsed to zero total weight: every drawn sample had "
            "zero density under this evidence. The combination you set may be impossible "
            "under the fitted model — loosen it, or intervene instead of observing."
        ))
    norm = weights / weights.sum() if float(weights.sum()) > 0 else np.full(
        num_samples, 1.0 / num_samples)
    ess = float(1.0 / np.sum(norm ** 2))
    return samples, norm, ess


def _mutilated_engine(p: ProjectState, interventions: dict):
    """do() by graph surgery: cut each intervened node's parents, pin its CPT to a point mass.

    `CausalInference.query(do=...)` is not usable here. It answers a *single* effect via
    backdoor adjustment and refuses outright when the query variable is not a descendant
    of the intervention ("Invalid causal query: there is a direct edge from ... to ...").
    But the board asks about every node at once, and for a non-descendant the answer is
    simply its unchanged marginal — a legitimate question, not an invalid one. Mutilating
    the network is the definition of do() anyway, and it answers all nodes uniformly.
    """
    from pgmpy.inference import VariableElimination  # noqa: PLC0415

    if not interventions:
        return VariableElimination(p.cbn)

    from pgmpy.factors.discrete import TabularCPD  # noqa: PLC0415
    from pgmpy.models import DiscreteBayesianNetwork  # noqa: PLC0415

    bn = p.cbn
    mutilated = DiscreteBayesianNetwork()
    mutilated.add_nodes_from(list(bn.nodes()))
    mutilated.add_edges_from([(u, v) for u, v in bn.edges() if v not in interventions])
    cpds = []
    for node in bn.nodes():
        cpd = bn.get_cpds(node)
        if node not in interventions:
            cpds.append(cpd)
            continue
        states = [str(s) for s in cpd.state_names[node]]
        wanted = str(interventions[node])
        if wanted not in states:
            raise HTTPException(400, (
                f"do({node} = {wanted}) names a level the fitted CPT has never seen; "
                f"known levels are {', '.join(states)}."
            ))
        cpds.append(TabularCPD(
            node, len(states), [[1.0 if s == wanted else 0.0] for s in states],
            state_names={node: states},
        ))
    mutilated.add_cpds(*cpds)
    return VariableElimination(mutilated)


def _cbn_predict(p: ProjectState, evidence: dict, interventions: dict) -> dict:
    """Exact variable elimination, one query per free node. No sampling, no ESS."""
    model = p.model
    levels = (p.model_info or {}).get("levels", {})
    pinned = {**evidence, **interventions}
    engine = _mutilated_engine(p, interventions)
    ev = {k: str(v) for k, v in evidence.items()}
    out: dict[str, dict] = {}
    for col in model.spec.columns:
        if col in pinned:
            continue
        try:
            factor = engine.query(variables=[col], evidence=ev or None, show_progress=False)
        except Exception as exc:  # noqa: BLE001 — the user's to read verbatim
            raise HTTPException(400, (
                f"Exact query for {col} failed: {exc}. Evidence that no row satisfies "
                "under this intervention has zero probability and cannot be conditioned on."
            )) from exc
        names = list(factor.state_names[col])
        values = np.asarray(factor.values, dtype=float).reshape(-1)
        total = float(values.sum()) or 1.0
        probs = {str(k): float(v) / total for k, v in zip(names, values)}
        ordered = levels.get(col)
        if ordered:
            probs = {lv: probs.get(lv, 0.0) for lv in ordered} | {
                k: v for k, v in probs.items() if k not in set(ordered)
            }
        out[col] = {"kind": "categorical", "probs": probs,
                    "predicted": max(probs, key=probs.get) if probs else None,
                    "mean": None, "std": None, "ci": None, "x": None, "y": None}
    return out


def _pinned_card(p: ProjectState, node: str, value, mode: str) -> dict:
    """A node the user has fixed: its 'distribution' is the point mass they chose.

    A categorical pin keeps every level and puts all the weight on one of them, so the
    card still shows the alternatives it is *not*. The continuous pin is the same idea:
    it keeps the node's own axis (W26) and puts a spike on it. Returning no grid at all
    left the browser with nothing to scale by, which is how a pinned `bmi` card ended up
    ruled 0 to 1 and unclickable for any value that meant anything.
    """
    dtype = p.model.spec.dtypes.get(node)
    if dtype in ("categorical", "ordinal"):
        levels = (p.model_info or {}).get("levels", {}).get(node, [])
        probs = {lv: (1.0 if str(lv) == str(value) else 0.0) for lv in levels}
        if str(value) not in probs:
            probs[str(value)] = 1.0
        return {"kind": "categorical", "probs": probs, "predicted": str(value),
                "mean": None, "std": None, "ci": None, "x": None, "y": None,
                "mode": mode}
    v = float(value)
    axis = _numeric_axis(p, node, v)
    xs, ys = _kde_grid(np.array([v]), axis=axis)
    series = pd.to_numeric(p.model.spec.data[node], errors="coerce").dropna()
    return {"kind": "numeric", "probs": None, "predicted": v,
            "mean": v, "std": 0.0, "ci": [v, v],
            "x": xs, "y": ys, "degenerate": True,
            "min": float(series.min()) if not series.empty else v,
            "max": float(series.max()) if not series.empty else v,
            "mode": mode}


def _board_estimand(evidence: dict, interventions: dict) -> str:
    """What the top bar shows. Rendered here so the client cannot drift from the server."""
    if not evidence and not interventions:
        return "P(V) — observational marginals"
    given = [f"do({k} = {v})" for k, v in interventions.items()]
    given += [f"{k} = {v}" for k, v in evidence.items()]
    return f"P(V | {', '.join(given)})"


@app.get("/api/projects/{pid}/model/marginals")
def model_marginals(pid: str):
    """Every node's observational marginal, straight off the training frame.

    This is the board's baseline state and the ghost layer every posterior is drawn
    against — without it a bar at 0.42 says nothing, because nobody knows whether the
    evidence moved it up or down.
    """
    p = _get_project(pid)
    model = _require_model(p)
    levels = (p.model_info or {}).get("levels", {})
    nodes = {
        col: _weighted_dist(model.spec.data[col].tolist(), model.spec.dtypes.get(col),
                            None, levels.get(col))
        for col in model.spec.columns
    }
    return {"n_rows": int(model.spec.data.shape[0]), "nodes": nodes}


class PredictRequest(BaseModel):
    evidence: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    num_samples: int = 5000


@app.post("/api/projects/{pid}/infer/predict")
def infer_predict(pid: str, body: PredictRequest):
    """The whole board's posterior under the current observations and interventions (W23)."""
    p = _get_project(pid)
    model = _require_model(p)
    _check_nodes(p, [*body.evidence, *body.interventions])
    overlap = set(body.evidence) & set(body.interventions)
    if overlap:
        raise HTTPException(400, (
            f"{', '.join(sorted(overlap))} is both observed and intervened on. A node can "
            "be one or the other: observing reads a value off the world, intervening "
            "replaces the mechanism that produced it."
        ))
    evidence = {k: _coerce(p, k, v) for k, v in body.evidence.items()}
    interventions = {k: _coerce(p, k, v) for k, v in body.interventions.items()}

    started = time.time()
    if p.model_kind == "cbn":
        nodes = _cbn_predict(p, evidence, interventions)
        backend, ess, n_samples = "pgmpy-exact", None, None
    else:
        n_samples = max(200, min(50_000, body.num_samples))
        samples, weights, ess = _joint_samples(model, evidence, interventions, n_samples)
        levels = (p.model_info or {}).get("levels", {})
        axes = _numeric_axes(p)
        pinned = {**evidence, **interventions}
        nodes = {
            col: _weighted_dist(samples[col], model.spec.dtypes.get(col), weights,
                                levels.get(col), axis=axes.get(col))
            for col in model.spec.columns if col not in pinned
        }
        backend = ("gcm.interventional-sampling" if interventions and evidence
                   else "gcm.interventional_samples" if interventions
                   else "likelihood-weighting" if evidence
                   else "ancestral-sampling")

    for node, value in evidence.items():
        nodes[node] = _pinned_card(p, node, value, "observed")
    for node, value in interventions.items():
        nodes[node] = _pinned_card(p, node, value, "intervened")

    payload = {
        "estimand": _board_estimand(evidence, interventions),
        "kind": "interventional" if interventions else "conditional",
        "model_kind": p.model_kind,
        "backend": backend,
        "n_samples": n_samples,
        "ess": ess,
        "low_confidence": bool(ess is not None and ess < LOW_ESS),
        "elapsed": round(time.time() - started, 2),
        "n_observed": len(evidence),
        "n_intervened": len(interventions),
        "evidence": {k: _py(v) for k, v in evidence.items()},
        "interventions": {k: _py(v) for k, v in interventions.items()},
        "nodes": nodes,
    }
    # Every board prediction lands in the same session log the single-target queries
    # use, so the query log stays the complete record of what was asked and the CSV
    # export stays the only durable copy of it (W14).
    _log_answer(p, {
        "kind": payload["kind"], "estimand": payload["estimand"],
        "target": "V (all nodes)", "entity": None,
        "predicted": None, "distribution": None, "mean": None, "std": None, "ci": None,
        "factual": None, "effect": None, "backend": backend, "n_samples": n_samples,
        "ess": ess, "low_confidence": payload["low_confidence"], "coupling": None,
        "board": {node: dist.get("predicted") for node, dist in nodes.items()},
        "evidence": payload["evidence"], "conditions": {},
        "interventions": payload["interventions"],
    })
    payload["answer_id"] = p.answers[-1]["answer_id"]
    return payload


# --------------------------------------------------------------------------- #
# W28 — a unit the user types, not one the KG supplies
#
# The counterfactual board used to require an `entity`, resolved to its rows in
# the materialised frame. That makes module 4 unusable in exactly the case an
# exported model is *for*: a model imported without its source KG has no
# entities to pick from, and neither does a user asking "what about a patient
# like this?" about someone who is not in the KG.
#
# So `observed` is accepted instead of `entity` — one synthetic row, the values
# the user typed. Everything downstream is unchanged: abduction reads a row,
# and does not care whether that row came from a SPARQL solution or a form.
#
# Two rules make it honest rather than convenient:
#
#   * every modelled variable must be given a value. A counterfactual abducts
#     noise from the *whole* observed state; leaving a node out would mean
#     silently inventing its value and then reporting the answer as if the user
#     had specified it. There is no sensible default — the mean is a value no
#     real unit has, and the modal level is a guess wearing a fact's clothes.
#   * the resulting world carries no `aboutEntity`. It describes nobody, so it
#     must not be mergeable back into the source KG (`cf_export` mints a
#     content-hashed IRI for it instead, hashing the observed values too so two
#     different hypothetical units cannot collide).
#
# The front end prefills the form from `model_info.levels` / `.ranges`, so the
# user picks from the levels the model was actually fitted on.
# --------------------------------------------------------------------------- #
#: Ceiling on the rows in one abduction batch, i.e. `draws x rows_per_draw`.
#: Sized so the batch is comfortably large enough to amortise scikit-learn's
#: per-call overhead (the thing that made the board slow) while the packed
#: Gumbel noise — one small object array per row per categorical node — stays
#: in the tens of megabytes even for an entity with hundreds of join rows.
#: A deliberate second copy of `causalway.inference.CF_BATCH_ROWS`: importing
#: that module here would be a top-level Plan 2 import and would charge every
#: discovery-only user dowhy's ~30s import (CLAUDE.md §2). **Change one, change
#: the other** — the two counterfactual paths are separate code (this one
#: answers about a hypothetical typed-in unit, W28) but the same trade-off.
_CF_BATCH_ROWS = 20_000


class CfPredictRequest(BaseModel):
    entity: Optional[str] = None
    observed: Optional[dict[str, Any]] = None
    interventions: dict[str, Any] = {}
    num_samples: int = 200


def _hypothetical_row(p: ProjectState, model, observed: dict[str, Any]) -> pd.DataFrame:
    """One typed-in row, validated against the fitted columns (W28)."""
    columns = list(model.spec.columns)
    _check_nodes(p, list(observed))
    missing = [c for c in columns if c not in observed]
    if missing:
        raise HTTPException(400, (
            f"A hypothetical unit needs a value for every modelled variable; "
            f"missing {sorted(missing)}. Abduction reads the whole observed state, "
            "so an unset node would be invented rather than assumed."
        ))
    row = {c: _coerce(p, c, observed[c]) for c in columns}
    frame = pd.DataFrame([row], columns=columns)
    for col in columns:
        if model.spec.dtypes.get(col) not in ("categorical", "ordinal"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


@app.post("/api/projects/{pid}/infer/counterfactual/predict")
def infer_counterfactual_predict(pid: str, body: CfPredictRequest):
    """Every node's counterfactual marginal for one unit, against its own factual state.

    The noise is abducted from this unit's rows once per draw and held fixed while the
    hypothetical is evaluated — that shared noise is the entire difference between this
    and an intervention. Every node is returned, because "what else would have been
    different" is the question a counterfactual is usually being asked to answer.

    The unit is either an `entity` from the source KG or an `observed` row the
    user typed (W28); exactly one of the two.
    """
    p = _get_project(pid)
    model = _require_model(p)
    if p.model_kind != "scm":
        raise HTTPException(400, (
            "This project is fitted as a causal Bayesian network. A CPT has no exogenous "
            "noise term, so there is nothing to abduct and no counterfactual to compute. "
            "Refit as a structural causal model in the Causal model panel."
        ))
    if not body.interventions:
        raise HTTPException(400, "A counterfactual needs at least one hypothetical value.")
    _check_nodes(p, list(body.interventions))

    from dowhy import gcm  # noqa: PLC0415
    from dowhy.gcm._noise import compute_noise_from_data  # noqa: PLC0415

    from causalway.entities import aggregate, population_rows  # noqa: PLC0415

    if (body.entity is None) == (body.observed is None):
        raise HTTPException(400, (
            "Give exactly one of `entity` (a unit from the source KG) or `observed` "
            "(a hypothetical unit's values). Both would be two different factual "
            "states for one counterfactual; neither leaves nothing to abduct from."
        ))
    if body.observed is not None:
        observed = _hypothetical_row(p, model, body.observed)
    else:
        if model.spec.mat is None or model.spec.data is None:
            raise HTTPException(400, (
                "This model has no materialised rows to resolve an entity against — it "
                "was imported without its source KG. Re-attach the source, or describe "
                "the unit directly with `observed`."
            ))
        rows = population_rows(model.spec.mat, body.entity)
        if len(rows) == 0:
            raise HTTPException(404, f"Entity {body.entity} has no rows in the materialised KG.")
        observed = model.spec.data.loc[rows, model.spec.columns]
    interventions = {k: _coerce(p, k, v) for k, v in body.interventions.items()}
    fns = {node: (lambda value: (lambda _x: value))(v) for node, v in interventions.items()}

    # Gumbel-max is a stochastic coupling, so a categorical node needs many draws to
    # have a distribution at all; an additive-noise node's abduction is an exact
    # residual and redrawing it would just repeat the same number (Plan 2 E1).
    any_categorical = any(model.spec.dtypes.get(c) == "categorical"
                          for c in model.spec.columns)
    n_draws = max(1, min(2000, body.num_samples)) if any_categorical else 1

    started = time.time()
    # The draws are *batched*, not looped one at a time.
    #
    # Both halves of a counterfactual are row-wise and independent across rows:
    # `InvertibleClassifierFCM.estimate_noise` abducts every row's Gumbel vector
    # in one vectorised top-down sample, and `evaluate` is an argmax over rows.
    # So drawing `n_draws` independent counterfactuals for one unit is the same
    # computation as one counterfactual for `n_draws` copies of that unit —
    # which means the whole board can be one pass through the graph instead of
    # `n_draws` passes.
    #
    # This is not a micro-optimisation. The per-pass cost here is almost all
    # fixed overhead: each pass walks every node and calls `predict_proba` on a
    # frame of one or two rows, where scikit-learn's own setup (and, for the
    # histogram-gradient-boosting classifiers gcm assigns by default, an OpenMP
    # fork/join) costs far more than the arithmetic. 200 passes over 1 row and
    # 1 pass over 200 rows do the same work; only the first pays the overhead
    # 200 times. Measured on the 11-node synthetic clinic model, the default
    # board went from ~16s to ~0.2s.
    #
    # Deliberately *not* spread across processes or threads. A process pool
    # would have to ship the fitted SCM to each worker and re-import dowhy in
    # it, which costs more than the query; a thread pool would have several
    # threads drawing from one mechanism's `np.random.Generator` at once, which
    # is neither thread-safe nor reproducible — and reproducibility is the
    # whole point of the per-node seeding in `CausalModel.fit`. One vectorised
    # pass beats eight cores running the overhead in parallel.
    #
    # Chunked so that an entity with many join rows cannot blow up memory:
    # a Gumbel noise column is one small object array *per row per categorical
    # node*, so `n_draws x rows_per_draw` is the number that has to stay
    # bounded, not `n_draws`. At one row per draw — a hypothetical unit, and
    # the common case — the cap never binds and the whole board is one pass.
    rows_per_draw = max(1, len(observed))
    draws_per_batch = max(1, _CF_BATCH_ROWS // rows_per_draw)
    draws = []
    try:
        remaining = n_draws
        while remaining > 0:
            k = min(draws_per_batch, remaining)
            batch = (observed if k == 1
                     else pd.concat([observed] * k, ignore_index=True))
            noise = compute_noise_from_data(model.scm, batch)
            draws.append(gcm.counterfactual_samples(model.scm, fns, noise_data=noise))
            remaining -= k
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Counterfactual failed: {exc}") from exc
    frame = pd.concat(draws, ignore_index=True) if len(draws) > 1 else draws[0]

    levels = (p.model_info or {}).get("levels", {})
    # Every numeric card on this board is drawn on the node's own axis (W26), which is
    # also the axis the unit's dashed factual line is placed on. A per-card axis put a
    # deterministic counterfactual — the normal case here, since an additive-noise
    # residual is abducted exactly — on its own invented +-5% ruler, where the factual
    # value was clamped to the edge and read as if it were inside the answer.
    axes = _numeric_axes(p)
    nodes = {}
    for col in model.spec.columns:
        if col in interventions:
            nodes[col] = _pinned_card(p, col, interventions[col], "intervened")
        else:
            nodes[col] = _weighted_dist(frame[col].tolist(), model.spec.dtypes.get(col),
                                        None, levels.get(col), axis=axes.get(col))
    # A hypothetical unit's factual state is exactly what the user typed — there
    # is nothing to aggregate, and one row is already the answer.
    factual = {
        col: _py(observed[col].iloc[0] if body.observed is not None
                 else aggregate(model.spec.data.loc[rows, col], model.spec.dtypes.get(col)))
        for col in model.spec.columns
    }
    do_text = ', '.join(f'{k}={v}' for k, v in interventions.items())
    unit_label = body.entity if body.entity is not None else "a hypothetical unit"
    estimand = f"P(V_{{{do_text}}} | {unit_label})"
    _log_answer(p, {
        "kind": "counterfactual", "estimand": estimand, "target": "V (all nodes)",
        "entity": body.entity, "predicted": None, "distribution": None, "mean": None,
        "std": None, "ci": None, "factual": None, "effect": None,
        "backend": "gumbel-max-abduction" if any_categorical else "anm-abduction",
        "n_samples": int(len(frame)), "ess": None, "low_confidence": False,
        "coupling": "gumbel-max" if any_categorical else None,
        "board": {node: dist.get("predicted") for node, dist in nodes.items()},
        "evidence": {}, "conditions": {},
        "interventions": {k: _py(v) for k, v in interventions.items()},
    })
    # The board is also kept as a *world*, separately from the query log. A
    # counterfactual about a named entity is the only answer this service
    # produces that describes a resource of the source KG, so it is the only one
    # that can be merged back into that KG and navigated entity-first; a
    # population-level answer is about the model. Different kind of artifact,
    # its own export.
    #
    # A hypothetical unit (W28) keeps `entity = None` all the way through, and
    # `cf_export` turns that into a world with no `cw:aboutEntity` and a
    # content-hashed IRI. It has to stay None rather than becoming a placeholder
    # string: an invented IRI in an exported TTL is a claim about a resource
    # that does not exist, and merging it into the source KG would be a
    # fabrication rather than a note.
    world = {
        "world_id": p.next_world_id,
        "entity": body.entity,
        "observed": (dict(body.observed) if body.observed is not None else None),
        "estimand": estimand,
        "model_id": getattr(model, "model_id", None),
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": "gumbel-max-abduction" if any_categorical else "anm-abduction",
        "coupling": "gumbel-max" if any_categorical else None,
        "n_samples": int(len(frame)),
        # How many KG rows the factual state was read off. A hypothetical unit is
        # one typed row, which is a different kind of "1" — `entity=None` in the
        # same record is what distinguishes them.
        "n_rows": 1 if body.observed is not None else int(len(rows)),
        # The model's seed, so the RDF can say how to recompute this world. An
        # unseeded fit records `null`, which is the honest answer: the Gumbel
        # draws behind a categorical counterfactual came from an unseeded
        # generator and nobody can reproduce them, including us.
        "random_seed": (p.model_info or {}).get("random_state"),
        "dtypes": dict(model.spec.dtypes),
        "interventions": {k: _py(v) for k, v in interventions.items()},
        "factual": factual,
        "counterfactual": {node: dist.get("predicted") for node, dist in nodes.items()},
        # `probs` is only present on a categorical/ordinal card; a numeric node's
        # counterfactual is a single abducted value with no distribution over
        # levels. The export keeps only the arg-max mass out of this (the user's
        # point 3: a counterfactual answers "most likely value", not "what is the
        # whole distribution"), but the board still needs the full shape.
        "distribution": {
            node: dist["probs"]
            for node, dist in nodes.items() if dist.get("probs")
        },
        # Spread of a *numeric* counterfactual across the draws. Usually 0 — an
        # additive-noise residual is abducted exactly, so redrawing repeats the
        # same number — but genuinely non-zero when the node has a stochastic
        # categorical ancestor, whose Gumbel-max coupling is not identified.
        # Exported so a reader can tell those two cases apart instead of
        # guessing which kind of "point estimate" they are looking at.
        "std": {
            node: dist["std"]
            for node, dist in nodes.items()
            if dist.get("std") is not None and node not in interventions
        },
    }
    p.next_world_id += 1
    p.cf_worlds.append(world)
    del p.cf_worlds[:-50]   # a session log, like p.answers

    return {
        "answer_id": p.answers[-1]["answer_id"],
        "world_id": world["world_id"],
        "entity": body.entity,
        "hypothetical": body.observed is not None,
        "estimand": estimand,
        "backend": world["backend"],
        "coupling": world["coupling"],
        "n_samples": int(len(frame)),
        "n_rows": world["n_rows"],
        "elapsed": round(time.time() - started, 2),
        "interventions": {k: _py(v) for k, v in interventions.items()},
        "factual": factual,
        "nodes": nodes,
    }


@app.get("/api/projects/{pid}/model/mechanism/{node}")
def model_mechanism(pid: str, node: str):
    """One node's mechanism, in whichever form the fitted model actually has.

    For a Bayesian network that is the conditional probability table itself, which is
    the readable object the CBN was chosen for. For an SCM it is the assigned function
    class plus the noise it draws — a table would be a lie there, since the mechanism
    is a fitted function, not a lookup.
    """
    p = _get_project(pid)
    model = _require_model(p)
    _check_nodes(p, [node])
    graph = model.scm.graph
    parents = sorted(graph.predecessors(node))
    dtype = model.spec.dtypes.get(node)
    info = (p.model_info or {})
    row = next((m for m in info.get("mechanisms", []) if m["node"] == node), None)
    out = {
        "node": node,
        "dtype": dtype,
        "kind": p.model_kind,
        "parents": parents,
        "children": sorted(graph.successors(node)),
        "is_root": not parents,
        "mechanism_type": (row or {}).get("mechanism_type"),
        "invertible": (row or {}).get("invertible", False),
        "levels": info.get("levels", {}).get(node),
        "range": info.get("ranges", {}).get(node),
        "cpt": None,
        "noise": None,
    }
    if p.model_kind == "cbn" and p.cbn is not None:
        cpd = p.cbn.get_cpds(node)
        if cpd is not None:
            values = np.asarray(cpd.get_values(), dtype=float)
            evidence_vars = [v for v in (cpd.variables or [])[1:]]
            combos = _cpt_columns(cpd, evidence_vars)
            out["cpt"] = {
                "variable": node,
                "states": [str(s) for s in cpd.state_names[node]],
                "evidence": evidence_vars,
                # One column per parent configuration; capped because a 5-parent CPT is
                # thousands of columns and no one reads those in a side panel.
                "columns": combos[:64],
                "values": [[float(v) for v in rowv[:64]] for rowv in values],
                "truncated": len(combos) > 64,
                "n_columns": len(combos),
            }
    else:
        series = model.spec.data[node]
        out["noise"] = {
            "description": (
                "root — fitted as the empirical distribution of the column itself"
                if not parents else
                "additive/classifier noise, drawn from the fitted residual distribution"
            ),
            "n_unique": int(series.nunique(dropna=True)),
        }
    return out


def _cpt_columns(cpd, evidence_vars: list[str]) -> list[str]:
    """Human-readable labels for each parent configuration column of a CPT."""
    if not evidence_vars:
        return ["—"]
    import itertools  # noqa: PLC0415

    states = [[str(s) for s in cpd.state_names[v]] for v in evidence_vars]
    return [", ".join(f"{v}={s}" for v, s in zip(evidence_vars, combo))
            for combo in itertools.product(*states)]


# --------------------------------------------------------------------------- #
# W24 — choosing the unit a counterfactual is about
#
# "Pick an entity from a list of 4000 IRIs" is not a usable way to choose a unit, and
# it is not how anyone thinks about one either: you want *a patient like this*, not
# `patient_1731`. So the unit is chosen by constraining the variables — gender =
# Female, stage = II — with the number of units still matching reported live, and the
# IRI picked only once the pool is small enough to be meaningful.
# --------------------------------------------------------------------------- #
class UnitSearch(BaseModel):
    # value is a level for a categorical column, or "lo|hi" for a numeric band.
    filters: dict[str, str] = {}
    limit: int = 200


@app.post("/api/projects/{pid}/units/search")
def search_units(pid: str, body: UnitSearch):
    p = _get_project(pid)
    model = _require_model(p)
    mat = model.spec.mat
    if mat is None or mat.entity_ids is None or mat.entity_ids.empty:
        return {"classes": [], "n_matching_rows": 0, "n_total_rows": 0}

    data = model.spec.data
    mask = pd.Series(True, index=data.index)
    unknown = [c for c in body.filters if c not in model.spec.columns]
    if unknown:
        raise HTTPException(400, f"Not columns in the fitted model: {unknown}")
    for col, raw in body.filters.items():
        if raw is None or raw == "":
            continue
        dtype = model.spec.dtypes.get(col)
        if dtype in ("categorical", "ordinal") and "|" not in str(raw):
            mask &= data[col].astype(str) == str(raw)
        else:
            lo, _, hi = str(raw).partition("|")
            numeric = pd.to_numeric(data[col], errors="coerce")
            if lo:
                mask &= numeric >= float(lo)
            if hi:
                mask &= numeric <= float(hi)

    rows = data.index[mask]
    ids = mat.entity_ids.loc[rows] if len(rows) else mat.entity_ids.iloc[:0]
    classes = []
    for column in mat.entity_ids.columns:
        values = [str(v) for v in pd.unique(ids[column].dropna())]
        classes.append({
            "var": str(column),
            "n_total": int(mat.entity_ids[column].nunique(dropna=True)),
            "n_matching": len(values),
            "entities": values[: max(1, body.limit)],
            "truncated": len(values) > body.limit,
        })
    return {
        "classes": classes,
        "n_matching_rows": int(len(rows)),
        "n_total_rows": int(len(data)),
    }


@app.get("/api/projects/{pid}/units/factual")
def unit_factual(pid: str, entity: str):
    """One unit's actual observed values — what a counterfactual is counter to."""
    p = _get_project(pid)
    model = _require_model(p)
    from causalway.entities import aggregate, population_rows  # noqa: PLC0415

    rows = population_rows(model.spec.mat, entity)
    if len(rows) == 0:
        raise HTTPException(404, f"Entity {entity} has no rows in the materialised KG.")
    return {
        "entity": entity,
        "n_rows": int(len(rows)),
        "values": {
            col: _py(aggregate(model.spec.data.loc[rows, col], model.spec.dtypes.get(col)))
            for col in model.spec.columns
        },
    }


@app.get("/api/projects/{pid}/units/bands")
def unit_bands(pid: str, quantiles: int = 5):
    """Quantile bands per numeric column, so the unit filter can offer ranges not free text."""
    p = _get_project(pid)
    model = _require_model(p)
    out: dict[str, list[dict]] = {}
    for col in model.spec.columns:
        if model.spec.dtypes.get(col) in ("categorical", "ordinal"):
            continue
        values = pd.to_numeric(model.spec.data[col], errors="coerce").dropna()
        if values.empty:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))
        bands = []
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            bands.append({"index": i, "lo": lo, "hi": hi,
                          "label": f"{_short_num(lo)} – {_short_num(hi)}"})
        out[col] = bands
    return out


def _short_num(v: float) -> str:
    a = abs(v)
    if a != 0 and (a < 0.01 or a >= 100_000):
        return f"{v:.2e}"
    return f"{v:.2f}" if a < 10 else f"{v:.1f}" if a < 1000 else f"{v:.0f}"


@app.get("/api/projects/{pid}/entities")
def list_entities(pid: str, q: str = "", limit: int = 200):
    """Entity IRIs per class variable — what a counterfactual is *about* (§5.1)."""
    p = _get_project(pid)
    model = _require_model(p)
    ids = model.spec.mat.entity_ids if model.spec.mat is not None else None
    if ids is None:
        return {"classes": []}
    needle = q.strip().lower()
    out = []
    for column in ids.columns:
        values = [str(v) for v in pd.unique(ids[column].dropna())]
        if needle:
            values = [v for v in values if needle in v.lower()]
        out.append({
            "var": str(column),
            "n_total": int(ids[column].nunique(dropna=True)),
            "entities": values[:limit],
        })
    return {"classes": out}


@app.get("/api/projects/{pid}/answers")
def list_answers(pid: str):
    return _get_project(pid).answers


@app.get("/api/projects/{pid}/answers/{aid}/export")
def export_answer(pid: str, aid: int, format: str = "ttl"):
    """One logged answer on its own — `json`, `csv`, or cw: `ttl`.

    The whole log is still `GET /export?what=answers`; this is the per-question
    counterpart, because the interesting unit of module 3 is usually one
    estimand rather than the session. Both go through the same `_render`, so the
    single answer is a filter over the log rather than a second serialiser.
    """
    p = _get_project(pid)
    payload, media_type, suffix = _render(p, "answers", format, [aid])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Response(
        payload, media_type=media_type,
        headers={"Content-Disposition":
                 f'attachment; filename="causalway-answer-{aid}-{stamp}.{suffix}"'},
    )


@app.delete("/api/projects/{pid}/answers")
def clear_answers(pid: str):
    p = _get_project(pid)
    n = len(p.answers)
    p.answers = []
    return {"ok": True, "cleared": n}


# --------------------------------------------------------------------------- #
# Export (W14) — the service holds no persistent user data, so every intermediate
# artifact has to be retrievable on demand. This is the recovery path, not a
# convenience feature.
# --------------------------------------------------------------------------- #
def _nodes_frame(p: ProjectState) -> pd.DataFrame:
    payload = get_nodes(p.id)["nodes"]
    return pd.DataFrame(payload)


def _ledger_frame(p: ProjectState) -> pd.DataFrame:
    edges = _ledger(p)["edges"]
    df = pd.DataFrame(edges)
    if not df.empty:
        df["methods"] = df["methods"].map(lambda m: "; ".join(m))
    return df


def _curated_ocg(p: ProjectState) -> OntologicalCausalGraph:
    """The curated selection as a cw:OntologicalCausalGraph — what module 3 fits.

    This is the graph behind "Take n edges to inference": the edge set the user
    settled on, as a first-class OCG instance rather than a list of pairs. Two
    pieces of curation provenance ride along, because a curated graph that
    cannot say where its edges came from is not reproducible:
    ``cw:manuallyAdded`` marks the edges the analyst drew by hand, and
    ``prov:wasDerivedFrom`` names the ``cw:DiscoveryRun``s the rest were
    selected out of.
    """
    ctx = p.discovery_context
    if ctx is None:
        raise HTTPException(400, "No discovery context, so no curated graph to build.")
    names = list(ctx.column_names)
    index = {name: i for i, name in enumerate(names)}
    adj = np.zeros((len(names), len(names)), dtype=int)
    for s, t in p.selected_edges:
        if s in index and t in index:
            adj[index[s], index[t]] = 1
    manual = {
        (index[s], index[t])
        for s, t in (tuple(e) for e in p.manual_edges)
        if s in index and t in index
    }
    return OntologicalCausalGraph.from_discovery(
        adj, ctx.constraint_kept, method="curated",
        params={"n_runs": len(p.discovery_runs), "n_manual": len(p.manual_edges)},
        constrained=None, source=ctx.source,
        manual_edges=manual,
        derived_from=[str(_run_ocg(p, run).run_iri) for run in p.discovery_runs],
    )


def _curated_turtle(p: ProjectState) -> str:
    """Serialise the curated selection as a cw: OntologicalCausalGraph."""
    return _curated_ocg(p).to_rdf(with_vocabulary=True).serialize(format="turtle")


# --------------------------------------------------------------------------- #
# W27 — the vocabulary, served
#
# `http://sdm-causalway.org/` does not dereference and this project does not
# control the domain, so a consumer of an exported TTL has no way to look up
# what `cw:predictedValue` means. Serving the ontology here is the honest
# substitute: wherever this service is deployed, `GET /api/vocab` is a URL that
# returns the term definitions.
#
# It is generated by `vocabulary()` on each request rather than read from
# `causalway/ontology.ttl`, so what the service serves is by construction what
# the code means — a checked-in snapshot can go stale against `vocab.py` and
# nothing would notice. The snapshot file exists for readers who have the repo
# but not a running server; `python -m causalway.vocab` regenerates it.
# --------------------------------------------------------------------------- #
@app.get("/api/vocab")
def get_vocab(format: str = "turtle"):
    """The `cw:` vocabulary — classes, properties, labels, comments, alignments."""
    graph = vocabulary()
    if format in ("json", "json-ld", "jsonld"):
        return Response(graph.serialize(format="json-ld", indent=2),
                        media_type="application/ld+json")
    if format in ("nt", "ntriples", "n-triples"):
        return Response(graph.serialize(format="nt"), media_type="application/n-triples")
    if format not in ("turtle", "ttl"):
        raise HTTPException(400, f"Unknown vocabulary format {format!r}; "
                                 "expected turtle, json-ld or nt.")
    return Response(graph.serialize(format="turtle"), media_type="text/turtle")


def _run_ocg(p: ProjectState, run: dict) -> OntologicalCausalGraph:
    """One stored discovery run, rebuilt as the OCG that run learned.

    A run record keeps ``adj``/``weights``/``columns``, which is everything an
    OCG needs, so each run in the session is exportable as its own
    ``cw:OntologicalCausalGraph`` — with the seed that produced it. Node IRIs
    are global, so several of these accumulate into one store and join on the
    shared node resources (this is what ``results/*/all_methods.ttl`` is).
    """
    ctx = p.discovery_context
    if ctx is None:
        raise HTTPException(400, "No discovery context, so no learned graph to build.")
    return OntologicalCausalGraph.from_discovery(
        np.asarray(run["adj"], dtype=int), ctx.constraint_kept,
        weights=np.asarray(run["weights"], dtype=float),
        method=run["method"], constrained=run["constrained"], source=ctx.source,
        graph_id=f"{p.id}-run{run['run_id']}-{_slugify(run['method'])}",
        params={
            "alpha": run["alpha"], "priors": run["priors"], "seed": run["seed"],
            "library": _algorithm_library(run["method"]),
        },
    )


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(text)) or "x"


def _learned_graphs(p: ProjectState, run_ids: Optional[list[int]] = None) -> list:
    """The selected discovery runs as OCG instances (all of them when unfiltered)."""
    if not p.discovery_runs:
        raise HTTPException(400, "No discovery runs to export.")
    wanted = set(run_ids) if run_ids else None
    chosen = [r for r in p.discovery_runs if wanted is None or r["run_id"] in wanted]
    if not chosen:
        raise HTTPException(404, f"No discovery run matches {sorted(wanted or [])}.")
    return [_run_ocg(p, run) for run in chosen]


#: The export menu, grouped by the module that produces each artifact. A flat
#: list of fourteen items gave no hint of which stage of the pipeline an item
#: belonged to; the group is the only thing this ordering adds, and it is the
#: order the modules run in.
#:
#: Four former items are gone from the menu but still ride inside the project
#: zip: the induced schema (re-derivable from the source KG), the canvas layout
#: (UI state, not a research artifact), the GES-Prior matrices and the model
#: evaluation.
#:
#: The last two are *inputs and diagnostics*, not results. `priors` is the
#: (M, B) pair that was fed to GES-Prior \u2014 it is already reproduced by the run
#: it belongs to (`runs` carries the seed and the prior flag) and it was either
#: imported from a file the user already has or is re-derivable by re-running
#: the estimate; `evaluation` is a goodness-of-fit report about the fit, read
#: beside the inference board, not a thing downstream work consumes. Neither
#: earns a line in a menu whose other entries are artifacts you publish or
#: re-import.
#:
#: Both stay in `_artifacts` and in `_render`, so the zip still carries them and
#: a direct `?what=priors` / `?what=evaluation` URL still resolves. `priors.json`
#: in particular *must* stay in the zip: `bundle.restore` reads it back, and
#: dropping it would silently lose the priors across an export/import round trip.
_ARTIFACT_GROUPS = [
    ("preprocess", "1 \u00b7 Preprocess", ["nodes", "materialisation", "sparql"]),
    ("discovery", "2 \u00b7 Causal discovery",
     ["runs", "learned_graphs", "ledger", "graph"]),
    ("inference", "3 \u00b7 Inference", ["model", "answers"]),
    ("counterfactual", "4 \u00b7 Counterfactual", ["cf_worlds"]),
    ("project", "Everything", ["project"]),
]


def _artifacts(p: ProjectState) -> dict[str, dict]:
    mat = p.last_materialization
    return {
        "schema": {
            "label": "Induced schema", "formats": ["json"],
            "available": p.schema is not None, "reason": "Load a source first.",
        },
        "nodes": {
            "label": "Candidate nodes & curation", "formats": ["json", "csv"],
            "available": bool(p.nodes), "reason": "Load a source first.",
        },
        "materialisation": {
            "label": "Materialised frame", "formats": ["csv", "json"],
            "available": mat is not None, "reason": "Run Materialise in module 1.",
        },
        "sparql": {
            "label": "Emitted SPARQL", "formats": ["txt"],
            "available": mat is not None, "reason": "Run Materialise in module 1.",
        },
        "runs": {
            "label": "Discovery runs (with seeds)", "formats": ["json"],
            "available": bool(p.discovery_runs), "reason": "Run a discovery method.",
        },
        "learned_graphs": {
            "label": "Learned causal graphs", "formats": ["json", "turtle"],
            "available": bool(p.discovery_runs), "reason": "Run a discovery method.",
            # Each selected run as its own cw:OntologicalCausalGraph. Node IRIs
            # are global, so several of them accumulate into one store and join
            # on the shared node resources.
            "select": [
                {"id": r["run_id"],
                 "label": f"#{r['run_id']} {r['method']}"
                          + (" (constrained)" if r["constrained"] else "")
                          + f" \u2014 seed {r['seed']}"}
                for r in p.discovery_runs
            ],
            "select_param": "runs",
        },
        "ledger": {
            "label": "Edge ledger", "formats": ["json", "csv"],
            "available": p.discovery_context is not None, "reason": "Run a discovery method.",
        },
        "priors": {
            "label": "GES-Prior matrices (M, B)", "formats": ["json"],
            "available": p.priors is not None, "reason": "Estimate or import priors in module 2.",
        },
        "graph": {
            "label": "Curated causal graph", "formats": ["json", "turtle"],
            "available": bool(p.selected_edges), "reason": "Select edges in module 2.",
        },
        "model": {
            "label": ("Fitted Bayesian network & CPT summary" if p.model_kind == "cbn"
                      else "Fitted SCM & mechanisms"),
            # No turtle for a CBN: `CausalModel.to_rdf` describes gcm mechanisms, which
            # a network of conditional probability tables does not have (W22).
            "formats": ["json", "csv"] if p.model_kind == "cbn" else ["json", "csv", "turtle"],
            "available": p.model is not None,
            "reason": "Fit a model in the Causal model panel.",
        },
        "evaluation": {
            "label": "Model evaluation", "formats": ["json", "csv"],
            "available": p.evaluation is not None,
            "reason": "Run evaluation beside the inference board.",
        },
        "answers": {
            "label": "Query log (conditional, interventional, counterfactual)",
            # `ttl` needs the fitted model to resolve node IRIs, which `json`
            # and `csv` do not — the flat log is self-contained, the RDF is not.
            "formats": ["json", "csv", "ttl"] if p.model is not None
                       else ["json", "csv"],
            "available": bool(p.answers), "reason": "Ask a question in module 3 or 4.",
        },
        "cf_worlds": {
            "label": "Counterfactual worlds", "formats": ["json", "turtle"],
            "available": bool(p.cf_worlds),
            "reason": "Run a counterfactual in module 4.",
        },
        "layout": {
            "label": "Canvas layout", "formats": ["json"],
            "available": True, "reason": "",
        },
        "project": {
            "label": "Whole project (zip)", "formats": ["zip"],
            "available": p.schema is not None, "reason": "Load a source first.",
        },
    }


@app.get("/api/projects/{pid}/export/manifest")
def export_manifest(pid: str):
    """The export menu, grouped by the module that produces each artifact.

    `items` is kept flat alongside `groups` so an older client keeps working;
    it lists exactly the grouped items, i.e. without the four the menu dropped
    (the induced schema, the canvas layout, the GES-Prior matrices and the model
    evaluation — all four still inside the zip).
    """
    p = _get_project(pid)
    artifacts = _artifacts(p)
    groups = [
        {
            "id": gid,
            "label": label,
            "items": [{"what": what, **artifacts[what]} for what in names
                      if what in artifacts],
        }
        for gid, label, names in _ARTIFACT_GROUPS
    ]
    return {
        "groups": groups,
        "items": [item for group in groups for item in group["items"]],
    }


def _cf_worlds_payload(p: ProjectState, fmt: str) -> tuple[bytes, str, str]:
    """The session's counterfactual worlds, as JSON or as cw: Turtle.

    The JSON is the very document the RML mapping reads, so the two formats are
    one artifact in two syntaxes rather than two hand-kept shapes.

    `dtypes` is passed through because the datatype of an exported value is
    decided by how the model *fitted* the column, not by the node's declared
    `R_p`: a column the user discretised is a set of labels whatever
    `xsd:double` the ontology claims, and typing it as a double would be a lie
    the reader cannot detect. A world recorded before this was wired through
    has no `dtypes` key, and its values simply export untyped.
    """
    if not p.cf_worlds:
        raise HTTPException(400, "No counterfactual worlds yet.")
    from causalway.cf_export import (  # noqa: PLC0415 — heavy Plan 2 import
        CounterfactualWorld, worlds_to_json, worlds_to_rdf,
    )

    ocg = _curated_ocg(p)
    worlds = [
        CounterfactualWorld(
            entity=w.get("entity"), interventions=w["interventions"],
            counterfactual=w["counterfactual"], factual=w["factual"],
            distribution=w.get("distribution") or {}, std=w.get("std") or {},
            row_count=w.get("n_rows"), random_seed=w.get("random_seed"),
            model_id=w.get("model_id"),
            computed_at=w.get("computed_at"), coupling=w.get("coupling"),
            label=w.get("estimand"),
        )
        for w in p.cf_worlds
    ]
    dtypes = next((w["dtypes"] for w in reversed(p.cf_worlds) if w.get("dtypes")), None)
    if fmt == "turtle":
        graph = worlds_to_rdf(worlds, ocg, dtypes=dtypes, with_vocabulary=True)
        return graph.serialize(format="turtle").encode(), "text/turtle", "ttl"
    return (json.dumps(worlds_to_json(worlds, ocg, dtypes=dtypes),
                       indent=2, default=str).encode(),
            "application/json", "json")


def _answers_turtle(p: ProjectState, records: list[dict]) -> tuple[bytes, str, str]:
    """The module-3 query log as cw: Turtle, through `query_mapping.rml.ttl`.

    The session log is a flat dict per answer; the export rebuilds the `Query`
    and `Answer` it came from so the RDF is produced by the same mapping the
    engine and the CLI use, rather than by a second shape maintained here.

    One asymmetry is deliberate, and it is the vocabulary's, not this function's:
    an entity attached to a *conditional* or *interventional* query is not the
    query's subject — those are population-level questions — so it travels as
    `cw:onEntity` on each intervention. Only a counterfactual has a
    `cw:aboutEntity`, which is why `Query` refuses `entity=` for the other two.
    """
    if p.model is None:
        raise HTTPException(400, "Fit a model before exporting answers as RDF.")
    from causalway.queries import (  # noqa: PLC0415 — heavy Plan 2 import
        Answer, Condition, Intervention, Query,
    )
    from causalway.query_export import queries_to_rdf

    dtypes = (getattr(p.model, "manifest", None) or {}).get("dtypes")
    pairs = []
    for rec in records:
        kind = rec.get("kind")
        if kind not in ("conditional", "interventional", "counterfactual"):
            continue
        entity = rec.get("entity")
        act_entity = entity if kind != "conditional" else None
        query = Query(
            kind=kind,
            model_id=p.model.model_id or "unfitted",
            target=[rec["target"]],
            interventions=[
                Intervention(node=k, value=v, entity=act_entity)
                for k, v in (rec.get("interventions") or {}).items()
            ],
            reference=[
                Intervention(node=k, value=v)
                for k, v in (rec.get("reference") or {}).items()
            ],
            evidence=[
                Condition(node=k, value=v)
                for k, v in (rec.get("evidence") or {}).items()
            ],
            condition_on=[
                Condition(node=k, value=v)
                for k, v in (rec.get("conditions") or {}).items()
            ],
            entity=entity if kind == "counterfactual" else None,
            label=rec.get("estimand"),
        )
        answer = Answer(
            kind=kind, target=rec["target"], entity=entity,
            predicted=rec.get("predicted"), effect=rec.get("effect"),
            factual=rec.get("factual"), coupling=rec.get("coupling"),
        )
        pairs.append((query, answer))
    if not pairs:
        raise HTTPException(400, "None of those answers carry a replayable query.")

    graph = queries_to_rdf(pairs, p.model, dtypes=dtypes, with_vocabulary=True)
    return graph.serialize(format="turtle").encode(), "text/turtle", "ttl"


def _render(p: ProjectState, what: str, fmt: str,
            select: Optional[list[int]] = None) -> tuple[bytes, str, str]:
    """-> (payload, media_type, filename suffix)."""
    def as_json(obj) -> tuple[bytes, str, str]:
        return json.dumps(obj, indent=2, default=str).encode(), "application/json", "json"

    if what == "schema":
        return as_json(_schema_payload(p))
    if what == "nodes":
        if fmt == "csv":
            return _nodes_frame(p).to_csv(index=False).encode(), "text/csv", "csv"
        return as_json(get_nodes(p.id))
    if what == "materialisation":
        if p.last_materialization is None:
            raise HTTPException(400, "No materialisation yet.")
        df = p.last_materialization.df
        if fmt == "json":
            return as_json({
                "columns": [c.name for c in p.last_materialization.columns],
                "n_rows": int(df.shape[0]),
                "multiplicity": p.last_materialization.multiplicity,
                "warnings": p.last_materialization.warnings,
                "skipped_patterns": p.last_materialization.skipped_patterns,
                "distinct_counts": p.distinct_counts,
            })
        return df.to_csv(index=False).encode(), "text/csv", "csv"
    if what == "sparql":
        if p.last_materialization is None:
            raise HTTPException(400, "No materialisation yet.")
        return p.last_materialization.query.encode(), "text/plain", "sparql"
    if what == "runs":
        return as_json(p.discovery_runs)
    if what == "ledger":
        if fmt == "csv":
            return _ledger_frame(p).to_csv(index=False).encode(), "text/csv", "csv"
        return as_json(_ledger(p))
    if what == "graph":
        # JSON and Turtle are the same cw:OntologicalCausalGraph in two
        # syntaxes — the JSON is the document the RML mapping reads. (The UI's
        # own board shape stays available at GET /graph.)
        if fmt == "turtle":
            return _curated_turtle(p).encode(), "text/turtle", "ttl"
        return as_json(_curated_ocg(p).to_json())
    if what == "learned_graphs":
        graphs = _learned_graphs(p, select)
        if fmt == "turtle":
            from rdflib import Graph as _Graph  # noqa: PLC0415
            store = _Graph()
            vocabulary(store)
            for ocg in graphs:
                ocg.to_rdf(graph=store)
            return store.serialize(format="turtle").encode(), "text/turtle", "ttl"
        return as_json([ocg.to_json() for ocg in graphs])
    if what == "cf_worlds":
        return _cf_worlds_payload(p, fmt)
    if what == "priors":
        if p.priors is None:
            raise HTTPException(400, "No priors loaded.")
        # Same {columns, M, B} shape learn_bn._load_priors reads, so an export can be
        # fed straight back into an offline run — or re-imported here.
        return as_json(p.priors)
    if what == "model":
        if p.model is None:
            raise HTTPException(400, "No fitted model.")
        if fmt == "csv":
            # A CBN has no gcm mechanisms to tabulate — `mechanism_table()` reads
            # `scm.causal_mechanism` and raises. Its mechanisms are CPTs, and the
            # per-node summary the UI shows is what the export carries (W22).
            rows = ((p.model_info or _model_payload(p))["mechanisms"]
                    if p.model_kind == "cbn"
                    else p.model.mechanism_table().to_dict(orient="records"))
            return pd.DataFrame(rows).to_csv(index=False).encode(), "text/csv", "csv"
        if fmt == "turtle":
            if p.model_kind == "cbn":
                raise HTTPException(400, (
                    "`CausalModel.to_rdf` describes gcm mechanisms, which a causal "
                    "Bayesian network does not have. Export the model as JSON or CSV, "
                    "or refit as a structural causal model."
                ))
            return (p.model.to_rdf().serialize(format="turtle").encode(),
                    "text/turtle", "ttl")
        return as_json(p.model_info or _model_payload(p))
    if what == "evaluation":
        if p.evaluation is None:
            raise HTTPException(400, "No evaluation run yet.")
        if fmt == "csv":
            return pd.DataFrame(p.evaluation["cv"]).to_csv(index=False).encode(), "text/csv", "csv"
        return as_json(p.evaluation)
    if what == "answers":
        if not p.answers:
            raise HTTPException(400, "No answers yet.")
        records = p.answers
        if select:
            wanted = set(select)
            records = [a for a in p.answers if a.get("answer_id") in wanted]
            if not records:
                raise HTTPException(
                    404, f"No answer with id in {sorted(wanted)}; the log keeps the "
                         "last 50 of a session and is lost on restart.")
        if fmt == "csv":
            flat = [
                {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
                 for k, v in a.items()}
                for a in records
            ]
            return pd.DataFrame(flat).to_csv(index=False).encode(), "text/csv", "csv"
        if fmt in ("ttl", "turtle"):
            return _answers_turtle(p, records)
        return as_json(records)
    if what == "layout":
        return as_json(p.layouts)
    raise HTTPException(400, f"Unknown artifact '{what}'.")


@app.get("/api/projects/{pid}/export")
def export(pid: str, what: str = "project", format: str = "json",
           runs: Optional[str] = None):
    """`runs` is a comma-separated list of run ids, for `what=learned_graphs`."""
    p = _get_project(pid)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        select = [int(r) for r in runs.split(",") if r.strip()] if runs else None
    except ValueError:
        raise HTTPException(400, f"runs= must be comma-separated integers, got {runs!r}")
    if what == "project":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps({
                "id": p.id, "name": p.name, "created_at": p.created_at,
                "source": p.source, "constraint_options": p.constraint_options,
                "excluded": sorted(p.excluded), "excluded_joins": sorted(p.excluded_joins),
                "component": p.last_component,
                # W29: the UI's own state, as itself. `curated_graph.json` is the
                # RML document and is the publishable artefact; these two lists
                # are what an import puts back on the board, without needing an
                # inverse of the RDF export.
                "selected_edges": [list(e) for e in p.selected_edges],
                "manual_edges": [list(e) for e in p.manual_edges],
                "type_options": dict(p.type_options),
                "prior_meta": dict(p.prior_meta),
            }, indent=2, default=str))
            failed: list[str] = []
            for what_i, fmt_i, name in [
                ("schema", "json", "schema.json"),
                ("nodes", "csv", "nodes.csv"),
                ("materialisation", "csv", "materialisation.csv"),
                ("sparql", "txt", "materialisation.sparql"),
                ("runs", "json", "discovery_runs.json"),
                ("ledger", "csv", "edge_ledger.csv"),
                ("graph", "json", "curated_graph.json"),
                ("graph", "turtle", "curated_graph.ttl"),
                ("learned_graphs", "json", "learned_graphs.json"),
                ("learned_graphs", "turtle", "learned_graphs.ttl"),
                ("priors", "json", "priors.json"),
                ("model", "json", "model.json"),
                ("model", "csv", "mechanisms.csv"),
                ("model", "turtle", "model.ttl"),
                ("evaluation", "csv", "evaluation.csv"),
                ("answers", "json", "answers.json"),
                ("cf_worlds", "json", "counterfactual_worlds.json"),
                ("cf_worlds", "turtle", "counterfactual_worlds.ttl"),
                ("layout", "json", "layout.json"),
            ]:
                try:
                    payload, _mt, _sfx = _render(p, what_i, fmt_i)
                except HTTPException:
                    continue  # artifact not produced yet; a partial project is still exportable
                except Exception as exc:  # noqa: BLE001
                    # This archive is the only durable copy of the session (W14),
                    # so one artifact that cannot render must not take the other
                    # seventeen with it — that failure mode loses the user's work
                    # to protect nothing. Recorded in the zip rather than swallowed,
                    # so a bug here is still visible to whoever opens it.
                    failed.append(f"{name}: {type(exc).__name__}: {exc}")
                    continue
                z.writestr(name, payload)

            # W29: the fitted model, as `model/` inside the same archive — one
            # importable artifact rather than a project zip and a model zip that
            # have to be kept together by hand. Absent when nothing is fitted.
            has_model = False
            if p.model is not None and p.model_kind == "scm":
                try:
                    has_model = bundle.write_model_into(z, p.model)
                except Exception as exc:  # noqa: BLE001 — a partial project still exports
                    z.writestr("model/ERROR.txt",
                               f"The fitted model could not be serialised: {exc}\n")
            if failed:
                z.writestr("ERRORS.txt",
                           "These artifacts could not be rendered and are missing from\n"
                           "this archive. Everything else in it is complete.\n\n"
                           + "\n".join(failed) + "\n")
            z.writestr(bundle.DESCRIPTOR, json.dumps(
                bundle.bundle_descriptor(p, has_model=has_model, versions=_library_versions()),
                indent=2, default=str))

            z.writestr("README.txt",
                       "CausalWay project export.\n"
                       "This service keeps no persistent user data (Plan 3 W14); this archive is\n"
                       "the only durable copy of the session. Re-import it with the Import\n"
                       "button, or POST it to /api/projects/import — that always creates a NEW\n"
                       "project and never overwrites one.\n"
                       "\n"
                       "The source knowledge graph is NOT in this archive, and neither is the\n"
                       "model's training data. Re-attach the source after importing to get\n"
                       "modules 1 and 2 back; the model in model/ answers module 3 and 4\n"
                       "queries without it, against a hypothetical unit.\n"
                       "\n"
                       "See 'Plan3 (web service).md' §8.2 for the headless reproduction path\n"
                       "through runners/.\n")
        return Response(
            buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="causalway-{p.id}-{stamp}.zip"'},
        )

    payload, media_type, suffix = _render(p, what, format, select)
    return Response(
        payload, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="causalway-{what}-{stamp}.{suffix}"'},
    )


# --------------------------------------------------------------------------- #
# W29 — import
#
# The counterpart to the export above, and the reason the two are one archive.
# See `service/api/bundle.py` for why it is a single bundle, why an import never
# overwrites, and why the source KG is re-attached rather than carried.
#
# The import machinery lives in that module rather than here: main.py is
# already ~4000 lines, and this is the one piece of it whose failure modes are
# entirely about *file* validity (corrupt zip, wrong version, mismatched
# sklearn, a KG that is not the one the model was fitted on) rather than about
# project state. Keeping those checks together, testable without a running
# FastAPI app, is worth the extra file. main.py itself is not split this round.
# --------------------------------------------------------------------------- #
def _library_versions() -> dict:
    import sklearn  # noqa: PLC0415

    import dowhy  # noqa: PLC0415
    return {
        "dowhy": getattr(dowhy, "__version__", "unknown"),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }


def _load_bundled_model(loaded, strict: bool):
    """`model/` -> a live `CausalModel`, or `None` plus an explanation."""
    from causalway.model import CausalModel  # noqa: PLC0415 — heavy Plan 2 import

    with tempfile.TemporaryDirectory() as tmp:
        directory = bundle.write_model_dir(loaded, os.path.join(tmp, "model"))
        try:
            # `reload_data=False`: the source KG is not here, and letting
            # `load` chase the manifest's recorded path would silently
            # re-materialise whatever happens to sit at that path on *this*
            # machine — the exact substitution `verify_source` exists to
            # prevent. The graph still loads, from the bundle's own copy.
            model = CausalModel.load(directory, strict=strict, reload_data=False)
        except Exception as exc:  # noqa: BLE001 — reported, not fatal
            return None, [f"The bundled model could not be loaded ({exc}); "
                          "the project imported without it."]
        ocg = CausalModel._reload_ocg(model.manifest, directory)
        if ocg is not None:
            model.spec.ocg = ocg
        notes = ["The model is loaded but has no rows behind it: entity search, "
                 "evaluation and the marginals need the source KG. Counterfactuals "
                 "on a hypothetical unit work now."]
        return model, notes


@app.post("/api/projects/import")
async def import_project(file: UploadFile = File(...), strict: bool = True):
    """Restore a project (and its model, if the bundle has one) as a NEW project.

    Never overwrites: `strict=false` relaxes the library-version check on the
    pickled model, and nothing relaxes the "new project" rule. The service has
    no undo and no second copy, so an import that could target an existing
    project would be one mis-click away from destroying an afternoon.
    """
    data = await file.read()
    try:
        loaded = bundle.read_bundle(data, installed=_library_versions(), strict=strict)
    except bundle.BundleError as exc:
        raise HTTPException(400, str(exc)) from exc

    p = ProjectState(name="Imported project")
    try:
        notes = bundle.restore_into(
            loaded, p, load_model=lambda b: _load_bundled_model(b, strict))
    except bundle.BundleError as exc:
        raise HTTPException(400, str(exc)) from exc

    if p.model is not None:
        p.model_info = _model_payload(p)
        p.model_key = None   # nothing to compare against until a source is attached
    p.import_notes = notes
    p.imported_source = (loaded.descriptor.get("source_kg") or {})
    PROJECTS[p.id] = p
    return {
        "id": p.id, "name": p.name,
        "bundle_version": loaded.descriptor.get("bundle_version"),
        "has_model": p.model is not None,
        "notes": notes,
        "source_hint": p.imported_source,
        "capabilities": _capabilities(p),
    }


@app.post("/api/projects/{pid}/sources/reattach")
async def reattach_source(pid: str, file: UploadFile = File(...), force: bool = False):
    """Attach the source KG an imported project was exported from.

    The offered file is hashed and compared against the sha256 the bundle
    recorded. A mismatch is refused: fitted mechanisms plus different rows is a
    failure with no symptom — every query still answers, and every answer is
    about rows that are not there. `force=true` is the deliberate override, and
    it drops the model rather than keeping it, because keeping it is exactly the
    combination being refused.
    """
    p = _get_project(pid)
    data = await file.read()
    expected = (getattr(p, "imported_source", None) or {}).get("sha256")
    try:
        digest = bundle.verify_source(data, None if force else expected)
    except bundle.BundleError as exc:
        raise HTTPException(409, str(exc)) from exc

    suffix = Path(file.filename or "source.ttl").suffix or ".ttl"
    handle, path = tempfile.mkstemp(suffix=suffix, prefix="causalway-src-")
    with os.fdopen(handle, "wb") as fh:
        fh.write(data)

    keep_model = p.model
    try:
        schema = resolve_schema(path)
    except Exception as exc:  # noqa: BLE001
        os.unlink(path)
        raise HTTPException(400, f"Could not parse that knowledge graph: {exc}") from exc

    excluded, excluded_joins = set(p.excluded), set(p.excluded_joins)
    selected, manual, layouts = list(p.selected_edges), list(p.manual_edges), p.layouts
    _set_schema(p, schema, {
        "kind": "file", "name": file.filename, "sha256": digest,
        "reattached": True,
    }, path)
    # `_set_schema` resets curation to defaults, which is right for a *new*
    # source and wrong here: this is the same KG the curation was made against,
    # verified by hash, so putting it back is restoring state rather than
    # guessing at it. Node names that no longer exist are dropped silently —
    # they cannot, given the hash matched, but a stale name would otherwise
    # fail a validation the user has no way to act on.
    names = {n.name for n in p.nodes}
    p.excluded = {n for n in excluded if n in names}
    p.excluded_joins = {n for n in excluded_joins if n in names}
    p.selected_edges, p.manual_edges, p.layouts = selected, manual, layouts

    if force and expected and digest != expected:
        p.model = None
        p.model_info = None
        note = ("Attached a different knowledge graph than the model was fitted on, "
                "as requested. The model was dropped — refit against these rows.")
    else:
        p.model = keep_model
        note = "Source re-attached and verified against the bundle's recorded hash."
        if keep_model is not None:
            # Give the model its rows back. The hash matched, so re-materialising
            # from this file reproduces exactly the frame it was fitted on —
            # including `entity_ids`, which is the only record of which entity
            # owns which row and therefore the only thing that makes an
            # entity-level counterfactual possible at all. Without this step a
            # verified re-attach would restore modules 1 and 2 and leave module 4
            # stuck on hypothetical units, which is the half-feature the
            # re-attach exists to avoid.
            keep_model.manifest["source_kg"] = {"kind": "file", "path": path,
                                                "sha256": digest}
            mat, data = type(keep_model)._reload_data(keep_model.manifest)
            if mat is not None:
                keep_model.spec.mat, keep_model.spec.data = mat, data
                note += " The model's rows were re-materialised, so entity-level queries work again."
            else:
                note += (" The model kept its mechanisms but could not re-materialise its "
                         "rows from this file; entity-level queries stay unavailable.")
        p.model_info = _model_payload(p) if keep_model is not None else None

    p.import_notes = [note]
    return {"ok": True, "note": note, "capabilities": _capabilities(p),
            "schema": {"n_classes": len(schema.classes), "n_nodes": len(p.nodes)}}


def _capabilities(p: ProjectState) -> dict:
    """Which panels can do anything, and why not when they cannot.

    An imported project has a model and no KG, which no panel had to cope with
    before — module 1 and 2 have nothing to show, and module 3's evaluation and
    entity search have no rows even though the model itself is live. Returning
    the reason with the flag is the difference between a greyed panel the user
    can act on and one that just looks broken.
    """
    has_source = p.schema is not None
    has_frame = p.last_materialization is not None
    has_model = p.model is not None
    reason_no_source = ("This project was imported without its source knowledge graph. "
                        "Re-attach it to use this panel.")
    return {
        "source": {"available": True, "reason": None},
        "ontology": {"available": has_source, "reason": None if has_source else reason_no_source},
        "nodes": {"available": has_source, "reason": None if has_source else reason_no_source},
        "materialise": {"available": has_source,
                        "reason": None if has_source else reason_no_source},
        "discovery": {"available": has_frame,
                      "reason": None if has_frame else (
                          reason_no_source if not has_source
                          else "Materialise the join first — discovery runs on the frame.")},
        "model": {"available": has_source or has_model,
                  "reason": None if (has_source or has_model) else reason_no_source},
        "inference": {"available": has_model,
                      "reason": None if has_model else "No fitted model yet."},
        "counterfactual": {
            "available": has_model and p.model_kind == "scm",
            "reason": None if (has_model and p.model_kind == "scm") else (
                "No fitted model yet." if not has_model
                else "A causal Bayesian network has no noise term to abduct."),
        },
        # Entity-level work specifically needs the *rows*, not just the model —
        # an imported model can still answer for a hypothetical unit.
        "entities": {
            "available": has_model and p.model.spec.mat is not None,
            "reason": None if (has_model and p.model.spec.mat is not None) else (
                "No rows behind this model — describe a hypothetical unit instead, "
                "or re-attach the source knowledge graph."),
        },
        "import_notes": list(getattr(p, "import_notes", []) or []),
    }


@app.get("/api/projects/{pid}/capabilities")
def get_capabilities(pid: str):
    return _capabilities(_get_project(pid))


# --------------------------------------------------------------------------- #
# Static frontend
#
# The React app (Plan3 W10) builds to web/dist. Serve that when it exists so a
# single `uvicorn` on one port is the whole deployment; during development run
# `npm run dev` in web/ instead, which proxies /api here.
# --------------------------------------------------------------------------- #
WEB_DIST = REPO_ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
else:
    @app.get("/")
    def _no_build():
        return Response(
            "<h1>CausalWay</h1><p>The frontend is not built yet. Run "
            "<code>npm install &amp;&amp; npm run build</code> in <code>web/</code>, "
            "or <code>npm run dev</code> for the dev server on port 5173.</p>",
            media_type="text/html",
        )
