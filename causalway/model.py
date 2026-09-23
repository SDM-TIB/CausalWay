"""``CausalModel`` — fit a ``dowhy.gcm`` SCM over an :class:`OntologicalCausalGraph` + KG.

Implements Plan 2 §3.3: align the causal graph's nodes to the KG's
materialised columns on node *identity* (``domain, prop, range``, not the
display-level ``name``), infer a dtype per column, let ``gcm.auto`` pick a
mechanism class, override every non-root categorical mechanism with
:class:`~causalway.mechanisms.InvertibleClassifierFCM` (E1), pre-validate
invertibility (gcm accepts a non-invertible mechanism silently and only fails
later, deep inside ``compute_noise_from_data`` — see Plan 2 §1.2), then fit.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

import dowhy
import sklearn
from dowhy import gcm
from dowhy.gcm.auto import AssignmentQuality
from dowhy.gcm.causal_mechanisms import (
    ClassifierFCM, DiscreteAdditiveNoiseModel, InvertibleFunctionalCausalModel,
)
from dowhy.gcm.util.general import set_random_seed
from dowhy.graph import is_root_node

from .bgp import Materialization, materialize
from .mechanisms import InvertibleClassifierFCM
from .nodes import PropertyNode, build_nodes
from .ontology import OntologySchema
from .result import OntologicalCausalGraph
from .sources import CausalGraphSource, GraphSource, is_endpoint, load_ocg, resolve_schema
from .vocab import CW, MODEL_RUN_STEM, MODEL_STEM, PROV

__all__ = ["ModelSpec", "CausalModel", "GRAPH_FILENAME"]

#: The OCG written beside ``scm.pkl`` by :meth:`CausalModel.save`, so a saved
#: model does not depend on the discovery output still sitting where it was
#: when the model was fitted. See :meth:`CausalModel.save`.
GRAPH_FILENAME = "curated_graph.ttl"

_QUALITY = {
    "good": AssignmentQuality.GOOD,
    "better": AssignmentQuality.BETTER,
    "best": AssignmentQuality.BEST,
}


def _identity(n: PropertyNode) -> tuple:
    return (str(n.domain), str(n.prop), str(n.range_))


def _materialize_kg(schema: OntologySchema, *, include_object_properties: bool,
                     object_value: str, key_properties: Optional[dict],
                     limit: Optional[int], excluded: Optional[set] = None,
                     excluded_joins: Optional[set] = None):
    """Materialise the flat join this model will be fitted on.

    ``build_nodes(include_object_properties=False)`` never builds an
    object-property node — that flag was meant to control whether a relation
    becomes a *value column*, but withholding the node also withholds its
    join, since :func:`~causalway.bgp.materialize` only joins classes through
    object nodes it is actually given. A schema whose classes are only
    connected through such a relation then has no way to reach a second
    class's properties: every active class ends up in its own disconnected
    group and the join is refused (``DisconnectedJoinError``) rather than
    silently computing their cross product. So the node is always built, and
    dropped as a *candidate variable* only, via ``excluded`` — the join
    itself stays via ``excluded_joins=set()``.

    ``excluded`` / ``excluded_joins`` let a caller that already curated and
    ran its own materialisation hand that curation straight through, so the
    fit re-runs the *same* SPARQL rather than a full-schema one of its own
    (Plan 3 §Preprocess revision 2 — the two axes are independent; see
    :mod:`causalway.bgp`'s module docstring). Left at ``None`` — every
    non-service caller — the behaviour is exactly the paragraph above: every
    property materialised, every join intact.

    This matters beyond column agreement. ``Materialization.entity_ids`` is
    what makes a *counterfactual* entity-level at all: it is the only record
    of which entity owns which cell, and both :func:`causalway.entities.
    population_rows` and the unit builder read a fit's own ``spec.mat`` for
    it. Materialising a different join here than the one the user curated
    would resolve "this patient's rows" against a row set they never saw.
    """
    kg_nodes = build_nodes(schema, include_object_properties=True)
    if excluded is None:
        excluded = (
            {n.name for n in kg_nodes if n.kind == "object"}
            if not include_object_properties else set()
        )
    return kg_nodes, materialize(
        schema, kg_nodes, excluded=set(excluded),
        excluded_joins=set() if excluded_joins is None else set(excluded_joins),
        include_object_properties=include_object_properties,
        object_value=object_value, key_properties=key_properties, limit=limit,
    )


@dataclass
class ModelSpec:
    """Everything :class:`CausalModel` needed to fit, kept for inspection and evaluation."""

    ocg: Optional[OntologicalCausalGraph]     # aligned to `columns`, acyclic
    columns: list                             # SCM node names == PropertyNode.name
    data: Optional[pd.DataFrame]              # prepared (dtype-cast) flat-join frame
    mat: Optional[Materialization]
    schema: Optional[OntologySchema]
    dtypes: dict                              # name -> "categorical"|"discrete"|"continuous"|"ordinal"
    alignment: dict                           # matched / missing_in_kg / extra_in_kg (node names)


def _align(ocg_nodes: list, kg_nodes: list):
    """Match OCG nodes to KG nodes on ``(domain, prop, range)`` identity, not ``name``."""
    kg_by_key: dict = {}
    for n in kg_nodes:
        kg_by_key.setdefault(_identity(n), n)

    matched_idx, matched_kg, missing_in_kg = [], [], []
    for i, on in enumerate(ocg_nodes):
        kn = kg_by_key.get(_identity(on))
        if kn is None:
            missing_in_kg.append(on)
        else:
            matched_idx.append(i)
            matched_kg.append(kn)

    matched_keys = {_identity(n) for n in matched_kg}
    extra_in_kg = [n for n in kg_nodes if _identity(n) not in matched_keys]
    return matched_idx, matched_kg, missing_in_kg, extra_in_kg


def _subset_ocg(ocg: OntologicalCausalGraph, keep_idx: list, rename_to: list) -> OntologicalCausalGraph:
    """Restrict ``ocg`` to ``keep_idx`` (structure) while using ``rename_to`` for node identity/names.

    Structure (adjacency, weights, edge labels) comes from ``ocg``; the node
    objects themselves come from ``rename_to`` (the KG's own build, so
    ``.name`` matches the materialised DataFrame's column names exactly).
    """
    adj = ocg.adj[np.ix_(keep_idx, keep_idx)]
    weights = None if ocg.weights is None else ocg.weights[np.ix_(keep_idx, keep_idx)]
    edge_labels = {}
    for ni, i in enumerate(keep_idx):
        for nj, j in enumerate(keep_idx):
            if (i, j) in ocg.edge_labels:
                edge_labels[(ni, nj)] = ocg.edge_labels[(i, j)]
    return OntologicalCausalGraph(
        nodes=list(rename_to), adj=adj, edge_labels=edge_labels, weights=weights,
        method=ocg.method, params=dict(ocg.params), constrained=ocg.constrained,
        source=ocg.source, graph_id=ocg.graph_id,
    )


def _infer_dtype(series: pd.Series, is_ordinal: bool, max_levels: int,
                 float_as_continuous: bool = True) -> str:
    """categorical (object/string, or <= max_levels distinct) | discrete (int) | continuous.

    ``max_levels`` only ever applies to *integer-valued* numeric columns: it is the
    line between "a small set of unordered labels" and "a number". A column whose
    values are genuinely fractional is continuous however few distinct values it
    happens to take, because binding 0.5/1.0/2.0 into three unrelated labels throws
    away the ordering the numbers already carry. Pass
    ``float_as_continuous=False`` for the pre-2026-09 behaviour, where a float
    column with few levels was also treated as categorical.
    """
    if is_ordinal:
        return "ordinal"
    coerced = pd.to_numeric(series, errors="coerce")
    is_numeric = coerced.notna().all()
    if not is_numeric:
        return "categorical"
    integral = bool((coerced.dropna() % 1 == 0).all())
    if not integral:
        return "continuous" if float_as_continuous else (
            "categorical" if series.nunique(dropna=True) <= max_levels else "continuous"
        )
    return "categorical" if series.nunique(dropna=True) <= max_levels else "discrete"


def _fit_manifest(ocg_aligned, columns, dtypes, ordinal_maps, prepared, alignment,
                  causal_graph, kg, include_object_properties, object_value,
                  key_properties, limit, quality, categorical_counterfactual,
                  random_state, excluded=None, excluded_joins=None) -> dict:
    """Everything a fitted model records about how it came to exist."""
    return {
        "model_id": f"{ocg_aligned.gid}-{ocg_aligned.fingerprint()[:8]}",
        "columns": columns,
        "dtypes": dtypes,
        "ordinal_maps": ordinal_maps,
        "category_levels": {
            col: sorted(prepared[col].unique().tolist())
            for col, dt in dtypes.items() if dt == "categorical"
        },
        # The numeric counterpart of `category_levels`, and recorded for the same
        # reason: these are the only description of a column's domain that
        # survives without the training rows. A model loaded from a bundle has no
        # frame to compute them from, and a UI that has to offer "type a value for
        # this variable" needs to know whether the model ever saw anything near
        # what the user typed. Summary statistics of a column, not the column.
        "numeric_ranges": {
            col: {
                "min": float(pd.to_numeric(prepared[col], errors="coerce").min()),
                "max": float(pd.to_numeric(prepared[col], errors="coerce").max()),
                "mean": float(pd.to_numeric(prepared[col], errors="coerce").mean()),
            }
            for col, dt in dtypes.items() if dt not in ("categorical",)
        },
        "alignment": alignment,
        "ocg_gid": ocg_aligned.gid,
        "ocg_fingerprint": ocg_aligned.fingerprint(),
        "causal_graph_source": _kg_provenance(causal_graph),
        "materialize_kwargs": {
            "include_object_properties": include_object_properties,
            "object_value": object_value,
            "key_properties": key_properties,
            "limit": limit,
            # Recorded so `load(reload_data=True)` re-runs the *same* join, and with
            # it the same entity_ids a counterfactual resolves its unit against.
            # `null` (not `[]`) means "uncurated", which is not the same thing.
            "excluded": None if excluded is None else sorted(excluded),
            "excluded_joins": None if excluded_joins is None else sorted(excluded_joins),
        },
        "source_kg": _kg_provenance(kg),
        "quality": quality if isinstance(quality, str) else quality.name.lower(),
        "categorical_counterfactual": categorical_counterfactual,
        "random_state": random_state,
        "training_rows": int(len(prepared)),
        "versions": {
            "dowhy": getattr(dowhy, "__version__", "unknown"),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _node_seeds(columns, random_state: Optional[int]) -> dict:
    """``{column: seed}`` derived from one ``random_state``, stable under node order.

    Keyed off the *sorted* column names, so a node's stream depends on its name
    and the model's seed and on nothing else: adding an unrelated node to the
    graph must not silently move the counterfactual answers already published
    for the others.  ``SeedSequence.spawn`` is used rather than ``seed + i``
    because spawned sequences are independent by construction, while integer
    offsets of one seed are not.

    ``random_state=None`` yields ``None`` for every node — an unseeded fit stays
    unseeded, and :class:`~causalway.mechanisms.InvertibleClassifierFCM` then
    behaves exactly as it did before mechanisms carried generators.
    """
    ordered = sorted(columns)
    if random_state is None:
        return {col: None for col in ordered}
    children = np.random.SeedSequence(random_state).spawn(len(ordered))
    return {col: int(child.generate_state(1, dtype=np.uint32)[0])
            for col, child in zip(ordered, children)}


def _kg_provenance(kg_source) -> dict:
    if isinstance(kg_source, str) and is_endpoint(kg_source):
        return {"kind": "endpoint", "url": kg_source,
                "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if isinstance(kg_source, (str, os.PathLike)) and os.path.exists(os.fspath(kg_source)):
        path = os.fspath(kg_source)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        return {"kind": "file", "path": path, "sha256": digest}
    return {"kind": type(kg_source).__name__}


@dataclass
class CausalModel:
    """A fitted ``dowhy.gcm.InvertibleStructuralCausalModel`` plus its provenance."""

    scm: "gcm.InvertibleStructuralCausalModel"
    spec: ModelSpec
    fit_summary: object = None
    ordinal_maps: dict = field(default_factory=dict)      # column -> {code: original_label}
    manifest: dict = field(default_factory=dict)

    @property
    def model_id(self) -> Optional[str]:
        return self.manifest.get("model_id")

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, causal_graph: CausalGraphSource, kg: GraphSource, *,
           quality: str = "good",
           categorical_counterfactual: str = "gumbel-max",
           ordinal: Optional[list] = None,
           on_missing: str = "error",
           on_cycle: str = "strict",
           cycle_constraint=None,
           include_object_properties: bool = True,
           object_value: str = "range_class",
           key_properties: Optional[dict] = None,
           excluded: Optional[set] = None,
           excluded_joins: Optional[set] = None,
           limit: Optional[int] = None,
           max_levels: int = 20,
           float_as_continuous: bool = True,
           fit_mechanisms: bool = True,
           random_state: Optional[int] = None) -> "CausalModel":
        """Fit an SCM over ``causal_graph`` using data materialised from ``kg``.

        See ``Plan2 (causal model & prediction).md`` §3.3 for the full step
        list; summarised in the module docstring above.

        ``excluded`` / ``excluded_joins`` are the curation of the join itself
        (see :func:`_materialize_kg`). Pass the same two sets the upstream
        materialisation used and this fit sees the same rows, the same
        columns and the same ``entity_ids`` it did; leave them ``None`` and
        the whole schema is materialised with every join intact, as before.
        """
        if categorical_counterfactual != "gumbel-max":
            raise ValueError(
                f"Unsupported categorical_counterfactual={categorical_counterfactual!r}; "
                "only 'gumbel-max' (E1) is implemented. List categorical columns that need "
                "a different (order-based) coupling in ordinal= instead."
            )
        if random_state is not None:
            set_random_seed(random_state)

        ordinal_set = set(ordinal or [])

        # 1. Resolve inputs.
        ocg = load_ocg(causal_graph)
        schema = resolve_schema(kg)

        # 2. Materialise the KG independently, using the KG's *own* node build
        #    (names may differ from the OCG's, hence identity alignment below).
        kg_nodes, mat = _materialize_kg(
            schema, include_object_properties=include_object_properties,
            object_value=object_value, key_properties=key_properties, limit=limit,
            excluded=excluded, excluded_joins=excluded_joins,
        )
        if isinstance(mat, list):
            raise ValueError(
                "CausalModel.fit: the KG has multiple connected components; materialize "
                "and fit each component separately."
            )

        # 3. Align OCG nodes to KG nodes on (domain, prop, range) identity.
        matched_idx, matched_kg, missing_in_kg, extra_in_kg = _align(ocg.nodes, kg_nodes)
        if missing_in_kg:
            names = [n.name for n in missing_in_kg]
            if on_missing == "error":
                raise ValueError(
                    "CausalModel.fit: the causal graph has node(s) the KG cannot supply: "
                    f"{names}. Pass on_missing='drop' to marginalise them — NOT a neutral "
                    "act: if one was a confounder, the surviving mechanisms are biased."
                )
            if on_missing != "drop":
                raise ValueError(f"Unknown on_missing: {on_missing!r}")
            dropped_idx = set(range(len(ocg.nodes))) - set(matched_idx)
            for d in dropped_idx:
                losers = [ocg.nodes[j].name for j in matched_idx if ocg.adj[d, j] == 1]
                if losers:
                    print(f"CausalModel.fit: dropping {ocg.nodes[d].name!r} removes a "
                         f"parent of {losers}")

        alignment = {
            "matched": [n.name for n in matched_kg],
            "missing_in_kg": [n.name for n in missing_in_kg],
            "extra_in_kg": [n.name for n in extra_in_kg],
        }

        # 4. Structure: subset OCG to matched nodes (renamed to the KG's own
        #    PropertyNode objects so names match the DataFrame), then acyclify.
        ocg_aligned = _subset_ocg(ocg, matched_idx, matched_kg)
        ocg_aligned = ocg_aligned.to_dag(strategy=on_cycle, constraint=cycle_constraint)

        columns = [n.name for n in ocg_aligned.nodes]
        df = mat.df[columns].copy()

        # 5. Dtype inference + casting.
        dtypes: dict = {}
        ordinal_maps: dict = {}
        prepared = df.copy()
        for node in ocg_aligned.nodes:
            col = node.name
            dt = _infer_dtype(prepared[col], col in ordinal_set, max_levels,
                              float_as_continuous=float_as_continuous)
            dtypes[col] = dt
            if dt == "categorical":
                prepared[col] = prepared[col].astype(str)
            elif dt == "ordinal":
                codes, uniques = pd.factorize(prepared[col], sort=True)
                ordinal_maps[col] = {int(c): (None if pd.isna(v) else v) for c, v in enumerate(uniques)}
                prepared[col] = codes.astype(int)
            elif dt == "discrete":
                prepared[col] = pd.to_numeric(prepared[col], errors="coerce").round().astype(int)
            else:
                prepared[col] = pd.to_numeric(prepared[col], errors="coerce").astype(float)

        # 6. Graph -> gcm.auto model selection.
        g = nx.DiGraph()
        g.add_nodes_from(columns)
        for i, j in ocg_aligned.edges():
            g.add_edge(ocg_aligned.nodes[i].name, ocg_aligned.nodes[j].name)
        scm = gcm.InvertibleStructuralCausalModel(g)

        if not fit_mechanisms:
            # Structure + prepared frame only. A discrete Bayesian network estimates its
            # own CPTs from `spec.data` and never touches a gcm mechanism, so paying for
            # gcm.auto's model selection would be ~30 s of work thrown away. Nothing that
            # reads `scm.causal_mechanism` may run against a model fitted this way.
            spec = ModelSpec(ocg=ocg_aligned, columns=columns, data=prepared, mat=mat,
                             schema=schema, dtypes=dtypes, alignment=alignment)
            manifest = _fit_manifest(
                ocg_aligned, columns, dtypes, ordinal_maps, prepared, alignment,
                causal_graph, kg, include_object_properties, object_value,
                key_properties, limit, quality, categorical_counterfactual, random_state,
                excluded=excluded, excluded_joins=excluded_joins,
            )
            manifest["mechanisms_fitted"] = False
            return cls(scm=scm, spec=spec, fit_summary=None,
                       ordinal_maps=ordinal_maps, manifest=manifest)

        q = _QUALITY[quality.lower()] if isinstance(quality, str) else quality
        for col, dt in dtypes.items():
            if dt == "ordinal":
                # Invertible out of the box; skip gcm.auto's own categorical/continuous
                # selection for this node and assign it directly when non-root.
                if not is_root_node(scm.graph, col):
                    scm.set_causal_mechanism(col, DiscreteAdditiveNoiseModel(
                        prediction_model=gcm.ml.create_hist_gradient_boost_regressor()))
        fit_summary = gcm.auto.assign_causal_mechanisms(
            scm, prepared, quality=q, override_models=False,
        )

        # 7. Mechanism override (E1): non-root ClassifierFCM -> InvertibleClassifierFCM.
        #
        # Each override gets its own seed, derived from `random_state` by *node
        # name* rather than by the order gcm happens to enumerate nodes in. Two
        # reasons, both about the answer rather than the code: a single shared
        # stream would make every counterfactual a function of that traversal
        # order (add an unrelated node and previously-published numbers move),
        # and `SeedSequence.spawn` gives independent streams rather than
        # correlated offsets of one. Unseeded stays unseeded — `_node_seeds`
        # returns None per node, which is exactly the old global-`np.random`
        # behaviour minus the global part.
        seeds = _node_seeds(columns, random_state)
        for col in columns:
            if is_root_node(scm.graph, col):
                continue
            mech = scm.causal_mechanism(col)
            if isinstance(mech, ClassifierFCM) and not isinstance(mech, InvertibleClassifierFCM):
                scm.set_causal_mechanism(col, InvertibleClassifierFCM.from_classifier_fcm(
                    mech, random_state=seeds[col]))

        # 8. Invertibility pre-validation (gcm accepts a non-invertible mechanism
        #    silently and only fails later inside compute_noise_from_data).
        for col in columns:
            if is_root_node(scm.graph, col):
                continue
            mech = scm.causal_mechanism(col)
            if not isinstance(mech, InvertibleFunctionalCausalModel):
                raise RuntimeError(
                    f"CausalModel.fit: node {col!r} was assigned a non-invertible mechanism "
                    f"({type(mech).__name__}); counterfactual queries on it would fail deep "
                    "inside dowhy.gcm._noise.compute_noise_from_data. Add it to ordinal= or "
                    "report this as an unsupported mechanism class."
                )

        # 9. Fit — after the overrides, never before (PARENTS_DURING_FIT).
        gcm.fit(scm, prepared)

        spec = ModelSpec(ocg=ocg_aligned, columns=columns, data=prepared, mat=mat,
                         schema=schema, dtypes=dtypes, alignment=alignment)

        manifest = _fit_manifest(
            ocg_aligned, columns, dtypes, ordinal_maps, prepared, alignment,
            causal_graph, kg, include_object_properties, object_value,
            key_properties, limit, quality, categorical_counterfactual, random_state,
            excluded=excluded, excluded_joins=excluded_joins,
        )
        manifest["mechanisms_fitted"] = True

        return cls(scm=scm, spec=spec, fit_summary=fit_summary,
                   ordinal_maps=ordinal_maps, manifest=manifest)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str, *, with_graph: bool = True) -> None:
        """Write ``scm.pkl`` (joblib) + ``manifest.json`` (+ ``curated_graph.ttl``) to ``path``.

        ``curated_graph.ttl`` is the OCG this model was fitted on, written
        beside the pickle so the directory is **self-contained**. Without it a
        saved model is only loadable on the machine that produced it: the
        manifest records the graph as a *path* (``causal_graph_source``), and
        :meth:`load` re-reads that path — which on any other machine, or after
        the file is moved, silently yields ``spec.ocg = None`` and a model that
        cannot answer a query. A model that only works where it was made is
        not a model you can export, which is the whole point of the bundle.

        The **training data is deliberately not written**, here or in the zip
        bundle built on top of this directory. A fitted SCM is a set of
        mechanism parameters; the rows it was fitted on are the user's KG, and
        packaging them into an artefact meant for sharing turns every model
        export into a data export nobody asked for. :meth:`load` re-materialises
        from the recorded source KG when it is reachable, and says plainly what
        does not work when it is not.
        """
        import joblib
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scm, os.path.join(path, "scm.pkl"))

        manifest = dict(self.manifest)
        if with_graph and self.spec.ocg is not None:
            graph_path = os.path.join(path, GRAPH_FILENAME)
            self.spec.ocg.to_rdf().serialize(destination=graph_path, format="turtle")
            # A marker, not a path: the loader resolves it relative to the
            # directory it is reading, so the bundle survives being moved,
            # zipped, or unpacked somewhere else entirely.
            manifest["bundled_graph"] = GRAPH_FILENAME

        with open(os.path.join(path, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str, strict: bool = True, reload_data: bool = True) -> "CausalModel":
        """Read back a model saved by :meth:`save`.

        With ``strict=True`` (default) refuses to load when the recorded
        library versions disagree with what's currently installed — a
        pickled sklearn model loaded under a different sklearn version can
        silently misbehave rather than error. Compare a live graph against
        the one this model was fitted on with :meth:`verify_graph`.

        With ``reload_data=True`` (default) the causal graph and KG are
        re-resolved from the provenance recorded in the manifest
        (``causal_graph_source`` / ``source_kg``), so the returned model can
        answer :mod:`causalway.inference` queries immediately — a query needs
        ``spec.data`` (the prepared frame) and ``spec.ocg`` (for reach
        validation and node lookup), neither of which lives in ``scm.pkl``.
        Reloading fails softly: if the original source is unreachable (moved
        file, offline endpoint, or an in-memory graph that was never a
        path/URL), the corresponding ``spec`` field stays ``None`` and a
        message explains what will not work until it is attached manually.
        """
        import joblib
        with open(os.path.join(path, "manifest.json")) as fh:
            manifest = json.load(fh)
        if strict:
            current = {
                "dowhy": getattr(dowhy, "__version__", "unknown"),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
            }
            recorded = manifest.get("versions", {})
            mismatched = {k: (recorded.get(k), v) for k, v in current.items()
                         if k in recorded and recorded.get(k) != v}
            if mismatched:
                raise RuntimeError(
                    f"CausalModel.load({path!r}): library version mismatch {mismatched} "
                    "(recorded vs. installed). Pass strict=False to load anyway."
                )
        scm = joblib.load(os.path.join(path, "scm.pkl"))

        ocg = None
        mat = None
        data = None
        if reload_data:
            ocg = cls._reload_ocg(manifest, path)
            mat, data = cls._reload_data(manifest)

        spec = ModelSpec(ocg=ocg, columns=manifest["columns"], data=data, mat=mat,
                         schema=None, dtypes=manifest["dtypes"], alignment=manifest["alignment"])
        return cls(scm=scm, spec=spec, fit_summary=None,
                  ordinal_maps=manifest.get("ordinal_maps", {}), manifest=manifest)

    @staticmethod
    def _reload_ocg(manifest: dict, path: Optional[str] = None) -> Optional[OntologicalCausalGraph]:
        """The bundled graph first, the recorded source path only as a fallback.

        Order matters. ``curated_graph.ttl`` travels with the model and is by
        construction the graph it was fitted on; ``causal_graph_source`` is a
        path on whatever machine did the fitting, which after an export is at
        best stale and at worst someone else's *different* graph that happens
        to live at the same path. Preferring the bundle is what makes an
        imported model answer queries at all.
        """
        bundled = manifest.get("bundled_graph")
        if path and bundled:
            candidate = os.path.join(path, bundled)
            if os.path.exists(candidate):
                try:
                    return load_ocg(candidate, graph_id=manifest.get("ocg_gid"))
                except Exception as e:  # noqa: BLE001 — fall through to the source path
                    print(f"CausalModel.load: the bundled graph {candidate!r} did not "
                          f"parse ({e}); falling back to the recorded source.")

        cg_src = manifest.get("causal_graph_source", {})
        source = cg_src.get("path") or cg_src.get("url")
        if not source:
            print("CausalModel.load: causal_graph_source was not a file/endpoint "
                 "(in-memory graph at fit time) — spec.ocg stays None; pass an ocg to "
                 "store_query/validate_query explicitly instead.")
            return None
        try:
            return load_ocg(source, graph_id=manifest.get("ocg_gid"))
        except Exception as e:  # noqa: BLE001 — surfaced as a message, not a hard failure
            print(f"CausalModel.load: could not reload the causal graph from {source!r} "
                 f"({e}); spec.ocg stays None.")
            return None

    @staticmethod
    def _reload_data(manifest: dict):
        kg_src = manifest.get("source_kg", {})
        source = kg_src.get("path") or kg_src.get("url")
        if not source:
            print("CausalModel.load: source_kg was not a file/endpoint (in-memory graph at "
                 "fit time) — spec.data/mat stay None; prediction calls needing them will fail.")
            return None, None
        try:
            mkw = manifest.get("materialize_kwargs", {})
            schema = resolve_schema(source)
            excluded = mkw.get("excluded")
            excluded_joins = mkw.get("excluded_joins")
            kg_nodes, mat = _materialize_kg(
                schema,
                include_object_properties=mkw.get("include_object_properties", True),
                object_value=mkw.get("object_value", "range_class"),
                key_properties=mkw.get("key_properties"), limit=mkw.get("limit"),
                excluded=None if excluded is None else set(excluded),
                excluded_joins=None if excluded_joins is None else set(excluded_joins),
            )
            if isinstance(mat, list):
                raise ValueError("KG has multiple connected components")
            data = mat.df[manifest["columns"]].copy()
            ordinal_maps = manifest.get("ordinal_maps", {})
            for col, dt in manifest["dtypes"].items():
                if dt == "categorical":
                    data[col] = data[col].astype(str)
                elif dt == "ordinal":
                    inverse = {label: int(code) for code, label in ordinal_maps.get(col, {}).items()}
                    data[col] = data[col].map(inverse)
                elif dt == "discrete":
                    data[col] = pd.to_numeric(data[col], errors="coerce").round().astype(int)
                else:
                    data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)
            return mat, data
        except Exception as e:  # noqa: BLE001
            print(f"CausalModel.load: could not re-materialise the source KG from {source!r} "
                 f"({e}); spec.data/mat stay None.")
            return None, None

    def verify_graph(self, ocg: OntologicalCausalGraph, strict: bool = True) -> bool:
        """Compare a live OCG's fingerprint against the one this model was fitted on."""
        ok = ocg.fingerprint() == self.manifest.get("ocg_fingerprint")
        if not ok and strict:
            raise RuntimeError(
                f"CausalModel.verify_graph: fingerprint mismatch — this model was fitted on "
                f"{self.manifest.get('ocg_fingerprint')}, given graph fingerprints as "
                f"{ocg.fingerprint()}."
            )
        return ok

    # ------------------------------------------------------------------ #
    # Prediction (thin delegation to causalway.inference — kept out of this
    # module so plain Plan 1 discovery use never imports pgmpy/scipy)
    # ------------------------------------------------------------------ #
    # Every one of these takes ``target`` as *either* a single node name or a
    # collection of them (Plan 2 §6.3: a Query has 1..n targets). One name in,
    # one ``Answer`` out; a list/tuple/set in, ``{node: Answer}`` out — and the
    # batch is answered off one shared computation rather than one per target,
    # so the answers cannot disagree with each other. See
    # :mod:`causalway.inference`.
    def condition(self, target, evidence: dict, *, conditions: Optional[dict] = None,
                 backend: str = "auto", num_samples: int = 10_000):
        from .inference import condition as _condition
        return _condition(self, target, evidence, conditions=conditions,
                         backend=backend, num_samples=num_samples)

    def intervene(self, interventions: dict, *, target,
                 reference: Optional[dict] = None, conditions: Optional[dict] = None,
                 entity: Optional[str] = None, num_samples: int = 10_000):
        from .inference import intervene as _intervene
        return _intervene(self, interventions, target=target, reference=reference,
                         conditions=conditions, entity=entity, num_samples=num_samples)

    def counterfactual(self, interventions: dict, *, entity: str, target,
                       num_samples: int = 200):
        from .inference import counterfactual as _counterfactual
        return _counterfactual(self, interventions, entity=entity, target=target,
                              num_samples=num_samples)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def mechanism_table(self) -> pd.DataFrame:
        """One row per node: mechanism class, invertibility, dtype, root-ness."""
        rows = []
        for col in self.spec.columns:
            root = is_root_node(self.scm.graph, col)
            mech = self.scm.causal_mechanism(col)
            rows.append({
                "node": col,
                "dtype": self.spec.dtypes.get(col),
                "is_root": root,
                "mechanism_type": type(mech).__name__,
                "invertible": isinstance(mech, InvertibleFunctionalCausalModel) or root,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # RDF export — Layer A (Plan2 §6.2)
    # ------------------------------------------------------------------ #
    def to_rdf(self, graph: Optional[Graph] = None, *,
               with_vocabulary: bool = False,
               workdir: Optional[str] = None) -> Graph:
        """Serialise the fitted model (not the query/answer) as ``cw:CausalModel``.

        A thin wrapper over :func:`causalway.model_export.model_to_rdf`: the
        shape of this export lives in ``model_mapping.rml.ttl``, not here.  The
        hand-written version this replaced also emitted the legacy
        ``cw:parameters`` JSON literal; the mapping emits ``cw:Parameter``
        resources instead, the same way a discovery run's hyper-parameters have
        been written since the RML export landed.
        """
        from .model_export import model_to_rdf

        return model_to_rdf(self, graph=graph, with_vocabulary=with_vocabulary,
                            workdir=workdir)

    @property
    def iri(self) -> URIRef:
        return URIRef(MODEL_STEM + (self.model_id or "unfitted"))

    @property
    def run_iri(self) -> URIRef:
        return URIRef(MODEL_RUN_STEM + (self.model_id or "unfitted"))
