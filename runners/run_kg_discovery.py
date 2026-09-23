"""Shared KG causal-discovery pipeline and CLI entry point.

The notebooks import the functions here rather than duplicating logic, so the
``.py`` files stay the source of truth.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from causalway.ontology import OntologySchema
from causalway.nodes import build_nodes
from causalway.constraints import EdgeConstraint
from causalway.bgp import Materialization, materialize
from causalway.encoding import drop_constant_columns, to_discrete_frame, to_numeric_frame
from causalway.llm_meta import build_var_meta, build_pair_meta, build_domain_str
from causalway.result import OntologicalCausalGraph, write_bundle
from algs.discovery_alg import CausalDiscovery
from algs.ges_prior.learn_bn import discover_structure, estimate_priors
from algs.ges_prior.llm_call import LLMClient

SCORE_BASED = {"PC"}
PRIOR_BASED = {
    "GES", "GES-M", "GES-B", "GES-M-B", "GES-M-B-HL", "GES-M-B-Greedy",
    "GES-HL", "GES-M-HL", "GES-B-HL",
}
CONTINUOUS = {"NOTEARS", "DAGMA", "LiNGAM", "DAG-GNN"}
ALL_METHODS = list(SCORE_BASED | PRIOR_BASED | CONTINUOUS)

# Public alias for the one PRIOR_BASED variant meant to be user-facing;
# resolved transparently in run_algorithm(). Every other PRIOR_BASED name
# (GES-M, GES-B, GES-M-B, GES-M-B-Greedy, GES-HL, GES-M-HL, GES-B-HL) stays
# internal-only -- reachable by its raw name for existing callers, but not
# part of the method list new pipelines (e.g. notebook 05) should offer.
METHOD_ALIASES = {"GES-Prior": "GES-M-B-HL"}
ALLOWED_METHODS = ["GES", "GES-Prior", "PC", "NOTEARS", "DAGMA", "LiNGAM", "DAG-GNN"]


@dataclass
class DiscoveryContext:
    """Everything the discovery step needs, derived from one KG."""

    schema: OntologySchema
    nodes: list                       # full node list (before constant-column drop)
    constraint: EdgeConstraint        # full constraint
    mat: Materialization              # raw materialization
    nodes_kept: list                  # nodes whose columns survived
    constraint_kept: EdgeConstraint   # constraint over kept columns
    discrete_df: pd.DataFrame         # for score-based / PC / GES-Prior
    continuous_df: pd.DataFrame       # for NOTEARS/DAGMA/LiNGAM/DAG-GNN
    dropped_columns: list = field(default_factory=list)
    source: Optional[str] = None      # path of the KG this context came from

    @property
    def column_names(self) -> list:
        return self.constraint_kept.names


def build_context(
    ttl_path: str,
    include_object_properties: bool = True,
    object_value: str = "range_class",
    key_properties: Optional[dict] = None,
    relation_direction: str = "both",
    allow_epsilon: bool = True,
    max_hops: int = 1,
    optional_data_properties: bool = False,
    limit: Optional[int] = None,
) -> DiscoveryContext:
    """Parse a KG and assemble the node set, constraint, and flat-join frames."""
    schema = OntologySchema.from_file(ttl_path)
    nodes = build_nodes(schema, include_object_properties=include_object_properties)
    constraint = EdgeConstraint.from_schema(
        schema, nodes, relation_direction=relation_direction,
        allow_epsilon=allow_epsilon, max_hops=max_hops,
    )
    mat = materialize(
        schema, nodes,
        include_object_properties=include_object_properties,
        object_value=object_value,
        key_properties=key_properties,
        optional_data_properties=optional_data_properties,
        limit=limit,
    )
    if isinstance(mat, list):
        raise ValueError(
            "The KG has multiple connected components; materialize and run each "
            "component separately (cross-component edges are forbidden anyway)."
        )

    df = mat.df.reindex(columns=[n.name for n in nodes])
    df_clean, dropped = drop_constant_columns(df, verbose=True)
    keep = [i for i, n in enumerate(nodes) if n.name in df_clean.columns]
    constraint_kept = constraint.subset(keep)
    nodes_kept = [nodes[i] for i in keep]

    return DiscoveryContext(
        schema=schema,
        nodes=nodes,
        constraint=constraint,
        mat=mat,
        nodes_kept=nodes_kept,
        constraint_kept=constraint_kept,
        discrete_df=to_discrete_frame(df_clean),
        continuous_df=to_numeric_frame(df_clean),
        dropped_columns=dropped,
        source=ttl_path,
    )


def run_algorithm(
    method: str,
    ctx: DiscoveryContext,
    constrained: bool = True,
    alpha: Optional[float] = None,
    priors: Optional[dict] = None,
    var_meta: Optional[dict] = None,
    pair_meta: Optional[dict] = None,
    domain: str = "general domain",
    llm_client=None,
    llm_model: str = "deepseek-v4-flash",
    verbose: bool = False,
    return_weights: bool = False,
):
    """Run one method and return its adjacency matrix.

    The DataFrame columns are ordered to match ``constraint_kept.names``, so the
    constraint matrices index the columns correctly.

    With ``return_weights=True`` the return value is ``(adj, weights)``, where
    ``weights`` is the method's ``(n, n)`` weight matrix (regression
    coefficients for LiNGAM/NOTEARS/DAGMA/DAG-GNN, 1.0 on every edge for the
    constraint- and score-based methods).
    """
    constraint = ctx.constraint_kept if constrained else None
    method = METHOD_ALIASES.get(method, method)

    if method in PRIOR_BASED:
        adj = np.asarray(discover_structure(
            df=ctx.discrete_df,
            algorithm=method,
            constraint=constraint,
            priors=priors,
            var_meta=var_meta,
            pair_meta=pair_meta,
            domain=domain,
            llm_client=llm_client,
            llm_model=llm_model,
        ), dtype=int)
        return (adj, adj.astype(float)) if return_weights else adj

    if method in CONTINUOUS:
        df = ctx.continuous_df
    else:  # GES, PC
        df = ctx.discrete_df

    cd = CausalDiscovery(verbose=verbose)
    graph, adj = cd.discover(df, method=method, alpha=alpha, constraint=constraint)
    adj = np.asarray(adj, dtype=int)
    if not return_weights:
        return adj
    return adj, _weight_matrix(graph, list(df.columns), adj)


def _weight_matrix(graph, columns: list, adj: np.ndarray) -> np.ndarray:
    """Read the per-edge weights off the returned DiGraph, aligned to ``adj``."""
    weights = np.zeros(adj.shape, dtype=float)
    index = {name: k for k, name in enumerate(columns)}
    for source, target, data in graph.edges(data=True):
        i, j = index.get(source), index.get(target)
        if i is None or j is None:
            continue
        weights[i, j] = float(data.get("weight", 1.0))
    # Any edge the graph did not weight still counts as present.
    weights[(adj == 1) & (weights == 0.0)] = 1.0
    return weights


def to_ocg(adj: np.ndarray, ctx: DiscoveryContext, constrained: bool,
           weights: Optional[np.ndarray] = None,
           method: Optional[str] = None,
           params: Optional[dict] = None,
           source: Optional[str] = None) -> OntologicalCausalGraph:
    """Wrap an adjacency matrix into an :class:`OntologicalCausalGraph`.

    ``method``/``params``/``source`` become the ``cw:DiscoveryRun`` provenance
    in the Turtle export; ``weights`` becomes ``cw:weight`` on each edge.

    The constraint is always ``constraint_kept``: every run — constrained or
    not — is fitted on the kept columns, so that is the only node set whose
    order matches ``adj``.  ``constrained`` records whether Assumption 1 was
    *enforced*, which is provenance, not indexing.  The relation labels are
    ontological facts either way; an unconstrained run that returns a
    topologically invalid edge simply gets no label (see
    :meth:`OntologicalCausalGraph.topological_validity`).
    """
    return OntologicalCausalGraph.from_discovery(
        adj, ctx.constraint_kept, weights=weights, method=method, params=params,
        constrained=constrained, source=source or getattr(ctx, "source", None),
    )


def estimate_and_cache_priors(
    ctx: DiscoveryContext,
    cache_path: str,
    llm_model: str = "deepseek-v4-flash",
    max_workers: int = 12,
):
    """Build var_meta / pair_meta from the schema and estimate (cached) priors."""
    schema = ctx.schema
    nodes = ctx.nodes_kept
    constraint = ctx.constraint_kept

    var_meta = build_var_meta(schema, nodes)
    pair_meta = build_pair_meta(schema, nodes, constraint)
    domain = build_domain_str(schema)

    client = LLMClient(model=llm_model)
    priors = estimate_priors(
        ctx.discrete_df,
        var_meta=var_meta,
        pair_meta=pair_meta,
        domain=domain,
        llm_client=client,
        max_workers=max_workers,
        cache_path=cache_path,
        allowed=constraint.allowed,
    )
    return priors, var_meta, pair_meta, domain


def run_experiment(
    ctx: DiscoveryContext,
    methods: Optional[list] = None,
    constrained: bool = True,
    alpha: Optional[float] = None,
    verbose: bool = False,
    return_weights: bool = False,
) -> dict:
    """Run each method (without LLM priors) and return ``{method: adj}``.

    With ``return_weights=True`` each value is ``(adj, weights)``.
    """
    methods = methods or ALL_METHODS
    out = {}
    for method in methods:
        out[method] = run_algorithm(method, ctx, constrained=constrained, alpha=alpha,
                                    verbose=verbose, return_weights=return_weights)
    return out


def run_and_export(
    ctx: DiscoveryContext,
    methods: Optional[list] = None,
    constrained: bool = True,
    alpha: Optional[float] = None,
    out_dir: Optional[str] = None,
    bundle_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Run each method and return ``{method: OntologicalCausalGraph}``.

    One ``<method>.ttl`` per method is written to ``out_dir`` when given, and
    all of them are merged into ``bundle_path`` when given — the single file to
    load into a triple store, since the property nodes carry global IRIs and so
    the per-method graphs join on them.
    """
    methods = methods or ALL_METHODS
    params = {"alpha": alpha, "constrained": constrained}
    ocgs = {}
    for method in methods:
        adj, weights = run_algorithm(method, ctx, constrained=constrained, alpha=alpha,
                                     verbose=verbose, return_weights=True)
        ocgs[method] = to_ocg(adj, ctx, constrained, weights=weights,
                              method=method, params=params, source=ctx.source)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            ocgs[method].to_turtle(os.path.join(out_dir, f"{method}.ttl"))
    if bundle_path:
        write_bundle(ocgs, bundle_path)
    return ocgs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Causal discovery over KG properties")
    parser.add_argument("--ttl", required=True, help="path to the TTL/NT KG")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--unconstrained", action="store_true",
                        help="run only unconstrained (default: both)")
    parser.add_argument("--object-value", default="range_class",
                        choices=["range_class", "key_property"])
    parser.add_argument("--relation-direction", default="both", choices=["both", "forward"])
    args = parser.parse_args(argv)

    ctx = build_context(args.ttl, object_value=args.object_value,
                        relation_direction=args.relation_direction)
    print(f"Nodes: {ctx.column_names}")
    print(f"Constraint stats: {ctx.constraint_kept.stats()}")

    os.makedirs(args.out, exist_ok=True)
    for constrained in ([False] if args.unconstrained else [True, False]):
        tag = "constrained" if constrained else "unconstrained"
        ocgs = run_and_export(
            ctx, constrained=constrained,
            bundle_path=os.path.join(args.out, f"all_methods_{tag}.ttl"),
        )
        for method, ocg in ocgs.items():
            path = os.path.join(args.out, f"{method}_{tag}.ttl")
            ocg.to_turtle(path)
            print(f"[{tag:13s}] {method:14s} -> {int(ocg.adj.sum())} edges -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
