"""Shared plumbing for the experiments: one way to build a context, one set of metrics.

Two decisions live here so that every experiment makes them the same way.

**Object properties are relationships.** ``runners.run_kg_discovery.build_context``
defaults to treating each object property as a join *and* as a variable whose
value is the range class. On a schema where one class holds two N:1 relations
that adds two ``?x a ?class`` patterns ahead of the join, and the query becomes a
cross product of every typed resource (minutes for 1,000 rows). The Studio's
default is the other role — relationship only (``context.md`` decision 25/W26) —
and ``context_from_ttl`` builds exactly that: the same schema induction, node
construction, constraint compilation and flat join the service runs, with the
object nodes excluded as variables and kept as joins.

**Metrics are defined once, on pairs.** A learner may return a CPDAG (PC, GES)
or a DAG (GES-Prior, anything after orientation), and the ground truth is a DAG
whose identifiable part is its CPDAG. Comparing a CPDAG with a DAG entry by
entry charges the learner for edges nobody could orient, so:

* adjacency precision / recall / F1 ignore direction — they score the skeleton;
* ``shd_cpdag`` counts the unordered pairs whose edge *type* (none, ``->``,
  ``<-``, ``-``) differs from the true CPDAG — the right target for a learner
  that only sees observational data;
* ``shd_dag`` / directed F1 compare with the true DAG and are reported only for
  outputs that are DAGs.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

OUT_DIR = os.path.join(_ROOT, "experiments", "out")

__all__ = [
    "OUT_DIR", "quiet", "context_from_ttl", "owners", "adjacency_scores", "shd_cpdag",
    "directed_scores", "true_cpdag", "mean_ci",
]


def quiet() -> None:
    """Silence library chatter; the experiments print their own progress."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ.setdefault("TQDM_DISABLE", "1")
    if "dowhy" in sys.modules or os.environ.get("CW_EXPERIMENT_GCM"):
        from dowhy import gcm
        gcm.config.disable_progress_bars()


def context_from_ttl(path: str, *, relation_direction: str = "both",
                     allow_epsilon: bool = True, max_hops: int = 1):
    """KG file -> ``DiscoveryContext`` with object properties as relationships only."""
    from causalway.bgp import materialize
    from causalway.constraints import EdgeConstraint
    from causalway.encoding import drop_constant_columns, to_discrete_frame, to_numeric_frame
    from causalway.nodes import build_nodes
    from causalway.ontology import OntologySchema
    from runners.run_kg_discovery import DiscoveryContext

    schema = OntologySchema.from_file(path)
    nodes = build_nodes(schema, include_object_properties=True)
    object_names = {n.name for n in nodes if n.kind == "object"}
    constraint = EdgeConstraint.from_schema(
        schema, nodes, relation_direction=relation_direction,
        allow_epsilon=allow_epsilon, max_hops=max_hops,
    )
    mat = materialize(schema, nodes, excluded=object_names, excluded_joins=set(),
                      include_object_properties=True)
    if isinstance(mat, list):
        raise ValueError("The KG has several connected components; run one at a time.")

    data_names = [n.name for n in nodes if n.name not in object_names]
    df, dropped = drop_constant_columns(mat.df.reindex(columns=data_names), verbose=False)
    keep = [i for i, n in enumerate(nodes) if n.name in df.columns]
    return DiscoveryContext(
        schema=schema, nodes=nodes, constraint=constraint, mat=mat,
        nodes_kept=[nodes[i] for i in keep], constraint_kept=constraint.subset(keep),
        discrete_df=to_discrete_frame(df), continuous_df=to_numeric_frame(df),
        dropped_columns=dropped, source=path,
    ), object_names


def owners(ctx) -> tuple:
    """``(owner_of, entity_ids)`` for the entity-level check, from the join's provenance.

    Every column of the flat join remembers the BGP variable whose entity holds
    it; ``Materialization.entity_ids`` has that entity's IRI on every row.
    """
    owner_of = {name: ctx.mat._colspec(name).subject_var for name in ctx.constraint_kept.names}
    return owner_of, ctx.mat.entity_ids.reset_index(drop=True)


# ------------------------------------------------------------------ metrics --

def _skeleton(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj) != 0
    return np.triu(a | a.T, 1)


def adjacency_scores(adj: np.ndarray, truth: np.ndarray) -> dict:
    est, tru = _skeleton(adj), _skeleton(truth)
    tp = int((est & tru).sum())
    fp = int((est & ~tru).sum())
    fn = int((~est & tru).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"adj_tp": tp, "adj_fp": fp, "adj_fn": fn, "adj_precision": p, "adj_recall": r,
            "adj_f1": 2 * p * r / (p + r) if p + r else 0.0}


def _pair_types(adj: np.ndarray) -> np.ndarray:
    a = (np.asarray(adj) != 0).astype(int)
    return np.triu(a + 2 * a.T, 1)          # 0 none, 1 i->j, 2 j->i, 3 undirected


def true_cpdag(truth: np.ndarray) -> np.ndarray:
    from algs.ges_prior.operators.graph import dag_to_cpdag
    return dag_to_cpdag(truth)


def shd_cpdag(adj: np.ndarray, truth_dag: np.ndarray) -> int:
    """Pairs whose edge type differs between ``adj``'s CPDAG and the true CPDAG.

    ``adj`` is first mapped to the CPDAG of its own equivalence class when it is
    a DAG, so a DAG-returning learner is not charged for its arbitrary choices.
    """
    from algs.ges_prior.operators.graph import dag_to_cpdag, is_dag
    a = (np.asarray(adj) != 0).astype(int)
    if not ((a == 1) & (a.T == 1)).any() and is_dag(a):
        a = dag_to_cpdag(a)
    return int((_pair_types(a) != _pair_types(true_cpdag(truth_dag))).sum())


def directed_scores(adj: np.ndarray, truth: np.ndarray) -> dict:
    """Directed precision / recall / F1 and SHD against the true DAG.

    An undirected edge in ``adj`` is credited to neither direction, so a CPDAG
    scores low here by design; use this for outputs that claim to be DAGs.
    """
    a = (np.asarray(adj) != 0)
    directed = a & ~a.T
    t = np.asarray(truth) != 0
    tp = int((directed & t).sum())
    fp = int((directed & ~t).sum()) + int(np.triu(a & a.T, 1).sum())
    fn = int((t & ~directed).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"dir_precision": p, "dir_recall": r,
            "dir_f1": 2 * p * r / (p + r) if p + r else 0.0,
            "shd_dag": int((_pair_types(a.astype(int)) != _pair_types(t.astype(int))).sum())}


def mean_ci(values, z: float = 1.96) -> tuple:
    """Mean and the half-width of its normal 95 % interval."""
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    half = z * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(half)
