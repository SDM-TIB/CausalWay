"""E7 — What the constraint costs when Assumption 1 is false.

E2 measures the constraint on KGs whose ground truth satisfies it, which is the
favourable case by construction. A schema is not a causal model: two classes the
KG does not relate directly can still be causally related directly, when the
attribute that mediates the effect was never recorded. This experiment builds
that case on purpose. KGBench draws, besides the admissible edges, true edges
from a class *two* relations away (``p_skip``), so ``k = 1`` forbids part of the
truth, and compares:

    unconstrained     no constraint: can find a skip edge, and any wrong one
    k = 1             the default: cannot find a skip edge
    k = 2             the hop budget raised to cover the violation

Reported: adjacency F1, and recall separately on the admissible true edges and
on the skip edges. The second number for ``k = 1`` is zero by construction; the
experiment is about what happens to everything else, and about whether raising
``k`` — the remedy the paper recommends — recovers the loss.

    python experiments/e7_violation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, adjacency_scores, context_from_ttl, quiet

OUT = os.path.join(OUT_DIR, "e7_violation.jsonl")
N_CLASSES, N_ATTRS, N_ROWS, P_SKIP = 6, 3, 1000, 0.2
SEEDS = range(20)


def run_cell(seed: int) -> list:
    quiet()
    import numpy as np
    from algs.ges_prior.learn_bn import discover_structure
    from experiments.kgbench import make_mechanism

    mech = make_mechanism(N_CLASSES, N_ATTRS, seed=7000 + seed, p_skip=P_SKIP)
    sample = mech.sample(N_ROWS, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kg.ttl")
        sample.write_ttl(path)
        ctx1, _ = context_from_ttl(path, max_hops=1)
        ctx2, _ = context_from_ttl(path, max_hops=2)
    names = ctx1.constraint_kept.names
    truth = mech.truth_adjacency(names)
    skel1 = ctx1.constraint_kept.allowed | ctx1.constraint_kept.allowed.T
    true_skel = np.triu((truth + truth.T) > 0, 1)
    skip = true_skel & ~skel1                       # true adjacencies k = 1 forbids
    admissible = true_skel & skel1

    arms = {"unconstrained": None, "k = 1": ctx1.constraint_kept, "k = 2": ctx2.constraint_kept}
    rows = []
    for arm, constraint in arms.items():
        adj = np.asarray(discover_structure(ctx1.discrete_df, algorithm="GES", constraint=constraint), dtype=int)
        est = np.triu((adj + adj.T) > 0, 1)
        row = {"seed": seed, "arm": arm, "n_nodes": len(names), "n_skip_edges": int(skip.sum()),
               "n_admissible_edges": int(admissible.sum()),
               "recall_skip": float((est & skip).sum() / skip.sum()) if skip.sum() else None,
               "recall_admissible": float((est & admissible).sum() / max(admissible.sum(), 1)),
               "pruning_rate": constraint.stats()["pruning_rate"] if constraint is not None else 0.0}
        row.update(adjacency_scores(adj, truth))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {r["seed"] for r in map(json.loads, f)}
    seeds = [s for s in SEEDS if s not in done]
    print(f"{len(seeds)} KGs to run, {len(done)} already in {OUT}", flush=True)
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_cell)(s) for s in seeds)
    with open(OUT, "a") as f:
        for rows in results:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"seed {rows[0]['seed']} (skip edges {rows[0]['n_skip_edges']}): " + "  ".join(
                f"{r['arm']} F1={r['adj_f1']:.2f}" for r in rows), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
