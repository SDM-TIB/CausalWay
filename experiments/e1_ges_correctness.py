"""E1 — Is the project's GES a correct GES, and what was the defect worth?

The GES that ships in ``algs/ges_prior`` (and that every constrained GES and
every GES-Prior run goes through) started as a copy of gCastle's. Two defects
came with it: the backward phase computes a score difference that is zero by
construction and so never deletes an edge, and the DAG -> CPDAG conversion keeps
some reversible edges directed. Both were fixed in the WWW'27 resource revision;
``legacy_ges=True`` keeps the old behaviour for this experiment.

Three arms on standard Bayesian-network benchmarks, which have nothing to do
with knowledge graphs — the point is the learner, not the pipeline:

    legacy      the search as it was (forward phase only, wrong CPDAG routine)
    corrected   the search as it is
    pgmpy       pgmpy's own GES under the same BIC score: an independent
                implementation, so agreement with it is evidence of correctness
                rather than of two copies of one mistake

Reported per run: SHD to the true CPDAG, adjacency F1, number of edges, the BIC
of the returned equivalence class (higher is better; a correct backward phase
can only raise it), whether the output *is* a CPDAG (re-deriving the CPDAG of
one of its extensions gives it back), and seconds.

    python experiments/e1_ges_correctness.py
    python experiments/e1_ges_correctness.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, adjacency_scores, quiet, shd_cpdag

OUT = os.path.join(OUT_DIR, "e1_ges_correctness.jsonl")
NETWORKS = ("asia", "sachs", "child", "insurance")
ROWS = (500, 5000)
SEEDS = range(10)


def run_cell(network: str, n_rows: int, seed: int) -> list:
    quiet()
    import networkx as nx
    import numpy as np
    from pgmpy.base import DAG
    from pgmpy.causal_discovery import GES
    from pgmpy.structure_score import get_scoring_method
    from pgmpy.utils import get_example_model

    from algs.ges_prior.ges_phase_I import ges_phase_i
    from algs.ges_prior.operators.graph import _to_dag, dag_to_cpdag

    model = get_example_model(network)
    cols = sorted(model.nodes())
    df = model.simulate(n_rows, seed=seed, show_progress=False)[cols].astype(str)
    df = df.loc[:, df.nunique() > 1]
    cols = list(df.columns)
    idx = {c: i for i, c in enumerate(cols)}
    truth = np.zeros((len(cols), len(cols)), dtype=int)
    for u, v in model.edges():
        if u in idx and v in idx:
            truth[idx[u], idx[v]] = 1

    scorer = get_scoring_method("bic-d", df)

    def bic(cpdag: np.ndarray) -> float:
        dag = DAG()
        dag.add_nodes_from(cols)
        dag.add_edges_from((cols[i], cols[j]) for i, j in np.argwhere(_to_dag(cpdag)))
        return float(scorer.score(dag))

    def ours(legacy: bool) -> np.ndarray:
        return np.asarray(ges_phase_i(df, True, "bic-d", legacy_ges=legacy), dtype=int)

    def pgmpy_ges() -> np.ndarray:
        est = GES(scoring_method="bic-d", return_type="pdag")
        est.fit(df)
        g = est.causal_graph_
        adj = np.zeros((len(cols), len(cols)), dtype=int)
        for u, v in g.edges():
            adj[idx[u], idx[v]] = 1
        return adj

    rows, results = [], {}
    for arm, learner in (("legacy", lambda: ours(True)), ("corrected", lambda: ours(False)),
                         ("pgmpy", pgmpy_ges)):
        t = time.perf_counter()
        adj = learner()
        secs = time.perf_counter() - t
        results[arm] = adj
        row = {"network": network, "n_rows": n_rows, "seed": seed, "arm": arm, "seconds": secs,
               "n_nodes": len(cols), "n_truth_edges": int(truth.sum()),
               "n_edges": int(np.triu((adj + adj.T) > 0, 1).sum()),
               "shd_cpdag": shd_cpdag(adj, truth), "bic": bic(adj),
               "is_cpdag": bool(np.array_equal(dag_to_cpdag(_to_dag(adj)), adj))}
        row.update(adjacency_scores(adj, truth))
        rows.append(row)
    for row in rows:
        row["same_as_pgmpy"] = bool(np.array_equal(results[row["arm"]], results["pgmpy"]))
        row["same_skeleton_as_pgmpy"] = bool(np.array_equal(
            (results[row["arm"]] + results[row["arm"]].T) > 0,
            (results["pgmpy"] + results["pgmpy"].T) > 0))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = OUT.replace(".jsonl", "_quick.jsonl") if args.quick else OUT
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["network"], r["n_rows"], r["seed"]) for r in map(json.loads, f)}

    networks = NETWORKS[:2] if args.quick else NETWORKS
    seeds = range(2) if args.quick else SEEDS
    cells = [(net, n, s) for net in networks for n in ROWS for s in seeds if (net, n, s) not in done]
    print(f"{len(cells)} cells to run, {len(done)} already in {out}", flush=True)

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_cell)(*cell) for cell in cells)
    with open(out, "a") as f:
        for k, rows in enumerate(results, 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k}/{len(cells)}] {rows[0]['network']} n={rows[0]['n_rows']} seed={rows[0]['seed']}: "
                  + "  ".join(f"{r['arm']} shd={r['shd_cpdag']} bic={r['bic']:.0f}" for r in rows), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
