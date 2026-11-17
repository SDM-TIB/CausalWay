"""E4 — Are the interventional answers right?

Discovery metrics score a graph. A causal resource is used for its *answers*,
and an answer can be scored only where the truth is known. KGBench keeps the
mechanism separate from the sample, so the true ``P(Y | do(X = x))`` over the
fact-row population is available by forward-sampling the intervened mechanism
(200,000 rows). This experiment fits a CausalWay model on each KG through the
public API (``CausalModel.fit`` on the KG file, ``model.intervene``) under four
causal graphs and measures the total-variation distance between the model's
interventional marginal and the true one:

    true graph      the generator's DAG: the error left when the structure is
                    right — estimation error of the mechanisms plus the
                    replicated-row effect of the join. The floor.
    pushed-down     GES with Assumption 1 in the search, CPDAG oriented by the
                    constraint-aware consistent extension (what ``to_ocg`` does)
    + entity check  pushed-down, then the entity-level check, oriented the same way
    unconstrained   GES without it, oriented the same way
    empty graph     no edges: the answer is the observational marginal. The
                    ceiling — a model must beat it to be worth fitting.

Targets are split by the truth: *descendants* of X (the intervention should move
them) and *non-descendants* (it should not). A wrong edge direction shows up in
the second group as an effect that does not exist.

The orientation of a CPDAG's undirected edges is a choice the data cannot make.
``orientation_spread`` re-orients the pushed-down CPDAG under five seeds and
reports the standard deviation of the resulting error — the size of the
arbitrariness a user inherits when they fit a model on a discovered graph
without curating it.

A fit that raises is recorded, not skipped: ``dowhy.gcm``'s automatic mechanism
selection cross-validates each node, and when a rare categorical level is absent
from one fold its ``log_loss`` call fails. CausalWay does not guard against
that yet, so a KG with such a level cannot be fitted through ``CausalModel.fit``
at the default quality. The report states how many KGs that was and computes
every comparison on the KGs where all graphs could be fitted.

    python experiments/e4_interventional.py
    python experiments/e4_interventional.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, context_from_ttl, owners, quiet

OUT = os.path.join(OUT_DIR, "e4_interventional.jsonl")
N_CLASSES = 4
N_ATTRS = 3
N_ROWS = 2000
SEEDS = range(12)
N_TRUTH = 200_000
MAX_INTERVENTIONS = 4
ORIENTATION_SEEDS = range(5)


def _tv(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def run_cell(seed: int) -> list:
    os.environ["CW_EXPERIMENT_GCM"] = "1"
    quiet()
    import numpy as np
    from algs.ges_prior.learn_bn import discover_structure
    from causalway.model import CausalModel
    from experiments.kgbench import make_mechanism
    from runners.run_kg_discovery import to_ocg

    mech = make_mechanism(N_CLASSES, N_ATTRS, seed=4000 + seed)
    sample = mech.sample(N_ROWS, seed=seed)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "kg.ttl")
    sample.write_ttl(path)
    ctx, object_names = context_from_ttl(path)
    names = ctx.constraint_kept.names
    truth = mech.truth_adjacency(names)
    node_of = {mech.name(n): n for n in mech.nodes}

    # Interventions: nodes with a true descendant, clamped to each of two levels.
    reach = np.linalg.matrix_power(truth + np.eye(len(names), dtype=int), len(names)) > 0
    np.fill_diagonal(reach, False)
    rng = np.random.default_rng(seed)
    causes = [i for i in range(len(names)) if reach[i].any()]
    causes = [causes[k] for k in sorted(rng.choice(len(causes), min(MAX_INTERVENTIONS, len(causes)),
                                                   replace=False))]
    interventions = [(i, level) for i in causes for level in (0, 1)]

    true_dist = {}
    for i, level in interventions:
        flat = mech.sample(N_TRUTH, seed=10_000 + seed, do={node_of[names[i]]: level}).flat()
        true_dist[(i, level)] = {n: flat[n].value_counts(normalize=True).to_dict() for n in names}

    free = np.asarray(discover_structure(ctx.discrete_df, algorithm="GES"), dtype=int)
    pushed = np.asarray(discover_structure(ctx.discrete_df, algorithm="GES",
                                           constraint=ctx.constraint_kept), dtype=int)
    owner_of, ids = owners(ctx)
    checked = np.asarray(discover_structure(ctx.discrete_df, algorithm="GES", constraint=ctx.constraint_kept,
                                            entity_ids=ids, owner_of=owner_of), dtype=int)
    graphs = {
        "true graph": to_ocg(truth, ctx, True, method="ground_truth"),
        "pushed-down": to_ocg(pushed, ctx, True, method="GES"),
        "+ entity check": to_ocg(checked, ctx, True, method="GES"),
        "unconstrained": to_ocg(free, ctx, False, method="GES"),
        "empty graph": to_ocg(np.zeros_like(truth), ctx, True, method="empty"),
    }

    def evaluate(ocg) -> dict:
        t = time.perf_counter()
        model = CausalModel.fit(ocg, path, excluded=object_names, excluded_joins=set(),
                                random_state=seed)
        fit_s = time.perf_counter() - t
        desc, nondesc = [], []
        t = time.perf_counter()
        for i, level in interventions:
            answers = model.intervene({names[i]: f"L{level}"},
                                      target=[n for n in names if n != names[i]], num_samples=20_000)
            for j, n in enumerate(names):
                if j == i:
                    continue
                err = _tv(answers[n].distribution, true_dist[(i, level)][n])
                (desc if reach[i, j] else nondesc).append(err)
        return {"tv_descendants": float(np.mean(desc)), "tv_nondescendants": float(np.mean(nondesc)),
                "tv_all": float(np.mean(desc + nondesc)), "fit_seconds": fit_s,
                "query_seconds": (time.perf_counter() - t) / len(interventions)}

    rows = []
    for arm, ocg in graphs.items():
        row = {"seed": seed, "arm": arm, "n_nodes": len(names), "n_rows": N_ROWS,
               "n_interventions": len(interventions), "n_edges": int(np.asarray(ocg.adj).sum())}
        try:
            row.update(evaluate(ocg))
        except Exception as exc:  # noqa: BLE001 — recorded verbatim, see module docstring
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rows.append(row)
    if any("error" in r for r in rows):
        return rows

    # Orientation sensitivity of the pushed-down CPDAG.
    from causalway.result import OntologicalCausalGraph
    raw = OntologicalCausalGraph.from_discovery(pushed, ctx.constraint_kept, method="GES",
                                                constrained=True, source=path)
    n_undirected = int(np.triu((pushed == 1) & (pushed.T == 1), 1).sum())
    errs = []
    if n_undirected:
        for s in ORIENTATION_SEEDS:
            errs.append(evaluate(raw.to_dag(strategy="extension", seed=s))["tv_descendants"])
    rows.append({"seed": seed, "arm": "orientation_spread", "n_undirected": n_undirected,
                 "tv_descendants_by_seed": errs,
                 "tv_descendants_sd": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
                 "tv_descendants_range": float(max(errs) - min(errs)) if errs else 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = OUT.replace(".jsonl", "_quick.jsonl") if args.quick else OUT
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {r["seed"] for r in map(json.loads, f)}
    seeds = [s for s in (range(1) if args.quick else SEEDS) if s not in done]
    print(f"{len(seeds)} KGs to run, {len(done)} already in {out}", flush=True)

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_cell)(s) for s in seeds)
    with open(out, "a") as f:
        for rows in results:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            failed = [r["arm"] for r in rows if "error" in r]
            print(f"seed {rows[0]['seed']}: " + "  ".join(
                f"{r['arm']}={r['tv_descendants']:.3f}" for r in rows if "tv_descendants" in r)
                + (f"  FIT FAILED: {failed}" if failed else ""), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
