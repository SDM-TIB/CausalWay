"""E2 — What Assumption 1 buys, as the schema grows.

Question. The paper claims the schema-derived constraint is *pushed into* the
search rather than applied to its output. Three things could make that matter:
accuracy, cost, and validity. On the three-class clinic KG none of them is
visible (16 % of pairs forbidden, five of six methods return the same graph
with and without it). This experiment asks the question where it can have an
answer: schemas of 2 to 8 classes, 250 to 4,000 rows, ten KGs per cell.

Arms, for GES and PC:

    unconstrained   the learner sees the table and nothing else
    post-hoc        the unconstrained result with forbidden edges deleted —
                    the simplest way to obtain a valid graph, and the baseline
                    the paper's "pushed in, not applied to the output" must beat
    pushed-down     the constraint is given to the search
    + entity check  (GES only) pushed-down, then every edge whose family can be
                    read off one row per owning entity is kept only if those
                    entities support it (``algs/ges_prior/entity_check.py``)

Every KG is written as RDF and read back through the pipeline a user's KG would
take (schema induction, node construction, constraint compilation, flat join);
the join is checked against the generator's own table, so the experiment is also
a correctness test of the join over 120 schemas.

GES is the project's implementation in all three arms (``discover_structure``),
not pgmpy's unconstrained and ours constrained as the runner does: the arms
must differ in the constraint and in nothing else.

Reported per run: adjacency precision/recall/F1, SHD to the true CPDAG, the
number of forbidden adjacencies an arm proposed, seconds, and where the false
positives fall (within the fact class, within a replicated dimension class,
across adjacent classes, across non-adjacent classes). The second-to-last
breakdown is the measured size of the i.i.d. violation the paper documents.

    python experiments/e2_constraint.py            # full grid, resumable
    python experiments/e2_constraint.py --quick    # 2 seeds, smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, adjacency_scores, context_from_ttl, owners, quiet, shd_cpdag

OUT = os.path.join(OUT_DIR, "e2_constraint.jsonl")
CLASSES = (2, 3, 5, 8)
ROWS = (250, 1000, 4000)
SEEDS = range(10)
N_ATTRS = 3


def _fp_breakdown(adj, truth, names, mech) -> dict:
    import numpy as np
    est = np.triu(((adj + adj.T) > 0), 1)
    tru = np.triu(((truth + truth.T) > 0), 1)
    cls = [int(n.split(".")[0][1:]) for n in names]
    out = {"fp_fact": 0, "fp_dim": 0, "fp_adjacent": 0, "fp_forbidden": 0}
    for i, j in np.argwhere(est & ~tru):
        ci, cj = cls[i], cls[j]
        if ci == cj:
            out["fp_fact" if ci == 0 else "fp_dim"] += 1
        elif mech.parent_class[ci] == cj or mech.parent_class[cj] == ci:
            out["fp_adjacent"] += 1
        else:
            out["fp_forbidden"] += 1
    return out


def run_cell(n_classes: int, n_rows: int, seed: int, only_entity_check: bool = False) -> list:
    quiet()
    import numpy as np
    from algs.ges_prior.entity_check import entity_level_prune
    from algs.ges_prior.learn_bn import discover_structure
    from experiments.kgbench import make_mechanism
    from runners.run_kg_discovery import run_algorithm

    mech = make_mechanism(n_classes, N_ATTRS, seed=1000 * n_classes + seed)
    sample = mech.sample(n_rows, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kg.ttl")
        sample.write_ttl(path)
        t = time.perf_counter()
        ctx, _ = context_from_ttl(path)
        join_s = time.perf_counter() - t

    names = ctx.constraint_kept.names
    flat = sample.flat()
    kept = flat[names].sort_values(names).reset_index(drop=True)
    joined = ctx.discrete_df.astype(str).sort_values(names).reset_index(drop=True)
    join_ok = bool(kept.equals(joined))

    truth = mech.truth_adjacency(names)
    allowed = ctx.constraint_kept.allowed
    skel_allowed = allowed | allowed.T
    base = {
        "n_classes": n_classes, "n_rows": n_rows, "seed": seed, "n_nodes": len(names),
        "n_truth_edges": int(truth.sum()), "pruning_rate": ctx.constraint_kept.stats()["pruning_rate"],
        "join_seconds": join_s, "join_matches_generator": join_ok,
        "n_dropped_constant": len(flat.columns) - len(names),
    }

    def ges(constraint):
        return np.asarray(discover_structure(ctx.discrete_df, algorithm="GES", constraint=constraint), dtype=int)

    def pc(constraint):
        return np.asarray(run_algorithm("PC", ctx, constrained=constraint is not None), dtype=int)

    def record(method, arm, adj, secs) -> dict:
        row = dict(base, method=method, arm=arm, seconds=secs,
                   shd_cpdag=shd_cpdag(adj, truth),
                   n_forbidden=int(np.triu(((adj + adj.T) > 0) & ~skel_allowed, 1).sum()))
        row.update(adjacency_scores(adj, truth))
        row.update(_fp_breakdown(adj, truth, names, mech))
        return row

    def entity_checked() -> dict:
        owner_of, ids = owners(ctx)
        t = time.perf_counter()
        pushed = ges(ctx.constraint_kept)
        adj = entity_level_prune(pushed, ctx.discrete_df.astype(str), owner_of, ids)
        return record("GES", "+ entity check", adj, time.perf_counter() - t)

    if only_entity_check:
        return [entity_checked()]

    rows = []
    for method, learner in (("GES", ges), ("PC", pc)):
        t = time.perf_counter(); free = learner(None); free_s = time.perf_counter() - t
        t = time.perf_counter(); pushed = learner(ctx.constraint_kept); pushed_s = time.perf_counter() - t
        post = np.where(skel_allowed, free, 0)
        for arm, adj, secs in (("unconstrained", free, free_s), ("post-hoc", post, free_s),
                               ("pushed-down", pushed, pushed_s)):
            rows.append(record(method, arm, adj, secs))
    rows.append(entity_checked())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = OUT.replace(".jsonl", "_quick.jsonl") if args.quick else OUT
    done, checked = set(), set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                key = (r["n_classes"], r["n_rows"], r["seed"])
                (checked if r["arm"] == "+ entity check" else done).add(key)

    seeds = range(2) if args.quick else SEEDS
    grid = [(c, n, s) for c in CLASSES for n in ROWS for s in seeds]
    # A cell that has its learner arms but not the entity-check arm (the arm
    # was added later) only needs that one arm.
    cells = [cell + (False,) for cell in grid if cell not in done]
    cells += [cell + (True,) for cell in grid if cell in done and cell not in checked]
    print(f"{len(cells)} cells to run, {len(done)} already in {out}", flush=True)

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_cell)(*cell) for cell in cells)
    with open(out, "a") as f:
        for k, rows in enumerate(results, 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            r = rows[0]
            print(f"[{k}/{len(cells)}] classes={r['n_classes']} rows={r['n_rows']} seed={r['seed']} "
                  f"join_ok={r['join_matches_generator']}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
