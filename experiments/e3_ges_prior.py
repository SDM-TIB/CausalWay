"""E3 — Does GES-Prior use a prior well? (clinic KG, cached LLM priors)

GES-Prior adds LLM-elicited edge priors to the BIC score. Before the WWW'27
resource revision it scored *below* plain GES on the clinic KG (F1 .70 vs .80 in
the earlier draft), which is the wrong way round for a method whose input is
strictly richer. The revision changed three things (``algs/ges_prior``): the
search (backward phase, CPDAG conversion), the Phase II score (``BIC+M+B``
instead of the log odds of ``B``), and the orientation (inside the equivalence
class, acyclic by construction). This experiment measures what they are worth.

Data: ten clinic KGs from ``causalway.synthetic.generate`` (seeds 0-9; 500
patients, 20 hospitals), each pushed through the pipeline. Priors: the matrices
in ``results/priors_synthetic.json``, elicited once from the KG's own
``rdfs:label`` / ``rdfs:comment`` with an LLM and cached, so the experiment runs
offline and every seed sees the same priors.

**What this does and does not show.** The clinic KG's comments describe its
mechanisms in words ("disease stage, influenced by age, smoking, …"). The prior
is therefore close to an oracle, and the experiment measures how well a *good*
prior is exploited — the plumbing — not how much an LLM knows about a domain it
was not told about. A method that cannot beat GES here cannot beat it anywhere.

Arms (all with Assumption 1 pushed into the search):

    GES                     corrected search; CPDAG oriented by the
                            constraint-aware consistent extension (``to_ocg``)
    GES-Prior (pre-rev.)    the configuration before the revision: legacy
                            search, Phase II hill climbing under log(B/(1-B))
                            with eps = 1e-6
    GES-M-B                 corrected search with BIC+M, orientation by B
    GES-Prior (ext. start)  corrected search with BIC+M, hill climbing under
                            BIC+M+B from an arbitrary consistent extension
    GES-Prior               the same, hill climbing started from the member of
                            the class the prior prefers (= "GES-M-B-HL", the
                            default, what the UI calls GES-Prior)
    GES-Prior + entity      GES-Prior followed by the entity-level check
    GES + entity            GES followed by the entity-level check
    GES-Prior, no priors    the same pipeline with an empty prior: the cost of
                            Phase II when the LLM knows nothing

Reported: directed precision / recall / F1 and SHD against the true DAG.

    python experiments/e3_ges_prior.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, adjacency_scores, context_from_ttl, directed_scores, owners, quiet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(OUT_DIR, "e3_ges_prior.jsonl")
PRIORS = os.path.join(ROOT, "results", "priors_synthetic.json")
SEEDS = range(10)


def _aligned_priors(names: list) -> dict:
    import numpy as np
    with open(PRIORS) as f:
        raw = json.load(f)
    pos = {c: i for i, c in enumerate(raw["columns"])}
    missing = [n for n in names if n not in pos]
    if missing:
        raise ValueError(f"cached priors do not cover {missing}")
    order = [pos[n] for n in names]
    return {k: np.asarray(raw[k], dtype=float)[np.ix_(order, order)] for k in ("M", "B")}


def run_cell(seed: int) -> list:
    quiet()
    import numpy as np
    from algs.ges_prior.ges_prior import run_ges_prior
    from algs.ges_prior.learn_bn import _normalise_priors, discover_structure
    from algs.ges_prior.scores.prior_score import PriorScore, prior_weights
    from causalway.synthetic import generate, truth_adjacency
    from runners.run_kg_discovery import to_ocg

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clinic.ttl")
        generate(n_patients=500, n_hospitals=20, max_therapies=4, seed=seed).write_ttl(path)
        ctx, _ = context_from_ttl(path)
    names = ctx.constraint_kept.names
    truth = truth_adjacency(names)
    constraint = ctx.constraint_kept
    priors = _aligned_priors(names)
    df = ctx.discrete_df

    def pre_revision() -> np.ndarray:
        pri = _normalise_priors(priors, len(names))
        data = df.astype(str)
        s1 = PriorScore(data, "bic-d", W=prior_weights("BIC+M", pri["M"], pri["D"], pri["has"], eps=1e-6))
        s2 = PriorScore(data, "bic-d", W=prior_weights("BIC+B", pri["M"], pri["D"], pri["has"], eps=1e-6))
        return run_ges_prior(data, scoring_method_i=s1, scoring_method_ii=s2, orient_strategy=2,
                             allowed=constraint.allowed, expert_knowledge=constraint.to_expert_knowledge(),
                             B=pri["B"], legacy_ges=True)

    def ges() -> np.ndarray:
        cpdag = discover_structure(df, algorithm="GES", constraint=constraint)
        return np.asarray(to_ocg(np.asarray(cpdag, dtype=int), ctx, True, method="GES").adj, dtype=int)

    owner_of, ids = owners(ctx)

    def ges_entity() -> np.ndarray:
        cpdag = discover_structure(df, algorithm="GES", constraint=constraint,
                                   entity_ids=ids, owner_of=owner_of)
        return np.asarray(to_ocg(np.asarray(cpdag, dtype=int), ctx, True, method="GES").adj, dtype=int)

    def prior(**kw):
        return discover_structure(df, algorithm="GES-M-B-HL", constraint=constraint, **kw)

    arms = {
        "GES": ges,
        "GES + entity": ges_entity,
        "GES-Prior (pre-rev.)": pre_revision,
        "GES-M-B": lambda: discover_structure(df, algorithm="GES-M-B", constraint=constraint, priors=priors),
        "GES-Prior (ext. start)": lambda: prior(priors=priors, hc_start="extension"),
        "GES-Prior": lambda: prior(priors=priors),
        "GES-Prior + entity": lambda: prior(priors=priors, entity_ids=ids, owner_of=owner_of),
        "GES-Prior, no priors": lambda: prior(),
    }
    rows = []
    for arm, learner in arms.items():
        t = time.perf_counter()
        adj = np.asarray(learner(), dtype=int)
        row = {"seed": seed, "arm": arm, "seconds": time.perf_counter() - t,
               "n_rows": int(len(df)), "n_nodes": len(names), "n_edges": int(adj.sum()),
               "n_forbidden": int((adj.astype(bool) & ~constraint.allowed).sum())}
        row.update(directed_scores(adj, truth))
        row.update(adjacency_scores(adj, truth))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
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
            print(f"seed {rows[0]['seed']}: " + "  ".join(
                f"{r['arm']}={r['dir_f1']:.2f}" for r in rows), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
