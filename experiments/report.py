"""Turn experiments/out/*.jsonl into the tables the paper quotes.

Writes ``experiments/out/REPORT.md`` (readable) and ``summary.json`` (every
number the paper states, keyed, so a claim can be checked against a file
rather than against a PDF). Nothing is computed here that is not a mean, a
confidence interval, a count, or a paired Wilcoxon signed-rank test over the
per-run rows; the per-run rows are the evidence.

    python experiments/report.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.common import OUT_DIR

QUICK = "--quick" in sys.argv
SUFFIX = "_quick" if QUICK else ""


def load(name: str) -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f"{name}{SUFFIX}.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])


def ci(values) -> str:
    v = np.asarray(values, dtype=float)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return f"{v.mean():.3f} ± {half:.3f}"


def paired_p(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when every pair ties."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured Markdown table (pandas' own needs the optional ``tabulate``)."""
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def e1(md: list, summary: dict) -> None:
    df = load("e1_ges_correctness")
    if df.empty:
        return
    key = ["network", "n_rows", "seed"]
    wide = df.pivot_table(index=key, columns="arm", values=["shd_cpdag", "bic", "adj_f1", "n_edges"])
    rows = []
    for (net, n), g in df.groupby(["network", "n_rows"], sort=False):
        w = wide.loc[(net, n)]
        rows.append({
            "network": net, "rows": n, "runs": len(w),
            "SHD legacy": ci(w["shd_cpdag"]["legacy"]), "SHD corrected": ci(w["shd_cpdag"]["corrected"]),
            "SHD pgmpy": ci(w["shd_cpdag"]["pgmpy"]),
            "runs changed by fix": int((w["shd_cpdag"]["legacy"] != w["shd_cpdag"]["corrected"]).sum()
                                       + ((w["shd_cpdag"]["legacy"] == w["shd_cpdag"]["corrected"])
                                          & (w["bic"]["legacy"] != w["bic"]["corrected"])).sum()),
            "BIC corrected >= pgmpy": int((w["bic"]["corrected"] >= w["bic"]["pgmpy"] - 1e-6).sum()),
        })
    md += ["## E1. GES correctness\n", table(pd.DataFrame(rows))]
    cor, leg, pg = (df[df.arm == a].set_index(key) for a in ("corrected", "legacy", "pgmpy"))
    summary["e1"] = {
        "runs": int(len(cor)),
        "corrected_is_cpdag": int(cor.is_cpdag.sum()), "legacy_is_cpdag": int(leg.is_cpdag.sum()),
        "legacy_differs_from_corrected": int((leg.bic.sub(cor.bic).abs() > 1e-6).sum()
                                             + ((leg.bic.sub(cor.bic).abs() <= 1e-6)
                                                & (leg.shd_cpdag != cor.shd_cpdag)).sum()),
        "corrected_bic_ge_legacy": int((cor.bic >= leg.bic - 1e-6).sum()),
        "corrected_bic_ge_pgmpy": int((cor.bic >= pg.bic - 1e-6).sum()),
        "corrected_bic_gt_pgmpy": int((cor.bic > pg.bic + 1e-6).sum()),
        "corrected_same_graph_as_pgmpy": int(cor.same_as_pgmpy.sum()),
        "shd_corrected": ci(cor.shd_cpdag), "shd_legacy": ci(leg.shd_cpdag), "shd_pgmpy": ci(pg.shd_cpdag),
        "p_shd_corrected_vs_pgmpy": paired_p(cor.shd_cpdag, pg.shd_cpdag),
        "seconds_corrected": float(cor.seconds.mean()), "seconds_pgmpy": float(pg.seconds.mean()),
    }
    md += ["```json\n" + json.dumps(summary["e1"], indent=2) + "\n```\n"]


def e2(md: list, summary: dict) -> None:
    df = load("e2_constraint")
    if df.empty:
        return
    key = ["n_classes", "n_rows", "seed"]
    cells = df.drop_duplicates(key)
    summary["e2"] = {
        "kgs": int(len(cells)), "join_matches_generator": int(cells.join_matches_generator.sum()),
        "join_seconds_by_rows": {int(n): float(g.join_seconds.mean()) for n, g in cells.groupby("n_rows")},
        "by_classes": {},
    }
    md += ["## E2. What Assumption 1 buys\n",
           f"{len(cells)} KGs; the flat join reproduced the generator's table in "
           f"{int(cells.join_matches_generator.sum())} of them.\n"]

    rows = []
    for method in ("GES", "PC"):
        d = df[df.method == method]
        for c, g in d.groupby("n_classes"):
            arm = {a: x.set_index(["n_rows", "seed"]).sort_index() for a, x in g.groupby("arm")}
            free, post, push = arm["unconstrained"], arm["post-hoc"], arm["pushed-down"]
            rec = {
                "method": method, "classes": c, "nodes": int(g.n_nodes.max()),
                "pruned %": round(100 * g.pruning_rate.mean(), 1),
                "F1 unconstrained": ci(free.adj_f1), "F1 post-hoc": ci(post.adj_f1),
                "F1 pushed-down": ci(push.adj_f1),
                "SHD unconstrained": ci(free.shd_cpdag), "SHD post-hoc": ci(post.shd_cpdag),
                "SHD pushed-down": ci(push.shd_cpdag),
                "forbidden edges (unconstr.)": round(float(free.n_forbidden.mean()), 2),
                "speed-up": round(float((free.seconds / push.seconds).mean()), 2),
                "p F1 push vs free": paired_p(push.adj_f1, free.adj_f1),
                "p F1 push vs post": paired_p(push.adj_f1, post.adj_f1),
            }
            fp = push[["fp_fact", "fp_dim", "fp_adjacent"]].sum()
            rec["FP in replicated classes %"] = round(100 * fp["fp_dim"] / max(fp.sum(), 1), 1)
            if "+ entity check" in arm:
                ent = arm["+ entity check"]
                rec.update({"F1 + entity check": ci(ent.adj_f1), "SHD + entity check": ci(ent.shd_cpdag),
                            "precision pushed-down": ci(push.adj_precision),
                            "precision + entity check": ci(ent.adj_precision),
                            "recall pushed-down": ci(push.adj_recall),
                            "recall + entity check": ci(ent.adj_recall),
                            "p SHD entity vs push": paired_p(ent.shd_cpdag, push.shd_cpdag)})
            rows.append(rec)
            summary["e2"]["by_classes"][f"{method}-{c}"] = rec
    out = pd.DataFrame(rows)
    main_cols = ["method", "classes", "nodes", "pruned %", "F1 unconstrained", "F1 post-hoc", "F1 pushed-down",
                 "SHD unconstrained", "SHD pushed-down", "forbidden edges (unconstr.)", "speed-up",
                 "p F1 push vs free", "p F1 push vs post"]
    md += [table(out[main_cols])]
    ent_cols = ["classes", "FP in replicated classes %", "precision pushed-down", "precision + entity check",
                "recall pushed-down", "recall + entity check", "SHD pushed-down", "SHD + entity check",
                "p SHD entity vs push"]
    if "F1 + entity check" in out:
        md += ["### Entity-level check (GES)\n", table(out[out.method == "GES"][ent_cols])]

    # by sample size, 8 classes: where does the constraint matter most?
    rows = []
    for (c, n), g in df[df.method == "GES"].groupby(["n_classes", "n_rows"]):
        arm = {a: x.set_index("seed").sort_index() for a, x in g.groupby("arm")}
        rows.append({"classes": c, "rows": n, "F1 unconstrained": ci(arm["unconstrained"].adj_f1),
                     "F1 pushed-down": ci(arm["pushed-down"].adj_f1),
                     "seconds unconstrained": round(float(arm["unconstrained"].seconds.mean()), 2),
                     "seconds pushed-down": round(float(arm["pushed-down"].seconds.mean()), 2)})
    md += ["### GES by classes and rows\n", table(pd.DataFrame(rows))]
    summary["e2"]["ges_by_classes_rows"] = rows


def e3(md: list, summary: dict) -> None:
    df = load("e3_ges_prior")
    if df.empty:
        return
    arms = list(dict.fromkeys(df.arm))
    by = {a: df[df.arm == a].set_index("seed").sort_index() for a in arms}
    rows = [{"arm": a, "directed F1": ci(by[a].dir_f1), "precision": ci(by[a].dir_precision),
             "recall": ci(by[a].dir_recall), "SHD": ci(by[a].shd_dag), "adjacency F1": ci(by[a].adj_f1),
             "forbidden": int(by[a].n_forbidden.sum()),
             "p F1 vs GES": paired_p(by[a].dir_f1, by["GES"].dir_f1) if a != "GES" else None}
            for a in arms]
    md += ["## E3. GES-Prior on the clinic KG (cached LLM priors, 10 seeds)\n", table(pd.DataFrame(rows))]
    summary["e3"] = {r["arm"]: r for r in rows}
    summary["e3"]["p GES-Prior vs pre-revision"] = paired_p(by["GES-Prior"].dir_f1,
                                                             by["GES-Prior (pre-rev.)"].dir_f1)


def e4(md: list, summary: dict) -> None:
    df = load("e4_interventional")
    if df.empty:
        return
    failed = sorted(set(df[df.get("error").notna()].seed)) if "error" in df else []
    df = df[~df.seed.isin(failed)]
    main = df[df.arm != "orientation_spread"]
    arms = list(dict.fromkeys(main.arm))
    by = {a: main[main.arm == a].set_index("seed").sort_index() for a in arms}
    rows = [{"graph": a, "TV, descendants": ci(by[a].tv_descendants),
             "TV, non-descendants": ci(by[a].tv_nondescendants), "edges": round(float(by[a].n_edges.mean()), 1),
             "fit s": round(float(by[a].fit_seconds.mean()), 1),
             "p vs unconstrained": (paired_p(by[a].tv_descendants, by["unconstrained"].tv_descendants)
                                    if a != "unconstrained" and len(by[a]) > 1 else None)}
            for a in arms]
    md += ["## E4. Interventional accuracy against the generator's truth\n", table(pd.DataFrame(rows))]
    spread = df[df.arm == "orientation_spread"]
    with_und = spread[spread.n_undirected > 0]
    summary["e4"] = {
        "kgs": int(len(spread)), "kgs_fit_failed": len(failed),
        "arms": {r["graph"]: r for r in rows},
        "kgs_with_undirected_edges": int(len(with_und)),
        "orientation_sd_mean": float(with_und.tv_descendants_sd.mean()) if len(with_und) else 0.0,
        "orientation_range_max": float(with_und.tv_descendants_range.max()) if len(with_und) else 0.0,
        "undirected_edges_mean": float(with_und.n_undirected.mean()) if len(with_und) else 0.0,
    }
    md += [f"{len(spread)} KGs evaluated; on {len(failed)} more, `CausalModel.fit` raised for at least one "
           f"graph (a rare categorical level missing from a cross-validation fold of dowhy's model "
           f"selection) and the KG is excluded from every row above.\n"]
    md += [f"Orientation sensitivity: {len(with_und)} of {len(spread)} pushed-down CPDAGs had undirected "
           f"edges (mean {summary['e4']['undirected_edges_mean']:.1f}); across five orientation seeds the "
           f"descendant TV had mean s.d. {summary['e4']['orientation_sd_mean']:.4f} and maximum range "
           f"{summary['e4']['orientation_range_max']:.4f}.\n"]


def e7(md: list, summary: dict) -> None:
    df = load("e7_violation")
    if df.empty:
        return
    arms = list(dict.fromkeys(df.arm))
    by = {a: df[df.arm == a].set_index("seed").sort_index() for a in arms}
    rows = [{"arm": a, "pruned %": round(100 * by[a].pruning_rate.mean(), 1), "adjacency F1": ci(by[a].adj_f1),
             "recall, admissible edges": ci(by[a].recall_admissible),
             "recall, skip edges": ci(by[a].recall_skip.dropna()),
             "precision": ci(by[a].adj_precision),
             "p F1 vs unconstrained": paired_p(by[a].adj_f1, by["unconstrained"].adj_f1)
             if a != "unconstrained" else None} for a in arms]
    first = by[arms[0]]
    md += ["## E7. When Assumption 1 is false\n",
           f"{len(first)} KGs; on average {first.n_skip_edges.mean():.1f} of "
           f"{(first.n_skip_edges + first.n_admissible_edges).mean():.1f} true adjacencies skip a class "
           f"and are forbidden at k = 1.\n", table(pd.DataFrame(rows))]
    summary["e7"] = {"kgs": int(len(first)), "skip_edges_mean": float(first.n_skip_edges.mean()),
                     "true_edges_mean": float((first.n_skip_edges + first.n_admissible_edges).mean()),
                     "arms": {r["arm"]: r for r in rows}}


def e5(md: list, summary: dict) -> None:
    path = os.path.join(OUT_DIR, "e5_efficiency.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        summary["e5"] = json.load(f)
    md += ["## E5. Stage costs and the knowledge layer\n",
           "```json\n" + json.dumps(summary["e5"], indent=2) + "\n```\n"]


def main() -> None:
    md, summary = ["# CausalWay — experiment report\n",
                   "Generated by `experiments/report.py` from the per-run rows in this directory. "
                   "Intervals are normal 95 % intervals of the mean; p-values are two-sided paired "
                   "Wilcoxon signed-rank tests.\n"], {}
    for section in (e1, e2, e3, e4, e7, e5):
        section(md, summary)
    with open(os.path.join(OUT_DIR, f"REPORT{SUFFIX}.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(OUT_DIR, f"summary{SUFFIX}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
