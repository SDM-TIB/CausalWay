"""CLI for Plan 2: fit a CausalModel, run a query, optionally store it as RDF.

Examples
--------
    python -m runners.run_causal_model fit \\
        --graph results/sclc/GES.ttl --kg kgs/ttls/SCLC_patients.ttl \\
        --out results/models/sclc-ges

    python -m runners.run_causal_model query \\
        --model results/models/sclc-ges --kind conditional \\
        --target SCLCPatient.survival --evidence SCLCPatient.ageGroup=old
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from causalway.model import CausalModel
from causalway.queries import Condition, Intervention, Query, store_query, validate_query


def _parse_kv(pairs) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"expected NODE=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def cmd_fit(args) -> int:
    model = CausalModel.fit(
        args.graph, args.kg, quality=args.quality, on_missing=args.on_missing,
        on_cycle=args.on_cycle, limit=args.limit,
    )
    model.save(args.out)
    print(f"Fitted model {model.model_id} -> {args.out}")
    print(f"Alignment: {model.spec.alignment}")
    print(model.mechanism_table().to_string(index=False))
    return 0


def cmd_query(args) -> int:
    model = CausalModel.load(args.model)

    if args.kind == "conditional":
        evidence = _parse_kv(args.evidence)
        answer = model.condition(args.target, evidence, conditions=_parse_kv(args.condition_on))
        do = {}
    elif args.kind == "interventional":
        do = _parse_kv(args.do)
        reference = _parse_kv(args.reference) or None
        answer = model.intervene(do, target=args.target, reference=reference,
                                conditions=_parse_kv(args.condition_on) or None,
                                entity=args.entity)
    elif args.kind == "counterfactual":
        if not args.entity:
            raise SystemExit("counterfactual queries require --entity")
        do = _parse_kv(args.do)
        answer = model.counterfactual({(args.entity, k): v for k, v in do.items()},
                                     entity=args.entity, target=args.target)
    else:
        raise SystemExit(f"Unknown query kind: {args.kind}")

    print(json.dumps({
        "predicted": answer.predicted, "distribution": answer.distribution,
        "mean": answer.mean, "std": answer.std, "effect": answer.effect,
        "backend": answer.backend, "n_samples": answer.n_samples, "ess": answer.ess,
        "low_confidence": answer.low_confidence,
    }, default=str, indent=2))

    if args.store:
        is_cf = args.kind == "counterfactual"
        query = Query(
            kind=args.kind, model_id=model.model_id, target=[args.target],
            interventions=[Intervention(node=k, value=v, entity=args.entity if is_cf else None)
                          for k, v in do.items()],
            evidence=[Condition(node=k, value=v) for k, v in _parse_kv(args.evidence).items()],
            condition_on=[Condition(node=k, value=v) for k, v in _parse_kv(args.condition_on).items()],
            entity=args.entity if is_cf else None,
        )
        if model.spec.ocg is not None:
            validate_query(query, model.spec.ocg, model.spec.mat)
        g = store_query(answer, query, model, merge=args.merge)
        g.serialize(destination=args.store, format="turtle" if args.merge == "annotation" else "trig")
        print(f"Stored query+answer -> {args.store}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plan 2: causal model fitting and prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit a CausalModel and save it")
    p_fit.add_argument("--graph", required=True, help="causal graph: .ttl/.nt file or SPARQL endpoint")
    p_fit.add_argument("--kg", required=True, help="knowledge graph: .ttl/.nt file or SPARQL endpoint")
    p_fit.add_argument("--out", required=True, help="output dir for scm.pkl + manifest.json")
    p_fit.add_argument("--quality", default="good", choices=["good", "better", "best"])
    p_fit.add_argument("--on-missing", dest="on_missing", default="error", choices=["error", "drop"])
    p_fit.add_argument("--on-cycle", dest="on_cycle", default="strict", choices=["strict", "weight"])
    p_fit.add_argument("--limit", type=int, default=None)
    p_fit.set_defaults(func=cmd_fit)

    p_query = sub.add_parser("query", help="run a query against a saved model")
    p_query.add_argument("--model", required=True, help="path saved by `fit`")
    p_query.add_argument("--kind", required=True,
                         choices=["conditional", "interventional", "counterfactual"])
    p_query.add_argument("--target", required=True)
    p_query.add_argument("--evidence", nargs="*", default=[], help="NODE=VALUE ...")
    p_query.add_argument("--do", nargs="*", default=[], help="NODE=VALUE ... (do())")
    p_query.add_argument("--reference", nargs="*", default=[],
                         help="NODE=VALUE ... (interventional baseline arm)")
    p_query.add_argument("--condition-on", dest="condition_on", nargs="*", default=[],
                         help="NODE=VALUE ... (sub-population selector)")
    p_query.add_argument("--entity", default=None, help="entity IRI (required for counterfactual)")
    p_query.add_argument("--store", default=None, help="write query+answer RDF to this path")
    p_query.add_argument("--merge", default="annotation", choices=["annotation", "named-graph"])
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
