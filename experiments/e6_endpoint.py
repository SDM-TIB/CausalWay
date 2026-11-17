"""E6 — Schema induction and constraint compilation on a live public endpoint.

The other experiments read files. A resource for Web KGs has to work against an
endpoint it does not control, where the schema is whatever the endpoint
declares. This probe points the same two steps (``resolve_schema``,
``EdgeConstraint.from_schema``) at a public SPARQL endpoint and records what
they find and how long they take.

What it measures is deliberately narrow: schema-level facts. No instance data
is fetched and no discovery is run, so the pruning rate here is a statement
about the search space Assumption 1 leaves on that schema, not about accuracy.

It needs the network and the endpoint to be up, so ``run_all.sh`` does not call
it; the result file records the date of the run.

    python experiments/e6_endpoint.py
    python experiments/e6_endpoint.py https://example.org/sparql
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, quiet

OUT = os.path.join(OUT_DIR, "e6_endpoint.json")
DEFAULT = "https://semscout.istc.cnr.it/sparql"      # WHOW: water, health and weather open data


def probe(endpoint: str) -> dict:
    from causalway.constraints import EdgeConstraint
    from causalway.nodes import build_nodes
    from causalway.sources import resolve_schema

    rec = {"endpoint": endpoint, "date": datetime.date.today().isoformat()}
    t = time.perf_counter()
    schema = resolve_schema(endpoint)
    rec["schema_seconds"] = time.perf_counter() - t
    rec.update(schema.summary())
    nodes = build_nodes(schema, include_object_properties=True)
    rec["n_property_nodes"] = len(nodes)
    rec["n_data_nodes"] = sum(1 for n in nodes if n.kind == "data")
    for hops in (1, 2):
        t = time.perf_counter()
        stats = EdgeConstraint.from_schema(schema, nodes, max_hops=hops).stats()
        rec[f"constraint_seconds_k{hops}"] = time.perf_counter() - t
        rec[f"pruning_rate_k{hops}"] = stats["pruning_rate"]
        rec[f"n_allowed_k{hops}"] = stats["n_allowed"]
    return rec


if __name__ == "__main__":
    quiet()
    os.makedirs(OUT_DIR, exist_ok=True)
    result = probe(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    print(json.dumps(result, indent=2))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
