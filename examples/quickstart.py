"""CausalWay in one file: a KG in, answered causal queries and validated RDF out.

    python examples/quickstart.py

Runs on the bundled three-class clinic KG in about a minute, offline. Every call
below is the public API; the web application (Studio) drives the same functions.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.run_kg_discovery import build_context, run_algorithm, to_ocg

KG = "kgs/ttls/synthetic_clinic.ttl"

# 1. KG -> schema -> property nodes -> Assumption 1 -> flat join (with provenance).
ctx = build_context(KG)
print(f"{len(ctx.column_names)} causal variables, {len(ctx.discrete_df)} rows; "
      f"Assumption 1 forbids {ctx.constraint_kept.stats()['pruning_rate']:.0%} of ordered pairs")

# 2. Causal discovery with the constraint inside the search, then one DAG.
adj = run_algorithm("GES", ctx, constrained=True)
ocg = to_ocg(adj, ctx, constrained=True, method="GES")
names = [n.name for n in ocg.nodes]
for i, j in ocg.edges():
    print(f"  {names[i]} -> {names[j]}")

# 3. Fit a structural causal model on the KG's own instances.
from causalway.model import CausalModel  # noqa: E402 — imports dowhy; only needed from here on

model = CausalModel.fit(ocg, KG, include_object_properties=False, random_state=0)

# 4. Three rungs of the causal hierarchy.
target = "Patient.survival"
seen = model.condition(target, {"Patient.smoking": "Yes"})                       # Q1: observe
done = model.intervene({"Hospital.careQuality": "High"}, target=target)          # Q2: intervene
patient = sorted(set(model.spec.mat.entity_ids["Patient"]))[0]
would = model.counterfactual({(None, "Patient.smoking"): "No"},                  # Q3: counterfactual
                             entity=patient, target="Patient.tumorStage")
print("P(survival | smoking=Yes)        ", seen.distribution)
print("P(survival | do(careQuality=High))", done.distribution)
print(f"tumorStage of {patient.rsplit('/', 1)[-1]} had they not smoked:", would.predicted)

# 5. Everything as RDF under the cw: vocabulary, checked by the SHACL shapes.
from pyshacl import validate  # noqa: E402

from causalway.queries import Intervention, Query  # noqa: E402
from causalway.query_export import queries_to_rdf  # noqa: E402

g = model.spec.ocg.to_rdf()
model.to_rdf(graph=g)
query = Query(kind="interventional", interventions=[Intervention("Hospital.careQuality", "High")])
queries_to_rdf([(query, {target: done})], model, graph=g)
conforms, _, _ = validate(g, shacl_graph="causalway/shapes.ttl")
g.serialize("quickstart_causal_layer.ttl", format="turtle")
print(f"{len(g)} triples written to quickstart_causal_layer.ttl; SHACL conforms: {conforms}")
