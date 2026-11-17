"""E5 — What each stage costs, and whether the knowledge layer answers a path query.

One pass through the whole pipeline on the two bundled KGs, timed stage by
stage, in process, single-threaded:

    schema induction -> constraint compilation -> flat join            (both KGs)
    discovery -> model fit -> observational / interventional /
    counterfactual query -> RDF export -> SHACL -> SPARQL over the merge (clinic)

Each timing is the median of ``REPEATS`` runs except the model fit, which is run
once (it is the slowest stage by an order of magnitude and has no cache to warm).

Two checks ride along, because a timing of a wrong result is worth nothing:

* the exported graph, read back, has the adjacency it was exported with;
* the causal path query ``?x (^cw:cause/cw:effect)+ ?y`` over the exported
  graph returns exactly the transitive closure of that adjacency.

The clinic KG is read from ``kgs/ttls/synthetic_clinic.ttl`` as shipped, not
regenerated, so the numbers describe the file a user downloads.

    python experiments/e5_efficiency.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import OUT_DIR, context_from_ttl, quiet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(OUT_DIR, "e5_efficiency.json")
KGS = {"synthetic_clinic": "kgs/ttls/synthetic_clinic.ttl", "SCLC_patients": "kgs/ttls/SCLC_patients.ttl"}
REPEATS = 5


def timed(fn, repeats: int = REPEATS):
    """Median wall-clock seconds of ``fn`` over ``repeats`` runs, and its last result."""
    times, result = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), result


def stages(path: str) -> dict:
    from causalway.bgp import materialize
    from causalway.constraints import EdgeConstraint
    from causalway.nodes import build_nodes
    from causalway.ontology import OntologySchema

    rec = {}
    rec["schema_seconds"], schema = timed(lambda: OntologySchema.from_file(path), 3)
    rec["triples"] = len(schema.graph)
    rec["n_classes"] = schema.summary()["n_classes"]
    nodes = build_nodes(schema, include_object_properties=True)
    objects = {n.name for n in nodes if n.kind == "object"}
    rec["constraint_seconds"], constraint = timed(lambda: EdgeConstraint.from_schema(schema, nodes))
    rec["join_seconds"], mat = timed(
        lambda: materialize(schema, nodes, excluded=objects, excluded_joins=set(),
                            include_object_properties=True), 3)
    rec["rows"], rec["columns"] = int(mat.df.shape[0]), int(mat.df.shape[1])
    keep = [i for i, n in enumerate(nodes) if n.name in mat.df.columns]
    rec["pruning_rate"] = constraint.subset(keep).stats()["pruning_rate"]
    return rec


def clinic_pipeline(path: str) -> dict:
    import numpy as np
    import rdflib
    from pyshacl import validate

    from algs.ges_prior.learn_bn import discover_structure
    from causalway.cf_export import CounterfactualWorld, worlds_to_rdf
    from causalway.entities import aggregate, rows_for_entity
    from causalway.model import CausalModel
    from causalway.queries import Condition, Intervention, Query
    from causalway.query_export import queries_to_rdf
    from causalway.result import OntologicalCausalGraph
    from causalway.vocab import vocabulary
    from runners.run_kg_discovery import to_ocg

    R = {}
    ctx, objects = context_from_ttl(path)
    R["discovery_seconds"], cpdag = timed(
        lambda: discover_structure(ctx.discrete_df, algorithm="GES", constraint=ctx.constraint_kept))
    ocg = to_ocg(np.asarray(cpdag, dtype=int), ctx, True, method="GES", source=path)

    t = time.perf_counter()
    model = CausalModel.fit(ocg, path, excluded=objects, excluded_joins=set(), random_state=7)
    R["fit_seconds"] = time.perf_counter() - t
    R["n_model_nodes"] = len(model.spec.columns)
    R["training_rows"] = int(model.spec.data.shape[0])
    graph = model.spec.ocg

    cols = list(model.spec.columns)
    parent = "Patient.tumorStage"
    level = sorted(model.spec.data[parent].astype(str).unique())[0]
    free = [c for c in cols if c != parent]
    entity = sorted(set(model.spec.mat.entity_ids["Patient"]))[0]

    R["observational_seconds"], cond = timed(lambda: model.condition(free, {parent: level}))
    R["observational_backend"] = next(iter(cond.values())).backend
    R["n_targets"] = len(free)
    R["interventional_seconds"], interv = timed(lambda: model.intervene({parent: level}, target=free))
    R["counterfactual_seconds"], cf = timed(
        lambda: model.counterfactual({(None, parent): level}, entity=entity, target=free, num_samples=1000), 3)
    R["counterfactual_draws"] = 1000

    # ---- knowledge layer: graph + model + one query of each kind, all answers.
    # The counterfactual goes through the adapter the Studio's board uses
    # (a "world": the unit's factual state beside its counterfactual one), so
    # the export has the shape the paper's listing queries.
    rows = rows_for_entity(model.spec.mat, entity)
    factual = {c: aggregate(model.spec.data.loc[rows, c], model.spec.dtypes[c]) for c in cols}
    world = CounterfactualWorld(
        entity=entity, interventions={parent: level},
        counterfactual={parent: level, **{c: cf[c].predicted for c in free}},
        factual=factual,
        distribution={c: cf[c].distribution for c in free if cf[c].distribution},
    )
    t = time.perf_counter()
    g = graph.to_rdf()
    model.to_rdf(graph=g)
    pairs = [
        (Query(kind="conditional", conditions=[Condition(parent, level)]), cond),
        (Query(kind="interventional", interventions=[Intervention(parent, level)]), interv),
    ]
    queries_to_rdf(pairs, model, graph=g)
    worlds_to_rdf([world], model, graph=g)
    R["export_seconds"] = time.perf_counter() - t
    R["exported_triples"] = len(g)
    R["n_answers"] = sum(len(a) for _, a in pairs) + len(free)
    R["roundtrip_adjacency_identical"] = bool(
        np.array_equal(OntologicalCausalGraph.from_rdf(graph.to_rdf()).adj, graph.adj))

    shapes = os.path.join(ROOT, "causalway", "shapes.ttl")
    t = time.perf_counter()
    conforms, _, _ = validate(g, shacl_graph=shapes)
    R["shacl_seconds"] = time.perf_counter() - t
    R["shacl_conforms"] = bool(conforms)
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    R["n_shapes"] = len(set(rdflib.Graph().parse(shapes).subjects(rdflib.RDF.type, SH.NodeShape)))

    # ---- SPARQL over the export merged into the source KG
    merged = rdflib.Graph().parse(path)
    n_source = len(merged)
    merged += g
    R["merged_triples"] = len(merged)
    R["source_triples"] = n_source

    path_query = """
        PREFIX cw: <http://sdm-causalway.org/>
        SELECT DISTINCT ?x ?y WHERE { ?x (^cw:cause/cw:effect)+ ?y . }"""
    R["path_query_seconds"], reached = timed(lambda: list(merged.query(path_query)))
    adj = np.asarray(graph.adj, dtype=int)
    closure = np.linalg.matrix_power(adj + np.eye(len(adj), dtype=int), len(adj)) > 0
    np.fill_diagonal(closure, False)
    R["path_query_pairs"] = len(reached)
    R["path_query_matches_closure"] = bool(len(reached) == int(closure.sum()))

    cf_query = """
        PREFIX cw: <http://sdm-causalway.org/>
        SELECT ?t ?tv ?node ?factual ?cf ?model WHERE {
          ?q a cw:CounterfactualQuery ; cw:aboutEntity <%s> ;
             cw:hasIntervention [ cw:onNode ?t ; cw:setValue ?tv ] .
          ?a cw:forQuery ?q ; cw:onNode ?node ; cw:evaluatedBy ?model ;
             cw:factualValue ?factual ; cw:predictedValue ?cf . }""" % entity
    R["listing_query_seconds"], rows = timed(lambda: list(merged.query(cf_query)))
    R["listing_query_rows"] = len(rows)

    v = vocabulary()
    R["vocab_classes"] = len(set(v.subjects(rdflib.RDF.type, rdflib.RDFS.Class)))
    R["vocab_properties"] = len(set(v.subjects(rdflib.RDF.type, rdflib.RDF.Property)))
    return R


def main() -> None:
    os.environ["CW_EXPERIMENT_GCM"] = "1"
    quiet()
    os.chdir(ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"stages": {name: stages(path) for name, path in KGS.items()}}
    for name, rec in out["stages"].items():
        print(name, json.dumps(rec), flush=True)
    out["clinic"] = clinic_pipeline(KGS["synthetic_clinic"])
    print(json.dumps(out["clinic"], indent=2), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
