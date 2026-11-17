# CausalWay

**Causal discovery, causal modelling and counterfactual reasoning over the *properties* of a
knowledge graph — with every intermediate result as typed, provenance-bearing RDF.**

Give CausalWay a knowledge graph (Turtle/N-Triples file or SPARQL endpoint) and it induces the
schema, turns every property into a candidate causal variable, prunes the search space with an
ontological assumption, runs causal discovery, fits a causal model, and answers conditional,
interventional and entity-level counterfactual questions. Everything it learns exports as RDF.

| Component | What it is |
|---|---|
| `causalway` + `algs` + `runners` | Python package: KG → schema → flat join → discovery → SCM/CBN → queries → RDF |
| **CausalWay Studio** | Four-module web app over the same engine — one FastAPI process, one port |
| CLI runners | `run_kg_discovery.py` (discovery), `run_causal_model.py` (fit / query) |
| `examples/quickstart.py` | The whole path in one file, ~1 minute, offline |
| `experiments/` | **KGBench** (KGs with a known causal graph) and the evaluation, one command |

---

## The idea

A knowledge graph is a directed edge-labelled graph *KG = (V, L, E)*. Every property *p* has a
domain class *D<sub>p</sub>* and a range *R<sub>p</sub>*.

CausalWay does causal discovery not over entities but over **property nodes**:

> **Property node.** A causal variable is the triple **(D<sub>p</sub>, p, R<sub>p</sub>)** — the
> property, the class it hangs off, and its range. `Patient.smoking` and `Hospital.region` are
> different variables even for the same predicate, because they belong to different classes.

Which pairs may be causally related is decided by the ontology, not the data:

> **Assumption 1 (Topological Causal Assumption).** A causal relationship can exist between
> *p<sub>i</sub>* and *p<sub>j</sub>* **iff** either **(1) intra-class** —
> `domain(p_i) = domain(p_j)` — or **(2) inter-class** — their domains are connected by a relation
> path of at most *k* object-property hops, *k* chosen by the user.

**This is a hard constraint, not a soft prior.** It compiles to an *n × n* boolean feasibility
matrix pushed *inside* each algorithm — expert knowledge for PC/GES, L-BFGS-B bounds for NOTEARS,
a prior matrix for LiNGAM — so forbidden edges are never scored, not scored then deleted. Raising
*k* widens the admissible edge set; the UI reports the pruning rate as you move it.

The result is the **Ontological Causal Graph (OCG)**: nodes are property nodes, each edge carries a
relation label from **P**<sub>obj</sub> ∪ {ε} (ε = intra-entity). Multi-hop labels are SPARQL 1.1
property paths (`^<a>/<b>`).

**Materialisation is a flat join with no root class.** One SPARQL basic graph pattern per connected
component of the class-level schema graph: one variable per class, one row per solution binding.
1:N relations expand into several rows. Every column keeps a **double provenance** — the causal node
it realises *and* the BGP variable holding the entity that owns it — which is what lets an inferred
value be attributed back to the right entity.

> Row duplication from 1:N expansion violates i.i.d. This is a **documented limitation, not a bug**.
> `causalway.evaluation.row_weights` is an approximate mitigation and says so.

```
     ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────────┐
KG → │ 1 Preprocess │ → │ 2 Discovery   │ → │ 3 Inference  │ → │ 4 Counterfactual │ → RDF
     │ schema       │   │ PC · GES      │   │ fit SCM/CBN  │   │ abduct a unit    │
     │ nodes        │   │ GES-Prior     │   │ observe      │   │ hypothetical do()│
     │ Assumption 1 │   │ NOTEARS·DAGMA │   │ intervene    │   │ factual vs CF    │
     │ flat join    │   │ LiNGAM·DAG-GNN│   │ evaluate     │   │                  │
     └──────────────┘   └───────────────┘   └──────────────┘   └──────────────────┘
```

---

## Install

Python **3.11**. The scientific stack (torch, gcastle, dowhy, pgmpy) installs most smoothly under
conda.

```bash
git clone https://github.com/SDM-TIB/CausalWay.git && cd CausalWay
```

```bash
conda create -n causalway python=3.11 -y && conda activate causalway && pip install -r requirements.txt
```

There is **no `pip install -e .`** — every entry point puts the repository root on `sys.path`
itself. Run commands from the repository root.

RDF export needs **SDM-RDFizer** on `PATH`. Only the `GES-Prior` method needs an LLM key
(`cp algs/.env.example algs/.env`); it is read server-side and the UI has no API-key field. Every
other method, all fitting and all inference run offline.

---

## Run the web service

```bash
npm --prefix web install && npm --prefix web run build
```

```bash
python -m uvicorn service.api.main:app --host 127.0.0.1 --port 8123 --timeout-keep-alive 75
```

Then open <http://127.0.0.1:8123>. One process serves the API and the built frontend, so one port
is one deployment.

> `--timeout-keep-alive 75` is not decoration. uvicorn closes idle keep-alive connections after 5s;
> a browser reusing a socket the server just closed gets "Load failed" / "Failed to fetch".

Two sample graphs ship with it: `synthetic_clinic` (3 classes, known ground-truth DAG) and
`SCLC_patients` (single class, real). For frontend work, `npm --prefix web run dev` serves :5173
and proxies `/api` to :8123.

### The four modules

1. **Preprocess** — load a source, induce the schema, pick property nodes, set the constraint
   (*k* hops, relation direction, ε), preview the SPARQL, materialise the flat join.
2. **Discovery** — run methods, accumulate an edge ledger across runs, curate the graph by hand.
3. **Inference** — fit an SCM/CBN, evaluate it, then condition and intervene.
   Any answer, or the whole log, exports as `cw:` Turtle, with the model and graph behind it.
4. **Counterfactual** — abduct a named entity (or a hand-described unit), act, compare factual
   against counterfactual.

| | | | |
|---|---|---|---|
| ![Preprocess](docs/images/01-preprocess.png) | ![Discovery](docs/images/02-discovery.png) | ![Inference](docs/images/03-inference.png) | ![Counterfactual](docs/images/04-counterfactual.png) |

**Nothing is persisted.** `PROJECTS` is a dict in one process's memory; export is the only durable
copy. A project exports to a zip (including the fitted model, excluding training data and the
source KG) and imports back — so an imported model answers counterfactuals about a hand-described
unit without its source KG.

---

## Python package

The whole path in one runnable file — about a minute on the bundled clinic KG, offline:

```bash
python examples/quickstart.py
```

```python
from runners.run_kg_discovery import build_context, run_algorithm, to_ocg

KG = "kgs/ttls/synthetic_clinic.ttl"
ctx = build_context(KG)                               # schema, property nodes, Assumption 1, flat join
adj = run_algorithm("GES", ctx, constrained=True)     # the constraint is inside the search
ocg = to_ocg(adj, ctx, constrained=True, method="GES")  # one DAG, with relational-path labels
```

```python
from causalway.model import CausalModel

model = CausalModel.fit(ocg, KG, include_object_properties=False, random_state=0)
model.condition("Patient.survival", {"Patient.smoking": "Yes"})                 # observe
model.intervene({"Hospital.careQuality": "High"}, target="Patient.survival")    # intervene
patient = sorted(set(model.spec.mat.entity_ids["Patient"]))[0]
model.counterfactual({(None, "Patient.smoking"): "No"}, entity=patient,        # counterfactual
                     target="Patient.tumorStage")
```

```python
g = model.spec.ocg.to_rdf()           # the causal graph as cw: RDF ...
model.to_rdf(graph=g)                 # ... plus the fitted model; see examples/quickstart.py
g.serialize("causal_layer.ttl")       # for queries, answers and SHACL validation
```

Plan 2 modules (`model`, `mechanisms`, `entities`, `inference`, `queries`, `evaluation`) depend on
dowhy/pgmpy and are **deliberately not imported by the package `__init__`**, so discovery-only users
do not pay a ~30s import.

CLI: `python -m runners.run_kg_discovery --help`, `python -m runners.run_causal_model --help`.

### Notebooks

`notebooks/` holds the experiments; the logic is imported from `runners/`, never copied.

| Notebook | What it does |
|---|---|
| `01_sclc_causal_discovery` | Every method on the real single-class SCLC graph |
| `02_synthetic_kg_experiments` | Constrained vs. unconstrained discovery, scored against a known DAG |
| `03_sclc_causal_model` | Fit, query, store as RDF, replay the stored queries |
| `04_synthetic_causal_model` | Graph error vs. model error, scored against the true SEM |
| `05_endpoint_causal_pipeline` | The whole pipeline over a SPARQL endpoint instead of a file |

01 and 02 read cached LLM priors from `results/priors_*.json`, so they need no API key. 05 expects
`kgs/ttls/synthetic_clinic.ttl` loaded into a repository `syn_clinic` at
`http://localhost:7200` (e.g. GraphDB); it runs every method twice and takes ~35 minutes.

---

## RDF output and the `cw:` vocabulary

Discovered graphs, fitted models, and conditional / interventional / counterfactual queries with
their answers all round-trip to RDF under `cw:` (`http://sdm-causalway.org/`). The model in one
screen:

```
cw:OntologicalCausalGraph   cw:discoveredBy cw:CausalDiscoveryRun
cw:CausalDiscoveryRun       cw:sourceKG, cw:byMethod, cw:discoveredTime, cw:hasParameter cw:Parameter
cw:Parameter                cw:parameterName, cw:parameterValue (typed), rdfs:comment
cw:CausalNode               rdfs:label, cw:inGraph, cw:domain, cw:property, cw:range      -- (D_p, p, R_p)
cw:CausalEdge               rdfs:label, cw:inGraph, cw:cause, cw:effect, cw:curated,
                            cw:weight?, cw:viaRelationalPath*  ("^<treatedAt>/<receives>")

cw:CausalModel              cw:onGraph, cw:fittedBy cw:ModelFittingRun, cw:artifactChecksum,
                            cw:hasMechanism cw:CausalMechanism (cw:onNode, cw:mechanismType, cw:invertible)

cw:{Conditional|Interventional|Counterfactual}Query
                            cw:onGraph, cw:hasIntervention (X), cw:hasCondition (Z), cw:aboutEntity (e)
cw:Intervention / cw:Condition   cw:onNode, cw:onProperty, cw:setValue / cw:observedValue
cw:{Conditional|Interventional|Counterfactual}Answer        -- one per target node
                            cw:forQuery, cw:evaluatedBy (a model), cw:evaluatedTime, cw:onNode,
                            cw:onProperty, cw:predictedValue, cw:probability | cw:std, cw:factualValue
```

A **query is posed on a graph; an answer is evaluated by a model** — so one query can carry the
answers of several models, and every answer traces to the model (and through it, the fitting run,
the source KG and the artifact checksum) that produced it. The target set V is not stored: it is
the nodes the answers are about. An average causal effect is two interventional queries.

Five rules:

1. **`causalway/vocab.py` is the single source of truth** for every term and IRI stem.
   `python -m causalway.vocab` regenerates `causalway/ontology.ttl`; the service serves the same
   at `GET /api/vocab`.
2. **Every exported triple is produced by an RML mapping**, never by `g.add(...)`. Three mappings,
   one per artifact: `ocg_mapping`, `model_mapping`, `query_mapping` (a counterfactual *world* is a
   counterfactual query and its answers). To change an export's shape, edit the mapping — their
   headers document SDM-RDFizer's quirks.
3. **`causalway/shapes.ttl` states the constraints**, as SHACL: the graph is a DAG (a PC/GES CPDAG is
   oriented into one DAG of its Markov equivalence class before export — no new v-structure, the
   choice recorded — and the exporter refuses a cyclic graph), answer and query
   types match, V, X and Z are disjoint (a counterfactual may condition on what it intervenes on —
   that is what makes it one), Assumption 1 holds where the run enforced it.
   `tests/test_shapes.py` runs it against real exports and against data broken on purpose.
4. **IRIs are minted deterministically**, so the same question on the same graph is one resource.
5. **The vocabulary is subtractive.** Nothing is minted that other triples already entail (a node's
   kind is read off its range; an intra-class edge is one with no relational path), and no foreign
   vocabulary is aligned to for the sake of a citation.

`(D_p, p, R_p)` is deliberately **not** expressed with `rdfs:domain`/`rdfs:range`: their entailments
would type every node as `rdf:Property`, and multiple domains are a *conjunctive* global constraint
— the opposite of the per-node local scoping meant here. Causal nodes belong to one graph; compare
two graphs by joining on `(cw:domain, cw:property, cw:range)`.

The **counterfactual** export is the only artifact that names entities of the source KG
(`cw:aboutEntity`), and it never asserts `entity property value`, so it merges back into the source
KG without contradicting it.

---

## Reproducing the evaluation

```bash
bash experiments/run_all.sh
```

About 25 minutes on 8 cores (`--quick` for a 3-minute smoke test; `PY=/path/to/python` to choose
the interpreter). Every run appends one JSON line to `experiments/out/*.jsonl`, so an interrupted run
resumes; `experiments/out/REPORT.md` and `summary.json` hold the aggregated tables. The outputs of
the run reported in the paper are committed in `experiments/out/`.

| Script | Question | Data |
|---|---|---|
| `e1_ges_correctness.py` | Is the project's GES a correct GES? (vs. the inherited search, vs. pgmpy) | Asia, Sachs, Child, Insurance |
| `e2_constraint.py` | What does Assumption 1 buy as the schema grows — unconstrained vs. post-hoc filter vs. in the search; what does the entity-level check buy? | 120 KGBench KGs |
| `e3_ges_prior.py` | Does GES-Prior use a prior well? | clinic KG × 10 seeds, cached LLM priors |
| `e4_interventional.py` | Are interventional answers right? Scored against the generator's truth | 12 KGBench KGs |
| `e5_efficiency.py` | What does each stage cost; does the knowledge layer answer a path query? | bundled KGs |
| `e6_endpoint.py` | Schema induction on a live public endpoint (needs network; not in `run_all.sh`) | WHOW |

**KGBench** (`experiments/kgbench.py`) generates snowflake-schema KGs — one fact class, a random
tree of dimension classes — with a ground-truth causal graph that satisfies Assumption 1 and
additive-logit categorical mechanisms. It writes plain RDF and can sample the *intervened*
mechanism, which is what makes an interventional answer checkable:

```python
from experiments.kgbench import make_mechanism

mech = make_mechanism(n_classes=5, n_attrs=3, seed=0)
mech.sample(1000, seed=0).write_ttl("bench.ttl")              # the KG
truth = mech.sample(200_000, seed=1, do={(1, 0): 0}).flat()   # P(. | do(C1.a1_0 = L0))
```

### The learner (`algs/ges_prior`)

GES and GES-Prior are this repository's own. The search began as a copy of gCastle's; two inherited
defects are fixed (the backward phase never deleted an edge; the DAG→CPDAG conversion kept some
reversible edges directed) and the forward phase's validity test is now a reachability check
instead of an enumeration of paths. `discover_structure(..., entity_ids=, owner_of=)` adds the
**entity-level check**: an edge is kept only if the entities that own its effect support it, not
merely the rows the join repeats. `tests/test_ges_prior.py` pins each of these.

---

## Testing

```bash
python -m pytest tests -q
```

238 passing. Covers `causalway/`, the learner in `algs/ges_prior`, plus `service/api/bundle.py`, an
export→import→export round trip through real FastAPI routes, counterfactual batching, the RML
exports, and the SHACL shapes — run against every export and against data broken on purpose
(`tests/test_shapes.py`). **The frontend has no tests.**

---

## Known limitations

Documented on purpose — the module docstrings say so too; please don't delete those comments to
make the source look tidy.

- **Row duplication breaks i.i.d.** A 1:N join repeats an entity's attributes, and a score computed
  on the join counts the copies as samples. The entity-level check (`algs/ges_prior/entity_check.py`)
  removes the false edges this creates where an edge's family has one row per owning entity; it does
  not cover a cause on the N side of a relation, and `evaluation.row_weights` remains an
  approximation for model fitting. The check is a library option, not yet a Studio control.
- **The runner's default join role can be a cross product.** `runners.build_context` treats each
  object property as a join *and* a variable; on a class with two N:1 relations to large classes the
  query degenerates. Studio's default (relationship only) does not; `experiments/common.py`
  (`context_from_ttl`) shows the equivalent library call.
- **`CausalModel.fit` can fail on a rare categorical level.** dowhy's automatic mechanism selection
  cross-validates each node and raises when a level is absent from a fold.
- **A counterfactual answer from the Python API carries no factual value.** Studio's export does
  (`cw:factualValue`); to get the same from the library, export through
  `causalway.cf_export.CounterfactualWorld` as `experiments/e5_efficiency.py` does.
- **Categorical counterfactuals are not point-identified.** Gumbel-max coupling is one admissible
  choice among many.
- **`condition` tier 1b is unimplemented** — the ladder skips pgmpy-exact to likelihood-weighting.
- **Nothing is persisted.** No Postgres, Redis, Celery, MinIO or WebSocket channel; everything is
  synchronous and in-process.
- **Cancelling a materialise job abandons it, it does not stop it.** A Python thread cannot be
  killed from outside; the UI says so.
- **`CausalModel.fit` re-materialises from the source**, paying the join twice.
- **A unit spanning many rows has an aggregated factual state**, and the unit filter matches *rows*.
- **No save-back-to-KG.** There is no SPARQL UPDATE push of inferred values into the source graph.
- **CSV ingest is not wired up.** `kgs/csvs/` cannot yet be used through the web service.
- **Bundles are version 2.** A bundle exported before the CausalKG → CausalWay rename is refused
  with an explicit message rather than a `ModuleNotFoundError` from a stale pickle.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
