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
3. **Inference** — fit an SCM/CBN, evaluate it, then condition, intervene, and contrast arms.
   Any answer, or the whole log, exports as `cw:` Turtle.
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

```python
from causalway import OntologySchema, candidate_nodes, EdgeConstraint, materialize

schema = OntologySchema.from_file("kgs/ttls/synthetic_clinic.ttl", infer_missing=True)
nodes = candidate_nodes(schema)
constraint = EdgeConstraint.from_schema(schema, nodes, max_hops=2)   # Assumption 1, k = 2
mat = materialize(schema, nodes, source="kgs/ttls/synthetic_clinic.ttl")
```

```python
from runners.run_kg_discovery import run_algorithm, to_ocg

adj = run_algorithm("GES", mat.frame, constraint=constraint)
ocg = to_ocg(adj, constraint, method="GES")
ocg.to_rdf().serialize("results/ges.ttl", format="turtle")
```

```python
from causalway.model import CausalModel

model = CausalModel(ocg).fit(source="kgs/ttls/synthetic_clinic.ttl")
model.intervene({"Patient.smoking": "No"}, target="Patient.survival")
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

Discovered graphs, fitted models, queries with their observational conditions, answers, and
counterfactual worlds all round-trip to RDF under `cw:` (`http://sdm-causalway.org/`). Four rules:

1. **`causalway/vocab.py` is the single source of truth** for every term and IRI stem. No IRI is
   hard-coded elsewhere. `python -m causalway.vocab` regenerates `causalway/ontology.ttl`; the
   service serves the same at `GET /api/vocab`.
2. **Every exported triple is produced by an RML mapping**, never by `g.add(...)`. Four mappings,
   one per artifact: `ocg_mapping`, `model_mapping`, `query_mapping`, `cf_mapping`. To change an
   export's shape, edit the mapping — their headers document SDM-RDFizer's quirks. (The one
   exception is the opt-in named-graph projection, which asserts entity-level *quads* and so cannot
   travel through a mapping whose contract is that it never does.)
3. **IRIs are minted deterministically**, so the same model/query/intervention is the same resource
   across runs and queries round-trip.
4. **The vocabulary is subtractive.** Nothing is minted that other triples already entail, or that
   standard syntax already expresses.

The vocabulary keeps *seeing* and *doing* apart at the link: `cw:hasEvidence` is a conditional
query's observed evidence; `cw:hasCondition` is the sub-population an interventional estimate is
restricted to (a CATE). Both point at a `cw:Condition`.

`(D_p, p, R_p)` is deliberately **not** expressed with `rdfs:domain`/`rdfs:range`: their entailments
would type every node as `rdf:Property`, and multiple domains are a *conjunctive* global constraint
— the opposite of the per-node local scoping meant here.

The **counterfactual world** export is the only artifact describing named entities from the source
KG, and therefore the only one that can be merged back into it. Everything else loads *alongside*
the source graph.

---

## Testing

```bash
python -m pytest tests -q
```

166 passing. Covers `causalway/`, plus `service/api/bundle.py`, an export→import→export round trip
through real FastAPI routes, counterfactual batching, and the query RML export. **The frontend has
no tests.**

---

## Known limitations

Documented on purpose — the module docstrings say so too; please don't delete those comments to
make the source look tidy.

- **Row duplication breaks i.i.d.** `evaluation.row_weights` is an approximation, not a fix.
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
