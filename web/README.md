# CausalWay web frontend

React 18 + TypeScript + Vite + Tailwind v4 + [React Flow](https://reactflow.dev) + Zustand.
Implements Plan 3 in **four** modules: preprocess, causal discovery, inference, counterfactual.
The old "causal model" step is gone — choosing and fitting a model happens in the inference rail,
beside the board it feeds (W22).

## Run it

Two ways, both talking to the FastAPI app in `service/api/main.py`.

**Single port (what a deployment looks like).** Build once, then let uvicorn serve `web/dist`:

```bash
npm --prefix web install && npm --prefix web run build
```

```bash
python -m uvicorn service.api.main:app --host 127.0.0.1 --port 8123 --timeout-keep-alive 75
```

`--timeout-keep-alive` is not decoration. uvicorn's 5-second default closes an idle
keep-alive connection, and a browser that reuses the pooled socket at that moment gets a
bare network error — "Load failed" in Safari, "Failed to fetch" in Chrome — on a request
that never reached the server. `api/client.ts` retries GETs and a short allowlist of POSTs
once for the same reason; the two together are what stopped "click Retrain, get Load failed".

Open <http://127.0.0.1:8123>.

**Two ports (what iterating looks like).** Keep the uvicorn command above running, and add the Vite
dev server — it proxies `/api` to port 8123 and hot-reloads the UI:

```bash
npm --prefix web run dev
```

Open <http://localhost:5173>.

## Layout

```
src/
├── main.tsx App.tsx          shell: TopBar + the active module
├── styles/tokens.css         the semantic palette, one value per scheme
├── api/client.ts             typed client for the §9 HTTP surface
├── stores/useStore.ts        one Zustand store: project, nodes, ledger, priors, model, answers, UI
├── app/                      TopBar, AppShell, ExportMenu, ImportButton, modules registry, ui primitives
├── lib/num.ts                number/name formatting, used by every chart and card
├── flow/                     cards, floating edges, DAG layout, layout sync
│   ├── DistCard.tsx          the distribution card modules 3 and 4 both render
│   └── charts/               BarDistribution (categorical) · DensityCurve (continuous)
└── modules/
    ├── preprocess/           source · schema · constraint · object encoding · ontology canvas ·
    │                         materialise · curation
    ├── discovery/            method · priors · runs · causal canvas · add-edge · ledger · curated graph
    ├── query/                shared by 3 and 4: the board, the command bar, the right-click menu,
    │                         the causal-model + evaluation panels, the query log
    ├── inference/            observe · intervene · predict every node
    └── counterfactual/       unit builder (constraint filtering *or* a described unit) · hypothetical do()
```

## Two rules about panels

**`step` is `module·position`** — `1·2`, never a bare running number. A single counter across the
app did not survive the modules being reshuffled, and it cannot survive `CausalModelPanel` being
rendered by modules 3 *and* 4: any number written inside a shared panel is wrong in one of its two
homes. Shared panels take `step` from the module that renders them. Panels that *report* rather than
ask — evaluation, the edge ledger, the query log, pinned answers — carry no number at all, because
numbering one implies an order to do it in that does not exist.

**Availability comes from the server, not from `schema !== null`.** A project imported from a bundle
has a fitted model and *no* source knowledge graph, and modules 3 and 4 are exactly the ones that
still work in that state. `capabilities` (from `GET /capabilities`) carries a reason per panel;
`TopBar` reads it and falls back to the old "everything except module 1 needs a schema" rule when it
is `null`, which is every ordinary project.

## Things worth knowing before changing it

- **Colour is load-bearing.** teal = class/entity, amber = literal, sky = numeric, indigo =
  causal/selected/target, green = valid/observed, orange = conflict/intervened/severed, red =
  invalid/failing. Every value is defined per scheme in `tokens.css`; nothing hard-codes a hex.
- **Tailwind extracts class names statically**, so tones are looked up as whole literals
  (`TONES` and `ROLE` in `flow/cards.tsx`) rather than built by interpolation. `` `text-${tone}` ``
  silently produces no CSS.
- **React Flow's `fitView` prop is init-only.** `flow/FitOnResize.tsx` re-fits on container resize;
  without it the graph drifts off-centre when a sidebar collapses. It takes an optional `focus`
  getter for a canvas whose subject is smaller than itself (the ontology canvas's chosen
  component) — without it, a panel below growing by one row silently zooms the user back out
  from the component they picked to all seventy.
- **Fitting to nodes that were just replaced does nothing.** Rebuilding the node array — which
  the ontology canvas does whenever the chosen component changes, because the dim styles live
  on the nodes — makes React Flow re-adopt every card as *unmeasured*, and a `fitView` in that
  window returns `true` and moves the camera not at all. Neither `useNodesInitialized` nor
  `nodes === seedNodes` marks the far edge of it reliably: both are already true on renders
  where the measurements still describe the outgoing cards. This is why "zoom to component" is
  a button rather than an automatic fly-to — which also suits W13, where the user's own
  arrangement is the authoritative one.
- **Node positions are server state**, not browser state — `PUT /api/projects/{id}/layout`,
  debounced on drag stop, one record per canvas. See Plan 3 W13. **Pane geometry is the
  opposite**: rail widths and centre-panel heights live in `localStorage` under `ckg-panes`
  and never touch the API, because a node's place in a graph is part of the artifact and
  how wide someone likes their sidebar is not.
- **A module fills the window and then lets you argue with it.** The centre column is
  `min-h-full`; the primary panel carries `grow` (absorbs the leftover space, so its bottom
  edge is the window's bottom edge, with a `clamp(320px, 52vh, 720px)` floor for when a
  sibling panel is taller than the window) and `resizeId` (a drag handle at its bottom that
  pins an explicit height; double-click removes the pin). Nothing is sized from a `vh`
  clamp any more — that is what used to leave a strip of background under the last panel.
- **React Flow inside a `grow` panel must be absolute-filled.** Its root is `height: 100%`,
  and a percentage height does not resolve against a parent sized by `flex: 1 1 0%` — the
  canvas silently collapses to 0 and the graph disappears. Every board is
  `<div className="relative min-h-0 flex-1"><div className="absolute inset-0">…`.
- **Two edge colourings in module 2, and both are load-bearing.** *Method* (the default)
  strokes an edge once per discovery method that found it, interleaving the dashes, so a
  striped edge is two methods agreeing; the map is fixed in `CausalCanvas.tsx`'s
  `METHOD_TONE`, not assigned in arrival order, or deleting a run would recolour the graph.
  *Status* is the original green/orange/red validity signal, one click away.
- **The prior is only ghosted behind a card where it means something** (`DistCard.showPrior`).
  A pinned card's posterior is a point mass, and module 4's reference is the unit's factual
  state, not the population marginal — in both cases the grey ghost bar reads as probability
  the answer does not contain.
- **Modules 3 and 4 share one board.** The difference between an intervention and a counterfactual is
  the *unit*, not the drawing. Don't fork `modules/query/`.
- **Three layout seeds on purpose.** Module 1's ontology canvas seeds on a **force-directed**
  relaxation (`flow/forceLayout.ts`) — a T-Box has no direction and no ordering, what it has is
  adjacency. Module 2 seeds on a ring (a ledger of competing claims with no agreed direction);
  modules 3–4 seed on topological layers, so depth means causal depth. All three are *seeds*:
  one-shot, deterministic, and overruled by the stored layout from the first drag. Nothing
  animates — a live simulation would re-settle on every curation click and rearrange a canvas
  the user had already organised.
- **The connected component is a choice (W31).** One `materialise` is one basic graph pattern
  over one component of the class graph — Assumption 1 forbids every edge between two classes
  with no relation path, so joining two of them could only make a cross product. What was
  missing is that nothing ever *picked* one: the server fell back to the largest and the client
  never sent anything else, which is invisible on the single-component samples and hides 69 of
  70 on a real endpoint. `Join scope` (1·3) picks it, `component: null` still means *auto* (the
  largest, re-decided as the curation changes), and materialising pins whatever ran. The
  curation list scopes itself to that component, the ontology canvas draws the rest of the
  T-Box faint rather than hiding it, and clicking any class card switches to its component.
- **An object property pointing at a class with no data properties starts as a `variable`**
  (W30). An age band or a tumour stage modelled as a class contributes no columns through a
  join, so the `relationship` default was dropping the one fact it carries. The `cat` badge in
  the curation panel marks them; it is a fact about the *schema*, so it stays on after the user
  overrides the role.
- **A card is the answer.** `POST /infer/predict` returns every node's posterior off one shared
  sample set, and the cards draw it; there is no single-target answer panel any more. On a free
  card the prior stays drawn behind the posterior, so a bar that did not move is visibly a bar
  that did not move. There is no "How this was answered" panel either — the backend, the
  effective sample size and the elapsed time are badges in the command bar, beside the estimand
  they belong to.
- **Densities are computed server-side.** The KDE grid arrives as `x`/`y` on the response
  (`_kde_grid`). The browser never estimates a density — two clients would then disagree about the
  same model.
- **The estimand string is rendered by the server**, from the evidence it actually computed with.
  The client shows its own only while the board is dirty, and flags that with a
  "board moved — predict again" chip. It **wraps** rather than truncating: an estimand cut off
  mid-condition is not a statement of what was asked.
- **Materialise is a job, not a request.** Unlimited by default, started automatically, polled at
  2 Hz, cancellable — and cancelling *abandons* rather than halts, which the UI says out loud.
- **The service persists nothing.** A project is a session-scoped workspace in the API process's
  memory; the Export menu is the only durable copy. See Plan 3 W14.
