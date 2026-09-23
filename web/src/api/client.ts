/** Typed client for the §9 HTTP surface. One place that knows about fetch. */

export interface PanelCapability {
  available: boolean
  /** Why not, in the user's terms; `null` when it is available. */
  reason: string | null
}

/** W29 — which panels can do anything in this project, and why not when they cannot. */
export interface Capabilities {
  source: PanelCapability
  ontology: PanelCapability
  nodes: PanelCapability
  materialise: PanelCapability
  discovery: PanelCapability
  model: PanelCapability
  inference: PanelCapability
  counterfactual: PanelCapability
  /** Entity-level work needs the *rows*, not just the model. */
  entities: PanelCapability
  import_notes: string[]
}

export interface ImportResult {
  id: string
  name: string
  bundle_version: string
  has_model: boolean
  notes: string[]
  source_hint: { kind?: string; name?: string; url?: string; sha256?: string } | null
  capabilities: Capabilities
}

export type Kind = 'data' | 'object'

/**
 * W26. `relationship` joins the two classes into one row; `variable` makes the
 * related entity's IRI a column and *drops* that join; `dropped` is neither.
 * The default is `relationship`.
 */
export type ObjectRole = 'relationship' | 'variable' | 'dropped'

export interface RoleOptions {
  role: ObjectRole
  can_be_relationship: boolean
  /** False when this join is the only thing holding two classes in the same row. */
  can_be_variable: boolean
  /** Why the unavailable role is unavailable; `null` when both are open. */
  reason: string | null
}
export type ValueHint = 'numeric' | 'string' | 'date' | 'entity'

export interface NodeInfo {
  name: string
  domain: string
  prop: string
  range: string
  kind: Kind
  value_hint: ValueHint
  excluded: boolean
  /** Object properties only: is the relationship joined? */
  join_excluded: boolean
  /**
   * W26 — an object property is a relationship XOR a causal variable, never
   * both. `null` for a data property, which has no role to choose.
   */
  role: ObjectRole | null
  role_options: RoleOptions | null
  /**
   * W30 — object properties only: the range class declares no data properties,
   * so it is an attribute someone modelled as a class (an age band, a stage).
   * These start in the `variable` role rather than `relationship`, because
   * joining a class with no columns contributes nothing. A fact about the
   * schema, so it stays `true` after the user overrides the role.
   */
  auto_variable: boolean
  /**
   * W31 — index of the connected component this node's *domain* sits in. Only
   * one component is materialised at a time, so this is what says whether the
   * node is in the query currently on screen.
   */
  component: number | null
  n_distinct: number | null
  n_rows: number | null
  near_unique: boolean
  constant: boolean
  /** How this column will be typed at fit time, under the current type options (W20). */
  dtype: DType | null
}

export interface TypeOptions {
  max_levels: number
  float_as_continuous: boolean
  default_max_levels: number
  note: string
}

export interface ConstraintOptions {
  relation_direction: 'both' | 'forward'
  allow_epsilon: boolean
  max_hops: number
}

export interface ConstraintStats {
  n_nodes: number
  n_allowed: number
  n_forbidden: number
  pruning_rate: number
  [k: string]: number
}

/**
 * One connected component of the class graph — one candidate join (W31).
 *
 * `classes` is what makes it pickable: an index is not a choice a human can make
 * among seventy. `n_retained` is how many of its nodes are candidate variables
 * right now, and zero is the one value `materialise` refuses outright, so the
 * picker can say so before the click rather than after.
 */
export interface ComponentInfo {
  index: number
  n_classes: number
  n_nodes: number
  n_retained: number
  classes: string[]
}

export interface NodesPayload {
  nodes: NodeInfo[]
  edges: { source: string; target: string; label: string; bidirectional: boolean }[]
  stats: ConstraintStats
  constraint_options: ConstraintOptions
  type_options: TypeOptions
  dtypes: Record<string, DType>
  components: ComponentInfo[]
  /** What `component: null` resolves to at this curation; `null` when nothing is retained. */
  auto_component: number | null
}

export interface SchemaPayload {
  source: { kind: string; name: string; sha256: string | null; bytes: number | null } | null
  schema: {
    n_classes: number
    n_object_properties: number
    n_data_properties: number
    n_object_domain_range: number
    n_data_domain_range: number
    n_inferred: number
    classes: string[]
    conflicts: string[]
  }
  n_nodes: number
}

export interface MatResult {
  component: number
  n_components_total: number
  row_count: number
  column_count: number
  columns: string[]
  classes: string[]
  multiplicity: Record<string, number>
  skipped_patterns: string[]
  warnings: string[]
  sparql: string
  preview_rows: Record<string, unknown>[]
  distinct_counts: Record<string, number>
  dtypes: Record<string, DType>
  has_multi_relation: boolean
  truncated: boolean
  limit: number | null
  near_unique_columns: string[]
}

/** The SPARQL `materialise` would run, without having run it (live preview). */
export interface GraphPatternPreview {
  component: number
  sparql: string
  warnings: string[]
}

export type MatJobState = 'idle' | 'running' | 'done' | 'error' | 'cancelled'

export interface MatJob {
  job_id?: string
  state: MatJobState
  elapsed: number
  limit?: number | null
  component?: number
  error?: string | null
  /** Past `slow_after` seconds — the point at which a row limit is worth offering. */
  slow?: boolean
  slow_after?: number
  note: string
  result?: MatResult
}

export interface MethodInfo {
  method: string
  continuous: boolean
  disabled: boolean
  needs_priors: boolean
  reason?: string | null
}

export interface RunInfo {
  run_id: number
  method: string
  constrained: boolean
  alpha: number | null
  priors: string | null
  seed: number
  timestamp: number
  columns: string[]
  n_edges: number
  topological_validity: number
}

export interface LedgerEdge {
  source: string
  target: string
  frequency: number
  reverse_frequency: number
  adjacency_frequency: number
  conflict: boolean
  methods: string[]
  relation_label: string
  topologically_valid: boolean
  manual: boolean
  selected: boolean
  /** Selecting this edge would close a cycle with what is already selected (W19). */
  blocked: boolean
  blocked_path: string[] | null
}

export interface Ledger {
  columns: string[]
  n_runs: number
  n_selected?: number
  edges: LedgerEdge[]
}

export interface CuratedGraph {
  columns: string[]
  nodes: string[]
  edges: {
    source: string
    target: string
    manual: boolean
    relation_label: string
    topologically_valid: boolean | null
  }[]
  n_edges: number
  n_nodes: number
  acyclic: boolean
  topological_validity: number | null
}

export interface CtxInfo {
  columns: string[]
  dropped_columns: string[]
  n_rows: number
  constraint_stats: ConstraintStats
}

/* ---------------------------------------------------------------- *
 * Module 2 — GES-Prior metadata (W15)
 * ---------------------------------------------------------------- */
export interface PriorVariable {
  column: string
  /** rdfs:label / rdfs:comment from the T-Box, free and always present. */
  schema_text: string
  /** What the user wrote; wins over everything else. */
  user_text: string
  /** QID (or IRI) on the external endpoint this column maps onto. */
  entity_id: string
}

export interface PriorsInfo {
  columns: string[]
  domain: string
  domain_default: string
  endpoint: string
  llm_model: string
  variables: PriorVariable[]
  n_pairs: number
  max_pairs: number
  llm_configured: boolean
  llm_key_env: string | null
  loaded: boolean
  source: 'llm' | 'imported' | null
  n_informative: number
  elapsed?: number
  kg_errors?: string[]
  n_columns_matched?: number
}

/* ---------------------------------------------------------------- *
 * Modules 3-5
 * ---------------------------------------------------------------- */
export type DType = 'categorical' | 'ordinal' | 'discrete' | 'continuous'

export interface Mechanism {
  node: string
  dtype: DType
  is_root: boolean
  mechanism_type: string
  invertible: boolean
  parents: string[]
  n_levels: number | null
}

export type ModelKind = 'scm' | 'cbn'

export interface ModelInfo {
  fitted?: boolean
  model_id: string
  kind: ModelKind
  kind_label: string
  supports_counterfactual: boolean
  fitted_at: number
  /** Rows in the *live* frame — 0 for a model imported without its source KG. */
  n_rows: number
  /** Rows it was actually fitted on; recorded in the manifest, so it survives an export. */
  training_rows: number | null
  has_rows: boolean
  /** W29 — the seed the mechanisms were spawned from; `null` when the fit was unseeded. */
  random_state: number | null
  columns: string[]
  dtypes: Record<string, DType>
  alignment: { matched: string[]; missing_in_kg: string[]; extra_in_kg: string[] }
  edges: { source: string; target: string }[]
  all_discrete: boolean
  mechanisms: Mechanism[]
  roots: string[]
  non_invertible: string[]
  levels: Record<string, string[]>
  ranges: Record<string, { min: number | null; max: number | null; mean: number | null }>
  elapsed?: number
}

export interface CvRow {
  node: string
  metric: 'accuracy' | 'r2'
  model: number
  baseline: number
  beats_baseline: boolean
  macro_f1: number | null
}

export interface Evaluation {
  model_kind: ModelKind
  cv: CvRow[]
  cv_error: string | null
  n_failing: number
  n_scored: number
  summary: {
    nodes_beating_baseline: number
    nodes_scored: number
    mean_lift: number | null
    worst_node: string | null
  }
  structure: {
    n_nodes: number
    n_edges: number
    acyclic: boolean
    topological_validity: number | null
    falsification: { summary: string; rejected: boolean | null } | null
  }
  calibration: {
    log_likelihood?: number
    n_parameters?: number
    bic?: number
    per_row_log_likelihood?: number
    error?: string
  } | null
  abduction: {
    supported: boolean
    non_invertible: string[]
    pnl_assumptions: Record<string, unknown> | null
  }
  gcm_summary: string | null
  pnl_assumptions: Record<string, unknown> | null
  elapsed?: number
}

/* ---------------------------------------------------------------- *
 * The board (W23) — every node's distribution, not one target's
 * ---------------------------------------------------------------- */
export interface NodeDistribution {
  kind: 'categorical' | 'numeric'
  probs: Record<string, number> | null
  predicted: string | number | null
  mean: number | null
  std: number | null
  ci: number[] | null
  x: number[] | null
  y: number[] | null
  /** The answer is a single value, not a spread — drawn as a spike, labelled `=` not `μ`. */
  degenerate?: boolean
  min?: number
  max?: number
  mode?: 'observed' | 'intervened'
}

export interface Marginals {
  n_rows: number
  nodes: Record<string, NodeDistribution>
}

export interface BoardPrediction {
  answer_id: number
  estimand: string
  kind: 'conditional' | 'interventional'
  model_kind: ModelKind
  backend: string
  n_samples: number | null
  ess: number | null
  low_confidence: boolean
  elapsed: number
  n_observed: number
  n_intervened: number
  evidence: Record<string, unknown>
  interventions: Record<string, unknown>
  nodes: Record<string, NodeDistribution>
}

export interface CfPrediction {
  answer_id: number
  /** `null` for a hypothetical unit (W28) — it describes nobody in the KG. */
  entity: string | null
  hypothetical: boolean
  estimand: string
  backend: string
  coupling: string | null
  n_samples: number
  n_rows: number
  elapsed: number
  interventions: Record<string, unknown>
  factual: Record<string, string | number | null>
  nodes: Record<string, NodeDistribution>
}

export interface MechanismDetail {
  node: string
  dtype: DType
  kind: ModelKind
  parents: string[]
  children: string[]
  is_root: boolean
  mechanism_type: string | null
  invertible: boolean
  levels: string[] | null
  range: { min: number | null; max: number | null; mean: number | null } | null
  cpt: {
    variable: string
    states: string[]
    evidence: string[]
    columns: string[]
    values: number[][]
    truncated: boolean
    n_columns: number
  } | null
  noise: { description: string; n_unique: number } | null
}

export interface UnitClass {
  var: string
  n_total: number
  n_matching: number
  entities: string[]
  truncated: boolean
}

export interface UnitSearchResult {
  classes: UnitClass[]
  n_matching_rows: number
  n_total_rows: number
}

export interface Band {
  index: number
  lo: number
  hi: number
  label: string
}

export type AnswerKind = 'conditional' | 'interventional' | 'counterfactual'

export interface AnswerInfo {
  answer_id: number
  at: number
  kind: AnswerKind
  estimand: string
  target: string
  entity: string | null
  predicted: string | number | null
  distribution: Record<string, number> | null
  mean: number | null
  std: number | null
  ci: number[] | null
  factual: string | number | null
  effect: Record<string, number> | number | null
  backend: string | null
  n_samples: number | null
  ess: number | null
  low_confidence: boolean
  coupling: string | null
  evidence: Record<string, unknown>
  conditions: Record<string, unknown>
  interventions: Record<string, unknown>
  reference?: Record<string, unknown>
}

export interface EntityClass {
  var: string
  n_total: number
  entities: string[]
}

export type CanvasId = 'ontology' | 'causal' | 'model' | 'inference' | 'counterfactual'
export type Positions = Record<string, { x: number; y: number }>
export type Layouts = Record<CanvasId, Positions>

export interface ExportArtifact {
  what: string
  label: string
  formats: string[]
  available: boolean
  reason: string
  /** Present when the artifact is a *set* the user picks from — the discovery
   *  runs, for `learned_graphs`. Empty/absent means "export everything". */
  select?: { id: number; label: string }[]
  /** Query parameter the chosen ids go into, e.g. `runs`. */
  select_param?: string
}

/** The export menu, grouped by the module that produces each artifact. */
export interface ExportManifest {
  groups: { id: string; label: string; items: ExportArtifact[] }[]
  items: ExportArtifact[]
}

export class ApiError extends Error {
  detail: unknown
  status: number
  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

/**
 * POSTs that are safe to send twice, because the server either dedupes them or
 * recomputes the same thing. `model/fit` is keyed on the curation + kind + options and
 * returns the cached model unchanged for a repeat; `model/evaluate` overwrites its own
 * result. Nothing else is on this list — re-sending `discovery/run` would append a
 * second run and re-sending `materialise` would 409.
 */
const RETRY_SAFE_POSTS = [/\/model\/fit$/, /\/model\/evaluate$/]

/**
 * A `fetch` that failed before any response existed — not an HTTP error, a dead socket.
 *
 * uvicorn closes an idle keep-alive connection after 5 s (`--timeout-keep-alive`), and a
 * browser that reuses a pooled socket at exactly the moment the server closes it gets a
 * bare network `TypeError`: WebKit calls it "Load failed", Chromium "Failed to fetch".
 * Browsers retry an idempotent GET themselves but never a POST, which is why this
 * surfaced as "click Retrain, get Load failed" on a button whose request had not even
 * reached the server. One retry is therefore not a second fit — it is the first one.
 */
function isNetworkFailure(e: unknown): boolean {
  return e instanceof TypeError
}

function connectionError(e: unknown, retried: boolean): ApiError {
  const raw = e instanceof Error ? e.message : String(e)
  return new ApiError(
    `The connection to the API dropped before the server answered (${raw})` +
      (retried ? ', and again on one retry' : '') +
      '. Check that the service is still running on this port, then try again.',
    0,
    null,
  )
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase()
  const retryable = method === 'GET' || RETRY_SAFE_POSTS.some((re) => re.test(path))
  let res: Response
  try {
    res = await fetch(path, init)
  } catch (e) {
    if (!isNetworkFailure(e) || !retryable) throw connectionError(e, false)
    await new Promise((r) => setTimeout(r, 200))
    try {
      res = await fetch(path, init)
    } catch (again) {
      throw connectionError(again, true)
    }
  }
  const text = await res.text()
  let body: any = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = text
  }
  if (!res.ok) {
    const detail = body?.detail ?? body
    // FastAPI's 422 for a cycle carries a structured detail (Plan 3 §4.4); keep the
    // whole object on the error so the caller can name the cycle rather than print JSON.
    const message =
      typeof detail === 'string'
        ? detail
        : detail?.message ?? detail?.error ?? res.statusText
    throw new ApiError(message, res.status, detail)
  }
  return body as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

const put = (body: unknown): RequestInit => ({ ...json(body), method: 'PUT' })

export const api = {
  createProject: (name = 'Untitled project') =>
    req<{ id: string; name: string }>('/api/projects', json({ name })),

  samples: () => req<{ sample_id: string; exists: boolean }[]>('/api/samples'),
  methods: () => req<MethodInfo[]>('/api/discovery/methods'),

  loadSample: (pid: string, sample_id: string) =>
    req<SchemaPayload>(`/api/projects/${pid}/sources/sample`, json({ sample_id })),

  uploadFile: (pid: string, file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<SchemaPayload>(`/api/projects/${pid}/sources/upload`, { method: 'POST', body: fd })
  },

  loadEndpoint: (pid: string, url: string, infer_missing: boolean | null) =>
    req<SchemaPayload>(`/api/projects/${pid}/sources/endpoint`, json({ url, infer_missing })),

  /* --- W29: bundles ------------------------------------------------- *
   * `importBundle` takes no project id on purpose — the server always
   * creates a new one. There is no undo here and no second copy, so an
   * import that could target an open project would be one mis-click from
   * destroying an afternoon's work.
   */
  importBundle: (file: File, strict = true) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<ImportResult>(`/api/projects/import?strict=${strict}`, {
      method: 'POST',
      body: fd,
    })
  },

  /** Attach the source KG an imported project was exported from. 409 = wrong file. */
  reattachSource: (pid: string, file: File, force = false) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<{ ok: true; note: string; capabilities: Capabilities }>(
      `/api/projects/${pid}/sources/reattach?force=${force}`,
      { method: 'POST', body: fd },
    )
  },

  capabilities: (pid: string) => req<Capabilities>(`/api/projects/${pid}/capabilities`),

  nodes: (pid: string) => req<NodesPayload>(`/api/projects/${pid}/nodes`),

  saveNodes: (
    pid: string,
    excluded: string[],
    excludedJoins: string[],
    c: ConstraintOptions,
    o?: {
      max_levels?: number
      float_as_continuous?: boolean
      /**
       * W26. Sent as roles rather than folded into the two sets here: the
       * server owns the resolution rule (`bgp.object_property_role`), and a
       * client that encoded it itself would be a second copy to keep in step.
       */
      object_roles?: Record<string, ObjectRole>
    },
  ) =>
    req<NodesPayload>(
      `/api/projects/${pid}/nodes`,
      put({ excluded, excluded_joins: excludedJoins, ...c, ...o }),
    ),

  /** Starts the join and returns immediately; poll `matStatus` (W21). */
  materialise: (pid: string, component: number | null, limit: number | null) =>
    req<MatJob>(`/api/projects/${pid}/materialise`, json({ component, limit })),
  matStatus: (pid: string) => req<MatJob>(`/api/projects/${pid}/materialise/status`),
  matCancel: (pid: string) =>
    req<MatJob>(`/api/projects/${pid}/materialise/cancel`, { method: 'POST' }),
  /** Cheap — no SPARQL execution — so it can be called on every curation or component change. */
  materialisePreview: (pid: string, limit: number | null, component: number | null) =>
    req<GraphPatternPreview>(
      `/api/projects/${pid}/materialise/preview?` +
        [
          limit === null ? '' : `limit=${limit}`,
          component === null ? '' : `component=${component}`,
        ]
          .filter(Boolean)
          .join('&'),
    ),

  context: (pid: string, component: number | null, limit: number | null) =>
    req<CtxInfo>(
      `/api/projects/${pid}/discovery/context` +
        (limit === null ? '?' : `?limit=${limit}&`) +
        (component === null ? '' : `component=${component}`),
    ),

  runDiscovery: (
    pid: string,
    body: {
      method: string
      constrained: boolean
      alpha: number | null
      component: number | null
      limit: number | null
      seed: number | null
    },
  ) => req<RunInfo>(`/api/projects/${pid}/discovery/run`, json(body)),

  runs: (pid: string) => req<RunInfo[]>(`/api/projects/${pid}/discovery/runs`),
  deleteRun: (pid: string, id: number) =>
    req<{ ok: true }>(`/api/projects/${pid}/discovery/runs/${id}`, { method: 'DELETE' }),
  clearRuns: (pid: string) =>
    req<{ ok: true; cleared: number }>(`/api/projects/${pid}/discovery/runs`, { method: 'DELETE' }),

  ledger: (pid: string) => req<Ledger>(`/api/projects/${pid}/discovery/ledger`),

  addEdge: (pid: string, source: string, target: string) =>
    req<Ledger>(`/api/projects/${pid}/graph/edges`, json({ source, target })),
  removeEdge: (pid: string, source: string, target: string) =>
    req<Ledger>(
      `/api/projects/${pid}/graph/edges?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}`,
      { method: 'DELETE' },
    ),
  clearManualEdges: (pid: string) =>
    req<{ ok: true; cleared: number; ledger: Ledger }>(`/api/projects/${pid}/graph/edges/all`, {
      method: 'DELETE',
    }),
  setSelection: (pid: string, edges: [string, string][]) =>
    req<Ledger>(`/api/projects/${pid}/graph/selection`, put({ edges })),
  autoSelect: (pid: string, min_frequency = 1, top_k: number | null = null) =>
    req<{
      ledger: Ledger
      n_selected: number
      skipped: { source: string; target: string; frequency: number; path: string[] }[]
      n_candidates: number
      cut_off: number
      min_kept_frequency: number | null
    }>(`/api/projects/${pid}/graph/selection/auto`, json({ min_frequency, top_k })),

  graph: (pid: string) => req<CuratedGraph>(`/api/projects/${pid}/graph`),

  /* --- priors (W15) --- */
  priors: (pid: string) => req<PriorsInfo>(`/api/projects/${pid}/priors`),
  savePriorMeta: (
    pid: string,
    body: {
      domain?: string
      var_text?: Record<string, string>
      entity_map?: Record<string, string>
      endpoint?: string
      llm_model?: string
    },
  ) => req<PriorsInfo>(`/api/projects/${pid}/priors/meta`, put(body)),
  estimatePriors: (pid: string, body: { use_kg: boolean; llm_model?: string }) =>
    req<PriorsInfo>(`/api/projects/${pid}/priors/estimate`, json(body)),
  importPriors: (pid: string, file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<PriorsInfo>(`/api/projects/${pid}/priors/import`, { method: 'POST', body: fd })
  },
  clearPriors: (pid: string) =>
    req<PriorsInfo>(`/api/projects/${pid}/priors`, { method: 'DELETE' }),

  /* --- the causal model (W22) --- */
  fitModel: (pid: string, body: { kind: ModelKind; ordinal: string[] }) =>
    req<ModelInfo>(`/api/projects/${pid}/model/fit`, json(body)),
  model: (pid: string) => req<ModelInfo & { fitted: boolean }>(`/api/projects/${pid}/model`),
  dropModel: (pid: string) =>
    req<{ ok: true }>(`/api/projects/${pid}/model`, { method: 'DELETE' }),
  evaluate: (pid: string, body: { cv_splits: number; gcm: boolean; falsify: boolean }) =>
    req<Evaluation>(`/api/projects/${pid}/model/evaluate`, json(body)),
  mechanism: (pid: string, node: string) =>
    req<MechanismDetail>(`/api/projects/${pid}/model/mechanism/${encodeURIComponent(node)}`),

  /* --- the board (W23) --- */
  marginals: (pid: string) => req<Marginals>(`/api/projects/${pid}/model/marginals`),
  predict: (
    pid: string,
    body: {
      evidence: Record<string, string | number>
      interventions: Record<string, string | number>
      num_samples?: number
    },
  ) => req<BoardPrediction>(`/api/projects/${pid}/infer/predict`, json(body)),
  /**
   * W28 — exactly one of `entity` (a unit from the source KG) or `observed`
   * (every variable's value, typed by the user). Both would be two different
   * factual states for one counterfactual; neither leaves nothing to abduct
   * from, and the server refuses either way.
   */
  predictCounterfactual: (
    pid: string,
    body: {
      entity?: string
      observed?: Record<string, string | number>
      interventions: Record<string, string | number>
      num_samples?: number
    },
  ) => req<CfPrediction>(`/api/projects/${pid}/infer/counterfactual/predict`, json(body)),

  /* --- the unit (W24) --- */
  searchUnits: (pid: string, filters: Record<string, string>, limit = 200) =>
    req<UnitSearchResult>(`/api/projects/${pid}/units/search`, json({ filters, limit })),
  unitFactual: (pid: string, entity: string) =>
    req<{ entity: string; n_rows: number; values: Record<string, string | number | null> }>(
      `/api/projects/${pid}/units/factual?entity=${encodeURIComponent(entity)}`,
    ),
  unitBands: (pid: string) => req<Record<string, Band[]>>(`/api/projects/${pid}/units/bands`),

  /* --- modules 4-5 ---
   *
   * These three take 1..n targets, and the request shape decides the response shape:
   * `target: string` answers one `AnswerInfo`, `target: string[]` answers
   * `{ answers: AnswerInfo[] }` — one per target, off a single shared computation
   * server-side, so a batch cannot disagree with itself. The overloads below keep that
   * in the type system rather than in a comment nobody reads. The board (modules 3-5)
   * goes through `predict` / `predictCounterfactual` instead; these stay the
   * single-question surface. */
  condition: <T extends string | string[]>(
    pid: string,
    body: {
      target: T
      evidence: Record<string, unknown>
      conditions: Record<string, unknown>
      backend: string
    },
  ) =>
    req<T extends string ? AnswerInfo : { answers: AnswerInfo[] }>(
      `/api/projects/${pid}/infer/condition`,
      json(body),
    ),
  intervene: <T extends string | string[]>(
    pid: string,
    body: {
      target: T
      interventions: Record<string, unknown>
      reference: Record<string, unknown>
      conditions: Record<string, unknown>
    },
  ) =>
    req<T extends string ? AnswerInfo : { answers: AnswerInfo[] }>(
      `/api/projects/${pid}/infer/intervene`,
      json(body),
    ),
  counterfactual: <T extends string | string[]>(
    pid: string,
    body: {
      target: T
      interventions: Record<string, unknown>
      entity: string
      num_samples: number
    },
  ) =>
    req<T extends string ? AnswerInfo : { answers: AnswerInfo[] }>(
      `/api/projects/${pid}/infer/counterfactual`,
      json(body),
    ),
  entities: (pid: string, q = '', limit = 200) =>
    req<{ classes: EntityClass[] }>(
      `/api/projects/${pid}/entities?limit=${limit}&q=${encodeURIComponent(q)}`,
    ),
  answers: (pid: string) => req<AnswerInfo[]>(`/api/projects/${pid}/answers`),
  clearAnswers: (pid: string) =>
    req<{ ok: true; cleared: number }>(`/api/projects/${pid}/answers`, { method: 'DELETE' }),

  layout: (pid: string) => req<Layouts>(`/api/projects/${pid}/layout`),
  saveLayout: (pid: string, canvas: CanvasId, positions: Positions) =>
    req<{ ok: true }>(`/api/projects/${pid}/layout`, put({ canvas, positions })),
  resetLayout: (pid: string, canvas: CanvasId) =>
    req<{ ok: true }>(`/api/projects/${pid}/layout/${canvas}`, { method: 'DELETE' }),

  exportManifest: (pid: string) => req<ExportManifest>(`/api/projects/${pid}/export/manifest`),
  exportUrl: (pid: string, what: string, format: string, select?: {
    param: string
    ids: number[]
  }) => {
    const base = `/api/projects/${pid}/export?what=${what}&format=${format}`
    return select && select.ids.length
      ? `${base}&${select.param}=${select.ids.join(',')}`
      : base
  },

  /**
   * One logged answer, rather than the whole session log. `ttl` renders the
   * query, its interventions, the observational conditions that scope it and
   * the estimate as `cw:` RDF; it needs a fitted model, because a node IRI
   * cannot be resolved without one.
   */
  answerExportUrl: (pid: string, aid: number, format: 'json' | 'csv' | 'ttl') =>
    `/api/projects/${pid}/answers/${aid}/export?format=${format}`,
}
