import { create } from 'zustand'
import {
  api,
  ApiError,
  type AnswerInfo,
  type Band,
  type BoardPrediction,
  type CanvasId,
  type CfPrediction,
  type ComponentInfo,
  type ConstraintOptions,
  type ConstraintStats,
  type CtxInfo,
  type CuratedGraph,
  type DType,
  type Evaluation,
  type GraphPatternPreview,
  type Layouts,
  type Ledger,
  type Marginals,
  type MatJob,
  type MatResult,
  type MechanismDetail,
  type MethodInfo,
  type ModelInfo,
  type ModelKind,
  type Capabilities,
  type NodeInfo,
  type ObjectRole,
  type Positions,
  type PriorsInfo,
  type RunInfo,
  type SchemaPayload,
  type TypeOptions,
  type UnitSearchResult,
} from '../api/client'

/**
 * Four modules, not five. Fitting a model is not a destination — it is the thing you
 * do in order to ask a question, and it was its own step only because the fit panel
 * had nowhere else to live. It now lives in the Inference rail beside the board it
 * feeds, which is also where re-fitting belongs: you discover you need a different
 * model *while* querying, never before.
 */
export type ModuleId = 'preprocess' | 'discovery' | 'inference' | 'counterfactual'

/** A node's role on the query board. */
export type Assignment = { mode: 'observed' | 'intervened'; value: string | number }
export type Assignments = Record<string, Assignment>

export interface Toast {
  kind: 'ok' | 'warn' | 'err'
  text: string
  id: number
}

interface State {
  // --- project ------------------------------------------------------- //
  projectId: string | null
  /**
   * W29 — which panels can do anything, and why not when they cannot. `null`
   * for an ordinary project, where everything is reachable by working through
   * the modules in order; non-null once a bundle is imported, because that
   * project has a fitted model and no source knowledge graph, a state no panel
   * had to cope with before.
   */
  capabilities: Capabilities | null
  importNotes: string[]
  schema: SchemaPayload | null
  nodes: NodeInfo[]
  stats: ConstraintStats | null
  constraint: ConstraintOptions
  typeOptions: TypeOptions
  dtypes: Record<string, DType>
  components: ComponentInfo[]
  /**
   * Which connected component the join runs over (W31). `null` == auto, i.e.
   * `autoComponent` — whichever holds the most retained nodes, re-decided by the
   * server on every curation change. Setting a number pins it, the same way a
   * row limit pins "no limit".
   */
  component: number | null
  /** What the server would pick for `component: null` at this curation. */
  autoComponent: number | null
  /** null == no SPARQL LIMIT. The default, and the whole point of W21. */
  limit: number | null
  mat: MatResult | null
  matJob: MatJob | null
  /** The SPARQL `materialise` would run right now, kept live so the user can check the
   *  join's shape before running it — never touches the join itself (Preprocess revision 3). */
  graphPattern: GraphPatternPreview | null
  /** Why there is no graph pattern — the server's own reason, not a blank box (W31). */
  graphPatternError: string | null
  layouts: Layouts

  // --- discovery ----------------------------------------------------- //
  methods: MethodInfo[]
  ctx: CtxInfo | null
  runs: RunInfo[]
  ledger: Ledger | null
  graph: CuratedGraph | null
  priors: PriorsInfo | null
  samples: { sample_id: string; exists: boolean }[]

  // --- the model and the board (modules 3-4) ------------------------- //
  model: ModelInfo | null
  evaluation: Evaluation | null
  answers: AnswerInfo[]
  /** The observational marginals every card draws behind its posterior. */
  marginals: Marginals | null
  /** The last Predict. Null means the board is showing priors. */
  prediction: BoardPrediction | null
  /** Module 4's last counterfactual, and the unit it was about. */
  cfPrediction: CfPrediction | null
  assignments: Assignments
  /** Module 4's hypothetical do()s — separate from `assignments`, which module 3 owns. */
  hypotheticals: Assignments
  selectedNode: string | null
  mechanism: MechanismDetail | null
  // --- the unit (module 4) ------------------------------------------- //
  unitFilters: Record<string, string>
  units: UnitSearchResult | null
  bands: Record<string, Band[]>
  entity: string | null
  factual: Record<string, string | number | null> | null
  /** How many flat-join rows the chosen unit occupies — 1 for a subject, N for a class
   *  the join fans out over. Above 1 the "factual" value is an aggregate (W24). */
  factualRows: number

  // --- ui ------------------------------------------------------------ //
  module: ModuleId
  leftOpen: boolean
  rightOpen: boolean
  /** Rail widths in px — dragged by the user, remembered across reloads. */
  railWidth: { left: number; right: number }
  /**
   * Height overrides for the centre panels, keyed by panel id. Absent means "fill the
   * space the module has left", which is the default and the reason a module never
   * leaves a gap at the bottom of the window.
   */
  panelHeight: Record<string, number>
  theme: 'light' | 'dark'
  busy: string | null
  toast: Toast | null
  cycle: string[] | null

  // --- actions ------------------------------------------------------- //
  init: () => Promise<void>
  newProject: () => Promise<void>
  /**
   * W29 — restore a project from an exported bundle. Always lands in a *new*
   * project (the server enforces it), so this replaces the session the way
   * `newProject` does rather than merging into it.
   */
  importBundle: (f: File, strict?: boolean) => Promise<void>
  /** Attach the source KG an imported project was exported from. */
  reattachSource: (f: File, force?: boolean) => Promise<void>
  refreshCapabilities: () => Promise<void>
  notify: (kind: Toast['kind'], text: string) => void
  dismiss: () => void
  setModule: (m: ModuleId) => void
  toggleLeft: () => void
  toggleRight: () => void
  setRailWidth: (side: 'left' | 'right', px: number) => void
  /** `null` hands the panel back to the flex layout, so it fills the window again. */
  setPanelHeight: (id: string, px: number | null) => void
  toggleTheme: () => void
  setLimit: (n: number | null) => void
  /** W31 — pin the component the join runs over, or `null` to hand it back to auto. */
  setComponent: (i: number | null) => Promise<void>
  /** Refetches the live "Graph Pattern" preview for the current curation state. */
  refreshGraphPattern: () => Promise<void>

  loadSample: (id: string) => Promise<void>
  uploadFile: (f: File) => Promise<void>
  loadEndpoint: (url: string) => Promise<void>
  refreshNodes: () => Promise<void>
  toggleNode: (name: string) => Promise<void>
  /** Object properties only: toggle whether the relationship is joined. */
  toggleJoin: (name: string) => Promise<void>
  /** W26 — set an object property's role. Rejected by the server (400) when
   *  `variable` would leave the query computing a cross product. */
  setObjectRole: (name: string, role: ObjectRole) => Promise<void>
  setExclusions: (names: string[], excluded: boolean) => Promise<void>
  setConstraint: (patch: Partial<ConstraintOptions>) => Promise<void>
  setTypeOptions: (patch: Partial<Pick<TypeOptions, 'max_levels' | 'float_as_continuous'>>) => Promise<void>
  materialise: (limit?: number | null) => Promise<void>
  cancelMaterialise: () => Promise<void>

  ensureContext: () => Promise<void>
  runDiscovery: (body: {
    method: string
    constrained: boolean
    alpha: number | null
    seed: number | null
  }) => Promise<void>
  deleteRun: (id: number) => Promise<void>
  clearRuns: () => Promise<void>
  clearManualEdges: () => Promise<void>
  addEdge: (s: string, t: string) => Promise<void>
  removeEdge: (s: string, t: string) => Promise<void>
  toggleSelected: (s: string, t: string) => Promise<void>
  selectAll: (which: 'valid' | 'none') => Promise<void>
  selectTopK: (k: number) => Promise<void>

  loadPriors: () => Promise<void>
  savePriorMeta: (body: {
    domain?: string
    var_text?: Record<string, string>
    entity_map?: Record<string, string>
    endpoint?: string
    llm_model?: string
  }) => Promise<void>
  estimatePriors: (useKg: boolean) => Promise<void>
  importPriors: (f: File) => Promise<void>
  clearPriors: () => Promise<void>

  loadModel: () => Promise<void>
  fitModel: (body: { kind: ModelKind; ordinal: string[] }) => Promise<void>
  dropModel: () => Promise<void>
  evaluate: (body: { cv_splits: number; gcm: boolean; falsify: boolean }) => Promise<void>
  clearAnswers: () => Promise<void>

  // --- the board ------------------------------------------------------ //
  assign: (node: string, mode: 'observed' | 'intervened', value: string | number) => void
  release: (node: string) => void
  clearBoard: () => void
  predict: () => Promise<void>
  selectNode: (node: string | null) => Promise<void>

  // --- module 4 ------------------------------------------------------- //
  setUnitFilter: (node: string, value: string) => Promise<void>
  clearUnitFilters: () => Promise<void>
  searchUnits: () => Promise<void>
  chooseEntity: (iri: string | null) => Promise<void>
  /**
   * W28 — the unit the user typed instead of picking. `null` means "use an
   * entity from the KG"; a record means "this hypothetical unit", and the two
   * are mutually exclusive by construction (setting one clears the other),
   * because a counterfactual has exactly one factual state.
   */
  observed: Record<string, string | number> | null
  setObserved: (values: Record<string, string | number> | null) => void
  setObservedValue: (node: string, value: string | number) => void
  setHypothetical: (node: string, value: string | number) => void
  releaseHypothetical: (node: string) => void
  clearHypotheticals: () => void
  predictCounterfactual: () => Promise<void>

  saveLayout: (canvas: CanvasId, positions: Positions) => void
  resetLayout: (canvas: CanvasId) => Promise<void>
}

/** Layout writes are chatty (one per drag) and cheap to lose; coalesce them. */
const layoutTimers: Record<string, number> = {}

/**
 * Pane geometry is *browser* state, unlike node positions (which are server state,
 * W13). A node's place in a causal graph is part of the artifact and has to travel with
 * the project; how wide this person likes their rails is about this screen, so it lives
 * in localStorage and never touches the API.
 */
export const RAIL_MIN = 210
export const RAIL_MAX = 620
const UI_KEY = 'cw-panes'

function loadPanes(): { railWidth: { left: number; right: number }; panelHeight: Record<string, number> } {
  const fallback = { railWidth: { left: 288, right: 320 }, panelHeight: {} }
  try {
    const raw = localStorage.getItem(UI_KEY)
    if (!raw) return fallback
    const p = JSON.parse(raw)
    return {
      railWidth: {
        left: clampRail(p?.railWidth?.left ?? fallback.railWidth.left),
        right: clampRail(p?.railWidth?.right ?? fallback.railWidth.right),
      },
      panelHeight: p?.panelHeight && typeof p.panelHeight === 'object' ? p.panelHeight : {},
    }
  } catch {
    return fallback
  }
}

function clampRail(px: number) {
  return Math.max(RAIL_MIN, Math.min(RAIL_MAX, Math.round(px)))
}

function savePanes(railWidth: { left: number; right: number }, panelHeight: Record<string, number>) {
  try {
    localStorage.setItem(UI_KEY, JSON.stringify({ railWidth, panelHeight }))
  } catch {
    /* private browsing, or a full quota — the panes just stop being remembered */
  }
}

let toastId = 0

export const useStore = create<State>((set, get) => {
  async function guard<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    set({ busy: label })
    try {
      return await fn()
    } catch (e) {
      const err = e as ApiError
      if (err instanceof ApiError && (err.detail as any)?.error === 'cycle') {
        set({ cycle: (err.detail as any).path })
      }
      get().notify('err', err.message || String(e))
      return undefined
    } finally {
      set({ busy: null })
    }
  }

  /** Everything downstream of a fitted model, cleared in one place. */
  const MODEL_RESET = {
    model: null, evaluation: null, answers: [], marginals: null,
    prediction: null, cfPrediction: null, assignments: {}, hypotheticals: {},
    selectedNode: null, mechanism: null, unitFilters: {}, units: null, bands: {},
    entity: null, factual: null, factualRows: 0,
  }

  async function afterSource(payload: SchemaPayload) {
    set({
      schema: payload, mat: null, matJob: null, ctx: null, runs: [], ledger: null,
      graph: null, priors: null, graphPattern: null, graphPatternError: null,
      // A component index only means anything against the schema it came from,
      // so a new source hands the choice back to auto rather than keeping a
      // number that now points at an unrelated part of a different ontology.
      component: null, ...MODEL_RESET,
    })
    await get().refreshNodes()
    const layouts = await api.layout(get().projectId!)
    set({ layouts })
    await get().refreshGraphPattern()
  }

  function applyNodes(data: Awaited<ReturnType<typeof api.nodes>>) {
    set({
      nodes: data.nodes,
      stats: data.stats,
      constraint: data.constraint_options,
      typeOptions: data.type_options,
      dtypes: data.dtypes,
      components: data.components,
      autoComponent: data.auto_component,
    })
  }

  async function pushNodes(
    excluded: string[],
    excludedJoins: string[],
    constraint: ConstraintOptions,
  ) {
    const pid = get().projectId!
    const data = await api.saveNodes(pid, excluded, excludedJoins, constraint)
    applyNodes(data)
    // A node, constraint or encoding change rebuilds the discovery context, so the
    // ledger it indexed is gone, and so is anything fitted to it (§4.3 cascade).
    set({
      ctx: null, runs: [], ledger: null, graph: null, mat: null, matJob: null,
      priors: null, ...MODEL_RESET,
    })
    await get().refreshGraphPattern()
  }

  async function refreshLedger(ledger?: Ledger) {
    const pid = get().projectId!
    const l = ledger ?? (await api.ledger(pid))
    const graph = await api.graph(pid)
    set({ ledger: l, graph, cycle: null })
  }

  return {
    projectId: null,
    capabilities: null,
    importNotes: [],
    observed: null,
    schema: null,
    nodes: [],
    stats: null,
    constraint: { relation_direction: 'both', allow_epsilon: true, max_hops: 1 },
    typeOptions: {
      max_levels: 6,
      float_as_continuous: true,
      default_max_levels: 6,
      note: '',
    },
    dtypes: {},
    components: [],
    component: null,
    autoComponent: null,
    limit: null,
    mat: null,
    matJob: null,
    graphPattern: null,
    graphPatternError: null,
    layouts: { ontology: {}, causal: {}, model: {}, inference: {}, counterfactual: {} },

    methods: [],
    ctx: null,
    runs: [],
    ledger: null,
    graph: null,
    priors: null,
    samples: [],

    model: null,
    evaluation: null,
    answers: [],
    marginals: null,
    prediction: null,
    cfPrediction: null,
    assignments: {},
    hypotheticals: {},
    selectedNode: null,
    mechanism: null,
    unitFilters: {},
    units: null,
    bands: {},
    entity: null,
    factual: null,
    factualRows: 0,

    module: 'preprocess',
    leftOpen: true,
    rightOpen: true,
    ...loadPanes(),
    theme: document.documentElement.classList.contains('dark') ? 'dark' : 'light',
    busy: null,
    toast: null,
    cycle: null,

    notify: (kind, text) => set({ toast: { kind, text, id: ++toastId } }),
    dismiss: () => set({ toast: null }),
    setModule: (module) => set({ module }),
    toggleLeft: () => set((s) => ({ leftOpen: !s.leftOpen })),
    toggleRight: () => set((s) => ({ rightOpen: !s.rightOpen })),

    setRailWidth: (side, px) =>
      set((s) => {
        const railWidth = { ...s.railWidth, [side]: clampRail(px) }
        savePanes(railWidth, s.panelHeight)
        return { railWidth }
      }),

    setPanelHeight: (id, px) =>
      set((s) => {
        const panelHeight = { ...s.panelHeight }
        if (px === null) delete panelHeight[id]
        else panelHeight[id] = Math.max(140, Math.round(px))
        savePanes(s.railWidth, panelHeight)
        return { panelHeight }
      }),

    toggleTheme: () =>
      set((s) => {
        const theme = s.theme === 'dark' ? 'light' : 'dark'
        document.documentElement.classList.toggle('dark', theme === 'dark')
        localStorage.setItem('cw-theme', theme)
        return { theme }
      }),
    setLimit: (limit) => {
      set({ limit })
      void get().refreshGraphPattern()
    },

    /*
     * W31 — picking a component is picking a different join, so everything
     * fitted to the old one goes with it. Same cascade as a curation change
     * (`pushNodes`): a frame drawn from component 11 is not a subset of
     * component 3's, it is unrelated data with different columns, and a ledger
     * indexed into the first one cannot be read against the second. The one
     * thing kept is the stored layout, which is per-canvas and per-node, so the
     * cards the user has already arranged stay where they put them.
     */
    setComponent: async (i) => {
      if (get().component === i) return
      set({
        component: i,
        mat: null, matJob: null, ctx: null, runs: [], ledger: null, graph: null,
        priors: null, cycle: null, ...MODEL_RESET,
      })
      await get().refreshGraphPattern()
    },

    init: async () => {
      const [samples, methods] = await Promise.all([api.samples(), api.methods()])
      set({ samples, methods })
      if (!get().projectId) await get().newProject()
    },

    newProject: async () => {
      const p = await api.createProject()
      set({
        projectId: p.id,
        schema: null,
        nodes: [],
        stats: null,
        components: [],
        component: null,
        autoComponent: null,
        limit: null,
        mat: null,
        matJob: null,
        graphPattern: null,
        graphPatternError: null,
        ctx: null,
        runs: [],
        ledger: null,
        graph: null,
        priors: null,
        ...MODEL_RESET,
        layouts: { ontology: {}, causal: {}, model: {}, inference: {}, counterfactual: {} },
        module: 'preprocess',
        cycle: null,
        capabilities: null,
        importNotes: [],
      })
    },

    importBundle: async (file, strict = true) => {
      // Deliberately not wrapped in `guard`: `guard` turns every failure into a
      // toast and returns undefined, and one failure here — "this model was
      // fitted with different libraries" — is a *question*, not a verdict. The
      // caller offers "import anyway", which needs the error to reach it.
      // Everything else still surfaces as the same toast guard would have shown.
      set({ busy: 'Importing bundle…' })
      let r
      try {
        r = await api.importBundle(file, strict)
      } catch (e) {
        const err = e as ApiError
        if (!/different libraries/i.test(err.message ?? '')) {
          get().notify('err', err.message || String(e))
          return
        }
        throw e
      } finally {
        set({ busy: null })
      }
      set({
        projectId: r.id,
        capabilities: r.capabilities,
        importNotes: r.notes,
        // Everything derived from the source KG is genuinely absent, not merely
        // unloaded: a bundle does not carry the KG. Leaving stale values from
        // the previous session here would let module 1 render a schema that has
        // nothing to do with this project.
        schema: null, nodes: [], stats: null, components: [], component: null,
        autoComponent: null, graphPatternError: null,
        mat: null, matJob: null, graphPattern: null, ctx: null, runs: [], ledger: null,
        graph: null, priors: null, cycle: null,
        ...MODEL_RESET,
      })
      await get().loadModel()
      // Land in the module the imported project can actually use. A bundle with
      // a model opens on the counterfactual board; without one there is nothing
      // to do until a source is attached.
      set({ module: r.has_model ? 'counterfactual' : 'preprocess' })
      get().notify('ok', r.has_model
        ? `Imported "${r.name}" with its fitted model.`
        : `Imported "${r.name}". No model in the bundle.`)
    },

    reattachSource: async (file, force = false) => {
      const pid = get().projectId!
      const r = await guard('Attaching source…', () => api.reattachSource(pid, file, force))
      if (!r) return
      set({ capabilities: r.capabilities, importNotes: [r.note] })
      await get().refreshNodes()
      await get().loadModel()
      get().notify('ok', r.note)
    },

    refreshCapabilities: async () => {
      const pid = get().projectId
      if (!pid) return
      try {
        set({ capabilities: await api.capabilities(pid) })
      } catch {
        // Availability is an enhancement, not a gate — an older server without
        // the route leaves every panel enabled, which is how it behaved before.
        set({ capabilities: null })
      }
    },

    loadSample: async (id) => {
      const r = await guard('Parsing sample…', () => api.loadSample(get().projectId!, id))
      if (r) await afterSource(r)
    },
    uploadFile: async (f) => {
      const r = await guard(`Parsing ${f.name}…`, () => api.uploadFile(get().projectId!, f))
      if (r) await afterSource(r)
    },
    loadEndpoint: async (url) => {
      const r = await guard('Resolving endpoint…', () =>
        api.loadEndpoint(get().projectId!, url, null),
      )
      if (r) await afterSource(r)
    },

    refreshNodes: async () => {
      applyNodes(await api.nodes(get().projectId!))
    },

    refreshGraphPattern: async () => {
      const { projectId, schema, limit, component } = get()
      if (!projectId || !schema) {
        set({ graphPattern: null, graphPatternError: null })
        return
      }
      try {
        set({
          graphPattern: await api.materialisePreview(projectId, limit, component),
          graphPatternError: null,
        })
      } catch (e) {
        // Still no toast — a preview failing is not an action failing. But the
        // reason is kept and shown in the box: with a component picker the
        // commonest failure is "component 3 has no retained nodes", and an
        // empty box for that reads as a bug rather than as an answer (W31).
        set({ graphPattern: null, graphPatternError: (e as ApiError).message || String(e) })
      }
    },

    toggleNode: async (name) => {
      const { nodes } = get()
      const next = nodes.some((n) => n.name === name && n.excluded)
        ? nodes.filter((n) => n.excluded && n.name !== name).map((n) => n.name)
        : [...nodes.filter((n) => n.excluded).map((n) => n.name), name]
      const joins = nodes.filter((n) => n.join_excluded).map((n) => n.name)
      await guard('Saving curation…', () => pushNodes(next, joins, get().constraint))
    },

    toggleJoin: async (name) => {
      const { nodes } = get()
      const excluded = nodes.filter((n) => n.excluded).map((n) => n.name)
      const next = nodes.some((n) => n.name === name && n.join_excluded)
        ? nodes.filter((n) => n.join_excluded && n.name !== name).map((n) => n.name)
        : [...nodes.filter((n) => n.join_excluded).map((n) => n.name), name]
      await guard('Saving curation…', () => pushNodes(excluded, next, get().constraint))
    },

    setObjectRole: async (name, role) => {
      // The two sets go up unchanged and `object_roles` overrides this one node.
      // Computing the sets here would mean re-implementing
      // `bgp.object_property_role` in TypeScript, and the two copies would drift
      // the first time either side changed.
      const { nodes, constraint } = get()
      const excluded = nodes.filter((n) => n.excluded).map((n) => n.name)
      const joins = nodes.filter((n) => n.join_excluded).map((n) => n.name)
      const pid = get().projectId!
      await guard('Saving curation…', async () => {
        const data = await api.saveNodes(pid, excluded, joins, constraint, {
          object_roles: { [name]: role },
        })
        applyNodes(data)
        set({
          ctx: null, runs: [], ledger: null, graph: null, mat: null, matJob: null,
          priors: null, ...MODEL_RESET,
        })
        await get().refreshGraphPattern()
      })
    },

    setExclusions: async (names, excluded) => {
      const current = new Set(get().nodes.filter((n) => n.excluded).map((n) => n.name))
      names.forEach((n) => (excluded ? current.add(n) : current.delete(n)))
      const joins = get().nodes.filter((n) => n.join_excluded).map((n) => n.name)
      await guard('Saving curation…', () => pushNodes([...current], joins, get().constraint))
    },

    setConstraint: async (patch) => {
      const constraint = { ...get().constraint, ...patch }
      const excluded = get()
        .nodes.filter((n) => n.excluded)
        .map((n) => n.name)
      const joins = get().nodes.filter((n) => n.join_excluded).map((n) => n.name)
      await guard('Recomputing constraint…', () => pushNodes(excluded, joins, constraint))
    },

    setTypeOptions: async (patch) => {
      const excluded = get().nodes.filter((n) => n.excluded).map((n) => n.name)
      const joins = get().nodes.filter((n) => n.join_excluded).map((n) => n.name)
      // Sent through the same endpoint but deliberately *not* through pushNodes: the
      // threshold does not change the join, so re-running it would throw away minutes
      // of work to change one integer (W20). The server invalidates only the model.
      const data = await guard('Re-typing columns…', () =>
        api.saveNodes(get().projectId!, excluded, joins, get().constraint, patch),
      )
      if (!data) return
      applyNodes(data)
      set({ ...MODEL_RESET })
    },

    /**
     * Start the join and poll it to completion.
     *
     * Unlimited by default: a row limit is a SPARQL LIMIT, the first N solutions in
     * whatever order the store returned them, and silently making that the default
     * meant every number downstream was computed on an arbitrary slice (W21).
     */
    materialise: async (limitOverride) => {
      const { projectId, component } = get()
      // Two renders can reach the auto-run effect before the first request answers.
      // The server rejects the second with a 409, which is correct but shows up as a
      // console error for a race the user never caused — so it is caught here instead.
      if (get().matJob?.state === 'running') return
      const limit = limitOverride === undefined ? get().limit : limitOverride
      if (limitOverride !== undefined) set({ limit: limitOverride })
      set({ matJob: { state: 'running', elapsed: 0, limit, note: '' } })
      let job: MatJob | undefined
      try {
        job = await api.materialise(projectId!, component, limit)
      } catch (e) {
        const err = e as ApiError
        if (err.status === 409) {
          job = await api.matStatus(projectId!).catch(() => undefined)
        } else {
          set({ matJob: null })
          get().notify('err', err.message)
          return
        }
      }
      if (!job) {
        set({ matJob: null })
        return
      }
      set({ matJob: job, busy: null })

      // Polled rather than pushed: this service has no WebSocket channel, and a
      // half-second poll on one job is cheaper than the machinery to avoid it.
      while (job && job.state === 'running') {
        await new Promise((r) => setTimeout(r, 500))
        if (get().projectId !== projectId) return
        try {
          job = await api.matStatus(projectId!)
        } catch {
          return
        }
        set({ matJob: job })
      }
      if (!job) return
      if (job.state === 'done' && job.result) {
        // Pinning the component the join actually ran on, which under W31 is a
        // deliberate narrowing rather than the only way it ever got set. Auto
        // re-decides on every curation change, and module 2 passes `component`
        // to `discovery/context` — leaving it on auto after a run would let a
        // later exclusion move discovery onto a component the frame in `mat`
        // did not come from.
        set({ mat: job.result, component: job.result.component })
        await get().refreshNodes() // picks up distinct counts and dtypes for the cards
        get().notify(
          'ok',
          `${job.result.row_count} rows × ${job.result.column_count} columns in ` +
            `${job.elapsed}s` +
            (job.result.truncated
              ? `, truncated at the ${job.result.limit}-row limit.`
              : ' — the whole join, no limit.'),
        )
      } else if (job.state === 'error') {
        get().notify('err', job.error ?? 'Materialisation failed.')
      } else if (job.state === 'cancelled') {
        get().notify('warn', `Stopped after ${job.elapsed}s. ${job.note}`)
      }
    },

    cancelMaterialise: async () => {
      // `mat` is deliberately left alone: cancelling discards the *in-flight* result,
      // and the server still holds whatever frame was committed before it, so clearing
      // it here would show an empty table for a materialisation that still exists.
      const job = await api.matCancel(get().projectId!).catch(() => null)
      if (job) set({ matJob: job })
    },

    ensureContext: async () => {
      if (get().ctx || !get().schema) return
      const { projectId, component, limit } = get()
      const ctx = await guard('Building discovery context…', () =>
        api.context(projectId!, component, limit),
      )
      if (ctx) {
        set({ ctx })
        await refreshLedger()
        await get().loadPriors()
      }
    },

    runDiscovery: async (body) => {
      const { projectId, component, limit } = get()
      const run = await guard(`Running ${body.method}…`, () =>
        api.runDiscovery(projectId!, { ...body, component, limit }),
      )
      if (!run) return
      const ctx = await api.context(projectId!, component, limit)
      const runs = await api.runs(projectId!)
      set({ ctx, runs })
      await refreshLedger()
      get().notify(
        'ok',
        `${run.method} — ${run.n_edges} edges, topological validity ${(
          run.topological_validity * 100
        ).toFixed(0)}%, seed ${run.seed}`,
      )
    },

    deleteRun: async (id) => {
      await guard('Removing run…', async () => {
        await api.deleteRun(get().projectId!, id)
        set({ runs: await api.runs(get().projectId!) })
        await refreshLedger()
      })
    },

    clearRuns: async () => {
      await guard('Clearing runs…', async () => {
        const r = await api.clearRuns(get().projectId!)
        set({ runs: [] })
        await refreshLedger()
        get().notify('ok', `Cleared ${r.cleared} run${r.cleared === 1 ? '' : 's'}.`)
      })
    },

    clearManualEdges: async () => {
      await guard('Clearing hand-authored edges…', async () => {
        const r = await api.clearManualEdges(get().projectId!)
        await refreshLedger(r.ledger)
        get().notify('ok', `Cleared ${r.cleared} hand-authored edge${r.cleared === 1 ? '' : 's'}.`)
      })
    },

    addEdge: async (s, t) => {
      const l = await guard('Adding edge…', () => api.addEdge(get().projectId!, s, t))
      if (l) {
        await refreshLedger(l)
        get().notify('ok', `${s} → ${t} added at frequency 0 — select it to include it.`)
      }
    },

    removeEdge: async (s, t) => {
      const l = await guard('Removing edge…', () => api.removeEdge(get().projectId!, s, t))
      if (l) await refreshLedger(l)
    },

    toggleSelected: async (s, t) => {
      const ledger = get().ledger
      if (!ledger) return
      const current = ledger.edges
        .filter((e) => e.selected)
        .map((e) => [e.source, e.target] as [string, string])
      const on = current.some(([a, b]) => a === s && b === t)
      const next = on
        ? current.filter(([a, b]) => !(a === s && b === t))
        : [...current, [s, t] as [string, string]]
      const l = await guard('Updating selection…', () => api.setSelection(get().projectId!, next))
      if (l) await refreshLedger(l)
    },

    selectAll: async (which) => {
      const ledger = get().ledger
      if (!ledger) return
      if (which === 'none') {
        const l = await guard('Updating selection…', () => api.setSelection(get().projectId!, []))
        if (l) await refreshLedger(l)
        return
      }
      // Not a filter over the ledger — a greedy acyclic construction on the server.
      // Filtering was the old bug: every run returns a DAG, their union need not be
      // one, and the whole selection was then rejected for a cycle the user never
      // drew. See POST /graph/selection/auto (Plan 3 W19).
      const r = await guard('Selecting the strongest acyclic set…', () =>
        api.autoSelect(get().projectId!, 1),
      )
      if (!r) return
      await refreshLedger(r.ledger)
      const skipped = r.skipped.length
      get().notify(
        skipped ? 'warn' : 'ok',
        `${r.n_selected} edge${r.n_selected === 1 ? '' : 's'} selected` +
          (skipped
            ? `; ${skipped} skipped because they would have closed a cycle — ` +
              r.skipped
                .slice(0, 3)
                .map((s) => `${s.source}→${s.target}`)
                .join(', ') +
              (skipped > 3 ? ` and ${skipped - 3} more` : '') +
              '. Lower-frequency orientations lose, so this is where your methods disagree.'
            : '. The graph is acyclic and every edge satisfies Assumption 1.'),
      )
    },

    /** The k most-agreed-on edges that still form a DAG — same construction, capped. */
    selectTopK: async (k) => {
      const r = await guard(`Selecting the top ${k} edges…`, () =>
        api.autoSelect(get().projectId!, 1, k),
      )
      if (!r) return
      await refreshLedger(r.ledger)
      get().notify(
        r.n_selected < k ? 'warn' : 'ok',
        `${r.n_selected} of the top ${k} selected` +
          (r.min_kept_frequency !== null
            ? `, down to frequency ${r.min_kept_frequency}`
            : '') +
          (r.skipped.length
            ? `; ${r.skipped.length} skipped for closing a cycle`
            : '') +
          (r.cut_off ? `; ${r.cut_off} left below the cut.` : '.'),
      )
    },

    /* --- priors (W15) --------------------------------------------------- */
    loadPriors: async () => {
      const pid = get().projectId
      if (!pid || !get().ctx) return
      try {
        set({ priors: await api.priors(pid) })
      } catch {
        /* no context yet — the panel renders its own empty state */
      }
    },

    savePriorMeta: async (body) => {
      const r = await guard('Saving metadata…', () => api.savePriorMeta(get().projectId!, body))
      if (r) set({ priors: r })
    },

    estimatePriors: async (useKg) => {
      const r = await guard('Querying the LLM for every admissible pair…', () =>
        api.estimatePriors(get().projectId!, { use_kg: useKg }),
      )
      if (!r) return
      set({ priors: r })
      get().notify(
        'ok',
        `Priors estimated in ${r.elapsed}s — ${r.n_informative} of the pair entries are ` +
          'informative; the rest stay at the neutral 0.5. Run GES-Prior to use them.' +
          (r.kg_errors?.length ? ` Endpoint enrichment failed: ${r.kg_errors[0]}` : ''),
      )
    },

    importPriors: async (f) => {
      const r = await guard(`Importing ${f.name}…`, () => api.importPriors(get().projectId!, f))
      if (!r) return
      set({ priors: r })
      get().notify(
        r.n_columns_matched ? 'ok' : 'warn',
        `Imported priors over ${r.columns.length} columns; ${r.n_columns_matched} match this ` +
          "project's variables. Unmatched pairs fall back to the neutral 0.5.",
      )
    },

    clearPriors: async () => {
      const r = await guard('Clearing priors…', () => api.clearPriors(get().projectId!))
      if (r) set({ priors: r })
    },

    /* --- modules 3-5 ---------------------------------------------------- */
    loadModel: async () => {
      const pid = get().projectId
      if (!pid) return
      const m = await api.model(pid)
      if (!m.fitted) {
        set({ ...MODEL_RESET })
        return
      }
      set({ model: m, answers: await api.answers(pid) })
      // The prior is what every card draws before the first Predict, and what every
      // posterior is drawn against afterwards, so it is loaded with the model, not
      // on demand (W23).
      try {
        set({ marginals: await api.marginals(pid) })
      } catch {
        set({ marginals: null })
      }
    },

    fitModel: async (body) => {
      const m = await guard(
        body.kind === 'cbn'
          ? 'Estimating conditional probability tables…'
          : 'Fitting mechanisms — this takes a while…',
        () => api.fitModel(get().projectId!, body),
      )
      if (!m) return
      set({ ...MODEL_RESET, model: m })
      await get().loadModel()
      get().notify(
        'ok',
        `${m.kind_label} fitted over ${m.n_rows} rows in ${m.elapsed}s. ` +
          (m.kind === 'cbn'
            ? 'Exact inference throughout; counterfactuals need a structural causal model.'
            : m.all_discrete
              ? 'Every variable is discrete, so exact inference is available too.'
              : 'Some variables are continuous, so inference is sampled.'),
      )
    },

    dropModel: async () => {
      await api.dropModel(get().projectId!)
      set({ ...MODEL_RESET })
    },

    evaluate: async (body) => {
      const e = await guard(
        body.gcm ? 'Evaluating mechanisms (gcm — minutes)…'
          : body.falsify ? 'Testing the graph against its own independencies…'
            : 'Cross-validating each node…',
        () => api.evaluate(get().projectId!, body),
      )
      if (!e) return
      set({ evaluation: e })
      get().notify(
        e.n_failing ? 'warn' : 'ok',
        e.n_failing
          ? `${e.n_failing} node${e.n_failing === 1 ? '' : 's'} cannot beat its own marginal ` +
            'baseline from its parents. Interventional answers about those nodes are noise.'
          : `${e.summary.nodes_beating_baseline}/${e.summary.nodes_scored} nodes beat their ` +
            `baseline (${e.elapsed}s).`,
      )
    },

    /* --- the board (W23) ------------------------------------------------ */
    assign: (node, mode, value) =>
      set((s) => ({ assignments: { ...s.assignments, [node]: { mode, value } } })),

    release: (node) =>
      set((s) => {
        const next = { ...s.assignments }
        delete next[node]
        return { assignments: next }
      }),

    clearBoard: () => set({ assignments: {}, prediction: null }),

    predict: async () => {
      const { projectId, assignments } = get()
      const evidence: Record<string, string | number> = {}
      const interventions: Record<string, string | number> = {}
      for (const [node, a] of Object.entries(assignments)) {
        ;(a.mode === 'intervened' ? interventions : evidence)[node] = a.value
      }
      const r = await guard('Predicting every node…', () =>
        api.predict(projectId!, { evidence, interventions }),
      )
      if (!r) return
      set({ prediction: r, answers: await api.answers(projectId!) })
      if (r.low_confidence) {
        get().notify(
          'warn',
          `Effective sample size ${r.ess?.toFixed(0)} — almost all the weight is on a handful ` +
            'of samples, so these distributions are not reliable. Loosen the evidence.',
        )
      }
    },

    selectNode: async (node) => {
      set({ selectedNode: node, mechanism: null })
      if (!node || !get().model) return
      try {
        set({ mechanism: await api.mechanism(get().projectId!, node) })
      } catch {
        set({ mechanism: null })
      }
    },

    /* --- module 4: the unit (W24) --------------------------------------- */
    setUnitFilter: async (node, value) => {
      const next = { ...get().unitFilters }
      if (value) next[node] = value
      else delete next[node]
      set({ unitFilters: next })
      await get().searchUnits()
    },

    clearUnitFilters: async () => {
      set({ unitFilters: {} })
      await get().searchUnits()
    },

    searchUnits: async () => {
      const pid = get().projectId
      if (!pid || !get().model) return
      try {
        const units = await api.searchUnits(pid, get().unitFilters)
        set({ units })
        // A constraint that excludes the chosen unit has to release it, or the panel
        // shows a factual state that no longer satisfies the filters on screen.
        const chosen = get().entity
        if (chosen && !units.classes.some((c) => c.entities.includes(chosen))) {
          await get().chooseEntity(null)
        }
      } catch {
        set({ units: null })
      }
      if (Object.keys(get().bands).length === 0) {
        try {
          set({ bands: await api.unitBands(pid) })
        } catch {
          /* every column categorical — no bands to offer */
        }
      }
    },

    setObserved: (values) =>
      // Choosing a hypothetical unit releases the KG entity and vice versa —
      // there is one factual state, and holding both would leave the board
      // showing one unit's baseline under the other's answer.
      set({
        observed: values,
        entity: values ? null : get().entity,
        factual: values ?? null,
        factualRows: values ? 1 : 0,
        cfPrediction: null,
      }),

    setObservedValue: (node, value) =>
      set((s) => {
        const next = { ...(s.observed ?? {}), [node]: value }
        return { observed: next, factual: next, cfPrediction: null }
      }),

    chooseEntity: async (iri) => {
      set({ entity: iri, observed: null, factual: null, factualRows: 0, cfPrediction: null })
      if (!iri) return
      try {
        const f = await api.unitFactual(get().projectId!, iri)
        set({ factual: f.values, factualRows: f.n_rows })
      } catch (e) {
        get().notify('err', (e as ApiError).message)
      }
    },

    setHypothetical: (node, value) =>
      set((s) => ({
        hypotheticals: { ...s.hypotheticals, [node]: { mode: 'intervened', value } },
      })),

    releaseHypothetical: (node) =>
      set((s) => {
        const next = { ...s.hypotheticals }
        delete next[node]
        return { hypotheticals: next }
      }),

    clearHypotheticals: () => set({ hypotheticals: {}, cfPrediction: null }),

    predictCounterfactual: async () => {
      const { projectId, entity, observed, hypotheticals } = get()
      if (!entity && !observed) return
      const interventions: Record<string, string | number> = {}
      for (const [node, a] of Object.entries(hypotheticals)) interventions[node] = a.value
      const r = await guard('Abducting this unit’s noise, then evaluating…', () =>
        api.predictCounterfactual(
          projectId!,
          observed ? { observed, interventions } : { entity: entity!, interventions },
        ),
      )
      if (!r) return
      set({ cfPrediction: r, answers: await api.answers(projectId!) })
    },

    clearAnswers: async () => {
      await api.clearAnswers(get().projectId!)
      set({ answers: [] })
    },

    saveLayout: (canvas, positions) => {
      const pid = get().projectId
      if (!pid) return
      set((s) => ({ layouts: { ...s.layouts, [canvas]: positions } }))
      window.clearTimeout(layoutTimers[canvas])
      layoutTimers[canvas] = window.setTimeout(() => {
        api.saveLayout(pid, canvas, positions).catch(() => {})
      }, 400)
    },

    resetLayout: async (canvas) => {
      const pid = get().projectId!
      await api.resetLayout(pid, canvas)
      set((s) => ({ layouts: { ...s.layouts, [canvas]: {} } }))
    },
  }
})
