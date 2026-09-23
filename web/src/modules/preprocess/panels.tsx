import { useMemo, useRef, useState } from 'react'
import { useStore } from '../../stores/useStore'
import type { NodeInfo, ObjectRole } from '../../api/client'
import { Badge, Button, Field, Icon, Info, Input, Note, Panel, Select, StatGrid, Switch } from '../../app/ui'

export function SourcePanel() {
  const { samples, projectId, schema } = useStore()
  const { loadSample, uploadFile, loadEndpoint } = useStore()
  const [sample, setSample] = useState('')
  const [endpoint, setEndpoint] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const first = samples.find((s) => s.exists)?.sample_id ?? ''
  const chosen = sample || first

  return (
    <Panel step="1·1" title="Source" bodyClass="space-y-2.5">
      <Field label="Sample knowledge graph">
        <Select value={chosen} onChange={(e) => setSample(e.target.value)}>
          {samples.map((s) => (
            <option key={s.sample_id} value={s.sample_id} disabled={!s.exists}>
              {s.sample_id}
              {s.exists ? '' : ' (missing)'}
            </option>
          ))}
        </Select>
      </Field>
      <Button
        className="w-full"
        disabled={!projectId || !chosen}
        onClick={() => loadSample(chosen)}
      >
        Load sample
      </Button>

      <div className="flex items-center gap-2 pt-1">
        <span className="h-px flex-1 bg-line" />
        <span className="text-[10px] uppercase tracking-wide text-faint">or</span>
        <span className="h-px flex-1 bg-line" />
      </div>

      <input
        ref={fileRef}
        type="file"
        accept=".ttl,.nt,.rdf,.xml,.jsonld,.n3,.trig,.nq"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) uploadFile(f)
          e.target.value = ''
        }}
      />
      <Button className="w-full" onClick={() => fileRef.current?.click()} disabled={!projectId}>
        <Icon name="layers" className="text-[14px]" />
        Upload RDF file
      </Button>

      <Field label="SPARQL endpoint">
        <Input
          value={endpoint}
          placeholder="http://host/repositories/kg"
          onChange={(e) => setEndpoint(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && endpoint && loadEndpoint(endpoint)}
        />
      </Field>
      <Button
        className="w-full"
        disabled={!projectId || !endpoint.trim()}
        onClick={() => loadEndpoint(endpoint.trim())}
      >
        Resolve endpoint
      </Button>

      {schema?.source && (
        <Note tone="ok">
          <strong>{schema.source.name}</strong> parsed — {schema.n_nodes} candidate nodes.
        </Note>
      )}
    </Panel>
  )
}

export function SchemaPanel() {
  const schema = useStore((s) => s.schema)
  if (!schema) return null
  const s = schema.schema
  return (
    <Panel
      title="Induced schema"
      right={
        <Info>
          <code>resolve_schema</code> defaults <code>infer_missing</code> to true for files and
          false for endpoints. A KG with no T-Box yields zero candidate nodes without it; a public
          endpoint can be brought down by it (Plan 3 §3.1).
        </Info>
      }
    >
      <StatGrid
        rows={[
          ['Classes', s.n_classes],
          ['Object properties', s.n_object_properties],
          ['Data properties', s.n_data_properties],
          ['Inferred declarations', s.n_inferred],
          ['Candidate nodes', schema.n_nodes],
        ]}
      />
      {s.conflicts.length > 0 && (
        <div className="mt-2">
          <Note tone="warn">
            {s.conflicts.length} declared/inferred conflict
            {s.conflicts.length === 1 ? '' : 's'} — the declaration wins (Plan 1 §3.1).
          </Note>
        </div>
      )}
      {/* Capped and scrollable: this is a *summary* panel, and an endpoint with
          97 classes turned it into a wall of chips that pushed every panel
          under it off the screen. */}
      <div className="mt-2 flex max-h-44 flex-wrap gap-1 overflow-y-auto">
        {s.classes.map((c) => (
          <Badge key={c} tone="teal">
            {c}
          </Badge>
        ))}
      </div>
    </Panel>
  )
}

/**
 * Which connected component the join runs over (W31).
 *
 * One `materialise` is one basic graph pattern over one component, and that is
 * not a limitation to be lifted: Assumption 1 forbids every edge between two
 * classes with no relation path, so joining two components could only produce a
 * cross product with no admissible edge across it. The limitation was that the
 * component was never *chosen* — the server picked the largest and the client
 * never sent anything else, which is invisible on a one-component sample KG and
 * hides sixty-nine out of seventy on a real endpoint.
 *
 * `auto` is kept as the resting state rather than replaced by a forced pick,
 * because it is right on every single-component KG and it re-decides as the
 * curation changes. Pinning one is the override, exactly as a row limit
 * overrides "no limit" in the materialise panel.
 */
export function ComponentPanel() {
  const { components, component, autoComponent, schema, setComponent } = useStore()
  const [open, setOpen] = useState(false)
  if (!schema || components.length === 0) return null

  const effective = component ?? autoComponent
  const active = components.find((c) => c.index === effective) ?? null
  // Ordered by size for the picker only — `index` stays the server's, because it
  // is the value every route is called with.
  const ordered = [...components].sort(
    (a, b) => b.n_retained - a.n_retained || b.n_nodes - a.n_nodes || a.index - b.index,
  )

  // One component is the ordinary case (both bundled samples), and a picker
  // offering a choice of one is noise. Report it and stop.
  if (components.length === 1) {
    return (
      <Panel step="1·3" title="Join scope">
        <Note tone="ok">
          The class graph is connected — one component, {components[0].n_nodes} candidate
          nodes — so there is only one join to make and nothing to choose.
        </Note>
      </Panel>
    )
  }

  return (
    <Panel
      step="1·3"
      title="Join scope — connected component"
      right={
        <div className="flex items-center gap-1.5">
          <Badge tone="teal">{components.length}</Badge>
          <Info>
            The flat join is <strong>one</strong> SPARQL basic graph pattern over{' '}
            <strong>one</strong> connected component of the class graph. Assumption 1 forbids
            every edge between two classes with no relation path, so an edge across two
            components is inadmissible by construction — joining them would compute a cross
            product that no causal edge could ever cross (Plan 3 §3.6).
            <br />
            So a KG with several components is several separate analyses, not one. This is
            where you choose which. Everything below follows it: the curation list, the live
            SPARQL, the join, and the discovery run.
            <br />
            <strong>auto</strong> means the component holding the most retained nodes,
            re-decided whenever you change the curation. Materialising pins whichever one
            actually ran.
          </Info>
        </div>
      }
      bodyClass="space-y-2.5"
    >
      <Field label="Component">
        <Select
          value={effective === null ? '' : String(effective)}
          onChange={(e) => setComponent(e.target.value === '' ? null : Number(e.target.value))}
        >
          {effective === null && <option value="">— nothing retained —</option>}
          {/*
            Components with nothing to select are listed, not hidden. On this
            endpoint 61 of 70 are single classes declaring no properties at all,
            and dropping them would quietly renumber the user's mental model of
            a schema they can see in full on the canvas. Marked instead, and
            sorted last, so the pickable ones are the ones at the top.
          */}
          {ordered.map((c) => (
            <option key={c.index} value={c.index}>
              {c.index} · {c.n_retained}/{c.n_nodes} nodes · {c.classes.slice(0, 3).join(', ')}
              {c.classes.length > 3 ? ` +${c.classes.length - 3}` : ''}
              {c.n_retained === 0 ? ' — nothing to select' : ''}
            </option>
          ))}
        </Select>
      </Field>

      <div className="flex flex-wrap items-center gap-1.5">
        {component === null ? (
          <Badge tone="muted" title="Following the curation — the component with the most retained nodes.">
            auto
          </Badge>
        ) : (
          <Button size="sm" variant="ghost" onClick={() => setComponent(null)}>
            Back to auto
          </Button>
        )}
        {active && (
          <>
            <Badge tone="teal">
              {active.n_classes} class{active.n_classes === 1 ? '' : 'es'}
            </Badge>
            <Badge tone={active.n_retained > 0 ? 'indigo' : 'red'}>
              {active.n_retained}/{active.n_nodes} retained
            </Badge>
          </>
        )}
      </div>

      {/* Two different states that both read as "0 retained", with two
          different fixes — one is curation, the other is the ontology. */}
      {active && active.n_retained === 0 && (
        <Note tone="err">
          {active.n_nodes === 0 ? (
            <>
              {active.n_classes === 1 ? 'This class declares' : 'These classes declare'} no
              properties at all, so this component yields no candidate nodes and there is
              nothing here to join. Nothing to fix in the curation — pick another component.
            </>
          ) : (
            <>
              All {active.n_nodes} of this component&apos;s nodes are excluded, so there is no
              column to select and the join has nothing to return. Keep at least one of them in
              the curation panel, or pick another component.
            </>
          )}
        </Note>
      )}

      {active && (
        <div>
          <button
            className="text-[11px] text-faint underline decoration-dotted underline-offset-2 hover:text-ink"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? 'Hide' : 'Show'} its {active.n_classes} class
            {active.n_classes === 1 ? '' : 'es'}
          </button>
          {open && (
            <div className="mt-1.5 flex max-h-40 flex-wrap gap-1 overflow-y-auto">
              {active.classes.map((c) => (
                <Badge key={c} tone="teal">
                  {c}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

export function ConstraintPanel() {
  const { constraint, stats, setConstraint, schema } = useStore()
  if (!schema) return null
  return (
    <Panel
      title="Assumption 1 — constraint"
      right={
        <Info>
          These control the admissible causal search space, not the ontology drawing. An object
          property's domain → range direction is fixed by the T-Box and does not move when these
          change (Plan 3 §3.4).
        </Info>
      }
      bodyClass="space-y-2.5"
    >
      <Field label="Relation direction" inline>
        <Select
          value={constraint.relation_direction}
          onChange={(e) => setConstraint({ relation_direction: e.target.value as 'both' | 'forward' })}
        >
          <option value="both">both</option>
          <option value="forward">forward</option>
        </Select>
      </Field>
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-muted">ε — intra-class edges</span>
        <Switch
          label="Allow intra-class edges"
          checked={constraint.allow_epsilon}
          onChange={(v) => setConstraint({ allow_epsilon: v })}
        />
      </div>
      <Field label="Max hops" inline>
        <Input
          type="number"
          min={1}
          max={3}
          value={constraint.max_hops}
          onChange={(e) => setConstraint({ max_hops: Number(e.target.value) || 1 })}
        />
      </Field>
      {stats && (
        <div className="rounded-lg border border-line bg-surface2 p-2">
          <StatGrid
            rows={[
              ['Nodes', stats.n_nodes],
              ['Allowed edges', stats.n_allowed],
              ['Forbidden edges', stats.n_forbidden],
              [
                'Pruning rate',
                <span className={stats.pruning_rate > 0 ? 'text-indigo' : 'text-faint'}>
                  {(stats.pruning_rate * 100).toFixed(1)}%
                </span>,
              ],
            ]}
          />
        </div>
      )}
      {stats?.pruning_rate === 0 && (
        <Note>
          Nothing is pruned: with one class and no object properties every pair is intra-class, so
          the topological constraint is vacuous here and constrained runs must match unconstrained
          ones exactly.
        </Note>
      )}
    </Panel>
  )
}

export function NodeCurationPanel() {
  const { nodes, setExclusions, toggleNode, setObjectRole, schema } = useStore()
  const { components, component, autoComponent } = useStore()
  const [filter, setFilter] = useState('')
  const [only, setOnly] = useState<'all' | 'data' | 'object' | 'excluded'>('all')
  /*
   * W31 — the list is scoped to the component being joined, because a node
   * outside it is not in the query on screen and cannot be put there by
   * curating it. Including or excluding one still *works* — it moves that other
   * component's own join — it just has no visible effect here, which used to
   * make "Exclude shown" feel broken on a 70-component schema. Off is the
   * escape hatch for curating a component you are not currently joining.
   */
  const [scoped, setScoped] = useState(true)

  const effective = component ?? autoComponent
  const multi = components.length > 1

  const inScope = useMemo(
    () => (multi && scoped && effective !== null
      ? nodes.filter((n) => n.component === effective)
      : nodes),
    [nodes, multi, scoped, effective],
  )

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase()
    return inScope.filter((n) => {
      if (f && !n.name.toLowerCase().includes(f)) return false
      if (only === 'data') return n.kind === 'data'
      if (only === 'object') return n.kind === 'object'
      if (only === 'excluded') return n.excluded
      return true
    })
  }, [inScope, filter, only])

  if (!schema) return null

  return (
    <Panel
      step="1·4"
      title="Candidate nodes & curation"
      right={
        <div className="flex items-center gap-1.5">
          <Badge tone="indigo">
            {inScope.filter((n) => !n.excluded).length}/{inScope.length}
          </Badge>
          <Info>
            {multi && (
              <>
                The list is scoped to <strong>component {effective}</strong> — the one the
                join above runs over. A node in another component is not in that query and
                cannot be curated into it; switch components, or turn the scope off to reach
                them anyway.
                <br />
              </>
            )}
            Excluding a <strong className="text-muted">data</strong> property only drops its{' '}
            <strong className="text-muted">column</strong> — its required pattern stays either
            way, so <strong className="text-teal">the row count never changes.</strong>
            <br />
            An <strong className="text-green">object</strong> property is a{' '}
            <strong>relationship</strong> <em>or</em> a <strong>causal variable</strong>, never
            both. As a relationship it joins the two classes into one row (the default). As a
            variable its value is the related entity, and that <em>costs the join</em> — the
            range class's own properties no longer ride along. It used to be allowed to be both,
            which put a column in the frame that was a deterministic function of the join that
            produced the row.
            <br />
            An object property whose range class declares{' '}
            <strong className="text-muted">no data properties</strong> is marked{' '}
            <strong className="text-green">cat</strong> and starts as a{' '}
            <strong>variable</strong>: an age band or a tumour stage modelled as a class has
            nothing to contribute through a join, and the one thing it does say — which instance
            this entity points at — is only a column if the property is a variable. The role
            selector still moves it back.
          </Info>
        </div>
      }
      bodyClass="space-y-2"
    >
      {multi && (
        <div className="flex items-center justify-between rounded-lg border border-line bg-surface2 px-2 py-1.5">
          <span className="text-[11px] text-muted">
            {scoped ? (
              <>
                Component <strong className="text-teal">{effective}</strong> only
              </>
            ) : (
              <>All {components.length} components</>
            )}
          </span>
          <Switch
            label="Scope the list to the component being joined"
            checked={scoped}
            onChange={setScoped}
          />
        </div>
      )}
      <Input placeholder="Filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="flex flex-wrap gap-1">
        {(['all', 'data', 'object', 'excluded'] as const).map((k) => (
          <Button key={k} size="sm" variant="ghost" active={only === k} onClick={() => setOnly(k)}>
            {k}
          </Button>
        ))}
      </div>
      <div className="flex gap-1.5">
        <Button
          size="sm"
          className="flex-1"
          onClick={() => setExclusions(shown.map((n) => n.name), false)}
        >
          Include shown
        </Button>
        <Button
          size="sm"
          className="flex-1"
          onClick={() => setExclusions(shown.map((n) => n.name), true)}
        >
          Exclude shown
        </Button>
      </div>
      {/*
        There is deliberately no "make every object property a variable" button.
        Each such choice drops a join, and dropping several at once is how a
        query ends up computing a cross product — the server refuses that, so a
        bulk control would mostly produce a bulk rejection. One at a time, with
        the conflict shown against that node, is the honest control.
      */}

      <ul className="-mx-1 max-h-[46vh] overflow-y-auto">
        {shown.map((n) => (
          <li key={n.name}>
            <div
              className={`flex w-full items-center gap-2 rounded-lg px-1.5 py-1 text-left transition-colors hover:bg-surface2 ${
                isInactive(n) ? 'opacity-45' : ''
              }`}
            >
              {/*
                An object property has no checkbox: "is this a candidate
                variable?" is not a separate question from its role, and a
                checkbox beside a role selector would offer a fourth state that
                does not exist. The role selector on the right is the whole
                control.
              */}
              {n.kind === 'object' ? (
                <span className="size-3.5 shrink-0" />
              ) : (
                <input
                  type="checkbox"
                  checked={!n.excluded}
                  onChange={() => toggleNode(n.name)}
                  aria-label={`Keep ${n.name} as a candidate node`}
                  className="size-3.5 shrink-0 accent-[var(--ck-indigo)]"
                />
              )}
              <span
                className="size-2.5 shrink-0 rounded-[3px]"
                style={{
                  background:
                    n.kind === 'object'
                      ? 'var(--ck-green)'
                      : n.value_hint === 'numeric'
                        ? 'var(--ck-sky)'
                        : 'var(--ck-amber)',
                  opacity: n.excluded ? 0.5 : 1,
                }}
              />
              <span
                className={`min-w-0 flex-1 truncate font-mono text-[11.5px] ${
                  isInactive(n) ? 'line-through' : ''
                }`}
                title={`${n.name} — ${n.range}`}
              >
                {n.name}
              </span>
              {/*
                W30 — the range class has no data properties of its own, so it
                is an attribute modelled as a class. Shown as a fact about the
                schema, not as a state of the curation: it stays marked when the
                user moves the role back to `rel`, because the reason it was
                seeded as a variable is still true.
              */}
              {/* Only while the scope is off — with it on, every row has the
                  same value and the column is pure noise (W31). */}
              {multi && !scoped && n.component !== null && (
                <span
                  className={`shrink-0 rounded px-1 py-0.5 font-mono text-[9px] ${
                    n.component === effective ? 'bg-teal/15 text-teal' : 'text-faint'
                  }`}
                  title={
                    n.component === effective
                      ? `Component ${n.component} — in the join currently on screen.`
                      : `Component ${n.component} — not in the join currently on screen, so curating it changes a query you are not looking at.`
                  }
                >
                  c{n.component}
                </span>
              )}
              {n.auto_variable && (
                <span
                  className="shrink-0 rounded bg-green/15 px-1 py-0.5 font-mono text-[9px] uppercase text-green"
                  title={`${n.range} declares no data properties — it is an attribute modelled as a class, so joining it contributes no columns. Started as a causal variable for that reason (W30); the role selector overrides it.`}
                >
                  cat
                </span>
              )}
              {n.n_distinct !== null && (
                <span
                  className={`shrink-0 font-mono text-[10px] ${
                    n.constant ? 'text-red' : n.near_unique ? 'text-orange' : 'text-faint'
                  }`}
                  title={
                    n.constant
                      ? 'Constant — dropped before fitting'
                      : n.near_unique
                        ? 'Near-unique — an identifier, not a variable'
                        : `${n.n_distinct} distinct values`
                  }
                >
                  {n.n_distinct}
                </span>
              )}
              {n.kind === 'object' && n.role_options ? (
                <RoleToggle node={n} onPick={setObjectRole} />
              ) : (
                <span className="shrink-0 text-[10px] text-faint">{n.kind}</span>
              )}
            </div>
          </li>
        ))}
        {shown.length === 0 && (
          <li className="px-1.5 py-3 text-center text-[11.5px] text-faint">No nodes match.</li>
        )}
      </ul>
    </Panel>
  )
}

/**
 * Where a numeric column stops being a set of labels and becomes a number (W20).
 *
 * This used to be a bare integer in the fit dialog, two modules downstream of the data
 * it describes, with no indication of what it did. It belongs here, next to the column
 * it retypes, showing exactly which columns it moves and what that costs — because the
 * threshold is not a tuning knob, it is a modelling claim about what those integers are.
 */
export function ColumnTypePanel() {
  const { typeOptions, dtypes, nodes, mat, setTypeOptions, busy } = useStore()
  const [draft, setDraft] = useState<string>(String(typeOptions.max_levels))

  useMemo(() => setDraft(String(typeOptions.max_levels)), [typeOptions.max_levels])

  const entries = Object.entries(dtypes)
  if (!mat || entries.length === 0) return null

  const counts = entries.reduce<Record<string, number>>((acc, [, dt]) => {
    acc[dt] = (acc[dt] ?? 0) + 1
    return acc
  }, {})
  const numericLabels = entries.filter(
    ([col, dt]) =>
      dt === 'categorical' &&
      Number.isFinite(Number(mat.preview_rows[0]?.[col])) &&
      nodes.some((n) => n.name === col && n.value_hint === 'numeric'),
  )
  const notDiscrete = entries.filter(([, dt]) => dt === 'continuous' || dt === 'discrete')

  return (
    <Panel
      title="Column types"
      right={
        <Info>
          <code>max levels</code> applies to <strong>integer</strong> columns only: at or
          below it they are modelled as unordered labels with a probability table each, and
          above it as numbers with a regressor each. A column whose values are genuinely
          fractional is always continuous however few distinct values it takes — three
          unrelated labels would throw away the ordering the numbers already carry. Nothing
          is ever dropped either way.
        </Info>
      }
      bodyClass="space-y-2.5"
      note="Changing this re-types the columns and discards any fitted model, but never re-runs the join."
    >
      <div className="flex items-end gap-2">
        <Field label="Max levels (integer columns)">
          <Input
            type="number"
            min={2}
            max={200}
            value={draft}
            disabled={!!busy}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => {
              const v = Number(draft)
              if (Number.isFinite(v) && v >= 2 && v <= 200 && v !== typeOptions.max_levels) {
                setTypeOptions({ max_levels: v })
              } else {
                setDraft(String(typeOptions.max_levels))
              }
            }}
            className="w-20"
          />
        </Field>
        {typeOptions.max_levels !== typeOptions.default_max_levels && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setTypeOptions({ max_levels: typeOptions.default_max_levels })}
          >
            Reset to {typeOptions.default_max_levels}
          </Button>
        )}
      </div>

      <StatGrid
        rows={(['categorical', 'ordinal', 'discrete', 'continuous'] as const)
          .filter((k) => counts[k])
          .map((k) => [k, counts[k]] as [string, number])}
      />

      {notDiscrete.length > 0 ? (
        <Note tone="warn">
          <code>{notDiscrete.map(([c]) => c).join(', ')}</code>{' '}
          {notDiscrete.length === 1 ? 'is' : 'are'} modelled as{' '}
          {notDiscrete.length === 1 ? 'a number' : 'numbers'}, so the network is not fully
          discrete: exact inference is off and a causal Bayesian network cannot be fitted.
          Raising the threshold above{' '}
          {Math.max(...notDiscrete.map(([c]) => mat.distinct_counts[c] ?? 0))} would bring{' '}
          {notDiscrete.length === 1 ? 'it' : 'them'} back — at the cost of the ordering.
        </Note>
      ) : (
        <Note tone="ok">
          Every column is discrete at this threshold, so both model kinds are available and
          inference is exact.
        </Note>
      )}
      {numericLabels.length > 0 && (
        <Note>
          Numeric but read as labels at this threshold:{' '}
          <code>{numericLabels.map(([c]) => c).join(', ')}</code>. The model cannot know that
          one of their values lies between two others.
        </Note>
      )}
    </Panel>
  )
}


/* ------------------------------------------------------------------ *
 * W26 — the object-property role selector.
 *
 * Three buttons rather than a checkbox pair, because the three roles are
 * genuinely exclusive and two checkboxes would render a fourth state the
 * materialiser cannot produce.
 *
 * `variable` is disabled, with the server's own reason on hover, when this
 * join is the only thing holding two classes in the same row. The verdict
 * comes from `bgp.join_role_options`, i.e. from the same function that would
 * refuse the change — the button is never disabled for a reason the server
 * would not also give, and never enabled for a choice the server would reject.
 * ------------------------------------------------------------------ */
/**
 * Dimmed and struck through means "this node does nothing", which for a data
 * property is `excluded` and for an object property is only the `dropped` role.
 * Reading `excluded` for both drew every *relationship* as struck out — a
 * relationship has `excluded = true` because it is not a candidate variable,
 * yet it is the one doing the most work in the query.
 */
function isInactive(n: NodeInfo): boolean {
  return n.kind === 'object' ? n.role === 'dropped' : n.excluded
}

const ROLE_LABEL = {
  relationship: 'rel',
  variable: 'var',
  dropped: 'off',
} as const

const ROLE_TITLE = {
  relationship: 'Relationship — joins the two classes into one row. Not a column.',
  variable:
    "Causal variable — a column holding the related entity's IRI. Drops the join, so the " +
    "range class's own properties no longer ride along.",
  dropped: 'Neither — no join and no column.',
} as const

function RoleToggle({
  node,
  onPick,
}: {
  node: NodeInfo
  onPick: (name: string, role: ObjectRole) => void
}) {
  const opts = node.role_options!
  return (
    <span className="flex shrink-0 gap-0.5">
      {(['relationship', 'variable', 'dropped'] as const).map((role) => {
        const blocked = role === 'variable' && !opts.can_be_variable
        const active = opts.role === role
        return (
          <button
            key={role}
            type="button"
            disabled={blocked && !active}
            onClick={() => !active && onPick(node.name, role)}
            title={blocked ? opts.reason ?? ROLE_TITLE[role] : ROLE_TITLE[role]}
            className={`rounded px-1 py-0.5 font-mono text-[9.5px] uppercase transition-colors ${
              active
                ? 'bg-indigo/15 text-indigo'
                : blocked
                  ? 'cursor-not-allowed text-faint opacity-40'
                  : 'text-faint hover:bg-surface2 hover:text-muted'
            }`}
          >
            {ROLE_LABEL[role]}
          </button>
        )
      })}
    </span>
  )
}
