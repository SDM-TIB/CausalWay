import { useState } from 'react'
import { useStore } from '../../stores/useStore'
import type { GraphPatternPreview } from '../../api/client'
import { Badge, Button, Field, Icon, Info, Input, Note, Panel, StatGrid, Tabs } from '../../app/ui'

/**
 * The flat join, run as a job rather than as a request (W21).
 *
 * It only runs when the user clicks Materialise — never on arrival, never on a
 * curation change. A row limit is a SPARQL LIMIT — the first N solutions in whatever
 * arbitrary order the store returned them — so making it the silent default meant
 * everything downstream was quietly computed on a slice nobody chose. The default is
 * the whole join, with the cost of that decision on screen while it is being paid:
 * elapsed time, a progress bar, and a Stop that returns control immediately. The Graph
 * Pattern box above the run controls is what makes running on demand safe to do
 * *without* pre-flighting it — it shows the same query text live, at no query cost, so
 * the user can check the join's shape before choosing to pay for it (§Preprocess
 * revision 3).
 *
 * Stop is honest about what it can do. rdflib evaluates the SELECT inside a generator
 * this app does not own and a Python thread cannot be killed from outside, so stopping
 * discards the result rather than halting the query.
 */
export function MaterialisePanel() {
  const { components, component, autoComponent, limit, mat, matJob, dtypes } = useStore()
  const { graphPattern, graphPatternError } = useStore()
  const { setLimit, materialise, cancelMaterialise } = useStore()
  const [tab, setTab] = useState<'rows' | 'sparql' | 'types' | 'stats'>('rows')
  const [draftLimit, setDraftLimit] = useState('')

  const running = matJob?.state === 'running'
  const multi = mat?.has_multi_relation
  // W31: which component the query above is for. `null` means auto, and auto is
  // the server's own choice, reported back on `auto_component`.
  const effective = component ?? autoComponent
  const active = components.find((c) => c.index === effective) ?? null
  const empty = active !== null && active.n_retained === 0

  return (
    <Panel
      step="1·5"
      title="Materialisation"
      right={
        <Info>
          A knowledge graph has no rows, and every discovery algorithm needs an n × p matrix.
          This is where one comes from: one SPARQL basic graph pattern over the whole
          component, each solution a row, each candidate node a column. It is not a preview
          of a separate step — module 2 fits on exactly this frame. The emitted query is shown
          verbatim so the join is auditable outside this app (Plan 3 §3.8, W18).
        </Info>
      }
      bodyClass="space-y-2.5"
    >
      <div className="flex flex-wrap items-end gap-2">
        {running ? (
          <Button variant="danger" onClick={cancelMaterialise}>
            <Icon name="x" className="text-[12px]" />
            Stop
          </Button>
        ) : (
          <Button
            variant="primary"
            disabled={empty}
            title={empty ? `Component ${effective} has no retained nodes to select.` : undefined}
            onClick={() => materialise(limit)}
          >
            <Icon name="play" className="text-[12px]" />
            {mat ? 'Re-run' : 'Materialise'}
          </Button>
        )}
        {components.length > 1 && effective !== null && (
          <Badge tone="teal" title="The connected component this query runs over. Change it in Join scope.">
            component {effective} of {components.length}
            {component === null ? ' · auto' : ''}
          </Badge>
        )}
      </div>

      {running && <Progress job={matJob!} onLimit={cancelMaterialise} />}

      {matJob?.state === 'cancelled' && (
        <Note tone="warn">
          Stopped after {matJob.elapsed}s and the result discarded. {matJob.note} Set a row
          limit below and run again, or wait it out.
        </Note>
      )}
      {matJob?.state === 'error' && <Note tone="err">{matJob.error}</Note>}

      {/* The limit is the escape hatch, not the default, so it lives below the run
          control rather than beside it — you reach for it once unlimited has hurt. */}
      {!running && (
        <div className="flex flex-wrap items-end gap-2">
          <Field
            label="Row limit"
            hint="A SPARQL LIMIT — the first N solutions the store returns, not a random sample. Empty means no limit."
          >
            <Input
              type="number"
              min={1}
              placeholder="no limit"
              value={draftLimit}
              onChange={(e) => {
                setDraftLimit(e.target.value)
                setLimit(e.target.value === '' ? null : Number(e.target.value) || null)
              }}
              className="w-28"
            />
          </Field>
          {limit !== null && (
            <Button size="sm" variant="ghost" onClick={() => { setDraftLimit(''); setLimit(null) }}>
              Clear the limit
            </Button>
          )}
        </div>
      )}

      <GraphPatternBox preview={graphPattern} error={graphPatternError} />

      {components.length > 1 && (
        <Note>
          The class graph has {components.length} connected components, and this is the query
          for <strong>component {effective}</strong>
          {active ? ` (${active.n_retained} of ${active.n_nodes} nodes retained)` : ''}. Pick a
          different one in <strong>Join scope</strong> on the left and the query above changes
          with it. Assumption 1 forbids every edge between two classes with no relation path, so
          a cross-component edge is inadmissible by construction: running one component at a time
          loses nothing, but each component is its own analysis, not a slice of a larger one
          (Plan 3 §3.6, W31).
        </Note>
      )}

      {mat && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="indigo">{mat.row_count} rows</Badge>
            <Badge tone="indigo">{mat.column_count} columns</Badge>
            <Badge tone="teal">
              component {mat.component} of {mat.n_components_total}
            </Badge>
            {matJob?.elapsed ? <Badge tone="muted">{matJob.elapsed}s</Badge> : null}
            {multi && <Badge tone="orange">multiplicity &gt; 1</Badge>}
            {mat.truncated ? (
              <Badge tone="orange">truncated at {mat.limit}</Badge>
            ) : (
              <Badge tone="green">whole join</Badge>
            )}
          </div>

          {mat.truncated && (
            <Note tone="warn">
              The join returned at least the {mat.limit}-row limit, so this frame is the{' '}
              <strong>first</strong> {mat.limit} solutions in whatever order the store
              produced them — arbitrary truncation, not a random sample. Everything
              downstream is fitted on this subset. Clear the limit, or accept it knowingly.
            </Note>
          )}
          {mat.near_unique_columns.length > 0 && (
            <Note tone="warn">
              Near-unique after the join, so effectively an identifier rather than a
              variable: <code>{mat.near_unique_columns.join(', ')}</code>.
            </Note>
          )}

          {multi && (
            <Note tone="warn">
              A 1:N relation duplicates rows here, so the frame is not i.i.d.: every downstream
              score is computed over a row set with repeated entities and will read optimistically.
              <code className="ml-1">evaluation.row_weights</code> /{' '}
              <code>resample_by_weight</code> are partial mitigations, not fixes (Plan 1 §9.1).
            </Note>
          )}
          {mat.warnings.map((w, i) => (
            <Note key={i} tone="warn">
              {w}
            </Note>
          ))}

          <div className="flex items-center gap-2">
            <Tabs
              value={tab}
              onChange={setTab}
              tabs={[
                { id: 'rows', label: 'Preview rows' },
                { id: 'types', label: 'Column types' },
                { id: 'sparql', label: 'SPARQL' },
                { id: 'stats', label: 'Per-class' },
              ]}
            />
            {tab === 'sparql' && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigator.clipboard?.writeText(mat.sparql)}
              >
                Copy
              </Button>
            )}
          </div>

          {tab === 'rows' && (
            <div className="max-h-72 overflow-auto rounded-lg border border-line">
              <table className="w-full border-collapse text-[11px]">
                <thead>
                  <tr>
                    {mat.columns.map((c) => (
                      <th
                        key={c}
                        className="sticky top-0 z-10 whitespace-nowrap border-b border-line bg-surface2 px-2 py-1.5 text-left font-medium text-muted"
                        title={c}
                      >
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mat.preview_rows.map((r, i) => (
                    <tr key={i} className="hover:bg-surface2/60">
                      {mat.columns.map((c) => (
                        <td
                          key={c}
                          className="whitespace-nowrap border-b border-linesoft px-2 py-1 font-mono text-ink/85"
                        >
                          {r[c] === null || r[c] === undefined ? (
                            <span className="text-faint">—</span>
                          ) : (
                            String(r[c])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {tab === 'types' && <TypeTable dtypes={dtypes} counts={mat.distinct_counts} />}

          {tab === 'sparql' && (
            <pre className="max-h-72 overflow-auto rounded-lg border border-line bg-surface2 p-2.5 font-mono text-[11px] leading-relaxed">
              {mat.sparql}
            </pre>
          )}

          {tab === 'stats' && (
            <div className="rounded-lg border border-line bg-surface2 p-2.5">
              <StatGrid
                rows={Object.entries(mat.multiplicity).map(([k, v]) => [
                  k.split('/').pop()?.split('#').pop() ?? k,
                  <span className={v > 1.0001 ? 'text-orange' : undefined}>{v.toFixed(2)}×</span>,
                ])}
              />
              {mat.skipped_patterns.length > 0 && (
                <p className="mt-2 text-[11px] text-faint">
                  Skipped: {mat.skipped_patterns.join(', ')}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </Panel>
  )
}

const TONE: Record<string, string> = {
  categorical: 'text-amber',
  ordinal: 'text-teal',
  discrete: 'text-sky',
  continuous: 'text-sky',
}

/** What each column will be at fit time, at the current threshold (W20). */
function TypeTable({
  dtypes,
  counts,
}: {
  dtypes: Record<string, string>
  counts: Record<string, number>
}) {
  const rows = Object.entries(dtypes)
  const anyContinuous = rows.some(([, d]) => d === 'continuous' || d === 'discrete')
  return (
    <div className="space-y-2">
      <div className="max-h-64 overflow-auto rounded-lg border border-line">
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr>
              {['Column', 'Distinct', 'Modelled as'].map((h) => (
                <th
                  key={h}
                  className="sticky top-0 z-10 whitespace-nowrap border-b border-line bg-surface2 px-2 py-1.5 text-left font-medium text-muted"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([col, dt]) => (
              <tr key={col} className="hover:bg-surface2/60">
                <td className="whitespace-nowrap border-b border-linesoft px-2 py-1 font-mono">
                  {col}
                </td>
                <td className="border-b border-linesoft px-2 py-1 text-right font-mono tabular-nums text-faint">
                  {counts[col] ?? '—'}
                </td>
                <td className={`border-b border-linesoft px-2 py-1 ${TONE[dt] ?? 'text-muted'}`}>
                  {dt}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {anyContinuous ? (
        <Note tone="warn">
          At least one column is numeric rather than categorical, so the network is not fully
          discrete: exact inference is unavailable and a causal Bayesian network cannot be
          fitted at all. Raise the max levels above those columns&apos; distinct counts to
          read them as labels instead — at the cost of losing their ordering.
        </Note>
      ) : (
        <Note tone="ok">
          Every column is discrete, so exact inference is available and both model kinds can
          be fitted.
        </Note>
      )}
    </div>
  )
}

/**
 * The query text `materialise` would run right now, kept live from curation alone —
 * no join, no schema.graph.query — so it updates on every node, join or row-limit
 * change and lets the user catch a wrong pattern (an unintended cross product, a
 * relation that didn't join) before spending the time on the real thing.
 */
function GraphPatternBox({
  preview,
  error,
}: {
  preview: GraphPatternPreview | null
  error: string | null
}) {
  return (
    <Field
      label="Query"
      hint="The SPARQL this Materialise run will use, live from the component and curation above — nothing has been queried yet."
    >
      {preview ? (
        <>
          {preview.warnings.map((w, i) => (
            <Note key={i} tone="warn">
              {w}
            </Note>
          ))}
          <pre className="max-h-56 overflow-auto rounded-lg border border-line bg-surface2 p-2.5 font-mono text-[11px] leading-relaxed">
            {preview.sparql}
          </pre>
        </>
      ) : (
        /* The server's own reason rather than one guess for every cause. With a
           component picker "every node is excluded" is usually wrong — the
           common case is a component whose nodes are all excluded, which is a
           different sentence and a different fix (W31). */
        <p className="rounded-lg border border-dashed border-line px-2.5 py-2 text-[11px] text-faint">
          {error ?? 'Nothing to query yet — every node is excluded.'}
        </p>
      )}
    </Field>
  )
}

function Progress({ job, onLimit }: { job: { elapsed: number; slow?: boolean; slow_after?: number; note: string }; onLimit: () => void }) {
  const slowAfter = job.slow_after ?? 180
  // An indeterminate bar until the job passes the "this is slow" mark, then a real
  // proportion of it: there is no row count to measure against, and inventing one
  // would be a lie about progress rather than a report of it.
  const pct = Math.min(100, (job.elapsed / slowAfter) * 100)
  return (
    <div className="space-y-1.5 rounded-lg border border-indigo/30 bg-indigo/8 p-2.5">
      <div className="flex items-baseline gap-2 text-[11.5px]">
        <span className="font-medium text-indigo">Running the join…</span>
        <span className="flex-1" />
        <span className="font-mono tabular-nums text-muted">{job.elapsed.toFixed(1)}s</span>
      </div>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-surface3">
        <span
          className={job.slow ? '' : 'ck-indeterminate'}
          style={{
            position: 'absolute',
            inset: '0 auto 0 0',
            width: job.slow ? '100%' : '38%',
            borderRadius: 999,
            background: job.slow ? 'var(--ck-orange)' : 'var(--ck-indigo)',
            ...(job.slow ? {} : { transform: `translateX(${pct * 1.6}%)` }),
          }}
        />
      </div>
      {job.slow ? (
        <p className="text-[11px] leading-relaxed text-orange">
          Past {slowAfter}s. This join is large enough that a limit may be the better
          trade — stop it, set a row limit below and re-run, or leave it to finish.{' '}
          <button className="underline decoration-dotted underline-offset-2" onClick={onLimit}>
            Stop now
          </button>
          .
        </p>
      ) : (
        <p className="text-[11px] leading-relaxed text-faint">{job.note}</p>
      )}
    </div>
  )
}
