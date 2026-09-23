import { useStore } from '../../stores/useStore'
import { Badge, Button, Icon, Note, Panel } from '../../app/ui'
import { api } from '../../api/client'
import type { AnswerKind } from '../../api/client'

const BACKEND_NOTE: Record<string, string> = {
  'pgmpy-exact':
    'Exact variable elimination over the fitted conditional probability tables. An ' +
    'intervention is answered on the mutilated network — the intervened node’s parents ' +
    'cut, its table pinned — so do() is graph surgery, not an adjustment formula. No ' +
    'sampling error.',
  'ancestral-sampling':
    'Forward simulation from the fitted model with nothing conditioned on: the ' +
    'observational marginals, and the baseline every other answer is read against.',
  'likelihood-weighting':
    'One ancestral draw with the evidence nodes clamped and each sample weighted by its ' +
    'density there. Never rejects a sample — but check the effective sample size.',
  'gcm.interventional_samples':
    'Sampling from the mutilated SCM: each intervened node is fixed, its incoming edges ' +
    'cut, and everything downstream regenerated.',
  'gcm.interventional-sampling':
    'One pass that both mutilates and weights: intervened nodes are set and contribute no ' +
    'likelihood factor, observed nodes are set and weight the sample by their density. ' +
    'Every card comes off the same draw, so the numbers on different cards are mutually ' +
    'consistent.',
  'gumbel-max-abduction':
    'This unit’s noise is abducted from its factual rows, then held fixed while the ' +
    'hypothetical is evaluated — that shared noise is what makes it counterfactual rather ' +
    'than merely interventional.',
  'anm-abduction':
    'The noise is an exact residual for an additive-noise mechanism, so one draw is the ' +
    'whole answer; redrawing would just repeat it.',
}

/** The disclosure strip under the board: which engine answered, and how well. */
export function BackendNote({
  backend,
  ess,
  lowConfidence,
  coupling,
  nSamples,
}: {
  backend: string | null
  ess?: number | null
  lowConfidence?: boolean
  coupling?: string | null
  nSamples?: number | null
}) {
  if (!backend) return null
  return (
    <div className="space-y-2">
      <div className="rounded-lg border border-line bg-surface2 p-2">
        <div className="mb-1 flex flex-wrap items-center gap-1.5">
          <Badge tone="indigo">{backend}</Badge>
          {nSamples ? <Badge tone="muted">{nSamples} samples</Badge> : null}
          {ess !== null && ess !== undefined && (
            <Badge tone={lowConfidence ? 'red' : 'green'}>ESS {ess.toFixed(0)}</Badge>
          )}
          {coupling && <Badge tone="orange">{coupling}</Badge>}
        </div>
        <p className="text-[10.5px] leading-relaxed text-faint">
          {BACKEND_NOTE[backend] ?? 'Backend reported as returned by the engine.'}
        </p>
      </div>
      {lowConfidence && (
        <Note tone="err">
          The effective sample size is below the warning threshold: almost all the weight
          sits on a handful of samples, so these distributions are not reliable. Loosen the
          evidence, or intervene instead of observing.
        </Note>
      )}
      {coupling === 'gumbel-max' && (
        <Note tone="warn">
          A categorical counterfactual is <em>not</em> identified by the observational
          distribution — several couplings fit the same data and disagree here. Gumbel-max
          fixes one, and the export records that choice so the answer is never read as
          assumption-free.
        </Note>
      )}
    </div>
  )
}

/**
 * Every query this session has run.
 *
 * Nothing here is persisted server-side — the log is capped at the last 50 answers and
 * dies with the process — so the download is not a convenience, it is the only durable
 * record that these numbers were ever produced (W14). The board itself now carries the
 * *answer*; this carries the history of asking.
 */
export function AnswerLogPanel({ kinds }: { kinds?: AnswerKind[] }) {
  const { answers, clearAnswers, projectId, model } = useStore()
  const mine = kinds ? answers.filter((a) => kinds.includes(a.kind)) : answers
  // RDF needs the fitted model to resolve node IRIs; CSV does not. Offering a
  // ttl link that can only 400 would be worse than not offering it.
  const canRdf = Boolean(projectId && model)
  return (
    <Panel
      title="Query log"
      right={
        <div className="flex items-center gap-1.5">
          <Badge tone="muted">{mine.length}</Badge>
          {answers.length > 0 && projectId && canRdf && (
            <a
              href={api.exportUrl(projectId, 'answers', 'ttl')}
              download
              title="Export the whole log as cw: RDF — each query with its interventions, its observational conditions and its estimate"
              className="rounded-md border border-line px-1.5 py-[2px] font-mono text-[10px] uppercase text-muted transition-colors hover:border-indigo/50 hover:bg-indigo/10 hover:text-indigo"
            >
              ttl
            </a>
          )}
          {answers.length > 0 && projectId && (
            <a
              href={`/api/projects/${projectId}/export?what=answers&format=csv`}
              download
              title="Export every answer with its estimand and backend"
            >
              <Button size="sm" variant="ghost">
                <Icon name="download" className="text-[13px]" />
              </Button>
            </a>
          )}
          {answers.length > 0 && (
            <Button size="sm" variant="ghost" onClick={clearAnswers}>
              Clear
            </Button>
          )}
        </div>
      }
      bodyClass="space-y-1"
      note="Every executed query, newest first — capped at the last 50 and never persisted. Export it or lose it when the session ends."
    >
      {mine.length === 0 && (
        <p className="py-2 text-center text-[11.5px] text-faint">No questions asked yet.</p>
      )}
      <ul className="-mx-1 max-h-[34vh] overflow-y-auto">
        {[...mine].reverse().map((a) => (
          <li key={a.answer_id} className="group rounded-lg px-1.5 py-1 hover:bg-surface2">
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono text-[10px] text-faint">#{a.answer_id}</span>
              <span className="min-w-0 flex-1 truncate text-[11px]" title={a.estimand}>
                {a.estimand}
              </span>
              {canRdf && projectId && (
                <a
                  href={api.answerExportUrl(projectId, a.answer_id, 'ttl')}
                  download
                  title="This one query and its estimate, as cw: RDF"
                  className="shrink-0 font-mono text-[9.5px] uppercase text-faint opacity-0 transition-opacity hover:text-indigo group-hover:opacity-100"
                >
                  ttl
                </a>
              )}
            </div>
            <div className="flex flex-wrap gap-1.5 text-[9.5px] text-faint">
              <span
                className={
                  a.kind === 'conditional'
                    ? 'text-green'
                    : a.kind === 'interventional'
                      ? 'text-orange'
                      : 'text-indigo'
                }
              >
                {a.kind}
              </span>
              <span>{a.backend}</span>
              {a.low_confidence && <span className="text-red">low ESS</span>}
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  )
}
