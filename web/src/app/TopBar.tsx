import { useStore } from '../stores/useStore'
import { MODULES } from './modules'
import type { ModuleId } from '../stores/useStore'
import { Button, Icon, IconButton, Spinner } from './ui'
import { ExportMenu } from './ExportMenu'
import { ImportButton } from './ImportButton'

function Wordmark() {
  return (
    <div className="flex shrink-0 items-center gap-2 pr-1">
      {/* Three nodes and a directed edge — the thing the app is about. */}
      <svg viewBox="0 0 28 28" className="size-[22px]" aria-hidden>
        <path d="M8 8.5 L20 8.5" stroke="var(--ck-faint)" strokeWidth="1.5" />
        <path
          d="M8 8.5 L14 19.5"
          stroke="var(--ck-indigo)"
          strokeWidth="1.8"
          strokeLinecap="round"
        />
        <circle cx="8" cy="8.5" r="3.6" fill="var(--ck-teal)" />
        <circle cx="20" cy="8.5" r="2.6" fill="var(--ck-sky)" />
        <circle cx="14" cy="19.5" r="3" fill="var(--ck-indigo)" />
      </svg>
      <span className="text-[14px] font-semibold tracking-tight">CausalWay</span>
    </div>
  )
}

function ModuleNav() {
  const module = useStore((s) => s.module)
  const setModule = useStore((s) => s.setModule)
  const hasSchema = useStore((s) => s.schema !== null)
  const graph = useStore((s) => s.graph)
  const caps = useStore((s) => s.capabilities)

  /**
   * W29. "No source loaded" used to be the only way a module could be
   * unreachable, so the rule was `everything except module 1 needs a schema`.
   * An imported bundle breaks that: it arrives with a fitted model and *no*
   * source, and modules 3 and 4 are exactly the ones that still work. So the
   * server's own capability report decides when it has one, and the old rule
   * stays as the fallback for an ordinary project.
   */
  const lockOf = (id: ModuleId): string | null => {
    if (caps) {
      const key = id === 'preprocess' ? 'source' : id
      const cap = caps[key as keyof typeof caps] as { available: boolean; reason: string | null }
      return cap && !cap.available ? cap.reason ?? 'Not available yet' : null
    }
    return id !== 'preprocess' && !hasSchema ? 'Load a source in module 1 first' : null
  }

  return (
    <nav
      aria-label="Pipeline modules"
      className="flex items-center gap-0.5 rounded-xl border border-line bg-surface2/70 p-[3px] backdrop-blur"
    >
      {MODULES.map((m) => {
        const active = module === m.id
        const lockReason = lockOf(m.id)
        const locked = lockReason !== null
        const done =
          (m.id === 'preprocess' && hasSchema) ||
          (m.id === 'discovery' && (graph?.n_edges ?? 0) > 0)
        return (
          <button
            key={m.id}
            onClick={() => !locked && setModule(m.id)}
            disabled={locked}
            title={lockReason ?? m.sub}
            aria-current={active ? 'page' : undefined}
            className={`group relative flex shrink-0 items-center gap-1.5 rounded-lg px-2 py-1.5 text-left transition-all duration-150 lg:gap-2 lg:px-2.5 ${
              active
                ? 'bg-surface text-ink shadow-[var(--ck-shadow)]'
                : locked
                  ? 'cursor-not-allowed text-faint/50'
                  : 'text-muted hover:bg-surface/70 hover:text-ink'
            }`}
          >
            <span
              className={`grid size-[18px] shrink-0 place-items-center rounded-md font-mono text-[10px] font-semibold transition-colors ${
                active
                  ? 'bg-indigo text-white dark:text-[#0b0e14]'
                  : done
                    ? 'bg-green/15 text-green'
                    : 'bg-surface3 text-faint'
              }`}
            >
              {done && !active ? <Icon name="check" className="text-[11px]" /> : m.n}
            </span>
            <span className="hidden flex-col leading-tight lg:flex">
              <span className="text-[12px] font-medium">{m.label}</span>
              <span className="text-[9.5px] uppercase tracking-wide text-faint">
                {m.built ? m.sub : 'planned'}
              </span>
            </span>
            <span className="text-[12px] font-medium lg:hidden">{m.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

export function TopBar() {
  const { leftOpen, rightOpen, theme, busy, projectId } = useStore()
  const { toggleLeft, toggleRight, toggleTheme, newProject } = useStore()

  return (
    <header className="themed z-30 flex h-13 shrink-0 items-center gap-3 border-b border-line bg-surface/85 px-3 py-2 backdrop-blur-xl">
      <Wordmark />

      <div className="flex shrink-0 items-center gap-0.5 border-l border-line pl-2">
        <IconButton
          label={leftOpen ? 'Hide left sidebar (⌘[)' : 'Show left sidebar (⌘[)'}
          variant="ghost"
          size="sm"
          active={leftOpen}
          onClick={toggleLeft}
        >
          <Icon name="panelLeft" className="text-[15px]" />
        </IconButton>
        <IconButton
          label={rightOpen ? 'Hide right sidebar (⌘])' : 'Show right sidebar (⌘])'}
          variant="ghost"
          size="sm"
          active={rightOpen}
          onClick={toggleRight}
        >
          <Icon name="panelRight" className="text-[15px]" />
        </IconButton>
      </div>

      {/* The nav is the one thing allowed to shrink and scroll: the sidebar and
          theme toggles must stay reachable at every width, so they never yield.
          `mx-auto` on the child rather than `justify-center` on the box — centring
          an overflowing flex container clips both ends and strands the scroll. */}
      <div className="flex min-w-0 flex-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <div className="mx-auto">
          <ModuleNav />
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1.5">
        {busy && (
          <span className="mr-1 hidden items-center gap-1.5 text-[11.5px] text-muted md:flex">
            <Spinner className="text-indigo" />
            {busy}
          </span>
        )}
        <ExportMenu />
        <ImportButton />
        <IconButton
          label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          variant="ghost"
          size="sm"
          onClick={toggleTheme}
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} className="text-[15px]" />
        </IconButton>
        <Button size="sm" onClick={newProject} title="Discard this session and start over">
          New project
        </Button>
        {projectId && (
          <code
            className="hidden rounded-md border border-line bg-surface2 px-1.5 py-[3px] font-mono text-[10px] text-faint xl:block"
            title="Session-scoped project id — no account, nothing persisted server-side (Plan 3 W6/W14)"
          >
            {projectId}
          </code>
        )}
      </div>
    </header>
  )
}
