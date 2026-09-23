/**
 * Arrowheads, once, document-global. SVG markers resolve by id across the
 * document, so both canvases reference these rather than each edge shipping
 * its own <defs> (which is what makes per-edge colour cheap).
 */
export const ARROW_TONES = [
  'green',
  'red',
  'orange',
  'indigo',
  'faint',
  'teal',
  'sky',
  'amber',
] as const
export type ArrowTone = (typeof ARROW_TONES)[number]

export const arrow = (tone: ArrowTone) => `url(#cw-arrow-${tone})`

export function Defs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden>
      <defs>
        {ARROW_TONES.map((tone) => (
          <marker
            key={tone}
            id={`cw-arrow-${tone}`}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5.5"
            markerHeight="5.5"
            orient="auto-start-reverse"
          >
            <path d="M0.5,1 L9.5,5 L0.5,9 z" fill={`var(--ck-${tone})`} />
          </marker>
        ))}
      </defs>
    </svg>
  )
}
