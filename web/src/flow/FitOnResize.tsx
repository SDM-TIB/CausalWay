import { useEffect, useRef } from 'react'
import { useReactFlow } from '@xyflow/react'

/**
 * React Flow's `fitView` prop only runs on init, so collapsing a sidebar (which
 * changes the canvas width) would leave the graph drifting off-centre. Refit on
 * container resize instead — rendered as a child of <ReactFlow> so it can find
 * the container without threading a ref through.
 */
export function FitOnResize({
  padding = 0.18,
  focus,
}: {
  padding?: number
  /**
   * The cards a refit should frame, when the canvas has a subject smaller than
   * itself — the ontology canvas's chosen connected component (W31). Without
   * it, any resize refits to *everything*, which on a 70-component schema means
   * the panel below growing by a row silently zooms the user back out from the
   * component they just picked to the whole T-Box at 15%.
   *
   * A getter rather than an array, read through a ref, so the observer is
   * installed once instead of being town down and rebuilt on every curation
   * change. Returning null or an empty list means "fit everything", as before.
   */
  focus?: () => { id: string }[] | null
}) {
  const { fitView } = useReactFlow()
  const ref = useRef<HTMLDivElement>(null)
  const focusRef = useRef(focus)
  focusRef.current = focus

  useEffect(() => {
    const el = ref.current?.closest('.react-flow') as HTMLElement | null
    if (!el) return
    let timer = 0
    const ro = new ResizeObserver(() => {
      window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        const nodes = focusRef.current?.() ?? null
        void fitView({
          padding,
          maxZoom: 1.1,
          duration: 220,
          ...(nodes && nodes.length > 0 ? { nodes } : {}),
        })
      }, 140)
    })
    ro.observe(el)
    return () => {
      ro.disconnect()
      window.clearTimeout(timer)
    }
  }, [fitView, padding])

  return <div ref={ref} className="hidden" />
}
