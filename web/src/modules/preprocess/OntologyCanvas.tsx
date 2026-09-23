import { useCallback, useEffect, useMemo } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  Panel as FlowPanel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import { useStore } from '../../stores/useStore'
import { ontologyNodeTypes } from '../../flow/cards'
import { edgeTypes } from '../../flow/FloatingEdge'
import { FitOnResize } from '../../flow/FitOnResize'
import { forceLayout, type ForceLink, type ForceNode } from '../../flow/forceLayout'
import { ringAngle, useLayoutSync } from '../../flow/useLayoutSync'
import { Badge, Button, Icon, Info, Panel } from '../../app/ui'

const CLASS_R = 250
const SAT_R = 195

// Card footprints, used as collision radii and to place a centre as a top-left
// corner.
//
// These are *estimated from the label*, not constants, and not measured from
// the DOM either — the seed is computed before React Flow has rendered
// anything, so there is nothing to measure. A fixed guess is what a schema with
// short class names lets you get away with: `flow/cards.tsx` gives every card a
// `min-w` and then lets it grow with its text, so an endpoint whose classes are
// called "Healthcare Indicator Calculation" draws them 269px wide against the
// 133px of a "Water Body" — and a collider sized for the latter lets the former
// overlap its neighbours. Coefficients calibrated against rendered `offsetWidth`
// on a 66-card canvas: the class line is 12.5px mono, the property line 11.5px
// mono, the sub-line 9.5px sans.
const CLASS_H = 42
const PROP_H = 40
const classWidth = (label: string) => Math.max(128, Math.round(7.7 * label.length + 26))
const propWidth = (prop: string, sub: string) =>
  Math.max(104, Math.round(Math.max(7.1 * prop.length, 4.8 * sub.length) + 22))

/**
 * Positions for the ontology canvas, relaxed by `forceLayout`.
 *
 * The graph handed to the simulation is the drawing itself: one body per class
 * card and per retained property card, a short stiff spring from each property
 * to its class (satellites orbit their owner), and a long slack spring for each
 * object property's relationship (joined classes end up near each other, and
 * unjoined ones are free to drift apart — which is the right picture, since a
 * dropped join really does stop relating them).
 *
 * The ring/fan arrangement this replaced is kept as the simulation's *starting*
 * placement. It is already a reasonable, deterministic, overlap-free guess, so
 * the simulation spends its ticks improving a layout rather than untangling a
 * pile at the origin.
 */
function ontologyPositions(nodesData: ReturnType<typeof useStore.getState>['nodes']) {
  const classSet = new Set<string>()
  nodesData.forEach((n) => {
    classSet.add(n.domain)
    if (n.kind === 'object') classSet.add(n.range)
  })
  const classes = [...classSet].sort()
  const nC = classes.length
  const centre: Record<string, { x: number; y: number }> = {}
  classes.forEach((c, i) => {
    const a = ringAngle(i, nC)
    const r = nC === 1 ? 0 : CLASS_R + Math.max(0, nC - 4) * 34
    centre[c] = { x: r * Math.cos(a), y: r * Math.sin(a) }
  })

  // A property's satellite card — data or object — is drawn only while it is a
  // retained candidate variable, exactly like a data property (Preprocess
  // revision 2). An object property's relationship (the edge below) is a
  // separate, independent axis and stays drawn regardless.
  const byClass: Record<string, typeof nodesData> = {}
  nodesData
    .filter((n) => !n.excluded)
    .forEach((n) => (byClass[n.domain] = [...(byClass[n.domain] ?? []), n]))

  const bodies: ForceNode[] = []
  const springs: ForceLink[] = []
  const widthOf: Record<string, number> = {}
  classes.forEach((c, i) => {
    const props = byClass[c] ?? []
    const cw = classWidth(c)
    widthOf[`c:${c}`] = cw
    bodies.push({
      id: `c:${c}`,
      w: cw,
      h: CLASS_H,
      seed: centre[c],
      charge: -1900,
      pad: 22,
    })
    // Fan this class's properties outward from the class, away from the centre
    // of the ring, so the simulation starts with satellites already sorted by
    // owner rather than interleaved.
    const base = nC === 1 ? -Math.PI / 2 : ringAngle(i, nC)
    const spread = nC === 1 ? Math.PI * 2 * ((props.length - 1) / props.length) : Math.PI * 0.95
    props.forEach((p, k) => {
      const a = props.length === 1 ? base : base - spread / 2 + (spread * k) / (props.length - 1)
      const pw = propWidth(p.prop, p.n_distinct === null ? p.range : `${p.range} · x distinct`)
      bodies.push({
        id: `p:${p.name}`,
        w: pw,
        h: PROP_H,
        seed: {
          x: centre[c].x + SAT_R * Math.cos(a),
          y: centre[c].y + SAT_R * Math.sin(a),
        },
        charge: -620,
        pad: 12,
      })
      // Rest lengths are measured edge-to-edge, not centre-to-centre: a fixed
      // centre distance parks a satellite *inside* a wide class card.
      springs.push({
        source: `c:${c}`,
        target: `p:${p.name}`,
        distance: 105 + cw / 2 + pw / 2,
        strength: 0.85,
      })
    })
  })
  nodesData
    .filter((n) => n.kind === 'object')
    .forEach((n) => {
      const gap = (widthOf[`c:${n.domain}`] ?? 128) / 2 + (widthOf[`c:${n.range}`] ?? 128) / 2
      springs.push({
        source: `c:${n.domain}`,
        target: `c:${n.range}`,
        // A dropped join is a weaker claim about these two classes belonging in
        // one row, and the layout says so: the spring is longer and slacker.
        // Only a little, though — the dashed stroke already carries that signal,
        // and a range class flung to the edge of the canvas (which is what a
        // much slacker spring produces, since a value-only class has no
        // satellites of its own to anchor it) inflates the bounding box every
        // `fitView` then has to zoom out to cover.
        distance: gap + (n.join_excluded ? 350 : 270),
        strength: n.join_excluded ? 0.26 : 0.45,
      })
    })

  return { classes, byClass, positions: forceLayout(bodies, springs) }
}

/**
 * W31 — which component each *class* is in, read off the nodes.
 *
 * The server sends a component per node, keyed on the node's domain, and an
 * object property's range is in the same component as its domain by
 * construction (the relationship is the edge that puts them there). So both
 * ends of every node place a class, and a class with no nodes of its own —
 * something that is only ever a range — is placed by whatever points at it.
 */
function classComponents(nodesData: ReturnType<typeof useStore.getState>['nodes']) {
  const out: Record<string, number> = {}
  nodesData.forEach((n) => {
    if (n.component === null) return
    out[n.domain] = n.component
    if (n.kind === 'object') out[n.range] = n.component
  })
  return out
}

/** The same thing keyed by React Flow node id, for `fitView({ nodes })`. */
function cardComponents(
  nodesData: ReturnType<typeof useStore.getState>['nodes'],
  ofClass: Record<string, number>,
) {
  const out: Record<string, number> = {}
  Object.entries(ofClass).forEach(([cls, i]) => (out[`c:${cls}`] = i))
  nodesData.forEach((n) => {
    if (n.component !== null) out[`p:${n.name}`] = n.component
  })
  return out
}

function Canvas() {
  const nodesData = useStore((s) => s.nodes)
  const toggleNode = useStore((s) => s.toggleNode)
  const toggleJoin = useStore((s) => s.toggleJoin)
  const setComponent = useStore((s) => s.setComponent)
  const component = useStore((s) => s.component)
  const autoComponent = useStore((s) => s.autoComponent)
  const nComponents = useStore((s) => s.components.length)
  const { place, commit } = useLayoutSync('ontology')
  const { fitView, getNodes } = useReactFlow()

  const effective = component ?? autoComponent
  const ofClass = useMemo(() => classComponents(nodesData), [nodesData])
  const ofCard = useMemo(() => cardComponents(nodesData, ofClass), [nodesData, ofClass])
  // Dimming is off entirely on a connected schema: there is nothing to
  // distinguish, and greying nothing still costs a re-render per node.
  const scoping = nComponents > 1 && effective !== null

  // Kept apart from `place` on purpose: the simulation depends only on the
  // graph's shape, while `place` changes on every drag commit. Folding the two
  // into one memo would re-run the simulation after every drag — harmless,
  // since it is deterministic, but pure waste.
  const relaxed = useMemo(() => ontologyPositions(nodesData), [nodesData])

  const { seedNodes, seedEdges } = useMemo(() => {
    const { classes, byClass, positions } = relaxed
    const at = (id: string) => positions[id] ?? { x: 0, y: 0 }
    /*
     * Out of scope is drawn faint, not hidden. The whole T-Box is the picture
     * that explains *why* there are components in the first place — the user
     * needs to see the seventy islands to understand that the query covers one
     * of them — and hiding the rest would turn a legible diagram into a claim
     * that the ontology is smaller than it is (W31).
     */
    const dimOf = (cls: string) =>
      scoping && ofClass[cls] !== effective ? { opacity: 0.22 } : undefined

    const rfNodes: Node[] = []
    classes.forEach((c) => {
      const props = byClass[c] ?? []
      rfNodes.push({
        id: `c:${c}`,
        type: 'classCard',
        position: place(`c:${c}`, at(`c:${c}`)),
        style: dimOf(c),
        data: {
          label: c,
          nProps: nodesData.filter((n) => n.kind === 'data' && !n.excluded && n.domain === c)
            .length,
          nObjects: nodesData.filter(
            (n) => n.kind === 'object' && !n.join_excluded && n.domain === c,
          ).length,
        },
        draggable: true,
      })
      props.forEach((p) => {
        rfNodes.push({
          id: `p:${p.name}`,
          type: 'propCard',
          position: place(`p:${p.name}`, at(`p:${p.name}`)),
          style: dimOf(c),
          data: {
            label: p.prop,
            range: p.range,
            numeric: p.value_hint === 'numeric',
            kind: p.kind,
            distinct: p.n_distinct,
            full:
              p.kind === 'object'
                ? `${p.name} — value is the related ${p.range}'s local name`
                : `${p.name} — ${p.range}`,
          },
          draggable: true,
        })
      })
    })

    const rfEdges: Edge[] = []
    nodesData
      .filter((n) => !n.excluded)
      .forEach((n) =>
        rfEdges.push({
          id: `t:${n.name}`,
          source: `c:${n.domain}`,
          target: `p:${n.name}`,
          type: 'floating',
          selectable: false,
          style: dimOf(n.domain),
          data: { tone: 'faint', head: false, width: 1, dim: true },
        }),
      )
    // The relationship edge is the join: always drawn, solid when joined and
    // dashed when not. Clicking it toggles the join — which under W26 is the
    // same thing as switching the property between its two roles, since a
    // dropped join is what makes it a variable. The curation panel's three-way
    // selector is the explicit form of the same choice.
    nodesData
      .filter((n) => n.kind === 'object')
      .forEach((n) => {
        rfEdges.push({
          id: `o:${n.name}`,
          source: `c:${n.domain}`,
          target: `c:${n.range}`,
          type: 'floating',
          style: dimOf(n.domain),
          data: {
            tone: 'green',
            label: n.prop,
            width: 2,
            dashed: n.join_excluded,
            dim: n.join_excluded,
            title: n.join_excluded
              ? `${n.domain} —${n.prop}→ ${n.range}\nNot joined — ${n.range}'s own properties cannot ride into a ${n.domain} row through this relation.\nClick to join it.`
              : `${n.domain} —${n.prop}→ ${n.range}\nThe join this relation contributes to the flat join.\nClick to remove it.`,
          },
        })
      })

    return { seedNodes: rfNodes, seedEdges: rfEdges }
    // `place` changes with the stored layout, which is exactly when we want to rebuild.
  }, [nodesData, place, relaxed, scoping, effective, ofClass])

  const [nodes, setNodes, onNodesChange] = useNodesState(seedNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(seedEdges)
  useEffect(() => setNodes(seedNodes), [seedNodes, setNodes])
  useEffect(() => setEdges(seedEdges), [seedEdges, setEdges])

  /** The cards of the component being joined — what a fit should frame. */
  const focus = useCallback(() => {
    if (!scoping) return null
    return getNodes()
      .filter((n) => ofCard[n.id] === effective)
      .map((n) => ({ id: n.id }))
  }, [scoping, effective, ofCard, getNodes])

  /*
   * Zoom to the chosen component (W31). This canvas covers 70 components at
   * about 15% zoom, and at that scale "the lit one is component 16" is not
   * something the user can actually read off the picture.
   *
   * A button rather than an automatic fly-to on every component change, for
   * one bad reason and one good one. The bad one: changing the component
   * rebuilds `seedNodes` (the dim styles change), React Flow re-adopts every
   * card as *unmeasured*, and a `fitView` anywhere in that window returns true
   * and moves nothing — and neither `useNodesInitialized` nor
   * `nodes === seedNodes` identifies the far edge of the window reliably,
   * because both are already true on the renders where the measurements still
   * describe the outgoing cards. The good one: W13 makes the user's own
   * arrangement of this canvas authoritative, and moving their camera without
   * being asked sits badly beside that. On demand it is unambiguous, and the
   * `FitOnResize` focus below already keeps a panel reflow from quietly zooming
   * them back out to all seventy.
   */
  const fitComponent = useCallback(() => {
    const ids = focus()
    if (!ids || ids.length === 0) return
    void fitView({ nodes: ids, padding: 0.3, maxZoom: 1.1, duration: 320 })
  }, [focus, fitView])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={ontologyNodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeDragStop={commit}
      onNodeClick={(_, n) => {
        if (n.type === 'propCard') return void toggleNode(n.id.slice(2))
        // W31: a class card is the handle on its component — the diagram is
        // where the components are legible, so it is also where picking one
        // should be possible. Only when there is a choice to make.
        const i = ofClass[n.id.slice(2)]
        if (n.type === 'classCard' && nComponents > 1 && i !== undefined && i !== effective) {
          void setComponent(i)
        }
      }}
      onEdgeClick={(_, e) => e.id.startsWith('o:') && toggleJoin(e.id.slice(2))}
      nodesConnectable={false}
      elementsSelectable
      fitView
      fitViewOptions={{ padding: 0.18, maxZoom: 1.1 }}
      minZoom={0.15}
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--ck-grid)" />
      {scoping && (
        <FlowPanel position="top-left">
          <button
            onClick={fitComponent}
            title={`Zoom to component ${effective} — the one being joined. The rest of the T-Box stays drawn, faint, where it is.`}
            className="rounded-lg border border-teal/40 bg-surface/90 px-2 py-1 text-[11px] font-medium text-teal shadow-sm backdrop-blur transition-colors hover:bg-teal/10"
          >
            Zoom to component {effective}
          </button>
        </FlowPanel>
      )}
      <Controls showInteractive={false} position="bottom-right" />
      <FitOnResize focus={focus} />
    </ReactFlow>
  )
}

export function OntologyCanvas() {
  const resetLayout = useStore((s) => s.resetLayout)
  const nodesData = useStore((s) => s.nodes)
  const components = useStore((s) => s.components)
  const component = useStore((s) => s.component)
  const autoComponent = useStore((s) => s.autoComponent)
  const nExcluded = nodesData.filter((n) => n.excluded).length
  const effective = component ?? autoComponent
  const scoping = components.length > 1 && effective !== null

  return (
    <Panel
      step="1·2"
      title="Ontology — classes, properties & relationships"
      flush
      grow
      resizeId="ontology-canvas"
      right={
        <div className="flex items-center gap-1.5">
          {scoping && (
            <Badge
              tone="teal"
              title={`Lit: component ${effective}, the one being joined. The rest of the T-Box is drawn faint, not hidden — the islands are the reason there is a choice to make.`}
            >
              component {effective} of {components.length}
            </Badge>
          )}
          {nExcluded > 0 && <Badge tone="orange">{nExcluded} excluded</Badge>}
          <Button size="sm" variant="ghost" onClick={() => resetLayout('ontology')}>
            <Icon name="refresh" className="text-[13px]" />
            Re-layout
          </Button>
          <Info>
            The induced T-Box, not the causal search space. One card per class, one satellite card
            per retained property (green for object, amber/sky for data), one directed labelled
            edge per object property — the card is the property as a candidate variable, the edge
            is its relationship being joined, and the two are curated independently.
            <br />
            When the class graph falls into several <strong>connected components</strong>, only
            one is joined at a time, and it is the one drawn at full strength — the others stay
            visible but faint, because the islands are what explain why there is a choice to
            make at all. Click any class card to switch the join to its component, and{' '}
            <strong>Zoom to component</strong> to get close enough to read it (W31).
            <br />
            Positions are a <strong>force-directed</strong> relaxation: properties orbit the
            class that owns them, joined classes are pulled together and unjoined ones drift
            apart. It runs once and then stops — drag anything and your arrangement is what is
            saved. <strong>Re-layout</strong> throws your positions away and runs it again, which
            for a given ontology always produces the same picture. Assumption
            1's admissible edge set is still computed — it is the pruning rate in the curation
            panel — it just isn't what's drawn here (Plan 3 §3.4, W7).
          </Info>
        </div>
      }
    >
      {/* React Flow's root is `height: 100%`, and a percentage height does not resolve
          against a parent whose own height came from `flex: 1 1 0%` — the canvas
          collapses to 0 and the graph vanishes. Absolute-filling a relative wrapper is
          the reliable way to hand a flex-sized box to a library that wants percentages. */}
      <div className="relative min-h-0 w-full flex-1">
        <div className="absolute inset-0">
          <ReactFlowProvider>
            <Canvas />
          </ReactFlowProvider>
        </div>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-t border-linesoft px-3 py-2 text-[11px] text-faint">
        <Legend colour="var(--ck-teal)" label="class" />
        <Legend colour="var(--ck-sky)" label="numeric data property" />
        <Legend colour="var(--ck-amber)" label="string / date data property" />
        <Legend colour="var(--ck-green)" label="object property — as a causal variable (card)" />
        <span className="flex items-center gap-1.5">
          <svg width="20" height="8" aria-hidden>
            <path d="M1 4h13" stroke="var(--ck-green)" strokeWidth="2" />
            <path d="M13 1.5 18 4l-5 2.5z" fill="var(--ck-green)" />
          </svg>
          object property — as a relationship (edge), dashed when not joined
        </span>
        <span className="ml-auto">
          Drag any card. Click a data-property card to drop its column, or a green relationship
          edge to switch that object property between its two roles — it is a relationship{' '}
          <em>or</em> a causal variable, never both.
          {scoping && ' Click a class card to join its component instead.'}
        </span>
      </div>
    </Panel>
  )
}

function Legend({ colour, label }: { colour: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block h-3 w-4 rounded-[4px] border"
        style={{
          borderColor: colour,
          background: `color-mix(in srgb, ${colour} 16%, transparent)`,
        }}
      />
      {label}
    </span>
  )
}
