import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from 'd3-force'
import type { SimulationLinkDatum, SimulationNodeDatum } from 'd3-force'
import type { XY } from './useLayoutSync'

/**
 * A force-directed *seed* for a canvas, not a live simulation.
 *
 * The three canvases seed their positions three different ways, and each says
 * something: module 2 rings (a ledger of competing claims with no agreed
 * direction), modules 3-4 layer (direction is settled, so depth is causal
 * depth), and the ontology canvas now relaxes (a T-Box has no direction and no
 * ordering — what it has is *adjacency*, and adjacency is what a force layout
 * draws). A ring could not: it fixed every class at an equal angle whatever its
 * degree, so two classes joined by a relationship could sit on opposite sides
 * of the circle with their edge crossing every satellite in between.
 *
 * Deliberately **one-shot**. The simulation is ticked to convergence here and
 * the result is handed over as a starting arrangement; nothing animates, and
 * the moment the user drags a card the stored layout wins (`useLayoutSync`,
 * W13). A live simulation would fight that: it would have to re-settle on every
 * curation click, rearranging a canvas the user had already organised, and
 * positions would only be committable once it cooled.
 *
 * Deterministic for a given graph. Node order is the server's (sorted), the
 * initial placement is the caller's seed rather than d3's own phyllotaxis, and
 * `randomSource` is a seeded generator instead of `Math.random` — d3 reaches
 * for randomness to jiggle exactly-coincident nodes apart, which is rare but
 * would otherwise make the same schema draw differently on each mount. The
 * memoised call site re-runs this on every layout commit, and an identical
 * result is what stops that being a visible twitch.
 */

export interface ForceNode {
  id: string
  /**
   * Card size. `w` drives the collision radius and both convert centre →
   * top-left. Pass the card's *rendered* width, not a constant: these cards
   * grow with their label, and a KG whose classes are called
   * "Healthcare Indicator Calculation" draws them nearly twice as wide as one
   * whose classes are called "Patient".
   */
  w: number
  h: number
  /** Where the simulation starts. A good seed converges faster and more stably. */
  seed: XY
  /** Repulsion. Class cards push harder so their satellites clear each other. */
  charge: number
  /** Extra padding around this card's collision circle. */
  pad?: number
}

export interface ForceLink {
  source: string
  target: string
  /** Rest length of the spring, in the same units as the seed positions. */
  distance: number
  strength: number
}

interface SimNode extends SimulationNodeDatum, ForceNode {}

/** mulberry32 — a two-line PRNG, so "deterministic" needs no dependency. */
function seededRandom(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Run the simulation to rest and return each node's **top-left** corner, which
 * is the coordinate React Flow wants (d3 works in centres).
 */
export function forceLayout(
  nodes: ForceNode[],
  links: ForceLink[],
  opts: { ticks?: number } = {},
): Record<string, XY> {
  if (nodes.length === 0) return {}
  const ticks = opts.ticks ?? 400

  const sim: SimNode[] = nodes.map((n) => ({ ...n, x: n.seed.x, y: n.seed.y, vx: 0, vy: 0 }))
  const ids = new Set(sim.map((n) => n.id))
  const edges: SimulationLinkDatum<SimNode>[] = links
    .filter((l) => ids.has(l.source) && ids.has(l.target))
    .map((l) => ({ ...l }) as unknown as SimulationLinkDatum<SimNode>)

  const simulation = forceSimulation(sim)
    .randomSource(seededRandom(0x0ca7f107))
    .force(
      'link',
      forceLink<SimNode, SimulationLinkDatum<SimNode>>(edges)
        .id((d) => d.id)
        .distance((l) => (l as unknown as ForceLink).distance)
        .strength((l) => (l as unknown as ForceLink).strength),
    )
    .force(
      'charge',
      forceManyBody<SimNode>().strength((d) => d.charge),
    )
    // Separation is driven by half the *width*, not by half the diagonal. A
    // card is always far wider than it is tall, so the diagonal both
    // under-separates the wide ones horizontally (the direction that actually
    // collides) and over-separates every card vertically, which inflates the
    // canvas for nothing. d3 has no rectangular collider; width + padding is
    // the honest circular approximation of a wide, short box.
    .force(
      'collide',
      forceCollide<SimNode>().radius((d) => d.w / 2 + (d.pad ?? 10)).iterations(2),
    )
    // forceX/forceY rather than forceCenter: centring by translation lets a
    // disconnected component drift off with nothing to stop it, and a T-Box
    // with two components is ordinary (`_schema_components`).
    .force('x', forceX<SimNode>(0).strength(0.035))
    .force('y', forceY<SimNode>(0).strength(0.035))
    .stop()

  simulation.tick(ticks)

  const out: Record<string, XY> = {}
  sim.forEach((n) => {
    out[n.id] = { x: (n.x ?? 0) - n.w / 2, y: (n.y ?? 0) - n.h / 2 }
  })
  return out
}
