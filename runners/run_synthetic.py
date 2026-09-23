"""Synthetic preset — the 3-class KG that actually exercises the pruning."""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from causalway.synthetic import generate, GROUND_TRUTH_EDGES, truth_adjacency
from runners.run_kg_discovery import build_context, DiscoveryContext

SYNTHETIC_TTL = os.path.join(_PROJECT_ROOT, "kgs", "ttls", "synthetic_clinic.ttl")


def build_synthetic_context(
    ttl_path: str = SYNTHETIC_TTL,
    n_patients: int = 500,
    n_hospitals: int = 20,
    max_therapies: int = 4,
    seed: int = 42,
    object_value: str = "range_class",
    relation_direction: str = "both",
) -> tuple[DiscoveryContext, np.ndarray]:
    """Generate, round-trip, and return ``(context, ground_truth_adj)``.

    The continuous frame is replaced with the true linear-SEM values (the
    discrete frame still comes from the round-tripped TTL).
    """
    synth = generate(
        n_patients=n_patients,
        n_hospitals=n_hospitals,
        max_therapies=max_therapies,
        seed=seed,
    )
    synth.write_ttl(ttl_path)

    ctx = build_context(
        ttl_path,
        object_value=object_value,
        relation_direction=relation_direction,
    )

    # Replace the integer-coded continuous frame with the true linear-SEM data,
    # aligned to the (kept) node order.
    cont = synth.continuous_df.reindex(columns=ctx.constraint_kept.names)
    ctx.continuous_df = cont

    truth_adj = truth_adjacency(ctx.constraint_kept.names)
    return ctx, truth_adj
