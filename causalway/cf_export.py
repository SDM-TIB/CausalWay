"""Counterfactual *worlds* — the service's board output — as cw: counterfactual queries.

A world is one unit, one joint intervention over one or more causal variables,
and — for every non-intervened node of the model — the value that node would
most likely have taken, beside the value it actually had.  In the cw: model
that is nothing but a ``cw:CounterfactualQuery`` (the world *is* the query: its
IRI names the hypothetical) with one ``cw:CounterfactualAnswer`` per node.  So
this module owns no mapping: :func:`world_to_query` turns a world into a
:class:`~causalway.queries.Query` and its :class:`~causalway.queries.Answer`s,
and :func:`worlds_to_rdf` hands them to :mod:`causalway.query_export`.  A world
exported here and the same question stored through ``store_query`` are the same
resources.

What a world says
-----------------
* **``cw:factualValue``** beside every ``cw:predictedValue``.  "dosage would
  have been 40" means nothing until "dosage was 25" is beside it.  The
  contrast between them is deliberately not minted: it is a subtraction.
* **Every non-intervened node**, not just nominated outcomes — a world is
  interesting because it says what *else* would have been different.  An
  intervened node gets no answer: its value is the intervention.
* **``cw:probability``** on a categorical answer (the mass of the arg-max
  level; the full distribution is not exported) and **``cw:std``** on a
  continuous one.

Units without a name
--------------------
A world is normally about a named entity of the source KG (``cw:aboutEntity``),
whose factual state is in that KG.  The counterfactual module can also pose a
*hypothetical* unit — values typed by hand, no entity behind them — which is
what makes an imported model usable without its training data.  Such a world
has no ``cw:aboutEntity``; its observed state is its ``cw:hasCondition``s, one
per modelled variable, and they are part of its IRI, so two hypothetical units
differing only in what was observed are two queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rdflib import Graph

from .queries import Answer, Condition, Intervention, Query
from .query_export import queries_to_json, queries_to_rdf

__all__ = ["CounterfactualWorld", "world_to_query", "worlds_to_json", "worlds_to_rdf"]


@dataclass
class CounterfactualWorld:
    """One unit under one joint intervention, with every node's two values.

    ``interventions``, ``factual``, ``counterfactual``, ``distribution`` and
    ``std`` are all keyed by **node name** (the SCM column / ``CausalNode``
    label).  ``entity`` is ``None`` for a hypothetical unit, whose ``factual``
    values are then the observed state that was typed in.  ``distribution`` is
    accepted in full (``{node: {level: probability}}``) but only the mass of
    the reported level survives into RDF.
    """

    interventions: dict                              # node name -> value set by do()
    counterfactual: dict                             # node name -> value it would have had
    entity: Optional[str] = None                     # None => hypothetical unit
    factual: dict = field(default_factory=dict)      # node name -> value it had
    distribution: dict = field(default_factory=dict) # node name -> {level: probability}
    std: dict = field(default_factory=dict)          # node name -> sd across draws
    label: Optional[str] = None
    evaluated_time: Optional[str] = None             # ISO-8601; "now" at export when unset


def world_to_query(world: CounterfactualWorld, graph_id: Optional[str] = None) -> tuple:
    """``(Query, [Answer, ...])`` — the cw: counterfactual query this world is.

    ``graph_id`` is the graph the query is posed on, which is part of its IRI;
    the exporters pass the evaluating model's graph.
    """
    conditions = ([] if world.entity else
                  [Condition(node=node, value=value)
                   for node, value in sorted(world.factual.items()) if value is not None])
    query = Query(
        kind="counterfactual",
        interventions=[Intervention(node=node, value=value)
                       for node, value in sorted(world.interventions.items())],
        conditions=conditions,
        entity=world.entity or None,
        graph_id=graph_id,
        label=world.label,
    )
    answers = []
    for node in sorted(world.counterfactual):
        if node in world.interventions:
            continue
        predicted = world.counterfactual[node]
        answers.append(Answer(
            kind="counterfactual", target=node, entity=world.entity,
            predicted=predicted,
            distribution=world.distribution.get(node) or None,
            std=world.std.get(node),
            factual=world.factual.get(node),
            evaluated_time=world.evaluated_time,
        ))
    return query, answers


def worlds_to_json(worlds, model, *, dtypes: Optional[dict] = None) -> dict:
    """The flat document ``query_mapping.rml.ttl`` reads, for these worlds."""
    gid = model.spec.ocg.gid
    return queries_to_json([world_to_query(w, gid) for w in worlds], model, dtypes=dtypes)


def worlds_to_rdf(worlds, model, *, dtypes: Optional[dict] = None,
                  graph: Optional[Graph] = None, with_vocabulary: bool = False,
                  workdir: Optional[str] = None) -> Graph:
    """Serialise counterfactual worlds as cw: counterfactual queries and answers.

    ``model`` is the model that evaluated them (``cw:evaluatedBy``); its graph is
    the one the queries are posed on.  Never asserts ``entity property value``,
    so the output merges into the source KG without contradicting it.
    """
    gid = model.spec.ocg.gid
    return queries_to_rdf([world_to_query(w, gid) for w in worlds], model, dtypes=dtypes,
                          graph=graph, with_vocabulary=with_vocabulary, workdir=workdir)
