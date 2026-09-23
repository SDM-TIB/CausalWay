"""Derive LLM ``var_meta`` / ``pair_meta`` from an *external* KG endpoint.

``causalway.llm_meta`` builds ``var_meta`` / ``pair_meta`` from the causal KG's
own T-Box (``rdfs:label`` / ``rdfs:comment``). This module builds the same two
shapes from a *different*, general-purpose knowledge base (Wikidata by
default) that the discovery columns have been manually mapped onto, e.g.::

    entity_map = {
        'Hospital.airPollution': 'Q131123',
        'Patient.age': 'Q185836',
        ...
    }

For each mapped column, :func:`build_var_meta_from_kg` fetches the entity's
1-hop neighborhood (predicate label -> object label) from the endpoint and
asks an LLM to summarize it into one or two sentences. For each mapped pair,
:func:`build_pair_meta_from_kg` looks for a relational path within 1 hop
(a direct edge between the two entities, or a shared 1-hop neighbor) and asks
an LLM to summarize *that*; a pair with no such path gets ``""``, per
``causalway.llm_meta.build_pair_meta``'s ``{(a, b): text}`` shape.

Either dict can be overridden per-column / per-pair with hand-written natural
-language text via ``text_meta`` (requirement (1)); the KG lookup covers
requirement (2). Both are meant to be passed straight into
``run_algorithm(..., var_meta=..., pair_meta=...)`` for the LLM-prior methods
(e.g. ``GES-Prior``).
"""

from __future__ import annotations

import re
import time
from typing import Optional

import rdflib
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from algs.ges_prior.learn_bn import _call_llm
from algs.ges_prior.llm_call import LLMClient

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# A public endpoint like query.wikidata.org throttles bursts (HTTP 429); a
# small fixed delay before each request plus one retry-with-backoff keeps an
# O(n^2) pair sweep from tripping that, without adding real latency for a
# handful of columns.
_QUERY_DELAY = 0.2
_QUERY_RETRIES = 2

# Wikidata requires a descriptive User-Agent on query.wikidata.org (unlabelled
# requests get a bare 403); see https://meta.wikimedia.org/wiki/User-Agent_policy.
_USER_AGENT = "CausalWay-api/0.1 (https://github.com/; causal-discovery research tool)"

_QID_RE = re.compile(r"^Q\d+$")

# rdfs:label / rdfs:comment on Wikidata are tagged "mul" (identical across all
# languages) rather than "en" whenever the label happens to match verbatim in
# English -- e.g. most proper nouns and many technical terms. Both must be
# accepted or such entities silently return no label at all.
def _lang_filter(var: str) -> str:
    return f'lang({var}) = "en" || lang({var}) = "mul"'

# No "PREFIX rdfs: ..." here: rdflib's default NamespaceManager already binds
# rdfs (and much else) on every Graph, and re-declaring it in the query text
# makes the SPARQLStore-prepended prologue clash with a "multiple prefix
# declarations" error on the server side. wd:/wikibase: are not auto-bound,
# so those still need declaring.
# Wikidata bookkeeping properties that connect almost any two articles (shared
# bibliography, Wikimedia-project housekeeping, images, ...) without implying
# a real-world relationship; excluded from the shared-1-hop-neighbor search so
# it doesn't surface one of these as if it were a substantive path.
_WIKIDATA_NOISY_PROPS = [
    "P1343",  # described by source
    "P910",   # topic's main category
    "P5008",  # on focus list of Wikimedia project
    "P18",    # image
    "P373",   # Commons category
    "P1424",  # topic's main template
    "P2670",  # has parts of the class
]

_PREFIXES = ""
_WIKIDATA_PREFIXES = (
    "PREFIX wd: <http://www.wikidata.org/entity/>\n"
    "PREFIX wikibase: <http://wikiba.se/ontology#>\n"
)

__all__ = [
    "WIKIDATA_ENDPOINT",
    "fetch_one_hop_neighbors",
    "fetch_relational_path",
    "build_var_meta_from_kg",
    "build_pair_meta_from_kg",
    "build_kg_meta",
]


def _entity_term(entity_id: str, endpoint: str) -> tuple[str, bool]:
    """Resolve a mapping value into a SPARQL term, plus whether it's a Wikidata QID.

    ``entity_id`` may be a full ``http(s)://`` IRI (any endpoint), or a bare
    ``Q<digits>`` Wikidata QID when ``endpoint`` points at Wikidata.
    """
    if entity_id.startswith("http://") or entity_id.startswith("https://"):
        return f"<{entity_id}>", False
    if "wikidata.org" in endpoint and _QID_RE.match(entity_id):
        return f"wd:{entity_id}", True
    raise ValueError(
        f"Cannot resolve entity id {entity_id!r} for endpoint {endpoint!r}; "
        "pass a full http(s):// IRI, or a bare 'Q<digits>' id against a "
        "Wikidata endpoint."
    )


def _endpoint_graph(endpoint: str, timeout: int) -> rdflib.Graph:
    store = SPARQLStore(endpoint, returnFormat="json", timeout=timeout,
                        headers={"User-Agent": _USER_AGENT})
    return rdflib.Graph(store=store)


def _label(row, var: str) -> Optional[str]:
    value = row[var]
    return str(value) if value is not None else None


def _run_query(graph: rdflib.Graph, query: str) -> list:
    """``graph.query`` with a fixed pre-request delay and one retry-with-backoff on 429."""
    for attempt in range(_QUERY_RETRIES + 1):
        time.sleep(_QUERY_DELAY)
        try:
            return list(graph.query(query))
        except Exception as e:
            if attempt < _QUERY_RETRIES and "429" in str(e):
                time.sleep(2.0 * (attempt + 1))
                continue
            raise


def fetch_one_hop_neighbors(entity_id: str, *, endpoint: str = WIKIDATA_ENDPOINT,
                            limit: int = 50, timeout: int = 30) -> list:
    """``[(predicate_label, object_label), ...]`` for ``entity_id``'s 1-hop neighborhood."""
    term, is_wikidata = _entity_term(entity_id, endpoint)
    prefixes = _WIKIDATA_PREFIXES if is_wikidata else _PREFIXES
    directclaim = "?prop wikibase:directClaim ?p ." if is_wikidata else ""
    prop_source = "?prop" if is_wikidata else "?p"
    query = f"""{prefixes}
SELECT ?propLabel ?objLabel WHERE {{
  {term} ?p ?obj .
  {directclaim}
  FILTER(isIRI(?obj))
  OPTIONAL {{ {prop_source} rdfs:label ?propLabel . FILTER({_lang_filter("?propLabel")}) }}
  OPTIONAL {{ ?obj rdfs:label ?objLabel . FILTER({_lang_filter("?objLabel")}) }}
}}
LIMIT {int(limit)}
"""
    graph = _endpoint_graph(endpoint, timeout)
    out = []
    for row in _run_query(graph, query):
        p_label, o_label = _label(row, "propLabel"), _label(row, "objLabel")
        if p_label and o_label:
            out.append((p_label, o_label))
    return out


def fetch_relational_path(entity_a: str, entity_b: str, *, endpoint: str = WIKIDATA_ENDPOINT,
                          timeout: int = 30) -> str:
    """Natural-language relational path between two entities within 1 hop, or ``""``.

    Tries a direct edge first (``A -[p]-> B`` or ``B -[p]-> A``); falls back to
    a shared 1-hop neighbor (``A -[p1]-> M <-[p2]- B``, ``M`` named in the
    text); returns ``""`` when neither exists.
    """
    term_a, wd_a = _entity_term(entity_a, endpoint)
    term_b, wd_b = _entity_term(entity_b, endpoint)
    is_wikidata = wd_a and wd_b
    prefixes = _WIKIDATA_PREFIXES if is_wikidata else _PREFIXES
    directclaim = "?prop wikibase:directClaim ?p ." if is_wikidata else ""
    prop_source = "?prop" if is_wikidata else "?p"

    graph = _endpoint_graph(endpoint, timeout)

    direct_query = f"""{prefixes}
SELECT ?propLabel ?dir WHERE {{
  {{ {term_a} ?p {term_b} . BIND("forward" AS ?dir) }}
  UNION
  {{ {term_b} ?p {term_a} . BIND("backward" AS ?dir) }}
  {directclaim}
  OPTIONAL {{ {prop_source} rdfs:label ?propLabel . FILTER({_lang_filter("?propLabel")}) }}
}}
LIMIT 1
"""
    for row in _run_query(graph, direct_query):
        p_label = _label(row, "propLabel")
        if not p_label:
            continue
        direction = _label(row, "dir")
        return (f"{entity_a} --[{p_label}]--> {entity_b}" if direction == "forward"
                else f"{entity_b} --[{p_label}]--> {entity_a}")

    prop_a_source = "?propA" if is_wikidata else "?pa"
    prop_b_source = "?propB" if is_wikidata else "?pb"
    directclaim_ab = (
        "?propA wikibase:directClaim ?pa . ?propB wikibase:directClaim ?pb ."
        if is_wikidata else ""
    )
    noisy_filter = ""
    if is_wikidata:
        noisy = ", ".join(f"wd:{pid}" for pid in _WIKIDATA_NOISY_PROPS)
        noisy_filter = f"FILTER(?propA NOT IN ({noisy}) && ?propB NOT IN ({noisy}))"
    shared_query = f"""{prefixes}
SELECT ?paLabel ?pbLabel ?midLabel WHERE {{
  {term_a} ?pa ?mid .
  {term_b} ?pb ?mid .
  FILTER(isIRI(?mid))
  {directclaim_ab}
  {noisy_filter}
  OPTIONAL {{ {prop_a_source} rdfs:label ?paLabel . FILTER({_lang_filter("?paLabel")}) }}
  OPTIONAL {{ {prop_b_source} rdfs:label ?pbLabel . FILTER({_lang_filter("?pbLabel")}) }}
  OPTIONAL {{ ?mid rdfs:label ?midLabel . FILTER({_lang_filter("?midLabel")}) }}
}}
LIMIT 1
"""
    for row in _run_query(graph, shared_query):
        pa_label, pb_label, mid_label = (
            _label(row, "paLabel"), _label(row, "pbLabel"), _label(row, "midLabel")
        )
        if pa_label and pb_label and mid_label:
            return f"{entity_a} --[{pa_label}]--> {mid_label} <--[{pb_label}]-- {entity_b}"

    return ""


def _summarize_var(llm_client, column: str, neighbors: list, max_facts: int = 15) -> str:
    if not neighbors:
        return ""
    facts = "; ".join(f"{p}: {o}" for p, o in neighbors[:max_facts])
    prompt = (
        f"In one or two sentences, summarize what the real-world entity behind "
        f"the variable '{column}' is, based only on these knowledge-graph facts "
        f"about it (predicate: object):\n{facts}\n"
        "Be concise and factual; do not invent information not present above."
    )
    return _call_llm(llm_client, prompt).strip()


def _summarize_pair(llm_client, col_a: str, col_b: str, path: str) -> str:
    if not path:
        return ""
    prompt = (
        f"In one sentence, describe the real-world relationship implied by this "
        f"knowledge-graph path between the entities behind variables '{col_a}' "
        f"and '{col_b}':\n{path}\n"
        "Be concise and factual; do not invent information not present above."
    )
    return _call_llm(llm_client, prompt).strip()


def build_var_meta_from_kg(
    entity_map: dict,
    *,
    text_meta: Optional[dict] = None,
    endpoint: str = WIKIDATA_ENDPOINT,
    llm_client=None,
    llm_model: str = "deepseek-v4-flash",
    limit: int = 50,
    timeout: int = 30,
) -> dict:
    """``{column: natural-language description}`` for every column in ``entity_map`` or ``text_meta``.

    ``text_meta`` entries are used verbatim (requirement (1) — hand-written
    text); every other column mapped in ``entity_map`` is summarized by an LLM
    from its 1-hop neighborhood on ``endpoint`` (requirement (2)).
    """
    text_meta = text_meta or {}
    if llm_client is None and any(c not in text_meta for c in entity_map):
        llm_client = LLMClient(model=llm_model)

    meta = {}
    for column, entity_id in entity_map.items():
        if column in text_meta:
            continue
        try:
            neighbors = fetch_one_hop_neighbors(entity_id, endpoint=endpoint, limit=limit,
                                                timeout=timeout)
            meta[column] = _summarize_var(llm_client, column, neighbors)
        except Exception as e:
            print(f"build_var_meta_from_kg: {column} ({entity_id}) failed "
                  f"({type(e).__name__}: {e}); leaving it out of var_meta.")
    meta.update(text_meta)
    return meta


def build_pair_meta_from_kg(
    entity_map: dict,
    *,
    text_meta: Optional[dict] = None,
    endpoint: str = WIKIDATA_ENDPOINT,
    llm_client=None,
    llm_model: str = "deepseek-v4-flash",
    timeout: int = 30,
    pairs: Optional[list] = None,
) -> dict:
    """``{(col_i, col_j): relationship description}`` for mapped column pairs.

    ``pairs`` restricts which unordered pairs are looked up (default: every
    pair of columns in ``entity_map``, i.e. O(n^2) SPARQL round-trips — pass
    e.g. ``constraint.allowed`` derived pairs to cut that down on a larger
    schema). ``text_meta`` overrides individual pairs (requirement (1));
    everything else comes from :func:`fetch_relational_path` (requirement
    (2)), and is ``""`` when no path exists within 1 hop.
    """
    text_meta = text_meta or {}
    columns = list(entity_map)
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(columns) for b in columns[i + 1:]]

    remaining = [
        (a, b) for a, b in pairs
        if (a, b) not in text_meta and (b, a) not in text_meta
    ]
    if llm_client is None and remaining:
        llm_client = LLMClient(model=llm_model)

    meta = {}
    for col_a, col_b in remaining:
        entity_a, entity_b = entity_map.get(col_a), entity_map.get(col_b)
        if entity_a is None or entity_b is None:
            meta[(col_a, col_b)] = ""
            continue
        try:
            path = fetch_relational_path(entity_a, entity_b, endpoint=endpoint, timeout=timeout)
            meta[(col_a, col_b)] = _summarize_pair(llm_client, col_a, col_b, path)
        except Exception as e:
            print(f"build_pair_meta_from_kg: ({col_a}, {col_b}) failed "
                  f"({type(e).__name__}: {e}); defaulting to \"\".")
            meta[(col_a, col_b)] = ""
    meta.update(text_meta)
    return meta


def build_kg_meta(
    entity_map: dict,
    *,
    var_text: Optional[dict] = None,
    pair_text: Optional[dict] = None,
    endpoint: str = WIKIDATA_ENDPOINT,
    llm_client=None,
    llm_model: str = "deepseek-v4-flash",
    pairs: Optional[list] = None,
) -> tuple:
    """Convenience wrapper: ``(var_meta, pair_meta)`` in one call."""
    var_meta = build_var_meta_from_kg(
        entity_map, text_meta=var_text, endpoint=endpoint,
        llm_client=llm_client, llm_model=llm_model,
    )
    pair_meta = build_pair_meta_from_kg(
        entity_map, text_meta=pair_text, endpoint=endpoint,
        llm_client=llm_client, llm_model=llm_model, pairs=pairs,
    )
    return var_meta, pair_meta
