"""Parse a TTL/NT knowledge graph into an :class:`OntologySchema`.

The schema captures the T-Box (classes, data/object properties, domains, ranges,
labels, comments) and materialises the class-level schema graph.  Declared
information is read first; when ``infer_missing=True`` properties without a
declaration are inferred from the A-Box (needed for KGs without a T-Box).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD

from ._util import is_vocab, local_name

RDF_TYPE = RDF.type


@dataclass
class OntologySchema:
    """Class-level view of a knowledge graph (the ``T-Box``).

    Attributes
    ----------
    graph: the parsed rdflib graph.
    classes: set of class URIRefs.
    object_properties: mapping ``p -> [(domain, range)]``.
    data_properties: mapping ``p -> [(domain, datatype)]``.
    labels / comments: per-URI ``rdfs:label`` / ``rdfs:comment``.
    class_graph: MultiDiGraph whose nodes are classes and whose edges are the
        object properties connecting them (``domain -> range``, edge keyed by
        the object-property URI).
    inferred: set of properties whose declaration had to be inferred.
    """

    graph: Graph
    classes: set
    object_properties: dict
    data_properties: dict
    labels: dict
    comments: dict
    class_graph: nx.MultiDiGraph
    inferred: set = field(default_factory=set)
    conflicts: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, format: Optional[str] = None,
                  infer_missing: bool = True) -> "OntologySchema":
        g = Graph()
        g.parse(path, format=format)
        return cls.from_graph(g, infer_missing=infer_missing)

    @classmethod
    def from_endpoint(cls, url: str, infer_missing: bool = False,
                      timeout: int = 60) -> "OntologySchema":
        """Parse the T-Box of a SPARQL endpoint. ``infer_missing`` defaults to
        ``False`` here (unlike :meth:`from_file`) — the inference queries scan
        the whole A-Box, which is fine on a KG-sized file and hostile to a
        public endpoint. See :func:`causalway.sources.resolve_schema`, which
        this delegates to.
        """
        from .sources import resolve_schema
        return resolve_schema(url, infer_missing=infer_missing, timeout=timeout)

    @classmethod
    def from_source(cls, source, **kw) -> "OntologySchema":
        """Polymorphic entry point: file path, endpoint URL, or ``rdflib.Graph``.

        Thin wrapper over :func:`causalway.sources.resolve_schema` (imported
        lazily to avoid a circular import).
        """
        from .sources import resolve_schema
        return resolve_schema(source, **kw)

    @classmethod
    def from_graph(cls, graph: Graph, infer_missing: bool = True) -> "OntologySchema":
        classes = cls._declared_classes(graph)
        obj_props = cls._declared_properties(graph, RDF.type, owl=OWL.ObjectProperty)
        dat_props = cls._declared_properties(graph, RDF.type, owl=OWL.DatatypeProperty)

        # A property declared as both data and object (or vice versa) wins on the
        # side where it was declared; keep both but flag the overlap.
        labels = cls._literals(graph, RDFS.label)
        comments = cls._literals(graph, RDFS.comment)

        inferred: set = set()
        conflicts: list = []

        if infer_missing:
            inf_obj, inf_dat, inf_confl = cls._infer_properties(
                graph, classes, set(obj_props) | set(dat_props)
            )
            for p, drs in inf_obj.items():
                if p not in obj_props:
                    obj_props[p] = drs
                    inferred.add(p)
            for p, drs in inf_dat.items():
                if p not in dat_props:
                    dat_props[p] = drs
                    inferred.add(p)
            conflicts.extend(inf_confl)

        class_graph = cls._build_class_graph(classes, obj_props, labels)

        return cls(
            graph=graph,
            classes=classes,
            object_properties=obj_props,
            data_properties=dat_props,
            labels=labels,
            comments=comments,
            class_graph=class_graph,
            inferred=inferred,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------ #
    # Extraction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _declared_classes(graph: Graph) -> set:
        q = """
        SELECT DISTINCT ?c WHERE {
            { ?c a owl:Class } UNION { ?c a rdfs:Class }
        }
        """
        classes = set()
        for row in graph.query(q, initNs={"owl": OWL, "rdfs": RDFS}):
            c = row.c
            if isinstance(c, URIRef) and not is_vocab(c):
                classes.add(c)
        return classes

    @staticmethod
    def _declared_properties(graph: Graph, rdf_type, owl) -> dict:
        q = """
        SELECT ?p ?d ?r WHERE {
            ?p a <OwlType> .
            OPTIONAL { ?p rdfs:domain ?d }
            OPTIONAL { ?p rdfs:range ?r }
        }
        """.replace("OwlType", str(owl))
        out: dict = {}
        for row in graph.query(q, initNs={"rdfs": RDFS}):
            p = row.p
            if not isinstance(p, URIRef) or is_vocab(p):
                continue
            d = row.d if isinstance(row.d, URIRef) and not is_vocab(row.d) else None
            r = row.r
            if r is not None and not isinstance(r, URIRef):
                r = None
            if d is None or r is None:
                # Cannot form a (domain, range) node; skip but keep the property name.
                continue
            out.setdefault(p, []).append((d, r))
        return out

    @staticmethod
    def _literals(graph: Graph, predicate) -> dict:
        out: dict = {}
        for s, o in graph.subject_objects(predicate):
            if isinstance(s, URIRef) and isinstance(o, Literal):
                out[s] = str(o)
        return out

    @staticmethod
    def _build_class_graph(classes, object_properties, labels) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        g.add_nodes_from(classes)
        for p, drs in object_properties.items():
            for (d, r) in drs:
                if d not in classes or r not in classes:
                    continue
                g.add_edge(d, r, key=p, property=p, label=labels.get(p, local_name(p)))
        return g

    # ------------------------------------------------------------------ #
    # Inference (for KGs without a T-Box)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_properties(graph: Graph, classes, declared) -> tuple:
        obj_props: dict = {}
        dat_props: dict = {}
        conflicts: list = []

        predicates = {p for p in graph.predicates() if isinstance(p, URIRef)}
        for p in predicates - declared:
            if is_vocab(p):
                continue
            objects = list(graph.objects(subject=None, predicate=p))
            if not objects:
                continue
            domains = OntologySchema._infer_domains(graph, p, classes)
            if not domains:
                continue
            if all(isinstance(o, Literal) for o in objects):
                datatype = OntologySchema._infer_datatype(objects)
                for d in domains:
                    dat_props.setdefault(p, []).append((d, datatype))
            elif any(isinstance(o, URIRef) for o in objects):
                ranges = OntologySchema._infer_ranges(graph, p, classes)
                if not ranges:
                    continue
                for d in domains:
                    for r in ranges:
                        obj_props.setdefault(p, []).append((d, r))
            else:
                conflicts.append((p, "mixed literal and IRI objects; skipped"))
        return obj_props, dat_props, conflicts

    @staticmethod
    def _infer_domains(graph: Graph, p: URIRef, classes) -> list:
        q = "SELECT DISTINCT ?C WHERE { ?s <Prop> ?o . ?s a ?C }".replace("Prop", str(p))
        domains = set()
        for row in graph.query(q):
            c = row.C
            if isinstance(c, URIRef) and not is_vocab(c):
                domains.add(c)
        return sorted(domains)

    @staticmethod
    def _infer_ranges(graph: Graph, p: URIRef, classes) -> list:
        q = "SELECT DISTINCT ?C WHERE { ?s <Prop> ?o . ?o a ?C }".replace("Prop", str(p))
        ranges = set()
        for row in graph.query(q):
            c = row.C
            if isinstance(c, URIRef) and not is_vocab(c):
                ranges.add(c)
        return sorted(ranges)

    @staticmethod
    def _infer_datatype(objects) -> URIRef:
        counter = Counter()
        for o in objects:
            if isinstance(o, Literal) and o.datatype is not None:
                counter[o.datatype] += 1
            else:
                counter[XSD.string] += 1
        return counter.most_common(1)[0][0] if counter else XSD.string

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        n_obj = sum(len(v) for v in self.object_properties.values())
        n_dat = sum(len(v) for v in self.data_properties.values())
        return {
            "n_classes": len(self.classes),
            "n_object_properties": len(self.object_properties),
            "n_object_domain_range": n_obj,
            "n_data_properties": len(self.data_properties),
            "n_data_domain_range": n_dat,
            "n_inferred": len(self.inferred),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"OntologySchema(classes={s['n_classes']}, "
            f"object_properties={s['n_object_properties']}, "
            f"data_properties={s['n_data_properties']})"
        )
