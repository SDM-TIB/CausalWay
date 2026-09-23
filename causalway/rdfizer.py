"""Run SDM-RDFizer over a JSON document and an RML mapping, in one call.

Two exports go through this: the ontological causal graph
(:mod:`causalway.result` + ``ocg_mapping.rml.ttl``) and the counterfactual worlds
(:mod:`causalway.cf_export` + ``cf_mapping.rml.ttl``).  They share every quirk of
the engine, so they share this module rather than each carrying a copy.

The quirks worth knowing here — the mapping files document the rest:

* The engine is invoked as a **subprocess**, not as ``rdfizer.semantify(...)``.
  It resolves a mapping's ``rml:source`` against the *process* working
  directory, so an in-process call would need a global ``os.chdir``, which is
  not safe in the threaded web service; a fresh interpreter also drops the
  large module-level state it accumulates between runs.
* A literal it emits has both ``"`` and ``'`` rewritten to ``\\'``,
  irreversibly.  :func:`check_quote_free` refuses to export one rather than let
  a knowledge graph quietly disagree with the object it describes.
* An *empty* reference value is not treated as absent: the engine writes the
  literal ``"None"``.  Callers must omit a key entirely, never set it to ``""``.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from rdflib import Graph

__all__ = ["materialise", "check_quote_free"]

_CONFIG = """\
[default]
main_directory: {work}

[datasets]
number_of_datasets: 1
output_folder: ${{default:main_directory}}/out
remove_duplicate: yes
all_in_one_file: yes
name: {name}
enrichment: no
ordered: yes
output_format: ntriples

[dataset1]
name: {name}
mapping: ${{default:main_directory}}/{mapping}
"""


def check_quote_free(payload: dict, iri_fields: set) -> None:
    """Raise ``ValueError`` naming the field if any literal carries a quote.

    ``iri_fields`` are exempt: they travel through ``rr:template`` +
    ``rr:termType rr:IRI``, which the engine does not touch (and an IRI cannot
    contain a quote anyway).
    """
    for section, records in payload.items():
        if not isinstance(records, list):
            continue
        for record in records:
            for key, value in record.items():
                if key in iri_fields or not isinstance(value, str):
                    continue
                if '"' in value or "'" in value:
                    raise ValueError(
                        f"Cannot export {section}[].{key} = {value!r} through "
                        "SDM-RDFizer: it rewrites both \" and ' to \\' in a "
                        "literal, irreversibly. Rename the property or strip "
                        "the quote before exporting."
                    )


def materialise(payload: dict, mapping_path: str, *, source_name: str,
                graph: Optional[Graph] = None,
                workdir: Optional[str] = None) -> Graph:
    """Apply ``mapping_path`` to ``payload`` and return the resulting triples.

    ``source_name`` must equal the ``rml:source`` the mapping names (e.g.
    ``"ocg.json"``).  Pass ``graph`` to accumulate into an existing store, and
    ``workdir`` to keep the generated JSON, config and N-Triples on disk for
    inspection instead of in a directory deleted on return — the only way to see
    what the engine actually received when a mapping change misbehaves.
    """
    g = Graph() if graph is None else graph
    temp = None
    try:
        if workdir is None:
            temp = work = tempfile.mkdtemp(prefix="cw-rdfizer-")
        else:
            work = os.path.abspath(workdir)
            os.makedirs(work, exist_ok=True)

        mapping_name = os.path.basename(mapping_path)
        with open(os.path.join(work, source_name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        shutil.copyfile(mapping_path, os.path.join(work, mapping_name))
        with open(os.path.join(work, "config.ini"), "w", encoding="utf-8") as fh:
            fh.write(_CONFIG.format(
                work=work, name=os.path.splitext(source_name)[0],
                mapping=mapping_name,
            ))

        proc = subprocess.run(
            [sys.executable, "-m", "rdfizer", "-c", "config.ini"],
            cwd=work, capture_output=True, text=True,
        )
        produced = sorted(glob.glob(os.path.join(work, "out", "*")))
        if proc.returncode != 0 or not produced:
            raise RuntimeError(
                f"SDM-RDFizer failed to apply {mapping_name} "
                f"(exit {proc.returncode}).\nstdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        for path in produced:
            g.parse(path, format="nt")
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    return g
