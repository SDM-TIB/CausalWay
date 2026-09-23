"""Project bundles — export a session to a zip, import one back.

This service keeps nothing (Plan 3 W14): `PROJECTS` is a dict in one process
and a restart loses every project in it. The export menu has always been the
only durable copy, but it was a one-way street — you could download a project
and never put it back. A model took minutes to fit and could not survive lunch.

**One artifact, not two.** The user's original ask was for a *model* export, so
that a fitted model could be reused in modules 3 and 4. A model on its own
turns out not to be a useful thing to hand back: module 3 draws a board of
nodes that only the curated graph names, module 4 needs the node dtypes and
levels, and both want the layout so the board is not a pile of cards at the
origin. All of that is project state. Shipping a model zip *and* a project zip
that must be imported together, in the right order, with matching ids, is two
formats and a failure mode; one zip with a `model/` directory inside it is one
format. The model is optional — a project exported before it was fitted simply
has no `model/` — so the merge costs nothing to a user who only wanted the
curation back.

**Import always creates a new project.** Never overwrite: an import that
targets an existing project can destroy hours of unsaved work with one
mis-click, and the service has no undo, no history, and no second copy. A new
project is cheap and the old one is still there to compare against.

**No training data, either way.** The bundle carries the fitted mechanisms and
the manifest, never the rows. Those rows are the user's KG, and an artefact
meant for sharing that silently contains them turns every model export into a
data export. `materialisation.csv` *is* in the project zip — it always has
been, it is the frame the user is looking at, and it is what `what=project`
means — but `model/` adds nothing to that, and an import that re-attaches a
source re-materialises from the source rather than trusting a CSV.

**Re-attaching a source is verified.** The manifest records the source KG's
sha256. On re-attach the file is hashed and compared, and a mismatch is a hard
error rather than a warning: a model whose mechanisms were fitted on different
rows than the ones now being queried gives confidently wrong answers, and
nothing downstream can detect it. The override is to import the bundle without
a source and treat it as a fresh project.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from typing import Any, Optional

__all__ = [
    "BUNDLE_VERSION",
    "BundleError",
    "bundle_descriptor",
    "write_model_into",
    "read_bundle",
    "restore_into",
    "verify_source",
]

#: Bumped when the bundle's *layout* changes in a way an older reader would
#: misread. Not a version of the service — a project zip from a service that
#: has since grown three new panels still restores fine, it just leaves the new
#: panels empty, which is what `notes` is for.
#:
#: 2 — the CausalKG → CausalWay rename. A version-1 bundle is unreadable here
#: for a reason no descriptor check would catch on its own: its ``scm.pkl`` is
#: a joblib pickle whose mechanism classes are recorded under their old module
#: path (``causalkg.mechanisms.InvertibleClassifierFCM``), and that module no
#: longer exists, so ``joblib.load`` would die with a bare ModuleNotFoundError
#: somewhere deep in the restore. We deliberately do *not* ship a
#: ``sys.modules["causalkg"]`` alias to paper over it: aliasing a name that
#: exists nowhere else in the project is permanent cruft in exchange for
#: reading artifacts produced by a pre-release version. Fail early, say why.
BUNDLE_VERSION = "2"

#: What ``kind`` said before the rename. Recognised only to produce a better
#: error than "unexpected bundle kind".
_PRE_RENAME_KIND = "causalkg-project"

MODEL_DIR = "model"
DESCRIPTOR = "bundle.json"


class BundleError(ValueError):
    """A bundle that cannot be read, or can be read but should not be trusted."""


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def bundle_descriptor(p, *, has_model: bool, versions: dict) -> dict:
    """The small file a reader looks at *before* unpacking anything else."""
    return {
        "bundle_version": BUNDLE_VERSION,
        "kind": "causalway-project",
        "project": {"id": p.id, "name": p.name, "created_at": p.created_at},
        "has_model": has_model,
        # What produced it. Checked on import against what is installed, because
        # a pickled sklearn estimator loaded under a different sklearn can
        # misbehave silently rather than raise.
        "versions": versions,
        "source_kg": _source_fingerprint(p),
        "contains_training_data": False,
    }


def _source_fingerprint(p) -> Optional[dict]:
    """Enough to recognise the source KG on re-attach, without carrying it.

    A path is a fact about the exporting machine and useless to anyone else, so
    it is kept only as a *hint* for the file picker; the sha256 is what decides
    whether a re-attached file is the same KG.
    """
    src = p.source or {}
    return {
        "kind": src.get("kind"),
        "name": src.get("name") or src.get("filename"),
        "url": src.get("url"),
        "sha256": src.get("sha256"),
        "path_hint": p.source_ref if isinstance(p.source_ref, str) else None,
    }


def write_model_into(z: zipfile.ZipFile, model) -> bool:
    """Write `model/scm.pkl` + `model/manifest.json` + `model/curated_graph.ttl`.

    `CausalModel.save` writes a directory, so this goes through a temporary one
    rather than reimplementing the layout here — the bundle and a local
    `results/models/<id>/` are then the same three files, and `CausalModel.load`
    reads either without knowing which it got.
    """
    if model is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        for name in sorted(os.listdir(tmp)):
            path = os.path.join(tmp, name)
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    z.writestr(f"{MODEL_DIR}/{name}", fh.read())
    return True


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
class LoadedBundle:
    """A parsed bundle, held in memory until something asks to restore it."""

    def __init__(self, files: dict, descriptor: dict):
        self.files = files
        self.descriptor = descriptor

    def json(self, name: str) -> Optional[Any]:
        raw = self.files.get(name)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"{name} in this bundle is not readable JSON ({exc}).") from exc

    @property
    def has_model(self) -> bool:
        return f"{MODEL_DIR}/scm.pkl" in self.files


def read_bundle(data: bytes, *, installed: dict, strict: bool = True) -> LoadedBundle:
    """Parse and vet a bundle. Raises `BundleError` rather than returning junk."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            bad = z.testzip()
            if bad is not None:
                raise BundleError(f"The archive is corrupt at {bad!r}.")
            files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    except zipfile.BadZipFile as exc:
        raise BundleError(f"That is not a readable zip archive ({exc}).") from exc

    bundle = LoadedBundle(files, {})
    descriptor = bundle.json(DESCRIPTOR)
    if descriptor is None:
        # Every project zip this service has ever written contains project.json;
        # accepting one without a descriptor is what lets a zip exported before
        # bundles existed still import (as a project, with no model).
        if "project.json" not in files:
            raise BundleError(
                "This zip has neither bundle.json nor project.json, so it is not a "
                "CausalWay export. Export a project from the Export menu to get one."
            )
        descriptor = {"bundle_version": "0", "kind": "causalway-project",
                      "has_model": False, "versions": {}}
    bundle.descriptor = descriptor

    if descriptor.get("kind") == _PRE_RENAME_KIND:
        raise BundleError(
            "This bundle was exported before the CausalKG → CausalWay rename. Its "
            "vocabulary IRIs are in the retired http://sdm-causalkg.org/ namespace "
            "and, if it carries a model, its pickle names modules that no longer "
            "exist. It cannot be restored here — re-run discovery and fitting on the "
            "source KG to produce a current bundle."
        )
    if descriptor.get("kind") != "causalway-project":
        raise BundleError(f"Unexpected bundle kind {descriptor.get('kind')!r}.")

    version = str(descriptor.get("bundle_version", "0"))
    if version > BUNDLE_VERSION:
        raise BundleError(
            f"This bundle is version {version}; this service reads up to "
            f"{BUNDLE_VERSION}. Newer bundles can hold things this version would "
            "drop silently, so it is refused rather than partly restored."
        )

    if strict and bundle.has_model:
        recorded = descriptor.get("versions") or {}
        mismatched = {
            k: {"bundle": recorded[k], "installed": v}
            for k, v in installed.items()
            if k in recorded and recorded[k] != v
        }
        if mismatched:
            raise BundleError(
                f"This bundle's model was fitted with different libraries than are "
                f"installed here: {json.dumps(mismatched)}. A pickled estimator "
                "loaded under a different version can misbehave without raising, so "
                "this is refused by default. Re-send with strict=false to load it "
                "anyway, or import without the model and refit."
            )
    return bundle


def _pairs(value) -> list[tuple]:
    if not isinstance(value, list):
        return []
    return [tuple(e) for e in value if isinstance(e, (list, tuple)) and len(e) == 2]


def restore_into(bundle: LoadedBundle, p, *, load_model) -> list[str]:
    """Populate a *fresh* project from a bundle. Returns human-readable notes.

    Restores only what does not need the source KG. The schema, the candidate
    nodes and the materialised frame are all derived from the KG, and this
    deliberately does not fake them: a project whose nodes were restored from a
    zip but whose KG is absent would let the user curate against a schema
    nothing can check, and fail at Materialise with an error about a file they
    were never asked for. Instead they are left empty and named in `notes`, and
    `verify_source` is the way to fill them in.
    """
    notes: list[str] = []

    project = bundle.json("project.json") or {}
    p.name = project.get("name") or p.name
    p.source = project.get("source")
    p.constraint_options = project.get("constraint_options") or p.constraint_options
    p.excluded = set(project.get("excluded") or [])
    p.excluded_joins = set(project.get("excluded_joins") or [])
    p.last_component = project.get("component")

    layouts = bundle.json("layout.json")
    if isinstance(layouts, dict):
        # Merge rather than replace: a bundle written before a board existed has
        # no key for it, and dropping the default would leave `p.layouts[...]`
        # raising KeyError on a board the user has not opened yet.
        for board, positions in layouts.items():
            if board in p.layouts and isinstance(positions, dict):
                p.layouts[board] = positions

    # The curated graph is read from `project.json`, not from
    # `curated_graph.json`. The latter is the RML document `to_json()` produces
    # — a full cw:OntologicalCausalGraph with node slots, reified edges and a
    # discovery run — and reversing it would mean writing an inverse of the
    # export that has to be kept in step with `ocg_mapping.rml.ttl` forever.
    # The two selection lists are the actual UI state, so they travel as
    # themselves; the RDF stays the publishable artefact it was written to be.
    p.selected_edges = _pairs(project.get("selected_edges"))
    p.manual_edges = _pairs(project.get("manual_edges"))
    if isinstance(project.get("type_options"), dict):
        p.type_options = {**p.type_options, **project["type_options"]}
    if isinstance(project.get("prior_meta"), dict):
        p.prior_meta = {**p.prior_meta, **project["prior_meta"]}

    priors = bundle.json("priors.json")
    if isinstance(priors, dict) and priors.get("M"):
        p.priors = {"columns": priors.get("columns"), "M": priors.get("M"), "B": priors.get("B")}
        # Whatever produced them, they arrived here in a file. Calling them "llm"
        # would claim a provenance this bundle cannot prove.
        p.priors_source = "imported"

    runs = bundle.json("discovery_runs.json")
    if isinstance(runs, list) and runs:
        notes.append(
            f"{len(runs)} discovery run(s) are in the bundle but not restored — a run "
            "is only interpretable against the materialised frame it was computed on. "
            "Re-attach the source KG and rerun, or read discovery_runs.json directly."
        )

    answers = bundle.json("answers.json")
    if isinstance(answers, list):
        p.answers = answers
        p.next_answer_id = max((a.get("answer_id", 0) for a in answers), default=0) + 1

    worlds = bundle.json("counterfactual_worlds.json")
    if isinstance(worlds, list):
        # The export shape is the RML document, not `p.cf_worlds`; restoring the
        # log from it would need an inverse of `worlds_to_json` that does not
        # exist. The TTL in the bundle is the durable copy.
        notes.append(
            f"{len(worlds)} counterfactual world(s) are in the bundle as RDF but not "
            "restored to the session log — the exported Turtle is their durable form."
        )

    if bundle.has_model:
        model, model_notes = load_model(bundle)
        notes.extend(model_notes)
        if model is not None:
            p.model = model
            p.model_kind = "scm"
    else:
        notes.append("This bundle has no fitted model; module 3 starts from Fit.")

    notes.append(
        "The source KG is not carried in a bundle. The schema, candidate nodes and "
        "materialised frame stay empty until you re-attach it — modules 1 and 2 are "
        "unavailable until then."
    )
    return notes


def write_model_dir(bundle: LoadedBundle, directory: str) -> str:
    """Unpack `model/` to a real directory, because `CausalModel.load` reads paths."""
    os.makedirs(directory, exist_ok=True)
    prefix = f"{MODEL_DIR}/"
    for name, raw in bundle.files.items():
        if not name.startswith(prefix):
            continue
        leaf = os.path.basename(name)
        if not leaf or leaf.startswith("."):
            continue
        with open(os.path.join(directory, leaf), "wb") as fh:
            fh.write(raw)
    return directory


# --------------------------------------------------------------------------- #
# Re-attaching a source
# --------------------------------------------------------------------------- #
def verify_source(data: bytes, expected_sha256: Optional[str]) -> str:
    """Hash the offered KG and refuse it if it is not the one the model was fitted on.

    A hard error, not a warning. The failure this prevents is silent: the
    mechanisms stay exactly as fitted, every query still answers, and the answers
    are about rows that no longer exist. There is no downstream check that would
    catch it and no symptom the user could notice. Nothing about the cost of
    being wrong here justifies a dismissible dialog.
    """
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise BundleError(
            "This file is not the knowledge graph the bundle's model was fitted on "
            f"(sha256 {digest[:12]}… vs the recorded {expected_sha256[:12]}…). Attaching "
            "it would leave the mechanisms fitted on one set of rows and every answer "
            "computed against another, with nothing downstream able to tell. Attach the "
            "original file, or import the bundle without a source and refit."
        )
    return digest
