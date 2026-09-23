"""Bundle validation — the checks that decide whether an archive is trustworthy.

These run against `service.api.bundle` directly, without a FastAPI app: every
failure mode here is about *file* validity (corrupt zip, wrong version,
mismatched libraries, a KG that is not the one the model was fitted on), and a
test that has to boot the service to exercise them would be slower and would
prove less.
"""

import io
import json
import zipfile

import pytest

from service.api import bundle

INSTALLED = {"dowhy": "0.12", "scikit_learn": "1.5.0", "numpy": "2.1.0"}


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, payload in files.items():
            z.writestr(name, payload if isinstance(payload, (bytes, str))
                       else json.dumps(payload))
    return buf.getvalue()


def _descriptor(**over) -> dict:
    base = {
        "bundle_version": bundle.BUNDLE_VERSION,
        "kind": "causalway-project",
        "has_model": False,
        "versions": dict(INSTALLED),
        "source_kg": {"sha256": "a" * 64},
        "contains_training_data": False,
    }
    base.update(over)
    return base


# --- reading ----------------------------------------------------------------


def test_a_plain_project_zip_is_readable():
    data = _zip({"bundle.json": _descriptor(), "project.json": {"name": "p"}})
    loaded = bundle.read_bundle(data, installed=INSTALLED)
    assert loaded.has_model is False
    assert loaded.json("project.json")["name"] == "p"


def test_a_zip_from_before_bundles_existed_still_imports():
    """Every project export this service has written contains project.json. A
    descriptor-less archive is an old export, not a foreign file."""
    loaded = bundle.read_bundle(_zip({"project.json": {"name": "old"}}), installed=INSTALLED)
    assert loaded.descriptor["bundle_version"] == "0"
    assert loaded.json("project.json")["name"] == "old"


def test_something_that_is_not_a_causalway_export_is_refused():
    with pytest.raises(bundle.BundleError, match="not a CausalWay export"):
        bundle.read_bundle(_zip({"notes.txt": "hello"}), installed=INSTALLED)


def test_a_file_that_is_not_a_zip_is_refused():
    with pytest.raises(bundle.BundleError, match="readable zip"):
        bundle.read_bundle(b"PK not really", installed=INSTALLED)


def test_a_newer_bundle_is_refused_rather_than_partly_restored():
    """A newer layout can hold things this reader would drop without noticing;
    silently importing 80% of a project is worse than refusing it."""
    data = _zip({"bundle.json": _descriptor(bundle_version="99"), "project.json": {}})
    with pytest.raises(bundle.BundleError, match="version 99"):
        bundle.read_bundle(data, installed=INSTALLED)


# --- the library-version gate ----------------------------------------------


def test_a_model_fitted_with_other_libraries_is_refused_by_default():
    data = _zip({
        "bundle.json": _descriptor(has_model=True,
                                   versions={**INSTALLED, "scikit_learn": "1.2.0"}),
        "project.json": {},
        "model/scm.pkl": b"\x00",
    })
    with pytest.raises(bundle.BundleError, match="scikit_learn"):
        bundle.read_bundle(data, installed=INSTALLED)


def test_the_version_gate_can_be_overridden_deliberately():
    data = _zip({
        "bundle.json": _descriptor(has_model=True,
                                   versions={**INSTALLED, "scikit_learn": "1.2.0"}),
        "project.json": {},
        "model/scm.pkl": b"\x00",
    })
    loaded = bundle.read_bundle(data, installed=INSTALLED, strict=False)
    assert loaded.has_model is True


def test_the_version_gate_does_not_fire_without_a_model():
    """Nothing is unpickled when there is no model, so the versions cannot
    matter — refusing there would block an import for no reason."""
    data = _zip({
        "bundle.json": _descriptor(versions={**INSTALLED, "scikit_learn": "1.2.0"}),
        "project.json": {},
    })
    assert bundle.read_bundle(data, installed=INSTALLED).has_model is False


# --- source verification ----------------------------------------------------


def test_the_right_kg_verifies():
    data = b"@prefix ex: <http://example.org/> ."
    import hashlib
    digest = hashlib.sha256(data).hexdigest()
    assert bundle.verify_source(data, digest) == digest


def test_the_wrong_kg_is_a_hard_error():
    """Fitted mechanisms plus different rows is a failure with no symptom: every
    query still answers, about rows that are not there."""
    with pytest.raises(bundle.BundleError, match="not the knowledge graph"):
        bundle.verify_source(b"different bytes", "b" * 64)


def test_no_recorded_hash_means_nothing_to_check():
    """A project exported without a source has no hash; that is not a mismatch."""
    assert bundle.verify_source(b"anything", None)


# --- restoring --------------------------------------------------------------


class _FakeProject:
    def __init__(self):
        self.name = "Imported project"
        self.source = None
        self.constraint_options = {"relation_direction": "both"}
        self.excluded = set()
        self.excluded_joins = set()
        self.last_component = None
        self.layouts = {"ontology": {}, "causal": {}}
        self.selected_edges = []
        self.manual_edges = []
        self.type_options = {"max_levels": 20}
        self.prior_meta = {"domain": ""}
        self.priors = None
        self.priors_source = None
        self.answers = []
        self.next_answer_id = 1
        self.model = None
        self.model_kind = "scm"


def test_the_curated_selection_round_trips_through_project_json():
    """The selection travels as itself, not inside the RML document — reversing
    `to_json()` would mean maintaining an inverse of the RDF export forever."""
    data = _zip({
        "bundle.json": _descriptor(),
        "project.json": {
            "name": "restored", "excluded": ["A.x"], "excluded_joins": ["A.r"],
            "selected_edges": [["A.x", "A.y"], ["A.y", "A.z"]],
            "manual_edges": [["A.y", "A.z"]],
            "type_options": {"max_levels": 7},
        },
        "layout.json": {"causal": {"A.x": {"x": 1, "y": 2}}, "unknown_board": {}},
    })
    loaded = bundle.read_bundle(data, installed=INSTALLED)
    p = _FakeProject()
    notes = bundle.restore_into(loaded, p, load_model=lambda b: (None, []))

    assert p.name == "restored"
    assert p.excluded == {"A.x"} and p.excluded_joins == {"A.r"}
    assert p.selected_edges == [("A.x", "A.y"), ("A.y", "A.z")]
    assert p.manual_edges == [("A.y", "A.z")]
    assert p.type_options["max_levels"] == 7
    assert p.layouts["causal"] == {"A.x": {"x": 1, "y": 2}}
    # A board the bundle does not know about keeps its default rather than vanishing
    assert p.layouts["ontology"] == {}
    assert "unknown_board" not in p.layouts
    assert any("source KG is not carried" in n for n in notes)


def test_imported_priors_are_never_labelled_as_llm_output():
    """They arrived in a file. Calling them "llm" claims a provenance the bundle
    cannot prove."""
    data = _zip({
        "bundle.json": _descriptor(),
        "project.json": {},
        "priors.json": {"columns": ["a", "b"], "M": [[0, 1], [0, 0]], "B": [[0, 1], [0, 0]]},
    })
    p = _FakeProject()
    bundle.restore_into(bundle.read_bundle(data, installed=INSTALLED), p,
                        load_model=lambda b: (None, []))
    assert p.priors_source == "imported"
    assert p.priors["columns"] == ["a", "b"]


def test_discovery_runs_are_reported_not_silently_dropped():
    data = _zip({
        "bundle.json": _descriptor(),
        "project.json": {},
        "discovery_runs.json": [{"run_id": 1}, {"run_id": 2}],
    })
    p = _FakeProject()
    notes = bundle.restore_into(bundle.read_bundle(data, installed=INSTALLED), p,
                               load_model=lambda b: (None, []))
    assert any("2 discovery run(s)" in n for n in notes)
