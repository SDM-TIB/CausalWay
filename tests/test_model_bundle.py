"""A saved model must be loadable somewhere other than where it was fitted.

`CausalModel.save` used to write only `scm.pkl` + `manifest.json`, and the
manifest recorded the causal graph as a *path*. That is fine as a local cache
and useless as an export: move the directory to another machine and `load`
silently returns a model with `spec.ocg = None`, which cannot answer a query.
These tests pin the bundle being self-contained, and pin the training data
staying out of it.
"""

import json
import os

import pytest

from causalway.model import CausalModel, GRAPH_FILENAME

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_TTL = os.path.join(_ROOT, "kgs", "ttls", "synthetic_clinic.ttl")
CLINIC_OCG = os.path.join(_ROOT, "results", "synthetic", "best_GES_constrained.ttl")


@pytest.fixture(scope="module")
def fitted():
    return CausalModel.fit(CLINIC_OCG, CLINIC_TTL,
                           include_object_properties=False,
                           on_cycle="weight", random_state=11)


def test_the_bundle_holds_the_graph_and_not_the_data(fitted, tmp_path):
    out = str(tmp_path / "bundle")
    fitted.save(out)

    assert sorted(os.listdir(out)) == [GRAPH_FILENAME, "manifest.json", "scm.pkl"]

    # The training rows are the user's KG. An export artefact that carries them
    # turns every model share into a data share nobody asked for.
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert "data" not in manifest
    assert not any(f.endswith((".csv", ".parquet", ".feather")) for f in os.listdir(out))


def test_the_graph_reloads_from_the_bundle_when_the_source_path_is_gone(fitted, tmp_path):
    """The point of the bundle: `causal_graph_source` is a path on the fitting
    machine, so after an export it is at best stale and at worst a *different*
    graph at the same path."""
    out = str(tmp_path / "bundle")
    fitted.save(out)

    mf = os.path.join(out, "manifest.json")
    manifest = json.load(open(mf))
    manifest["causal_graph_source"] = {"kind": "file", "path": "/nonexistent/gone.ttl"}
    with open(mf, "w") as fh:
        json.dump(manifest, fh, indent=2)

    back = CausalModel.load(out)
    assert back.spec.ocg is not None
    assert back.verify_graph(back.spec.ocg, strict=False)
    assert (back.spec.ocg.adj == fitted.spec.ocg.adj).all()
    assert sorted(n.name for n in back.spec.ocg.nodes) == \
        sorted(n.name for n in fitted.spec.ocg.nodes)


def test_the_bundled_graph_wins_over_a_stale_recorded_path(fitted, tmp_path):
    """Not just a fallback — the bundled graph is preferred, because a path that
    still resolves is exactly the case where the wrong graph loads silently."""
    out = str(tmp_path / "bundle")
    fitted.save(out)

    # Point the recorded source at a real, loadable, but *different* OCG.
    other = os.path.join(_ROOT, "results", "sclc", "GES.ttl")
    mf = os.path.join(out, "manifest.json")
    manifest = json.load(open(mf))
    manifest["causal_graph_source"] = {"kind": "file", "path": other}
    with open(mf, "w") as fh:
        json.dump(manifest, fh, indent=2)

    back = CausalModel.load(out)
    assert back.verify_graph(back.spec.ocg, strict=False), \
        "load() took the stale recorded path instead of the bundled graph"


def test_save_without_the_graph_still_works(fitted, tmp_path):
    """`with_graph=False` keeps the old two-file layout for callers that only
    want a local cache and already know where their graph lives."""
    out = str(tmp_path / "cache")
    fitted.save(out, with_graph=False)
    assert sorted(os.listdir(out)) == ["manifest.json", "scm.pkl"]
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert "bundled_graph" not in manifest
