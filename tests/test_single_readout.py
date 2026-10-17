import importlib.util
from pathlib import Path
import pytest
from curiosity_study.artifacts import RunStore
from curiosity_study.single_curiosity import build_manifest

spec = importlib.util.spec_from_file_location(
    "readout", Path(__file__).resolve().parents[1] / "scripts/single_curiosity_readout.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_matched_readout_extremes_and_missingness(tmp_path):
    manifest = build_manifest(5, 7, {"model_id": "mock"}, "behavior")
    store = RunStore(tmp_path, manifest)
    for a in manifest["assignments"][:-1]:
        store.save(
            dict(
                trial_id=a["trial_id"],
                assignment=a,
                terminal=True,
                first_service="clue" if a["condition"] == "unresolved" else "stop",
            ),
            "fixture",
        )
    with pytest.raises(AssertionError):
        module.summarize(tmp_path)
    a = manifest["assignments"][-1]
    store.save(
        dict(
            trial_id=a["trial_id"],
            assignment=a,
            terminal=True,
            first_service="clue" if a["condition"] == "unresolved" else "stop",
        ),
        "fixture",
    )
    result = module.summarize(tmp_path)
    for r in result["contrasts"]:
        expected = {"clue": 1, "stop": -1}.get(r["outcome"], 0)
        assert r["difference"] == expected and r["ci95"] == [expected, expected]
    for r in result["headroom"]:
        assert r["upward_with_invalid_and_technical_fixed"] == (r["condition"] == "resolved")
    assert all(r["assigned"] == 10 for r in result["rows"])
