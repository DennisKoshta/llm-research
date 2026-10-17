import importlib.util
from pathlib import Path
import pytest
from curiosity_study.artifacts import RunStore
from curiosity_study.credit import build_manifest

spec = importlib.util.spec_from_file_location(
    "credit_uncertainty", Path(__file__).resolve().parents[1] / "scripts/credit_uncertainty.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_cluster_intervals_preserve_resolution_and_invalid_denominators(tmp_path):
    m = build_manifest(2, 123, {"model_id": "mock"}, "behavior")
    store = RunStore(tmp_path, m)
    for a in m["assignments"]:
        # Identical within-pair layout choices; entire pair is invalid in one cluster.
        choice = "invalid" if a["index"] == 0 else ("Y" if a["condition"] == "x_resolved" else "X")
        store.save({"trial_id": a["trial_id"], "terminal": True, "first_service": choice}, "fixture")
    result = analysis.analyze(tmp_path, repetitions=100, seed=4)
    for family in result["rows"]:
        metrics = family["metrics"]
        assert family["pair_count"] == 2
        assert metrics["p/both_unresolved/invalid"]["estimate"] == 0.5
        assert metrics["p/both_unresolved/X"]["estimate"] == 0.5
        assert metrics["difference/x_resolved-both_unresolved/X"]["estimate"] == -0.5
        assert metrics["difference/x_resolved-both_unresolved/Y"]["estimate"] == 0.5
        assert metrics["difference/x_resolved-both_unresolved/invalid"][
            "pointwise_percentile_95_interval"
        ] == [0, 0]
        assert metrics["p/both_unresolved/invalid"]["pointwise_percentile_95_interval"] == [0, 1]


def test_quantile_interpolates():
    assert analysis.quantile([0, 10], 0.25) == pytest.approx(2.5)
