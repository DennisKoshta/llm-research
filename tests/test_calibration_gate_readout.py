import importlib.util
from pathlib import Path

from curiosity_study.artifacts import RunStore
from curiosity_study.credit import build_manifest

spec = importlib.util.spec_from_file_location(
    "credit_calibration_gates", Path(__file__).resolve().parents[1] / "scripts/credit_calibration_gates.py"
)
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)


def test_validity_boundary_and_failed_cell_identity(tmp_path):
    for invalid_count in (5, 6):
        root = tmp_path / str(invalid_count)
        m = build_manifest(100, 19, {"model_id": "mock"}, "behavior")
        store = RunStore(root, m)
        for a in m["assignments"]:
            affected = (a["family"], a["condition"], a["layout"]) == ("machines", "x_resolved", "XY")
            choice = "X" if a["index"] % 2 == 0 else "Y"
            if affected and a["index"] < invalid_count:
                choice = "invalid"
            store.save(
                {"trial_id": a["trial_id"], "assignment": a, "terminal": True, "first_service": choice},
                "fixture",
            )
        result = readout.explain(root)
        assert result["passed"] is (invalid_count == 5)
        if invalid_count == 6:
            assert len(result["failed_gates"]) == 1
            failed = result["failed_gates"][0]
            assert (failed["family"], failed["condition"], failed["layout"]) == (
                "machines",
                "x_resolved",
                "XY",
            )
            assert failed["criterion"] == "minimum_valid_format"
            assert failed["observed"] == 0.94
