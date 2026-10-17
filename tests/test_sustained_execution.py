"""Local durable-stage tests; no data freeze, cloud call or model outcomes."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study import sustained_engineering as engineering
from curiosity_study import report, runner
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.concept_capture import MODEL
from test_sustained_engineering import inputs, model_and_direction

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collection_coverage import CollectionCoverageError, check_collection_coverage  # noqa: E402


def manifest(cap=128):
    unit = np.arange(16, dtype=np.float64) - 7.5
    unit /= np.linalg.norm(unit)
    requests = [{**request, "labels": list("ABCD")} for request in inputs()]
    result = dict(
        assay="sustained_engineering",
        phase="engineering",
        scientific_admission=False,
        model=deepcopy(MODEL),
        execution=deepcopy(engineering.EXECUTION),
        execution_gpu="H200",
        arithmetic=deepcopy(engineering.ARITHMETIC),
        schedule=engineering.SCHEDULE,
        layer=15,
        cap=cap,
        requests=requests,
        source_binding={"run": "synthetic-source"},
        engine_sha256=engineering.engine_hashes(),
        direction=dict(
            source="curiosity_vs_indifference",
            corpus="A",
            layer=15,
            position="record",
            unit=unit.tolist(),
            unit_dtype=str(unit.dtype),
            unit_binding=engineering.array_binding(unit),
            norm=0.25,
            dose=0.25,
            multiplier=1.0,
        ),
        assignments=[],
        diagnostics=engineering.assignment_plan(requests, cap),
        short_stage=None,
    )
    result["assignment_sha256"] = digest(result["diagnostics"])
    result["compatibility_sha256"] = digest(engineering.compatibility_binding(result))
    return result


def model():
    return SimpleNamespace(metadata=lambda: dict(**MODEL, dtype="bfloat16", attention="eager"))


def result(passed=True):
    harness = dict(
        passed=passed,
        conditions={
            condition: dict(
                attempted_forwards=1,
                rows=[dict(completion=dict(generated_token_ids=[row + 1])) for row in range(32)],
            )
            for condition in engineering.CONDITIONS
        },
    )
    return harness, dict(
        before=np.zeros((2, 16), dtype=np.float32),
        after=np.ones((2, 16), dtype=np.float32),
        direction_float32=np.ones(16, dtype=np.float32) / 4,
    )


@pytest.mark.parametrize("passed", [True, False])
def test_evidence_is_durable_before_all128_operational_records(tmp_path, monkeypatch, passed):
    frozen = manifest()
    monkeypatch.setattr(engineering, "run_engineering", lambda *a, **k: result(passed))
    original = RunStore.save_many
    seen = []

    def check_before_publish(store, records, attempt, offset):
        report = records[0]["report_artifact"]
        artifact = records[0]["activation_artifact"]
        assert hashlib.sha256((tmp_path / report["path"]).read_bytes()).hexdigest() == report["sha256"]
        assert hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
        wrapper = read(tmp_path / report["path"])
        assert wrapper["engineering_passed"] is passed and wrapper["status"] == "measured"
        assert wrapper["vector_artifact"] == artifact
        assert artifact["byte_count"] == (tmp_path / artifact["path"]).stat().st_size
        assert report["byte_count"] == (tmp_path / report["path"]).stat().st_size
        assert wrapper["performance"]["total_forward_count"] == 4
        assert wrapper["performance"]["generated_tokens_by_condition"] == dict.fromkeys(
            engineering.CONDITIONS, 32
        )
        assert wrapper["performance"]["peak_cuda_allocated_bytes"] is None
        seen.append(offset)
        original(store, records, attempt, offset)

    monkeypatch.setattr(RunStore, "save_many", check_before_publish)
    rows = runner.execute(tmp_path, frozen, model())
    assert len(rows) == 128 and seen == ["0000000", "0000032", "0000064", "0000096"]
    assert all(row["engineering_passed"] is passed and "failure" not in row for row in rows.values())
    assert check_collection_coverage(tmp_path, expected_manifest_sha256=digest(frozen))["passed"]
    # Operational completion intentionally cannot certify engineering success.
    assert str(passed) in engineering.render(frozen, rows)
    assert report.render(tmp_path) == engineering.render(frozen, rows)
    with pytest.raises(ValueError, match="cannot be rerun"):
        engineering.execute(tmp_path, frozen, model())


def test_runtime_failure_retains128_unavailable_dispositions_and_cannot_retry(tmp_path, monkeypatch):
    frozen = manifest()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(engineering, "run_engineering", fail)
    rows = engineering.execute(tmp_path, frozen, model())
    assert len(rows) == 128 and calls == [1]
    for row in rows.values():
        assert row["failure"] == row["status"] == "execution_error"
        assert row["condition_row_sha256"] is None and row["activation_artifact"] is None
        assert not row["engineering_passed"] and not row["scientific_admission"]
    wrapper = read(tmp_path / next(iter(rows.values()))["report_artifact"]["path"])
    assert wrapper["harness"] is None and wrapper["error"]["message"] == "synthetic forward failure"
    assert wrapper["performance"]["total_forward_count"] is None
    assert wrapper["performance"]["generated_tokens_by_condition"] is None
    with pytest.raises(CollectionCoverageError, match="execution-error"):
        check_collection_coverage(tmp_path, expected_manifest_sha256=digest(frozen))
    with pytest.raises(ValueError, match="cannot be rerun"):
        engineering.execute(tmp_path, frozen, model())
    assert calls == [1]


def test_expired_deadline_does_not_invoke_model_and_remains_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(engineering, "run_engineering", lambda *a, **k: pytest.fail("Expired stage ran"))
    rows = engineering.execute(tmp_path, manifest(), model(), deadline=0)
    assert len(rows) == 128 and all(row["error_type"] == "TimeoutError" for row in rows.values())


def test_partial_publication_and_prior_admission_refuse_fresh_compute(tmp_path, monkeypatch):
    monkeypatch.setattr(engineering, "run_engineering", lambda *a, **k: result())
    original = RunStore.save_many

    def partial(store, records, attempt, offset):
        if offset == "0000032":
            raise OSError("synthetic persistence failure")
        original(store, records, attempt, offset)

    monkeypatch.setattr(RunStore, "save_many", partial)
    frozen = manifest()
    with pytest.raises(OSError, match="persistence"):
        engineering.execute(tmp_path, frozen, model())
    assert len(RunStore(tmp_path, frozen).completed()) == 32
    with pytest.raises(ValueError, match="cannot be rerun"):
        engineering.execute(tmp_path, frozen, model())
    other = tmp_path / "previous-admission"
    write_once(other / "admissions/a.json", {})
    write_once(other / "admissions/b.json", {})
    with pytest.raises(ValueError, match="cannot be rerun"):
        engineering.execute(other, frozen, model())


def test_deadline_guard_cleans_real_model_observers_before_any_forward():
    model_value, unit = model_and_direction()
    with pytest.raises(TimeoutError, match="deadline"):
        engineering.run_engineering(model_value, inputs(), direction=unit, dose=0.25, cap=4, deadline=0)
    assert not engineering.hook_inventory(model_value) and model_value.active_hooks == 0


def test_full_cap_requires_exact_short_stage_and_cap_cannot_certify_itself():
    full = manifest(4096)
    with pytest.raises(ValueError, match="short stage"):
        engineering.validate_manifest(full)
    audit = dict(
        assay="sustained_engineering_audit",
        passed=True,
        complete=True,
        scientific_admission=False,
        cap=128,
        manifest_sha256="short",
        compatibility_sha256=full["compatibility_sha256"],
        report_artifact={"sha256": "report"},
        vector_artifact={"sha256": "vectors"},
    )
    full["short_stage"] = dict(manifest_sha256="short", audit=audit, audit_sha256=digest(audit))
    assert len(engineering.validate_manifest(full)) == 128
    for key, value in (("passed", False), ("cap", 4096), ("compatibility_sha256", "other")):
        altered = deepcopy(full)
        altered["short_stage"]["audit"][key] = value
        altered["short_stage"]["audit_sha256"] = digest(altered["short_stage"]["audit"])
        with pytest.raises(ValueError, match="short stage"):
            engineering.validate_manifest(altered)


@pytest.mark.parametrize("mutation", ["dose", "layer", "seed", "order", "engine"])
def test_changed_locked_execution_contract_is_rejected(mutation):
    frozen = manifest()
    if mutation == "dose":
        frozen["direction"]["dose"] *= 2
    elif mutation == "layer":
        frozen["direction"]["layer"] = 31
    elif mutation == "seed":
        frozen["requests"][0]["seed"] += 1
    elif mutation == "order":
        frozen["diagnostics"][0], frozen["diagnostics"][1] = (
            frozen["diagnostics"][1],
            frozen["diagnostics"][0],
        )
    else:
        frozen["engine_sha256"]["model.py"] = "changed"
    with pytest.raises(ValueError):
        engineering.validate_manifest(frozen)
