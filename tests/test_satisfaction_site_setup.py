"""Unscored setup durability and provenance checks; synthetic CPU data only."""

import base64
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from curiosity_study import satisfaction_site_setup as setup
from curiosity_study.artifacts import RunStore, digest, read


@pytest.fixture
def inputs(monkeypatch):
    # Only this NEW wrapper's dimensional contract is reduced; old engines are unchanged.
    monkeypatch.setattr(setup, "HIDDEN_SIZE", 4)
    requests = [dict(source_trial_id=f"synthetic-{i:03d}", seed=i, labels="ABCD",
                     token_prefix=[1, 2, 3], token_prefix_sha256=digest([1, 2, 3]),
                     prefix_positions=dict(record=1, final=2)) for i in range(288)]
    hidden = np.ones((288, 1, 2, 4), dtype=np.float32)
    historical = dict(source_manifest_sha256=setup.study.PRIOR_MANIFEST_SHA256,
                      source_audit_sha256="a" * 64, record_bindings_sha256="b" * 64,
                      artifact_bindings_sha256="c" * 64, requests_sha256=digest(requests),
                      hidden=setup.engineering.array_binding(hidden))
    units = {str(site): dict(source="satisfaction_vs_neutral", corpus="A", layer=site,
                            position="record", orientation=1, source_raw_norm=2.0,
                            unit=[1.0, 0.0, 0.0, 0.0], unit_sha256=digest([1.0, 0.0, 0.0, 0.0]),
                            source_unit_sha256=digest([1.0, 0.0, 0.0, 0.0]),
                            directions_byte_sha256=setup.study.LOCKED_DIRECTION_BYTES)
             for site in setup.study.SITES}
    assignments = setup.assignment_plan(requests)
    manifest = dict(assay=setup.ASSAY, phase="engineering", scientific_admission=False,
                    full_cap_certified=False, model=setup.MODEL, execution=setup.EXECUTION,
                    execution_gpu="H200", arithmetic=setup.ARITHMETIC, schedule=setup.SCHEDULE,
                    sites=[31, 47], cap=128, original_dose=setup.study.ORIGINAL_S_DOSE,
                    budget=deepcopy(setup.resources.POLICIES["setup"]), requests=requests,
                    requests_sha256=digest(requests), historical_hidden=setup.encode_historical(hidden),
                    historical_binding=historical, historical_binding_sha256=digest(historical),
                    source_units=units, source_units_sha256=digest(units), assignments=[],
                    diagnostics=assignments, assignment_sha256=digest(assignments),
                    engine_sha256=setup.engine_hashes())
    metadata = dict(**setup.MODEL, dtype="bfloat16", attention="eager", device="cuda", gpu="NVIDIA H200")
    model = SimpleNamespace(metadata=lambda: deepcopy(metadata), active_hooks=0,
                            model=SimpleNamespace(config=SimpleNamespace(hidden_size=4),
                                                  named_modules=lambda: []))
    return manifest, model


def install_collectors(monkeypatch, manifest, *, reference_passed=True, site_failed=None, site_error=None):
    calls = []

    def reference(model, requests, **kwargs):
        calls.append("reference")
        assert requests == manifest["requests"]
        assert kwargs["historical_binding"] == manifest["historical_binding"]
        assert kwargs["original_dose"] == setup.study.ORIGINAL_S_DOSE
        assert np.array_equal(kwargs["historical_layer15_hidden"], setup.decode_historical(manifest["historical_hidden"]))
        hidden = np.ones((288, 3, 2, 4), dtype=np.float32)
        hidden[:, 1] *= 2
        hidden[:, 2] *= 3
        norms = np.linalg.norm(hidden[:, :, 0].astype(np.float64), axis=-1)
        arrays = dict(hidden=hidden, record_norms_float64=norms)
        medians = {str(site): float(v) for site, v in zip((15, 31, 47), np.median(norms, axis=0))}
        doses = {str(site): setup.study.ORIGINAL_S_DOSE * medians[str(site)] / medians["15"] for site in (31, 47)}
        cohorts = [dict(offset=start, measurements=[dict(token_prefix_sha256=r["token_prefix_sha256"])
                                                   for r in requests[start:start + 32]],
                        engineering=dict(passed=True), historical_layer15_equal=reference_passed)
                   for start in range(0, 288, 32)]
        inventory = dict(requests_sha256=digest(requests), cohorts_sha256=digest(cohorts),
                         arrays={k: setup.engineering.array_binding(v) for k, v in arrays.items()})
        dose_reference = dict(formula=setup.study.DOSE_FORMULA, original_dose=setup.study.ORIGINAL_S_DOSE,
                              base_branches=288, source_screen_manifest_sha256=setup.study.PRIOR_MANIFEST_SHA256,
                              capture_sha256=digest(inventory), median_record_norm=medians, doses=doses)
        return dict(passed=reference_passed, checks={"historical_layer15_equal": reference_passed},
                    cohorts=cohorts, capture_inventory=inventory, dose_reference=dose_reference,
                    doses=doses), arrays

    def engineering(model, requests, *, layer, direction, dose, cap, deadline):
        calls.append(layer)
        assert requests == manifest["requests"][:32]
        assert cap == 128 and np.array_equal(direction, manifest["source_units"][str(layer)]["unit"])
        assert dose == setup.study.ORIGINAL_S_DOSE * (2 if layer == 31 else 3)
        if layer == site_error:
            raise RuntimeError("synthetic partial site failure")
        conditions = {name: dict(attempted_forwards=2, rows=[dict(completion=dict(generated_token_ids=[4, 5]))
                                                           for _ in range(32)]) for name in setup.CONDITIONS}
        return dict(passed=layer != site_failed, checks={"full_sham_identity": layer != site_failed},
                    conditions=conditions, malformed_direction=dict(attempted_forwards=0)), dict(
                        before=np.ones((2, 4), dtype=np.float32), after=np.full((2, 4), 2, dtype=np.float32),
                        direction_float32=np.asarray(direction, dtype=np.float32))

    monkeypatch.setattr(setup.capture, "capture_reference", reference)
    monkeypatch.setattr(setup.engineering, "run_engineering", engineering)
    return calls


def summary(root):
    return read(next((root / "setup").glob("*/summary.json")))


def test_production_encoding_preserves_real5120_width_and_exact_float32_bytes():
    value = np.zeros((288, 1, 2, 5120), dtype=np.float32)
    value[111, 0, 1, 382] = -1.25
    encoded = setup.encode_historical(value)
    assert np.array_equal(setup.decode_historical(encoded), value)
    assert encoded["raw_byte_sha256"] == hashlib.sha256(value.tobytes()).hexdigest()
    with pytest.raises(ValueError):
        setup.encode_historical(value.astype(np.float64))


@pytest.mark.parametrize("problem", ["shape", "hash", "base64", "trailing", "oversize", "nonfinite", "extra"])
def test_rejects_encoded_corruption_even_when_outer_manifest_could_be_rehashed(inputs, problem):
    manifest, _ = inputs
    encoded = deepcopy(manifest["historical_hidden"])
    if problem == "shape":
        encoded["shape"][0] = True
    elif problem == "hash":
        encoded["raw_byte_sha256"] = "f" * 64
    elif problem == "base64":
        encoded["data"] = "not base64!"
    elif problem in ("trailing", "oversize"):
        raw = np.ones((288, 1, 2, 4), dtype=np.float32).tobytes()
        compressed = zlib.compress(raw) + zlib.compress(b"trailing") if problem == "trailing" else zlib.compress(raw + b"x")
        encoded["data"] = base64.b64encode(compressed).decode()
    elif problem == "nonfinite":
        bad = np.ones((288, 1, 2, 4), dtype=np.float32)
        bad[0, 0, 0, 0] = np.nan
        encoded["data"] = base64.b64encode(zlib.compress(bad.tobytes())).decode()
        encoded["raw_byte_sha256"] = hashlib.sha256(bad.tobytes()).hexdigest()
    else:
        encoded["replacement"] = True
    with pytest.raises(ValueError):
        setup.decode_historical(encoded)


@pytest.mark.parametrize("problem", ["reorder", "seed", "dose", "direction", "layer", "unit_hash",
                                     "assignment", "source", "engine", "budget", "provenance"])
def test_manifest_tampering_rejects_before_model_execution(inputs, problem):
    manifest, _ = inputs
    bad = deepcopy(manifest)
    if problem == "reorder":
        bad["requests"][0], bad["requests"][1] = bad["requests"][1], bad["requests"][0]
    elif problem == "seed":
        bad["requests"][1]["seed"] = bad["requests"][0]["seed"]
    elif problem == "dose":
        bad["source_units"]["31"]["dose"] = 2
    elif problem == "direction":
        bad["source_units"]["31"]["orientation"] = -1
    elif problem == "layer":
        bad["source_units"]["31"]["layer"] = 15
    elif problem == "unit_hash":
        bad["source_units"]["31"]["source_unit_sha256"] = "f" * 64
    elif problem == "assignment":
        bad["diagnostics"][-1]["source_row"] = 0
    elif problem == "source":
        bad["historical_binding"]["source_manifest_sha256"] = "f" * 64
    elif problem == "engine":
        bad["engine_sha256"]["state_backend.py"] = "f" * 64
    elif problem == "budget":
        bad["budget"]["stage_ceiling_usd"] += 1
    else:
        bad.update(provenance={"claimed": True}, provenance_sha256="f" * 64)
    bad["source_units_sha256"] = digest(bad["source_units"])
    bad["historical_binding_sha256"] = digest(bad["historical_binding"])
    with pytest.raises(ValueError):
        setup.validate_manifest(bad)


def test_complete_execution_checkpoints544_rows_and_actual_physical_work(inputs, monkeypatch, tmp_path):
    manifest, model = inputs
    calls = install_collectors(monkeypatch, manifest)
    checkpoints = []

    def checkpoint():
        rows = RunStore(tmp_path, manifest).completed()
        checkpoints.append(len(rows))
        # Published records never refer to unpublished reports or NPZ bytes.
        for row in rows.values():
            assert (tmp_path / row["report_artifact"]["path"]).exists()
            assert (tmp_path / row["activation_artifact"]["path"]).exists()

    records = setup.execute(tmp_path, manifest, model, checkpoint=checkpoint)
    assert calls == ["reference", 31, 47] and checkpoints == [288, 416, 544, 544]
    assert len(records) == 544 and all(r["status"] == "measured" for r in records.values())
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from collection_coverage import check_collection_coverage

    assert check_collection_coverage(tmp_path, expected_manifest_sha256=digest(manifest))["passed"]
    assert sorted(int(path.stem) for path in (tmp_path / "cohorts").glob("*/*.json")) == list(range(0, 544, 32))
    assert all(r["engineering_passed"] and not r["scientific_admission"] for r in records.values())
    result = summary(tmp_path)
    assert result["passed"] and result["unavailable"] == 0 and not result["full_cap_certified"]
    reference = read(tmp_path / result["reports"]["reference"]["path"])
    assert reference["performance"]["attempted_model_forwards"] == 18
    for site in (31, 47):
        evidence = read(tmp_path / result["reports"][f"layer{site}"]["path"])
        assert evidence["performance"]["attempted_model_forwards"] == 8
        assert evidence["performance"]["generated_tokens"] == 256
        with np.load(tmp_path / evidence["vector_artifact"]["path"], allow_pickle=False) as data:
            assert set(data.files) == {"before", "after", "direction_float32"}
    assert "544/544" in setup.render(manifest, records)
    with pytest.raises(ValueError, match="cannot be rerun"):
        setup.execute(tmp_path, manifest, model)
    assert calls == ["reference", 31, 47]


def test_measured_failed_check_remains_available_and_does_not_hide_other_site(inputs, monkeypatch, tmp_path):
    manifest, model = inputs
    calls = install_collectors(monkeypatch, manifest, site_failed=31)
    records = setup.execute(tmp_path, manifest, model)
    assert calls == ["reference", 31, 47]
    assert not summary(tmp_path)["passed"] and summary(tmp_path)["unavailable"] == 0
    rows = [r for r in records.values() if r["site"] == 31]
    assert len(rows) == 128 and all(r["status"] == "measured" and not r["engineering_passed"] for r in rows)


def test_failed_reference_is_saved_but_no_site_is_executed(inputs, monkeypatch, tmp_path):
    manifest, model = inputs
    calls = install_collectors(monkeypatch, manifest, reference_passed=False)
    records = setup.execute(tmp_path, manifest, model)
    assert calls == ["reference"] and len(records) == 544
    assert not summary(tmp_path)["passed"] and summary(tmp_path)["unavailable"] == 256
    assert all(r["status"] == "prerequisite_failed" and r["measurement_sha256"] is None
               for r in records.values() if r["site"] is not None)


def test_partial_site_exception_preserves_reference_and_other_site_without_fabricating_counts(inputs, monkeypatch, tmp_path):
    manifest, model = inputs
    calls = install_collectors(monkeypatch, manifest, site_error=31)
    records = setup.execute(tmp_path, manifest, model)
    result = summary(tmp_path)
    assert calls == ["reference", 31, 47] and result["unavailable"] == 128 and not result["passed"]
    failed = read(tmp_path / result["reports"]["layer31"]["path"])
    assert failed["harness"] is None and failed["vector_artifact"] is None
    assert failed["performance"]["attempted_model_forwards"] is None
    assert all(r["status"] == "execution_error" and r["measurement_sha256"] is None
               for r in records.values() if r["site"] == 31)


@pytest.mark.parametrize("reason", ["deadline", "environment"])
def test_startup_failure_retains_all544_unavailable_without_collector_calls(inputs, monkeypatch, tmp_path, reason):
    manifest, model = inputs
    calls = install_collectors(monkeypatch, manifest)
    if reason == "environment":
        model.metadata = lambda: {**setup.MODEL, "dtype": "float32"}
    records = setup.execute(tmp_path, manifest, model, deadline=0 if reason == "deadline" else None)
    assert not calls and len(records) == 544 and summary(tmp_path)["unavailable"] == 544
    assert not summary(tmp_path)["passed"]


def test_interruption_after_reference_keeps_atomic_checkpoint_and_forbids_resume(inputs, monkeypatch, tmp_path):
    manifest, model = inputs
    calls = install_collectors(monkeypatch, manifest)

    def interrupt():
        raise SystemExit("simulated process interruption")

    with pytest.raises(SystemExit):
        setup.execute(tmp_path, manifest, model, checkpoint=interrupt)
    assert calls == ["reference"] and len(RunStore(tmp_path, manifest).completed()) == 288
    assert not list((tmp_path / "setup").glob("*/summary.json"))
    with pytest.raises(ValueError, match="cannot be rerun"):
        setup.execute(tmp_path, manifest, model)
    assert calls == ["reference"]


def test_real_cpu_executor_connects_reference_and_both_site_harnesses(inputs, monkeypatch, tmp_path):
    """Exercise the actual collectors at31/47, reducing only NEW wrapper size/cap."""
    import torch

    from curiosity_study.state_backend import StateBackend
    from test_causal_site_engineering import inputs as real_requests, model_and_direction

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(setup, "HIDDEN_SIZE", 16)
        monkeypatch.setattr(setup, "CAP", 3)
        manifest, _ = inputs
        model, direction = model_and_direction()
        requests = real_requests(288)
        model.tokenizer.encode = lambda label, **kwargs: [100 + "ABCD".index(label)]
        for request in requests:
            request["labels"] = "ABCD"
        original = np.concatenate([StateBackend(model).forward(requests[i:i + 32], layers=[15])[1]
                                   for i in range(0, 288, 32)])
        manifest.update(cap=3, requests=requests, requests_sha256=digest(requests),
                        historical_hidden=setup.encode_historical(original))
        manifest["historical_binding"].update(requests_sha256=digest(requests),
                                              hidden=setup.engineering.array_binding(original))
        manifest["historical_binding_sha256"] = digest(manifest["historical_binding"])
        for binding in manifest["source_units"].values():
            binding.update(unit=direction.tolist(), unit_sha256=digest(direction.tolist()),
                           source_unit_sha256=digest(direction.tolist()))
        manifest["source_units_sha256"] = digest(manifest["source_units"])
        manifest["diagnostics"] = setup.assignment_plan(requests)
        manifest["assignment_sha256"] = digest(manifest["diagnostics"])
        # Hardware provenance is synthetic here; actual torch forwards remain CPU BF16.
        model.metadata = lambda: dict(**setup.MODEL, dtype="bfloat16", attention="eager",
                                      device="cuda", gpu="SYNTHETIC H200 metadata, CPU execution")
        records = setup.execute(tmp_path, manifest, model)
        result = summary(tmp_path)
        assert len(records) == 544 and result["passed"] and result["unavailable"] == 0
        for layer in (31, 47):
            wrapper = read(tmp_path / result["reports"][f"layer{layer}"]["path"])
            assert wrapper["harness"]["checks"]["independently_recomputed_geometry"]
            assert wrapper["harness"]["checks"]["full_sham_identity"]
            assert wrapper["harness"]["checks"]["every_perturbed_event_engaged"]
        assert not setup.engineering.hook_inventory(model) and model.active_hooks == 0
    finally:
        torch.set_num_threads(previous)


def test_actual_closed_source_and_cached_tokenizer_match_production_setup_contract(monkeypatch):
    """Read only the completed S source/candidate artifacts; no C outputs or inference."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from satisfaction_site_sources import historical_baseline, setup_units
    from prepare_satisfaction_screen import tokenizer_model
    from curiosity_study import satisfaction_capture as original

    parent, requests, hidden, historical, provenance = historical_baseline(root, reproduce=False)
    units, _ = setup_units(root)
    rendered = original.bind_preflight(parent["screen_plan"], tokenizer_model())
    assert [item["request"] for item in rendered["requests"]] == requests
    assert {type(row["labels"]) for row in requests} == {str}
    rows = setup.assignment_plan(requests)
    manifest = dict(
        assay=setup.ASSAY, phase="engineering", scientific_admission=False, full_cap_certified=False,
        model=setup.MODEL, execution=setup.EXECUTION, execution_gpu="H200", arithmetic=setup.ARITHMETIC,
        schedule=setup.SCHEDULE, sites=[31, 47], cap=128, original_dose=setup.study.ORIGINAL_S_DOSE,
        budget=setup.resources.POLICIES["setup"], requests=requests, requests_sha256=digest(requests),
        historical_hidden=setup.encode_historical(hidden), historical_binding=historical,
        historical_binding_sha256=digest(historical), source_units=units, source_units_sha256=digest(units),
        assignments=[], diagnostics=rows, assignment_sha256=digest(rows), engine_sha256=setup.engine_hashes(),
        provenance=provenance, provenance_sha256=digest(provenance),
    )
    assert setup.validate_manifest(manifest) == rows
    assert len(rows) == 544 and hidden.shape == (288, 1, 2, 5120)
