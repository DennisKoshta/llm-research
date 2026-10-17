"""Durable, unscored setup for the two prospectively fixed satisfaction sites.

The 544 assigned rows are 288 read-only references and 256 short engineering
dispositions. Identity forwards and the malformed-direction guard are additional
physical work, not extra scientific observations. This module grants no admission.
"""

import base64
import binascii
import hashlib
import math
import os
from pathlib import Path
import time
import uuid
import zlib

import numpy as np

from . import causal_site_engineering as engineering
from . import satisfaction_causal_sites as study
from . import satisfaction_site_capture as capture
from . import satisfaction_site_resources as resources
from .artifacts import RunStore, digest, write_once
from .concept_capture import MODEL

ASSAY = resources.SETUP_ASSAY
HIDDEN_SIZE = 5120
REFERENCE_ROWS = 288
CONDITIONS = ("baseline", "zero_sham", "one_row_nonzero", "clean_repeat")
EXECUTION = dict(backend="torch_batched_v1", batch_size=32)
ARITHMETIC = engineering.ARITHMETIC
SCHEDULE = engineering.SCHEDULE
CAP = 128
require = engineering.require


def encode_historical(array):
    """Encode the exact original float32 record/final values without conversion."""
    value = np.asarray(array)
    require(value.dtype == np.float32 and value.shape == (REFERENCE_ROWS, 1, 2, HIDDEN_SIZE)
            and np.isfinite(value).all(), "Malformed historical array")
    raw = np.ascontiguousarray(value).tobytes()
    return dict(encoding="zlib_base64", dtype="float32", shape=list(value.shape),
                raw_byte_sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b64encode(zlib.compress(raw)).decode("ascii"))


def decode_historical(binding):
    """Bound decompression, reject trailing streams, and verify actual array bytes."""
    shape = [REFERENCE_ROWS, 1, 2, HIDDEN_SIZE]
    size = math.prod(shape) * 4
    require(isinstance(binding, dict)
            and set(binding) == {"encoding", "dtype", "shape", "raw_byte_sha256", "data"}
            and binding["encoding"] == "zlib_base64" and binding["dtype"] == "float32"
            and binding["shape"] == shape and all(type(x) is int for x in binding["shape"])
            and study.valid_sha(binding["raw_byte_sha256"])
            and isinstance(binding["data"], str)
            and len(binding["data"]) <= 4 * ((size + 65536 + 2) // 3),
            "Malformed encoded historical array")
    try:
        compressed = base64.b64decode(binding["data"], validate=True)
        stream = zlib.decompressobj()
        raw = stream.decompress(compressed, size + 1)
    except (ValueError, binascii.Error, zlib.error) as error:
        raise ValueError("Invalid compressed historical array") from error
    require(len(raw) == size and stream.eof and not stream.unused_data and not stream.unconsumed_tail
            and hashlib.sha256(raw).hexdigest() == binding["raw_byte_sha256"],
            "Historical array size, stream or byte binding changed")
    value = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    require(np.isfinite(value).all(), "Nonfinite historical array")
    return value


def assignment_plan(requests):
    """Keep reference288 and both sites' four first32 conditions explicitly assigned."""
    capture._requests(requests, REFERENCE_ROWS)
    rows = []
    for index, request in enumerate(requests):
        rows.append(dict(trial_id=f"{ASSAY}:reference:{index:03d}", kind="engineering",
                         stage="reference", site=None, condition="read_only_reference",
                         source_row=index, source_trial_id=request["source_trial_id"],
                         request_sha256=digest(request), cap=0))
    for site in study.SITES:
        for condition in CONDITIONS:
            for index, request in enumerate(requests[:32]):
                rows.append(dict(trial_id=f"{ASSAY}:layer{site}:{condition}:{index:02d}",
                                 kind="engineering", stage="engineering", site=site,
                                 condition=condition, source_row=index,
                                 source_trial_id=request["source_trial_id"],
                                 request_sha256=digest(request), cap=CAP))
    return rows


def engine_hashes():
    folder = Path(__file__).parent
    names = (
        "satisfaction_site_setup.py", "satisfaction_site_capture.py", "causal_site_engineering.py",
        "satisfaction_causal_sites.py", "satisfaction_site_resources.py", "sustained_engineering.py",
        "voluntary_capture.py", "state_backend.py", "model.py", "batched.py",
    )
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}


def validate_manifest(manifest):
    require(manifest.get("assay") == ASSAY and manifest.get("phase") == "engineering"
            and manifest.get("scientific_admission") is False and manifest.get("full_cap_certified") is False
            and manifest.get("model") == MODEL and manifest.get("execution") == EXECUTION
            and manifest.get("execution_gpu") == "H200" and manifest.get("arithmetic") == ARITHMETIC
            and manifest.get("schedule") == SCHEDULE and manifest.get("sites") == list(study.SITES)
            and type(manifest.get("cap")) is int and manifest["cap"] == CAP
            and type(manifest.get("original_dose")) in (int, float)
            and manifest["original_dose"] == study.ORIGINAL_S_DOSE,
            "Unsupported fixed two-site setup contract")
    resources.job_wall_limit(manifest)
    requests = manifest["requests"]
    expected = assignment_plan(requests)
    require(manifest.get("requests_sha256") == digest(requests)
            and len({row["seed"] for row in requests}) == REFERENCE_ROWS
            and all(row.get("labels") == "ABCD" for row in requests),
            "Require all288 distinct original prompts and sampling seeds")
    historical = decode_historical(manifest["historical_hidden"])
    binding = manifest["historical_binding"]
    keys = {"source_manifest_sha256", "source_audit_sha256", "record_bindings_sha256",
            "artifact_bindings_sha256", "requests_sha256", "hidden"}
    require(set(binding) == keys and binding["source_manifest_sha256"] == study.PRIOR_MANIFEST_SHA256
            and all(study.valid_sha(binding[key]) for key in keys - {"hidden"})
            and binding["requests_sha256"] == digest(requests)
            and binding["hidden"] == engineering.array_binding(historical)
            and manifest.get("historical_binding_sha256") == digest(binding),
            "Historical source, request order or original vector binding changed")
    units = manifest["source_units"]
    require(set(units) == {str(site) for site in study.SITES}
            and manifest.get("source_units_sha256") == digest(units), "Both original site units required")
    for site in study.SITES:
        item = units[str(site)]
        unit = item["unit"]
        require(item["source"] == "satisfaction_vs_neutral" and item["corpus"] == "A"
                and type(item["layer"]) is int and item["layer"] == site
                and item["position"] == "record" and type(item["orientation"]) is int
                and item["orientation"] == 1 and "dose" not in item
                and type(item["source_raw_norm"]) in (int, float)
                and math.isfinite(item["source_raw_norm"]) and item["source_raw_norm"] > 0
                and item["directions_byte_sha256"] == study.LOCKED_DIRECTION_BYTES
                and isinstance(unit, list) and len(unit) == HIDDEN_SIZE
                and all(type(x) in (int, float) and math.isfinite(x) for x in unit)
                and math.isclose(math.sqrt(math.fsum(x * x for x in unit)), 1, abs_tol=1e-6)
                and item["source_unit_sha256"] == item["unit_sha256"] == digest(unit),
                "Require unchanged positive A satisfaction unit at each specified site")
    require(manifest.get("assignments") == [] and manifest.get("diagnostics") == expected
            and manifest.get("assignment_sha256") == digest(expected)
            and manifest.get("engine_sha256") == engine_hashes(),
            "Setup assignment inventory or implementation bytes changed")
    for optional in ("source_certificate", "provenance"):
        if optional in manifest or f"{optional}_sha256" in manifest:
            require(isinstance(manifest.get(optional), dict)
                    and manifest.get(f"{optional}_sha256") == digest(manifest[optional]),
                    "Optional provenance content binding changed")
    return expected


def _artifact(root, relative):
    path = Path(root) / relative
    return dict(path=str(relative), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                byte_count=path.stat().st_size)


def _save_arrays(root, relative, arrays):
    destination = Path(root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        require(not destination.exists(), "Setup arrays already exist")
        os.rename(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {**_artifact(root, relative),
            "array_bindings": {name: engineering.array_binding(value) for name, value in arrays.items()}}


def _exception(error):
    return dict(type=type(error).__name__, message=str(error))


def _deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Site setup deadline elapsed before the next stage")


def _persist_stage(root, manifest, store, attempt, assignments, *, name, harness, arrays,
                   error, prerequisite_failed, environment_artifact, elapsed):
    vector_artifact = None
    if arrays is not None:
        vector_artifact = _save_arrays(root, Path("setup") / attempt / name / "arrays.npz", arrays)
    measured = harness is not None and error is None and not prerequisite_failed
    status = "measured" if measured else "prerequisite_failed" if prerequisite_failed else "execution_error"
    passed = measured and harness.get("passed") is True
    if measured and name == "reference":
        physical_forwards = 2 * len(harness["cohorts"])
        counts = dict(read_only_capture_forwards=len(harness["cohorts"]),
                      read_only_identity_forwards=len(harness["cohorts"]), generated_tokens=0,
                      attempted_model_forwards=physical_forwards)
    elif measured:
        by_condition = {key: value["attempted_forwards"] for key, value in harness["conditions"].items()}
        counts = dict(forwards_by_condition=by_condition,
                      malformed_direction_attempts=harness["malformed_direction"]["attempted_forwards"],
                      attempted_model_forwards=sum(by_condition.values())
                      + harness["malformed_direction"]["attempted_forwards"],
                      generated_tokens=sum(len(row["completion"]["generated_token_ids"])
                                           for condition in harness["conditions"].values()
                                           for row in condition["rows"]))
    else:
        counts = dict(attempted_model_forwards=0 if prerequisite_failed else None, generated_tokens=None,
                      limitation="Unexpected helper exceptions do not expose partial forward counts; unknown is not zero.")
    wrapper = dict(schema_version=1, assay=ASSAY, manifest_sha256=digest(manifest), attempt=attempt,
                   stage=name, status=status, assigned=len(assignments), passed=passed,
                   scientific_admission=False, full_cap_certified=False, harness=harness, error=error,
                   vector_artifact=vector_artifact, environment_artifact=environment_artifact,
                   elapsed_seconds=elapsed, performance=counts)
    relative = Path("setup") / attempt / name / "report.json"
    write_once(Path(root) / relative, wrapper)
    report_artifact = _artifact(root, relative)
    parent_indices = {a["trial_id"]: i for i, a in enumerate(manifest["diagnostics"])}
    for offset in range(0, len(assignments), 32):
        records = []
        for assignment in assignments[offset:offset + 32]:
            row = None
            if measured:
                index = assignment["source_row"]
                row = (harness["cohorts"][index // 32]["measurements"][index % 32]
                       if name == "reference" else harness["conditions"][assignment["condition"]]["rows"][index])
            record = dict(trial_id=assignment["trial_id"], assignment=assignment, kind="engineering",
                          terminal=True, status=status, stage=assignment["stage"], site=assignment["site"],
                          condition=assignment["condition"], source_row=assignment["source_row"],
                          source_trial_id=assignment["source_trial_id"], request_sha256=assignment["request_sha256"],
                          engineering_passed=passed, scientific_admission=False, full_cap_certified=False,
                          measurement_sha256=None if row is None else digest(row),
                          report_artifact=report_artifact, activation_artifact=vector_artifact)
            if not measured:
                record.update(failure="execution_error", error_type=error["type"], error=error["message"])
            records.append(record)
        absolute = [parent_indices[record["trial_id"]] for record in records]
        require(absolute == list(range(absolute[0], absolute[0] + len(records))),
                "Setup cohort must preserve contiguous parent assignment order")
        store.save_many(records, attempt, f"{absolute[0]:07d}")
    return wrapper, report_artifact


def execute(root, manifest, model, checkpoint=None, deadline=None):
    """Persist each setup stage before moving on; an observed attempt cannot resume."""
    assignments = validate_manifest(manifest)
    root = Path(root)
    store = RunStore(root, manifest)
    require(not (root / "setup_started.json").exists() and not store.completed()
            and not any((root / "setup").glob("*/*"))
            and len(list((root / "admissions").glob("*.json"))) <= 1,
            "A started or partial setup cannot be rerun")
    attempt = uuid.uuid4().hex
    write_once(root / "setup_started.json", dict(manifest_sha256=digest(manifest), attempt=attempt))
    environment_artifact, pre_error = None, None
    try:
        _deadline(deadline)
        environment = model.metadata()
        relative = Path("environments") / f"{attempt}.json"
        write_once(root / relative, environment)
        environment_artifact = {**_artifact(root, relative), "content_sha256": digest(environment)}
        require(all(environment.get(key) == value for key, value in MODEL.items())
                and environment.get("dtype") == "bfloat16" and environment.get("attention") == "eager"
                and environment.get("device") == "cuda" and "H200" in str(environment.get("gpu"))
                and model.model.config.hidden_size == HIDDEN_SIZE,
                "Actual setup model or arithmetic differs from the frozen contract")
    except Exception as caught:
        pre_error = _exception(caught)
    reports = {}
    started = time.monotonic()
    reference, arrays, error = None, None, pre_error
    if error is None:
        try:
            reference, arrays = capture.capture_reference(
                model, manifest["requests"], historical_layer15_hidden=decode_historical(manifest["historical_hidden"]),
                historical_binding=manifest["historical_binding"], original_dose=manifest["original_dose"],
                deadline=deadline,
            )
        except Exception as caught:
            error = _exception(caught)
    reference_wrapper, artifact = _persist_stage(
        root, manifest, store, attempt, assignments[:REFERENCE_ROWS], name="reference",
        harness=reference, arrays=arrays, error=error, prerequisite_failed=False,
        environment_artifact=environment_artifact, elapsed=time.monotonic() - started,
    )
    reports["reference"] = artifact
    if checkpoint:
        checkpoint()
    dose_error = None
    if reference_wrapper["passed"]:
        try:
            study.validate_reference(reference["dose_reference"])
            require(reference["dose_reference"]["capture_sha256"] == digest(reference["capture_inventory"])
                    and reference["doses"] == reference["dose_reference"]["doses"],
                    "Reference capture and derived doses differ")
        except Exception as caught:
            dose_error = _exception(caught)
    ready = reference_wrapper["passed"] and dose_error is None
    results = [reference_wrapper]
    for site in study.SITES:
        name, started = f"layer{site}", time.monotonic()
        harness, arrays, error = None, None, None
        if not ready:
            error = dose_error or dict(type="ReferenceNotPassed", message="Read-only reference did not pass; no site forward attempted")
        else:
            try:
                _deadline(deadline)
                require(model.active_hooks == 0 and not engineering.hook_inventory(model), "Setup hook cleanup failed")
                harness, arrays = engineering.run_engineering(
                    model, manifest["requests"][:32], layer=site,
                    direction=np.asarray(manifest["source_units"][str(site)]["unit"], dtype=np.float64),
                    dose=reference["doses"][str(site)], cap=CAP, deadline=deadline,
                )
            except Exception as caught:
                error = _exception(caught)
        wrapper, artifact = _persist_stage(
            root, manifest, store, attempt, [a for a in assignments if a["site"] == site], name=name,
            harness=harness, arrays=arrays, error=error, prerequisite_failed=not ready,
            environment_artifact=environment_artifact, elapsed=time.monotonic() - started,
        )
        reports[name] = artifact
        results.append(wrapper)
        if checkpoint:
            checkpoint()
    completed = store.completed()
    require(set(completed) == {a["trial_id"] for a in assignments}, "Setup terminal inventory differs")
    write_once(root / "setup" / attempt / "summary.json", dict(
        schema_version=1, assay=ASSAY, manifest_sha256=digest(manifest), attempt=attempt,
        complete=True, assigned=544, terminal=len(completed), passed=ready and all(r["passed"] for r in results),
        scientific_admission=False, full_cap_certified=False, reports=reports, dose_error=dose_error,
        unavailable=sum(r["status"] != "measured" for r in completed.values()),
        physical_work_note="Reference288 uses nine capture plus nine identity forwards. Engineering256 uses two sites × four32-row conditions; the malformed-direction guard adds no assigned row.",
        limitation="Observed cap128 evidence is not full4096 new-site certification; original source/direction provenance and later scientific execution require independent audit.",
    ))
    if checkpoint:
        checkpoint()
    return completed


def render(manifest, records):
    passed = len(records) == 544 and all(row.get("engineering_passed") is True for row in records.values())
    return (f"Two-site setup: {len(records)}/544 terminal dispositions; recorded checks passed: {passed}.\n\n"
            "Read-only dose reference and observed128-token checks only. No scientific admission or full-cap certification.")
