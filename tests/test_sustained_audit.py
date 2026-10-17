"""Synthetic immutable evidence tests; no research outputs or GPU execution."""

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from curiosity_study import sustained_engineering as engine
from curiosity_study.artifacts import digest, read, write_once
from test_sustained_engineering import model_and_direction
from test_sustained_execution import manifest as base_manifest, model as metadata_model

spec = importlib.util.spec_from_file_location(
    "audit_sustained", Path(__file__).parents[1] / "scripts/audit_sustained_engineering.py"
)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def packed(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def source_documents(monkeypatch):
    manifest = base_manifest()
    requests = manifest["requests"]
    source_assignments = [dict(trial_id=r["source_trial_id"], sampling_seed=r["seed"]) for r in requests]
    records = []
    for index, (a, request) in enumerate(zip(source_assignments, requests)):
        records.append(
            dict(
                trial_id=a["trial_id"],
                assignment=a,
                activation_artifact=dict(
                    positions=["record", "prefill"],
                    token_positions=list(request["prefix_positions"].values()),
                ),
                inference=dict(
                    unpadded_input_ids=request["token_prefix"],
                    sampling=dict(seed=a["sampling_seed"]),
                    batch=dict(row=index, size=32, cohort_ids=[r["source_trial_id"] for r in requests]),
                ),
            )
        )
    source_assignments.extend(dict(trial_id=f"unused-source-{i}", sampling_seed=i) for i in range(32, 288))
    unit = np.asarray(manifest["direction"]["unit"], dtype=manifest["direction"]["unit_dtype"])
    stream = io.BytesIO()
    np.savez(
        stream,
        sources=np.array(["curiosity_vs_indifference"]),
        layers=np.array([15]),
        positions=np.array(["record"]),
        unit=unit.reshape(1, 1, 1, -1),
        contrast_norm=np.array([[[0.25]]]),
    )
    vector_bytes = stream.getvalue()
    vector_sha = hashlib.sha256(vector_bytes).hexdigest()
    candidate = dict(directions_byte_sha256=vector_sha)
    source = dict(
        assay="voluntary_resolution",
        task_revision="voluntary-resolution-v2",
        model=engine.MODEL,
        source_sha256="synthetic-source",
        assignments=[],
        diagnostics=source_assignments,
        candidate_source=candidate,
    )
    monkeypatch.setattr(auditor, "SOURCE_MANIFEST_SHA256", digest(source))
    archive_bytes = packed({"candidate/directions.npz": vector_bytes})
    source_freeze = dict(
        manifest_sha256=digest(source),
        source_sha256=source["source_sha256"],
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        documents={"candidate/directions.npz": vector_sha},
    )
    source_audit = dict(
        manifest_sha256=digest(source),
        source_sha256=source["source_sha256"],
        complete=True,
        integrity_passed=True,
        readiness=dict(passed=False),
        source_freeze=source_freeze,
        candidate_source=candidate,
        outcomes=[dict(trial_id=r["trial_id"], record_sha256=digest(r)) for r in records]
        + [dict(trial_id=a["trial_id"], record_sha256=digest(a)) for a in source_assignments[32:]],
    )
    envelope = dict(manifest_hash=digest(source), records=records, sha256=digest(records))
    documents = {
        "runs/source/manifest.json": encoded(source),
        "reports/source-audit.json": encoded(source_audit),
        "runs/source/freeze.json": encoded(source_freeze),
        "runs/source/source.tar.gz": archive_bytes,
        "runs/source/cohorts/attempt/0000000.json": encoded(envelope),
    }
    manifest["source_binding"] = dict(
        run="runs/source",
        manifest_path="runs/source/manifest.json",
        manifest_sha256=digest(source),
        audit_path="reports/source-audit.json",
        audit_sha256=digest(source_audit),
        audit_byte_sha256=hashlib.sha256(encoded(source_audit)).hexdigest(),
        freeze_path="runs/source/freeze.json",
        freeze_sha256=digest(source_freeze),
        archive_path="runs/source/source.tar.gz",
        archive_byte_sha256=source_freeze["archive_sha256"],
        first_cohort_path="runs/source/cohorts/attempt/0000000.json",
        first_cohort_sha256=digest(envelope),
        first_cohort_byte_sha256=hashlib.sha256(encoded(envelope)).hexdigest(),
        readiness_passed=False,
        records=[
            dict(trial_id=r["trial_id"], record_sha256=digest(r), assignment_sha256=digest(r["assignment"]))
            for r in records
        ],
    )
    manifest["direction"].update(
        directions_document="candidate/directions.npz",
        directions_byte_sha256=vector_sha,
        candidate_source=candidate,
    )
    manifest["compatibility_sha256"] = digest(engine.compatibility_binding(manifest))
    return manifest, documents


def freeze_fixture(root, manifest, documents):
    root.mkdir(parents=True)
    manifest["document_bindings"] = [
        dict(path=name, byte_sha256=hashlib.sha256(payload).hexdigest())
        for name, payload in sorted(documents.items())
    ]
    implementation = {
        "src/curiosity_study/" + name: (Path(engine.__file__).parent / name).read_bytes()
        for name in engine.engine_hashes()
    }
    manifest["source_sha256"] = digest({name: value.decode() for name, value in implementation.items()})
    archive = packed({**documents, **implementation})
    (root / "source.tar.gz").write_bytes(archive)
    write_once(root / "manifest.json", manifest)
    (root / "freeze.json").write_bytes(
        encoded(
            dict(
                manifest_sha256=digest(manifest),
                source_sha256=manifest["source_sha256"],
                archive_sha256=hashlib.sha256(archive).hexdigest(),
                documents={b["path"]: b["byte_sha256"] for b in manifest["document_bindings"]},
            )
        )
    )


@pytest.fixture(scope="module")
def evidence():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, _ = model_and_direction()
        model.model_id, model.revision = engine.MODEL["model_id"], engine.MODEL["revision"]
        model.tokenizer.encode = lambda label, **kwargs: [100 + "ABCD".index(label)]
        model.model.generation_config.eos_token_id = 127
        forward = model.model.forward

        def finite_trajectory(*args, **kwargs):
            cache = kwargs.get("past_key_values")
            step = 0 if cache is None else int(cache.get_seq_length()) - 128 + 1
            result = forward(*args, **kwargs)
            logits = torch.full_like(result.logits, -1000)
            logits[:, -1, 127 if step == 3 else 7] = 1000
            result.logits = logits
            return result

        model.model.forward = finite_trajectory
        manifest = base_manifest()
        return engine.run_engineering(
            model,
            manifest["requests"],
            direction=np.asarray(manifest["direction"]["unit"]),
            dose=0.25,
            cap=128,
        )
    finally:
        torch.set_num_threads(previous)


def measured_run(tmp_path, monkeypatch, evidence, *, failed=False):
    manifest, documents = source_documents(monkeypatch)
    root = tmp_path / "runs" / "short"
    freeze_fixture(root, manifest, documents)
    harness, arrays = deepcopy(evidence)
    if failed:
        harness["conditions"]["zero_sham"]["forwards"][1]["full_fp32_logits_sha256"][17] = "0" * 64
        harness["checks"]["full_sham_identity"] = False
        harness["passed"] = False
    monkeypatch.setattr(engine, "run_engineering", lambda *a, **k: (harness, arrays))
    engine.execute(root, manifest, metadata_model())
    return root, manifest, documents


def test_full_artifact_audit_uses_real_geometry_and_does_not_gate_failed_source_readiness(
    tmp_path, monkeypatch, evidence
):
    root, _, _ = measured_run(tmp_path, monkeypatch, evidence)
    result = auditor.audit(root)
    assert result["passed"] and result["complete"] and result["integrity_passed"]
    assert not result["scientific_admission"] and result["source"]["source_readiness_passed"] is False
    assert result["observed_maximum_tokens"] == 4
    assert result["observed_lengths_by_condition"] == {name: [4] * 32 for name in engine.CONDITIONS}
    assert result["performance"]["forward_counts_by_condition"] == {name: 4 for name in engine.CONDITIONS}
    assert result["performance"]["generated_tokens_by_condition"] == {name: 128 for name in engine.CONDITIONS}
    assert result["performance"]["total_forward_count"] == 16
    assert result == auditor.audit(root)


def test_complete_false_engineering_stays_failed_despite_correct_terminal_coverage(
    tmp_path, monkeypatch, evidence
):
    root, _, _ = measured_run(tmp_path, monkeypatch, evidence, failed=True)
    result = auditor.audit(root)
    assert result["complete"] and result["integrity_passed"] and result["measurement_status"] == "measured"
    assert not result["passed"] and not result["checks"]["full_sham_identity"]


def test_forged_pass_boolean_cannot_override_actual_logit_fingerprints(evidence):
    harness, arrays = deepcopy(evidence)
    harness["conditions"]["zero_sham"]["forwards"][1]["full_fp32_logits_sha256"][17] = "0" * 64
    m = base_manifest()
    with pytest.raises(ValueError, match="pass flags"):
        auditor.recompute_checks(
            harness, arrays, m["requests"], np.asarray(m["direction"]["unit"]), 0.25, 128
        )


@pytest.mark.parametrize("field", ["source_order", "token", "seed", "position", "candidate"])
def test_source_and_candidate_changes_fail_independent_reconstruction(monkeypatch, field):
    m, docs = source_documents(monkeypatch)
    if field == "source_order":
        m["requests"][0], m["requests"][1] = m["requests"][1], m["requests"][0]
    elif field == "token":
        m["requests"][0]["token_prefix"][0] += 1
    elif field == "seed":
        m["requests"][0]["seed"] += 1
    elif field == "position":
        m["requests"][0]["prefix_positions"]["record"] += 1
    else:
        m["direction"]["unit"][0] *= -1
    with pytest.raises(ValueError):
        auditor.verify_sources(m, docs)


def test_modified_vector_archive_rejected_even_with_complete_rows(tmp_path, monkeypatch, evidence):
    root, _, _ = measured_run(tmp_path, monkeypatch, evidence)
    path = next((root / "engineering").glob("*/vectors.npz"))
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="Vector archive location/bytes"):
        auditor.audit(root)


def test_rehashed_source_record_cannot_override_bound_audit(monkeypatch):
    m, docs = source_documents(monkeypatch)
    source = m["source_binding"]
    envelope = json.loads(docs[source["first_cohort_path"]])
    record = envelope["records"][0]
    record["inference"]["unpadded_input_ids"][0] += 1
    envelope["sha256"] = digest(envelope["records"])
    payload = encoded(envelope)
    docs[source["first_cohort_path"]] = payload
    source["first_cohort_sha256"] = digest(envelope)
    source["first_cohort_byte_sha256"] = hashlib.sha256(payload).hexdigest()
    source["records"][0]["record_sha256"] = digest(record)
    request = m["requests"][0]
    request["token_prefix"] = record["inference"]["unpadded_input_ids"]
    request["token_prefix_sha256"] = digest(request["token_prefix"])
    with pytest.raises(ValueError, match="Audited source record"):
        auditor.verify_sources(m, docs)


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_source_audit_requires_full_unique_record_inventory(monkeypatch, change):
    m, docs = source_documents(monkeypatch)
    source = m["source_binding"]
    audit = json.loads(docs[source["audit_path"]])
    if change == "duplicate":
        audit["outcomes"][-1] = audit["outcomes"][0]
    else:
        audit["outcomes"].pop()
    payload = encoded(audit)
    docs[source["audit_path"]] = payload
    source["audit_sha256"] = digest(audit)
    source["audit_byte_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="all 288"):
        auditor.verify_sources(m, docs)


def test_rehashed_performance_metadata_cannot_override_observed_counts(tmp_path, monkeypatch, evidence):
    root, _, _ = measured_run(tmp_path, monkeypatch, evidence)
    path = next((root / "engineering").glob("*/report.json"))
    wrapper = read(path)
    wrapper["performance"]["forward_counts_by_condition"]["baseline"] += 1
    wrapper["performance"]["total_forward_count"] += 1
    path.write_bytes(encoded(wrapper))
    for cohort_path in (root / "cohorts").glob("*/*.json"):
        envelope = read(cohort_path)
        for row in envelope["records"]:
            row["report_artifact"].update(sha256=auditor.sha(path), byte_count=path.stat().st_size)
        envelope["sha256"] = digest(envelope["records"])
        cohort_path.write_bytes(encoded(envelope))
    with pytest.raises(ValueError, match="Performance descriptions"):
        auditor.audit(root)


@pytest.mark.parametrize("change", ["duplicate", "nested", "order"])
def test_duplicate_nested_or_reordered_cohorts_rejected(tmp_path, monkeypatch, evidence, change):
    root, _, _ = measured_run(tmp_path, monkeypatch, evidence)
    path = next((root / "cohorts").glob("*/0000000.json"))
    if change in ("duplicate", "nested"):
        dest = root / "cohorts" / "other" / ("nested" if change == "nested" else "") / path.name
        dest.parent.mkdir(parents=True)
        dest.write_bytes(path.read_bytes())
    else:
        envelope = read(path)
        envelope["records"].reverse()
        envelope["sha256"] = digest(envelope["records"])
        path.write_bytes(encoded(envelope))
    with pytest.raises(ValueError):
        auditor.audit(root)


def test_hard_kill_partial_collection_is_incomplete(tmp_path, monkeypatch, evidence):
    root, _, _ = measured_run(tmp_path, monkeypatch, evidence)
    next((root / "cohorts").glob("*/0000096.json")).unlink()
    result = auditor.audit(root)
    assert not result["complete"] and not result["passed"] and result["terminal_records"] == 96
    assert result["measurement_status"] == "incomplete"


def test_execution_errors_retain128_unavailable_rows(tmp_path, monkeypatch):
    m, docs = source_documents(monkeypatch)
    root = tmp_path / "error"
    freeze_fixture(root, m, docs)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(engine, "run_engineering", fail)
    engine.execute(root, m, metadata_model())
    result = auditor.audit(root)
    assert result["complete"] and result["terminal_records"] == 128 and not result["passed"]
    assert result["measurement_status"] == "unavailable_execution_error"


def test_long_stage_rechecks_short_npz_instead_of_accepting_pass_flag(tmp_path, monkeypatch, evidence):
    short, m, docs = measured_run(tmp_path, monkeypatch, evidence)
    short_audit = auditor.audit(short)
    long = deepcopy(m)
    long.update(cap=4096, diagnostics=engine.assignment_plan(long["requests"], 4096))
    long["assignment_sha256"] = digest(long["diagnostics"])
    stage = dict(
        run="runs/short",
        manifest_path="runs/short/manifest.json",
        manifest_sha256=digest(m),
        audit_path="reports/short-audit.json",
        audit_sha256=digest(short_audit),
        audit_byte_sha256=hashlib.sha256(encoded(short_audit)).hexdigest(),
        audit=short_audit,
    )
    long["short_stage"] = stage
    docs.update({stage["manifest_path"]: encoded(m), stage["audit_path"]: encoded(short_audit)})
    root = tmp_path / "runs" / "long"
    freeze_fixture(root, long, docs)
    path = next((short / "engineering").glob("*/vectors.npz"))
    path.write_bytes(path.read_bytes() + b"tampered after short audit")
    with pytest.raises(ValueError, match="Vector archive location/bytes"):
        auditor.audit(root)
