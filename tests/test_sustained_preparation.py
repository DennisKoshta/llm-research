"""Synthetic frozen-input admission only; no preparation/freezing or cloud calls."""

from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study.artifacts import canonical, digest, read, write_once
from curiosity_study.concept_capture import MODEL
from curiosity_study import sustained_engineering as engineering

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_sustained_engineering as prepare  # noqa: E402
from prepare_voluntary_resolution import candidate_binding  # noqa: E402


def json_bytes(value):
    return (canonical(value) + "\n").encode()


def candidate_payloads():
    unit = np.ones((2, 4, 2, 16), dtype=np.float64) / 4
    binary = BytesIO()
    np.savez_compressed(
        binary,
        unit=unit,
        contrast_norm=np.full((2, 4, 2), 0.25),
        sources=np.array(["curiosity_vs_indifference", "satisfaction_vs_neutral"]),
        layers=np.array([15, 31, 47, 57]),
        positions=np.array(["record", "final"]),
    )
    directions = binary.getvalue()
    direction_sha = hashlib.sha256(directions).hexdigest()
    manifest = dict(model=MODEL, source_sha256="candidate-source")
    audit = dict(passed=True, complete=True, assigned=384, manifest_sha256=digest(manifest))
    analysis = dict(
        assay="described_state_candidate_analysis",
        development_only=True,
        model=MODEL,
        source_sha256="candidate-source",
        manifest_sha256=digest(manifest),
        audit_sha256=digest(audit),
        directions_sha256=direction_sha,
        candidates=[
            dict(
                contrast=name,
                passed=True,
                a_selection_passed=True,
                b_transfer_passed=True,
                b_specificity_point_screen_passed=True,
                selected_layer=15,
                position="record",
                a_raw_contrast_norm=0.25,
            )
            for name in ("curiosity_vs_indifference", "satisfaction_vs_neutral")
        ],
    )
    analysis_sha = hashlib.sha256(json_bytes(analysis)).hexdigest()
    check = dict(
        passed=True,
        records=384,
        capture_manifest_sha256=digest(manifest),
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=analysis_sha,
        audit_sha256=digest(audit),
        directions_byte_sha256=direction_sha,
    )
    binding = candidate_binding(manifest, audit, analysis, check, analysis_sha, direction_sha)
    payloads = {
        key: json_bytes(value)
        for key, value in dict(manifest=manifest, audit=audit, analysis=analysis, check=check).items()
    }
    return {**payloads, "directions": directions}, binding


def source_tree(tmp_path, monkeypatch):
    source = tmp_path / "runs/source"
    source.mkdir(parents=True)
    payloads, candidate = candidate_payloads()
    assignments = [dict(trial_id=f"source-{i:03d}", sampling_seed=500 + i) for i in range(288)]
    manifest = dict(
        assay="voluntary_resolution",
        task_revision="voluntary-resolution-v2",
        model=MODEL,
        execution=engineering.EXECUTION,
        source_sha256="source",
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        candidate_source=candidate,
    )
    monkeypatch.setattr(prepare, "SOURCE_MANIFEST_SHA256", digest(manifest))
    write_once(source / "manifest.json", manifest)
    outcomes = []
    for offset in range(0, 288, 32):
        cohort = assignments[offset : offset + 32]
        records = [
            dict(
                trial_id=a["trial_id"],
                assignment=a,
                terminal=True,
                valid=False,
                correct=False,
                inference=dict(
                    unpadded_input_ids=[1, 2, 3, 4],
                    enable_thinking=True,
                    sampling=dict(seed=a["sampling_seed"]),
                    batch=dict(
                        cohort_ids=[a["trial_id"] for a in cohort],
                        row=row,
                        size=32,
                        backend="torch_batched_v1",
                        id=f"batch-{offset}",
                    ),
                ),
                activation_artifact=dict(positions=["record", "prefill"], token_positions=[1, 3]),
            )
            for row, a in enumerate(cohort)
        ]
        write_once(
            source / "cohorts/source-attempt" / f"{offset:07d}.json",
            dict(manifest_hash=digest(manifest), records=records, sha256=digest(records)),
        )
        outcomes.extend(dict(trial_id=row["trial_id"], record_sha256=digest(row)) for row in records)
    with tarfile.open(source / "source.tar.gz", "w:gz") as archive:
        for key, payload in payloads.items():
            info = tarfile.TarInfo(prepare.CANDIDATE_DOCUMENTS[key])
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    frozen = dict(
        manifest_sha256=digest(manifest),
        source_sha256="source",
        archive_sha256=prepare.sha(source / "source.tar.gz"),
        documents={
            prepare.CANDIDATE_DOCUMENTS[key]: hashlib.sha256(value).hexdigest()
            for key, value in payloads.items()
        },
    )
    write_once(source / "freeze.json", frozen)
    audit = dict(
        complete=True,
        integrity_passed=True,
        assigned=288,
        manifest_sha256=digest(manifest),
        source_sha256="source",
        source_freeze=dict(archive_sha256=frozen["archive_sha256"]),
        readiness=dict(passed=False),
        candidate_source=candidate,
        outcomes=outcomes,
    )
    audit_path, plan = tmp_path / "reports/source-audit.json", tmp_path / "reports/engineering-plan.md"
    write_once(audit_path, audit)
    plan.write_text("Prospective synthetic engineering plan; no execution.\n")
    return source, audit_path, plan


def test_first_actual_cohort_and_locked_axis_are_bound_despite_failed_readiness(tmp_path, monkeypatch):
    source, audit, plan = source_tree(tmp_path, monkeypatch)
    manifest = prepare.build_manifest(tmp_path, source, audit, plan, 128)
    assert len(manifest["diagnostics"]) == 128 and len(manifest["requests"]) == 32
    assert [row["source_trial_id"] for row in manifest["requests"]] == [f"source-{i:03d}" for i in range(32)]
    assert [row["seed"] for row in manifest["requests"]] == list(range(500, 532))
    assert manifest["source_binding"]["readiness_passed"] is False
    assert manifest["direction"]["dose"] == manifest["direction"]["norm"] == 0.25
    assert manifest["direction"]["unit"] == [0.25] * 16
    assert manifest["direction"]["corpus"] == "A"
    assert all(row["token_prefix"] == [1, 2, 3, 4] for row in manifest["requests"])
    assert len(manifest["document_bindings"]) == 6
    assert not (tmp_path / "executions").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "source_manifest",
        "missing_cohort",
        "source_seed",
        "source_prefix",
        "source_integrity",
        "candidate_archive",
    ],
)
def test_changed_or_incomplete_source_fails_without_replacement(tmp_path, monkeypatch, mutation):
    source, audit, plan = source_tree(tmp_path, monkeypatch)
    if mutation == "source_manifest":
        monkeypatch.setattr(prepare, "SOURCE_MANIFEST_SHA256", "other")
    elif mutation == "missing_cohort":
        (source / "cohorts/source-attempt/0000000.json").unlink()
    elif mutation in ("source_seed", "source_prefix"):
        path = source / "cohorts/source-attempt/0000000.json"
        value = read(path)
        if mutation == "source_seed":
            value["records"][0]["inference"]["sampling"]["seed"] += 1
        else:
            value["records"][0]["inference"]["unpadded_input_ids"][0] += 1
        value["sha256"] = digest(value["records"])
        path.write_text(canonical(value))
    elif mutation == "source_integrity":
        value = read(audit)
        value["integrity_passed"] = False
        audit.write_text(canonical(value))
    else:
        with (source / "source.tar.gz").open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError):
        prepare.build_manifest(tmp_path, source, audit, plan, 128)


def test_full_cap_reaudits_short_evidence_and_rejects_same_source_with_changed_seed(tmp_path, monkeypatch):
    source, source_audit, plan = source_tree(tmp_path, monkeypatch)
    short_manifest = prepare.build_manifest(tmp_path, source, source_audit, plan, 128)
    short = tmp_path / "runs/short"
    write_once(short / "manifest.json", short_manifest)
    receipt = dict(
        assay="sustained_engineering_audit",
        cap=128,
        complete=True,
        passed=True,
        scientific_admission=False,
        manifest_sha256=digest(short_manifest),
        compatibility_sha256=short_manifest["compatibility_sha256"],
        report_artifact=dict(path="engineering/report.json", sha256="report"),
        vector_artifact=dict(path="engineering/vectors.npz", sha256="vectors"),
    )
    audit_path = tmp_path / "reports/short-audit.json"
    write_once(audit_path, receipt)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "audit_sustained_engineering",
        SimpleNamespace(audit=lambda path: calls.append(path) or receipt),
    )
    full = prepare.build_manifest(
        tmp_path, source, source_audit, plan, 4096, short_run=short, short_audit=audit_path
    )
    assert calls == [short] and full["short_stage"]["audit"] == receipt
    assert full["compatibility_sha256"] == short_manifest["compatibility_sha256"]
    altered = deepcopy(short_manifest)
    altered["requests"][0]["seed"] += 1
    altered["compatibility_sha256"] = digest(engineering.compatibility_binding(altered))
    (short / "manifest.json").write_text(canonical(altered))
    with pytest.raises(ValueError, match="exact source"):
        prepare.build_manifest(
            tmp_path, source, source_audit, plan, 4096, short_run=short, short_audit=audit_path
        )


def test_full_cap_cannot_trust_unreproducible_short_pass_flag(tmp_path, monkeypatch):
    source, source_audit, plan = source_tree(tmp_path, monkeypatch)
    previous = prepare.build_manifest(tmp_path, source, source_audit, plan, 128)
    short = tmp_path / "runs/short"
    write_once(short / "manifest.json", previous)
    path = tmp_path / "reports/short-audit.json"
    write_once(path, dict(passed=True))
    monkeypatch.setitem(
        sys.modules, "audit_sustained_engineering", SimpleNamespace(audit=lambda _: dict(passed=False))
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        prepare.build_manifest(tmp_path, source, source_audit, plan, 4096, short_run=short, short_audit=path)
