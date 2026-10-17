"""Synthetic artifact-chain tests; no model outcomes or compute are used."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

from curiosity_study import thinking_information as thinking
from curiosity_study import voluntary_capture as capture
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import RunStore, digest, read, write_once

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "voluntary_analyzer_integration", SCRIPTS / "analyze_voluntary_resolution.py"
)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
from audit_voluntary_resolution import compact_record  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays():
    shape = (8, 4, 2, 8)
    units = np.broadcast_to(np.eye(8)[:, None, None, :], shape).copy()
    centers = np.full(shape, 2.0)
    norms = np.full((8, 4, 2), 4.0)
    return dict(
        sources=analyzer.CONTRASTS,
        layers=[15, 31, 47, 57],
        positions=["record", "final"],
        unit=units,
        center=centers,
        contrast_norm=norms,
        threshold=np.full((8, 4, 2), 2.0),
        contrast=units * norms[..., None],
    )


@pytest.fixture
def bound_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    directions = tmp_path / "directions.npz"
    np.savez_compressed(directions, **arrays())
    plan = voluntary.assignment_plan()
    candidate = dict(
        directions_byte_sha256=sha(directions),
        selected_sites={name: dict(layer=15, position="record") for name in analyzer.CONTRASTS[:2]},
    )
    document = "reports/voluntary-resolution-v1-plan.md"
    document_bytes = b"Synthetic prospective plan fixture, no model data."
    manifest = dict(
        assay="voluntary_resolution",
        task_revision=voluntary.REVISION,
        source_sha256="synthetic-source",
        model=capture.MODEL,
        assignments=[],
        diagnostics=plan["assignments"],
        assignment_sha256=plan["assignment_sha256"],
        analysis_metadata=analyzer.REQUIRED_ANALYSIS_METADATA,
        candidate_source=candidate,
        document_bindings=[dict(path=document, byte_sha256=hashlib.sha256(document_bytes).hexdigest())],
        prior_exposure=dict(independent_holdout_claim_supported=False),
    )
    store = RunStore(run, manifest)
    source, bindings = {}, []
    for offset in range(0, 288, 32):
        cohort = manifest["diagnostics"][offset : offset + 32]
        hidden = np.full((32, 4, 7, 8), 2.0)
        for row, a in enumerate(cohort):
            hidden[row] += (a["world_slot"] + (a["delivery"] == "decisive_evidence")) / 10
        available = np.ones((32, 4, 7), dtype=bool)
        positions = ["record", "prefill", "token_32", "token_128", "token_512", "think_end", "decision"]
        token_positions = np.tile(np.arange(7), (32, 1))
        path = run / "activations" / "synthetic" / f"{offset:07d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            hidden=hidden,
            available=available,
            layers=[15, 31, 47, 57],
            positions=positions,
            token_positions=token_positions,
            trial_ids=[a["trial_id"] for a in cohort],
        )
        rows = []
        for index, a in enumerate(cohort):
            if a["kind"] == "choice":
                service = "report" if a["world_slot"] % 3 == 0 else "stop"
                label = next(
                    k for k, value in voluntary.task(a)["action_mapping"].items() if value == service
                )
            else:
                label = voluntary.prompt(a)[2]
            inference = dict(
                raw=f"Synthetic reasoning.</think> {label}",
                ended_with_eos=True,
                generated_token_ids=[9, 10],
                unpadded_input_ids=[1, 2, 3],
            )
            record = thinking.record_from_inference(a, inference)
            record["activation_artifact"] = dict(
                path=str(path.relative_to(run)),
                sha256=sha(path),
                row=index,
                layers=[15, 31, 47, 57],
                positions=positions,
                token_positions=token_positions[index].tolist(),
                available=available[index].tolist(),
                max_new_tokens=4096,
            )
            source[a["trial_id"]] = record
            rows.append(record)
            bindings.append(
                dict(
                    trial_id=a["trial_id"],
                    record_sha256=digest(record),
                    artifact_sha256=sha(path),
                    row=index,
                    token_prefix_sha256=digest(inference["unpadded_input_ids"]),
                )
            )
        store.save_many(rows, "synthetic", f"{offset:07d}")
    choices = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
    receipt_bindings, receipt_cohorts = [], []
    for offset in range(0, 144, 32):
        cohort = choices[offset : offset + 32]
        sidecars = [capture.choice_sidecar(source[a["trial_id"]]) for a in cohort]
        active = [r for r in sidecars if r["receipt_required"]]
        if active:
            hidden = np.full((len(active), 4, 2, 8), 3.0)
            available = np.ones((len(active), 4, 2), dtype=bool)
            tokens = np.tile([0, 1], (len(active), 1))
            path = run / "receipt_activations" / "synthetic" / f"{offset:07d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                hidden=hidden,
                available=available,
                token_positions=tokens,
                layers=[15, 31, 47, 57],
                positions=["record", "final"],
                trial_ids=[r["source_trial_id"] for r in active],
            )
            for index, row in enumerate(active):
                row.update(
                    status="completed",
                    receipt_model_pass_completed=True,
                    token_binding=dict(token_prefix_sha256=digest([5, 6])),
                    activation_artifact=dict(
                        path=str(path.relative_to(run)),
                        sha256=sha(path),
                        row=index,
                        layers=[15, 31, 47, 57],
                        positions=["record", "final"],
                        token_positions=[0, 1],
                        available=available[index].tolist(),
                    ),
                )
                receipt_bindings.append(
                    dict(
                        source_trial_id=row["source_trial_id"],
                        sidecar_sha256=digest(row),
                        source_record_sha256=row["source_record_sha256"],
                        artifact_sha256=sha(path),
                        row=index,
                        token_prefix_sha256=digest([5, 6]),
                    )
                )
        envelope = dict(manifest_sha256=digest(manifest), records=sidecars, records_sha256=digest(sidecars))
        path = run / "receipt_cohorts" / "synthetic" / f"{offset:07d}.json"
        write_once(path, envelope)
        receipt_cohorts.append(dict(path=str(path.relative_to(run)), envelope_sha256=digest(envelope)))
    auditor_bytes = (SCRIPTS / "audit_voluntary_resolution.py").read_bytes()
    with tarfile.open(run / "source.tar.gz", "w:gz") as archive:
        for name, payload in (
            (analyzer.DIRECTION_DOCUMENT, directions.read_bytes()),
            ("scripts/audit_voluntary_resolution.py", auditor_bytes),
            (document, document_bytes),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    freeze = dict(
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        archive_sha256=sha(run / "source.tar.gz"),
    )
    write_once(run / "freeze.json", freeze)
    audit = dict(
        assay="voluntary_resolution",
        complete=True,
        integrity_passed=True,
        passed=False,
        readiness=dict(
            passed=False, diagnostic_cells=[dict(passed=False, family="objects", kind="new_fact")]
        ),
        causal_admission=False,
        choice_correctness=None,
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        model=manifest["model"],
        analysis_metadata=manifest["analysis_metadata"],
        candidate_source=candidate,
        source_freeze=dict(
            freeze_sha256=digest(freeze), documents={document: hashlib.sha256(document_bytes).hexdigest()}
        ),
        auditor_byte_sha256=hashlib.sha256(auditor_bytes).hexdigest(),
        outcomes=[compact_record(a, source[a["trial_id"]]) for a in manifest["diagnostics"]],
        generation_record_bindings_sha256=digest(bindings),
        receipt_bindings=dict(
            cohorts=receipt_cohorts,
            record_bindings=receipt_bindings,
            record_bindings_sha256=digest(receipt_bindings),
        ),
    )
    audit_path = tmp_path / "audit.json"
    write_once(audit_path, audit)
    return run, audit_path, directions


def test_bound_end_to_end_analysis_preserves_failed_readiness_and_exports_every_choice(bound_run, tmp_path):
    run, audit_path, directions = bound_run
    out = tmp_path / "analysis"
    result = analyzer.analyze(run, audit_path, directions, out)
    assert not result["readiness_passed"] and not result["causal_admission"]
    assert result["assigned_choices"] == 144 and result["assigned_worlds"] == 24
    assert len(result["generation_record_bindings"]) == 288
    assert result["receipt_status_counts"] == {"completed": 48, "stop": 96}
    counts = {r["path"]: r["rows"] for r in result["exports"]}
    assert counts["choices.csv"] == 144
    assert counts["choice-projections.csv"] == counts["receipt-projections.csv"] == 2304
    assert all(sha(out / item["path"]) == item["byte_sha256"] for item in result["exports"])
    assert read(out / "analysis.json")["readiness"] == read(audit_path)["readiness"]
    row = next(
        r
        for r in result["projection_contrasts"]
        if r["family"] == "pooled" and r["contrast"] == "decisive_minus_redundant" and r["scale"] == "scaled"
    )
    assert row["estimate"] == pytest.approx(0.025)
    assert row["ci95"] == pytest.approx([0.025, 0.025])
    assert all(
        r["receipt_centered_raw"]["n"] == 0
        for r in result["receipt_summaries"]
        if r["selected_service"] == "stop"
    )
    with pytest.raises(ValueError, match="immutable"):
        analyzer.analyze(run, audit_path, directions, out)


def test_direction_file_cannot_be_replaced_even_if_numerically_plausible(bound_run):
    run, audit_path, directions = bound_run
    data = arrays()
    data["center"] += 1
    np.savez_compressed(directions, **data)
    with pytest.raises(ValueError, match="direction bytes"):
        analyzer.bind_inputs(run, audit_path, directions)


@pytest.mark.parametrize("field", ["complete", "integrity_passed"])
def test_incomplete_or_failed_integrity_audit_is_not_analyzed(bound_run, field):
    run, audit_path, directions = bound_run
    a = read(audit_path)
    a[field] = False
    audit_path.write_text(__import__("json").dumps(a))
    with pytest.raises(ValueError, match="integrity-passed"):
        analyzer.bind_inputs(run, audit_path, directions)


def test_changed_capture_bytes_rejected_before_projection(bound_run):
    run, audit_path, directions = bound_run
    path = next((run / "activations").glob("*/*.npz"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte hash"):
        analyzer.bind_inputs(run, audit_path, directions)


def test_rehashed_receipt_cannot_change_source_or_selected_capture(bound_run):
    run, audit_path, directions = bound_run
    path = next((run / "receipt_cohorts").glob("*/*.json"))
    envelope = read(path)
    row = next(r for r in envelope["records"] if r["status"] == "completed")
    row["activation_artifact"]["row"] = (row["activation_artifact"]["row"] + 1) % 8
    envelope["records_sha256"] = digest(envelope["records"])
    path.write_text(__import__("json").dumps(envelope))
    with pytest.raises(ValueError, match="row/source binding|bindings differ"):
        analyzer.bind_inputs(run, audit_path, directions)


def test_invalid_choice_cannot_leak_its_parsed_action():
    assert analyzer.category(dict(valid=False, thinking_closed=False, selected_service=None)) == "invalid"
    assert analyzer.category(dict(valid=True, thinking_closed=True, selected_service="stop")) == "stop"
    with pytest.raises(ValueError, match="Invalid choices"):
        analyzer.category(dict(valid=False, thinking_closed=False, selected_service="report"))
    assert analyzer.category(None) == "missing"
    assert analyzer.category(dict(failure="execution_error")) == "execution_failure"
