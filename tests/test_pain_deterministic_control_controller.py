"""Finite two-phase admission, immutable evidence and pricing; no cloud calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
SPEC = importlib.util.spec_from_file_location(
    "pain_deterministic_control_controller_test", ROOT / "infra/launch_pain_deterministic_control.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)

STATE = dict(
    all_stopped_zero_tasks=True,
    available_compute_estimate_usd=100.0,
    timestamp_utc="synthetic",
    phase_compute_estimate_usd=20.0,
)


@pytest.fixture
def manifest():
    return dict(
        name=controller.NAME,
        resources=deepcopy(controller.POLICIES),
        campaign_ceiling_usd=5.0,
        original_manifest={"synthetic": "immutable original scientific inputs"},
        original_campaign_remains_failed=True,
    )


def ledger(project, phase):
    return project / "campaign_admissions" / controller.NAME / f"{phase}.json"


def run_dir(project):
    return project / "runs" / controller.NAME


def rewrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def setup_evidence(tmp_path, manifest):
    admission = controller.claim(tmp_path, manifest, STATE, "setup", now=1000.0)
    run = run_dir(tmp_path)
    evidence = {
        "original_execution/phases/setup/summary.json": dict(
            complete=True, passed=True, manifest_sha256=controller.ORIGINAL_MANIFEST_SHA
        ),
        "phases/setup/phase-execution.json": dict(
            complete=True, passed=True, phase="setup", manifest_sha256=controller.digest(manifest),
            admission_sha256=controller.digest(admission),
        ),
        "phases/setup/audit.json": dict(
            complete=True, passed=True, phase="setup", manifest_sha256=controller.digest(manifest),
            benchmark_forecast_seconds=1440.0,
        ),
        "launches/setup-outcome.json": dict(
            all_stopped_zero_tasks=True, export_error=None, failure=None,
        ),
    }
    for name, value in evidence.items():
        controller.write_once(run / name, value)
    return run, admission


def test_setup_reserves_both_phases_and_preserves_storage_allowance(tmp_path, manifest):
    total = 3.9287808
    assert sum(p["maximum_liability_usd"] for p in controller.POLICIES.values()) == pytest.approx(total)
    with pytest.raises(ValueError, match="Combined finite reservation"):
        controller.claim(tmp_path, manifest, dict(STATE, available_compute_estimate_usd=10 + total - 1e-6),
                         "setup", now=1000.0)
    assert not ledger(tmp_path, "setup").exists()
    receipt = controller.claim(tmp_path, manifest, dict(STATE, available_compute_estimate_usd=10 + total + 1e-6),
                               "setup", now=1000.0)
    assert receipt["deadline"] == 1900.0
    assert receipt["maximum_liability_usd"] == pytest.approx(1.3630464)
    assert receipt["prior_evidence"] == {}
    assert json.loads(ledger(tmp_path, "setup").read_text()) == receipt
    assert json.loads((run_dir(tmp_path) / "admissions/setup.json").read_text()) == receipt
    assert ledger(tmp_path, "setup").read_bytes() == (run_dir(tmp_path) / "admissions/setup.json").read_bytes()
    assert not (tmp_path / "runs" / controller.PARENT).exists()
    assert not (run_dir(tmp_path) / "original_execution/admissions").exists()


def test_benchmark_requires_prior_stage_and_exact_prior_bytes(tmp_path, manifest, setup_evidence):
    run, _ = setup_evidence
    # Noncanonical serialization proves these are actual byte bindings, not JSON digests.
    audit_path = run / "phases/setup/audit.json"
    rewrite(audit_path, json.loads(audit_path.read_text()))
    receipt = controller.claim(tmp_path, manifest, STATE, "benchmark", now=2000.0)
    assert receipt["deadline"] == 3800.0
    assert receipt["maximum_liability_usd"] == pytest.approx(2.5657344)
    assert receipt["prior_evidence"] == {
        name: controller.sha(run / name) for name in (
            "admissions/setup.json", "original_execution/phases/setup/summary.json",
            "phases/setup/phase-execution.json", "phases/setup/audit.json", "launches/setup-outcome.json",
        )
    }
    assert receipt["prior_evidence"]["phases/setup/audit.json"] != controller.digest(json.loads(audit_path.read_text()))
    assert json.loads(ledger(tmp_path, "benchmark").read_text()) == receipt
    assert ledger(tmp_path, "benchmark").read_bytes() == (run / "admissions/benchmark.json").read_bytes()
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, "benchmark", now=2001.0)


def test_benchmark_cannot_skip_setup(tmp_path, manifest):
    with pytest.raises((ValueError, FileNotFoundError)):
        controller.claim(tmp_path, manifest, STATE, "benchmark", now=1000.0)
    assert not ledger(tmp_path, "benchmark").exists()


@pytest.mark.parametrize("case", ["new", "existing_identical", "corrupt_payload", "conflicting_existing"])
def test_prior_transport_preserves_exact_noncanonical_bytes_before_worker(tmp_path, manifest, monkeypatch, case):
    # Exercise the transport function locally; neither Modal nor a child process can run.
    name = "phases/setup/audit.json"
    payload = b'{\n  "passed": true, "complete": true\n}\n'
    cloud_root = tmp_path / "synthetic-cloud"
    destination = cloud_root / controller.NAME / name
    if case in ("existing_identical", "conflicting_existing"):
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload if case == "existing_identical" else payload + b" ")
    manifest["original_execution_files"] = {}
    admission = dict(
        phase="benchmark", manifest_sha256=controller.digest(manifest),
        policy=controller.POLICIES["benchmark"], phase_consumed=True, retry_allowed=False,
        created=1000.0, deadline=2800.0,
        prior_evidence={name: controller.hashlib.sha256(payload).hexdigest()},
    )

    def local_path(*parts):
        value = Path(*parts)
        return cloud_root if value == Path("/artifacts") else value

    def before_worker(*args):
        raise RuntimeError("synthetic stop before worker execution")

    monkeypatch.setattr(controller, "Path", local_path)
    monkeypatch.setattr(controller, "modal", SimpleNamespace(Volume=SimpleNamespace(
        from_name=lambda *args, **kwargs: SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )))
    monkeypatch.setattr(controller.time, "time", lambda: 1001.0)
    monkeypatch.setattr(controller, "check_files", before_worker)
    supplied = payload + b" " if case == "corrupt_payload" else payload
    if case in ("corrupt_payload", "conflicting_existing"):
        with pytest.raises(ValueError, match="bytes changed"):
            controller.run_cloud(manifest, admission, {name: supplied}, "synthetic-image")
        if case == "corrupt_payload":
            assert not destination.exists()
        else:
            assert destination.read_bytes() == payload + b" "
    else:
        result = controller.run_cloud(manifest, admission, {name: supplied}, "synthetic-image")
        assert result["error"]["message"] == "synthetic stop before worker execution"
        assert destination.read_bytes() == payload


@pytest.mark.parametrize("name", [
    "admissions/setup.json", "original_execution/phases/setup/summary.json",
    "phases/setup/phase-execution.json", "phases/setup/audit.json", "launches/setup-outcome.json",
])
def test_missing_setup_evidence_blocks_before_consumption(tmp_path, manifest, setup_evidence, name):
    run, _ = setup_evidence
    (run / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        controller.claim(tmp_path, manifest, STATE, "benchmark", now=2000.0)
    assert not ledger(tmp_path, "benchmark").exists()


@pytest.mark.parametrize("name,key,value", [
    ("phases/setup/audit.json", "passed", False),
    ("phases/setup/audit.json", "complete", False),
    ("phases/setup/audit.json", "phase", "benchmark"),
    ("phases/setup/audit.json", "manifest_sha256", "0" * 64),
    ("phases/setup/audit.json", "benchmark_forecast_seconds", 1440.000001),
    ("phases/setup/audit.json", "benchmark_forecast_seconds", float("nan")),
    ("phases/setup/audit.json", "benchmark_forecast_seconds", True),
    ("original_execution/phases/setup/summary.json", "passed", False),
    ("original_execution/phases/setup/summary.json", "manifest_sha256", "0" * 64),
    ("launches/setup-outcome.json", "all_stopped_zero_tasks", False),
    ("launches/setup-outcome.json", "export_error", "partial export"),
    ("launches/setup-outcome.json", "failure", {"type": "TimeoutError"}),
])
def test_failed_audit_forecast_or_shutdown_blocks_admission(
    tmp_path, manifest, setup_evidence, name, key, value
):
    run, _ = setup_evidence
    record = json.loads((run / name).read_text())
    record[key] = value
    rewrite(run / name, record)
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, "benchmark", now=2000.0)
    assert not ledger(tmp_path, "benchmark").exists()


@pytest.mark.parametrize("change", ["missing_global", "different_global", "wrong_local_manifest", "unconsumed_local"])
def test_benchmark_requires_setup_global_claim_and_matching_mirror(tmp_path, manifest, setup_evidence, change):
    run, _ = setup_evidence
    path = ledger(tmp_path, "setup") if change.endswith("global") else run / "admissions/setup.json"
    if change == "missing_global":
        path.unlink()
    else:
        value = json.loads(path.read_text())
        if change == "different_global":
            value["created"] += 1
        elif change == "wrong_local_manifest":
            value["manifest_sha256"] = "0" * 64
        else:
            value["phase_consumed"] = False
        rewrite(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        controller.claim(tmp_path, manifest, STATE, "benchmark", now=2000.0)
    assert not ledger(tmp_path, "benchmark").exists()


@pytest.mark.parametrize("key,value", [
    ("complete", False), ("passed", False), ("phase", "benchmark"),
    ("manifest_sha256", "0" * 64), ("admission_sha256", "0" * 64),
])
def test_phase_receipt_must_match_successful_actual_setup_admission(
    tmp_path, manifest, setup_evidence, key, value
):
    run, _ = setup_evidence
    path = run / "phases/setup/phase-execution.json"
    record = json.loads(path.read_text())
    record[key] = value
    rewrite(path, record)
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, "benchmark", now=2000.0)
    assert not ledger(tmp_path, "benchmark").exists()


def test_simultaneous_setup_claims_have_one_winner(tmp_path, manifest):
    def attempt(_):
        try:
            return controller.claim(tmp_path, manifest, STATE, "setup", now=1000.0)
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(attempt, range(2)))
    assert sum(isinstance(row, dict) for row in result) == 1


def test_failed_local_mirror_keeps_global_phase_consumed(tmp_path, manifest, monkeypatch):
    def fail(*args):
        raise OSError("interrupted mirror")
    monkeypatch.setattr(controller, "write_once", fail)
    with pytest.raises(OSError, match="interrupted mirror"):
        controller.claim(tmp_path, manifest, STATE, "setup", now=1000.0)
    assert ledger(tmp_path, "setup").exists()
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, "setup", now=1001.0)


@pytest.mark.parametrize("change", ["wall", "rate", "second_container", "retry", "ceiling", "active", "nan_budget"])
def test_modified_policy_and_invalid_budget_cannot_consume(tmp_path, manifest, change):
    state = dict(STATE)
    if change == "ceiling":
        manifest["campaign_ceiling_usd"] = 6
    elif change == "active":
        state["all_stopped_zero_tasks"] = False
    elif change == "nan_budget":
        state["available_compute_estimate_usd"] = float("nan")
    else:
        key, value = {"wall": ("wall_seconds", 901), "rate": ("rate_usd_per_second", .001),
                      "second_container": ("max_containers", 2), "retry": ("retries", 1)}[change]
        manifest["resources"]["setup"][key] = value
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, state, "setup", now=1000.0)
    assert not ledger(tmp_path, "setup").exists()


def test_frozen_shared_code_inventory_rejects_changes_and_symlinks(tmp_path):
    root = tmp_path / "original"
    (root / "infra").mkdir(parents=True)
    code = root / "infra/budget.py"
    code.write_text("# immutable inherited source\n")
    inventory = {"infra/budget.py": controller.sha(code)}
    controller.check_files(root, inventory)
    old = code.read_bytes()
    code.write_bytes(old + b"# changed\n")
    with pytest.raises(ValueError, match="identity"):
        controller.check_files(root, inventory)
    code.write_bytes(old)
    target = tmp_path / "same.py"
    target.write_bytes(old)
    code.unlink()
    code.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        controller.check_files(root, inventory)


@pytest.fixture
def preparation_tree(tmp_path, monkeypatch):
    # This executable fixture is authored here; no downloaded source is executed.
    for source in controller.FILES.values():
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic bound source\n")
    preset = dict(schema_version=1, wall_seconds=900, execution_reserve_seconds=60,
                  deterministic_algorithms=True, deterministic_warn_only=False,
                  sdpa_backends=["math"], cublas_workspace_config=":4096:8")
    worker_path = tmp_path / controller.FILES["diagnostic_reference_worker.py"]
    worker_path.write_text(f"PRESET = {preset!r}\n")
    original = {"scientific_inputs": "unchanged synthetic plan"}
    parent = tmp_path / "runs" / controller.PARENT
    reference = tmp_path / "runs" / controller.DIAGNOSTIC
    controller.write_once(parent / "manifest.json", original)
    archive_path = parent / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"# original immutable scientific source\n"
        member = tarfile.TarInfo("src/curiosity_study/original.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    controller.write_once(parent / "freeze.json", dict(
        manifest_sha256=controller.digest(original), archive_sha256=controller.sha(archive_path)
    ))
    reference_manifest = dict(
        preset=preset,
        diagnostic_files={worker_path.name: controller.sha(worker_path)},
    )
    rewrite(reference / "manifest.json", reference_manifest)
    rewrite(reference / "summary.json", dict(passed=True, complete=True, completed_rows=148))
    rewrite(tmp_path / controller.FILES["reference-independent-check.json"], dict(
        passed=True, manifest_sha256=controller.digest(reference_manifest)
    ))
    for key, value in {
        "ORIGINAL_MANIFEST_SHA": controller.digest(original),
        "ARCHIVE_SHA": controller.sha(archive_path),
        "DIAGNOSTIC_MANIFEST_SHA": controller.digest(reference_manifest),
        "DIAGNOSTIC_SUMMARY_SHA": controller.sha(reference / "summary.json"),
        "DIAGNOSTIC_CHECK_SHA": controller.sha(tmp_path / controller.FILES["reference-independent-check.json"]),
    }.items():
        monkeypatch.setattr(controller, key, value)
    return worker_path, preset


def test_preparation_preserves_original_artifacts_and_reuses_bound_preset(tmp_path, preparation_tree):
    _, preset = preparation_tree
    originals = {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()}
    controller.prepare(tmp_path)
    manifest = json.loads((run_dir(tmp_path) / "manifest.json").read_text())
    assert manifest["preset"] == {k: v for k, v in preset.items() if k not in {"wall_seconds", "execution_reserve_seconds"}}
    assert manifest["assigned_calls"] == dict(setup=21, benchmark=65)
    assert manifest["assigned_rows"] == dict(setup=612, benchmark=2020)
    assert manifest["original_campaign_remains_failed"] is True
    assert manifest["scientific_rules_changed"] is False
    assert all(path.read_bytes() == payload for path, payload in originals.items())
    assert not (tmp_path / "campaign_admissions").exists()
    assert not (run_dir(tmp_path) / "original_execution/admissions").exists()
    with pytest.raises(ValueError, match="already prepared"):
        controller.prepare(tmp_path)


def test_tampered_successful_reference_worker_rejected_before_import(tmp_path, preparation_tree):
    worker_path, _ = preparation_tree
    worker_path.write_text("raise RuntimeError('unverified reference code executed')\n")
    with pytest.raises(ValueError):
        controller.prepare(tmp_path)
    assert not (run_dir(tmp_path) / "freeze.json").exists()
    assert not (tmp_path / "campaign_admissions").exists()


@pytest.mark.parametrize("audit_passes", [False, True])
def test_benchmark_cli_reaudits_frozen_setup_before_budget_or_claim(tmp_path, manifest, monkeypatch, audit_passes):
    execution = tmp_path / "executions" / controller.NAME
    for folder in ("original", "diagnostic"):
        (execution / folder).mkdir(parents=True)
    manifest.update(original_execution_files={}, diagnostic_files={})
    run = run_dir(tmp_path)
    controller.write_once(run / "manifest.json", manifest)
    controller.write_once(run / "freeze.json", dict(
        manifest_sha256=controller.digest(manifest), execution_root=str(execution)
    ))
    monkeypatch.setattr(controller, "__file__", str(execution / "diagnostic/launch_pain_deterministic_control.py"))
    observed = []

    def audit(command, **kwargs):
        observed.append("audit")
        assert command == [str(tmp_path / ".pain-runtime-venv/bin/python"),
                           str(execution / "diagnostic/audit_pain_deterministic_control.py"),
                           "--run", str(run), "--phase", "setup", "--verify-existing"]
        assert kwargs["check"] is True and kwargs["timeout"] == 120
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        if not audit_passes:
            raise subprocess.CalledProcessError(1, command)

    def snapshot(**kwargs):
        observed.append("budget")
        return STATE

    def claim(*args, **kwargs):
        observed.append("claim")
        raise RuntimeError("stop before image creation or cloud calls")

    monkeypatch.setattr(controller.subprocess, "run", audit)
    monkeypatch.setattr(controller.budget, "snapshot", snapshot)
    monkeypatch.setattr(controller.budget, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "claim", claim)
    with pytest.raises(RuntimeError if audit_passes else subprocess.CalledProcessError):
        controller.launch(tmp_path, "benchmark")
    assert observed == (["audit", "budget", "claim"] if audit_passes else ["audit"])
    assert not ledger(tmp_path, "benchmark").exists()


def test_both_new_apps_are_priced_at_h100_96gib_without_changing_legacy(tmp_path, monkeypatch):
    names = [row["app_name"] for row in controller.POLICIES.values()]
    names += ["curiosity-study-pilot-h100", "curiosity-study-pilot-h200", "unrelated-app"]
    apps = [dict(**{"App ID": str(i), "Description": name,
                   "Created at": "2026-10-15T00:00:00+00:00", "Stopped at": "2026-10-15T00:01:00+00:00"},
                 State="stopped", Tasks="0") for i, name in enumerate(names)]
    monkeypatch.setattr(controller.budget, "ROOT", tmp_path)
    monkeypatch.setattr(controller.budget.subprocess, "check_output", lambda *args, **kwargs: json.dumps(apps))
    state = controller.budget.snapshot(dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc))
    by_id = {row["app_id"]: row for row in state["app_charges"]}
    assert set(by_id) == {"0", "1", "2", "3"}
    for key in ("0", "1"):
        assert by_id[key]["gpu"] == "H100"
        assert by_id[key]["resource_profile"] == "pain-deterministic-control-2cpu-96gib"
        assert by_id[key]["rate_usd_per_second"] == pytest.approx(.00133632)
        assert by_id[key]["compute_estimate_usd"] == pytest.approx(60 * .00133632)
    assert by_id["2"]["rate_usd_per_second"] == controller.budget.RATES["H100"]
    assert by_id["3"]["rate_usd_per_second"] == controller.budget.RATES["H200"]
