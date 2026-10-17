"""Synthetic once-only local phase claims; never real budget or cloud admission."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tarfile

import pytest

from curiosity_study import pain_profile_admission as admission
from curiosity_study import pain_profile_design as design
from curiosity_study import pain_profile_resources as resources
from curiosity_study.artifacts import digest

STATE = dict(
    all_stopped_zero_tasks=True, available_compute_estimate_usd=100.0, timestamp_utc="synthetic test"
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def campaign(tmp_path):
    run = tmp_path / "runs" / admission.CAMPAIGN_NAME
    execution = tmp_path / "executions" / admission.CAMPAIGN_NAME
    contents = {
        "src/curiosity_study/example.py": "VALUE = 1\n",
        "infra/launch.py": "# synthetic launcher\n",
        "scripts/analyze.py": "# synthetic analyzer\n",
        "tests/test_example.py": "# synthetic test\n",
        "reports/plan.md": "Synthetic prospectively fixed plan.\n",
    }
    for name, content in contents.items():
        path = execution / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = digest({"example.py": contents["src/curiosity_study/example.py"]})
    documents = {"reports/plan.md": admission.sha(execution / "reports/plan.md")}
    manifest = dict(
        assay=resources.ASSAY,
        revision=design.REVISION,
        source_sha256=source,
        resources={p: resources.policy(p) for p in resources.POLICIES},
        documents=documents,
    )
    put(run / "manifest.json", manifest)
    with tarfile.open(run / "source.tar.gz", "w:gz") as archive:
        for name in contents:
            archive.add(execution / name, arcname=name)
    frozen = dict(
        budget_root=str(tmp_path),
        execution_root=str(execution),
        manifest_sha256=digest(manifest),
        source_sha256=source,
        archive_sha256=admission.sha(run / "source.tar.gz"),
        documents=documents,
    )
    put(run / "freeze.json", frozen)
    return run, manifest, frozen


def ledger(run):
    return run.parents[1] / "campaign_admissions" / resources.REVISION


def evidence(run, *, setup=False, seconds=100):
    manifest = json.loads((run / "manifest.json").read_text())
    put(
        run / "phases/prefetch/summary.json",
        dict(phase="prefetch", manifest_sha256=digest(manifest), complete=True, passed=True),
    )
    if setup:
        summary = dict(
            phase="setup",
            manifest_sha256=digest(manifest),
            complete=True,
            passed=True,
            forecast=dict(benchmark_seconds=seconds, limit_seconds=2880, passed=True),
        )
        put(run / "phases/setup/summary.json", summary)
        put(
            run / "phases/setup/audit.json",
            dict(
                phase="setup",
                manifest_sha256=digest(manifest),
                complete=True,
                passed=True,
                summary_byte_sha256=admission.sha(run / "phases/setup/summary.json"),
            ),
        )


def prefetched(run):
    claim = admission.claim(run, "prefetch", STATE, now=1000)
    evidence(run)
    return claim


def setup_done(run, *, seconds=100):
    prefetch = prefetched(run)
    setup = admission.claim(run, "setup", STATE, now=3000)
    evidence(run, setup=True, seconds=seconds)
    return prefetch, setup


def test_real_design_revision_and_exact_frozen_archive_are_accepted(campaign):
    run, manifest, frozen = campaign
    observed, observed_freeze, root = admission.verify_frozen_campaign(run)
    assert observed == manifest and observed_freeze == frozen and root == run.parents[1]
    assert observed["revision"] == design.REVISION


@pytest.mark.parametrize(
    "name",
    [
        "src/curiosity_study/example.py",
        "infra/launch.py",
        "scripts/analyze.py",
        "tests/test_example.py",
        "reports/plan.md",
    ],
)
def test_changed_frozen_source_launcher_test_or_document_cannot_start(campaign, name):
    run, _, frozen = campaign
    path = Path(frozen["execution_root"]) / name
    path.write_text(path.read_text() + "# edited\n")
    with pytest.raises(ValueError):
        admission.claim(run, "prefetch", STATE, now=1000)
    assert not (ledger(run) / "prefetch.json").exists()


def test_added_execution_file_is_rejected(campaign):
    run, _, frozen = campaign
    (Path(frozen["execution_root"]) / "infra/extra.py").write_text("# extra")
    with pytest.raises(ValueError, match="Unexpected frozen execution"):
        admission.verify_frozen_campaign(run)


def test_added_execution_symlink_is_rejected(campaign, tmp_path):
    run, _, frozen = campaign
    external = tmp_path / "outside.py"
    external.write_text("# not archived")
    (Path(frozen["execution_root"]) / "infra/extra.py").symlink_to(external)
    with pytest.raises(ValueError):
        admission.verify_frozen_campaign(run)


def test_intentional_top_level_venv_and_python_cache_do_not_invalidate_source(campaign, tmp_path):
    run, _, frozen = campaign
    execution = Path(frozen["execution_root"])
    environment = tmp_path / ".venv"
    environment.mkdir()
    (execution / ".venv").symlink_to(environment, target_is_directory=True)
    cache = execution / "src/curiosity_study/__pycache__/example.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"synthetic cache")
    admission.verify_frozen_campaign(run)


def test_aliased_campaign_path_cannot_consume_phase(campaign, tmp_path):
    run, _, _ = campaign
    renamed = tmp_path / "runs/alias"
    run.rename(renamed)
    with pytest.raises(ValueError, match="aliased"):
        admission.claim(renamed, "prefetch", STATE, now=1000)


def test_exact_once_only_order_and_deadlines_under12(campaign):
    run, manifest, _ = campaign
    prefetch, setup = setup_done(run)
    benchmark = admission.claim(run, "benchmark", STATE, now=5000)
    claims = dict(prefetch=prefetch, setup=setup, benchmark=benchmark)
    for phase, receipt in claims.items():
        assert receipt["deadline"] - receipt["created"] == resources.policy(phase)["wall_seconds"]
        assert receipt["manifest_sha256"] == digest(manifest)
        assert receipt["phase_consumed"] is True and receipt["retry_allowed"] is False
        assert json.loads((run / "admissions" / f"{phase}.json").read_text()) == receipt
        with pytest.raises(ValueError, match="consumed"):
            admission.claim(run, phase, STATE, now=10000)
    assert sum(row["policy"]["maximum_compute_usd"] for row in claims.values()) < 12
    assert setup["prior_admissions"] == {"prefetch": digest(prefetch)}
    assert benchmark["prior_admissions"] == {"prefetch": digest(prefetch), "setup": digest(setup)}
    assert benchmark["prior_evidence"]["setup_audit_sha256"] == admission.sha(run / "phases/setup/audit.json")


@pytest.mark.parametrize("phase", ["setup", "benchmark"])
def test_skipped_phase_cannot_consume_claim(campaign, phase):
    run, _, _ = campaign
    with pytest.raises(ValueError, match="requires"):
        admission.claim(run, phase, STATE, now=1000)
    assert not (ledger(run) / f"{phase}.json").exists()


@pytest.mark.parametrize("available", [11.99, float("nan"), float("inf"), True, None])
def test_invalid_or_insufficient_budget_fails_before_consumption(campaign, available):
    run, _, _ = campaign
    state = dict(STATE, available_compute_estimate_usd=available)
    with pytest.raises(ValueError, match="allowance"):
        admission.claim(run, "prefetch", state, now=1000)
    assert not (ledger(run) / "prefetch.json").exists()


@pytest.mark.parametrize("now", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_time_is_rejected(campaign, now):
    run, _, _ = campaign
    with pytest.raises(ValueError, match="admission time"):
        admission.claim(run, "prefetch", STATE, now=now)


def test_active_app_vetoes_before_phase_consumption(campaign):
    run, _, _ = campaign
    with pytest.raises(ValueError, match="still active"):
        admission.claim(run, "prefetch", dict(STATE, all_stopped_zero_tasks=False), now=1000)
    assert not (ledger(run) / "prefetch.json").exists()


@pytest.mark.parametrize("change", ["incomplete", "false_pass", "wrong_manifest"])
def test_cpu_completion_required_before_setup(campaign, change):
    run, _, _ = campaign
    prefetched(run)
    path = run / "phases/prefetch/summary.json"
    value = json.loads(path.read_text())
    value[{"incomplete": "complete", "false_pass": "passed", "wrong_manifest": "manifest_sha256"}[change]] = (
        "0" * 64 if change == "wrong_manifest" else False
    )
    put(path, value)
    with pytest.raises(ValueError):
        admission.claim(run, "setup", STATE, now=3000)
    assert not (ledger(run) / "setup.json").exists()


@pytest.mark.parametrize("change", ["missing_mirror", "mirror_edited", "global_edited", "prefetch_rebound"])
def test_prior_receipts_and_evidence_cannot_be_changed_before_benchmark(campaign, change):
    run, _, _ = campaign
    setup_done(run)
    path = run / "admissions/prefetch.json" if change.startswith("mi") else ledger(run) / "prefetch.json"
    if change == "missing_mirror":
        path.unlink()
    elif change == "prefetch_rebound":
        path = run / "phases/prefetch/summary.json"
        value = json.loads(path.read_text())
        value["extra"] = "changed after setup bound it"
        put(path, value)
    else:
        value = json.loads(path.read_text())
        value["deadline"] += 1
        put(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        admission.claim(run, "benchmark", STATE, now=5000)
    assert not (ledger(run) / "benchmark.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "audit_failed",
        "audit_incomplete",
        "summary_failed",
        "summary_rehashed",
        "wrong_campaign",
        "over_limit",
        "nan",
        "zero",
        "boolean",
    ],
)
def test_benchmark_requires_bound_complete_setup_and_conservative_finite_forecast(campaign, change):
    run, _, _ = campaign
    setup_done(run)
    summary_path, audit_path = run / "phases/setup/summary.json", run / "phases/setup/audit.json"
    summary, audit = json.loads(summary_path.read_text()), json.loads(audit_path.read_text())
    if change == "audit_failed":
        audit["passed"] = False
    elif change == "audit_incomplete":
        audit["complete"] = False
    elif change == "summary_failed":
        summary["passed"] = False
    elif change == "summary_rehashed":
        summary["extra"] = "not bound by audit"
    elif change == "wrong_campaign":
        audit["manifest_sha256"] = "0" * 64
    else:
        summary["forecast"]["benchmark_seconds"] = {
            "over_limit": 2880.01,
            "nan": float("nan"),
            "zero": 0,
            "boolean": True,
        }[change]
    put(summary_path, summary)
    if change != "summary_rehashed":
        audit["summary_byte_sha256"] = admission.sha(summary_path)
    put(audit_path, audit)
    with pytest.raises(ValueError):
        admission.claim(run, "benchmark", STATE, now=5000)
    assert not (ledger(run) / "benchmark.json").exists()


def test_exact2880_forecast_boundary_is_allowed(campaign):
    run, _, _ = campaign
    setup_done(run, seconds=2880)
    assert (
        admission.claim(run, "benchmark", STATE, now=5000)["prior_evidence"]["benchmark_forecast_seconds"]
        == 2880
    )


def test_combined_liability_guard_does_not_consume_failed_phase(campaign, monkeypatch):
    run, _, _ = campaign
    prior = prefetched(run)
    total = prior["policy"]["maximum_compute_usd"] + resources.policy("setup")["maximum_compute_usd"]
    monkeypatch.setattr(resources, "CAMPAIGN_CEILING_USD", total - 0.01)
    with pytest.raises(ValueError, match="Cumulative"):
        admission.claim(run, "setup", STATE, now=3000)
    assert not (ledger(run) / "setup.json").exists()


def test_interrupted_mirror_publication_cannot_retry(campaign, monkeypatch):
    run, _, _ = campaign
    original = admission.write_once

    def interrupted(path, value):
        if Path(path) == run / "admissions/prefetch.json":
            raise OSError("synthetic mirror write failure")
        return original(path, value)

    monkeypatch.setattr(admission, "write_once", interrupted)
    with pytest.raises(OSError, match="mirror write"):
        admission.claim(run, "prefetch", STATE, now=1000)
    assert (ledger(run) / "prefetch.json").exists()
    assert not (run / "admissions/prefetch.json").exists()
    monkeypatch.setattr(admission, "write_once", original)
    with pytest.raises(ValueError, match="consumed"):
        admission.claim(run, "prefetch", STATE, now=1100)


def test_two_local_contenders_can_consume_only_one_synthetic_phase(campaign):
    run, _, _ = campaign

    def attempt(_):
        try:
            return admission.claim(run, "prefetch", STATE, now=1000)
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(isinstance(value, dict) for value in results) == 1
    assert len(list(ledger(run).glob("*.json"))) == 1
