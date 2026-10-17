"""Exclusive three-app resource admission and immutable transport; no cloud."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest

import launch_curiosity_coverage_bridge as controller
import curiosity_coverage_bridge_worker as worker

STATE = dict(
    all_stopped_zero_tasks=True,
    available_compute_estimate_usd=207.21713708,
    phase_compute_estimate_usd=92.78286292,
    timestamp_utc="synthetic",
)


@pytest.fixture
def manifest():
    return dict(
        name=controller.NAME,
        resources=controller.resources(),
        scientific_admission=False,
        causal_admission=False,
        resource_amendment=dict(amended_resource_check_passed=True, original_forecast_passed=False),
        workers=[
            dict(execution_group=g, app_name=controller.APPS[i]) for i, g in enumerate(controller.GROUPS)
        ],
    )


def test_exact_bound_and_one_parent_consuming_three_distinct_worker_claims(tmp_path, manifest):
    admission = controller.claim(tmp_path, manifest, STATE, now=1000.0)
    assert admission["deadline"] == 29000.0
    assert admission["maximum_liability_usd"] == 117.5775936
    assert len(set(controller.APPS)) == 3 and all(n.endswith("-h200") for n in controller.APPS)
    global_root = tmp_path / "campaign_admissions" / controller.NAME
    local_root = tmp_path / "runs" / controller.NAME
    assert (global_root / "admission.json").read_bytes() == (local_root / "admission.json").read_bytes()
    for i, group in enumerate(controller.GROUPS):
        claim = controller.claim_worker(tmp_path, manifest, admission, group, worker, now=1001.0 + i)
        assert claim["parent_created"] == 1000 and claim["deadline"] == 29000
        assert claim["app_name"] == controller.APPS[i]
        assert (global_root / "workers" / (group + ".json")).read_bytes() == (
            local_root / "workers" / group / "worker-claim.json"
        ).read_bytes()
        with pytest.raises(FileExistsError):
            controller.claim_worker(tmp_path, manifest, admission, group, worker, now=2000)
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=2000)


def test_concurrent_parent_claim_has_one_winner(tmp_path, manifest):
    def attempt(_):
        try:
            return controller.claim(tmp_path, manifest, STATE, now=1000)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(x is not None for x in pool.map(attempt, range(2))) == 1


def test_parent_local_mirror_failure_consumes_campaign(tmp_path, manifest, monkeypatch):
    original = controller.write_once

    def failed(path, value):
        if "runs" in path.parts:
            raise OSError("synthetic local mirror failure")
        original(path, value)

    monkeypatch.setattr(controller, "write_once", failed)
    with pytest.raises(OSError):
        controller.claim(tmp_path, manifest, STATE, now=1000)
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=1000)


@pytest.mark.parametrize("change", ["deadline", "manifest", "parent_time", "app", "consumed"])
def test_worker_claim_cannot_reset_or_swap_parent(tmp_path, manifest, change):
    admission = controller.claim(tmp_path, manifest, STATE, now=1000)
    claim = controller.claim_worker(tmp_path, manifest, admission, "worker_0", worker, now=1010)
    key, value = {
        "deadline": ("deadline", 29001),
        "manifest": ("manifest_sha256", "0" * 64),
        "parent_time": ("parent_created", 1001),
        "app": ("app_name", controller.APPS[1]),
        "consumed": ("consumed", 1),
    }[change]
    with pytest.raises(ValueError):
        worker.validate_worker_claim(manifest, admission, dict(claim, **{key: value}), "worker_0", 1020)


@pytest.mark.parametrize(
    "field,value",
    [
        ("all_stopped_zero_tasks", False),
        ("all_stopped_zero_tasks", 1),
        ("available_compute_estimate_usd", 127.57),
        ("available_compute_estimate_usd", float("nan")),
        ("phase_compute_estimate_usd", True),
    ],
)
def test_invalid_or_insufficient_budget_never_claims(tmp_path, manifest, field, value):
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, dict(STATE, **{field: value}), now=1000)
    assert not (tmp_path / "campaign_admissions").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_seconds", 28001),
        ("workers", 2),
        ("retries", 1),
        ("max_containers_per_app", 3),
        ("ceiling_usd", 121),
        ("memory_gib", 96),
    ],
)
def test_resource_changes_rejected_before_claim(tmp_path, manifest, field, value):
    manifest["resources"][field] = value
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, now=1000)
    assert not (tmp_path / "campaign_admissions").exists()


def test_failed_amendment_cannot_admit(tmp_path, manifest):
    manifest["resource_amendment"]["amended_resource_check_passed"] = False
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, now=1000)


def test_existing_freeze_blocks_prepare_before_source_work(tmp_path, monkeypatch):
    (tmp_path / "executions" / controller.NAME).mkdir(parents=True)
    monkeypatch.setattr(
        controller, "read", lambda *a: pytest.fail("No source work after consumed destination")
    )
    with pytest.raises(ValueError, match="already prepared"):
        controller.prepare(tmp_path)


def test_worker_import_checks_all_bytes_before_execution(tmp_path):
    path = tmp_path / controller.WORKER
    path.write_text("# original")
    bound = dict(wrapper_files={path.name: controller.sha(path)})
    sentinel = tmp_path / "executed"
    path.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()")
    with pytest.raises(ValueError):
        controller.load_worker(tmp_path, bound)
    assert not sentinel.exists()


def test_export_only_preserves_exact_bytes_and_never_admits(tmp_path, monkeypatch):
    remote = controller.NAME + "/workers/worker_0/raw.json"

    class Volume:
        def iterdir(self, prefix, recursive):
            assert prefix == controller.NAME and recursive
            return [SimpleNamespace(type=1, path=remote)]

        def read_file(self, path):
            assert path == remote
            return [b'{"unaltered": true}\n']

    monkeypatch.setattr(controller.modal.Volume, "from_name", lambda *a, **kw: Volume())
    monkeypatch.setattr(controller, "claim", lambda *a: pytest.fail("Export never admits"))
    assert controller.collect(tmp_path) == 1
    assert controller.collect(tmp_path) == 1
    path = tmp_path / "runs" / remote
    assert path.read_bytes() == b'{"unaltered": true}\n'
    path.write_text("different")
    with pytest.raises(ValueError, match="immutable"):
        controller.collect(tmp_path)


def test_distinct_names_and_creation_cutoff_exclude_old_or_unrelated_apps():
    rows = [
        dict(Description=name, **{"Created at": timestamp})
        for name, timestamp in [
            (controller.APPS[0], "1970-01-01T00:16:40+00:00"),
            (controller.APPS[1], "1970-01-01T00:00:00+00:00"),
            ("unrelated-h200", "1970-01-01T00:16:40+00:00"),
        ]
    ]
    assert controller.matching_apps(rows, dict(created=1000)) == rows[:1]


def test_changed_original_forecast_input_rejected_without_reinterpretation(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("unchanged")
    lifecycle = dict(raw_input_files={}, external_files={})
    root = tmp_path / "runs" / controller.ENGINEERING_NAME
    controller.write_once(root / "lifecycle-audit.json", lifecycle)
    bindings = [dict(path=str(source), byte_sha256=controller.sha(source), byte_count=source.stat().st_size)]
    controller.write_once(
        tmp_path / controller.FORECAST,
        dict(input_files=bindings, input_files_sha256=controller.digest(bindings)),
    )
    assert controller.verify_live_evidence(tmp_path, dict(documents={}))
    source.write_text("changed")
    with pytest.raises(ValueError, match="forecast input"):
        controller.verify_live_evidence(tmp_path, dict(documents={}))


def test_interrupted_export_publishes_no_partial_file_and_can_be_recovered(tmp_path, monkeypatch):
    destination = tmp_path / "runs/campaign/raw.json"
    original = controller.os.link
    monkeypatch.setattr(
        controller.os, "link", lambda *a: (_ for _ in ()).throw(OSError("synthetic interruption"))
    )
    with pytest.raises(OSError):
        controller.publish_download(tmp_path, destination, b"complete payload")
    assert not destination.exists()
    assert not list(tmp_path.glob(".coverage-download-*"))
    monkeypatch.setattr(controller.os, "link", original)
    controller.publish_download(tmp_path, destination, b"complete payload")
    assert destination.read_bytes() == b"complete payload"


@pytest.mark.parametrize("problem", [None, "active", "missing_marker", "wrong_app", "expired"])
def test_only_verified_stopped_exports_receive_separate_local_window(tmp_path, monkeypatch, problem):
    admission = dict(created=1000, deadline=29000)
    app = {
        "App ID": "ap-fixed",
        "Description": controller.APPS[0],
        "Created at": "1970-01-01T00:16:40+00:00",
        "State": "stopped",
        "Tasks": 0,
    }
    marker = dict(
        execution_group="worker_0",
        app_id="ap-fixed",
        app_stopped_zero_tasks=True,
        admission_sha256=controller.digest(admission),
    )
    if problem != "missing_marker":
        controller.write_once(tmp_path / "workers/worker_0/export-started.json", marker)
    if problem == "active":
        app["State"], app["Tasks"] = "ephemeral", 1
    if problem == "wrong_app":
        app["App ID"] = "ap-replacement"
    monkeypatch.setattr(controller, "app_rows", lambda *a: [app])
    now = 29121 if problem != "expired" else 32721
    if problem is None:
        controller.verify_local_export_window(tmp_path, tmp_path, admission, ["worker_0"], now)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            controller.verify_local_export_window(tmp_path, tmp_path, admission, ["worker_0"], now)
