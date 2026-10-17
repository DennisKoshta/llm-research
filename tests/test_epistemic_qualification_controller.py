"""One-time admission, source immutability and terminal coverage; no cloud calls."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
import launch_epistemic_readout_qualification as controller  # noqa: E402

STATE = dict(all_stopped_zero_tasks=True, available_compute_estimate_usd=100.,
             phase_compute_estimate_usd=90., timestamp_utc="synthetic")


@pytest.fixture
def manifest():
    assignments = [dict(trial_id=f"synthetic-{i:02d}") for i in range(48)]
    return dict(name=controller.NAME, resources=controller.resources(), qualification_only=True, causal_admission=False,
        plan=dict(assignments=assignments, execution_cohorts=[dict(offset=i,
            trial_ids=[a["trial_id"] for a in assignments[i:i + 32]]) for i in (0, 32)]))


def test_bound_cost_and_one_global_admission_exact_local_bytes(tmp_path, manifest):
    result = controller.claim(tmp_path, manifest, STATE, now=1000.)
    assert result["deadline"] == 2800 and result["maximum_liability_usd"] == pytest.approx(2.6760192)
    global_path = tmp_path / "campaign_admissions" / controller.NAME / "admission.json"
    local_path = tmp_path / "runs" / controller.NAME / "admission.json"
    assert global_path.read_bytes() == local_path.read_bytes()
    assert json.loads(global_path.read_text()) == result
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=2000.)


def test_concurrent_admission_has_one_winner(tmp_path, manifest):
    def attempt(_):
        try:
            return controller.claim(tmp_path, manifest, STATE, now=1000.)
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(r is not None for r in results) == 1


def test_local_mirror_failure_still_consumes_campaign(tmp_path, manifest, monkeypatch):
    original = controller.write_once
    def failing(path, value):
        if "runs" in path.parts:
            raise OSError("synthetic mirror failure")
        original(path, value)
    monkeypatch.setattr(controller, "write_once", failing)
    with pytest.raises(OSError):
        controller.claim(tmp_path, manifest, STATE, now=1000.)
    assert (tmp_path / "campaign_admissions" / controller.NAME / "admission.json").exists()
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=1000.)


@pytest.mark.parametrize("field,value", [("wall_seconds", 1801), ("gpu", "H100"), ("cpu", 4),
    ("memory_gib", 96), ("ceiling_usd", 4), ("retries", 1), ("rate_usd_per_second", .001)])
def test_policy_changes_rejected_before_admission(tmp_path, manifest, field, value):
    manifest["resources"][field] = value
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, now=1000.)
    assert not (tmp_path / "campaign_admissions").exists()


@pytest.mark.parametrize("field,value", [("all_stopped_zero_tasks", False), ("all_stopped_zero_tasks", 1),
    ("available_compute_estimate_usd", 12.), ("available_compute_estimate_usd", float("nan")),
    ("phase_compute_estimate_usd", True)])
def test_active_or_invalid_or_insufficient_budget_rejected(tmp_path, manifest, field, value):
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, dict(STATE, **{field: value}), now=1000.)
    assert not (tmp_path / "campaign_admissions").exists()


@pytest.mark.parametrize("change", ["bytes", "extra", "missing", "symlink"])
def test_frozen_inventory_rejects_changes(tmp_path, change):
    tree = tmp_path / "frozen"
    tree.mkdir()
    path = tree / "worker.py"
    path.write_text("original")
    inventory = {"worker.py": controller.sha(path)}
    controller.check_files(tree, inventory)
    if change == "bytes":
        path.write_text("changed")
    elif change == "extra":
        (tree / "extra.py").write_text("extra")
    elif change == "missing":
        path.unlink()
    else:
        other = tmp_path / "same"
        other.write_text("original")
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(ValueError):
        controller.check_files(tree, inventory)


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/absolute", "file"),
                                      ("safe", "symlink"), ("duplicate", "duplicate")])
def test_archive_rejects_unsafe_inventory_before_any_extraction(tmp_path, monkeypatch, name, kind):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for _ in range(2 if kind == "duplicate" else 1):
            entry = tarfile.TarInfo(name)
            if kind == "symlink":
                entry.type, entry.linkname = tarfile.SYMTYPE, "/outside"
                stream.addfile(entry)
            else:
                entry.size = 1
                stream.addfile(entry, io.BytesIO(b"x"))
    monkeypatch.setattr(controller, "ARCHIVE_SHA", controller.sha(archive))
    with pytest.raises(ValueError, match="Unsafe"):
        controller.extract_original(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_hardkill_without_summary_preserves_all48_missing(tmp_path, manifest):
    result = controller.coverage(tmp_path, manifest)
    assert len(result["rows"]) == 48 and all(r["status"] == "missing" for r in result["rows"])
    assert result["complete"] is False and result["causal_admission"] is False


def test_hardkill_in_second_batch_retains32_and16_failed(tmp_path, manifest):
    run = tmp_path / "runs" / controller.NAME
    first, second = manifest["plan"]["execution_cohorts"]
    rows = [dict(trial_id=a["trial_id"], assignment_sha256=controller.digest(a), inference={})
            for a in manifest["plan"]["assignments"][:32]]
    controller.write_once(run / "cohorts/0000.json", dict(manifest_sha256=controller.digest(manifest),
        cohort=first, status="completed", records=rows))
    controller.write_once(run / "attempts/0032.json", dict(cohort=second))
    result = controller.coverage(tmp_path, manifest)
    assert [sum(r["status"] == status for r in result["rows"]) for status in
            ("completed", "execution_error", "missing")] == [32, 16, 0]
    assert not result["complete"]


def test_rehashed_cohort_cannot_swap_assignment_binding(tmp_path, manifest):
    run = tmp_path / "runs" / controller.NAME
    first = manifest["plan"]["execution_cohorts"][0]
    rows = [dict(trial_id=a["trial_id"], assignment_sha256=controller.digest(a))
            for a in manifest["plan"]["assignments"][:32]]
    rows[0]["assignment_sha256"] = rows[1]["assignment_sha256"]
    controller.write_once(run / "cohorts/0000.json", dict(manifest_sha256=controller.digest(manifest),
        cohort=first, status="completed", records=rows))
    result = controller.coverage(tmp_path, manifest)
    assert len(result["rows"]) == 48 and len(result["issues"]) == 1
    assert not result["complete"]


def test_import_guard_rejects_tampered_worker_before_executing(tmp_path):
    path = tmp_path / controller.WORKER
    sentinel = tmp_path / "executed"
    path.write_text("# original")
    bound = dict(wrapper_files={controller.WORKER: controller.sha(path)})
    path.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()")
    with pytest.raises(ValueError):
        controller.load_worker(tmp_path, bound)
    assert not sentinel.exists()


def test_snapshot_runs_frozen_accounting_separately_without_model_namespace(tmp_path, monkeypatch):
    seen = {}
    def run(args, **kwargs):
        seen.update(args=args, **kwargs)
        return json.dumps(STATE)
    monkeypatch.setattr(controller.subprocess, "check_output", run)
    wrapper = tmp_path / "qualification"
    assert controller.snapshot(tmp_path, wrapper) == STATE
    assert seen["env"]["PYTHONPATH"] == str(wrapper / "accounting")
    assert seen["env"]["CURIOSITY_BUDGET_ROOT"] == str(tmp_path)
    assert seen["timeout"] == 25
