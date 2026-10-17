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
import launch_curiosity_coverage_engineering as controller  # noqa: E402

STATE = dict(
    all_stopped_zero_tasks=True,
    available_compute_estimate_usd=100.0,
    phase_compute_estimate_usd=90.0,
    timestamp_utc="synthetic",
)


@pytest.fixture
def manifest():
    assignments = [dict(trial_id=f"synthetic-{i:02d}") for i in range(48)]
    return dict(
        name=controller.NAME,
        resources=controller.resources(),
        engineering_only=True,
        scientific_admission=False,
        causal_admission=False,
        plan=dict(
            assignments=assignments,
            execution_cohorts=[
                dict(offset=i, trial_ids=[a["trial_id"] for a in assignments[i : i + 32]]) for i in (0, 32)
            ],
        ),
    )


def test_bound_cost_and_one_global_admission_exact_local_bytes(tmp_path, manifest):
    result = controller.claim(tmp_path, manifest, STATE, now=1000.0)
    assert result["deadline"] == 4400 and result["maximum_liability_usd"] == pytest.approx(4.9060352)
    global_path = tmp_path / "campaign_admissions" / controller.NAME / "admission.json"
    local_path = tmp_path / "runs" / controller.NAME / "admission.json"
    assert global_path.read_bytes() == local_path.read_bytes()
    assert json.loads(global_path.read_text()) == result
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=2000.0)


def test_concurrent_admission_has_one_winner(tmp_path, manifest):
    def attempt(_):
        try:
            return controller.claim(tmp_path, manifest, STATE, now=1000.0)
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
        controller.claim(tmp_path, manifest, STATE, now=1000.0)
    assert (tmp_path / "campaign_admissions" / controller.NAME / "admission.json").exists()
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=1000.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_seconds", 3401),
        ("gpu", "H100"),
        ("cpu", 4),
        ("memory_gib", 96),
        ("ceiling_usd", 4),
        ("retries", 1),
        ("rate_usd_per_second", 0.001),
    ],
)
def test_policy_changes_rejected_before_admission(tmp_path, manifest, field, value):
    manifest["resources"][field] = value
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, now=1000.0)
    assert not (tmp_path / "campaign_admissions").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("all_stopped_zero_tasks", False),
        ("all_stopped_zero_tasks", 1),
        ("available_compute_estimate_usd", 12.0),
        ("available_compute_estimate_usd", float("nan")),
        ("phase_compute_estimate_usd", True),
    ],
)
def test_active_or_invalid_or_insufficient_budget_rejected(tmp_path, manifest, field, value):
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, dict(STATE, **{field: value}), now=1000.0)
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


@pytest.mark.parametrize(
    "name,kind",
    [("../escape", "file"), ("/absolute", "file"), ("safe", "symlink"), ("duplicate", "duplicate")],
)
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
    wrapper = tmp_path / "engineering"
    assert controller.snapshot(tmp_path, wrapper) == STATE
    assert seen["env"]["PYTHONPATH"] == str(wrapper / "accounting")
    assert seen["env"]["CURIOSITY_BUDGET_ROOT"] == str(tmp_path)
    assert seen["timeout"] == 25


def test_runtime_has_only_two_declared_overlays_and_original_is_untouched(tmp_path):
    original, project, runtime = (tmp_path / name for name in ("original", "project", "runtime"))
    for root in (original, project):
        (root / "src/curiosity_study").mkdir(parents=True)
    (original / "src/curiosity_study/__init__.py").write_text("original init\n")
    (original / "src/curiosity_study/model.py").write_text("original model\n")
    (original / "src/curiosity_study/state_backend.py").write_text("original backend\n")
    for name in controller.OVERLAYS:
        (project / "src/curiosity_study" / name).write_text("reviewed " + name)
    before = controller.inventory(original)
    files, delta = controller.build_runtime(project, original, runtime)
    assert controller.inventory(original) == before
    assert set(delta) == {"state_backend.py", "coverage_engineering.py"}
    assert delta["coverage_engineering.py"]["original_byte_sha256"] is None
    assert delta["state_backend.py"]["original_byte_sha256"] == before["src/curiosity_study/state_backend.py"]
    assert files["curiosity_study/model.py"] == before["src/curiosity_study/model.py"]
    assert files["curiosity_study/__init__.py"] == before["src/curiosity_study/__init__.py"]
    assert set(files) == {
        "curiosity_study/" + name for name in ("__init__.py", "model.py", *controller.OVERLAYS)
    }


def test_prepare_rejects_consumed_destination_before_source_or_tokenizer_work(tmp_path, monkeypatch):
    execution = tmp_path / "executions" / controller.NAME
    execution.mkdir(parents=True)
    monkeypatch.setattr(controller, "read", lambda *args: pytest.fail("must reject before source access"))
    with pytest.raises(ValueError, match="already prepared"):
        controller.prepare(tmp_path)
    assert not (tmp_path / "runs").exists()


def test_claim_does_not_admit_scientific_work(tmp_path, manifest):
    manifest["scientific_admission"] = True
    with pytest.raises(ValueError, match="policy differs"):
        controller.claim(tmp_path, manifest, STATE, now=1000.0)
    assert not (tmp_path / "campaign_admissions").exists()
