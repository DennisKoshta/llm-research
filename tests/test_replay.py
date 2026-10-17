import pytest
from curiosity_study.artifacts import RunStore, read
from curiosity_study.assignments import make_assignments
from curiosity_study.runner import build_manifest, execute
from curiosity_study.model import MockModel


def test_replay_does_not_regenerate_terminal_outputs(tmp_path):
    model = MockModel()
    manifest = build_manifest(make_assignments(1), model.metadata())
    first = execute(tmp_path, manifest, model)

    class NeverCalled(MockModel):
        def infer(self, *args, **kwargs):
            raise AssertionError("Replay attempted new inference")

    second = execute(tmp_path, manifest, NeverCalled())
    assert first == second and len(first) == 28


def test_immutable_artifacts_and_hash_validation(tmp_path):
    manifest = {"model": "mock"}
    store = RunStore(tmp_path, manifest)
    record = {"trial_id": "x", "terminal": True, "first_service": "invalid"}
    store.save(record, "one")
    assert store.completed()["x"]["first_service"] == "invalid"
    with pytest.raises(ValueError):
        store.save(dict(record, first_service="target"), "one")
    path = tmp_path / "attempts/one/x.json"
    envelope = read(path)
    envelope["record"]["first_service"] = "target"
    path.write_text(__import__("json").dumps(envelope))
    with pytest.raises(ValueError, match="integrity"):
        store.completed()


def test_conflicting_terminal_attempts_are_rejected(tmp_path):
    store = RunStore(tmp_path, {})
    store.save({"trial_id": "x", "terminal": True, "raw": "A"}, "one")
    store.save({"trial_id": "x", "terminal": True, "raw": "B"}, "two")
    with pytest.raises(ValueError, match="Conflicting"):
        store.completed()


def test_interrupted_attempt_can_resume_without_overwriting(tmp_path):
    store = RunStore(tmp_path, {})
    store.save({"trial_id": "x", "terminal": False, "failure": "interruption"}, "one")
    assert not store.completed()
    store.save({"trial_id": "x", "terminal": True, "raw": "A"}, "two")
    assert len(store.completed()) == 1
    assert len(list((tmp_path / "attempts").glob("*/*.json"))) == 2


def test_single_writer_volume_without_hardlinks(tmp_path, monkeypatch):
    import errno
    from curiosity_study.artifacts import write_once

    def unavailable(*args):
        raise PermissionError(errno.EPERM, "Hard links unavailable")

    monkeypatch.setattr("os.link", unavailable)
    path = tmp_path / "record.json"
    write_once(path, {"value": 1})
    write_once(path, {"value": 1})
    assert read(path) == {"value": 1}
    with pytest.raises(ValueError):
        write_once(path, {"value": 2})
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_cohort_storage_validates_every_record_and_conflicting_attempt(tmp_path):
    store = RunStore(tmp_path, {})
    rows = [
        {"trial_id": "a", "terminal": True, "answer": "A"},
        {"trial_id": "b", "terminal": True, "answer": "B"},
    ]
    store.save_many(rows, "one", "cohort0")
    assert set(store.completed()) == {"a", "b"}
    store.save({"trial_id": "a", "terminal": True, "answer": "C"}, "two")
    with pytest.raises(ValueError, match="Conflicting terminal"):
        store.completed()


def test_cohort_corruption_is_not_silently_accepted(tmp_path):
    import json

    store = RunStore(tmp_path, {})
    store.save_many([{"trial_id": "a", "terminal": True, "answer": "A"}], "one", "cohort0")
    path = tmp_path / "cohorts/one/cohort0.json"
    payload = read(path)
    payload["records"][0]["answer"] = "B"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity failure"):
        store.completed()
