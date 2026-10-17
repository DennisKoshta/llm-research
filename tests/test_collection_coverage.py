"""Synthetic storage fixtures for operational completion, without model outcomes."""

import copy
from pathlib import Path
import sys

import pytest

from curiosity_study.artifacts import RunStore, digest, read, write_once

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collection_coverage import CollectionCoverageError, check_collection_coverage  # noqa: E402


def record(assignment):
    return dict(
        trial_id=assignment["trial_id"],
        assignment=assignment,
        terminal=True,
        # Neither scientific success nor parse validity is an operational gate.
        correct=False,
        valid=False,
        answer={"opaque": "The coverage checker must not interpret this field"},
    )


def fixture(tmp_path, count=5, size=2, with_batch=False):
    rows = [dict(trial_id=f"trial-{i}", index=i, seed=710) for i in range(count)]
    manifest = dict(
        assay="thinking_information",
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        execution=dict(backend="torch_batched_v1", batch_size=size),
    )
    root = tmp_path / "child"
    store = RunStore(root, manifest)
    for offset in range(0, count, size):
        cohort = rows[offset : offset + size]
        records = [record(a) for a in cohort]
        if with_batch:
            for row, r in enumerate(records):
                r["inference"] = dict(
                    batch=dict(
                        backend="torch_batched_v1",
                        id=f"batch-{offset}",
                        cohort_ids=[a["trial_id"] for a in cohort],
                        size=len(cohort),
                        row=row,
                    )
                )
        store.save_many(records, "attempt-one", f"{offset:07d}")
    return root, manifest


def check(root, manifest):
    return check_collection_coverage(root, expected_manifest_sha256=digest(manifest))


def replace(path, value):
    path.unlink()
    write_once(path, value)


def alter_records(root, callback, offset=0):
    path = root / "cohorts/attempt-one" / f"{offset:07d}.json"
    envelope = read(path)
    callback(envelope["records"])
    envelope["sha256"] = digest(envelope["records"])
    replace(path, envelope)


@pytest.mark.parametrize("with_batch", [False, True])
def test_complete_atomic_collection_passes_without_scientific_success_or_writes(tmp_path, with_batch):
    root, manifest = fixture(tmp_path, with_batch=with_batch)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    receipt = check(root, manifest)
    assert receipt["passed"]
    assert receipt["assigned_trials"] == receipt["terminal_trials"] == 5
    assert receipt["verified_cohorts"] == receipt["expected_cohorts"] == 3
    assert receipt["scientific_outcomes_checked"] is False
    assert receipt == check(root, manifest)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_deadline_return_with_missing_final_cohort_cannot_look_complete(tmp_path):
    root, manifest = fixture(tmp_path)
    (root / "cohorts/attempt-one/0000004.json").unlink()
    # A normal return profile or exit code cannot override physical coverage.
    write_once(root / "profiles/clean-exit.json", dict(completed=4, assigned=5, exit_code=0))
    with pytest.raises(CollectionCoverageError, match="Incomplete collection: 4/5"):
        check(root, manifest)


@pytest.mark.parametrize(
    "problem",
    ["nonterminal", "execution_error", "wrong_assignment", "wrong_trial", "missing_record", "reverse_order"],
)
def test_terminal_assignment_and_cohort_order_are_required(tmp_path, problem):
    root, manifest = fixture(tmp_path)

    def mutate(records):
        if problem == "nonterminal":
            records[0]["terminal"] = False
        elif problem == "execution_error":
            records[0]["failure"] = "execution_error"
        elif problem == "wrong_assignment":
            records[0]["assignment"]["seed"] += 1
        elif problem == "wrong_trial":
            records[0]["trial_id"] = "extra-trial"
        elif problem == "missing_record":
            records.pop()
        else:
            records.reverse()

    alter_records(root, mutate)
    with pytest.raises(CollectionCoverageError):
        check(root, manifest)


def test_identical_replayed_cohort_is_rejected_even_when_runstore_deduplicates(tmp_path):
    root, manifest = fixture(tmp_path)
    original = read(root / "cohorts/attempt-one/0000000.json")
    write_once(root / "cohorts/attempt-two/0000000.json", original)
    assert len(RunStore(root, manifest).completed()) == 5
    with pytest.raises(CollectionCoverageError, match="Duplicate physical cohort"):
        check(root, manifest)


def test_assignment_binding_rejects_type_drift_even_when_python_equality_matches(tmp_path):
    root, manifest = fixture(tmp_path)
    alter_records(root, lambda rows: rows[0]["assignment"].update(index=0.0))
    changed = read(root / "cohorts/attempt-one/0000000.json")["records"][0]["assignment"]
    assert changed == manifest["diagnostics"][0]
    assert digest(changed) != digest(manifest["diagnostics"][0])
    with pytest.raises(CollectionCoverageError, match="Record assignment"):
        check(root, manifest)


def test_individually_complete_rows_cannot_substitute_for_atomic_cohorts(tmp_path):
    root, manifest = fixture(tmp_path)
    original = read(root / "cohorts/attempt-one/0000000.json")
    (root / "cohorts/attempt-one/0000000.json").unlink()
    store = RunStore(root, manifest)
    for r in original["records"]:
        store.save(r, "individual")
    assert len(store.completed()) == 5
    with pytest.raises(CollectionCoverageError, match="Individual attempt records"):
        check(root, manifest)


def test_partial_cohort_retries_are_rejected_even_if_all_ids_eventually_exist(tmp_path):
    root, manifest = fixture(tmp_path)
    first = read(root / "cohorts/attempt-one/0000000.json")["records"]
    alter_records(root, lambda rows: rows.pop())
    RunStore(root, manifest).save_many(first[1:], "attempt-two", "0000000")
    assert len(RunStore(root, manifest).completed()) == 5
    with pytest.raises(CollectionCoverageError, match="Atomic cohort membership"):
        check(root, manifest)


@pytest.mark.parametrize("offset", ["0000001", "0000006", "not-an-offset"])
def test_unexpected_or_unaligned_extra_cohorts_are_rejected(tmp_path, offset):
    root, manifest = fixture(tmp_path)
    write_once(
        root / "cohorts/extra" / f"{offset}.json",
        read(root / "cohorts/attempt-one/0000000.json"),
    )
    with pytest.raises(CollectionCoverageError):
        check(root, manifest)


def test_duplicate_trials_within_one_atom_are_rejected(tmp_path):
    root, manifest = fixture(tmp_path)
    alter_records(root, lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
    with pytest.raises(CollectionCoverageError, match="Atomic cohort membership"):
        check(root, manifest)


@pytest.mark.parametrize("field", ["manifest_hash", "sha256"])
def test_corrupt_envelope_bindings_are_rejected(tmp_path, field):
    root, manifest = fixture(tmp_path)
    path = root / "cohorts/attempt-one/0000000.json"
    envelope = read(path)
    envelope[field] = "corrupt"
    replace(path, envelope)
    with pytest.raises(CollectionCoverageError):
        check(root, manifest)


def test_expected_parent_binding_cannot_be_replaced_by_current_manifest_hash(tmp_path):
    root, manifest = fixture(tmp_path)
    changed = {**manifest, "unplanned_change": True}
    replace(root / "manifest.json", changed)
    with pytest.raises(CollectionCoverageError, match="frozen expected binding"):
        check(root, manifest)


@pytest.mark.parametrize(
    "problem", ["duplicate_ids", "wrong_digest", "empty", "bad_batch_size", "unsupported_backend"]
)
def test_malformed_frozen_assignments_do_not_pass(tmp_path, problem):
    root, manifest = fixture(tmp_path)
    if problem == "duplicate_ids":
        manifest["diagnostics"][1]["trial_id"] = manifest["diagnostics"][0]["trial_id"]
    elif problem == "empty":
        manifest["diagnostics"] = []
    elif problem == "bad_batch_size":
        manifest["execution"]["batch_size"] = True
    elif problem == "unsupported_backend":
        manifest["execution"]["backend"] = "unplanned"
    manifest["assignment_sha256"] = "wrong" if problem == "wrong_digest" else digest(manifest["diagnostics"])
    replace(root / "manifest.json", manifest)
    with pytest.raises(CollectionCoverageError):
        check(root, manifest)


@pytest.mark.parametrize("problem", ["cohort_ids", "row", "size", "id", "missing_one_batch"])
def test_present_inference_batch_metadata_must_match_atomic_row_order(tmp_path, problem):
    root, manifest = fixture(tmp_path, with_batch=True)

    def mutate(records):
        batch = records[0]["inference"]["batch"]
        if problem == "cohort_ids":
            batch["cohort_ids"].reverse()
        elif problem == "row":
            batch["row"] = 1
        elif problem == "size":
            batch["size"] = 1
        elif problem == "id":
            batch["id"] = "different-batch"
        else:
            del records[0]["inference"]["batch"]

    alter_records(root, mutate)
    with pytest.raises(CollectionCoverageError):
        check(root, manifest)


def test_published_json_outside_cohort_layout_is_not_silently_ignored(tmp_path):
    root, manifest = fixture(tmp_path)
    write_once(root / "cohorts/unexpected/nested/0000000.json", {})
    with pytest.raises(CollectionCoverageError, match="exactly one attempt directory"):
        check(root, manifest)


def test_combined_assignment_then_diagnostic_order_is_preserved(tmp_path):
    root, manifest = fixture(tmp_path)
    rows = manifest["diagnostics"]
    manifest["assignments"], manifest["diagnostics"] = rows[:1], rows[1:]
    replace(root / "manifest.json", manifest)
    for path in (root / "cohorts").rglob("*.json"):
        envelope = read(path)
        envelope["manifest_hash"] = digest(manifest)
        replace(path, envelope)
    assert check(root, manifest)["passed"]
