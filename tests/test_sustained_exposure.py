"""Input-only history reconstruction; all engineering evidence is synthetic."""

from copy import deepcopy
import gzip
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import pytest

from curiosity_study import information_state, voluntary_resolution as voluntary
from curiosity_study.artifacts import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_voluntary_freshness as freshness  # noqa: E402
import sustained_exposure as adapter  # noqa: E402


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def tar_bytes(files):
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, payload in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    return gzip.compress(output.getvalue(), mtime=0)


def write(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def fixture_state(root, *, executed=True, cap=128, run="runs/engineering", offsets=(0,), failed=True):
    assignments = voluntary.assignments()
    sources = {
        Path(module.__file__).name: Path(module.__file__).read_text()
        for module in (information_state, voluntary)
    }
    source_archive = tar_bytes(
        {"src/curiosity_study/" + name: text.encode() for name, text in sources.items()}
    )
    source_manifest = dict(
        assay="voluntary_resolution",
        task_revision=voluntary.REVISION,
        source_sha256=digest(sources),
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
    )
    source_freeze = dict(
        manifest_sha256=digest(source_manifest),
        source_sha256=digest(sources),
        archive_sha256=adapter.sha_bytes(source_archive),
    )
    records, requests = [], []
    for index, assignment in enumerate(assignments[:32]):
        tokens = [1, index + 10, 3]
        records.append(
            dict(
                trial_id=assignment["trial_id"],
                assignment=assignment,
                inference=dict(
                    unpadded_input_ids=tokens,
                    enable_thinking=True,
                    sampling=dict(seed=assignment["sampling_seed"]),
                    batch=dict(row=index, size=32, cohort_ids=[a["trial_id"] for a in assignments[:32]]),
                ),
                activation_artifact=dict(positions=["record", "prefill"], token_positions=[1, 2]),
                # This deliberately is not a score schema. Exposure must ignore it.
                correct="must never be interpreted",
                raw="not an answer",
            )
        )
        requests.append(
            dict(
                source_trial_id=assignment["trial_id"],
                token_prefix=tokens,
                token_prefix_sha256=digest(tokens),
                prefix_positions=dict(record=1, final=2),
                labels=list("ABCD"),
                seed=assignment["sampling_seed"],
            )
        )
    cohort = dict(manifest_hash=digest(source_manifest), records=records, sha256=digest(records))
    record_hashes = {row["trial_id"]: digest(row) for row in records}
    source_audit = dict(
        complete=True,
        integrity_passed=True,
        passed=False,
        assigned=288,
        manifest_sha256=digest(source_manifest),
        source_sha256=digest(sources),
        source_freeze=dict(archive_sha256=source_freeze["archive_sha256"]),
        readiness=dict(passed=False),
        outcomes=[
            dict(trial_id=a["trial_id"], record_sha256=record_hashes.get(a["trial_id"], "unneeded"))
            for a in assignments
        ],
    )
    engine_sources = {**sources, "synthetic_engine.py": "ENGINEERING = True\n"}
    manifest = dict(
        assay="sustained_engineering",
        phase="engineering",
        scientific_admission=False,
        cap=cap,
        source_sha256=digest(engine_sources),
        requests=requests,
        assignments=[],
    )
    manifest["diagnostics"] = [
        dict(
            trial_id=f"sustained-{cap}-{condition}-{row:02d}",
            kind="engineering",
            condition=condition,
            source_row=row,
            source_trial_id=request["source_trial_id"],
            request_sha256=digest(request),
            cap=cap,
        )
        for condition in adapter.CONDITIONS
        for row, request in enumerate(requests)
    ]
    manifest["assignment_sha256"] = digest(manifest["diagnostics"])
    state = dict(
        root=root,
        run=run,
        executed=executed,
        offsets=offsets,
        failed=failed,
        sources=sources,
        source_archive=source_archive,
        source_manifest=source_manifest,
        source_freeze=source_freeze,
        cohort=cohort,
        source_audit=source_audit,
        engine_sources=engine_sources,
        manifest=manifest,
    )
    seal(state)
    return state


def seal(state):
    """Bind changed synthetic evidence without repairing the original audit ledger."""
    root, manifest = state["root"], state["manifest"]
    source_paths = dict(
        manifest_path="runs/source/manifest.json",
        audit_path="reports/source-audit.json",
        freeze_path="runs/source/freeze.json",
        archive_path="runs/source/source.tar.gz",
        first_cohort_path="runs/source/cohorts/source/0000000.json",
    )
    cohort = state["cohort"]
    cohort["sha256"] = digest(cohort["records"])
    documents = {
        source_paths["manifest_path"]: json_bytes(state["source_manifest"]),
        source_paths["audit_path"]: json_bytes(state["source_audit"]),
        source_paths["freeze_path"]: json_bytes(state["source_freeze"]),
        source_paths["archive_path"]: state["source_archive"],
        source_paths["first_cohort_path"]: json_bytes(cohort),
    }
    for name, payload in documents.items():
        write(root, name, payload)
    manifest["source_binding"] = dict(
        **source_paths,
        manifest_sha256=digest(state["source_manifest"]),
        audit_sha256=digest(state["source_audit"]),
        audit_byte_sha256=adapter.sha_bytes(documents[source_paths["audit_path"]]),
        freeze_sha256=digest(state["source_freeze"]),
        archive_byte_sha256=adapter.sha_bytes(state["source_archive"]),
        first_cohort_sha256=digest(cohort),
        first_cohort_byte_sha256=adapter.sha_bytes(documents[source_paths["first_cohort_path"]]),
        records=[
            dict(trial_id=r["trial_id"], record_sha256=digest(r), assignment_sha256=digest(r["assignment"]))
            for r in cohort["records"]
        ],
    )
    manifest["document_bindings"] = [
        dict(path=name, byte_sha256=adapter.sha_bytes(payload)) for name, payload in sorted(documents.items())
    ]
    archive = tar_bytes(
        {
            **documents,
            **{
                "src/curiosity_study/" + name: text.encode() for name, text in state["engine_sources"].items()
            },
        }
    )
    write(root, state["run"] + "/manifest.json", json_bytes(manifest))
    write(root, state["run"] + "/source.tar.gz", archive)
    write(
        root,
        state["run"] + "/freeze.json",
        json_bytes(
            dict(
                manifest_sha256=digest(manifest),
                source_sha256=manifest["source_sha256"],
                archive_sha256=adapter.sha_bytes(archive),
                documents={name: adapter.sha_bytes(payload) for name, payload in documents.items()},
            )
        ),
    )
    if state["executed"]:
        for offset in state["offsets"]:
            rows = [
                dict(
                    assignment=a,
                    trial_id=a["trial_id"],
                    terminal=True,
                    engineering_passed=not state["failed"],
                    status="execution_error" if state["failed"] else "measured",
                )
                for a in manifest["diagnostics"][offset : offset + 32]
            ]
            write(
                root,
                f"{state['run']}/cohorts/attempt/{offset:07d}.json",
                json_bytes(
                    dict(
                        manifest_hash=digest(manifest),
                        records=rows,
                        sha256=digest(rows),
                    )
                ),
            )


def test_failed_partial_engineering_recovers_all32_bound_original_inputs(tmp_path):
    state = fixture_state(tmp_path, failed=True, offsets=(64,))
    result = adapter.source_inputs(tmp_path, state["run"], state["manifest"])
    assert result["assignments"] == state["source_manifest"]["diagnostics"][:32]
    assert result["source_sha256"] == state["source_manifest"]["source_sha256"]
    assert result["sources"]["sources"] == state["sources"]
    binding = result["provenance"]
    assert binding["adapter_byte_sha256"] == adapter.sha_bytes(Path(adapter.__file__).read_bytes())
    assert binding["source_audit_sha256"] == digest(state["source_audit"])
    assert len(binding["terminal_evidence"]) == 1


def test_prepared_only_engineering_contributes_no_exposure_and_does_not_need_adapter(tmp_path, monkeypatch):
    fixture_state(tmp_path, executed=False)
    monkeypatch.setattr(adapter, "source_inputs", lambda *args: pytest.fail("Prepared engine was adapted"))
    result = freshness.audit(tmp_path, pool_candidates=0)
    entry = next(row for row in result["historical_registry"] if row["run"] == "runs/engineering")
    assert entry["semantic_screen"] == "not_verified_executed"
    assert all("runs/engineering" not in row["source_runs"] for row in result["exact_world_exclusions"])


def test_short_and_full_reexposure_add_provenance_without_new_semantic_ids(tmp_path):
    short = fixture_state(tmp_path, run="runs/short", cap=128)
    fixture_state(tmp_path, run="runs/full", cap=4096)
    result = freshness.audit(tmp_path, pool_candidates=0)
    expected = {
        digest(voluntary.canonical_world(a["family"], voluntary.task(a)))
        for a in short["source_manifest"]["diagnostics"]
    }
    assert set(result["historical_exact_world_ids"]) == expected
    first32 = {
        digest(voluntary.canonical_world(a["family"], voluntary.task(a)))
        for a in short["source_manifest"]["diagnostics"][:32]
    }
    for row in result["exact_world_exclusions"]:
        assert row["source_runs"] == (
            ["runs/full", "runs/short", "runs/source"]
            if row["semantic_world_id"] in first32
            else ["runs/source"]
        )
    assert len(expected) == 24
    entries = {row["run"]: row for row in result["historical_registry"]}
    assert all("source_input_adapter" in entries[name] for name in ("runs/short", "runs/full"))
    history = freshness.history_receipt(result)
    assert history["historical_registry_sha256"] == digest(history["historical_registry"])


def test_rehashed_source_record_cannot_escape_original_audit_ledger(tmp_path):
    state = fixture_state(tmp_path)
    state["cohort"]["records"][0]["raw"] = "changed even though this text is never interpreted"
    seal(state)
    with pytest.raises(ValueError, match="audit ledger"):
        adapter.source_inputs(tmp_path, state["run"], state["manifest"])


@pytest.mark.parametrize("mutation", ["seed", "prefix", "row", "assignment", "audit_coverage"])
def test_source_request_and_ledger_bindings_fail_closed(tmp_path, mutation):
    state = fixture_state(tmp_path)
    if mutation == "seed":
        state["manifest"]["requests"][0]["seed"] += 1
    elif mutation == "prefix":
        state["manifest"]["requests"][0]["token_prefix"][0] = 999
    elif mutation == "row":
        state["cohort"]["records"][0]["inference"]["batch"]["row"] = 31
    elif mutation == "assignment":
        state["cohort"]["records"][0]["assignment"] = deepcopy(state["cohort"]["records"][1]["assignment"])
    else:
        state["source_audit"]["outcomes"].pop()
    for assignment in state["manifest"]["diagnostics"]:
        assignment["request_sha256"] = digest(state["manifest"]["requests"][assignment["source_row"]])
    state["manifest"]["assignment_sha256"] = digest(state["manifest"]["diagnostics"])
    seal(state)
    with pytest.raises(ValueError):
        adapter.source_inputs(tmp_path, state["run"], state["manifest"])


@pytest.mark.parametrize("mutation", ["nonterminal", "wrong_manifest", "duplicate_attempt", "unbound_bytes"])
def test_only_bound_durable_terminal_evidence_is_admitted(tmp_path, mutation):
    state = fixture_state(tmp_path)
    path = tmp_path / state["run"] / "cohorts/attempt/0000000.json"
    envelope = json.loads(path.read_text())
    if mutation == "nonterminal":
        envelope["records"][0]["terminal"] = False
        envelope["sha256"] = digest(envelope["records"])
        path.write_bytes(json_bytes(envelope))
    elif mutation == "wrong_manifest":
        envelope["manifest_hash"] = "other"
        path.write_bytes(json_bytes(envelope))
    elif mutation == "duplicate_attempt":
        write(tmp_path, state["run"] + "/cohorts/other/0000000.json", path.read_bytes())
    else:
        envelope["records"][0]["terminal"] = False
        path.write_bytes(json_bytes(envelope))
    with pytest.raises(ValueError):
        adapter.source_inputs(tmp_path, state["run"], state["manifest"])


def test_prepared_source_cannot_be_claimed_executed_by_manifest_alone(tmp_path):
    state = fixture_state(tmp_path, executed=False)
    with pytest.raises(ValueError, match="no durable terminal"):
        adapter.source_inputs(tmp_path, state["run"], state["manifest"])


def test_unknown_assay_cannot_hide_behind_known_task_revision():
    with pytest.raises(ValueError, match="Unknown finite-task revision/assay"):
        freshness.revision_adapter(voluntary.REVISION, "unrecognized_engine")
