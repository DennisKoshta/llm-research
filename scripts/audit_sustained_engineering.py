"""Recompute bounded sustained-engineering checks from immutable saved evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study import sustained_engineering as engine
from curiosity_study.artifacts import canonical, digest, read, write_once

SOURCE_MANIFEST_SHA256 = "b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    return path


def frozen_documents(root, manifest):
    frozen = read(root / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha(root / "source.tar.gz"),
        "Engineering source freeze mismatch",
    )
    documents = {}
    with tarfile.open(root / "source.tar.gz") as archive:
        for binding in manifest["document_bindings"]:
            name = str(safe_relative(binding["path"]))
            require(name not in documents, "Duplicate frozen document")
            member = archive.getmember(name)
            require(member.isfile(), "Frozen document is not a regular file")
            payload = archive.extractfile(member).read()
            require(
                hashlib.sha256(payload).hexdigest() == binding["byte_sha256"] == frozen["documents"][name],
                "Frozen engineering document changed",
            )
            documents[name] = payload
        for name, expected in manifest["engine_sha256"].items():
            payload = archive.extractfile("src/curiosity_study/" + name).read()
            require(
                hashlib.sha256(payload).hexdigest() == expected, "Frozen engineering implementation changed"
            )
    return documents, dict(freeze_sha256=digest(frozen), archive_sha256=frozen["archive_sha256"])


def verify_sources(manifest, documents):
    """Bind the first actual source cohort; source readiness is recorded, not gated."""
    source = manifest["source_binding"]
    source_manifest = json.loads(documents[source["manifest_path"]])
    source_audit = json.loads(documents[source["audit_path"]])
    source_freeze = json.loads(documents[source["freeze_path"]])
    envelope = json.loads(documents[source["first_cohort_path"]])
    require(
        digest(source_manifest) == source["manifest_sha256"] == SOURCE_MANIFEST_SHA256
        and source_manifest["assay"] == "voluntary_resolution"
        and source_manifest["task_revision"] == "voluntary-resolution-v2"
        and source_manifest["model"] == manifest["model"]
        and digest(source_audit) == source["audit_sha256"]
        and hashlib.sha256(documents[source["audit_path"]]).hexdigest() == source["audit_byte_sha256"]
        and digest(source_freeze) == source["freeze_sha256"]
        and hashlib.sha256(documents[source["archive_path"]]).hexdigest()
        == source["archive_byte_sha256"]
        == source_freeze["archive_sha256"]
        and source_freeze["manifest_sha256"] == digest(source_manifest)
        and source_freeze["source_sha256"] == source_manifest["source_sha256"],
        "Source manifest/audit/freeze/archive bindings disagree",
    )
    require(
        source_audit["manifest_sha256"] == digest(source_manifest)
        and source_audit["complete"] is True
        and source_audit["integrity_passed"] is True
        and source_audit["source_sha256"] == source_manifest["source_sha256"]
        and source_audit["source_freeze"]["archive_sha256"] == source_freeze["archive_sha256"]
        and source["readiness_passed"] is source_audit["readiness"]["passed"],
        "Require complete source integrity and faithfully reported readiness",
    )
    require(
        digest(envelope) == source["first_cohort_sha256"]
        and hashlib.sha256(documents[source["first_cohort_path"]]).hexdigest()
        == source["first_cohort_byte_sha256"]
        and envelope["manifest_hash"] == digest(source_manifest)
        and envelope["sha256"] == digest(envelope["records"]),
        "Source first-cohort envelope changed",
    )
    assignments = (source_manifest.get("assignments") or []) + source_manifest["diagnostics"]
    audited = source_audit["outcomes"]
    require(
        len(assignments) == len(audited) == 288
        and len({a["trial_id"] for a in assignments}) == 288
        and len({r["trial_id"] for r in audited}) == 288
        and {r["trial_id"] for r in audited} == {a["trial_id"] for a in assignments},
        "Source audit does not bind exactly all 288 assigned records",
    )
    audited_hashes = {r["trial_id"]: r["record_sha256"] for r in audited}
    records = envelope["records"]
    require(
        len(records) == 32 and [r["trial_id"] for r in records] == [a["trial_id"] for a in assignments[:32]],
        "Engineering sources must be the first actual32 original rows in order",
    )
    bindings = []
    for index, (record, assignment, request) in enumerate(
        zip(records, assignments[:32], manifest["requests"])
    ):
        inference, artifact = record["inference"], record["activation_artifact"]
        tokens = inference["unpadded_input_ids"]
        positions = dict(zip(artifact["positions"], artifact["token_positions"]))
        expected = dict(
            source_trial_id=record["trial_id"],
            token_prefix=tokens,
            token_prefix_sha256=digest(tokens),
            prefix_positions=dict(record=positions["record"], final=len(tokens) - 1),
            labels=list("ABCD"),
            seed=assignment["sampling_seed"],
        )
        require(
            digest(record) == audited_hashes[record["trial_id"]]
            and record["assignment"] == assignment
            and positions["prefill"] == len(tokens) - 1
            and inference["sampling"]["seed"] == assignment["sampling_seed"]
            and inference["batch"]["row"] == index
            and inference["batch"]["size"] == 32
            and inference["batch"]["cohort_ids"] == [r["trial_id"] for r in records]
            and request == expected,
            "Audited source record, original prompt tokens, positions, seed, companions or source order changed",
        )
        bindings.append(
            dict(
                trial_id=record["trial_id"],
                record_sha256=digest(record),
                assignment_sha256=digest(assignment),
            )
        )
    require(bindings == source["records"], "Source record bindings changed")
    direction = manifest["direction"]
    with tarfile.open(fileobj=io.BytesIO(documents[source["archive_path"]])) as archive:
        member = archive.getmember(direction["directions_document"])
        require(member.isfile(), "Direction archive member is not a regular file")
        payload = archive.extractfile(member).read()
    require(
        hashlib.sha256(payload).hexdigest() == direction["directions_byte_sha256"],
        "Direction archive changed",
    )
    require(
        source_freeze["documents"][direction["directions_document"]] == direction["directions_byte_sha256"],
        "Direction bytes are not bound by source freeze",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        s = z["sources"].tolist().index("curiosity_vs_indifference")
        layer = z["layers"].tolist().index(15)
        position = z["positions"].tolist().index("record")
        unit, norm = z["unit"][s, layer, position], float(z["contrast_norm"][s, layer, position])
        require(
            engine.array_binding(unit) == direction["unit_binding"]
            and np.array_equal(unit, np.asarray(direction["unit"], dtype=direction["unit_dtype"]))
            and norm == direction["norm"] == direction["dose"]
            and direction["candidate_source"]
            == source_manifest["candidate_source"]
            == source_audit["candidate_source"]
            and source_manifest["candidate_source"]["directions_byte_sha256"]
            == direction["directions_byte_sha256"],
            "Require exactly the frozen corpus-A15/record curiosity direction and raw norm",
        )
    if manifest["cap"] == 4096:
        stage = manifest["short_stage"]
        short_audit = json.loads(documents[stage["audit_path"]])
        short_manifest = json.loads(documents[stage["manifest_path"]])
        require(
            short_audit == stage["audit"]
            and digest(short_audit) == stage["audit_sha256"]
            and hashlib.sha256(documents[stage["audit_path"]]).hexdigest() == stage["audit_byte_sha256"]
            and digest(short_manifest) == stage["manifest_sha256"]
            and digest(engine.compatibility_binding(short_manifest)) == manifest["compatibility_sha256"],
            "Full-cap stage does not bind the same independently audited short-stage inputs",
        )
    return dict(
        source_manifest_sha256=digest(source_manifest), source_readiness_passed=source["readiness_passed"]
    )


def recompute_checks(harness, arrays, requests, direction, dose, cap):
    """Ignore saved check booleans while recalculating from observations."""
    import torch

    require(set(arrays) == {"before", "after", "direction_float32"}, "Unexpected vector arrays")
    effective = np.asarray(direction, dtype=np.float32)
    require(np.array_equal(arrays["direction_float32"], effective), "Actual steering direction changed")
    bindings = {key: engine.array_binding(value) for key, value in arrays.items()}
    require(
        harness["arrays"] == bindings and harness["arrays_sha256"] == digest(bindings),
        "Vector content bindings changed",
    )
    require(
        harness["schema_version"] == 1
        and harness["scientific_admission"] is False
        and harness["schedule"] == engine.SCHEDULE
        and harness["layer"] == 15
        and harness["cap"] == cap
        and harness["batch_size"] == 32
        and harness["perturbed_row"] == 0
        and harness["request_sha256"] == digest(requests)
        and harness["source_trial_ids"] == [r["source_trial_id"] for r in requests]
        and harness["seeds"] == [r["seed"] for r in requests]
        and harness["dose"] == dose
        and harness["effective_direction"] == bindings["direction_float32"]
        and harness["prefill_padding_multiple"] == engine.PREFILL_PADDING_MULTIPLE
        and harness["supplied_direction"] == engine.array_binding(np.asarray(direction)),
        "Harness contract differs from frozen requests/direction/schedule",
    )
    conditions = harness["conditions"]
    require(set(conditions) == set(engine.CONDITIONS), "Missing or extra engineering conditions")
    zero = dict(layer=15, position="record", mode="add", direction=effective, dose=0.0)
    patches = dict(
        baseline=[None] * 32,
        zero_sham=[dict(zero) for _ in range(32)],
        one_row_nonzero=[{**zero, "dose": dose}] + [dict(zero) for _ in range(31)],
        clean_repeat=[None] * 32,
    )
    checks = {
        f"{name}_timeline": engine.validate_condition(
            condition,
            requests,
            patches[name],
            cap=cap,
            pad_token_id=harness["pad_token_id"],
            eos_ids=set(harness["eos_token_ids"]),
        )
        for name, condition in conditions.items()
    }
    malformed = harness["malformed_direction"]
    checks.update(
        decode_observed_in_every_condition=all(
            any(
                frame["phase"] == "decode" and any(frame["active_prediction_rows"]) for frame in c["forwards"]
            )
            for c in conditions.values()
        ),
        nonzero_decode_engagement=any(
            e["phase"] == "decode" and e["delta_norm"] > 0
            for e in conditions["one_row_nonzero"]["rows"][0]["intervention_trace"]["events"]
        ),
        full_sham_identity=engine.identical_rows(conditions["baseline"], conditions["zero_sham"], range(32)),
        unaffected31_identity=engine.identical_rows(
            conditions["baseline"], conditions["one_row_nonzero"], range(1, 32)
        ),
        cleanup_repeat_identity=engine.identical_rows(
            conditions["baseline"], conditions["clean_repeat"], range(32)
        ),
        malformed_rejected_cleanly=malformed["error"] is not None
        and malformed["error"]["type"] == "ValueError"
        and malformed["error"]["message"] == "Invalid sustained direction or dose"
        and not malformed["forwards"]
        and malformed["attempted_forwards"] == 0
        and malformed["hooks_clean"],
        independently_recomputed_geometry=engine.check_geometry(
            conditions["one_row_nonzero"], arrays, effective, dose, torch
        ),
        unchanged_requests=harness["request_sha256"] == digest(requests),
        hook_cleanup=all(
            c["hooks_clean"] and all(r["hooks_remaining"] == 0 for r in c["rows"])
            for c in conditions.values()
        )
        and malformed["hooks_clean"],
    )
    require(
        harness["checks"] == checks and harness["passed"] is all(checks.values()),
        "Saved engineering pass flags disagree with recomputed evidence",
    )
    require(
        harness["observed_baseline_max_tokens"]
        == max(len(r["completion"]["generated_token_ids"]) for r in conditions["baseline"]["rows"])
        and harness["observed_nonzero_row_tokens"]
        == len(conditions["one_row_nonzero"]["rows"][0]["completion"]["generated_token_ids"]),
        "Observed trajectory lengths differ from measured tokens",
    )
    return checks


def terminal_records(root, manifest):
    assignments = manifest["diagnostics"]
    require(not any((root / "attempts").rglob("*.json")), "Engineering requires atomic cohorts")
    rows, offsets, attempts = [], set(), set()
    for path in sorted((root / "cohorts").rglob("*.json")):
        require(len(path.relative_to(root / "cohorts").parts) == 2, "Unexpected cohort artifact location")
        offset = int(path.stem)
        require(
            offset in (0, 32, 64, 96) and offset not in offsets, "Duplicate or unexpected engineering cohort"
        )
        envelope = read(path)
        cohort = envelope["records"]
        require(
            envelope["manifest_hash"] == digest(manifest) and envelope["sha256"] == digest(cohort),
            "Engineering cohort binding changed",
        )
        require(
            len(cohort) == 32 and [r["assignment"] for r in cohort] == assignments[offset : offset + 32],
            "Engineering cohort membership/order changed",
        )
        require(
            all(r["terminal"] is True and r["trial_id"] == r["assignment"]["trial_id"] for r in cohort),
            "Nonterminal or rebound engineering row",
        )
        offsets.add(offset)
        attempts.add(path.parent.name)
        rows.extend(cohort)
    require(len(attempts) <= 1, "Engineering cannot combine multiple attempts")
    return rows, offsets == {0, 32, 64, 96}


def audit(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    engine.validate_manifest(manifest)
    documents, frozen = frozen_documents(root, manifest)
    source = verify_sources(manifest, documents)
    if manifest["cap"] == 4096:
        stage = manifest["short_stage"]
        candidates = [parent / safe_relative(stage["run"]) for parent in root.resolve().parents]
        short_root = next(
            (
                path
                for path in candidates
                if (path / "manifest.json").is_file()
                and digest(read(path / "manifest.json")) == stage["manifest_sha256"]
            ),
            None,
        )
        require(
            short_root is not None and read(short_root / "manifest.json")["cap"] == 128,
            "Bound short-stage evidence is unavailable",
        )
        require(
            audit(short_root) == stage["audit"],
            "Bound short-stage audit no longer reproduces from report/vectors",
        )
    rows, complete = terminal_records(root, manifest)
    receipt = dict(
        schema_version=1,
        assay="sustained_engineering_audit",
        manifest_sha256=digest(manifest),
        compatibility_sha256=manifest["compatibility_sha256"],
        cap=manifest["cap"],
        complete=complete,
        terminal_records=len(rows),
        integrity_passed=False,
        engineering_passed=False,
        passed=False,
        scientific_admission=False,
        report_artifact=None,
        vector_artifact=None,
        source=source,
        freeze=frozen,
        auditor_byte_sha256=sha(__file__),
        validation_scope=(
            "Offline reconstruction from bound records and vector bytes, independent of generation; "
            "timeline, identity and geometry checks reuse the harness validators. "
            "Only observed trajectories are certified; the configured cap is an allowance."
        ),
    )
    if not complete:
        return {**receipt, "measurement_status": "incomplete"}
    reports = {digest(row["report_artifact"]) for row in rows}
    require(len(reports) == 1, "Terminal rows bind different engineering reports")
    report_artifact = rows[0]["report_artifact"]
    relative = safe_relative(report_artifact["path"])
    require(
        len(relative.parts) == 3 and relative.parts[0] == "engineering" and relative.name == "report.json",
        "Invalid engineering report path",
    )
    require(
        sha(root / relative) == report_artifact["sha256"]
        and (root / relative).stat().st_size == report_artifact["byte_count"],
        "Engineering report bytes changed",
    )
    wrapper = read(root / relative)
    require(
        wrapper["schema_version"] == 1
        and wrapper["assay"] == "sustained_engineering"
        and wrapper["manifest_sha256"] == digest(manifest)
        and wrapper["compatibility_sha256"] == manifest["compatibility_sha256"]
        and wrapper["cap"] == manifest["cap"]
        and wrapper["scientific_admission"] is False,
        "Engineering report contract changed",
    )
    for row in rows:
        a = row["assignment"]
        require(
            row["kind"] == "engineering"
            and row["condition"] == a["condition"]
            and row["source_row"] == a["source_row"]
            and row["request_sha256"] == a["request_sha256"]
            and row["status"] == wrapper["status"]
            and row["engineering_passed"] is wrapper["engineering_passed"]
            and row["scientific_admission"] is False
            and row["activation_artifact"] == wrapper["vector_artifact"],
            "Terminal engineering row/report binding mismatch",
        )
    performance = wrapper["performance"]
    harness = wrapper["harness"]
    forward_counts = (
        None
        if harness is None
        else {name: len(condition["forwards"]) for name, condition in harness["conditions"].items()}
    )
    token_counts = (
        None
        if harness is None
        else {
            name: sum(len(row["completion"]["generated_token_ids"]) for row in condition["rows"])
            for name, condition in harness["conditions"].items()
        }
    )
    require(
        performance["elapsed_seconds"] == wrapper["elapsed_seconds"]
        and performance["forward_counts_by_condition"] == forward_counts
        and performance["total_forward_count"]
        == (None if forward_counts is None else sum(forward_counts.values()))
        and performance["generated_tokens_by_condition"] == token_counts
        and performance["harness_json_bytes"] == (0 if harness is None else len(canonical(harness).encode()))
        and performance["vector_file_bytes"]
        == (0 if wrapper["vector_artifact"] is None else wrapper["vector_artifact"]["byte_count"]),
        "Performance descriptions disagree with saved observations",
    )
    receipt.update(report_artifact=report_artifact, performance=performance)
    if wrapper["status"] == "execution_error":
        require(
            wrapper["error"] is not None
            and wrapper["engineering_passed"] is False
            and all(row.get("failure") == "execution_error" for row in rows),
            "Execution error cannot be certified as measured",
        )
        for row in rows:
            observed = (
                None
                if wrapper["harness"] is None
                else wrapper["harness"]["conditions"][row["condition"]]["rows"][row["source_row"]]
            )
            require(
                row["condition_row_sha256"] == (None if observed is None else digest(observed))
                and row["error_type"] == wrapper["error"]["type"]
                and row["error"] == wrapper["error"]["message"],
                "Execution error row differs from its report",
            )
        return {
            **receipt,
            "integrity_passed": True,
            "measurement_status": "unavailable_execution_error",
            "error": wrapper["error"],
        }
    require(
        wrapper["status"] == "measured"
        and wrapper["error"] is None
        and all("failure" not in row for row in rows),
        "Unknown or inconsistent engineering execution status",
    )
    environment = read(root / "environments" / f"{relative.parts[1]}.json")
    require(
        digest(environment) == wrapper["environment_sha256"]
        and all(environment.get(k) == v for k, v in manifest["model"].items())
        and environment.get("dtype") == "bfloat16"
        and environment.get("attention") == "eager",
        "Actual engineering environment differs",
    )
    artifact = wrapper["vector_artifact"]
    vector_path = safe_relative(artifact["path"])
    require(
        vector_path == relative.parent / "vectors.npz"
        and sha(root / vector_path) == artifact["sha256"]
        and (root / vector_path).stat().st_size == artifact["byte_count"],
        "Vector archive location/bytes changed",
    )
    with np.load(root / vector_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    require(
        {key: engine.array_binding(value) for key, value in arrays.items()} == artifact["array_bindings"],
        "Vector archive contents disagree with wrapper",
    )
    harness = wrapper["harness"]
    require(
        all(harness["model"].get(k) == v for k, v in manifest["model"].items())
        and harness["model"]["parameter_dtype"] == "torch.bfloat16"
        and all(
            dtype == "torch.bfloat16"
            for dtype in harness["conditions"]["one_row_nonzero"]["vector_activation_dtypes"]
        ),
        "Require actual bound BF16 model and observed BF16 interventions",
    )
    for row in rows:
        observed = harness["conditions"][row["condition"]]["rows"][row["source_row"]]
        require(row["condition_row_sha256"] == digest(observed), "Terminal condition row is rebound")
    direction = manifest["direction"]
    checks = recompute_checks(
        harness,
        arrays,
        manifest["requests"],
        np.asarray(direction["unit"], dtype=direction["unit_dtype"]),
        direction["dose"],
        manifest["cap"],
    )
    passed = all(checks.values())
    require(wrapper["engineering_passed"] is passed, "Wrapper engineering pass differs from measurements")
    return dict(
        **{
            **receipt,
            "integrity_passed": True,
            "engineering_passed": passed,
            "passed": passed,
            "vector_artifact": {key: artifact[key] for key in ("path", "sha256", "byte_count")},
        },
        measurement_status="measured",
        checks=checks,
        environment_sha256=digest(environment),
        observed_lengths_by_condition={
            name: [len(row["completion"]["generated_token_ids"]) for row in condition["rows"]]
            for name, condition in harness["conditions"].items()
        },
        observed_maximum_tokens=max(
            len(row["completion"]["generated_token_ids"])
            for condition in harness["conditions"].values()
            for row in condition["rows"]
        ),
        record_bindings_sha256=digest(
            [dict(trial_id=row["trial_id"], record_sha256=digest(row)) for row in rows]
        ),
        interpretation="Bound engineering evidence only; no answer-success selection, scientific admission, or evidence of subjective experience.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    write_once(args.out, result)
    print(
        json.dumps(
            {key: result[key] for key in ("complete", "integrity_passed", "passed", "measurement_status")}
        )
    )
