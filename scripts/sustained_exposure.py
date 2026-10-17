"""Recover engineering exposure from frozen source inputs, never generated answers.

A durable terminal cohort conservatively admits all 32 original source requests,
including failed or partial engineering runs. Preparation alone is not exposure.
The source integrity ledger is checked; behavioral readiness is never an exclusion.
"""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

from curiosity_study.artifacts import digest, read

CONDITIONS = ("baseline", "zero_sham", "one_row_nonzero", "clean_repeat")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def safe_relative(value):
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe engineering exposure path")
    return path


def member_bytes(archive, name):
    safe_relative(name)
    matches = [member for member in archive.getmembers() if member.name == str(name)]
    require(len(matches) == 1 and matches[0].isfile(), "Missing or ambiguous frozen exposure member")
    return archive.extractfile(matches[0]).read()


def python_sources(archive):
    prefix = "src/curiosity_study/"
    names = [
        member.name
        for member in archive.getmembers()
        if member.name.startswith(prefix) and member.name.endswith(".py")
    ]
    require(bool(names) and len(names) == len(set(names)), "Missing or duplicate frozen task sources")
    return {name[len(prefix) :]: member_bytes(archive, name).decode() for name in names}


def terminal_evidence(root, run, manifest):
    """Check disposition bindings without interpreting status, answers or checks."""
    directory = root / safe_relative(run)
    require(not any((directory / "attempts").rglob("*.json")), "Engineering exposure requires atomic cohorts")
    assignments = manifest["diagnostics"]
    paths = sorted((directory / "cohorts").rglob("*.json"))
    require(bool(paths), "Prepared engineering has no durable terminal exposure evidence")
    offsets, attempts, evidence = set(), set(), []
    for path in paths:
        relative = path.relative_to(directory / "cohorts")
        require(len(relative.parts) == 2 and path.stem.isdigit(), "Unexpected engineering cohort path")
        offset = int(path.stem)
        require(
            offset in (0, 32, 64, 96) and offset not in offsets, "Duplicate or unexpected engineering cohort"
        )
        payload = path.read_bytes()
        envelope = json.loads(payload)
        records = envelope["records"]
        require(
            envelope["manifest_hash"] == digest(manifest)
            and envelope["sha256"] == digest(records)
            and len(records) == 32
            and [record["assignment"] for record in records] == assignments[offset : offset + 32]
            and all(
                record["terminal"] is True and record["trial_id"] == record["assignment"]["trial_id"]
                for record in records
            ),
            "Engineering terminal exposure binding changed",
        )
        offsets.add(offset)
        attempts.add(relative.parts[0])
        evidence.append(
            dict(
                path=str(path.relative_to(root)),
                byte_sha256=sha_bytes(payload),
                envelope_sha256=digest(envelope),
                assigned_rows=32,
            )
        )
    require(len(attempts) == 1, "Engineering exposure cannot combine attempts")
    return evidence


def source_inputs(root, run, manifest):
    """Return original voluntary assignments and their hash-matched task oracle."""
    root = Path(root).resolve()
    directory = root / safe_relative(run)
    require(
        manifest.get("assay") == "sustained_engineering"
        and manifest.get("phase") == "engineering"
        and manifest.get("scientific_admission") is False
        and type(manifest.get("cap")) is int
        and manifest["cap"] in (128, 4096)
        and manifest.get("assignments") == [],
        "Unknown sustained-engineering exposure schema",
    )
    requests = manifest["requests"]
    require(
        len(requests) == 32 and len({r["source_trial_id"] for r in requests}) == 32,
        "Require exactly32 original requests",
    )
    expected = [
        dict(
            trial_id=f"sustained-{manifest['cap']}-{condition}-{index:02d}",
            kind="engineering",
            condition=condition,
            source_row=index,
            source_trial_id=request["source_trial_id"],
            request_sha256=digest(request),
            cap=manifest["cap"],
        )
        for condition in CONDITIONS
        for index, request in enumerate(requests)
    ]
    require(
        manifest["diagnostics"] == expected and manifest["assignment_sha256"] == digest(expected),
        "Engineering assignments do not bind the four fixed source cohorts",
    )
    evidence = terminal_evidence(root, run, manifest)
    frozen = read(directory / "freeze.json")
    archive_path = directory / "source.tar.gz"
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha_bytes(archive_path.read_bytes()),
        "Engineering exposure freeze changed",
    )
    source = manifest["source_binding"]
    required_documents = [
        source[key]
        for key in (
            "manifest_path",
            "audit_path",
            "freeze_path",
            "archive_path",
            "first_cohort_path",
        )
    ]
    document_bindings = {row["path"]: row["byte_sha256"] for row in manifest["document_bindings"]}
    require(
        len(document_bindings) == len(manifest["document_bindings"]), "Duplicate exposure document binding"
    )
    documents = {}
    with tarfile.open(archive_path) as archive:
        require(
            digest(python_sources(archive)) == manifest["source_sha256"],
            "Engineering archived source digest differs",
        )
        for name in required_documents:
            payload = member_bytes(archive, name)
            require(
                sha_bytes(payload) == document_bindings.get(name) == frozen["documents"].get(name),
                "Frozen source-input document binding differs",
            )
            documents[name] = payload
    source_manifest = json.loads(documents[source["manifest_path"]])
    source_audit = json.loads(documents[source["audit_path"]])
    source_freeze = json.loads(documents[source["freeze_path"]])
    cohort = json.loads(documents[source["first_cohort_path"]])
    source_archive = documents[source["archive_path"]]
    require(
        source_manifest.get("assay") == "voluntary_resolution"
        and source_manifest.get("task_revision") == "voluntary-resolution-v2"
        and digest(source_manifest) == source["manifest_sha256"] == source_freeze["manifest_sha256"]
        and source_manifest["source_sha256"] == source_freeze["source_sha256"]
        and digest(source_freeze) == source["freeze_sha256"]
        and sha_bytes(source_archive) == source["archive_byte_sha256"] == source_freeze["archive_sha256"],
        "Original engineering source provenance differs",
    )
    # Match the live source inventory too; archived aliases cannot invent a source.
    require(
        (root / safe_relative(source["manifest_path"])).read_bytes() == documents[source["manifest_path"]]
        and sha_bytes((root / safe_relative(source["archive_path"])).read_bytes())
        == source["archive_byte_sha256"],
        "Engineering source differs from the historical live source binding",
    )
    require(
        digest(source_audit) == source["audit_sha256"]
        and sha_bytes(documents[source["audit_path"]]) == source["audit_byte_sha256"]
        and source_audit.get("complete") is True
        and source_audit.get("integrity_passed") is True
        and source_audit.get("assigned") == 288
        and source_audit["manifest_sha256"] == digest(source_manifest)
        and source_audit["source_sha256"] == source_manifest["source_sha256"]
        and source_audit["source_freeze"]["archive_sha256"] == source_freeze["archive_sha256"],
        "Original source integrity audit binding differs",
    )
    assignments = (source_manifest.get("assignments") or []) + (source_manifest.get("diagnostics") or [])
    ledger = source_audit["outcomes"]
    require(
        len(assignments) == len(ledger) == len({a["trial_id"] for a in assignments}) == 288
        and len({row["trial_id"] for row in ledger}) == 288
        and {row["trial_id"] for row in ledger} == {a["trial_id"] for a in assignments},
        "Source audit must bind all288 original assignments",
    )
    ledger_hashes = {row["trial_id"]: row["record_sha256"] for row in ledger}
    records = cohort["records"]
    require(
        digest(cohort) == source["first_cohort_sha256"]
        and sha_bytes(documents[source["first_cohort_path"]]) == source["first_cohort_byte_sha256"]
        and cohort["manifest_hash"] == digest(source_manifest)
        and cohort["sha256"] == digest(records)
        and len(records) == 32
        and [record["assignment"] for record in records] == assignments[:32],
        "Original first32 source cohort binding differs",
    )
    record_bindings = []
    for index, (record, assignment, request) in enumerate(
        zip(records, assignments[:32], requests, strict=True)
    ):
        inference, artifact = record["inference"], record["activation_artifact"]
        tokens = inference["unpadded_input_ids"]
        positions = dict(zip(artifact["positions"], artifact["token_positions"], strict=True))
        require(
            record["trial_id"] == assignment["trial_id"]
            and digest(record) == ledger_hashes[record["trial_id"]]
            and inference.get("enable_thinking") is True
            and inference["sampling"]["seed"] == assignment["sampling_seed"]
            and inference["batch"]["row"] == index
            and inference["batch"]["size"] == 32
            and inference["batch"]["cohort_ids"] == [a["trial_id"] for a in assignments[:32]]
            and positions["prefill"] == len(tokens) - 1
            and request
            == dict(
                source_trial_id=record["trial_id"],
                token_prefix=tokens,
                token_prefix_sha256=digest(tokens),
                prefix_positions=dict(record=positions["record"], final=positions["prefill"]),
                labels=list("ABCD"),
                seed=assignment["sampling_seed"],
            ),
            "Original source record differs from the audit ledger or exact requested input",
        )
        record_bindings.append(
            dict(
                trial_id=record["trial_id"],
                record_sha256=digest(record),
                assignment_sha256=digest(assignment),
            )
        )
    require(record_bindings == source["records"], "Source record/assignment bindings changed")
    with tarfile.open(fileobj=BytesIO(source_archive)) as archive:
        sources = python_sources(archive)
    require(
        digest(sources) == source_manifest["source_sha256"], "Original frozen oracle source digest differs"
    )
    return dict(
        assignments=assignments[:32],
        assay=source_manifest["assay"],
        source_sha256=source_manifest["source_sha256"],
        sources=dict(
            archive=source["archive_path"],
            archive_byte_sha256=source["archive_byte_sha256"],
            member_prefix="",
            sources=sources,
        ),
        provenance=dict(
            adapter="sustained-engineering-source-inputs-v1",
            adapter_byte_sha256=sha_bytes(Path(__file__).read_bytes()),
            engineering_manifest_sha256=digest(manifest),
            engineering_freeze_sha256=digest(frozen),
            engineering_archive_path=str(archive_path.relative_to(root)),
            engineering_archive_byte_sha256=frozen["archive_sha256"],
            source_binding_sha256=digest(source),
            source_manifest_sha256=digest(source_manifest),
            source_audit_sha256=digest(source_audit),
            source_assignment_sha256=digest(assignments[:32]),
            source_record_bindings=record_bindings,
            terminal_evidence=evidence,
            inclusion_rule="All32 original inputs after any durable bound terminal cohort, including failed or partial engineering; no pass, correctness, answer or length filter",
        ),
    )
