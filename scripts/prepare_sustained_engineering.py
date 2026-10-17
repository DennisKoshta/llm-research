"""Prepare one fixed engineering stage; never launch or admit scientific steering."""

import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

from curiosity_study import sustained_engineering as engineering
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.concept_capture import MODEL
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_execution import freeze  # noqa: E402
from collection_coverage import check_collection_coverage  # noqa: E402
from prepare_voluntary_resolution import candidate_binding  # noqa: E402

SOURCE_MANIFEST_SHA256 = "b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6"
CANDIDATE_DOCUMENTS = dict(
    manifest="runs/qwen32-curiosity-satisfaction-candidates-v1/manifest.json",
    audit="reports/qwen32-curiosity-satisfaction-candidates-v1-audit.json",
    analysis="analyses/curiosity-satisfaction-candidates-v1/analysis.json",
    directions="analyses/curiosity-satisfaction-candidates-v1/directions.npz",
    check="reports/curiosity-satisfaction-candidates-v1-independent-check.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(root, path):
    return str(Path(path).resolve().relative_to(root))


def fixed_source(root, source, audit_path):
    """Use original inputs from the first actual cohort, without outcome selection."""
    manifest_path, freeze_path = source / "manifest.json", source / "freeze.json"
    manifest, frozen, audit = read(manifest_path), read(freeze_path), read(audit_path)
    archive_path = source / "source.tar.gz"
    require(
        digest(manifest) == SOURCE_MANIFEST_SHA256
        and manifest.get("assay") == "voluntary_resolution"
        and manifest.get("task_revision") == "voluntary-resolution-v2"
        and manifest.get("model") == MODEL
        and manifest.get("execution") == engineering.EXECUTION,
        "Require the fixed v2 source manifest and pinned batch32 model",
    )
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha(archive_path)
        and audit.get("complete") is True
        and audit.get("integrity_passed") is True
        and audit.get("assigned") == 288
        and audit.get("manifest_sha256") == digest(manifest)
        and audit.get("source_sha256") == manifest["source_sha256"]
        and audit.get("source_freeze", {}).get("archive_sha256") == frozen["archive_sha256"],
        "Require complete source integrity and unchanged archived provenance, regardless of readiness",
    )
    check_collection_coverage(source, expected_manifest_sha256=digest(manifest))
    assignments = (manifest.get("assignments") or []) + manifest["diagnostics"]
    paths = list((source / "cohorts").glob("*/0000000.json"))
    require(len(paths) == 1, "Require the single first physical source cohort")
    cohort_path = paths[0]
    cohort = read(cohort_path)
    records = cohort["records"]
    audited = audit["outcomes"]
    audited_records = {row["trial_id"]: row.get("record_sha256") for row in audited}
    require(
        cohort["manifest_hash"] == digest(manifest)
        and cohort["sha256"] == digest(records)
        and len(records) == 32
        and [row["assignment"] for row in records] == assignments[:32]
        and [row["trial_id"] for row in records] == [a["trial_id"] for a in assignments[:32]],
        "First actual cohort membership differs from frozen source order",
    )
    require(
        len(audited) == len(audited_records) == len(assignments)
        and set(audited_records) == {a["trial_id"] for a in assignments}
        and all(audited_records[row["trial_id"]] == digest(row) for row in records),
        "First source records differ from the complete passing integrity audit",
    )
    requests, record_bindings = [], []
    for row, record in enumerate(records):
        assignment, inference = record["assignment"], record["inference"]
        positions = dict(
            zip(
                record["activation_artifact"]["positions"],
                record["activation_artifact"]["token_positions"],
                strict=True,
            )
        )
        tokens = inference["unpadded_input_ids"]
        require(
            inference.get("enable_thinking") is True
            and inference["sampling"]["seed"] == assignment["sampling_seed"]
            and inference["batch"]["cohort_ids"] == [a["trial_id"] for a in assignments[:32]]
            and inference["batch"]["row"] == row
            and inference["batch"]["size"] == 32
            and positions["prefill"] == len(tokens) - 1,
            "Source original-prefix, capture or row layout binding differs",
        )
        requests.append(
            dict(
                source_trial_id=record["trial_id"],
                token_prefix=tokens,
                token_prefix_sha256=digest(tokens),
                prefix_positions=dict(record=positions["record"], final=positions["prefill"]),
                labels=list("ABCD"),
                seed=assignment["sampling_seed"],
            )
        )
        record_bindings.append(
            dict(
                trial_id=record["trial_id"],
                record_sha256=digest(record),
                assignment_sha256=digest(assignment),
            )
        )
    binding = dict(
        run=relative(root, source),
        manifest_path=relative(root, manifest_path),
        manifest_sha256=digest(manifest),
        audit_path=relative(root, audit_path),
        audit_sha256=digest(audit),
        audit_byte_sha256=sha(audit_path),
        freeze_path=relative(root, freeze_path),
        freeze_sha256=digest(frozen),
        archive_path=relative(root, archive_path),
        archive_byte_sha256=sha(archive_path),
        first_cohort_path=relative(root, cohort_path),
        first_cohort_sha256=digest(cohort),
        first_cohort_byte_sha256=sha(cohort_path),
        records=record_bindings,
        readiness_passed=audit.get("readiness", {}).get("passed"),
        selection_rule="first32 frozen source assignments and their actual first cohort; no answer, validity, correctness or length filter",
    )
    return manifest, audit, frozen, requests, binding


def locked_direction(source, manifest, audit, frozen):
    with tarfile.open(source / "source.tar.gz", "r:gz") as archive:
        payloads = {key: archive.extractfile(name).read() for key, name in CANDIDATE_DOCUMENTS.items()}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()}
    require(
        all(frozen["documents"].get(name) == hashes[key] for key, name in CANDIDATE_DOCUMENTS.items()),
        "Candidate documents differ from the source freeze",
    )
    values = {key: json.loads(value) for key, value in payloads.items() if key != "directions"}
    binding = candidate_binding(
        values["manifest"],
        values["audit"],
        values["analysis"],
        values["check"],
        hashes["analysis"],
        hashes["directions"],
    )
    require(
        binding == manifest["candidate_source"] == audit["candidate_source"],
        "The candidate must be the unchanged independently checked source candidate",
    )
    with np.load(BytesIO(payloads["directions"]), allow_pickle=False) as arrays:
        require(
            list(arrays["layers"]) == [15, 31, 47, 57] and list(arrays["positions"]) == ["record", "final"],
            "Frozen candidate coordinate axes changed",
        )
        index = list(arrays["sources"]).index("curiosity_vs_indifference")
        unit = arrays["unit"][index, 0, 0].copy()
        norm = float(arrays["contrast_norm"][index, 0, 0])
    selected = next(
        row for row in values["analysis"]["candidates"] if row["contrast"] == "curiosity_vs_indifference"
    )
    require(norm == selected["a_raw_contrast_norm"], "Candidate raw norm differs from locked analysis")
    return dict(
        source="curiosity_vs_indifference",
        corpus="A",
        layer=15,
        position="record",
        unit=unit.tolist(),
        unit_dtype=str(unit.dtype),
        unit_binding=engineering.array_binding(unit),
        norm=norm,
        dose=norm,
        multiplier=1.0,
        directions_document=CANDIDATE_DOCUMENTS["directions"],
        directions_byte_sha256=hashes["directions"],
        candidate_source=binding,
    )


def bind_short(root, manifest, short_run, short_audit):
    require(
        short_run is not None and short_audit is not None,
        "Full cap requires the separate short run and audit",
    )
    from audit_sustained_engineering import audit as audit_engineering

    short_run, short_audit = Path(short_run).resolve(), Path(short_audit).resolve()
    previous, receipt = read(short_run / "manifest.json"), read(short_audit)
    require(
        previous["cap"] == 128
        and previous["compatibility_sha256"] == manifest["compatibility_sha256"]
        and engineering.compatibility_binding(previous) == engineering.compatibility_binding(manifest),
        "Short stage does not bind the exact source, dose, layout, model and engine",
    )
    require(receipt == audit_engineering(short_run), "Short audit does not reproduce from immutable evidence")
    require(
        receipt.get("complete") is True
        and receipt.get("passed") is True
        and receipt.get("cap") == 128
        and receipt.get("scientific_admission") is False,
        "Short engineering did not pass; a full-cap stage is unavailable",
    )
    return dict(
        run=relative(root, short_run),
        manifest_path=relative(root, short_run / "manifest.json"),
        manifest_sha256=digest(previous),
        audit_path=relative(root, short_audit),
        audit_sha256=digest(receipt),
        audit_byte_sha256=sha(short_audit),
        audit=receipt,
    )


def build_manifest(root, source, source_audit, plan, cap, *, short_run=None, short_audit=None):
    root, source, source_audit, plan = map(
        lambda value: Path(value).resolve(), (root, source, source_audit, plan)
    )
    require(type(cap) is int and cap in (128, 4096), "Only explicit128 or4096 engineering stages are allowed")
    require(
        plan.is_file() and bool(plan.read_text().strip()), "Require an explicit prospective engineering plan"
    )
    source_manifest, source_audit_value, frozen, requests, source_binding = fixed_source(
        root, source, source_audit
    )
    direction = locked_direction(source, source_manifest, source_audit_value, frozen)
    assignments = engineering.assignment_plan(requests, cap)
    manifest = dict(
        schema_version=1,
        assay="sustained_engineering",
        phase="engineering",
        scientific_admission=False,
        model=deepcopy(MODEL),
        execution=deepcopy(engineering.EXECUTION),
        execution_gpu="H200",
        arithmetic=deepcopy(engineering.ARITHMETIC),
        schedule=engineering.SCHEDULE,
        layer=15,
        cap=cap,
        source_sha256=source_hash(),
        engine_sha256=engineering.engine_hashes(),
        requests=requests,
        source_binding=source_binding,
        direction=direction,
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        short_stage=None,
        invalid_output_policy="No answer scoring; terminal engineering failures retained without retry",
        interpretation="Engineering only; cap128 pass can admit only an exactly bound cap4096 engineering stage, never scientific steering.",
    )
    manifest["compatibility_sha256"] = digest(engineering.compatibility_binding(manifest))
    documents = [relative(root, plan)] + [
        source_binding[key]
        for key in ("manifest_path", "audit_path", "freeze_path", "archive_path", "first_cohort_path")
    ]
    if cap == 4096:
        manifest["short_stage"] = bind_short(root, manifest, short_run, short_audit)
        documents += [manifest["short_stage"][key] for key in ("manifest_path", "audit_path")]
    else:
        require(
            short_run is None and short_audit is None,
            "A short stage must not supply a prior-stage certificate",
        )
    manifest["plan_path"] = relative(root, plan)
    manifest["document_bindings"] = [
        dict(path=name, byte_sha256=sha(root / name)) for name in sorted(set(documents))
    ]
    engineering.validate_manifest(manifest)
    return manifest


def prepare(out, source, source_audit, plan, cap, *, root=ROOT, short_run=None, short_audit=None):
    out, root = Path(out).resolve(), Path(root).resolve()
    require(not out.exists(), "Engineering collections are immutable; choose a new output directory")
    manifest = build_manifest(
        root, source, source_audit, plan, cap, short_run=short_run, short_audit=short_audit
    )
    write_once(out / "manifest.json", manifest)
    frozen = freeze(root, out, [row["path"] for row in manifest["document_bindings"]])
    return dict(
        run=str(out),
        assigned=128,
        cap=cap,
        manifest_sha256=digest(manifest),
        source_sha256=frozen["source_sha256"],
        gpu_launched=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "source", "source-audit", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cap", type=int, choices=(128, 4096), required=True)
    parser.add_argument("--short-run", type=Path)
    parser.add_argument("--short-audit", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.out,
            args.source,
            args.source_audit,
            args.plan,
            args.cap,
            short_run=args.short_run,
            short_audit=args.short_audit,
        )
    )
