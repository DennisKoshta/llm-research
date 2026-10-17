"""Reporting-only leading-dot adapter-path correction for one frozen campaign.

The archived auditor, model execution, numerical checks and analysis rules remain
unchanged. This wrapper canonicalizes only expected adapter inventory keys using
the same PurePosixPath operation as the frozen loader, with collision rejection.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

from curiosity_study.artifacts import digest, read, write_once

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/qwen25-pain-profile-control-v1"
AMENDMENT = ROOT / "reports/pain-profile-canonical-paths-amendment.json"
MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
SOURCE_SHA = "63bc3e330b25f5805724bfa5b8bf2f75d6fe37d048963524e1d28138e4bd4ff9"
ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
AUDITOR_SHA = "ebacd48847f51754f9becd0afe20ca21103c2963e8ee97765818723a41432bc3"
ANALYZER_SHA = "c0158b2846e65108a2e245c805ac4634a83cb515d6e69fe0b4c1ca8df87882da"
CPU_SUMMARY_SHA = "32034eaa3b28214d62377057a58fcefe310db9ac8c6de0687357b2169af7b0f6"
AUDITOR = "scripts/audit_pain_profile_control.py"
ANALYZER = "scripts/analyze_pain_profile_control.py"
ADAPTER_RECEIPT = "references/pain-axis-assets/asset-receipt.json"
CORRECTION = "adapter_inventory_leading_dot_path_canonicalization_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_adapter_files(raw):
    require(isinstance(raw, dict) and raw, "Missing raw adapter inventory")
    canonical = {}
    for name, checksum in raw.items():
        require(isinstance(name, str) and bool(name), "Invalid adapter member path")
        path = PurePosixPath(name)
        clean = str(path)
        require(not path.is_absolute() and ".." not in path.parts and clean not in ("", ".")
                and name in (clean, "./" + clean), "Only a single leading ./ may be normalized")
        require(clean not in canonical, "Adapter canonical path collision")
        require(isinstance(checksum, str) and re.fullmatch("[a-f0-9]{64}", checksum), "Invalid adapter content hash")
        canonical[clean] = checksum
    return canonical


def frozen_inputs(run):
    """Read only original archive and CPU metadata; never inspect setup outcomes."""
    run = Path(run).resolve()
    require(run == RUN.resolve(), "Correction is limited to the original fixed campaign")
    manifest, frozen = read(run / "manifest.json"), read(run / "freeze.json")
    require(digest(manifest) == frozen["manifest_sha256"] == MANIFEST_SHA
            and manifest["source_sha256"] == frozen["source_sha256"] == SOURCE_SHA
            and sha(run / "source.tar.gz") == frozen["archive_sha256"] == ARCHIVE_SHA,
            "Original frozen campaign identity differs")
    execution = Path(frozen["execution_root"]).resolve()
    require(execution == ROOT / "executions/qwen25-pain-profile-control-v1", "Unexpected frozen execution path")
    inventory, selected, sources = {}, {}, {}
    with tarfile.open(run / "source.tar.gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and str(path) == member.name and member.name not in inventory, "Unsafe or duplicate frozen archive member")
            data = archive.extractfile(member).read()
            checksum = hashlib.sha256(data).hexdigest()
            inventory[member.name] = checksum
            if member.name.startswith("src/curiosity_study/") and member.name.endswith(".py"):
                sources[member.name.removeprefix("src/curiosity_study/")] = data.decode()
            if member.name in (AUDITOR, ANALYZER, ADAPTER_RECEIPT):
                selected[member.name] = data
    require(digest(sources) == SOURCE_SHA and inventory.get(AUDITOR) == AUDITOR_SHA
            and inventory.get(ANALYZER) == ANALYZER_SHA
            and inventory.get(ADAPTER_RECEIPT) == manifest["documents"][ADAPTER_RECEIPT],
            "Original source/auditor/analyzer/adapter metadata bytes differ")
    for name in inventory:
        if name.startswith(("src/", "scripts/", "infra/")):
            path = execution / name
            require(path.is_file() and not path.is_symlink() and sha(path) == inventory[name],
                    "Frozen execution code changed")
    receipt = json.loads(selected[ADAPTER_RECEIPT])
    members = [row for row in receipt["members"] if row["type"] == "file"]
    raw = {row["name"]: row["sha256"] for row in members}
    require(len(members) == len(raw) == 7, "Original seven-file adapter inventory changed")
    canonical = canonical_adapter_files(raw)
    cpu_path = run / "phases/prefetch/summary.json"
    cpu = read(cpu_path)
    require(sha(cpu_path) == CPU_SUMMARY_SHA and cpu["manifest_sha256"] == MANIFEST_SHA
            and cpu["source_sha256"] == SOURCE_SHA and cpu["complete"] is cpu["passed"] is True
            and cpu["model_inference_performed"] is False
            and cpu["assets"]["adapter_files"] == canonical,
            "CPU receipt does not prove the exact seven path-only equivalences")
    return dict(run=run, execution=execution, manifest=manifest, freeze=frozen, inventory=inventory,
                raw=raw, canonical=canonical, cpu=cpu, cpu_path=cpu_path)


def amendment_payload(run=RUN):
    inputs = frozen_inputs(run)
    rows = [dict(raw_path=name, canonical_path=str(PurePosixPath(name)),
                 original_byte_sha256=checksum,
                 cpu_byte_sha256=inputs["canonical"][str(PurePosixPath(name))], identical_bytes=True)
            for name, checksum in sorted(inputs["raw"].items())]
    return dict(schema_version=1, correction=CORRECTION, scope="Reporting-only; remove a single leading ./ from expected adapter inventory keys",
        manifest_sha256=MANIFEST_SHA, source_sha256=SOURCE_SHA, source_archive_sha256=ARCHIVE_SHA,
        freeze_byte_sha256=sha(inputs["run"] / "freeze.json"),
        original_auditor=dict(path=AUDITOR, byte_sha256=AUDITOR_SHA),
        original_analyzer=dict(path=ANALYZER, byte_sha256=ANALYZER_SHA),
        adapter_receipt_byte_sha256=inputs["inventory"][ADAPTER_RECEIPT],
        cpu_summary=dict(path="phases/prefetch/summary.json", byte_sha256=CPU_SUMMARY_SHA,
                         content_sha256=digest(inputs["cpu"])),
        member_count=7, members=rows, members_sha256=digest(rows),
        raw_inventory_sha256=digest(inputs["raw"]), canonical_inventory_sha256=digest(inputs["canonical"]),
        reporting_wrapper=dict(path=str(Path(__file__).relative_to(ROOT)), byte_sha256=sha(__file__)),
        timing="Recorded after setup GPU launch, prompted only by CPU metadata and before this correction's author inspected any setup outcomes; no claim of preceding measurements.",
        execution_or_scoring_changed=False, numerical_checks_changed=False, thresholds_changed=False,
        paid_retry_authorized=False, original_frozen_files_preserved=True)


def validate_amendment(run, path):
    path = Path(path).resolve()
    expected = amendment_payload(run)
    require(read(path) == expected, "Reporting amendment differs from exact frozen inputs or wrapper bytes")
    return expected, dict(correction=CORRECTION, amendment_path=str(path),
        amendment_byte_sha256=sha(path), amendment_sha256=digest(expected),
        reporting_wrapper_byte_sha256=sha(__file__), original_auditor_byte_sha256=AUDITOR_SHA,
        original_analyzer_byte_sha256=ANALYZER_SHA, raw_inventory_sha256=expected["raw_inventory_sha256"],
        canonical_inventory_sha256=expected["canonical_inventory_sha256"], member_count=7,
        execution_or_scoring_changed=False, numerical_checks_changed=False, thresholds_changed=False)


def attach_correction(original, raw_inventory, correction_metadata):
    """Modify a private loaded module only; recursive setup uses the same receipt."""
    canonical = canonical_adapter_files(raw_inventory)
    original_source, original_audit = original.source_provenance, original.audit

    def corrected_source(run, manifest):
        proof = original_source(run, manifest)
        require(proof["adapter_files"] == raw_inventory, "Unanticipated adapter contents or raw names")
        return dict(proof, adapter_files=dict(canonical))

    def corrected_audit(run, phase):
        result = original_audit(run, phase)
        require("reporting_correction" not in result, "Original auditor unexpectedly includes a correction")
        return dict(result, reporting_correction=dict(correction_metadata))

    original.source_provenance = corrected_source
    original.audit = corrected_audit
    return corrected_audit


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def corrected_runtime(run=RUN, amendment=AMENDMENT):
    expected, metadata = validate_amendment(run, amendment)
    inputs = frozen_inputs(run)
    names = ("freeze_execution", "prepare_pain_profile_control", "analyze_pain_profile_control", "_original_pain_profile_auditor")
    saved = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    old_no_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        execution = inputs["execution"]
        _load(names[0], execution / "infra/freeze_execution.py")
        _load(names[1], execution / "scripts/prepare_pain_profile_control.py")
        analysis = _load(names[2], execution / ANALYZER)
        original = _load(names[3], execution / AUDITOR)
        # Imported numerical helpers may already be resident, but their exact
        # bytes must be those archived before the paid campaign.
        for name, module in tuple(sys.modules.items()):
            if name.startswith("curiosity_study.pain_profile_") or name == "curiosity_study.artifacts":
                relative = "src/" + name.replace(".", "/") + ".py"
                require(relative in inputs["inventory"] and sha(module.__file__) == inputs["inventory"][relative],
                        "Loaded numerical helper differs from original archive")
        require(expected["members_sha256"] == digest(expected["members"]), "Member ledger changed")
        corrected = attach_correction(original, inputs["raw"], metadata)
        yield corrected, analysis, metadata
    finally:
        sys.dont_write_bytecode = old_no_bytecode
        sys.path[:] = old_path
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def audit(run=RUN, phase="setup", amendment=AMENDMENT):
    with corrected_runtime(run, amendment) as (corrected, _, _):
        return corrected(Path(run).resolve(), phase)


def analyze_run(run=RUN, audit_path=None, amendment=AMENDMENT):
    run = Path(run).resolve()
    path = run / "phases/benchmark/audit.json" if audit_path is None else Path(audit_path).resolve()
    receipt = read(path)
    with corrected_runtime(run, amendment) as (corrected, analysis, metadata):
        reproduced = corrected(run, "benchmark")
        require(receipt == reproduced, "Corrected audit does not reproduce from original raw evidence")
        manifest = read(run / "manifest.json")
        require(receipt["phase"] == "benchmark" and receipt["manifest_sha256"] == digest(manifest)
                and receipt["assigned_rows"] == len(receipt["outcomes"]) == 2020, "Wrong benchmark denominator")
        fields = (*analysis.METADATA, "base_request_id", "gen_seed", "status", "category", "valid", "cap_hit", "raw", "record_sha256")
        rows = [{key: row[key] for key in fields if key in row} for row in receipt["outcomes"]]
        result = analysis.analyze_tables(manifest["plan"], rows, dict(complete=receipt["complete"], passed=receipt["passed"]))
        result.update(manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
            source_archive_sha256=receipt["archive_byte_sha256"], freeze_byte_sha256=receipt["freeze_byte_sha256"],
            audit_sha256=digest(receipt), audit_byte_sha256=sha(path), auditor_byte_sha256=AUDITOR_SHA,
            analyzer_byte_sha256=ANALYZER_SHA, source_audit_recomputed=True, failed_integrity_can_qualify=False,
            reporting_correction=metadata)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--amendment", type=Path, default=AMENDMENT)
    parser.add_argument("--phase", choices=("setup", "benchmark"))
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(args.analyze != (args.phase is not None), "Choose one audit phase or --analyze")
    if args.analyze:
        require(not args.out.exists(), "Analysis directory already exists")
        result = analyze_run(args.run, args.audit, args.amendment)
        args.out.mkdir(parents=True, exist_ok=False)
        write_once(args.out / "analysis.json", result)
        write_once(args.out / "rates.json", result["rates"])
        write_once(args.out / "contrasts.json", result["contrasts"])
    else:
        result = audit(args.run, args.phase, args.amendment)
        write_once(args.out, result)
    print(json.dumps({key: result[key] for key in ("complete", "passed", "integrity_passed", "assigned_rows", "assigned", "observed_rows", "observed") if key in result}))
