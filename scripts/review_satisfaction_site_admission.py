"""Verify a prepared immutable site phase locally before its sole paid admission."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_site_resources as resources
from curiosity_study import satisfaction_site_execution as execution
from curiosity_study import satisfaction_site_setup as setup
from curiosity_study.artifacts import digest, read, write_once

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(run, *, root=ROOT):
    from run_bounded_partitions import validate_stage_budget
    from budget import _bound_job_limits
    from launch_job import select_pilot
    import modal_app

    root, run = Path(root).resolve(), Path(run).resolve()
    m, frozen = read(run / "manifest.json"), read(run / "freeze.json")
    phase = resources.phase_of(m)
    require(phase in resources.POLICIES and run.parent == root / "runs", "Unknown site phase or run location")
    require(digest(m) == frozen["manifest_sha256"] and m["source_sha256"] == frozen["source_sha256"]
            and sha(run / "source.tar.gz") == frozen["archive_sha256"], "Prepared source freeze changed")
    execution_root = Path(frozen["execution_root"])
    require(execution_root == root / "executions" / run.name, "Unexpected execution directory")
    source = {}
    verified = 0
    with tarfile.open(run / "source.tar.gz") as archive:
        members = archive.getmembers()
        files = [member for member in members if member.isfile()]
        require(len(files) == len(members) and len({member.name for member in files}) == len(files),
                "Archive must contain unique regular files only")
        for member in files:
            name = Path(member.name)
            require(not name.is_absolute() and ".." not in name.parts, "Unsafe archived path")
            payload = archive.extractfile(member).read()
            require(payload == (root / name).read_bytes() == (execution_root / name).read_bytes(),
                    "Current code/evidence or isolated execution copy differs from freeze")
            if member.name.startswith("src/curiosity_study/") and member.name.endswith(".py"):
                source[member.name.removeprefix("src/curiosity_study/")] = payload.decode()
            verified += 1
    require(digest(source) == frozen["source_sha256"], "Archived package inventory changed")
    require(all(sha(root / name) == expected for name, expected in frozen["documents"].items()),
            "Frozen document binding changed")
    if phase == "setup":
        from audit_satisfaction_site_setup import source_provenance

        setup.validate_manifest(m)
        source_provenance(run, m)
        manifests = [m]
        expected_cloud = modal_app.site_setup_pilot
    else:
        from audit_satisfaction_causal_sites import parent_children, provenance
        from prepare_satisfaction_screen import tokenizer_model

        manifests = [child for _, child in parent_children(run, m)]
        tokenizer = tokenizer_model()
        provenance(run, m, tokenizer.tokenizer)
        require(execution.bind_preflight(m["site_plan"], tokenizer) == m["generation_preflight"]
                and execution.bind_receipt_preflight(m["site_plan"], tokenizer) == m["receipt_preflight"],
                "Generation or possible-receipt tokenization changed")
        expected_cloud = modal_app.site_calibration_pilot if phase == "calibration" else modal_app.site_validation_pilot
    policy = resources.POLICIES[phase]
    require(_bound_job_limits(manifests) == (policy["shared_deadline_seconds"], policy["stage_ceiling_usd"]),
            "Actual budget dispatch differs from the new stage policy")
    validate_stage_budget(m, len(manifests), policy["shared_deadline_seconds"])
    require(all(select_pilot(child, run / "reservation.json") is expected_cloud for child in manifests),
            "Actual cloud entrypoint does not use the finite site policy")
    require(m["campaign_id"] == study.REVISION and not (run / "reservation.json").exists(),
            "Phase is not an unadmitted member of this campaign")
    return dict(schema_version=1, passed=True, phase=phase, manifest_sha256=digest(m),
                source_sha256=m["source_sha256"], archive_byte_sha256=frozen["archive_sha256"],
                verified_archive_files=verified, verified_documents=len(frozen["documents"]),
                assigned=len(m["diagnostics"]), workers=len(manifests), budget=policy,
                gpu_launched=False, scientific_outputs_inspected=False,
                scope="Exact current/archive/execution bytes, fixed complete assignments, setup or scientific provenance replay, token preflights and actual budget/cloud dispatch. No model inference or new-result certification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Prelaunch review is immutable")
    result = review(args.run)
    write_once(args.out, result)
    print(json.dumps(result))
