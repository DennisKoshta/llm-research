"""Preserve unused admission1 and refreeze its identical science after an audit-schema fix.

This does not select assignments, retokenize prompts, fit directions, change any
scientific criterion, create a reservation or schedule inference.
"""

from copy import deepcopy
from pathlib import Path
import hashlib
import sys
import tarfile

from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.runner import source_hash
from curiosity_study import curiosity_suppression_capture as capture
from curiosity_study import curiosity_suppression_screen as screen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402

ORIGINAL = "runs/qwen32-curiosity-suppression-screen-v1"
DESTINATION = "runs/qwen32-curiosity-suppression-screen-v1-admission2"
FAILED = "reports/curiosity-suppression-screen-v1-prelaunch-review-attempt1.json"
ORIGINAL_MANIFEST_SHA = "964cb17b1a1fdfc0f6a4bede5e087bf39acfb596efda630fb2a3bf5dfdedda29"
ORIGINAL_ARCHIVE_SHA = "720046cd763275d8ca76b197e957d6d3a2313a7bc38de51e63cabb713cf0c138"
FAILED_REVIEW_SHA = "3835f4325b5aeec73194c58927818ff37c78e5d0daabca67a41508dc6d2f9430"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    original, destination = ROOT / ORIGINAL, ROOT / DESTINATION
    manifest, frozen = read(original / "manifest.json"), read(original / "freeze.json")
    failed = read(ROOT / FAILED)
    descriptors = read(original / "children.json")
    require(
        digest(manifest) == frozen["manifest_sha256"] == ORIGINAL_MANIFEST_SHA
        and sha(original / "source.tar.gz") == frozen["archive_sha256"] == ORIGINAL_ARCHIVE_SHA
        and source_hash() == manifest["source_sha256"] == frozen["source_sha256"]
        and sha(ROOT / FAILED) == FAILED_REVIEW_SHA
        and failed["passed"] is False
        and failed["parent_manifest_sha256"] == ORIGINAL_MANIFEST_SHA
        and failed["archive_byte_sha256"] == ORIGINAL_ARCHIVE_SHA
        and failed["source_sha256"] == manifest["source_sha256"]
        and failed["freeze_sha256"] == digest(frozen)
        and failed["gpu_launched"] is False
        and failed["reservation_created"] is False
        and failed["new_study_generation_artifacts"] == 0
        and failed["six_result_fields_equal"] is True
        and failed["source_scientific_failure_preserved"] is True,
        "Require the intact unused original package and failed prelaunch receipt",
    )
    require(
        not any(
            (original / name).exists() for name in ("reservation.json", "outcomes.json", "coordinator.json")
        )
        and len(descriptors) == 3
        and [row["execution_group"] for row in descriptors] == list(screen.GROUPS),
        "Original package must never have been admitted or collected",
    )
    children_paths = []
    original_children = []
    for descriptor in descriptors:
        child_path = Path(descriptor["path"])
        require(child_path.parent == original.parent, "Original child path escaped runs")
        child = read(child_path / "manifest.json")
        expected = deepcopy(manifest)
        expected.update(
            execution_group=descriptor["execution_group"],
            parent_manifest_sha256=digest(manifest),
            diagnostics=screen.group_assignments(manifest["screen_plan"], descriptor["execution_group"]),
        )
        expected["assignment_sha256"] = digest(expected["diagnostics"])
        require(
            digest(child) == descriptor["manifest_sha256"]
            and child == expected
            and child["parent_manifest_sha256"] == digest(manifest)
            and {p.name for p in child_path.iterdir()} == {"manifest.json"},
            "Original child must contain only its immutable manifest and no execution evidence",
        )
        original_children.append(child)
        children_paths.append(
            destination.with_name(destination.name + "-" + child["execution_group"].replace("_", "-"))
        )
    require(
        not any(p.exists() for p in (destination, ROOT / "executions" / destination.name, *children_paths)),
        "Admission2 artifacts are immutable",
    )
    # Source modules and every scientific bound input remain byte-identical.
    with tarfile.open(original / "source.tar.gz", "r:gz") as archive:
        for binding in manifest["bound_receipts"]:
            require(
                sha(ROOT / binding["path"])
                == binding["byte_sha256"]
                == hashlib.sha256(archive.extractfile(binding["path"]).read()).hexdigest(),
                "A scientific bound input changed",
            )
        for script in ("analyze_curiosity_suppression_screen.py", "prepare_curiosity_suppression_screen.py"):
            require(
                sha(ROOT / "scripts" / script)
                == hashlib.sha256(archive.extractfile("scripts/" + script).read()).hexdigest(),
                "Frozen estimator or original preparation changed",
            )
        old_auditor_sha = hashlib.sha256(
            archive.extractfile("scripts/audit_curiosity_suppression_screen.py").read()
        ).hexdigest()
    new_auditor_sha = sha(ROOT / "scripts/audit_curiosity_suppression_screen.py")
    require(new_auditor_sha != old_auditor_sha, "Require the reviewed auditor schema correction")
    amendment = dict(
        revision="prelaunch_schema_fix_1",
        original_parent=ORIGINAL,
        original_manifest_sha256=digest(manifest),
        original_freeze_byte_sha256=sha(original / "freeze.json"),
        original_archive_sha256=ORIGINAL_ARCHIVE_SHA,
        failed_review=FAILED,
        failed_review_byte_sha256=sha(ROOT / FAILED),
        original_auditor_byte_sha256=old_auditor_sha,
        corrected_auditor_byte_sha256=new_auditor_sha,
        scientific_contract_sha256=digest(manifest),
        scientific_assignments_unchanged=True,
        original_local_execution_artifacts_absent=True,
        reason="Compare the independent prior S decision's exact six-field scientific schema with the same fixed projection of the full analysis decision; prose is not an independent numerical field. No criterion, prior failure or outcome is changed.",
    )
    parent = deepcopy(manifest)
    parent["prelaunch_amendment"] = amendment
    require({k: v for k, v in parent.items() if k != "prelaunch_amendment"} == manifest, "Science changed")
    children = []
    for old_child in original_children:
        child = deepcopy(old_child)
        child["prelaunch_amendment"] = deepcopy(amendment)
        child["parent_manifest_sha256"] = digest(parent)
        capture.validate_manifest(child)
        children.append(child)
    extra_documents = (
        ORIGINAL + "/manifest.json",
        ORIGINAL + "/freeze.json",
        ORIGINAL + "/source.tar.gz",
        FAILED,
    )
    documents = [binding["path"] for binding in parent["bound_receipts"]] + list(extra_documents)
    require(len(documents) == len(set(documents)), "Duplicate freeze document path")
    write_once(destination / "manifest.json", parent)
    descriptors = []
    for path, child in zip(children_paths, children, strict=True):
        write_once(path / "manifest.json", child)
        descriptors.append(
            dict(path=str(path), manifest_sha256=digest(child), execution_group=child["execution_group"])
        )
    write_once(destination / "children.json", descriptors)
    result = freeze(ROOT, destination, documents=documents)
    return dict(
        run=str(destination),
        manifest_sha256=digest(parent),
        source_sha256=result["source_sha256"],
        archive_sha256=result["archive_sha256"],
        original_preserved=True,
        assigned=4320,
        scientific_contract_unchanged=True,
        gpu_launched=False,
    )


if __name__ == "__main__":
    print(prepare())
