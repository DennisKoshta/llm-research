import copy
import hashlib
from pathlib import Path
import sys

import pytest

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_replay_interventions import direction_assignment_splits, split_provenance


def fixture():
    rows = target.assignments(items=2, kinds=("new", "repeat"))
    ranges = dict(calibration_indices=[0, 1], check_indices=[1, 2])
    manifest = dict(
        diagnostics=rows,
        assignment_sha256=digest(rows),
        development_split=ranges,
        task_module_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
    )
    ledger = [
        dict(
            trial_id=a["trial_id"], representation_split=("extraction" if a["index"] == 0 else "development")
        )
        for a in rows
    ]
    analysis = dict(
        manifest_sha256=digest(manifest),
        development_only=True,
        direction_schema="target_candidate_controls_v1",
        passed=True,
        candidate_eligible=True,
        controls_available=True,
        assignment_split=dict(
            schema_version=1,
            rule="manifest_development_split_v1",
            manifest_sha256=digest(manifest),
            development_split_sha256=digest(ranges),
            assignments=ledger,
            assignment_sha256=digest(ledger),
        ),
    )
    return manifest, analysis


def rebind(manifest, analysis):
    manifest["assignment_sha256"] = digest(manifest["diagnostics"])
    analysis["manifest_sha256"] = digest(manifest)
    binding = analysis["assignment_split"]
    binding["manifest_sha256"] = digest(manifest)
    binding["development_split_sha256"] = digest(manifest["development_split"])
    binding["assignment_sha256"] = digest(binding["assignments"])


def test_range_ledger_binds_all_worlds_without_mutating_frozen_source():
    manifest, analysis = fixture()
    original = copy.deepcopy(manifest)
    result = split_provenance(
        dict(inference_scope="development"), manifest, analysis, manifest["diagnostics"]
    )
    assert result["extraction_binding_verified"]
    assert not result["confirmation_eligible"]
    assert result["direction_assignment_split_sha256"] == digest(analysis["assignment_split"])
    assert all(len(v["world_ids"]) == 2 for v in result["overlaps"].values())
    assert manifest == original


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "reordered",
        "flipped",
        "extra",
        "range_overlap",
        "outside_range",
        "conflicting_explicit",
        "duplicate_id",
        "bad_hash",
        "bad_rule",
        "empty_ledger",
    ],
)
def test_rehashed_invalid_split_ledger_is_rejected(problem):
    manifest, analysis = fixture()
    binding = analysis["assignment_split"]
    if problem == "missing":
        binding["assignments"].pop()
    elif problem == "reordered":
        binding["assignments"].reverse()
    elif problem == "flipped":
        binding["assignments"][0]["representation_split"] = "other"
    elif problem == "extra":
        binding["ignored"] = True
    elif problem == "range_overlap":
        manifest["development_split"]["calibration_indices"] = [0, 2]
    elif problem == "outside_range":
        manifest["diagnostics"][0]["index"] = 2
    elif problem == "conflicting_explicit":
        manifest["diagnostics"][0]["representation_split"] = "other"
    elif problem == "duplicate_id":
        manifest["diagnostics"][1]["trial_id"] = manifest["diagnostics"][0]["trial_id"]
    elif problem == "bad_rule":
        binding["rule"] = "posthoc_split"
    elif problem == "empty_ledger":
        binding["assignments"] = []
    rebind(manifest, analysis)
    if problem == "bad_hash":
        binding["assignment_sha256"] = "stale"
    with pytest.raises(ValueError):
        direction_assignment_splits(manifest, analysis)


@pytest.mark.parametrize("key", ["passed", "candidate_eligible", "controls_available", "assignment_split"])
def test_failed_candidate_or_missing_ledger_cannot_enter_natural_science(key):
    manifest, analysis = fixture()
    analysis[key] = None if key == "assignment_split" else False
    with pytest.raises(ValueError, match="Target"):
        split_provenance(dict(inference_scope="development"), manifest, analysis, manifest["diagnostics"])
