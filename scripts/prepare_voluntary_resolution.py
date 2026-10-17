"""Freeze the bounded, unsteered optional-choice bridge; never launch compute."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np

from curiosity_study import voluntary_capture as capture
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.concept_capture import CONTRASTS
from curiosity_study.runner import source_hash
from curiosity_study.thinking_information import CAPTURE_PLAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402

TARGETS = ("curiosity_vs_indifference", "satisfaction_vs_neutral")
CAPTURE = "runs/qwen32-curiosity-satisfaction-candidates-v1/manifest.json"
AUDIT = "reports/qwen32-curiosity-satisfaction-candidates-v1-audit.json"
ANALYSIS = "analyses/curiosity-satisfaction-candidates-v1/analysis.json"
DIRECTIONS = "analyses/curiosity-satisfaction-candidates-v1/directions.npz"
CHECK = "reports/curiosity-satisfaction-candidates-v1-independent-check.json"
PLAN = "reports/voluntary-resolution-v2-plan.md"
HISTORY = "reports/voluntary-resolution-v2-history.json"
EXPOSURE = "reports/voluntary-resolution-v2-freshness.json"
PREFLIGHT = "reports/voluntary-resolution-v2-tokenization-preflight.json"
MATERIALS = "reports/voluntary-resolution-v2-material-comparison.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate_binding(manifest, audit, analysis, check, analysis_byte_sha, directions_sha):
    require(manifest.get("model") == capture.MODEL, "Require the pinned candidate model")
    require(
        audit.get("passed") is True
        and audit.get("complete") is True
        and audit.get("assigned") == 384
        and audit.get("manifest_sha256") == digest(manifest),
        "Require the complete passing384 capture audit",
    )
    require(
        analysis.get("assay") == "described_state_candidate_analysis"
        and analysis.get("development_only") is True
        and analysis.get("model") == manifest["model"]
        and analysis.get("manifest_sha256") == digest(manifest)
        and analysis.get("source_sha256") == manifest.get("source_sha256")
        and analysis.get("audit_sha256") == digest(audit)
        and analysis.get("directions_sha256") == directions_sha,
        "Candidate analysis and direction bindings disagree",
    )
    candidates = {item["contrast"]: item for item in analysis["candidates"]}
    require(set(candidates) == set(TARGETS), "Bind both prespecified description candidates")
    require(
        all(
            item.get("passed") is True
            and item.get("a_selection_passed") is True
            and item.get("b_transfer_passed") is True
            and item.get("b_specificity_point_screen_passed") is True
            and item.get("selected_layer") == 15
            and item.get("position") == "record"
            for item in candidates.values()
        ),
        "This pilot requires both already selected raw layer15/record candidates",
    )
    require(
        check.get("passed") is True
        and check.get("records") == 384
        and check.get("capture_manifest_sha256") == digest(manifest)
        and check.get("analysis_sha256") == digest(analysis)
        and check.get("analysis_byte_sha256") == analysis_byte_sha
        and check.get("audit_sha256") == digest(audit)
        and check.get("directions_byte_sha256") == directions_sha,
        "Require the matching independent primary-candidate check",
    )
    return dict(
        capture_manifest_sha256=digest(manifest),
        capture_audit_sha256=digest(audit),
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=analysis_byte_sha,
        directions_byte_sha256=directions_sha,
        independent_check_sha256=digest(check),
        selected_sites={target: dict(layer=15, position="record") for target in TARGETS},
        role="Fixed corpus-A raw directions; descriptive transfer only, no felt-state classification",
    )


def prepare(out, root=ROOT):
    # The auditor supplies the exact prospectively agreed metadata, with no outcomes.
    from audit_voluntary_resolution import (
        REQUIRED_ANALYSIS_METADATA,
        REQUIRED_CRITERIA,
        exposure_binding,
        history_plan,
        prospective_checks,
        recheck_history,
    )

    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.exists(), "Voluntary collections are immutable; choose a new run directory")
    candidate_manifest, candidate_audit, analysis, check = [
        read(root / name) for name in (CAPTURE, AUDIT, ANALYSIS, CHECK)
    ]
    candidate = candidate_binding(
        candidate_manifest, candidate_audit, analysis, check, sha(root / ANALYSIS), sha(root / DIRECTIONS)
    )
    with np.load(root / DIRECTIONS, allow_pickle=False) as vectors:
        require(
            list(vectors["sources"]) == list(CONTRASTS),
            "Require all eight raw contrast directions",
        )
        require(
            list(vectors["layers"]) == [15, 31, 47, 57]
            and list(vectors["positions"]) == ["record", "final"]
            and all(
                np.isfinite(vectors[key]).all()
                for key in ("unit", "contrast", "center", "threshold", "contrast_norm")
            ),
            "Candidate direction coordinates or numerical values disagree",
        )
        for target in TARGETS:
            index = list(vectors["sources"]).index(target)
            result = next(item for item in analysis["candidates"] if item["contrast"] == target)
            require(
                float(vectors["contrast_norm"][index, 0, 0]) == result["a_raw_contrast_norm"]
                and float(vectors["threshold"][index, 0, 0]) == result["a_threshold"],
                "Selected candidate normalization/threshold changed",
            )
    history = read(root / HISTORY)
    plan = history_plan(history)
    exposure = read(root / EXPOSURE)
    exposure_binding(history, exposure, plan)
    require(
        exposure.get("audit_script_byte_sha256") == sha(root / "scripts/audit_voluntary_freshness.py")
        and all(sha(root / name) == value for name, value in exposure["source_file_sha256"].items()),
        "Prior-exposure assessment does not bind these exact selected worlds",
    )
    recheck_history(root, history, plan, prospective=True)
    preflight, materials = read(root / PREFLIGHT), read(root / MATERIALS)
    prospective_checks(plan, preflight, materials, sha(root / "src/curiosity_study/voluntary_resolution.py"))
    require(
        preflight["preflight_script_byte_sha256"] == sha(root / "scripts/preflight_voluntary_tokenization.py")
        and materials["comparison_script_byte_sha256"]
        == sha(root / "scripts/compare_voluntary_materials.py"),
        "Prospective check script bytes changed",
    )
    documents = [
        CAPTURE,
        AUDIT,
        ANALYSIS,
        DIRECTIONS,
        CHECK,
        PLAN,
        HISTORY,
        EXPOSURE,
        PREFLIGHT,
        MATERIALS,
        "reports/curiosity-satisfaction-candidate-pilot-plan.md",
        "reports/voluntary-resolution-v1-results.md",
    ]
    bindings = [dict(path=name, byte_sha256=sha(root / name)) for name in documents]
    package = root / "src/curiosity_study"
    manifest = dict(
        schema_version=3,
        assay="voluntary_resolution",
        task_revision=voluntary.REVISION,
        phase="development_pilot",
        inference_scope="development",
        model=deepcopy(capture.MODEL),
        execution_gpu="H200",
        execution=deepcopy(voluntary.EXECUTION),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=deepcopy(plan["assignments"]),
        assignment_sha256=plan["assignment_sha256"],
        max_new_tokens=4096,
        inference_revision="thinking-4096-v1",
        capture_plan=deepcopy(CAPTURE_PLAN),
        receipt_capture_plan=deepcopy(capture.RECEIPT_PLAN),
        prechoice_capture_aliases=deepcopy(capture.PRECHOICE_SITES),
        task_module_sha256=sha(package / "voluntary_resolution.py"),
        engine_sha256={
            name: sha(package / name)
            for name in (
                "voluntary_capture.py",
                "thinking_information.py",
                "state_backend.py",
                "model.py",
                "batched.py",
                "reasoning_trace.py",
            )
        },
        readiness_criteria=deepcopy(REQUIRED_CRITERIA),
        analysis_metadata=deepcopy(REQUIRED_ANALYSIS_METADATA),
        world_selection=deepcopy(plan["selection"]),
        world_selection_sha256=plan["selection_sha256"],
        counterbalancing=deepcopy(plan["counterbalancing"]),
        candidate_source=candidate,
        historical_exclusions=dict(
            path=HISTORY,
            receipt_byte_sha256=sha(root / HISTORY),
            receipt_sha256=digest(history),
            receipt=history,
        ),
        tokenization_preflight=dict(
            path=PREFLIGHT,
            receipt_byte_sha256=sha(root / PREFLIGHT),
            receipt_sha256=digest(preflight),
            receipt=preflight,
        ),
        material_comparison=dict(
            path=MATERIALS,
            receipt_byte_sha256=sha(root / MATERIALS),
            receipt_sha256=digest(materials),
            receipt=materials,
        ),
        prior_exposure=dict(path=EXPOSURE, assessment_byte_sha256=sha(root / EXPOSURE), assessment=exposure),
        document_bindings=bindings,
        compute_ceiling_usd=10,
        invalid_output_policy="terminal_no_retry; no action without valid closed choice",
        interpretation="Bounded unsteered development only; semantic candidates, task exposure, and receipt-context changes limit interpretation. No causal or subjective-experience claim.",
    )
    capture.validate_manifest(manifest)
    write_once(out / "manifest.json", manifest)
    frozen = freeze(root, out, documents)
    return dict(
        run=str(out),
        assigned=288,
        choices=144,
        diagnostics=144,
        manifest_sha256=digest(manifest),
        source_sha256=frozen["source_sha256"],
        gpu_launched=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.out))
