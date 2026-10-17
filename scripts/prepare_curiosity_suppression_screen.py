"""Prepare the fixed controlled C-negative screen; never schedule inference."""

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study.artifacts import canonical, digest, read, write_once
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_execution import freeze  # noqa: E402
from prepare_voluntary_resolution import candidate_binding  # noqa: E402
from prepare_sustained_engineering import CANDIDATE_DOCUMENTS  # noqa: E402
from prepare_satisfaction_screen import (  # noqa: E402
    ENGINEERING,
    ENGINEERING_AUDIT,
    LOCKED_CANDIDATE_MANIFEST,
    bound_receipt,
    engineering_certificate as previous_engineering_certificate,
    receipt_preflight,
    tokenizer_model,
)

PREFIX = "reports/curiosity-suppression-screen-v1"
PLAN = PREFIX + "-plan.md"
GENERATION = PREFIX + "-tokenization-preflight.json"
RECEIPTS = PREFIX + "-receipt-preflight.json"
DIRECTIONS = PREFIX + "-direction-inventory.json"
FORECAST = PREFIX + "-resource-forecast.json"
SOURCE_SCREEN = "runs/qwen32-satisfaction-screen-v1"
SOURCE_AUDIT = "reports/qwen32-satisfaction-screen-v1-audit.json"
SOURCE_ANALYSIS = "analyses/satisfaction-screen-v1/analysis.json"
SOURCE_CHECK = "reports/satisfaction-screen-v1-independent-check.json"
SOURCE_SCORING = "reports/satisfaction-screen-v1-independent-scoring.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def locked_directions(root=ROOT):
    paths = {key: Path(root) / name for key, name in CANDIDATE_DOCUMENTS.items()}
    values = {key: read(path) for key, path in paths.items() if key != "directions"}
    direction_sha = sha(paths["directions"])
    require(
        digest(values["manifest"]) == LOCKED_CANDIDATE_MANIFEST
        and direction_sha == screen.LOCKED_DIRECTION_BYTES,
        "Require original independently selected candidate materials",
    )
    candidate = candidate_binding(
        values["manifest"],
        values["audit"],
        values["analysis"],
        values["check"],
        sha(paths["analysis"]),
        direction_sha,
    )
    require(
        candidate["selected_sites"]["curiosity_vs_indifference"] == dict(layer=15, position="record"),
        "Original C site changed",
    )
    bindings = {}
    with np.load(paths["directions"], allow_pickle=False) as arrays:
        require(
            list(arrays["layers"]) == [15, 31, 47, 57] and list(arrays["positions"]) == ["record", "final"],
            "Candidate array layout changed",
        )
        sources = list(arrays["sources"])
        c_index = sources.index("curiosity_vs_indifference")
        require(
            float(arrays["contrast_norm"][c_index, 0, 0]) == screen.LOCKED_C_NORM, "Original C norm changed"
        )
        semantic = (
            ("zero_sham", "curiosity_vs_indifference", 1),
            (screen.PRIMARY, "curiosity_vs_indifference", -1),
            *screen.NEIGHBORS,
        )
        for arm, source, orientation in semantic:
            index = sources.index(source)
            original = arrays["unit"][index, 0, 0].astype(np.float64)
            applied = original * orientation
            bindings[arm] = dict(
                source=source,
                corpus="A",
                layer=15,
                position="record",
                orientation=orientation,
                source_raw_norm=float(arrays["contrast_norm"][index, 0, 0]),
                source_unit_sha256=digest(original.tolist()),
                unit=applied.tolist(),
                unit_sha256=digest(applied.tolist()),
                reference_raw_norm=screen.LOCKED_C_NORM,
                dose=0.0 if arm == "zero_sham" else screen.LOCKED_C_NORM,
                directions_byte_sha256=direction_sha,
            )
    for arm, seed in zip(screen.RANDOM_ARMS, screen.RANDOM_SEEDS, strict=True):
        draw = np.random.default_rng(seed).standard_normal(5120, dtype=np.float64)
        unit = draw / np.linalg.norm(draw)
        bindings[arm] = dict(
            source="seeded_gaussian",
            corpus=None,
            layer=15,
            position="record",
            orientation=1,
            seed=seed,
            construction="numpy_default_rng_standard_normal_float64_unit_no_resampling",
            generator="PCG64",
            numpy_version=np.__version__,
            original_draw_byte_sha256=array_sha(draw),
            original_draw_norm=float(np.linalg.norm(draw)),
            unit=unit.tolist(),
            unit_sha256=digest(unit.tolist()),
            reference_raw_norm=screen.LOCKED_C_NORM,
            dose=screen.LOCKED_C_NORM,
        )
    return screen.validate_directions(bindings), candidate


def direction_inventory(bindings):
    arms = list(screen.ARMS[1:])
    units = np.array([bindings[arm]["unit"] for arm in arms], dtype=np.float64)
    normalized = units / np.linalg.norm(units, axis=1, keepdims=True)
    return dict(
        direction_bindings_sha256=digest(bindings),
        numpy_version=np.__version__,
        generator="PCG64",
        arm_order=arms,
        signed_unit_cosines=(normalized @ normalized.T).tolist(),
        directions=[
            dict(
                arm=arm,
                binding_sha256=digest(bindings[arm]),
                signed_unit_float64_byte_sha256=array_sha(units[i]),
                effective_float32_byte_sha256=array_sha(units[i].astype(np.float32)),
                float64_norm=float(np.linalg.norm(units[i])),
                effective_float32_norm=float(np.linalg.norm(units[i].astype(np.float32))),
            )
            for i, arm in enumerate(arms)
        ],
        resampled=False,
        orientation_selected_by_outcome=False,
        interpretation="All fixed applied directions; cosines are descriptive and do not validate psychological specificity",
    )


def source_certificate(root=ROOT, *, reproduce=True):
    root = Path(root)
    manifest, frozen, audit, analysis, check, scoring = (
        read(root / SOURCE_SCREEN / "manifest.json"),
        read(root / SOURCE_SCREEN / "freeze.json"),
        read(root / SOURCE_AUDIT),
        read(root / SOURCE_ANALYSIS),
        read(root / SOURCE_CHECK),
        read(root / SOURCE_SCORING),
    )
    if reproduce:
        from audit_satisfaction_screen import audit as audit_source

        require(
            audit_source(root / SOURCE_SCREEN) == audit, "Completed source-screen audit no longer reproduces"
        )
    require(
        digest(manifest) == screen.PRIOR_MANIFEST_SHA256 == frozen["manifest_sha256"]
        and sha(root / SOURCE_SCREEN / "source.tar.gz") == frozen["archive_sha256"]
        and audit["parent_manifest_sha256"] == digest(manifest)
        and all(
            audit[k] is True
            for k in (
                "complete",
                "integrity_passed",
                "sham_identity_passed",
                "engagement_passed",
                "receipts_passed",
            )
        )
        and analysis["parent_manifest_sha256"] == digest(manifest)
        and analysis["audit_sha256"] == digest(audit)
        and analysis["decision"]["classification"] == "readiness_failed"
        and analysis["decision"]["positive_screen"] is False
        and analysis["decision"]["effect_threshold_met"] is False
        and all(value == 0.0 for value in analysis["decision"]["primary_panel_differences"].values())
        and all(value == 0.0 for value in analysis["decision"]["primary_pooled_menu_differences"].values())
        and check["passed"] is True
        and check["parent_manifest_sha256"] == digest(manifest)
        and check["analysis_sha256"] == digest(analysis)
        and check["analysis_byte_sha256"] == sha(root / SOURCE_ANALYSIS)
        and check["audit_sha256"] == digest(audit)
        and check["audit_byte_sha256"] == sha(root / SOURCE_AUDIT)
        and scoring["passed"] is True
        and scoring["checked"] == scoring["assigned"] == 1152
        and scoring["parent_manifest_sha256"] == digest(manifest)
        and scoring["complete"] is True
        and scoring["all_scoring_agrees"] is True
        and not scoring["mismatches"],
        "Bind the complete prior screen while retaining its failed scientific decision",
    )
    return manifest, dict(
        manifest_sha256=digest(manifest),
        freeze_sha256=digest(frozen),
        archive_sha256=frozen["archive_sha256"],
        audit_sha256=digest(audit),
        analysis_sha256=digest(analysis),
        independent_check_sha256=digest(check),
        independent_scoring_sha256=digest(scoring),
        prior_classification=analysis["decision"]["classification"],
        prior_positive_screen=False,
        prior_exact_choice_effect=0.0,
        scope="Entire24-world prior development set reused; failed S readiness is retained, not converted to an admission pass",
    )


def resource_forecast(root=ROOT):
    from budget import RATES

    root = Path(root)
    profiles = []
    for suffix, cohorts in (("baseline-zero", 18), ("positive-half", 9), ("positive-full", 9)):
        paths = list((root / f"{SOURCE_SCREEN}-{suffix}" / "cloud_profiles").glob("*.json"))
        require(len(paths) == 1, "Require a unique completed source runtime profile")
        value = read(paths[0])
        require(
            value["gpu"] == "NVIDIA H200" and value["assigned"] == value["completed"] == cohorts * 32,
            "Runtime source incomplete",
        )
        require(
            type(value["execution_seconds"]) in (int, float)
            and type(value["load_seconds"]) in (int, float)
            and math.isfinite(value["execution_seconds"])
            and value["execution_seconds"] > 0
            and math.isfinite(value["load_seconds"])
            and value["load_seconds"] >= 0,
            "Runtime profile requires finite positive execution and finite nonnegative loading",
        )
        profiles.append(
            dict(
                path=str(paths[0].relative_to(root)),
                byte_sha256=sha(paths[0]),
                profile=value,
                source_cohorts=cohorts,
                extrapolated_execution_seconds_per_worker=value["execution_seconds"] / cohorts * 45,
            )
        )
    low = min(r["extrapolated_execution_seconds_per_worker"] for r in profiles)
    high = max(r["extrapolated_execution_seconds_per_worker"] for r in profiles)
    load = max(r["profile"]["load_seconds"] for r in profiles)
    return dict(
        schema_version=1,
        stage_workers=3,
        cohorts_per_worker=45,
        branches_per_worker=1440,
        source_profiles=profiles,
        estimated_execution_seconds_per_worker_range=[low, high],
        maximum_observed_load_seconds=load,
        startup_export_margin_seconds=120,
        estimated_combined_compute_usd_range=[
            3 * (low + load + 120) * RATES["H200"],
            3 * (high + load + 120) * RATES["H200"],
        ],
        rate_usd_per_second=RATES["H200"],
        shared_deadline_seconds=12000,
        stage_ceiling_usd=55.0,
        maximum_reserved_compute_usd=3 * (12000 + 120) * RATES["H200"],
        hypothetical_all_cap_seconds_per_worker=45 * 4096 * 0.218,
        completion_guaranteed=False,
        paid_retries=0,
        caveat="Cohort-based scaling from observed prior trajectories includes receipt work but does not adjust for unequal new worker receipt counts. It is a workload extrapolation, not a per-worker bound or guarantee for new directions. All-cap cohorts can exceed the deadline; retain incomplete full-parent denominators without retry or arm selection.",
    )


def build_manifests(
    plan, preflight, receipt_check, source, source_cert, candidate, certificate, inventory, forecast, bindings
):
    from curiosity_study import curiosity_suppression_capture as capture
    from curiosity_study import sustained_engineering as engineering
    from curiosity_study.curiosity_resource_limits import CURIOSITY_SCREEN_BUDGET
    from analyze_curiosity_suppression_screen import ANALYSIS_METADATA

    original_preflight = source["generation_preflight"]
    require(
        preflight["requests"] == original_preflight["requests"]
        and preflight["requests_sha256"] == original_preflight["requests_sha256"]
        and preflight["base_assignment_sha256"] == original_preflight["base_assignment_sha256"],
        "Require exact original token requests, positions, seeds and base assignment binding",
    )
    parent = dict(
        schema_version=1,
        assay=screen.ASSAY,
        task_revision=screen.REVISION,
        phase="development_pilot",
        inference_scope="development",
        model=deepcopy(capture.MODEL),
        execution_gpu="H200",
        execution=deepcopy(screen.EXECUTION),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=deepcopy(plan["assignments"]),
        assignment_sha256=plan["assignment_sha256"],
        screen_plan=deepcopy(plan),
        screen_plan_sha256=digest(plan),
        source_screen_manifest=deepcopy(source),
        source_screen_manifest_sha256=digest(source),
        source_screen_certificate=source_cert,
        generation_preflight=deepcopy(preflight),
        generation_preflight_sha256=digest(preflight),
        receipt_preflight=deepcopy(receipt_check),
        receipt_preflight_sha256=digest(receipt_check),
        direction_inventory_sha256=digest(inventory),
        resource_forecast_sha256=digest(forecast),
        max_new_tokens=4096,
        capture_plan=deepcopy(capture.CAPTURE_PLAN),
        arithmetic=deepcopy(engineering.ARITHMETIC),
        schedule=screen.SCHEDULE,
        candidate_source=deepcopy(candidate),
        engineering_certificate=certificate,
        engine_sha256=engineering.engine_hashes(),
        bound_receipts=bindings,
        analysis_metadata=deepcopy(ANALYSIS_METADATA),
        budget=deepcopy(CURIOSITY_SCREEN_BUDGET),
        invalid_output_policy="Terminal without replacement or retries; all4320 assignments retained",
    )
    children = []
    for group in screen.GROUPS:
        child = deepcopy(parent)
        child.update(
            execution_group=group,
            parent_manifest_sha256=digest(parent),
            diagnostics=screen.group_assignments(plan, group),
        )
        child["assignment_sha256"] = digest(child["diagnostics"])
        capture.validate_manifest(child)
        children.append(child)
    combined = sorted((r for c in children for r in c["diagnostics"]), key=lambda r: r["parent_row"])
    require(combined == parent["diagnostics"], "Children must preserve every parent row exactly once")
    return parent, children


def prepare(out, root=ROOT):
    from curiosity_study import curiosity_suppression_capture as capture

    root, out = Path(root).resolve(), Path(out).resolve()
    require(out.parent == root / "runs", "Place parent directly under runs")
    children_paths = [root / "runs" / f"{out.name}-{group.replace('_', '-')}" for group in screen.GROUPS]
    require(
        not any(
            p.exists()
            for p in [
                out,
                root / "executions" / out.name,
                *children_paths,
                *(root / n for n in (GENERATION, RECEIPTS, DIRECTIONS, FORECAST)),
            ]
        ),
        "Prepared study artifacts are immutable",
    )
    require((root / PLAN).is_file(), "Require prospective plan")
    source, source_cert = source_certificate(root)
    certificate = previous_engineering_certificate(root)
    certificate["scope"] = (
        "Existing full-cap C+ one-row engineering and S all-row evidence; new signed directions and all-row engagement require their own complete scientific audit"
    )
    directions, candidate = locked_directions(root)
    plan = screen.assignment_plan(source_manifest=source, direction_bindings=directions)
    model = tokenizer_model()
    preflight, receipt_check = capture.bind_preflight(plan, model), receipt_preflight(plan, model)
    inventory, forecast = direction_inventory(directions), resource_forecast(root)
    require(
        forecast["maximum_reserved_compute_usd"] < 55
        and max(forecast["estimated_execution_seconds_per_worker_range"])
        + forecast["maximum_observed_load_seconds"]
        + 120
        < 12000,
        "Insufficient measured resource headroom",
    )
    new_receipts = dict(
        (name, value)
        for name, value in (
            (GENERATION, preflight),
            (RECEIPTS, receipt_check),
            (DIRECTIONS, inventory),
            (FORECAST, forecast),
        )
    )
    documents = dict(
        prospective_plan=PLAN,
        generation_preflight=GENERATION,
        receipt_preflight=RECEIPTS,
        direction_inventory=DIRECTIONS,
        resource_forecast=FORECAST,
        source_screen_manifest=f"{SOURCE_SCREEN}/manifest.json",
        source_screen_audit=SOURCE_AUDIT,
        source_screen_freeze=f"{SOURCE_SCREEN}/freeze.json",
        source_screen_archive=f"{SOURCE_SCREEN}/source.tar.gz",
        source_screen_analysis=SOURCE_ANALYSIS,
        source_screen_independent_check=SOURCE_CHECK,
        source_screen_independent_scoring=SOURCE_SCORING,
        source_screen_results="reports/satisfaction-screen-v1-results.md",
        engineering_manifest=f"{ENGINEERING}/manifest.json",
        engineering_audit=ENGINEERING_AUDIT,
        engineering_freeze=f"{ENGINEERING}/freeze.json",
        engineering_archive=f"{ENGINEERING}/source.tar.gz",
    )
    for key, name in CANDIDATE_DOCUMENTS.items():
        documents["candidate_" + ("independent_check" if key == "check" else key)] = name
    bindings = [
        dict(
            role=role,
            path=name,
            byte_sha256=hashlib.sha256((canonical(new_receipts[name]) + "\n").encode()).hexdigest(),
            content_sha256=digest(new_receipts[name]),
        )
        if name in new_receipts
        else bound_receipt(root, role, name)
        for role, name in documents.items()
    ]
    parent, children = build_manifests(
        plan,
        preflight,
        receipt_check,
        source,
        source_cert,
        candidate,
        certificate,
        inventory,
        forecast,
        bindings,
    )
    # All source, tokenizer, resource and manifest validation precedes publication.
    for name, value in new_receipts.items():
        write_once(root / name, value)
    write_once(out / "manifest.json", parent)
    descriptors = []
    for path, child in zip(children_paths, children, strict=True):
        write_once(path / "manifest.json", child)
        descriptors.append(
            dict(path=str(path), manifest_sha256=digest(child), execution_group=child["execution_group"])
        )
    write_once(out / "children.json", descriptors)
    frozen = freeze(root, out, documents=list(documents.values()))
    return dict(
        run=str(out),
        assigned=4320,
        workers=3,
        manifest_sha256=digest(parent),
        source_sha256=frozen["source_sha256"],
        gpu_launched=False,
        known_worlds_reused=24,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(prepare(args.out))
