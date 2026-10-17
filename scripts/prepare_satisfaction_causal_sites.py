"""Prepare one fixed complete S-site phase; no launch and no outcome-based repair."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_site_capture as capture
from curiosity_study import satisfaction_site_execution as execution
from curiosity_study import satisfaction_site_resources as resources
from curiosity_study import causal_site_engineering as engineering
from curiosity_study.artifacts import canonical, digest, read, write_once
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402
from satisfaction_site_sources import locked_directions, sha, CANDIDATE_DOCUMENTS  # noqa: E402
from prepare_curiosity_suppression_screen import source_certificate  # noqa: E402
from prepare_satisfaction_screen import tokenizer_model, bound_receipt  # noqa: E402
from prepare_satisfaction_site_setup import DESIGN, DESIGN_SHA, PLAN, RESERVED, priority_gate, forecast  # noqa: E402

SETUP = "runs/qwen32-satisfaction-causal-sites-v1-setup"
SETUP_AUDIT = "reports/qwen32-satisfaction-causal-sites-v1-setup-audit.json"
CALIBRATION = "runs/qwen32-satisfaction-causal-sites-v1-calibration"
CALIBRATION_AUDIT = "reports/qwen32-satisfaction-causal-sites-v1-calibration-audit.json"
CALIBRATION_ANALYSIS = "analyses/satisfaction-causal-sites-v1-calibration/analysis.json"
CALIBRATION_CHECK = "reports/satisfaction-causal-sites-v1-calibration-independent-check.json"
require = study.require


def setup_certificate(root):
    from audit_satisfaction_site_setup import audit

    root = Path(root)
    manifest, frozen, receipt = read(root / SETUP / "manifest.json"), read(root / SETUP / "freeze.json"), read(root / SETUP_AUDIT)
    require(receipt == audit(root / SETUP) and receipt["complete"] is True and receipt["passed"] is True
            and receipt["full_cap_certified"] is False and receipt["assigned"] == 544,
            "Both prescribed site checks and complete reference must independently reproduce")
    reference = study.validate_reference(receipt["dose_reference"])
    require(receipt["manifest_sha256"] == digest(manifest) == frozen["manifest_sha256"]
            and receipt["source_sha256"] == manifest["source_sha256"]
            and frozen["archive_sha256"] == sha(root / SETUP / "source.tar.gz")
            and receipt["dose_reference_sha256"] == digest(reference), "Setup certificate binding changed")
    return dict(
        run=SETUP, manifest_path=SETUP + "/manifest.json", manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"], audit_path=SETUP_AUDIT, audit_sha256=digest(receipt),
        audit_byte_sha256=sha(root / SETUP_AUDIT), freeze_path=SETUP + "/freeze.json", freeze_sha256=digest(frozen),
        archive_path=SETUP + "/source.tar.gz", archive_byte_sha256=frozen["archive_sha256"],
        dose_reference=reference, dose_reference_sha256=digest(reference), passed=True, full_cap_certified=False,
        scope="Exact read-only288 reference and observed128-token S31/S47 engineering. Every scientific full-length sham, geometry, source, task and selected-receipt gate remains required.",
    )


def selection_certificate(root):
    """A complete independently checked calibration can select at most one site."""
    root = Path(root)
    m, a, result, check = (read(root / name) for name in
                          (CALIBRATION + "/manifest.json", CALIBRATION_AUDIT, CALIBRATION_ANALYSIS, CALIBRATION_CHECK))
    frozen = read(root / CALIBRATION / "freeze.json")
    require(m["assay"] == study.ASSAY and m["site_phase"] == m["site_plan"]["phase"] == "calibration"
            and m["site_plan"]["sites"] == list(study.SITES)
            and frozen["manifest_sha256"] == digest(m) and frozen["source_sha256"] == m["source_sha256"]
            and frozen["archive_sha256"] == sha(root / CALIBRATION / "source.tar.gz"),
            "Require the complete frozen two-site calibration")
    chosen = result["decision"]["selected_site"]
    require(a["complete"] is True and a["integrity_passed"] is True
            and a["parent_manifest_sha256"] == digest(m)
            and result["parent_manifest_sha256"] == digest(m) and result["audit_sha256"] == digest(a)
            and check["passed"] is True and check["parent_manifest_sha256"] == digest(m)
            and check["analysis_sha256"] == digest(result) and check["audit_sha256"] == digest(a)
            and check["analysis_byte_sha256"] == sha(root / CALIBRATION_ANALYSIS)
            and check["audit_byte_sha256"] == sha(root / CALIBRATION_AUDIT)
            and check["decision"] == result["decision"]
            and result["decision"]["positive_screen"] is True and chosen in study.SITES,
            "No qualified calibration site: the finite extension ends without validation")
    from check_satisfaction_site_decision import check as independent_check

    reproduced = independent_check(root / CALIBRATION, root / CALIBRATION_ANALYSIS, root / CALIBRATION_AUDIT)
    require(all(check[key] == reproduced[key] for key in
                ("checker_byte_sha256", "assigned", "compared_values", "decision", "recomputed_decision")),
            "Calibration selection and readiness must reproduce in the independent checker")
    return dict(calibration_run=CALIBRATION, manifest_path=CALIBRATION + "/manifest.json",
        manifest_sha256=digest(m), source_sha256=m["source_sha256"],
        audit_path=CALIBRATION_AUDIT, audit_sha256=digest(a), audit_byte_sha256=sha(root / CALIBRATION_AUDIT),
        freeze_path=CALIBRATION + "/freeze.json", freeze_sha256=digest(frozen),
        archive_path=CALIBRATION + "/source.tar.gz", archive_byte_sha256=frozen["archive_sha256"],
        analysis_path=CALIBRATION_ANALYSIS, analysis_sha256=digest(result),
        analysis_byte_sha256=sha(root / CALIBRATION_ANALYSIS), independent_check_path=CALIBRATION_CHECK,
        independent_check_sha256=digest(check), independent_check_byte_sha256=sha(root / CALIBRATION_CHECK),
        selected_site=chosen, decision=deepcopy(result["decision"]))


def incremental_exposure(root, bundle, *, setup_cert, selection_receipt=None):
    """Extend the pre-setup census solely with independently bound known-world work.

The original historical audit already covered the complete local/archive inputs.
Here every old registry entry and archive remains unchanged; newly executed work
must be the exact setup or the complete calibration, both confined to old S24.
Prepared-only additional manifests are recorded but are not treated as execution.
"""
    import audit_voluntary_freshness as freshness

    root = Path(root).resolve()
    history = bundle["history"]
    require(bundle["history_sha256"] == digest(history) == bundle["selection"]["history_sha256"],
            "Original reserved history binding changed")
    current, manifests = freshness.inventory(root)
    by_run = {row["run"]: row for row in current}
    excluded = {"input_exposure_included", "explicitly_active_assigned_parent"}
    old = {row["run"]: {key: value for key, value in row.items() if key not in excluded}
           for row in history["historical_registry"]}
    require(all(by_run.get(name) == value for name, value in old.items()),
            "Previously audited exposure inventory changed")
    allowed = {setup_cert["run"]: setup_cert["manifest_sha256"]}
    setup_manifest = manifests[setup_cert["run"]]
    source = read(root / "runs/qwen32-satisfaction-screen-v1/manifest.json")
    original_requests = [item["request"] for item in source["generation_preflight"]["requests"]]
    require(setup_manifest["requests"] == original_requests
            and setup_manifest["source_certificate"]["reserved_materials_sha256"] == digest(bundle),
            "New setup does not use exactly the old288 prefixes and fixed reserved commitment")
    if selection_receipt is not None:
        parent_path = root / selection_receipt["calibration_run"]
        parent = manifests[selection_receipt["calibration_run"]]
        require(parent["site_phase"] == "calibration"
                and digest(parent["site_plan"]["base_assignments"]) == study.CALIBRATION_BASE_SHA256
                and parent["reserved_materials_sha256"] == digest(bundle), "Calibration introduced other source worlds")
        allowed[selection_receipt["calibration_run"]] = selection_receipt["manifest_sha256"]
        for child in read(parent_path / "children.json"):
            name = str(Path(child["path"]).resolve().relative_to(root))
            require(manifests[name]["parent_manifest_sha256"] == digest(parent), "Calibration child changed parent")
            allowed[name] = child["manifest_sha256"]
    for name, expected in allowed.items():
        require(name in by_run and digest(manifests[name]) == expected,
                "Explicit known-world campaign collection is absent or changed")
    unexpected = [row["run"] for row in current if row["run"] not in old and row["run"] not in allowed
                  and row["execution_evidence"] != "none"]
    require(not unexpected, "An unreviewed collection executed after the reserved inventory")
    for binding in history["source_archive_bindings"].values():
        require(sha(root / binding["archive"]) == binding["archive_byte_sha256"], "Historical source archive changed")
    for binding in history["outer_source_bindings"]:
        require(sha(root / binding["run"] / "source.tar.gz") == binding["archive_byte_sha256"],
                "Historical outer source archive changed")
    closing_registry, closing_manifests = freshness.inventory(root)
    require(closing_registry == current and closing_manifests == manifests,
            "Exposure inventory changed during the incremental check")
    return dict(schema_version=1, passed=True, scientific_admission=False,
        initial_history_sha256=digest(history), reserved_materials_sha256=digest(bundle),
        registry=current, registry_sha256=digest(current), allowed_known_world_runs=allowed,
        closing_inventory_unchanged=True,
        newly_executed_worlds_outside_initial_history=0, reserved_exact_overlap_count=0,
        source_subset=study.CALIBRATION_BASE_SHA256,
        scope="Incremental live input-census check after the complete bound historical audit. Only exact288 known-world setup/calibration exposure is added. No independent replay of the whole historical archive census is claimed.")


def direction_inventory(directions):
    import numpy as np

    result = {}
    for site, bindings in directions.items():
        arms = list(bindings)
        values = np.asarray([bindings[arm]["unit"] for arm in arms], dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        normalized = values / norms[:, None]
        result[site] = dict(arms=arms, requested_doses=[bindings[arm]["dose"] for arm in arms],
            signed_unit_cosines=(normalized @ normalized.T).tolist(),
            direction_binding_sha256={arm: digest(bindings[arm]) for arm in arms},
            interpretation="Local vector geometry; named controls do not establish functional psychological manipulation")
    return dict(direction_bindings_sha256=digest(directions), sites=result, resampled=False, refitted=False)


def build_manifests(plan, *, source, source_cert, setup_cert, reserved, preflight, receipt_check,
                    candidate, inventory, estimate, exposure, bindings, selection_receipt=None):
    from analyze_satisfaction_causal_sites import ANALYSIS_METADATA

    phase = plan["phase"]
    parent = dict(
        schema_version=1, assay=study.ASSAY, task_revision=study.REVISION,
        phase="development_pilot", site_phase=phase, inference_scope="development",
        campaign_id=study.REVISION, model=deepcopy(execution.MODEL), execution_gpu="H200",
        execution=deepcopy(study.EXECUTION), source_sha256=source_hash(), assignments=[],
        diagnostics=deepcopy(plan["assignments"]), assignment_sha256=plan["assignment_sha256"],
        site_plan=deepcopy(plan), site_plan_sha256=digest(plan),
        source_screen_manifest=deepcopy(source), source_screen_manifest_sha256=digest(source),
        source_screen_certificate=deepcopy(source_cert), setup_certificate=deepcopy(setup_cert),
        setup_certificate_sha256=digest(setup_cert), reserved_materials=deepcopy(reserved),
        reserved_materials_sha256=digest(reserved), generation_preflight=deepcopy(preflight),
        generation_preflight_sha256=digest(preflight), receipt_preflight=deepcopy(receipt_check),
        receipt_preflight_sha256=digest(receipt_check), receipt_grouping=execution.RECEIPT_GROUPING,
        capture_plans={str(site): capture.capture_plan(site) for site in plan["sites"]},
        max_new_tokens=4096, arithmetic=deepcopy(engineering.ARITHMETIC), schedule=study.SCHEDULE,
        candidate_source=deepcopy(candidate), direction_inventory_sha256=digest(inventory),
        resource_forecast_sha256=digest(estimate), exposure_receipt_sha256=digest(exposure),
        bound_receipts=deepcopy(bindings), analysis_metadata=deepcopy(ANALYSIS_METADATA),
        budget=deepcopy(resources.POLICIES[phase]),
        invalid_output_policy=f"Terminal no retries or replacement; all{len(plan['assignments'])} rows remain assigned",
    )
    if selection_receipt is not None:
        parent.update(selected_site=selection_receipt["selected_site"], selection_receipt=deepcopy(selection_receipt),
                      selection_receipt_sha256=digest(selection_receipt))
    children = []
    for group in study.GROUPS:
        child = deepcopy(parent)
        child.update(execution_group=group, parent_manifest_sha256=digest(parent),
                     diagnostics=study.group_assignments(plan, group))
        child["assignment_sha256"] = digest(child["diagnostics"])
        execution.validate_manifest(child)
        children.append(child)
    require(sorted((row for child in children for row in child["diagnostics"]), key=lambda row: row["parent_row"])
            == parent["diagnostics"], "Child decomposition changed the complete grid")
    return parent, children


def prepare(out, *, phase, root=ROOT):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(phase in ("calibration", "validation") and out.parent == root / "runs", "Unknown phase/location")
    prefix = f"reports/satisfaction-causal-sites-v1-{phase}"
    names = {role: prefix + "-" + suffix + ".json" for role, suffix in
             (("generation_preflight", "tokenization-preflight"), ("receipt_preflight", "receipt-preflight"),
              ("direction_inventory", "direction-inventory"), ("resource_forecast", "resource-forecast"),
              ("exposure_receipt", "exposure-receipt"))}
    children_paths = [root / "runs" / f"{out.name}-{group.replace('_', '-')}" for group in study.GROUPS]
    require(not any(p.exists() for p in [out, *children_paths, root / "executions" / out.name,
                                        *(root / name for name in names.values())]), "Phase materials are immutable")
    require(sha(root / DESIGN) == DESIGN_SHA and (root / PLAN).is_file(), "Prospective design/plan absent or changed")
    require(not any(read(p).get("assay") == study.ASSAY and read(p).get("site_phase") == phase
                    for p in (root / "runs").glob("*/manifest.json")), "This finite scientific phase was already prepared")
    priority_gate(root)
    reserved = read(root / RESERVED)
    setup_cert = setup_certificate(root)
    source, source_cert = source_certificate(root, reproduce=False)
    selected = None if phase == "calibration" else selection_certificate(root)
    exposure = incremental_exposure(root, reserved, setup_cert=setup_cert, selection_receipt=selected)
    all_directions, candidate = locked_directions(root, doses=setup_cert["dose_reference"]["doses"])
    if phase == "calibration":
        base, selection = study.calibration_inputs(source)
        directions = {str(site): {arm: all_directions[str(site)][arm] for arm in study.CALIBRATION_ARMS[1:]}
                      for site in study.SITES}
    else:
        base, selection = reserved["base_assignments"], reserved["selection"]
        directions = {str(selected["selected_site"]): all_directions[str(selected["selected_site"])]}
    plan = study.assignment_plan(phase=phase, base_assignments=base, selection=selection,
        direction_bindings=directions, dose_reference=setup_cert["dose_reference"],
        selected_site=None if selected is None else selected["selected_site"])
    tokenizer = tokenizer_model()
    preflight, receipt_check = execution.bind_preflight(plan, tokenizer), execution.bind_receipt_preflight(plan, tokenizer)
    inventory, estimate = direction_inventory(directions), forecast(root, phase=phase)
    new = {names[role]: value for role, value in
           (("generation_preflight", preflight), ("receipt_preflight", receipt_check), ("direction_inventory", inventory),
            ("resource_forecast", estimate), ("exposure_receipt", exposure))}
    documents = dict(prospective_plan=PLAN, prospective_design=DESIGN, reserved_materials=RESERVED,
                     setup_manifest=setup_cert["manifest_path"], setup_audit=SETUP_AUDIT,
                     setup_freeze=setup_cert["freeze_path"], setup_archive=setup_cert["archive_path"],
                     source_screen_manifest="runs/qwen32-satisfaction-screen-v1/manifest.json", **names)
    for key, name in CANDIDATE_DOCUMENTS.items():
        documents["candidate_" + ("independent_check" if key == "check" else key)] = name
    for index, path in enumerate(sorted((root / SETUP).rglob("*"))):
        if path.is_file() and path.suffix in (".json", ".npz") and str(path.relative_to(root)) not in documents.values():
            documents[f"setup_artifact_{index:04d}"] = str(path.relative_to(root))
    if selected is not None:
        for key in ("manifest_path", "freeze_path", "archive_path", "audit_path", "analysis_path", "independent_check_path"):
            documents["calibration_" + key.removesuffix("_path")] = selected[key]
    bindings = [dict(role=role, path=name, byte_sha256=hashlib.sha256((canonical(new[name]) + "\n").encode()).hexdigest(),
                     content_sha256=digest(new[name])) if name in new else bound_receipt(root, role, name)
                for role, name in documents.items()]
    parent, children = build_manifests(plan, source=source, source_cert=source_cert, setup_cert=setup_cert,
        reserved=reserved, preflight=preflight, receipt_check=receipt_check, candidate=candidate, inventory=inventory,
        estimate=estimate, exposure=exposure, bindings=bindings, selection_receipt=selected)
    for name, value in new.items():
        write_once(root / name, value)
    write_once(out / "manifest.json", parent)
    descriptors = []
    for path, child in zip(children_paths, children, strict=True):
        write_once(path / "manifest.json", child)
        descriptors.append(dict(path=str(path), manifest_sha256=digest(child), execution_group=child["execution_group"]))
    write_once(out / "children.json", descriptors)
    frozen = freeze(root, out, documents=list(documents.values()))
    return dict(run=str(out), phase=phase, assigned=len(plan["assignments"]), manifest_sha256=digest(parent),
                source_sha256=frozen["source_sha256"], gpu_launched=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "validation"), required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.out, phase=args.phase)))
