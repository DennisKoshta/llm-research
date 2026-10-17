"""Prospective two-site S+ analysis; no site, dose, or outcome exclusion search.

Pure-table calculations only. Source/engine/material admission belongs to the
bound parent audit; these calculations never certify that evidence themselves.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study import satisfaction_causal_sites as screen
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once
from analyze_curiosity_suppression_screen import CATEGORIES, GATES, category, counts, write_csv

BOOTSTRAPS, SEED = screen.BOOTSTRAP_DRAWS, screen.BOOTSTRAP_SEED
PRIMARY = screen.PRIMARY
TOP_GATES = ("complete", "integrity_passed", "worker_policies_equal")
METADATA = ("trial_id", "base_trial_id", "panel_id", "semantic_world_id", "world_slot", "arm",
            "family", "kind", "delivery", "menu_layout", "phase", "site")
ANALYSIS_METADATA = dict(
    schema_version=1, bootstrap_samples=BOOTSTRAPS, bootstrap_seed=SEED,
    bootstrap_unit="whole_semantic_world_within_panel_or_reserved_half",
    shared_draw_banks_across_all_sites_and_endpoints=True,
    primary_arm=PRIMARY, comparator="zero_sham", primary_delivery="redundant_evidence",
    primary_category="stop", minimum_panel_difference=0.25,
    each_pooled_menu_difference_strictly_positive=True,
    minimum_each_control_panel_difference=0.125,
    each_control_pooled_menu_difference_nonnegative=True,
    comparator_favorable_invalidity_bound_required=True,
    calibration_sites=list(screen.SITES), calibration_arms=list(screen.CALIBRATION_ARMS),
    calibration_controls=list(screen.CALIBRATION_CONTROLS),
    validation_arms=list(screen.VALIDATION_ARMS), validation_controls=list(screen.VALIDATION_ARMS[3:]),
    calibration_selection="earliest_qualifying_31_else_47_else_none",
    calibration_readiness="each_site_arm_panel:diagnostic12_observed12_valid12_correct>=11_caps0;choice24_observed24_valid>=23_caps0",
    validation_readiness="each_arm_whole16:diagnostic16_observed16_valid16_correct>=15_caps0;choice32_observed32_valid>=31_caps0",
    validation_effect_halves="fixed_lexical8_plus8",
    whole_calibration_grid_integrity_required=True, interval_controls_advancement=False,
    scientific_confirmation=False, natural_satisfaction_claim=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def design(plan):
    phase = plan.get("phase")
    require(phase in ("calibration", "validation"), "Unknown analysis phase")
    calibration = phase == "calibration"
    sites = tuple(plan["sites"])
    arms = screen.CALIBRATION_ARMS if calibration else screen.VALIDATION_ARMS
    require((sites == screen.SITES if calibration else len(sites) == 1 and sites[0] in screen.SITES)
            and tuple(plan["arms"]) == arms, "Prescribed site/arm grid differs")
    return phase, sites, arms, ("panel_1", "panel_2") if calibration else ("half_1", "half_2"), 12 if calibration else 8


def inventory(plan, outcomes):
    """Require exact parent crossing and reconcile every outcome before scoring."""
    phase, sites, arms, panels, n = design(plan)
    assignments = plan["assignments"]
    expected_n = 4032 if phase == "calibration" else 2880
    require(digest(assignments) == plan["assignment_sha256"], "Parent assignment binding differs")
    require(len(assignments) == len(outcomes) == expected_n, "Require the complete parent, not a child or subset")
    assigned = {a["trial_id"]: a for a in assignments}
    observed = {r["trial_id"]: r for r in outcomes}
    require(len(assigned) == len(observed) == expected_n and set(assigned) == set(observed),
            "Missing, duplicate or extra outcomes")
    cells, bases, slots, ordinals = {}, {}, defaultdict(dict), {}
    for a in assignments:
        require(a["phase"] == phase and a["site"] in sites and a["arm"] in arms
                and a["panel_id"] in panels and a["family"] == "objects"
                and a["kind"] in ("choice", *voluntary.DIAGNOSTICS)
                and a["delivery"] in voluntary.DELIVERIES
                and type(a["menu_layout"]) is int and a["menu_layout"] in (0, 1)
                and type(a["world_slot"]) is int,
                "Unknown site/panel/arm/task cell")
        panel, world = a["panel_id"], a["semantic_world_id"]
        require(slots[panel].setdefault(world, a["world_slot"]) == a["world_slot"],
                "Semantic world changed source slot")
        if phase == "validation":
            require(type(a.get("reserved_ordinal")) is int
                    and ordinals.setdefault(world, a["reserved_ordinal"]) == a["reserved_ordinal"],
                    "Reserved ordinal changed")
        key = (a["site"], panel, world, a["arm"], a["kind"], a["delivery"], a["menu_layout"])
        require(key not in cells, "Duplicate semantic-world cell")
        row = deepcopy(observed[a["trial_id"]])
        require(all(row.get(k) == a[k] for k in METADATA)
                and (phase != "validation" or row.get("reserved_ordinal") == a["reserved_ordinal"]),
                "Outcome assignment metadata differs")
        require(row.get("status") in ("completed", "execution_error", "execution_failure", "missing"),
                "Unknown outcome status")
        if row["status"] == "completed":
            require(all(type(row.get(k)) is bool for k in ("valid", "thinking_closed", "cap_hit")),
                    "Completed outcomes require boolean validity/closure/cap")
            require(not row["valid"] or row["thinking_closed"], "Valid response must be closed")
            if a["kind"] == "choice":
                require(row.get("correct") is None, "Choices have no correctness label")
                require(row.get("selected_service") in voluntary.SERVICES if row["valid"]
                        else row.get("selected_service") is None, "Invalid choices cannot execute a service")
            else:
                require(type(row.get("correct")) is bool and (not row["correct"] or row["valid"]),
                        "Diagnostics require observed correctness and valid correct answers")
        else:
            require(row.get("valid") in (None, False) and row.get("correct") in (None, False)
                    and row.get("selected_service") is None, "Unavailable row fabricated a response")
        cells[key] = row
        stable = {k: a[k] for k in METADATA if k not in ("trial_id", "arm", "site")}
        if phase == "validation":
            stable["reserved_ordinal"] = a["reserved_ordinal"]
        require(bases.setdefault(a["base_trial_id"], stable) == stable, "Base branch differs across sites/arms")
    worlds = {p: sorted(slots[p]) for p in panels}
    expected_slots = set(range(12)) if n == 12 else {0, 1, 4, 5, 6, 7, 10, 11}
    require(len(bases) == 2*n*12 and all(len(worlds[p]) == n and set(slots[p].values()) == expected_slots for p in panels)
            and len(set().union(*(set(v) for v in worlds.values()))) == 2*n,
            "Require the full distinct-world and source-slot inventory")
    if phase == "validation":
        ordered_worlds = sorted(w for values in worlds.values() for w in values)
        require(worlds[panels[0]] == ordered_worlds[:8] and worlds[panels[1]] == ordered_worlds[8:]
                and all(ordinals[w] == i for i, w in enumerate(ordered_worlds)),
                "Reserved lexical halves/ordinals differ")
    variants = {("choice", d, m) for d in voluntary.DELIVERIES for m in (0, 1)}
    variants.update((k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
                    for k in voluntary.DIAGNOSTICS)
    expected = {(s, p, w, arm, *variant) for s in sites for p in panels for w in worlds[p]
                for arm in arms for variant in variants}
    require(set(cells) == expected, "Every site/world requires all arms and all twelve variants")
    return cells, worlds


def task_readiness(cells, *, phase, site, arms):
    """Independent counts: calibration per12 panel, validation whole16."""
    values = [r for r in cells.values() if r["site"] == site]
    groups = ("panel_1", "panel_2") if phase == "calibration" else ("all_reserved",)
    n = 12 if phase == "calibration" else 16
    diagnostic_cells, choice_cells = [], []
    for arm in arms:
        for panel in groups:
            group = [r for r in values if r["arm"] == arm and (panel == "all_reserved" or r["panel_id"] == panel)]
            for kind in voluntary.DIAGNOSTICS:
                result = counts([r for r in group if r["kind"] == kind])
                passed = (result["assigned"] == result["observed"] == result["valid"] == n
                          and result["correct"] >= n-1 and result["cap_hits"] == 0)
                diagnostic_cells.append(dict(site=site, arm=arm, panel_id=panel, kind=kind, **result, passed=passed))
            for delivery in voluntary.DELIVERIES:
                result = counts([r for r in group if r["kind"] == "choice" and r["delivery"] == delivery])
                result["correct"] = None
                passed = (result["assigned"] == result["observed"] == 2*n
                          and result["valid"] >= 2*n-1 and result["cap_hits"] == 0)
                choice_cells.append(dict(site=site, arm=arm, panel_id=panel, delivery=delivery, **result, passed=passed))
    return dict(passed=all(r["passed"] for r in diagnostic_cells+choice_cells),
                diagnostic_cells=diagnostic_cells, choice_cells=choice_cells,
                interpretation="Observed gates do not establish noninferiority, unchanged confidence, or psychological specificity.")


def bootstrap_bank(worlds):
    rng = np.random.default_rng(SEED)
    return {p: rng.integers(0, len(ws), size=(BOOTSTRAPS, len(ws))) for p, ws in worlds.items()}


def paired_endpoints(values, banks):
    panels = tuple(values)
    draws = {p: values[p][banks[p]].mean(axis=1) for p in panels}
    means = {p: float(values[p].mean()) for p in panels}
    draws["pooled"] = sum(draws.values()) / len(panels)
    means["pooled"] = sum(means.values()) / len(panels)
    return [dict(panel_id=p, estimate=means[p], ci95=np.quantile(draws[p], [0.025, 0.975]).tolist(),
                 assigned_worlds=sum(map(len, values.values())) if p == "pooled" else len(values[p]),
                 empirical_interval_degenerate=bool(np.ptp(draws[p]) == 0)) for p in (*panels, "pooled")]


def paired_values(cells, worlds, site, left, right, delivery, menus, endpoint, *, comparator_favorable=False):
    bad = {"invalid", "capped", "execution_failure", "missing"}
    def score(row, comparator=False):
        value = category(row)
        return float(value == endpoint or (comparator and comparator_favorable and value in bad))
    return {p: np.array([np.mean([
        score(cells[site, p, w, left, "choice", delivery, m])
        - score(cells[site, p, w, right, "choice", delivery, m], True) for m in menus])
        for w in ws], dtype=float) for p, ws in worlds.items()}


def effect_summary(rows, site, left, right, panels, *, bound=False):
    selected = [r for r in rows if r["site"] == site and r["left_arm"] == left and r["right_arm"] == right
                and r["category"] == "stop" and r["delivery"] == "redundant_evidence"]
    require(len(selected) == 9 and {(r["panel_id"], r["menu_layout"]) for r in selected}
            == {(p, m) for p in (*panels, "pooled") for m in (0, 1, "averaged")},
            "Incomplete fixed decision endpoint inventory")
    field = "lower_bound" if bound else "estimate"
    require(all(np.isfinite(r[field]) for r in selected), "Nonfinite decision effect")
    return dict(panel_differences={r["panel_id"]: r[field] for r in selected if r["menu_layout"] == "averaged"},
                pooled_menu_differences={str(r["menu_layout"]): r[field] for r in selected
                                        if r["panel_id"] == "pooled" and r["menu_layout"] in (0, 1)})


def threshold(summary, panels, *, specificity=False):
    return all(summary["panel_differences"][p] >= (0.125 if specificity else 0.25) for p in panels) and all(
        v >= 0 if specificity else v > 0 for v in summary["pooled_menu_differences"].values())


def decide_site(endpoints, bounds, *, site, panels, controls, integrity, readiness, audit_readiness):
    primary = effect_summary(endpoints, site, PRIMARY, "zero_sham", panels)
    lower = effect_summary(bounds, site, PRIMARY, "zero_sham", panels, bound=True)
    rivals = {}
    for control in controls:
        observed = effect_summary(endpoints, site, PRIMARY, control, panels)
        conservative = effect_summary(bounds, site, PRIMARY, control, panels, bound=True)
        rivals[control] = dict(observed=observed, comparator_favorable=conservative,
                              observed_threshold_met=threshold(observed, panels, specificity=True),
                              bounded_threshold_met=threshold(conservative, panels, specificity=True))
    effect = threshold(primary, panels) and threshold(lower, panels)
    specificity = all(v["observed_threshold_met"] and v["bounded_threshold_met"] for v in rivals.values())
    classification = ("global_integrity_or_engineering_failed" if not integrity else "readiness_failed"
                      if not readiness["passed"] or not audit_readiness["passed"] else
                      "no_substantial_primary_development_effect" if not effect else
                      "primary_effect_without_predefined_specificity" if not specificity else
                      "direction_selective_development_signal")
    return dict(classification=classification, positive_screen=classification == "direction_selective_development_signal",
                effect_threshold_met=effect, primary_observed_threshold_met=threshold(primary, panels),
                primary_comparator_favorable_threshold_met=threshold(lower, panels), specificity_threshold_met=specificity,
                primary_observed=primary, primary_comparator_favorable=lower, controls=rivals,
                recomputed_readiness_passed=readiness["passed"], audit_readiness_passed=audit_readiness["passed"])


def analyze_tables(plan, outcomes, *, checks, site_checks):
    phase, sites, arms, panels, n = design(plan)
    require(set(checks) == set(TOP_GATES) and all(type(v) is bool for v in checks.values()),
            "Require explicit global complete/integrity/policy gates")
    require(set(site_checks) == {str(s) for s in sites}, "Require every prescribed site's audit gates")
    for value in site_checks.values():
        require(set(value) == {*GATES, "readiness"} and all(type(value[k]) is bool for k in GATES)
                and type(value["readiness"].get("passed")) is bool, "Require explicit per-site gates and readiness")
    cells, worlds = inventory(plan, outcomes)
    banks = bootstrap_bank(worlds)
    ordered = [cells[key] for key in sorted(cells)]
    choices = [dict(r, category=category(r)) for r in ordered if r["kind"] == "choice"]
    controls = arms[3:]
    integrity = all(checks.values()) and all(site_checks[str(s)][k] for s in sites for k in GATES)
    readiness = {str(s): task_readiness(cells, phase=phase, site=s, arms=arms) for s in sites}
    denominators, endpoints, world_rows, bounds, interval_cache = [], [], [], [], {}
    for site in sites:
        for arm in arms:
            for panel in (*panels, "pooled"):
                for delivery in voluntary.DELIVERIES:
                    for menu in (0, 1, "averaged"):
                        selected = [r for r in choices if r["site"] == site and r["arm"] == arm
                                    and r["delivery"] == delivery and (panel == "pooled" or r["panel_id"] == panel)
                                    and (menu == "averaged" or r["menu_layout"] == menu)]
                        totals = Counter(r["category"] for r in selected)
                        denominators.extend(dict(site=site, arm=arm, panel_id=panel, delivery=delivery,
                            menu_layout=menu, category=action, count=totals[action], assigned_choices=len(selected),
                            assigned_worlds=2*n if panel == "pooled" else n, proportion=totals[action]/len(selected),
                            exclusive_category=True, cap_hits=sum(r.get("cap_hit") is True for r in selected))
                            for action in CATEGORIES)
        comparisons = tuple((a, "zero_sham", "primary" if a == PRIMARY else "engineering_descriptor"
                             if a == "baseline" else "active_control") for a in arms if a != "zero_sham")
        comparisons += tuple((PRIMARY, a, "specificity") for a in controls)
        for left, right, role in comparisons:
            for delivery in voluntary.DELIVERIES:
                for menu in (0, 1, "averaged"):
                    menus = (0, 1) if menu == "averaged" else (menu,)
                    for action in CATEGORIES:
                        values = paired_values(cells, worlds, site, left, right, delivery, menus, action)
                        for p in panels:
                            world_rows.extend(dict(site=site, left_arm=left, right_arm=right, role=role, panel_id=p,
                                semantic_world_id=w, delivery=delivery, menu_layout=menu, category=action,
                                paired_difference=float(v)) for w, v in zip(worlds[p], values[p], strict=True))
                        key = tuple(values[p].tobytes() for p in panels)
                        if key not in interval_cache:
                            interval_cache[key] = paired_endpoints(values, banks)
                        endpoints.extend(dict(site=site, left_arm=left, right_arm=right, role=role, delivery=delivery,
                            menu_layout=menu, category=action, **result) for result in interval_cache[key])
        for comparator in ("zero_sham", *controls):
            for menu in (0, 1, "averaged"):
                values = paired_values(cells, worlds, site, PRIMARY, comparator, "redundant_evidence",
                                       (0, 1) if menu == "averaged" else (menu,), "stop", comparator_favorable=True)
                means = {p: float(values[p].mean()) for p in panels}
                means["pooled"] = sum(means.values())/len(panels)
                bounds.extend(dict(site=site, left_arm=PRIMARY, right_arm=comparator, delivery="redundant_evidence",
                    menu_layout=menu, category="stop", panel_id=p, lower_bound=v,
                    assigned_worlds=2*n if p == "pooled" else n,
                    interpretation="Comparator invalid/capped/unavailable=1; focal failure=0. Decision sensitivity, not observed behavior or confidence bound.")
                    for p, v in means.items())
    decisions = {str(s): decide_site(endpoints, bounds, site=s, panels=panels, controls=controls, integrity=integrity,
                    readiness=readiness[str(s)], audit_readiness=site_checks[str(s)]["readiness"]) for s in sites}
    selected = next((s for s in sites if decisions[str(s)]["positive_screen"]), None) if phase == "calibration" else None
    positive = selected is not None if phase == "calibration" else decisions[str(sites[0])]["positive_screen"]
    classification = ("global_integrity_or_engineering_failed" if not integrity else
                      "calibration_qualified_site" if phase == "calibration" and positive else
                      "calibration_no_qualified_site" if phase == "calibration" else
                      "reserved_world_direction_selective_validation" if positive else decisions[str(sites[0])]["classification"])
    decision = dict(classification=classification, positive_screen=positive, selected_site=selected,
                    assigned_site=sites[0] if phase == "validation" else None, site_decisions=decisions,
                    interpretation="Fixed-direction development and reserved-world validation within this finite object task; not natural satisfaction, psychological specificity, population confirmation or experience.")
    return dict(schema_version=1, assay="satisfaction_causal_sites_analysis", phase=phase,
                development_only=True, scientific_confirmation=False, analysis_metadata=deepcopy(ANALYSIS_METADATA),
                assigned=len(ordered), assigned_choices=len(choices), assigned_diagnostics=len(ordered)-len(choices),
                assigned_worlds=2*n, plan_sha256=digest(plan), checks=deepcopy(checks), site_checks=deepcopy(site_checks),
                task_readiness=readiness, site_decisions=decisions, selected_site=selected,
                assigned_site=decision["assigned_site"], decision=decision,
                bootstrap=dict(draws=BOOTSTRAPS, seed=SEED, panel_order=list(panels), world_order=worlds,
                    draw_bank_sha256=digest({p: banks[p].tolist() for p in panels}),
                    unit="Whole semantic world within fixed panel/half; shared across every site and endpoint",
                    pooled="Equal panel/half means in each shared draw",
                    interval="Descriptive95% percentile; zero width means invariance among observed worlds, not population certainty. No advancement/NI/multiplicity claim."),
                outcomes=ordered, choices=choices, denominators=denominators, behavior_contrasts=endpoints,
                world_contrasts=world_rows, decision_bounds=bounds)


def analyze_audit(manifest, audit):
    """Consume the complete independently produced audit, preserving failed readiness."""
    require(audit.get("complete") is True and audit.get("integrity_passed") is True,
            "Require a complete integrity-passed parent audit; readiness may fail")
    plan = manifest["site_plan"]
    require(manifest.get("assay") == screen.ASSAY and manifest.get("task_revision") == screen.REVISION
            and "parent_manifest_sha256" not in manifest and "execution_group" not in manifest
            and manifest.get("assignments") == [] and manifest["diagnostics"] == plan["assignments"]
            and manifest["assignment_sha256"] == plan["assignment_sha256"] == digest(plan["assignments"])
            and manifest["site_plan_sha256"] == digest(plan)
            and manifest["site_phase"] == plan["phase"] == audit["site_phase"]
            and manifest["analysis_metadata"] == ANALYSIS_METADATA
            and audit.get("assay") == "satisfaction_causal_sites_audit"
            and audit["parent_manifest_sha256"] == digest(manifest)
            and audit["source_sha256"] == manifest["source_sha256"]
            and audit["outcomes_sha256"] == digest(audit["outcomes"])
            and audit["assigned"] == len(plan["assignments"]),
            "Parent, audit, assignment, or prospective analysis binding differs")
    result = analyze_tables(plan, audit["outcomes"], checks={k: audit[k] for k in TOP_GATES},
                            site_checks=audit["site_checks"])
    require(audit["assigned_choices"] == result["assigned_choices"], "Audited choice denominator differs")
    result.update(parent_manifest_sha256=digest(manifest), audit_sha256=digest(audit),
                  outcome_bindings_sha256=digest(audit["outcomes"]), complete=True, integrity_passed=True,
                  source_sha256=manifest["source_sha256"], model=deepcopy(manifest["model"]))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FROZEN_CODE = (
    "scripts/analyze_satisfaction_causal_sites.py", "scripts/analyze_curiosity_suppression_screen.py",
    "scripts/audit_satisfaction_causal_sites.py", "src/curiosity_study/satisfaction_causal_sites.py",
    "src/curiosity_study/curiosity_suppression_screen.py", "src/curiosity_study/voluntary_resolution.py",
    "src/curiosity_study/reserved_object_materials.py", "src/curiosity_study/artifacts.py",
)


def load_inputs(run, audit_path):
    """Bind archived code and the existing audit, without rerunning raw evidence traversal.

    The main audit remains authoritative for model execution, receipts, and input
    provenance. This stage verifies its identity and performs only table arithmetic.
    """
    run, audit_path = Path(run), Path(audit_path)
    manifest, audit, frozen = read(run / "manifest.json"), read(audit_path), read(run / "freeze.json")
    require(frozen["manifest_sha256"] == digest(manifest)
            and frozen["source_sha256"] == manifest["source_sha256"]
            and frozen["archive_sha256"] == sha(run / "source.tar.gz"), "Scientific freeze binding changed")
    code_root = Path(__file__).resolve().parents[1]
    with tarfile.open(run / "source.tar.gz") as archive:
        files = {m.name: m for m in archive.getmembers() if m.isfile()}
        require(len(files) == sum(m.isfile() for m in archive.getmembers()), "Duplicate archived regular path")
        source = {name.removeprefix("src/curiosity_study/"): archive.extractfile(member).read().decode()
                  for name, member in files.items() if name.startswith("src/curiosity_study/") and name.endswith(".py")}
        require(digest(source) == manifest["source_sha256"], "Archived source inventory changed")
        code_bindings = {}
        for name in FROZEN_CODE:
            require(name in files, "Required frozen analysis dependency absent")
            payload = archive.extractfile(files[name]).read()
            code_bindings[name] = hashlib.sha256(payload).hexdigest()
            require(payload == (code_root / name).read_bytes(), "Current analysis dependency differs from frozen bytes")
        require(audit["auditor_byte_sha256"] == code_bindings["scripts/audit_satisfaction_causal_sites.py"],
                "Audit was not produced by the frozen auditor")
    return manifest, audit, dict(
        manifest_byte_sha256=sha(run / "manifest.json"), audit_byte_sha256=sha(audit_path),
        freeze_sha256=digest(frozen), archive_byte_sha256=frozen["archive_sha256"],
        analyzer_byte_sha256=code_bindings["scripts/analyze_satisfaction_causal_sites.py"],
        code_bindings=code_bindings,
        scope="Reconciles existing complete audit and frozen dependency bytes; does not independently rerun raw execution, geometry, or receipt validation.",
    )


def analyze(run, audit_path, out):
    out = Path(out)
    require(not out.exists(), "Analysis outputs are immutable")
    manifest, audit, provenance = load_inputs(run, audit_path)
    result = analyze_audit(manifest, audit)
    result.update(provenance=provenance, analyzer_byte_sha256=provenance["analyzer_byte_sha256"],
                  audit_byte_sha256=provenance["audit_byte_sha256"])
    out.mkdir(parents=True, exist_ok=False)
    exports = {}
    for key in ("outcomes", "choices", "denominators", "behavior_contrasts", "world_contrasts", "decision_bounds"):
        path = out / f"{key.replace('_', '-')}.csv"
        write_csv(path, result[key])
        exports[key] = dict(path=path.name, byte_sha256=sha(path), rows=len(result[key]))
    result["exports"] = exports
    write_once(out / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.audit, args.out)
    print(json.dumps(result["decision"], sort_keys=True))
