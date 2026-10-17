"""Independent assigned-denominator, readiness and finite-site decision arithmetic.

This checker imports no task scorer, study constructor, analyzer or GPU auditor.
It checks compact outcomes supplied by the separately bound complete audit; it
does not certify raw responses, neural geometry, receipt delivery or intervals.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from curiosity_study.artifacts import digest, read, write_once

PRIMARY = "satisfaction_positive"
CALIBRATION_ARMS = ("baseline", "zero_sham", PRIMARY, "confidence_positive", "completion_positive",
                    "valence_positive", "random_01")
VALIDATION_ARMS = ("baseline", "zero_sham", PRIMARY, "confidence_positive", "completion_positive",
    "valence_positive", "valence_negative", "motivation_negative", "novelty_negative", "task_demand_negative",
    "random_01", "random_02", "random_03", "random_04", "random_05")
DIAGNOSTICS = ("identify_resolved", "new_fact", "repeated_fact", "pattern_independence",
               "instrument_useful_action", "action_consequence")
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
GATES = ("sham_identity_passed", "engagement_passed", "receipts_passed",
         "cross_arm_prepatch_identity_passed", "worker_policies_equal")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def totals(rows, *, choice=False):
    observed = [row for row in rows if row["status"] == "completed"]
    return dict(assigned=len(rows), observed=len(observed),
        valid=sum(row["valid"] is True and row["thinking_closed"] is True for row in observed),
        correct=None if choice else sum(row["correct"] is True for row in observed),
        cap_hits=sum(row["cap_hit"] is True for row in observed),
        unclosed=sum(row["thinking_closed"] is not True for row in observed),
        missing=sum(row["status"] == "missing" for row in rows),
        execution_failures=sum(row["status"] in ("execution_error", "execution_failure") for row in rows))


def recompute(plan, outcomes, *, checks, site_checks):
    """Use exact rational sums for all integer-denominator decision contrasts."""
    phase = plan["phase"]
    require(phase in ("calibration", "validation"), "Unknown phase")
    calibration = phase == "calibration"
    sites, arms = tuple(plan["sites"]), CALIBRATION_ARMS if calibration else VALIDATION_ARMS
    panels, n = (("panel_1", "panel_2"), 12) if calibration else (("half_1", "half_2"), 8)
    require((sites == (31, 47) if calibration else len(sites) == 1 and sites[0] in (31, 47))
            and tuple(plan["arms"]) == arms and digest(plan["assignments"]) == plan["assignment_sha256"],
            "Prescribed site/arm/assignment grid changed")
    assignments = {row["trial_id"]: row for row in plan["assignments"]}
    observed = {row["trial_id"]: row for row in outcomes}
    require(len(assignments) == len(plan["assignments"]) == len(observed) == len(outcomes)
            == (4032 if calibration else 2880) and set(assignments) == set(observed),
            "Missing, repeated or additional assigned rows")
    worlds, cells, slots, ordinals, base_ids = defaultdict(set), {}, {}, {}, {}
    for trial_id, assignment in assignments.items():
        row = observed[trial_id]
        fields = ("base_trial_id", "panel_id", "semantic_world_id", "world_slot", "arm", "family",
                  "kind", "delivery", "menu_layout", "phase", "site")
        require(all(row.get(key) == assignment[key] for key in fields), "Outcome identity changed")
        require(type(row["site"]) is int and row["site"] in sites and row["panel_id"] in panels
                and row["arm"] in arms and row["family"] == "objects" and row["phase"] == phase
                and row["kind"] in ("choice", *DIAGNOSTICS) and row["delivery"] in DELIVERIES
                and type(row["menu_layout"]) is int and row["menu_layout"] in (0, 1)
                and type(row["world_slot"]) is int, "Unknown world condition")
        world_key = (row["panel_id"], row["semantic_world_id"])
        require(slots.setdefault(world_key, row["world_slot"]) == row["world_slot"], "World source slot changed")
        variant = tuple(row[key] for key in ("kind", "delivery", "menu_layout"))
        require(base_ids.setdefault((*world_key, *variant), row["base_trial_id"]) == row["base_trial_id"],
                "Base identity changed across arms/sites")
        if not calibration:
            require(type(assignment.get("reserved_ordinal")) is int
                    and row.get("reserved_ordinal") == assignment["reserved_ordinal"]
                    and ordinals.setdefault(row["semantic_world_id"], row["reserved_ordinal"])
                    == row["reserved_ordinal"], "Reserved ordinal changed")
        require(row["status"] in ("completed", "missing", "execution_failure", "execution_error"),
                "Unknown disposition")
        if row["status"] == "completed":
            require(all(type(row[key]) is bool for key in ("valid", "thinking_closed", "cap_hit")),
                    "Completed outcomes need observed booleans")
            require(not row["valid"] or row["thinking_closed"], "Valid responses require closure")
            if row["kind"] == "choice":
                require(row.get("correct") is None and
                        (row.get("selected_service") in ("report", "repeat", "pattern", "stop")
                         if row["valid"] else row.get("selected_service") is None),
                        "Unknown selected service or invalid choice success")
            else:
                require(type(row.get("correct")) is bool and (not row["correct"] or row["valid"])
                        and row.get("selected_service") is None, "Malformed diagnostic outcome")
        else:
            require(row.get("valid") in (None, False) and row.get("correct") in (None, False)
                    and row.get("selected_service") is None, "Unavailable row fabricates a success")
        worlds[row["panel_id"]].add(row["semantic_world_id"])
        key = tuple(row[key] for key in ("site", "panel_id", "semantic_world_id", "arm", "kind", "delivery", "menu_layout"))
        require(key not in cells, "Repeated world condition")
        cells[key] = row
    require(set(worlds) == set(panels) and all(len(worlds[p]) == n for p in panels)
            and len(worlds[panels[0]] | worlds[panels[1]]) == 2*n, "Incomplete distinct-world halves")
    expected_slots = set(range(12)) if calibration else {0, 1, 4, 5, 6, 7, 10, 11}
    require(all({slots[p, w] for w in worlds[p]} == expected_slots for p in panels), "Changed source slots")
    variants = {("choice", d, m) for d in DELIVERIES for m in (0, 1)}
    variants |= {(k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact")
                  else "redundant_evidence", 0) for k in DIAGNOSTICS}
    expected_cells = {(s, p, w, a, *v) for s in sites for p in panels for w in worlds[p]
                      for a in arms for v in variants}
    require(set(cells) == expected_cells and len(set(base_ids.values())) == 2*n*12,
            "Every world requires every arm/site and all twelve distinct base variants")
    if not calibration:
        ordered = sorted(worlds[panels[0]] | worlds[panels[1]])
        require(sorted(worlds[panels[0]]) == ordered[:8] and sorted(worlds[panels[1]]) == ordered[8:]
                and all(ordinals[w] == i for i, w in enumerate(ordered)),
                "Validation halves are not the fixed lexical split")
    require(set(checks) == {"complete", "integrity_passed", "worker_policies_equal"}
            and all(type(value) is bool for value in checks.values())
            and set(site_checks) == {str(site) for site in sites}, "Missing whole-grid or site gates")
    for site in sites:
        require(set(site_checks[str(site)]) == {*GATES, "readiness"}
                and all(type(site_checks[str(site)][key]) is bool for key in GATES)
                and type(site_checks[str(site)]["readiness"]["passed"]) is bool, "Malformed site gates")
    global_passed = all(checks.values()) and all(site_checks[str(s)][key] for s in sites for key in GATES)
    readiness, decisions = {}, {}

    def stopped(row, *, comparator_bound=False):
        usable = (row["status"] == "completed" and row["valid"] is True
                  and row["thinking_closed"] is True and row["cap_hit"] is False)
        return int(row.get("selected_service") == "stop") if usable else int(comparator_bound)

    def difference(site, comparator, *, panel, menu, bounded):
        terms = []
        for p in panels if panel == "pooled" else (panel,):
            for world in sorted(worlds[p]):
                for m in (0, 1) if menu == "averaged" else (menu,):
                    left = cells[site, p, world, PRIMARY, "choice", "redundant_evidence", m]
                    right = cells[site, p, world, comparator, "choice", "redundant_evidence", m]
                    terms.append(stopped(left) - stopped(right, comparator_bound=bounded))
        return float(Fraction(sum(terms), len(terms)))

    def summary(site, comparator, bounded):
        return dict(panel_differences={p: difference(site, comparator, panel=p, menu="averaged", bounded=bounded)
                                       for p in (*panels, "pooled")},
                    pooled_menu_differences={str(m): difference(site, comparator, panel="pooled", menu=m, bounded=bounded)
                                            for m in (0, 1)})

    def threshold(value, *, rival=False):
        return (all(value["panel_differences"][p] >= (0.125 if rival else 0.25) for p in panels)
                and all(v >= 0 if rival else v > 0 for v in value["pooled_menu_differences"].values()))

    for site in sites:
        diag_cells, choice_cells = [], []
        groups, count = (panels, 12) if calibration else (("all_reserved",), 16)
        for arm in arms:
            for panel in groups:
                rows = [r for r in outcomes if r["site"] == site and r["arm"] == arm
                        and (panel == "all_reserved" or r["panel_id"] == panel)]
                for kind in DIAGNOSTICS:
                    values = totals([r for r in rows if r["kind"] == kind])
                    passed = values["assigned"] == values["observed"] == values["valid"] == count
                    passed = passed and values["correct"] >= count-1 and values["cap_hits"] == 0
                    diag_cells.append(dict(site=site, arm=arm, panel_id=panel, kind=kind, **values, passed=passed))
                for delivery in DELIVERIES:
                    values = totals([r for r in rows if r["kind"] == "choice" and r["delivery"] == delivery], choice=True)
                    passed = (values["assigned"] == values["observed"] == 2*count
                              and values["valid"] >= 2*count-1 and values["cap_hits"] == 0)
                    choice_cells.append(dict(site=site, arm=arm, panel_id=panel, delivery=delivery, **values, passed=passed))
        ready = all(cell["passed"] for cell in diag_cells + choice_cells)
        readiness[str(site)] = dict(passed=ready, diagnostic_cells=diag_cells, choice_cells=choice_cells)
        primary, lower = summary(site, "zero_sham", False), summary(site, "zero_sham", True)
        controls = {}
        for arm in arms[3:]:
            value, bound = summary(site, arm, False), summary(site, arm, True)
            controls[arm] = dict(observed=value, comparator_favorable=bound,
                                 observed_threshold_met=threshold(value, rival=True),
                                 bounded_threshold_met=threshold(bound, rival=True))
        effect = threshold(primary) and threshold(lower)
        specificity = all(v["observed_threshold_met"] and v["bounded_threshold_met"] for v in controls.values())
        audited_ready = site_checks[str(site)]["readiness"]["passed"]
        label = ("global_integrity_or_engineering_failed" if not global_passed else "readiness_failed"
                 if not ready or not audited_ready else "no_substantial_primary_development_effect"
                 if not effect else "primary_effect_without_predefined_specificity" if not specificity
                 else "direction_selective_development_signal")
        decisions[str(site)] = dict(classification=label, positive_screen=label == "direction_selective_development_signal",
            effect_threshold_met=effect, primary_observed_threshold_met=threshold(primary),
            primary_comparator_favorable_threshold_met=threshold(lower), specificity_threshold_met=specificity,
            primary_observed=primary, primary_comparator_favorable=lower, controls=controls,
            recomputed_readiness_passed=ready, audit_readiness_passed=audited_ready)
    selected = next((site for site in (31, 47) if decisions[str(site)]["positive_screen"]), None) if calibration else None
    positive = selected is not None if calibration else decisions[str(sites[0])]["positive_screen"]
    label = ("global_integrity_or_engineering_failed" if not global_passed else "calibration_qualified_site"
             if calibration and positive else "calibration_no_qualified_site" if calibration
             else "reserved_world_direction_selective_validation" if positive else decisions[str(sites[0])]["classification"])
    return dict(decision=dict(classification=label, positive_screen=positive, selected_site=selected,
                             assigned_site=None if calibration else sites[0], site_decisions=decisions),
                task_readiness=readiness)


def compare(expected, actual, *, path="root"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), f"Missing fields at{path}")
        return sum(compare(value, actual[key], path=f"{path}.{key}") for key, value in expected.items())
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Changed row count at{path}")
        return sum(compare(a, b, path=f"{path}[{i}]") for i, (a, b) in enumerate(zip(expected, actual, strict=True)))
    if type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12), f"Arithmetic mismatch at{path}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Decision mismatch at{path}")
    return 1


def check(run, analysis_path, audit_path):
    run, analysis_path, audit_path = Path(run), Path(analysis_path), Path(audit_path)
    manifest, frozen, analysis, audit = read(run / "manifest.json"), read(run / "freeze.json"), read(analysis_path), read(audit_path)
    require(frozen["manifest_sha256"] == digest(manifest) == analysis["parent_manifest_sha256"]
            == audit["parent_manifest_sha256"] and sha(run / "source.tar.gz") == frozen["archive_sha256"]
            and analysis["audit_sha256"] == digest(audit) and audit["complete"] is True,
            "Require the bound complete parent audit and analysis")
    gates = {key: audit[key] for key in ("complete", "integrity_passed", "worker_policies_equal")}
    require(analysis["checks"] == gates and analysis["site_checks"] == audit["site_checks"], "Audited gates changed")
    expected = recompute(manifest["site_plan"], audit["outcomes"], checks=gates, site_checks=audit["site_checks"])
    compared = compare(expected["decision"], analysis["decision"])
    compared += compare(expected["task_readiness"], analysis["task_readiness"])
    return dict(schema_version=1, assay="satisfaction_site_independent_decision_check", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), checker_byte_sha256=sha(__file__),
        parent_manifest_sha256=digest(manifest), analysis_sha256=digest(analysis), analysis_byte_sha256=sha(analysis_path),
        audit_sha256=digest(audit), audit_byte_sha256=sha(audit_path), assigned=len(audit["outcomes"]),
        compared_values=compared, decision=analysis["decision"], recomputed_decision=expected["decision"],
        scope="Independent exact-rational STOP contrasts, conservative invalidity bounds, every task-readiness count, whole-grid versus site gates, and earliest qualifying site/one-validation decision. No analyzer functions imported.",
        excluded_scope="Raw response parsing, source and GPU evidence, receipt execution, secondary outcome tables, exported CSVs and descriptive bootstrap intervals remain outside this decision check.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Independent decision receipt is immutable")
    result = check(args.run, args.analysis, args.audit)
    write_once(args.out, result)
    print(json.dumps({key: result[key] for key in ("passed", "assigned", "compared_values", "decision")}))
