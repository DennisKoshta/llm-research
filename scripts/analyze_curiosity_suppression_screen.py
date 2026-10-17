"""Fixed signed-curiosity development screen; all controls and assigned outcomes retained."""

import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once

PANELS = tuple(name for name, _ in screen.PANELS)
ARMS, CONTROLS, PRIMARY = screen.ARMS, screen.CONTROLS, screen.PRIMARY
ASSIGNED = 288 * len(ARMS)
CHOICE_COUNT = DIAGNOSTIC_COUNT = ASSIGNED // 2
CATEGORIES = (*voluntary.SERVICES, "invalid", "capped", "execution_failure", "missing")
NONREPORT = "valid_nonreport"
ENDPOINTS = (*CATEGORIES, NONREPORT)
METADATA = (
    "trial_id",
    "base_trial_id",
    "panel_id",
    "semantic_world_id",
    "world_slot",
    "arm",
    "family",
    "kind",
    "delivery",
    "menu_layout",
)
GATES = (
    "sham_identity_passed",
    "engagement_passed",
    "receipts_passed",
    "cross_arm_prepatch_identity_passed",
    "worker_policies_equal",
)
BOOTSTRAPS, SEED = 10000, 2026101440
ANALYSIS_METADATA = dict(
    schema_version=1,
    bootstrap_samples=BOOTSTRAPS,
    bootstrap_seed=SEED,
    bootstrap_unit="whole_semantic_world_within_panel",
    arms=list(ARMS),
    primary_arm=PRIMARY,
    comparator="zero_sham",
    primary_delivery="redundant_evidence",
    primary_category=NONREPORT,
    minimum_panel_difference=0.25,
    each_pooled_menu_difference_strictly_positive=True,
    specificity_controls=list(CONTROLS),
    minimum_each_control_panel_difference=0.125,
    each_control_pooled_menu_difference_nonnegative=True,
    comparator_favorable_invalidity_bound_required=True,
    all_arm_readiness_required=True,
    interval_controls_advancement=False,
    natural_suppression_claim=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(assignments, outcomes):
    """Check complete crossing and world identity before inspecting responses."""
    require(len(assignments) == ASSIGNED, "Require the complete combined parent, not a child or subset")
    assigned = {a["trial_id"]: a for a in assignments}
    observed = {r["trial_id"]: r for r in outcomes}
    require(
        len(assigned) == len(observed) == len(outcomes) == ASSIGNED and set(assigned) == set(observed),
        "Missing, duplicate or extra assigned outcomes",
    )
    cells, base, slots = {}, {}, defaultdict(dict)
    for a in assignments:
        require(
            a["panel_id"] in PANELS
            and a["arm"] in ARMS
            and a["family"] == "objects"
            and a["kind"] in ("choice", *voluntary.DIAGNOSTICS)
            and a["delivery"] in voluntary.DELIVERIES
            and type(a["menu_layout"]) is int
            and a["menu_layout"] in (0, 1)
            and type(a["world_slot"]) is int
            and a["world_slot"] in range(12),
            "Unknown panel/arm/family/task/cell",
        )
        panel, world = a["panel_id"], a["semantic_world_id"]
        slot = slots[panel].setdefault(world, a["world_slot"])
        require(slot == a["world_slot"], "Semantic world merged across slots")
        key = (panel, world, a["arm"], a["kind"], a["delivery"], a["menu_layout"])
        require(key not in cells, "Duplicate semantic-world cell")
        r = deepcopy(observed[a["trial_id"]])
        require(all(r.get(k) == a[k] for k in METADATA), "Outcome assignment metadata differs")
        require(
            r["status"] in ("completed", "execution_error", "execution_failure", "missing"),
            "Unknown outcome status",
        )
        if r["status"] == "completed":
            require(
                all(type(r.get(k)) is bool for k in ("valid", "thinking_closed", "cap_hit")),
                "Completed outcomes require observed boolean validity/closure/cap fields",
            )
            require(not r["valid"] or r["thinking_closed"], "A valid thinking response must be closed")
            if a["kind"] == "choice":
                require(r.get("correct") is None, "Optional choices have no correctness label")
                require(
                    (
                        r["selected_service"] in voluntary.SERVICES
                        if r["valid"]
                        else r["selected_service"] is None
                    ),
                    "Invalid choices cannot execute a parsed service; valid choices need a service",
                )
            else:
                require(type(r.get("correct")) is bool, "Diagnostics need observed boolean correctness")
                require(not r["correct"] or r["valid"], "Invalid diagnostics cannot be correct")
        else:
            require(
                r.get("valid") in (None, False)
                and r.get("correct") in (None, False)
                and r.get("selected_service") is None,
                "Unavailable rows cannot fabricate valid responses, correct answers or services",
            )
        cells[key] = r
        stable = {k: a[k] for k in METADATA if k not in ("trial_id", "arm")}
        require(base.setdefault(a["base_trial_id"], stable) == stable, "Base branch differs across arms")
    worlds = {panel: sorted(slots[panel]) for panel in PANELS}
    require(
        len(base) == 288
        and all(len(values) == 12 for values in worlds.values())
        and all(set(slots[p].values()) == set(range(12)) for p in PANELS)
        and len(set().union(*(set(values) for values in worlds.values()))) == 24,
        "Require24 distinct semantic worlds and twelve unique slots in each panel",
    )
    expected = {
        (panel, world, arm, "choice", delivery, menu)
        for panel in PANELS
        for world in worlds[panel]
        for arm in ARMS
        for delivery in voluntary.DELIVERIES
        for menu in (0, 1)
    }
    expected.update(
        (
            panel,
            world,
            arm,
            kind,
            "decisive_evidence" if kind in ("identify_resolved", "repeated_fact") else "redundant_evidence",
            0,
        )
        for panel in PANELS
        for world in worlds[panel]
        for arm in ARMS
        for kind in voluntary.DIAGNOSTICS
    )
    require(
        set(cells) == expected, "Every assigned world needs all arms, deliveries, menus and six diagnostics"
    )
    return cells, worlds


def category(row):
    if row["status"] == "missing":
        return "missing"
    if row["status"] in ("execution_error", "execution_failure"):
        return "execution_failure"
    if row["cap_hit"]:
        return "capped"
    if row["valid"] is not True or row["thinking_closed"] is not True:
        return "invalid"
    return row["selected_service"]


def counts(rows):
    completed = [r for r in rows if r["status"] == "completed"]
    return dict(
        assigned=len(rows),
        observed=len(completed),
        valid=sum(r["valid"] and r["thinking_closed"] for r in completed),
        correct=sum(r.get("correct") is True for r in completed),
        cap_hits=sum(r["cap_hit"] for r in completed),
        unclosed=sum(not r["thinking_closed"] for r in completed),
        missing=sum(r["status"] == "missing" for r in rows),
        execution_failures=sum(r["status"] in ("execution_error", "execution_failure") for r in rows),
    )


def task_readiness(cells):
    diagnostic_cells, choice_cells = [], []
    values = list(cells.values())
    for arm in ARMS:
        for panel in PANELS:
            group = [r for r in values if r["arm"] == arm and r["panel_id"] == panel]
            for kind in voluntary.DIAGNOSTICS:
                result = counts([r for r in group if r["kind"] == kind])
                passed = (
                    result["assigned"] == result["observed"] == result["valid"] == 12
                    and result["correct"] >= 11
                    and result["cap_hits"] == 0
                )
                diagnostic_cells.append(dict(arm=arm, panel_id=panel, kind=kind, **result, passed=passed))
            for delivery in voluntary.DELIVERIES:
                result = counts([r for r in group if r["kind"] == "choice" and r["delivery"] == delivery])
                result["correct"] = None
                passed = (
                    result["assigned"] == result["observed"] == 24
                    and result["valid"] >= 23
                    and result["cap_hits"] == 0
                )
                choice_cells.append(dict(arm=arm, panel_id=panel, delivery=delivery, **result, passed=passed))
    return dict(
        passed=all(r["passed"] for r in diagnostic_cells + choice_cells),
        diagnostic_cells=diagnostic_cells,
        choice_cells=choice_cells,
        interpretation="Passed observed-panel gates do not establish noninferiority or unchanged confidence/uncertainty.",
    )


def bootstrap_bank(worlds):
    rng = np.random.default_rng(SEED)
    return {panel: rng.integers(0, 12, size=(BOOTSTRAPS, 12)) for panel in PANELS}


def paired_endpoints(values, banks):
    draws = {p: values[p][banks[p]].mean(axis=1) for p in PANELS}
    estimates = {p: float(values[p].mean()) for p in PANELS}
    draws["pooled"] = (draws[PANELS[0]] + draws[PANELS[1]]) / 2
    estimates["pooled"] = (estimates[PANELS[0]] + estimates[PANELS[1]]) / 2
    return [
        dict(
            panel_id=p,
            estimate=estimates[p],
            ci95=np.quantile(draws[p], [0.025, 0.975]).tolist(),
            assigned_worlds=24 if p == "pooled" else 12,
            empirical_interval_degenerate=bool(np.ptp(draws[p]) == 0),
        )
        for p in (*PANELS, "pooled")
    ]


def indicator(row, endpoint, *, unavailable_as_alternative=False):
    observed = category(row)
    if endpoint != NONREPORT:
        return float(observed == endpoint)
    if observed in ("repeat", "pattern", "stop"):
        return 1.0
    if unavailable_as_alternative and observed in ("invalid", "capped", "execution_failure", "missing"):
        return 1.0
    return 0.0


def paired_values(cells, worlds, left, right, delivery, menus, endpoint, *, comparator_favorable=False):
    return {
        panel: np.array(
            [
                np.mean(
                    [
                        indicator(cells[panel, world, left, "choice", delivery, m], endpoint)
                        - indicator(
                            cells[panel, world, right, "choice", delivery, m],
                            endpoint,
                            unavailable_as_alternative=comparator_favorable,
                        )
                        for m in menus
                    ]
                )
                for world in worlds[panel]
            ],
            dtype=float,
        )
        for panel in PANELS
    }


def effect_summary(rows, left, right, *, bound=False):
    selected = [
        r
        for r in rows
        if r["left_arm"] == left
        and r["right_arm"] == right
        and r["category"] == NONREPORT
        and r["delivery"] == "redundant_evidence"
    ]
    require(len(selected) == 9, "Require complete fixed panel/menu effect inventory")
    keys = {(r["panel_id"], r["menu_layout"]) for r in selected}
    require(
        keys == {(p, m) for p in (*PANELS, "pooled") for m in (0, 1, "averaged")},
        "Duplicate or absent panel/menu effect",
    )
    field = "lower_bound" if bound else "estimate"
    require(all(np.isfinite(r[field]) for r in selected), "Nonfinite decision effect")
    return dict(
        panel_differences={r["panel_id"]: r[field] for r in selected if r["menu_layout"] == "averaged"},
        pooled_menu_differences={
            str(r["menu_layout"]): r[field]
            for r in selected
            if r["panel_id"] == "pooled" and r["menu_layout"] in (0, 1)
        },
    )


def threshold(summary, *, specificity=False):
    margin = 0.125 if specificity else 0.25
    return all(summary["panel_differences"][p] >= margin for p in PANELS) and all(
        value >= 0 if specificity else value > 0 for value in summary["pooled_menu_differences"].values()
    )


def decide(endpoints, bounds, checks, readiness, audit_readiness):
    primary = effect_summary(endpoints, PRIMARY, "zero_sham")
    primary_bound = effect_summary(bounds, PRIMARY, "zero_sham", bound=True)
    controls = {}
    for control in CONTROLS:
        observed = effect_summary(endpoints, PRIMARY, control)
        worst_case = effect_summary(bounds, PRIMARY, control, bound=True)
        controls[control] = dict(
            observed=observed,
            comparator_favorable=worst_case,
            observed_threshold_met=threshold(observed, specificity=True),
            bounded_threshold_met=threshold(worst_case, specificity=True),
        )
    effect_passed = threshold(primary) and threshold(primary_bound)
    specificity_passed = all(
        v["observed_threshold_met"] and v["bounded_threshold_met"] for v in controls.values()
    )
    if not all(checks.values()):
        classification = "engineering_or_receipt_gate_failed"
    elif not readiness["passed"] or not audit_readiness["passed"]:
        classification = "readiness_failed"
    elif not effect_passed:
        classification = "no_substantial_primary_development_effect"
    elif not specificity_passed:
        classification = "primary_effect_without_predefined_specificity"
    else:
        classification = "direction_selective_development_signal"
    return dict(
        classification=classification,
        positive_screen=classification == "direction_selective_development_signal",
        effect_threshold_met=effect_passed,
        primary_observed_threshold_met=threshold(primary),
        primary_comparator_favorable_threshold_met=threshold(primary_bound),
        specificity_threshold_met=specificity_passed,
        primary_observed=primary,
        primary_comparator_favorable=primary_bound,
        controls=controls,
        controls_with_substantial_observed_alternative_increase=[
            c for c in CONTROLS if threshold(effect_summary(endpoints, c, "zero_sham"))
        ],
        rule="C-negative minus sham valid alternatives: >=0.25 in each panel and >0 in both pooled menus; C-negative minus every active control: >=0.125 in each panel and >=0 in both pooled menus. All rules must also survive comparator-favorable invalidity bounds, and every audited/task gate must pass.",
        stopping="No automatic expansion, candidate/layer/sign/dose/prompt rescue, or reinterpretation of the closed satisfaction screen.",
        interpretation="Direction-selective development among these exact signed perturbations only; not natural curiosity suppression, psychological specificity, preserved confidence, population confirmation or inner experience.",
    )


def analyze_tables(assignments, outcomes, *, checks, audit_readiness):
    require(
        set(checks) == set(GATES) and all(type(v) is bool for v in checks.values()),
        "Require explicit audited engineering/receipt gates",
    )
    require(type(audit_readiness.get("passed")) is bool, "Require the original audit readiness result")
    cells, worlds = inventory(assignments, outcomes)
    banks = bootstrap_bank(worlds)
    ordered = [cells[key] for key in sorted(cells)]
    choices = [dict(r, category=category(r)) for r in ordered if r["kind"] == "choice"]
    readiness = task_readiness(cells)
    denominators = []
    for arm in ARMS:
        for panel in (*PANELS, "pooled"):
            for delivery in voluntary.DELIVERIES:
                for menu in (0, 1, "averaged"):
                    selected = [
                        r
                        for r in choices
                        if r["arm"] == arm
                        and r["delivery"] == delivery
                        and (panel == "pooled" or r["panel_id"] == panel)
                        and (menu == "averaged" or r["menu_layout"] == menu)
                    ]
                    totals = Counter(r["category"] for r in selected)
                    for action in ENDPOINTS:
                        number = (
                            sum(totals[k] for k in ("repeat", "pattern", "stop"))
                            if action == NONREPORT
                            else totals[action]
                        )
                        denominators.append(
                            dict(
                                arm=arm,
                                panel_id=panel,
                                delivery=delivery,
                                menu_layout=menu,
                                category=action,
                                count=number,
                                assigned_choices=len(selected),
                                assigned_worlds=24 if panel == "pooled" else 12,
                                proportion=number / len(selected),
                                exclusive_category=action != NONREPORT,
                                cap_hits=sum(r.get("cap_hit") is True for r in selected),
                            )
                        )
    comparisons = tuple(
        (
            arm,
            "zero_sham",
            "primary"
            if arm == PRIMARY
            else "engineering_descriptor"
            if arm == "baseline"
            else "active_control",
        )
        for arm in ARMS
        if arm != "zero_sham"
    ) + tuple((PRIMARY, arm, "specificity") for arm in CONTROLS)
    endpoints, world_rows, interval_cache = [], [], {}
    for left, right, role in comparisons:
        for delivery in voluntary.DELIVERIES:
            for menu in (0, 1, "averaged"):
                menus = (0, 1) if menu == "averaged" else (menu,)
                for action in ENDPOINTS:
                    values = paired_values(cells, worlds, left, right, delivery, menus, action)
                    for panel in PANELS:
                        world_rows.extend(
                            dict(
                                left_arm=left,
                                right_arm=right,
                                role=role,
                                panel_id=panel,
                                semantic_world_id=world,
                                delivery=delivery,
                                menu_layout=menu,
                                category=action,
                                paired_difference=float(value),
                            )
                            for world, value in zip(worlds[panel], values[panel], strict=True)
                        )
                    cache_key = tuple(values[p].tobytes() for p in PANELS)
                    if cache_key not in interval_cache:
                        interval_cache[cache_key] = paired_endpoints(values, banks)
                    endpoints.extend(
                        dict(
                            left_arm=left,
                            right_arm=right,
                            role=role,
                            delivery=delivery,
                            menu_layout=menu,
                            category=action,
                            **result,
                        )
                        for result in interval_cache[cache_key]
                    )
    bounds = []
    for comparator in ("zero_sham", *CONTROLS):
        for menu in (0, 1, "averaged"):
            menus = (0, 1) if menu == "averaged" else (menu,)
            values = paired_values(
                cells,
                worlds,
                PRIMARY,
                comparator,
                "redundant_evidence",
                menus,
                NONREPORT,
                comparator_favorable=True,
            )
            means = {p: float(values[p].mean()) for p in PANELS}
            means["pooled"] = sum(means.values()) / len(PANELS)
            bounds.extend(
                dict(
                    left_arm=PRIMARY,
                    right_arm=comparator,
                    delivery="redundant_evidence",
                    menu_layout=menu,
                    category=NONREPORT,
                    panel_id=p,
                    lower_bound=value,
                    assigned_worlds=24 if p == "pooled" else 12,
                    interpretation="Comparator invalid/capped/unavailable=1; target=0. Decision sensitivity, not observed behavior or an inferential confidence bound.",
                )
                for p, value in means.items()
            )
    decision = decide(endpoints, bounds, checks, readiness, audit_readiness)
    return dict(
        schema_version=1,
        assay="curiosity_suppression_screen_analysis",
        development_only=True,
        scientific_confirmation=False,
        analysis_metadata=deepcopy(ANALYSIS_METADATA),
        assigned=ASSIGNED,
        assigned_choices=CHOICE_COUNT,
        assigned_diagnostics=DIAGNOSTIC_COUNT,
        assigned_worlds=24,
        checks=checks,
        audit_readiness=deepcopy(audit_readiness),
        task_readiness=readiness,
        decision=decision,
        bootstrap=dict(
            draws=BOOTSTRAPS,
            seed=SEED,
            panel_order=list(PANELS),
            world_order=worlds,
            draw_bank_sha256=digest({p: banks[p].tolist() for p in PANELS}),
            unit="Whole semantic world within each panel; all arms/deliveries/menus/diagnostics retained together",
            pooled="Equal mean of panel means within shared draws",
            interval="Descriptive95% percentile; intervals do not control advancement or establish noninferiority",
        ),
        outcomes=ordered,
        choices=choices,
        denominators=denominators,
        behavior_contrasts=endpoints,
        world_contrasts=world_rows,
        decision_bounds=bounds,
    )


def analyze_audit(manifest, audit):
    require(
        audit.get("complete") is True and audit.get("integrity_passed") is True,
        "Require a complete integrity-passed parent audit; readiness may fail",
    )
    require(
        manifest.get("assay") == "curiosity_suppression_screen"
        and manifest.get("task_revision") == screen.REVISION
        and audit.get("assay") == "curiosity_suppression_screen_audit"
        and audit.get("parent_manifest_sha256") == digest(manifest)
        and manifest.get("analysis_metadata") == ANALYSIS_METADATA,
        "Parent identity or prospective analysis contract differs",
    )
    assignments = (manifest.get("assignments") or []) + (manifest.get("diagnostics") or [])
    require(digest(assignments) == manifest["assignment_sha256"], "Parent assignment binding changed")
    result = analyze_tables(
        assignments,
        audit["outcomes"],
        checks={k: audit[k] for k in GATES},
        audit_readiness=audit["readiness"],
    )
    result.update(
        parent_manifest_sha256=digest(manifest),
        audit_sha256=digest(audit),
        outcome_bindings_sha256=digest(audit["outcomes"]),
        complete=True,
        integrity_passed=True,
        source_sha256=manifest["source_sha256"],
        model=manifest["model"],
        analyzer_byte_sha256=sha(__file__),
    )
    return result


def load_inputs(run, audit_path):
    """Reproduce the bound combined audit before reading its compact measurements."""
    from audit_curiosity_suppression_screen import audit as audit_parent

    run, audit_path = Path(run), Path(audit_path)
    manifest, report = read(run / "manifest.json"), read(audit_path)
    require(audit_parent(run) == report, "Combined audit no longer reproduces from bound child evidence")
    return manifest, report


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def analyze(run, audit_path, out):
    manifest, audit = load_inputs(run, audit_path)
    result = analyze_audit(manifest, audit)
    result["audit_byte_sha256"] = sha(audit_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    exports = {}
    for key in (
        "outcomes",
        "choices",
        "denominators",
        "behavior_contrasts",
        "world_contrasts",
        "decision_bounds",
    ):
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
