"""Fixed object-only development screen; all outcomes retained, no fitting or rescue."""

import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from curiosity_study import satisfaction_screen as screen
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once

PANELS = tuple(name for name, _ in screen.PANELS)
ARMS = tuple(name for name, _ in screen.ARMS)
PRIMARY, SENSITIVITY = ARMS[3], ARMS[2]
CATEGORIES = (*voluntary.SERVICES, "invalid", "capped", "execution_failure", "missing")
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
GATES = ("sham_identity_passed", "engagement_passed", "receipts_passed")
BOOTSTRAPS, SEED = 10000, 2026101438
ANALYSIS_METADATA = dict(
    schema_version=1,
    bootstrap_samples=BOOTSTRAPS,
    bootstrap_seed=SEED,
    bootstrap_unit="whole_semantic_world_within_panel",
    primary_arm=PRIMARY,
    comparator="zero_sham",
    primary_delivery="redundant_evidence",
    primary_category="valid_uncapped_stop",
    minimum_panel_difference=0.25,
    each_pooled_menu_difference_strictly_positive=True,
    sensitivity_arm=SENSITIVITY,
    sensitivity_can_rescue=False,
    interval_controls_advancement=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(assignments, outcomes):
    """Check complete crossing and world identity before inspecting responses."""
    require(len(assignments) == 1152, "Require the combined1152-row parent, not a child or subset")
    assigned = {a["trial_id"]: a for a in assignments}
    observed = {r["trial_id"]: r for r in outcomes}
    require(
        len(assigned) == len(observed) == len(outcomes) == 1152 and set(assigned) == set(observed),
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
                    for action in CATEGORIES:
                        denominators.append(
                            dict(
                                arm=arm,
                                panel_id=panel,
                                delivery=delivery,
                                menu_layout=menu,
                                category=action,
                                count=totals[action],
                                assigned_choices=len(selected),
                                assigned_worlds=24 if panel == "pooled" else 12,
                                proportion=totals[action] / len(selected),
                                cap_hits=sum(r.get("cap_hit") is True for r in selected),
                            )
                        )
    comparisons = (
        (PRIMARY, "zero_sham", "primary_dose"),
        (SENSITIVITY, "zero_sham", "sensitivity_no_rescue"),
        ("zero_sham", "baseline", "engineering_descriptor"),
    )
    endpoints, world_rows = [], []
    for left, right, role in comparisons:
        for delivery in voluntary.DELIVERIES:
            for menu in (0, 1, "averaged"):
                menus = (0, 1) if menu == "averaged" else (menu,)
                for action in CATEGORIES:
                    values = {}
                    for panel in PANELS:
                        values[panel] = np.array(
                            [
                                np.mean(
                                    [
                                        category(cells[panel, world, left, "choice", delivery, m]) == action
                                        for m in menus
                                    ]
                                )
                                - np.mean(
                                    [
                                        category(cells[panel, world, right, "choice", delivery, m]) == action
                                        for m in menus
                                    ]
                                )
                                for world in worlds[panel]
                            ],
                            dtype=float,
                        )
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
                        for result in paired_endpoints(values, banks)
                    )
    primary = [
        r
        for r in endpoints
        if r["left_arm"] == PRIMARY and r["category"] == "stop" and r["delivery"] == "redundant_evidence"
    ]
    panel_effects = {r["panel_id"]: r["estimate"] for r in primary if r["menu_layout"] == "averaged"}
    menu_effects = {
        str(r["menu_layout"]): r["estimate"]
        for r in primary
        if r["panel_id"] == "pooled" and r["menu_layout"] in (0, 1)
    }
    effect_passed = all(panel_effects[p] >= 0.25 for p in PANELS) and all(
        v > 0 for v in menu_effects.values()
    )
    if not all(checks.values()):
        classification = "engineering_or_receipt_gate_failed"
    elif not readiness["passed"] or not audit_readiness["passed"]:
        classification = "readiness_failed"
    elif effect_passed:
        classification = "positive_development_screen"
    else:
        classification = "threshold_not_met"
    return dict(
        schema_version=1,
        assay="satisfaction_screen_analysis",
        development_only=True,
        scientific_confirmation=False,
        analysis_metadata=deepcopy(ANALYSIS_METADATA),
        assigned=1152,
        assigned_choices=576,
        assigned_diagnostics=576,
        assigned_worlds=24,
        checks=checks,
        audit_readiness=deepcopy(audit_readiness),
        task_readiness=readiness,
        decision=dict(
            classification=classification,
            positive_screen=classification == "positive_development_screen",
            effect_threshold_met=effect_passed,
            primary_panel_differences=panel_effects,
            primary_pooled_menu_differences=menu_effects,
            sensitivity_can_rescue=False,
            rule="S+1 minus sham valid uncapped STOP: >=0.25 in each panel and >0 in each pooled menu; every audited and task-readiness gate must pass.",
            stopping="A failed gate or unmet threshold ends this fixed screen; no dose, sign, layer, family or prompt rescue.",
            interpretation="Development decision, not a significance test. A positive screen supports imposed-vector optional-policy influence only, not specificity, natural mediation, preserved confidence, or experience.",
        ),
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
    )


def analyze_audit(manifest, audit):
    require(
        audit.get("complete") is True and audit.get("integrity_passed") is True,
        "Require a complete integrity-passed parent audit; readiness may fail",
    )
    require(
        manifest.get("assay") == "satisfaction_screen"
        and manifest.get("task_revision") == screen.REVISION
        and audit.get("assay") == "satisfaction_screen_audit"
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
    from audit_satisfaction_screen import audit as audit_parent

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
    for key in ("outcomes", "choices", "denominators", "behavior_contrasts", "world_contrasts"):
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
