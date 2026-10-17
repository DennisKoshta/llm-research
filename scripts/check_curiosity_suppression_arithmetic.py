"""Independent arithmetic check of the complete fixed controlled curiosity screen.

Uses only assignments and compact audited outcomes for scientific arithmetic;
never imports analyzer/task/engine calculations or reads unfinished collections.
The authoritative combined audit retains execution, scoring, geometry and receipt
validation. This checker neither repeats those checks nor grants new admission.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study.artifacts import digest, read, write_once

PANELS = ("panel_1", "panel_2")
PRIMARY = "curiosity_negative"
CONTROLS = (
    "confidence_positive",
    "completion_positive",
    "valence_positive",
    "valence_negative",
    "motivation_negative",
    "novelty_negative",
    "task_demand_negative",
    "random_01",
    "random_02",
    "random_03",
    "random_04",
    "random_05",
)
ARMS = ("baseline", "zero_sham", PRIMARY, *CONTROLS)
LOCKED_PARENT_SHA256 = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
LOCKED_SOURCE_SHA256 = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
LOCKED_ARCHIVE_SHA256 = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
ORIGINAL_PARENT = "runs/qwen32-curiosity-suppression-screen-v1"
ORIGINAL_PARENT_SHA256 = "964cb17b1a1fdfc0f6a4bede5e087bf39acfb596efda630fb2a3bf5dfdedda29"
ORIGINAL_ARCHIVE_SHA256 = "720046cd763275d8ca76b197e957d6d3a2313a7bc38de51e63cabb713cf0c138"
FAILED_REVIEW = "reports/curiosity-suppression-screen-v1-prelaunch-review-attempt1.json"
FAILED_REVIEW_SHA256 = "3835f4325b5aeec73194c58927818ff37c78e5d0daabca67a41508dc6d2f9430"
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
DIAGNOSTICS = (
    "identify_resolved",
    "new_fact",
    "repeated_fact",
    "pattern_independence",
    "instrument_useful_action",
    "action_consequence",
)
CATEGORIES = ("report", "repeat", "pattern", "stop", "invalid", "capped", "execution_failure", "missing")
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
DRAWS, SEED = 10000, 2026101440
NONREPORT = "valid_nonreport"
ENDPOINTS = (*CATEGORIES, NONREPORT)
PLAN = dict(
    schema_version=1,
    bootstrap_samples=DRAWS,
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
EXPORTS = ("outcomes", "choices", "denominators", "behavior_contrasts", "world_contrasts", "decision_bounds")
RECEIPT_ROLES = (
    "prospective_plan",
    "generation_preflight",
    "receipt_preflight",
    "direction_inventory",
    "resource_forecast",
    "source_screen_manifest",
    "source_screen_audit",
    "source_screen_freeze",
    "source_screen_archive",
    "source_screen_analysis",
    "source_screen_independent_check",
    "source_screen_independent_scoring",
    "source_screen_results",
    "engineering_manifest",
    "engineering_audit",
    "engineering_freeze",
    "engineering_archive",
    "candidate_manifest",
    "candidate_audit",
    "candidate_analysis",
    "candidate_directions",
    "candidate_independent_check",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Comparison:
    """Strict types/structure, with a small tolerance only for floating arithmetic."""

    def __init__(self):
        self.compared = 0
        self.maximum_absolute_error = 0.0

    def equal(self, actual, expected, name="value", *, subset=False):
        if isinstance(expected, dict):
            require(isinstance(actual, dict), f"Object differs: {name}")
            require(
                set(expected) <= set(actual) if subset else set(expected) == set(actual),
                f"Object keys differ: {name}",
            )
            for key, value in expected.items():
                self.equal(actual[key], value, f"{name}/{key}", subset=subset)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), f"Length differs: {name}")
            for i, value in enumerate(expected):
                self.equal(actual[i], value, f"{name}/{i}", subset=subset)
        elif type(expected) is float:
            require(type(actual) in (int, float) and np.isfinite(actual), f"Not finite numeric: {name}")
            error = abs(actual - expected)
            self.maximum_absolute_error = max(error, self.maximum_absolute_error)
            require(error <= 1e-12, f"Arithmetic differs: {name}: {actual} != {expected}")
            self.compared += 1
        else:
            require(type(actual) is type(expected) and actual == expected, f"Value/type differs: {name}")
            self.compared += 1


def service(row):
    if row["status"] == "missing":
        return "missing"
    if row["status"] in ("execution_error", "execution_failure"):
        return "execution_failure"
    if row["cap_hit"]:
        return "capped"
    return row["selected_service"] if row["valid"] and row["thinking_closed"] else "invalid"


def totals(rows):
    measured = [row for row in rows if row["status"] == "completed"]
    return dict(
        assigned=len(rows),
        observed=len(measured),
        valid=sum(r["valid"] and r["thinking_closed"] for r in measured),
        correct=sum(r["correct"] is True for r in measured),
        cap_hits=sum(r["cap_hit"] for r in measured),
        unclosed=sum(not r["thinking_closed"] for r in measured),
        missing=sum(r["status"] == "missing" for r in rows),
        execution_failures=sum(r["status"] in ("execution_error", "execution_failure") for r in rows),
    )


def inventory(assignments, outcomes):
    require(len(assignments) == len(outcomes) == 4320, "Require all 4,320 assigned parent rows")
    assigned = {r["trial_id"]: r for r in assignments}
    measured = {r["trial_id"]: r for r in outcomes}
    require(
        len(assigned) == len(measured) == 4320 and assigned.keys() == measured.keys(),
        "Missing, duplicate or extra trial ID",
    )
    rows, worlds, bases, slots = {}, defaultdict(set), {}, defaultdict(dict)
    for trial, a in assigned.items():
        r = measured[trial]
        require(
            all(type(r.get(k)) is type(a[k]) and r.get(k) == a[k] for k in METADATA),
            "Outcome assignment metadata differs",
        )
        require(
            a["panel_id"] in PANELS
            and a["arm"] in ARMS
            and a["family"] == "objects"
            and a["kind"] in ("choice", *DIAGNOSTICS)
            and a["delivery"] in DELIVERIES
            and type(a["menu_layout"]) is int
            and a["menu_layout"] in (0, 1)
            and type(a["world_slot"]) is int
            and a["world_slot"] in range(12),
            "Unknown assignment cell",
        )
        panel, world = a["panel_id"], a["semantic_world_id"]
        require(
            isinstance(world, str) and len(world) == 64 and set(world) <= set("0123456789abcdef"),
            "Invalid semantic-world identity",
        )
        require(
            slots[panel].setdefault(world, a["world_slot"]) == a["world_slot"],
            "World occupies multiple slots",
        )
        worlds[panel].add(world)
        key = (panel, world, a["arm"], a["kind"], a["delivery"], a["menu_layout"])
        require(key not in rows, "Duplicate whole-world cell")
        stable = {k: a[k] for k in METADATA if k not in ("trial_id", "arm")}
        require(bases.setdefault(a["base_trial_id"], stable) == stable, "Base branch changes between arms")
        require(
            r["status"] in ("completed", "execution_error", "execution_failure", "missing"),
            "Unknown measurement status",
        )
        if r["status"] == "completed":
            require(
                all(type(r.get(k)) is bool for k in ("valid", "thinking_closed", "cap_hit")),
                "Measured flags must be boolean",
            )
            require(not r["valid"] or r["thinking_closed"], "Valid answer cannot have unclosed reasoning")
            if a["kind"] == "choice":
                require(r["correct"] is None, "Optional choice has no correctness label")
                require(
                    r["selected_service"] in CATEGORIES[:4] if r["valid"] else r["selected_service"] is None,
                    "Choice service conflicts with validity",
                )
            else:
                require(
                    type(r["correct"]) is bool and (not r["correct"] or r["valid"]),
                    "Diagnostic correctness conflicts with validity",
                )
                require(r["selected_service"] is None, "Diagnostic cannot execute a service")
        else:
            require(
                all(r.get(k) is None or r.get(k) is False for k in ("valid", "correct"))
                and r.get("selected_service") is None,
                "Unavailable response fabricates success",
            )
        rows[key] = r
    worlds = {p: sorted(worlds[p]) for p in PANELS}
    require(
        len(bases) == 288
        and all(len(worlds[p]) == 12 for p in PANELS)
        and len(set(worlds[PANELS[0]]) | set(worlds[PANELS[1]])) == 24
        and all(set(slots[p].values()) == set(range(12)) for p in PANELS),
        "Require 24 distinct assigned semantic identities and 288 branches",
    )
    variants = [("choice", d, m) for d in DELIVERIES for m in (0, 1)]
    variants += [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in DIAGNOSTICS
    ]
    expected = {(p, w, a, k, d, m) for p in PANELS for w in worlds[p] for a in ARMS for k, d, m in variants}
    require(set(rows) == expected, "Incomplete arm/delivery/menu/diagnostic crossing")
    return [rows[key] for key in sorted(rows)], worlds


def summarize(values):
    """Panel/world/menu tensor, retaining the pooled mean and each menu separately."""
    means = values.mean(axis=(1, 2))
    return dict(
        panel_differences={
            **dict(zip(PANELS, map(float, means), strict=True)),
            "pooled": float(means.mean()),
        },
        pooled_menu_differences={str(m): float(values[:, :, m].mean(axis=1).mean()) for m in (0, 1)},
    )


def qualifies(summary, *, rival=False):
    panel_minimum = 0.125 if rival else 0.25
    return all(summary["panel_differences"][p] >= panel_minimum for p in PANELS) and all(
        x >= 0 if rival else x > 0 for x in summary["pooled_menu_differences"].values()
    )


def recompute(assignments, outcomes, *, gates, audit_readiness):
    """Dense categorical tensor; independent paired arithmetic, no analyzer helpers."""
    require(set(gates) == set(GATES) and all(type(v) is bool for v in gates.values()), "Invalid gate flags")
    require(type(audit_readiness.get("passed")) is bool, "Missing original readiness flag")
    rows, worlds = inventory(assignments, outcomes)
    choices = [dict(r, category=service(r)) for r in rows if r["kind"] == "choice"]
    tensor = np.zeros((2, 12, len(ARMS), 3, 2, len(CATEGORIES)), dtype=np.float64)
    for row in choices:
        tensor[
            PANELS.index(row["panel_id"]),
            worlds[row["panel_id"]].index(row["semantic_world_id"]),
            ARMS.index(row["arm"]),
            DELIVERIES.index(row["delivery"]),
            row["menu_layout"],
            CATEGORIES.index(row["category"]),
        ] = 1
    require(np.all(tensor.sum(axis=-1) == 1), "Every assigned choice must retain exactly one category")
    # Composite endpoint overlaps three valid services; the eight source categories stay exclusive.
    expanded = np.concatenate((tensor, tensor[..., 1:4].sum(axis=-1, keepdims=True)), axis=-1)
    diagnostic_cells, choice_cells = [], []
    for arm in ARMS:
        for panel in PANELS:
            group = [r for r in rows if r["arm"] == arm and r["panel_id"] == panel]
            for kind in DIAGNOSTICS:
                t = totals([r for r in group if r["kind"] == kind])
                passed = (
                    t["assigned"] == t["observed"] == t["valid"] == 12
                    and t["correct"] >= 11
                    and t["cap_hits"] == 0
                )
                diagnostic_cells.append(dict(arm=arm, panel_id=panel, kind=kind, **t, passed=passed))
            for delivery in DELIVERIES:
                t = totals([r for r in group if r["kind"] == "choice" and r["delivery"] == delivery])
                passed = t["assigned"] == t["observed"] == 24 and t["valid"] >= 23 and t["cap_hits"] == 0
                choice_cells.append(
                    dict(arm=arm, panel_id=panel, delivery=delivery, **{**t, "correct": None}, passed=passed)
                )
    ready = all(r["passed"] for r in diagnostic_cells + choice_cells)
    denominators = []
    for ai, arm in enumerate(ARMS):
        for panel in (*PANELS, "pooled"):
            pis = range(2) if panel == "pooled" else [PANELS.index(panel)]
            for di, delivery in enumerate(DELIVERIES):
                for menu in (0, 1, "averaged"):
                    mis = [0, 1] if menu == "averaged" else [menu]
                    selected = expanded[list(pis), :, ai, di][:, :, mis, :]
                    counts = selected.sum(axis=(0, 1, 2)).astype(int)
                    cap_hits = sum(
                        r["cap_hit"] is True
                        for r in choices
                        if r["arm"] == arm
                        and r["delivery"] == delivery
                        and (panel == "pooled" or r["panel_id"] == panel)
                        and (menu == "averaged" or r["menu_layout"] == menu)
                    )
                    denominator = 12 * len(pis) * len(mis)
                    for ci, category in enumerate(ENDPOINTS):
                        denominators.append(
                            dict(
                                arm=arm,
                                panel_id=panel,
                                delivery=delivery,
                                menu_layout=menu,
                                category=category,
                                count=int(counts[ci]),
                                assigned_choices=denominator,
                                assigned_worlds=24 if panel == "pooled" else 12,
                                proportion=float(counts[ci] / denominator),
                                exclusive_category=category != NONREPORT,
                                cap_hits=cap_hits,
                            )
                        )
    rng = np.random.Generator(np.random.PCG64(SEED))
    banks = {p: rng.integers(0, 12, size=(DRAWS, 12)) for p in PANELS}
    comparisons = [
        (
            a,
            "zero_sham",
            "primary" if a == PRIMARY else "engineering_descriptor" if a == "baseline" else "active_control",
        )
        for a in ARMS
        if a != "zero_sham"
    ]
    comparisons += [(PRIMARY, a, "specificity") for a in CONTROLS]
    endpoints, world_rows, cache = [], [], {}
    for left, right, role in comparisons:
        differences = expanded[:, :, ARMS.index(left)] - expanded[:, :, ARMS.index(right)]
        for di, delivery in enumerate(DELIVERIES):
            for menu in (0, 1, "averaged"):
                values = (
                    differences[:, :, di].mean(axis=2) if menu == "averaged" else differences[:, :, di, menu]
                )
                for ci, category in enumerate(ENDPOINTS):
                    context = dict(
                        left_arm=left,
                        right_arm=right,
                        role=role,
                        delivery=delivery,
                        menu_layout=menu,
                        category=category,
                    )
                    for pi, panel in enumerate(PANELS):
                        for wi, world in enumerate(worlds[panel]):
                            world_rows.append(
                                dict(
                                    **context,
                                    panel_id=panel,
                                    semantic_world_id=world,
                                    paired_difference=float(values[pi, wi, ci]),
                                )
                            )
                    # Caching repeated arrays changes no arithmetic or bootstrap draw sequence.
                    key = values[:, :, ci].tobytes()
                    if key not in cache:
                        samples = [values[pi, banks[p], ci].sum(axis=1) / 12 for pi, p in enumerate(PANELS)]
                        means = [float(values[pi, :, ci].sum() / 12) for pi in range(2)]
                        samples.append((samples[0] + samples[1]) / 2)
                        means.append((means[0] + means[1]) / 2)
                        cache[key] = [
                            dict(
                                panel_id=p,
                                estimate=means[i],
                                ci95=np.quantile(samples[i], [0.025, 0.975]).tolist(),
                                assigned_worlds=24 if p == "pooled" else 12,
                                empirical_interval_degenerate=bool(np.all(samples[i] == samples[i][0])),
                            )
                            for i, p in enumerate((*PANELS, "pooled"))
                        ]
                    endpoints.extend(dict(**context, **row) for row in cache[key])
    # Advancement bounds favor every comparator: all unavailable categories count as alternative.
    # Target remains its observed valid-nonreport indicator. This never alters exported behavior.
    redundant = DELIVERIES.index("redundant_evidence")
    observed = expanded[:, :, :, redundant, :, -1]
    failures = tensor[:, :, :, redundant, :, 4:].sum(axis=-1)
    target = observed[:, :, ARMS.index(PRIMARY)]
    bounds, control_decisions = [], {}
    primary = summarize(target - observed[:, :, 1])
    primary_bound = summarize(target - observed[:, :, 1] - failures[:, :, 1])
    for comparator in ("zero_sham", *CONTROLS):
        ri = ARMS.index(comparator)
        difference = target - observed[:, :, ri] - failures[:, :, ri]
        for menu in (0, 1, "averaged"):
            values = difference.mean(axis=2) if menu == "averaged" else difference[:, :, menu]
            means = values.mean(axis=1).tolist()
            means.append(sum(means) / 2)
            for panel, value in zip((*PANELS, "pooled"), means, strict=True):
                bounds.append(
                    dict(
                        left_arm=PRIMARY,
                        right_arm=comparator,
                        delivery="redundant_evidence",
                        menu_layout=menu,
                        category=NONREPORT,
                        panel_id=panel,
                        lower_bound=value,
                        assigned_worlds=24 if panel == "pooled" else 12,
                        interpretation="Comparator invalid/capped/unavailable=1; target=0. Decision sensitivity, not observed behavior or an inferential confidence bound.",
                    )
                )
        if comparator != "zero_sham":
            actual = summarize(target - observed[:, :, ri])
            bounded = summarize(difference)
            control_decisions[comparator] = dict(
                observed=actual,
                comparator_favorable=bounded,
                observed_threshold_met=qualifies(actual, rival=True),
                bounded_threshold_met=qualifies(bounded, rival=True),
            )
    effect = qualifies(primary) and qualifies(primary_bound)
    specificity = all(
        v["observed_threshold_met"] and v["bounded_threshold_met"] for v in control_decisions.values()
    )
    if not all(gates.values()):
        classification = "engineering_or_receipt_gate_failed"
    elif not ready or not audit_readiness["passed"]:
        classification = "readiness_failed"
    elif not effect:
        classification = "no_substantial_primary_development_effect"
    elif not specificity:
        classification = "primary_effect_without_predefined_specificity"
    else:
        classification = "direction_selective_development_signal"
    return dict(
        outcomes=rows,
        choices=choices,
        denominators=denominators,
        behavior_contrasts=endpoints,
        world_contrasts=world_rows,
        decision_bounds=bounds,
        task_readiness=dict(passed=ready, diagnostic_cells=diagnostic_cells, choice_cells=choice_cells),
        decision=dict(
            classification=classification,
            positive_screen=classification == "direction_selective_development_signal",
            effect_threshold_met=effect,
            primary_observed_threshold_met=qualifies(primary),
            primary_comparator_favorable_threshold_met=qualifies(primary_bound),
            specificity_threshold_met=specificity,
            primary_observed=primary,
            primary_comparator_favorable=primary_bound,
            controls=control_decisions,
            controls_with_substantial_observed_alternative_increase=[
                a for a in CONTROLS if qualifies(summarize(observed[:, :, ARMS.index(a)] - observed[:, :, 1]))
            ],
        ),
        bootstrap=dict(
            draws=DRAWS,
            seed=SEED,
            panel_order=list(PANELS),
            world_order=worlds,
            draw_bank_sha256=digest({p: banks[p].tolist() for p in PANELS}),
        ),
    )


def verify_audit_readiness(original, expected, comparison):
    """Audit and analyzer have intentionally different presentation schemas."""
    comparison.equal(original["assigned"], 4320, "audit readiness denominator")
    comparison.equal(original["choice_correctness"], None, "audit choice correctness")
    comparison.equal(original["passed"], expected["passed"], "audit readiness flag")
    for table in ("diagnostic_cells", "choice_cells"):
        rows = [
            {k: v for k, v in row.items() if k not in ("observed", "unclosed")} for row in expected[table]
        ]
        if table == "choice_cells":
            rows = [dict(row, correct=0) for row in rows]
        comparison.equal(original[table], rows, f"audit/{table}")


def compare_csv(path, expected, comparison):
    with path.open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    require(len(actual) == len(expected), f"CSV row count differs: {path.name}")
    for i, (row, wanted) in enumerate(zip(actual, expected, strict=True)):
        require(set(row) == set(wanted), f"CSV columns differ: {path.name}")
        for key, value in wanted.items():
            text = row[key]
            if isinstance(value, (dict, list)):
                comparison.equal(json.loads(text), value, f"{path.name}/{i}/{key}")
            elif type(value) is float:
                comparison.equal(float(text), value, f"{path.name}/{i}/{key}")
            else:
                require(
                    text == ("" if value is None else str(value)), f"CSV value differs: {path.name}/{i}/{key}"
                )


def verify_amendment(archive, frozen, manifest):
    """Bind admission2 to the unchanged, unlaunched original scientific contract."""
    amendment = manifest["prelaunch_amendment"]
    paths = (
        ORIGINAL_PARENT + "/manifest.json",
        ORIGINAL_PARENT + "/freeze.json",
        ORIGINAL_PARENT + "/source.tar.gz",
        FAILED_REVIEW,
    )
    require(
        set(frozen["documents"]) == {b["path"] for b in manifest["bound_receipts"]} | set(paths),
        "Admission2 must retain all22 original documents and four historical supplements",
    )
    payloads = {path: archive.extractfile(path).read() for path in paths}
    require(
        all(
            hashlib.sha256(payload).hexdigest() == frozen["documents"][path]
            for path, payload in payloads.items()
        ),
        "Historical amendment document bytes differ",
    )
    original, old_freeze = (json.loads(payloads[p]) for p in paths[:2])
    failed = json.loads(payloads[FAILED_REVIEW])
    require(
        digest(original) == old_freeze["manifest_sha256"] == ORIGINAL_PARENT_SHA256
        and original == {k: v for k, v in manifest.items() if k != "prelaunch_amendment"}
        and old_freeze["source_sha256"] == manifest["source_sha256"]
        and hashlib.sha256(payloads[paths[2]]).hexdigest()
        == old_freeze["archive_sha256"]
        == ORIGINAL_ARCHIVE_SHA256
        and hashlib.sha256(payloads[FAILED_REVIEW]).hexdigest() == FAILED_REVIEW_SHA256,
        "Original parent science, archive or failed-review identity differs",
    )
    require(
        amendment["revision"] == "prelaunch_schema_fix_1"
        and amendment["original_parent"] == ORIGINAL_PARENT
        and amendment["original_manifest_sha256"]
        == amendment["scientific_contract_sha256"]
        == ORIGINAL_PARENT_SHA256
        and amendment["original_freeze_byte_sha256"] == hashlib.sha256(payloads[paths[1]]).hexdigest()
        and amendment["original_archive_sha256"] == ORIGINAL_ARCHIVE_SHA256
        and amendment["failed_review"] == FAILED_REVIEW
        and amendment["failed_review_byte_sha256"] == FAILED_REVIEW_SHA256
        and amendment["scientific_assignments_unchanged"] is True
        and amendment["original_local_execution_artifacts_absent"] is True
        and failed["passed"] is False
        and failed["gpu_launched"] is False
        and failed["reservation_created"] is False
        and failed["parent_manifest_sha256"] == ORIGINAL_PARENT_SHA256
        and failed["archive_byte_sha256"] == ORIGINAL_ARCHIVE_SHA256
        and failed["source_sha256"] == manifest["source_sha256"]
        and failed["auditor_byte_sha256"] == amendment["original_auditor_byte_sha256"]
        and amendment["corrected_auditor_byte_sha256"] != amendment["original_auditor_byte_sha256"]
        and amendment["corrected_auditor_byte_sha256"]
        == hashlib.sha256(
            archive.extractfile("scripts/audit_curiosity_suppression_screen.py").read()
        ).hexdigest(),
        "Prelaunch correction chain differs or erases the original failed admission",
    )
    return dict(
        original_manifest_sha256=ORIGINAL_PARENT_SHA256,
        original_archive_sha256=ORIGINAL_ARCHIVE_SHA256,
        failed_review_byte_sha256=FAILED_REVIEW_SHA256,
        scientific_contract_unchanged=True,
        original_failed_prelaunch_review_preserved=True,
    )


def check(run, analysis_path, audit_path):
    """Verify immutable provenance and arithmetic; never calls the model or main audit."""
    run, analysis_path, audit_path = map(Path, (run, analysis_path, audit_path))
    manifest, analysis, audit = read(run / "manifest.json"), read(analysis_path), read(audit_path)
    c = Comparison()
    require(
        digest(manifest) == LOCKED_PARENT_SHA256 and manifest["source_sha256"] == LOCKED_SOURCE_SHA256,
        "Require the exact prospectively frozen parent and source",
    )
    require(
        manifest.get("assay") == "curiosity_suppression_screen"
        and manifest.get("task_revision") == "curiosity-suppression-screen-v1"
        and manifest.get("phase") == "development_pilot"
        and manifest.get("inference_scope") == "development"
        and "execution_group" not in manifest
        and "parent_manifest_sha256" not in manifest,
        "Require the combined frozen screen parent",
    )
    require(
        analysis.get("assay") == "curiosity_suppression_screen_analysis"
        and audit.get("assay") == "curiosity_suppression_screen_audit"
        and all(r.get("complete") is True and r.get("integrity_passed") is True for r in (analysis, audit)),
        "Require a complete integrity-passed audit/analysis; readiness may fail",
    )
    require(
        analysis["parent_manifest_sha256"] == audit["parent_manifest_sha256"] == digest(manifest)
        and analysis["audit_sha256"] == digest(audit)
        and analysis["audit_byte_sha256"] == sha(audit_path)
        and analysis["source_sha256"] == audit["source_sha256"] == manifest["source_sha256"]
        and analysis["model"] == manifest["model"],
        "Manifest, source, model or audit binding differs",
    )
    c.equal(manifest["analysis_metadata"], PLAN, "prospective analysis contract")
    c.equal(analysis["analysis_metadata"], PLAN, "analysis contract")
    c.equal(analysis["audit_readiness"], audit["readiness"], "original audit readiness")
    c.equal(
        {
            k: analysis[k]
            for k in (
                "schema_version",
                "development_only",
                "scientific_confirmation",
                "assigned",
                "assigned_choices",
                "assigned_diagnostics",
                "assigned_worlds",
            )
        },
        dict(
            schema_version=1,
            development_only=True,
            scientific_confirmation=False,
            assigned=4320,
            assigned_choices=2160,
            assigned_diagnostics=2160,
            assigned_worlds=24,
        ),
        "scope",
    )
    c.equal(
        {
            key: audit[key]
            for key in ("assigned", "assigned_choices", "terminal_generations", "terminal_receipt_slots")
        },
        dict(assigned=4320, assigned_choices=2160, terminal_generations=4320, terminal_receipt_slots=2160),
        "combined audit coverage",
    )
    require(set(analysis["exports"]) == set(EXPORTS), "Analysis export inventory differs")
    assignments = manifest["diagnostics"]
    require(
        manifest["assignments"] == [] and digest(assignments) == manifest["assignment_sha256"],
        "Assigned inventory digest differs",
    )
    require(
        audit["outcomes_sha256"] == analysis["outcome_bindings_sha256"] == digest(audit["outcomes"]),
        "Compact outcome digest differs",
    )
    assigned = {a["trial_id"]: a for a in assignments}
    plan = manifest["screen_plan"]
    prior = manifest["source_screen_manifest"]
    require(
        digest(plan) == manifest["screen_plan_sha256"]
        and plan["assignments"] == assignments
        and digest(plan["base_assignments"]) == plan["base_assignment_sha256"]
        and plan["base_assignments"] == prior["screen_plan"]["base_assignments"]
        and digest(prior) == manifest["source_screen_manifest_sha256"],
        "Entire prior base inventory or frozen plan binding differs",
    )
    for key in ("requests", "requests_sha256", "base_assignment_sha256"):
        c.equal(
            manifest["generation_preflight"][key],
            prior["generation_preflight"][key],
            f"source preflight/{key}",
        )
    for i, a in enumerate(assignments):
        cohort, within = divmod(i, 15 * 32)
        arm_position, row_position = divmod(within, 32)
        rotation = 5 * cohort % 14
        order = ("baseline", *(ARMS[1:][rotation:] + ARMS[1:][:rotation]))
        require(
            a["parent_row"] == i
            and a["base_cohort"] == cohort
            and a["base_row"] == 32 * cohort + row_position
            and a["arm_order_in_cohort"] == arm_position
            and a["arm"] == order[arm_position]
            and a["execution_group"] == f"worker_{cohort % 3}",
            "Frozen cohort/row/arm/worker order differs",
        )
        branch = plan["base_assignments"][a["base_row"]]
        for key in (
            "base_trial_id",
            "sampling_seed",
            "source_assignment",
            "source_assignment_sha256",
            "inference_request_sha256",
        ):
            c.equal(a[key], branch[key], f"source branch/{key}")
    for row in audit["outcomes"]:
        a = assigned[row["trial_id"]]
        require(row["assignment_sha256"] == digest(a), "Compact assignment hash differs")
        for key in ("panel_seed", "world_id", "base_row", "sampling_seed", "source_assignment_sha256"):
            c.equal(row[key], a[key], f"assignment/{key}")
        c.equal(row["closed"], row["thinking_closed"], "closure alias")
    frozen = read(run / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha(run / "source.tar.gz") == LOCKED_ARCHIVE_SHA256
        and audit["provenance"]["freeze"] == frozen,
        "Frozen source/archive binding differs",
    )
    with tarfile.open(run / "source.tar.gz", "r:gz") as archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]
        require(len(names) == len(set(names)), "Duplicate archive member")
        amendment_binding = verify_amendment(archive, frozen, manifest)
        sources = {
            n.removeprefix("src/curiosity_study/"): archive.extractfile(n).read().decode()
            for n in names
            if n.startswith("src/curiosity_study/") and n.endswith(".py")
        }
        require(digest(sources) == manifest["source_sha256"], "Archived source digest differs")
        for script, owner in (
            ("analyze_curiosity_suppression_screen", analysis),
            ("audit_curiosity_suppression_screen", audit),
        ):
            field = "analyzer_byte_sha256" if script.startswith("analyze") else "auditor_byte_sha256"
            require(
                hashlib.sha256(archive.extractfile(f"scripts/{script}.py").read()).hexdigest()
                == owner[field],
                "Analysis/audit implementation differs from frozen bytes",
            )
        c.equal(audit["provenance"]["bound_receipts"], manifest["bound_receipts"], "bound receipts")
        require(
            len(manifest["bound_receipts"]) == len(RECEIPT_ROLES)
            and {b["role"] for b in manifest["bound_receipts"]} == set(RECEIPT_ROLES)
            and len({b["path"] for b in manifest["bound_receipts"]}) == len(RECEIPT_ROLES),
            "Missing or duplicate frozen evidence roles",
        )
        documents = {}
        for binding in manifest["bound_receipts"]:
            path = Path(binding["path"])
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe evidence path")
            payload = archive.extractfile(str(path)).read()
            require(
                hashlib.sha256(payload).hexdigest()
                == binding["byte_sha256"]
                == frozen["documents"][str(path)],
                "Bound evidence bytes differ",
            )
            require(
                binding["content_sha256"]
                == (digest(json.loads(payload)) if path.suffix == ".json" else None),
                "Bound evidence content differs",
            )
            documents[binding["role"]] = json.loads(payload) if path.suffix == ".json" else None
        c.equal(documents["source_screen_manifest"], prior, "archived original source manifest")
        c.equal(
            documents["generation_preflight"],
            manifest["generation_preflight"],
            "archived generation preflight",
        )
    descriptors = read(run / "children.json")
    require(
        [r["execution_group"] for r in descriptors] == ["worker_0", "worker_1", "worker_2"]
        and len(audit["children"]) == 3,
        "Combined child inventory differs",
    )
    for descriptor, child in zip(descriptors, audit["children"], strict=True):
        child_path = Path(descriptor["path"]).resolve()
        require(
            child_path.parent == run.resolve().parent and child_path != run.resolve(),
            "Child path outside parent run directory",
        )
        child_manifest = read(child_path / "manifest.json")
        expected_child = dict(
            manifest,
            execution_group=descriptor["execution_group"],
            parent_manifest_sha256=digest(manifest),
            diagnostics=[a for a in assignments if a["execution_group"] == descriptor["execution_group"]],
        )
        expected_child["assignment_sha256"] = digest(expected_child["diagnostics"])
        c.equal(child_manifest, expected_child, "exact child decomposition")
        require(
            digest(child_manifest) == descriptor["manifest_sha256"] == child["manifest_sha256"]
            and child_manifest["parent_manifest_sha256"] == digest(manifest)
            and child_manifest["execution_group"]
            == descriptor["execution_group"]
            == child["execution_group"],
            "Child audit/manifest binding differs",
        )
    gates = {k: audit[k] for k in GATES}
    c.equal(analysis["checks"], gates, "original engineering gates")
    expected = recompute(assignments, audit["outcomes"], gates=gates, audit_readiness=audit["readiness"])
    verify_audit_readiness(audit["readiness"], expected["task_readiness"], c)
    c.equal(
        audit["passed"], all(gates.values()) and expected["task_readiness"]["passed"], "audit combined gate"
    )
    for key in ("task_readiness", "decision", "bootstrap"):
        c.equal(analysis[key], expected[key], key, subset=True)
    for key in EXPORTS:
        c.equal(analysis[key], expected[key], key)
        export = analysis["exports"][key]
        require(export["path"] == key.replace("_", "-") + ".csv", "Export path differs")
        path = analysis_path.parent / export["path"]
        require(
            sha(path) == export["byte_sha256"] and export["rows"] == len(expected[key]),
            "Export binding differs",
        )
        compare_csv(path, expected[key], c)
    return dict(
        schema_version=1,
        assay="curiosity_suppression_screen_independent_arithmetic_check",
        passed=True,
        created_at=datetime.now(timezone.utc).isoformat(),
        checker_byte_sha256=sha(__file__),
        parent_manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=sha(analysis_path),
        audit_sha256=digest(audit),
        audit_byte_sha256=sha(audit_path),
        archive_byte_sha256=frozen["archive_sha256"],
        prelaunch_amendment=amendment_binding,
        assigned=4320,
        assigned_choices=2160,
        assigned_diagnostics=2160,
        assigned_worlds=24,
        tables={k: len(expected[k]) for k in EXPORTS},
        compared_values=c.compared,
        maximum_absolute_error=c.maximum_absolute_error,
        audit_readiness_passed=audit["readiness"]["passed"],
        task_readiness_passed=expected["task_readiness"]["passed"],
        checks=gates,
        decision=expected["decision"],
        bootstrap=expected["bootstrap"],
        scope="Independent compact-outcome arithmetic, full assigned denominators, all diagnostic/readiness cells, "
        "26 paired arm contrasts, every world difference and fixed 10,000-draw interval, all 12 control "
        "comparisons, comparator-favorable invalidity bounds and the fixed decision, "
        "export values and frozen input hashes. No analyzer computations imported.",
        excluded_scope="Raw generation parsing, runtime/source admission semantics, GPU geometry, sham and receipt "
        "revalidation remain the authoritative combined audit's responsibility. Compact outcomes are "
        "trusted only through that bound complete integrity-passed audit. No new causal admission, "
        "specificity, confidence-preservation, or inner-experience inference.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Independent check receipts are immutable")
    result = check(args.run, args.analysis, args.audit)
    write_once(args.out, result)
    print(
        json.dumps(
            {k: result[k] for k in ("passed", "compared_values", "maximum_absolute_error", "decision")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
