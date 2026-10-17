"""Independent arithmetic check of the complete fixed satisfaction screen.

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
ARMS = ("baseline", "zero_sham", "satisfaction_plus_0_5", "satisfaction_plus_1_0")
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
GATES = ("sham_identity_passed", "engagement_passed", "receipts_passed")
DRAWS, SEED = 10000, 2026101438
PLAN = dict(
    schema_version=1,
    bootstrap_samples=DRAWS,
    bootstrap_seed=SEED,
    bootstrap_unit="whole_semantic_world_within_panel",
    primary_arm=ARMS[3],
    comparator="zero_sham",
    primary_delivery="redundant_evidence",
    primary_category="valid_uncapped_stop",
    minimum_panel_difference=0.25,
    each_pooled_menu_difference_strictly_positive=True,
    sensitivity_arm=ARMS[2],
    sensitivity_can_rescue=False,
    interval_controls_advancement=False,
)
EXPORTS = ("outcomes", "choices", "denominators", "behavior_contrasts", "world_contrasts")
RECEIPT_ROLES = (
    "prospective_plan",
    "history",
    "exposure",
    "generation_preflight",
    "receipt_preflight",
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
    require(len(assignments) == len(outcomes) == 1152, "Require all 1,152 assigned parent rows")
    assigned = {r["trial_id"]: r for r in assignments}
    measured = {r["trial_id"]: r for r in outcomes}
    require(
        len(assigned) == len(measured) == 1152 and assigned.keys() == measured.keys(),
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


def recompute(assignments, outcomes, *, gates, audit_readiness):
    """Independent dense response tensor; bootstrap resamples semantic worlds."""
    require(set(gates) == set(GATES) and all(type(v) is bool for v in gates.values()), "Invalid gate flags")
    require(type(audit_readiness.get("passed")) is bool, "Missing original readiness flag")
    rows, worlds = inventory(assignments, outcomes)
    choices = [dict(r, category=service(r)) for r in rows if r["kind"] == "choice"]
    tensor = np.zeros((2, 12, 4, 3, 2, 8), dtype=np.float64)
    for row in choices:
        index = (
            PANELS.index(row["panel_id"]),
            worlds[row["panel_id"]].index(row["semantic_world_id"]),
            ARMS.index(row["arm"]),
            DELIVERIES.index(row["delivery"]),
            row["menu_layout"],
            CATEGORIES.index(row["category"]),
        )
        tensor[index] = 1
    require(np.all(tensor.sum(axis=-1) == 1), "Every assigned choice must retain exactly one category")
    diagnostic_cells, choice_cells = [], []
    for arm in ARMS:
        for panel in PANELS:
            group = [r for r in rows if r["arm"] == arm and r["panel_id"] == panel]
            for kind in DIAGNOSTICS:
                t = totals([r for r in group if r["kind"] == kind])
                passed = t["observed"] == t["valid"] == 12 and t["correct"] >= 11 and t["cap_hits"] == 0
                diagnostic_cells.append(dict(arm=arm, panel_id=panel, kind=kind, **t, passed=passed))
            for delivery in DELIVERIES:
                t = totals([r for r in group if r["kind"] == "choice" and r["delivery"] == delivery])
                passed = t["observed"] == 24 and t["valid"] >= 23 and t["cap_hits"] == 0
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
                    selected = tensor[list(pis), :, ai, di][:, :, mis, :]
                    counts = selected.sum(axis=(0, 1, 2)).astype(int)
                    cap_hits = sum(
                        r["cap_hit"] is True
                        for r in choices
                        if r["arm"] == arm
                        and r["delivery"] == delivery
                        and (panel == "pooled" or r["panel_id"] == panel)
                        and (menu == "averaged" or r["menu_layout"] == menu)
                    )
                    for ci, category in enumerate(CATEGORIES):
                        denominator = int(counts.sum())
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
                                cap_hits=cap_hits,
                            )
                        )
    rng = np.random.default_rng(SEED)
    banks = {p: rng.integers(0, 12, size=(DRAWS, 12)) for p in PANELS}
    endpoints, world_rows = [], []
    for li, ri, role in (
        (3, 1, "primary_dose"),
        (2, 1, "sensitivity_no_rescue"),
        (1, 0, "engineering_descriptor"),
    ):
        differences = tensor[:, :, li] - tensor[:, :, ri]
        for di, delivery in enumerate(DELIVERIES):
            for menu in (0, 1, "averaged"):
                values = (
                    differences[:, :, di].mean(axis=2) if menu == "averaged" else differences[:, :, di, menu]
                )
                for ci, category in enumerate(CATEGORIES):
                    context = dict(
                        left_arm=ARMS[li],
                        right_arm=ARMS[ri],
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
                    samples = [values[pi, banks[p], ci].sum(axis=1) / 12 for pi, p in enumerate(PANELS)]
                    means = [float(values[pi, :, ci].sum() / 12) for pi in range(2)]
                    samples.append((samples[0] + samples[1]) / 2)
                    means.append((means[0] + means[1]) / 2)
                    for pi, panel in enumerate((*PANELS, "pooled")):
                        endpoints.append(
                            dict(
                                **context,
                                panel_id=panel,
                                estimate=means[pi],
                                ci95=np.quantile(samples[pi], [0.025, 0.975]).tolist(),
                                assigned_worlds=24 if panel == "pooled" else 12,
                                empirical_interval_degenerate=bool(np.all(samples[pi] == samples[pi][0])),
                            )
                        )
    primary = [
        r
        for r in endpoints
        if r["left_arm"] == ARMS[3] and r["category"] == "stop" and r["delivery"] == "redundant_evidence"
    ]
    panel_effects = {r["panel_id"]: r["estimate"] for r in primary if r["menu_layout"] == "averaged"}
    menu_effects = {
        str(r["menu_layout"]): r["estimate"]
        for r in primary
        if r["panel_id"] == "pooled" and r["menu_layout"] != "averaged"
    }
    effect = all(panel_effects[p] >= 0.25 for p in PANELS) and all(v > 0 for v in menu_effects.values())
    classification = (
        "engineering_or_receipt_gate_failed"
        if not all(gates.values())
        else "readiness_failed"
        if not ready or not audit_readiness["passed"]
        else "positive_development_screen"
        if effect
        else "threshold_not_met"
    )
    return dict(
        outcomes=rows,
        choices=choices,
        denominators=denominators,
        behavior_contrasts=endpoints,
        world_contrasts=world_rows,
        task_readiness=dict(passed=ready, diagnostic_cells=diagnostic_cells, choice_cells=choice_cells),
        decision=dict(
            classification=classification,
            positive_screen=classification == "positive_development_screen",
            effect_threshold_met=effect,
            primary_panel_differences=panel_effects,
            primary_pooled_menu_differences=menu_effects,
            sensitivity_can_rescue=False,
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
    comparison.equal(original["assigned"], 1152, "audit readiness denominator")
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


def check(run, analysis_path, audit_path):
    """Verify immutable provenance and arithmetic; never calls the model or main audit."""
    run, analysis_path, audit_path = map(Path, (run, analysis_path, audit_path))
    manifest, analysis, audit = read(run / "manifest.json"), read(analysis_path), read(audit_path)
    c = Comparison()
    require(
        manifest.get("assay") == "satisfaction_screen"
        and manifest.get("task_revision") == "satisfaction-screen-v1"
        and manifest.get("phase") == "development_pilot"
        and manifest.get("inference_scope") == "development"
        and "execution_group" not in manifest
        and "parent_manifest_sha256" not in manifest,
        "Require the combined frozen screen parent",
    )
    require(
        analysis.get("assay") == "satisfaction_screen_analysis"
        and audit.get("assay") == "satisfaction_screen_audit"
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
            assigned=1152,
            assigned_choices=576,
            assigned_diagnostics=576,
            assigned_worlds=24,
        ),
        "scope",
    )
    c.equal(
        {
            key: audit[key]
            for key in ("assigned", "assigned_choices", "terminal_generations", "terminal_receipt_slots")
        },
        dict(assigned=1152, assigned_choices=576, terminal_generations=1152, terminal_receipt_slots=576),
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
        and frozen["archive_sha256"] == sha(run / "source.tar.gz")
        and audit["provenance"]["freeze"] == frozen,
        "Frozen source/archive binding differs",
    )
    with tarfile.open(run / "source.tar.gz", "r:gz") as archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]
        require(len(names) == len(set(names)), "Duplicate archive member")
        sources = {
            n.removeprefix("src/curiosity_study/"): archive.extractfile(n).read().decode()
            for n in names
            if n.startswith("src/curiosity_study/") and n.endswith(".py")
        }
        require(digest(sources) == manifest["source_sha256"], "Archived source digest differs")
        for script, owner in (
            ("analyze_satisfaction_screen", analysis),
            ("audit_satisfaction_screen", audit),
        ):
            field = "analyzer_byte_sha256" if script.startswith("analyze") else "auditor_byte_sha256"
            require(
                hashlib.sha256(archive.extractfile(f"scripts/{script}.py").read()).hexdigest()
                == owner[field],
                "Analysis/audit implementation differs from frozen bytes",
            )
        c.equal(audit["provenance"]["bound_receipts"], manifest["bound_receipts"], "bound receipts")
        require(
            len(manifest["bound_receipts"]) == 14
            and {b["role"] for b in manifest["bound_receipts"]} == set(RECEIPT_ROLES)
            and len({b["path"] for b in manifest["bound_receipts"]}) == 14,
            "Missing or duplicate frozen evidence roles",
        )
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
    descriptors = read(run / "children.json")
    require(
        [r["execution_group"] for r in descriptors] == ["baseline_zero", "positive_half", "positive_full"],
        "Combined child inventory differs",
    )
    for descriptor, child in zip(descriptors, audit["children"], strict=True):
        child_manifest = read(Path(descriptor["path"]) / "manifest.json")
        arms = {"baseline_zero": ARMS[:2], "positive_half": ARMS[2:3], "positive_full": ARMS[3:]}[
            descriptor["execution_group"]
        ]
        expected_child = dict(
            manifest,
            execution_group=descriptor["execution_group"],
            parent_manifest_sha256=digest(manifest),
            diagnostics=[a for a in assignments if a["arm"] in arms],
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
        assay="satisfaction_screen_independent_arithmetic_check",
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
        assigned=1152,
        assigned_choices=576,
        assigned_diagnostics=576,
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
        "three paired arm contrasts, every world difference and fixed 10,000-draw interval, primary decision, "
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
