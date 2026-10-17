"""Pure, prospective arithmetic for the fixed 29-arm coverage bridge.

No source scoring, fitting, execution or admission. Inputs must be compact audited
outcomes for every assigned slot, including explicit missing/error dispositions.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

PANELS = ("panel_1", "panel_2")
SCHEDULES = ("record", "full")
PRIMARY = "curiosity_negative"
CONTROLS = ("confidence_positive", "completion_positive", "valence_positive", "valence_negative",
            "motivation_negative", "novelty_negative", "task_demand_negative",
            *(f"random_{i:02d}" for i in range(1, 6)))
DIRECTIONS = (PRIMARY, *CONTROLS)
ARMS = ("baseline", *(f"{s}__{a}" for a in ("zero_sham", *DIRECTIONS) for s in SCHEDULES))
DIAGNOSTICS = ("identify_resolved", "new_fact", "repeated_fact", "pattern_independence",
               "instrument_useful_action", "action_consequence")
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
SERVICES = ("report", "repeat", "pattern", "stop")
CATEGORIES = (*SERVICES, "invalid", "capped", "execution_failure", "missing")
NONREPORT = "valid_nonreport"
ENDPOINTS = (*CATEGORIES, NONREPORT)
FAILURES = ("invalid", "capped", "execution_failure", "missing")
GATES = ("sham_identity_passed", "engagement_passed", "receipts_passed",
         "cross_arm_prepatch_identity_passed", "worker_policies_equal")
METADATA = ("trial_id", "base_trial_id", "panel_id", "semantic_world_id", "world_slot", "arm",
            "schedule", "direction_key", "family", "kind", "delivery", "menu_layout")
ASSIGNED, BOOTSTRAPS, SEED = 8352, 10000, 2026101552
ANALYSIS_METADATA = dict(schema_version=1, revision="curiosity-coverage-bridge-v1",
    arms=list(ARMS), controls=list(CONTROLS), primary_direction=PRIMARY, schedules=list(SCHEDULES),
    primary_delivery="redundant_evidence", primary_category=NONREPORT,
    panel_effect_margin=.25, panel_specificity_margin=.125,
    primary_each_pooled_menu_strictly_positive=True, specificity_each_pooled_menu_nonnegative=True,
    coefficient_aggregation_before_invalidity_bound=True, all12_controls_required=True,
    diagnostic_cell_assigned=12, diagnostic_minimum_valid=11, diagnostic_minimum_correct=11,
    diagnostic_arm_assigned=144, diagnostic_arm_minimum_valid=137,
    choice_cell_assigned=24, choice_minimum_valid=23, maximum_caps=0,
    bootstrap_samples=BOOTSTRAPS, bootstrap_seed=SEED, bootstrap_generator="PCG64",
    bootstrap_unit="whole_semantic_world_within_panel", intervals_control_advancement=False,
    scientific_admission=False, causal_admission=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def arm(direction, schedule):
    return f"{schedule}__{direction}"


def category(row):
    if row["status"] == "missing":
        return "missing"
    if row["status"] in ("execution_error", "execution_failure"):
        return "execution_failure"
    if row["cap_hit"]:
        return "capped"
    if not row["valid"] or not row["thinking_closed"]:
        return "invalid"
    return row["selected_service"]


def inventory(assignments, outcomes):
    require(len(assignments) == len(outcomes) == ASSIGNED, "Require all8352 assigned compact slots")
    assigned = {a["trial_id"]: a for a in assignments}
    observed = {r["trial_id"]: r for r in outcomes}
    require(len(assigned) == len(observed) == ASSIGNED and set(assigned) == set(observed),
            "Missing, duplicate or extra response ID")
    cells, bases, slots = {}, {}, defaultdict(dict)
    for a in assignments:
        expected_schedule, direction = ("baseline", None) if a["arm"] == "baseline" else a["arm"].split("__")
        require(a["arm"] in ARMS and a["panel_id"] in PANELS and a["schedule"] == expected_schedule
                and a["direction_key"] == direction and a["family"] == "objects"
                and a["kind"] in ("choice", *DIAGNOSTICS) and a["delivery"] in DELIVERIES
                and type(a["world_slot"]) is int and a["world_slot"] in range(12)
                and type(a["menu_layout"]) is int and a["menu_layout"] in (0, 1), "Invalid fixed assignment")
        panel, world = a["panel_id"], a["semantic_world_id"]
        require(isinstance(world, str) and world, "Missing semantic identity")
        require(slots[panel].setdefault(world, a["world_slot"]) == a["world_slot"], "World slot changed")
        key = panel, world, a["arm"], a["kind"], a["delivery"], a["menu_layout"]
        require(key not in cells, "Duplicate semantic-world cell")
        r = deepcopy(observed[a["trial_id"]])
        require(all(r.get(k) == a[k] for k in METADATA), "Outcome metadata differs from assignment")
        require(r["status"] in ("completed", "execution_error", "execution_failure", "missing"), "Unknown status")
        if r["status"] == "completed":
            require(all(type(r.get(k)) is bool for k in ("valid", "thinking_closed", "cap_hit")),
                    "Completed rows need observed boolean flags")
            require(not r["valid"] or r["thinking_closed"], "A valid response must have closed thinking")
            if a["kind"] == "choice":
                require(r.get("correct") is None and (r.get("selected_service") in SERVICES if r["valid"]
                    else r.get("selected_service") is None), "Choice scoring/disposition differs")
            else:
                require(type(r.get("correct")) is bool and (not r["correct"] or r["valid"])
                        and r.get("selected_service") is None, "Diagnostic scoring differs")
        else:
            require(all(r.get(k) is None or r.get(k) is False for k in ("valid", "correct", "thinking_closed", "cap_hit"))
                    and r.get("selected_service") is None, "Unavailable rows fabricate observed responses")
        cells[key] = r
        stable = {k: a[k] for k in METADATA if k not in ("trial_id", "arm", "schedule", "direction_key")}
        require(bases.setdefault(a["base_trial_id"], stable) == stable, "Base branch changed across arms")
    worlds = {p: sorted(slots[p]) for p in PANELS}
    require(len(bases) == 288 and all(len(worlds[p]) == 12 and set(slots[p].values()) == set(range(12)) for p in PANELS)
            and len(set(worlds[PANELS[0]]) | set(worlds[PANELS[1]])) == 24, "Require24 distinct assigned worlds")
    branches = [("choice", d, m) for d in DELIVERIES for m in (0, 1)] + [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in DIAGNOSTICS]
    expected = {(p, w, a, k, d, m) for p in PANELS for w in worlds[p] for a in ARMS for k, d, m in branches}
    require(set(cells) == expected, "Incomplete world×arm×branch crossing")
    return cells, worlds


def counts(rows):
    completed = [r for r in rows if r["status"] == "completed"]
    return dict(assigned=len(rows), observed=len(completed), valid=sum(r["valid"] and r["thinking_closed"] for r in completed),
        correct=sum(r["correct"] is True for r in completed), cap_hits=sum(r["cap_hit"] for r in completed),
        unclosed=sum(not r["thinking_closed"] for r in completed), missing=sum(r["status"] == "missing" for r in rows),
        execution_failures=sum(r["status"] in ("execution_error", "execution_failure") for r in rows))


def task_readiness(cells):
    diagnostic_cells, choice_cells, arm_diagnostics = [], [], []
    grouped = defaultdict(list)
    for r in cells.values():
        grouped[r["arm"], r["panel_id"], r["kind"], r["delivery"]].append(r)
    for a in ARMS:
        diagnostics = []
        for p in PANELS:
            for k in DIAGNOSTICS:
                rows = [r for d in DELIVERIES for r in grouped[a, p, k, d]]
                diagnostics.extend(rows)
                c = counts(rows)
                passed = c["assigned"] == c["observed"] == 12 and c["valid"] >= 11 and c["correct"] >= 11 and c["cap_hits"] == 0
                diagnostic_cells.append(dict(arm=a, panel_id=p, kind=k, **c, passed=passed,
                                             original_pristine_format_passed=c["valid"] == 12))
            for d in DELIVERIES:
                c = counts(grouped[a, p, "choice", d])
                c["correct"] = None
                passed = c["assigned"] == c["observed"] == 24 and c["valid"] >= 23 and c["cap_hits"] == 0
                choice_cells.append(dict(arm=a, panel_id=p, delivery=d, **c, passed=passed))
        c = counts(diagnostics)
        arm_diagnostics.append(dict(arm=a, **c, passed=c["assigned"] == c["observed"] == 144 and c["valid"] >= 137 and c["cap_hits"] == 0))
    return dict(passed=all(c["passed"] for c in diagnostic_cells + choice_cells + arm_diagnostics),
        diagnostic_cells=diagnostic_cells, choice_cells=choice_cells, arm_diagnostics=arm_diagnostics,
        original_all_pristine_format_passed=all(c["original_pristine_format_passed"] for c in diagnostic_cells),
        interpretation="Observed instrument readiness only; no noninferiority, belief preservation or confidence equivalence.")


def aggregate_coefficients(terms):
    """Combine repeated response IDs BEFORE assigning any invalidity penalty."""
    coefficients = defaultdict(float)
    for response_id, coefficient in terms:
        require(isinstance(response_id, str) and type(coefficient) in (int, float) and np.isfinite(coefficient),
                "Invalid signed response coefficient")
        coefficients[response_id] += coefficient
    return {key: float(value) for key, value in sorted(coefficients.items()) if value != 0}


def contrast_specs():
    specs = []
    def add(name, role, terms):
        specs.append(dict(contrast=name, role=role, coefficients=aggregate_coefficients(terms)))
    for s in SCHEDULES:
        zero = arm("zero_sham", s)
        for d in DIRECTIONS:
            add(f"effect:{s}:{d}", "direction_minus_schedule_sham", [(arm(d, s), 1), (zero, -1)])
        add(f"baseline_minus_zero:{s}", "sham_descriptor", [("baseline", 1), (zero, -1)])
        for d in CONTROLS:
            add(f"selectivity:{s}:{d}", "target_minus_individual_control", [(arm(PRIMARY, s), 1), (arm(d, s), -1)])
    def interaction(d):
        return [(arm(d, "full"), 1), (arm(d, "record"), -1), (arm("zero_sham", "full"), -1), (arm("zero_sham", "record"), 1)]
    for d in DIRECTIONS:
        add(f"interaction:{d}", "coverage_interaction", interaction(d))
    for d in CONTROLS:
        add(f"interaction_selectivity:{d}", "target_minus_control_coverage_interaction",
            interaction(PRIMARY) + [(a, -c) for a, c in interaction(d)])
    return specs


def response_contrast(rows_by_arm, coefficients, endpoint, *, bound=False):
    """One world/menu/delivery; real response IDs are the uncertainty units."""
    terms, unique = [], {}
    for a, c in coefficients.items():
        r = rows_by_arm[a]
        require(r["trial_id"] not in unique or unique[r["trial_id"]] == r, "Conflicting shared response")
        unique[r["trial_id"]] = r
        terms.append((r["trial_id"], c))
    combined = aggregate_coefficients(terms)
    require(not bound or endpoint == NONREPORT, "Invalidity sensitivity applies only to valid_nonreport")
    value = 0.0
    for response_id, c in combined.items():
        observed = category(unique[response_id])
        y = int(observed in ("repeat", "pattern", "stop")) if endpoint == NONREPORT else int(observed == endpoint)
        value += c * y
        if bound and c < 0 and observed in FAILURES:
            value += c
    return value


def bootstrap_bank():
    rng = np.random.Generator(np.random.PCG64(SEED))
    return {p: rng.integers(0, 12, size=(BOOTSTRAPS, 12)) for p in PANELS}


def paired_summary(values, banks):
    draws = {p: values[p][banks[p]].mean(axis=1) for p in PANELS}
    estimates = {p: float(values[p].mean()) for p in PANELS}
    draws["pooled"] = (draws[PANELS[0]] + draws[PANELS[1]]) / 2
    estimates["pooled"] = (estimates[PANELS[0]] + estimates[PANELS[1]]) / 2
    return [dict(panel_id=p, estimate=estimates[p], ci95=np.quantile(draws[p], [.025, .975]).tolist(),
                 assigned_worlds=24 if p == "pooled" else 12, empirical_interval_degenerate=bool(np.ptp(draws[p]) == 0))
            for p in (*PANELS, "pooled")]


def decision_summary(rows, name, *, bound=False):
    selected = [r for r in rows if r["contrast"] == name and r["delivery"] == "redundant_evidence" and r["category"] == NONREPORT]
    keys = [(r["panel_id"], r["menu_layout"]) for r in selected]
    require(len(keys) == len(set(keys)) == 9 and set(keys) == {(p, m) for p in (*PANELS, "pooled") for m in (0, 1, "averaged")},
            "Incomplete fixed decision contrast")
    field = "lower_bound" if bound else "estimate"
    return dict(panel_differences={r["panel_id"]: r[field] for r in selected if r["menu_layout"] == "averaged"},
        pooled_menu_differences={str(r["menu_layout"]): r[field] for r in selected if r["panel_id"] == "pooled" and r["menu_layout"] in (0, 1)})


def threshold(summary, *, specificity=False):
    margin = .125 if specificity else .25
    return all(summary["panel_differences"][p] >= margin for p in PANELS) and all(
        v >= 0 if specificity else v > 0 for v in summary["pooled_menu_differences"].values())


def decide(effects, bounds, *, complete, integrity_passed, checks, readiness, audit_readiness):
    def assess(name, specificity=False):
        observed, conservative = decision_summary(effects, name), decision_summary(bounds, name, bound=True)
        op, bp = threshold(observed, specificity=specificity), threshold(conservative, specificity=specificity)
        return dict(observed=observed, conservative_invalidity=conservative,
                    observed_threshold_met=op, bounded_threshold_met=bp, passed=op and bp)
    primary = assess(f"effect:full:{PRIMARY}")
    selectivity = {d: assess(f"selectivity:full:{d}", True) for d in CONTROLS}
    interaction = assess(f"interaction:{PRIMARY}")
    interaction_selectivity = {d: assess(f"interaction_selectivity:{d}", True) for d in CONTROLS}
    specific = all(r["passed"] for r in selectivity.values())
    selective_interaction = interaction["passed"] and all(r["passed"] for r in interaction_selectivity.values())
    if not complete or not integrity_passed or not all(checks.values()):
        classification = "incomplete_or_integrity_failed"
    elif not readiness["passed"] or not audit_readiness["passed"]:
        classification = "readiness_failed"
    elif not primary["passed"]:
        classification = "no_substantial_full_curiosity_effect"
    elif not specific:
        classification = "full_effect_without_required_direction_selectivity"
    elif not selective_interaction:
        classification = "selective_full_effect_without_required_selective_coverage_interaction"
    else:
        classification = "qualifying_selective_coverage_signal"
    return dict(classification=classification, positive_screen=classification == "qualifying_selective_coverage_signal",
        full_effect_threshold_met=primary["passed"], full_selectivity_threshold_met=specific,
        coverage_interaction_threshold_met=interaction["passed"], selective_coverage_interaction_threshold_met=selective_interaction,
        primary_full_effect=primary, full_controls=selectivity, primary_coverage_interaction=interaction,
        interaction_controls=interaction_selectivity,
        interpretation="Exposed-development imposed-vector optional-policy perturbability only; no native recruitment, psychological specificity, preserved belief/confidence or inner-experience claim.",
        stopping="No automatic layer, sign, dose, prompt, vector, anchor or training search.")


def analyze_tables(assignments, outcomes, *, complete, integrity_passed, checks, audit_readiness):
    require(type(complete) is type(integrity_passed) is bool and set(checks) == set(GATES)
            and all(type(v) is bool for v in checks.values()) and type(audit_readiness.get("passed")) is bool,
            "Require explicit typed completeness, integrity and audit gates")
    cells, worlds = inventory(assignments, outcomes)
    ordered = [cells[key] for key in sorted(cells)]
    choices = [dict(r, category=category(r)) for r in ordered if r["kind"] == "choice"]
    readiness = task_readiness(cells)
    banks, cache = bootstrap_bank(), {}
    # Dense table retains every response, including failure mass. The only
    # contraction is the prespecified paired-menu average within each world.
    dense = np.zeros((2, 12, len(ARMS), 3, 2, len(ENDPOINTS)), dtype=np.float64)
    for pi, p in enumerate(PANELS):
        for wi, w in enumerate(worlds[p]):
            for ai, a in enumerate(ARMS):
                for di, d in enumerate(DELIVERIES):
                    for m in (0, 1):
                        c = category(cells[p, w, a, "choice", d, m])
                        dense[pi, wi, ai, di, m, CATEGORIES.index(c)] = 1
                        dense[pi, wi, ai, di, m, -1] = c in ("repeat", "pattern", "stop")
    unavailable = dense[..., [CATEGORIES.index(c) for c in FAILURES]].sum(axis=-1)
    specs = contrast_specs()
    endpoints, world_values, bounds, denominators = [], [], [], []
    def summarize(values):
        key = values.tobytes()
        if key not in cache:
            cache[key] = paired_summary({p: values[i] for i, p in enumerate(PANELS)}, banks)
        return cache[key]
    for spec in [dict(contrast=f"absolute:{a}", role="absolute_proportion", coefficients={a: 1}) for a in ARMS] + specs:
        coefficients = aggregate_coefficients(spec["coefficients"].items())
        c = np.array([coefficients.get(a, 0.) for a in ARMS])
        values = np.einsum("a,pwadmc->pwdmc", c, dense)
        penalties = np.einsum("a,pwadm->pwdm", np.minimum(c, 0), unavailable)
        absolute = spec["role"] == "absolute_proportion"
        for di, delivery in enumerate(DELIVERIES):
            for menu in (0, 1, "averaged"):
                v = values[:, :, di].mean(axis=2) if menu == "averaged" else values[:, :, di, menu]
                penalty = penalties[:, :, di].mean(axis=2) if menu == "averaged" else penalties[:, :, di, menu]
                for ci, endpoint in enumerate(ENDPOINTS):
                    metadata = dict(contrast=spec["contrast"], role=spec["role"], delivery=delivery, menu_layout=menu, category=endpoint)
                    for summary in summarize(v[:, :, ci]):
                        if absolute:
                            multiplier = (24 if summary["panel_id"] == "pooled" else 12) * (2 if menu == "averaged" else 1)
                            denominators.append(dict(arm=next(iter(coefficients)), delivery=delivery, menu_layout=menu, category=endpoint,
                                **summary, count=int(round(summary["estimate"] * multiplier)), assigned_choices=multiplier,
                                exclusive_category=endpoint != NONREPORT, proportion=summary["estimate"]))
                        else:
                            endpoints.append(dict(**metadata, **summary))
                    if not absolute:
                        world_values.extend(dict(**metadata, panel_id=p, values=v[pi, :, ci].tolist()) for pi, p in enumerate(PANELS))
                if not absolute:
                    lower = v[:, :, -1] + penalty
                    means = {p: float(lower[pi].mean()) for pi, p in enumerate(PANELS)}
                    means["pooled"] = (means[PANELS[0]] + means[PANELS[1]]) / 2
                    bounds.extend(dict(contrast=spec["contrast"], delivery=delivery, menu_layout=menu, category=NONREPORT,
                        panel_id=p, lower_bound=value, assigned_worlds=24 if p == "pooled" else 12,
                        world_values=None if p == "pooled" else lower[PANELS.index(p)].tolist()) for p, value in means.items())
    responses_complete = all(r["status"] == "completed" for r in ordered)
    decision = decide(endpoints, bounds, complete=complete and responses_complete, integrity_passed=integrity_passed, checks=checks,
                      readiness=readiness, audit_readiness=audit_readiness)
    return dict(schema_version=1, assay="curiosity_coverage_bridge_analysis", development_only=True,
        scientific_admission=False, causal_admission=False, analysis_metadata=deepcopy(ANALYSIS_METADATA),
        assigned=ASSIGNED, assigned_choices=4176, assigned_diagnostics=4176, assigned_worlds=24,
        complete=complete, integrity_passed=integrity_passed, all_assigned_responses_completed=responses_complete,
        checks=deepcopy(checks), audit_readiness=deepcopy(audit_readiness),
        task_readiness=readiness, decision=decision, outcomes=ordered, choices=choices, denominators=denominators,
        contrast_specs=specs, behavior_contrasts=endpoints, world_contrasts=world_values, decision_bounds=bounds,
        invalidity_scope="Aggregate coefficients on each response first; negative coefficients receive failure mass. These are sensitivity bounds, not observed behavior or confidence limits.",
        bootstrap=dict(draws=BOOTSTRAPS, seed=SEED, generator="PCG64", panel_order=list(PANELS), world_order=worlds,
            draw_bank_sha256=digest({p: banks[p].tolist() for p in PANELS}),
            world_values_order="Every world_contrasts.values and nonpooled bound.world_values follows this panel's world_order",
            unit="Whole world within panel, both menus and all arms/deliveries together",
            pooled="Equal panel means within each shared draw", interval="Descriptive95% percentile; never an advancement or equivalence test"))


def analyze_payload(payload):
    require(payload["analysis_metadata"] == ANALYSIS_METADATA
            and payload["assignment_sha256"] == digest(payload["assignments"]), "Prospective analysis/assignment binding differs")
    result = analyze_tables(payload["assignments"], payload["outcomes"], complete=payload["complete"],
        integrity_passed=payload["integrity_passed"], checks=payload["checks"], audit_readiness=payload["readiness"])
    result.update(input_sha256=digest(payload), assignment_sha256=payload["assignment_sha256"],
                  outcomes_sha256=digest(payload["outcomes"]), analyzer_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_artifacts_independently_verified=False,
                  input_scope="Arithmetic on supplied compact audited outcomes; upstream execution/artifact audit remains required")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Analysis output is immutable")
    raw = args.input.read_bytes()
    result = analyze_payload(json.loads(raw))
    result["input_byte_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
