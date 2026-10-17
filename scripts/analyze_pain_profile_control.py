"""Prospective five-arm analysis of the published first-choice benchmark.

The pure ``analyze_tables`` function consumes audited source-parser categories.
The CLI reproduces the raw audit and frozen inputs before exporting immutable
statistics. Neither interface performs inference or admits another paid phase.
"""

from copy import deepcopy
from fractions import Fraction
import hashlib

import numpy as np

from curiosity_study import pain_profile_design as design
from curiosity_study.artifacts import digest

ARMS, GROUPS = design.ARMS, design.GROUPS
POSITIONS, SEEDS = (0, 1), (1000, 2000)
CATEGORIES = ("target", "other", "malformed", "execution_error", "missing")
METADATA = ("trial_id", "arm", "context_id", "content_group", "initial_position", "seed_base", "seed")
BOOTSTRAPS, BOOTSTRAP_SEED = 10000, 2026101551
CONTRASTS = {
    "pain_full_minus_baseline": {"pain_full": 1, "baseline": -1},
    "fear_full_minus_baseline": {"fear_full": 1, "baseline": -1},
    "pain_full_minus_fear_full": {"pain_full": 1, "fear_full": -1},
    "pain_final_prefix_minus_baseline": {"pain_final_prefix": 1, "baseline": -1},
    "fear_final_prefix_minus_baseline": {"fear_final_prefix": 1, "baseline": -1},
    "pain_coverage": {"pain_full": 1, "pain_final_prefix": -1},
    "fear_coverage": {"fear_full": 1, "fear_final_prefix": -1},
    "coverage_difference_of_differences": {
        "pain_full": 1,
        "pain_final_prefix": -1,
        "fear_full": -1,
        "fear_final_prefix": 1,
    },
}
ANALYSIS_METADATA = dict(
    schema_version=1,
    arms=list(ARMS),
    all_assigned_denominators=True,
    primary_schedule_contrast="pain_full_minus_pain_final_prefix",
    parser="original_source_parser; strict exact-button parser is descriptive only",
    recognized_cap_answers_keep_category=True,
    minimum_cell_parseability=0.95,
    bootstrap_draws=BOOTSTRAPS,
    bootstrap_seed=BOOTSTRAP_SEED,
    bootstrap_unit="whole_first_context_stratified_by_original_content_group",
    intervals_control_qualification=False,
    fear_schedule_and_interaction_descriptive=True,
    independent_full_mask_and_schedule_decisions=True,
    narrow_cannot_rescue_full_mask=True,
    final_prefix_is_not_prior_qwen3_record_position=True,
    coverage_also_changes_aggregate_prefill_exposure=True,
    pain_specificity_claim=False,
    prior_qwen3_null_explanation=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(plan, rows):
    """Reconstruct the fixed full grid and retain absent observations as missing."""
    design.validate_plan(plan)
    require(isinstance(rows, (list, tuple)), "Rows must be a compact outcome sequence")
    assigned = {a["trial_id"]: a for a in plan["assignments"]}
    observed = {}
    for row in rows:
        require(isinstance(row, dict) and row.get("trial_id") in assigned, "Unknown outcome trial ID")
        require(row["trial_id"] not in observed, "Duplicate outcome trial ID")
        observed[row["trial_id"]] = row
    outcomes, cells = [], {}
    for assignment in plan["assignments"]:
        row = observed.get(assignment["trial_id"])
        if row is None:
            row = dict(
                **{key: assignment[key] for key in METADATA},
                status="missing",
                category=None,
                valid=None,
                cap_hit=None,
            )
        else:
            row = deepcopy(row)
            require(
                all(row.get(key) == assignment[key] for key in METADATA),
                "Outcome assignment metadata differs",
            )
            require(
                all(type(row[key]) is int for key in ("initial_position", "seed_base", "seed")),
                "Position and seed metadata must be integers",
            )
            for key in ("base_request_id", "gen_seed"):
                require(key not in row or row[key] == assignment[key], "Optional source identity differs")
        status = row.get("status")
        require(status in ("completed", "execution_error", "missing"), "Unknown outcome status")
        if status == "completed":
            require(row.get("category") in CATEGORIES[:3], "Completed response needs source parser category")
            require(
                type(row.get("valid")) is bool and type(row.get("cap_hit")) is bool,
                "Completed response needs boolean validity and cap flags",
            )
            require(row["valid"] == (row["category"] != "malformed"), "Source category and validity disagree")
        else:
            require(
                row.get("category") is None
                and all(row.get(key) is None or row.get(key) is False for key in ("valid", "cap_hit")),
                "Unavailable response cannot fabricate a parsed choice",
            )
        row["analysis_category"] = row["category"] if status == "completed" else status
        outcomes.append(row)
        key = (
            assignment["arm"],
            assignment["context_id"],
            assignment["initial_position"],
            assignment["seed_base"],
        )
        require(key not in cells, "Duplicate context/arm/position/seed cell")
        cells[key] = row
    return outcomes, cells


def count_rows(rows):
    counts = {category: sum(r["analysis_category"] == category for r in rows) for category in CATEGORIES}
    assigned = len(rows)
    return dict(
        assigned=assigned,
        completed=sum(r["status"] == "completed" for r in rows),
        **counts,
        parseable=counts["target"] + counts["other"],
        parseability=(counts["target"] + counts["other"]) / assigned,
        target_rate=counts["target"] / assigned,
        cap_hits=sum(r.get("cap_hit") is True for r in rows),
        parseability_passed=20 * (counts["target"] + counts["other"]) >= 19 * assigned,
    )


def _subset(rows, *, arm=None, group="all", position="averaged", seed="averaged"):
    return [
        r
        for r in rows
        if (arm is None or r["arm"] == arm)
        and (group == "all" or r["content_group"] == group)
        and (position == "averaged" or r["initial_position"] == position)
        and (seed == "averaged" or r["seed_base"] == seed)
    ]


def parseability_cells(outcomes):
    return [
        dict(
            arm=arm,
            content_group=group,
            initial_position=position,
            **count_rows(_subset(outcomes, arm=arm, group=group, position=position)),
        )
        for arm in ARMS
        for group in GROUPS
        for position in POSITIONS
    ]


def _difference(outcomes, left, right, position="averaged", *, conservative=False):
    a = count_rows(_subset(outcomes, arm=left, position=position))
    b = count_rows(_subset(outcomes, arm=right, position=position))
    require(a["assigned"] == b["assigned"], "Comparison denominators differ")
    unavailable = b["malformed"] + b["execution_error"] + b["missing"] if conservative else 0
    numerator = a["target"] - b["target"] - unavailable
    return dict(numerator=numerator, denominator=a["assigned"], estimate=numerator / a["assigned"])


def _at_least(value, threshold):
    return Fraction(value["numerator"], value["denominator"]) >= Fraction(threshold)


def _margin(outcomes, left, right, pooled, position, *, conservative=False):
    values = {
        str(p): _difference(outcomes, left, right, p, conservative=conservative)
        for p in ("averaged", *POSITIONS)
    }
    return dict(
        left_arm=left,
        right_arm=right,
        comparator_favorable=conservative,
        pooled_threshold=float(Fraction(pooled)),
        each_position_threshold=float(Fraction(position)),
        pooled=values["averaged"],
        positions={str(p): values[str(p)] for p in POSITIONS},
        threshold_met=_at_least(values["averaged"], pooled)
        and all(_at_least(values[str(p)], position) for p in POSITIONS),
    )


def decisions(outcomes, integrity):
    """Exact rational gate arithmetic; bootstrap intervals never decide admission."""
    require(
        isinstance(integrity, dict) and all(type(integrity.get(k)) is bool for k in ("complete", "passed")),
        "Require explicit boolean complete and passed integrity receipt fields",
    )
    measurements_complete = all(r["status"] == "completed" for r in outcomes)
    available = measurements_complete and integrity["complete"] and integrity["passed"]
    grammar = parseability_cells(outcomes)
    full_grammar = all(c["parseability_passed"] for c in grammar if c["arm"] in ARMS[:3])
    all_grammar = all(c["parseability_passed"] for c in grammar)
    rates = {arm: count_rows(_subset(outcomes, arm=arm)) for arm in ARMS[:3]}
    absolute = dict(
        pain_full_at_least_half=2 * rates["pain_full"]["target"] >= rates["pain_full"]["assigned"],
        baseline_at_most_five_percent=20 * rates["baseline"]["target"] <= rates["baseline"]["assigned"],
        fear_full_at_most_fifteen_percent=20 * rates["fear_full"]["target"]
        <= 3 * rates["fear_full"]["assigned"],
    )
    observed, conservative = {}, {}
    for comparator, pooled, position in (("baseline", "2/5", "1/4"), ("fear_full", "3/10", "1/5")):
        observed[comparator] = _margin(outcomes, "pain_full", comparator, pooled, position)
        conservative[comparator] = _margin(
            outcomes, "pain_full", comparator, pooled, position, conservative=True
        )
    full_thresholds = all(absolute.values()) and all(
        m["threshold_met"] for m in (*observed.values(), *conservative.values())
    )
    full_passed = available and full_grammar and full_thresholds
    directional = {}
    for direction, left, right in (
        ("positive", "pain_full", "pain_final_prefix"),
        ("negative", "pain_final_prefix", "pain_full"),
    ):
        obs = _margin(outcomes, left, right, "1/4", "1/10")
        bound = _margin(outcomes, left, right, "1/4", "1/10", conservative=True)
        directional[direction] = dict(
            observed=obs,
            comparator_favorable=bound,
            threshold_met=obs["threshold_met"] and bound["threshold_met"],
        )
    schedule_eligible = available and all_grammar
    qualified = (
        next((s for s, v in directional.items() if v["threshold_met"]), None) if schedule_eligible else None
    )
    return dict(
        measurements_complete=measurements_complete,
        integrity_receipt_complete=integrity["complete"],
        integrity_receipt_passed=integrity["passed"],
        eligible_for_behavioral_interpretation=available,
        parseability_cells=grammar,
        full_mask_positive_control=dict(
            passed=full_passed,
            parseability_passed=full_grammar,
            thresholds_met=full_thresholds,
            observed_absolute_rate_checks=absolute,
            observed_margins=observed,
            comparator_favorable_margins=conservative,
            classification=(
                "incomplete_or_integrity_failed"
                if not available
                else "parseability_failed"
                if not full_grammar
                else "qualified"
                if full_passed
                else "not_qualified"
            ),
        ),
        schedule_comparison=dict(
            eligible=schedule_eligible,
            parseability_passed=all_grammar,
            qualified_direction=qualified,
            appreciable_difference=qualified is not None,
            directional_tests=directional,
            classification=(
                "incomplete_or_integrity_failed"
                if not available
                else "parseability_failed"
                if not all_grammar
                else f"{qualified}_coverage_difference"
                if qualified
                else "no_qualifying_difference"
            ),
            interpretation="Coverage plus aggregate prefill exposure in this adapted model/assay; no equivalence or prior-Qwen3 explanation.",
        ),
        stopping="No automatic alternative dose, site, anchor, parser, batch, model, or paid retry; narrow behavior cannot rescue full-mask qualification.",
    )


class ContextBootstrap:
    """One bank per fixed stratum, reused across every endpoint and paired arm."""

    def __init__(self, contexts):
        self.contexts = contexts
        self.orders = {
            group: sorted(c["context_id"] for c in contexts if c["content_group"] == group)
            for group in GROUPS
        }
        self.index = {c["context_id"]: i for i, c in enumerate(contexts)}
        generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        self.banks = {
            group: generator.integers(0, len(self.orders[group]), size=(BOOTSTRAPS, len(self.orders[group])))
            for group in GROUPS
        }
        self.cache = {}

    def summarize(self, vector, group):
        vector = np.asarray(vector, dtype=np.float64)
        key = (group, vector.tobytes())
        if key in self.cache:
            return deepcopy(self.cache[key])
        groups = GROUPS if group == "all" else (group,)
        n = sum(len(self.orders[g]) for g in groups)
        samples = np.zeros(BOOTSTRAPS)
        total = 0.0
        for g in groups:
            values = vector[[self.index[c] for c in self.orders[g]]]
            total += float(values.sum())
            samples += values[self.banks[g]].mean(axis=1) * len(values) / n
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
        result = dict(
            estimate=total / n,
            ci95=interval,
            assigned_contexts=n,
            empirical_interval_degenerate=interval[0] == interval[1],
        )
        self.cache[key] = result
        return deepcopy(result)

    def metadata(self):
        return dict(
            draws=BOOTSTRAPS,
            seed=BOOTSTRAP_SEED,
            generator="numpy.PCG64",
            group_order=list(GROUPS),
            context_order=deepcopy(self.orders),
            group_weights={g: len(self.orders[g]) / len(self.contexts) for g in GROUPS},
            draw_bank_sha256=hashlib.sha256(
                b"".join(self.banks[g].astype("<i8").tobytes() for g in GROUPS)
            ).hexdigest(),
            array_encoding="group order; each (10000, group_count) C-order little-endian int64",
            unit="Whole first context; retain all arms, both positions and both seeds together",
            interval="Descriptive 95% percentile; not a qualification gate or population certainty",
        )


def analyze_tables(plan, rows, integrity):
    """Analyze the full fixed assignment plan, including explicit missing slots."""
    outcomes, cells = inventory(plan, rows)
    decision = decisions(outcomes, integrity)
    contexts = plan["contexts"]
    bootstrap = ContextBootstrap(contexts)
    estimates, contrasts = [], []
    for position in ("averaged", *POSITIONS):
        for seed in ("averaged", *SEEDS):
            positions = POSITIONS if position == "averaged" else (position,)
            seeds = SEEDS if seed == "averaged" else (seed,)
            vectors = {
                arm: np.array(
                    [
                        np.mean(
                            [
                                cells[arm, c["context_id"], p, s]["analysis_category"] == "target"
                                for p in positions
                                for s in seeds
                            ]
                        )
                        for c in contexts
                    ]
                )
                for arm in ARMS
            }
            for group in ("all", *GROUPS):
                scope = dict(content_group=group, initial_position=position, seed_base=seed)
                for arm in ARMS:
                    estimates.append(
                        dict(
                            arm=arm,
                            **scope,
                            **count_rows(
                                _subset(outcomes, arm=arm, group=group, position=position, seed=seed)
                            ),
                            **bootstrap.summarize(vectors[arm], group),
                        )
                    )
                for name, weights in CONTRASTS.items():
                    vector = sum(vectors[arm] * coefficient for arm, coefficient in weights.items())
                    contrasts.append(
                        dict(
                            contrast=name,
                            coefficients=weights.copy(),
                            **scope,
                            role="primary_schedule"
                            if name == "pain_coverage"
                            else "positive_control"
                            if name in ("pain_full_minus_baseline", "pain_full_minus_fear_full")
                            else "descriptive",
                            **bootstrap.summarize(vector, group),
                        )
                    )
    return dict(
        schema_version=1,
        assay="pain_profile_control_analysis",
        analysis_metadata=deepcopy(ANALYSIS_METADATA),
        plan_sha256=digest(plan),
        assignment_sha256=plan["assignment_sha256"],
        assigned=2020,
        assigned_per_arm=404,
        assigned_contexts=101,
        input_rows=len(rows),
        observed=sum(r["status"] == "completed" for r in outcomes),
        complete=decision["measurements_complete"] and integrity["complete"],
        integrity_passed=decision["eligible_for_behavioral_interpretation"],
        integrity=deepcopy(integrity),
        outcomes=outcomes,
        outcome_bindings_sha256=digest(outcomes),
        rates=estimates,
        contrasts=contrasts,
        bootstrap=bootstrap.metadata(),
        decision=decision,
        limitations=[
            "Source-parser categories are supplied by an independent audit; this function does not reparse tokens or verify GPU/source provenance.",
            "Missing and execution-failed assignments retain their denominators and prevent admission; observed values do not certify absent responses.",
            "Text clusters are a resampling unit, not proof of independent semantic worlds or population sampling.",
            "The adapted Qwen2.5 assay does not establish pain specificity, experience, or the cause of unchanged-Qwen3 nulls.",
        ],
    )


def analyze_run(run, audit_path=None):
    """Analyze a reproducible receipt, including authentic unavailable phases."""
    from pathlib import Path
    from curiosity_study.artifacts import read
    import audit_pain_profile_control as raw_audit

    run = Path(run).resolve()
    path = run / "phases/benchmark/audit.json" if audit_path is None else Path(audit_path).resolve()
    manifest = read(run / "manifest.json")
    receipt = read(path)
    reproduced = raw_audit.audit(run, "benchmark")
    require(receipt == reproduced, "Saved benchmark audit differs from independently reproduced raw evidence")
    require(receipt["phase"] == "benchmark" and receipt["manifest_sha256"] == digest(manifest)
            and receipt["assigned_rows"] == 2020 and len(receipt["outcomes"]) == 2020,
            "Audit belongs to another phase/manifest or omits assigned slots")
    fields = (*METADATA, "base_request_id", "gen_seed", "status", "category", "valid", "cap_hit",
              "raw", "record_sha256")
    rows = [{key: row[key] for key in fields if key in row} for row in receipt["outcomes"]]
    result = analyze_tables(manifest["plan"], rows,
                            dict(complete=receipt["complete"], passed=receipt["passed"]))
    result.update(manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
                  source_archive_sha256=receipt["archive_byte_sha256"],
                  freeze_byte_sha256=receipt["freeze_byte_sha256"], audit_sha256=digest(receipt),
                  audit_byte_sha256=raw_audit.sha(path), auditor_byte_sha256=receipt["auditor_byte_sha256"],
                  analyzer_byte_sha256=raw_audit.sha(__file__),
                  source_audit_recomputed=True, failed_integrity_can_qualify=False)
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from curiosity_study.artifacts import write_once

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New immutable analysis directory")
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "Analysis output already exists")
    output = analyze_run(arguments.run, arguments.audit)
    arguments.out.mkdir(parents=True, exist_ok=False)
    write_once(arguments.out / "analysis.json", output)
    write_once(arguments.out / "rates.json", output["rates"])
    write_once(arguments.out / "contrasts.json", output["contrasts"])
    print(json.dumps({key: output[key] for key in ("assigned", "observed", "complete", "integrity_passed", "decision")}))
