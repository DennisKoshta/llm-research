"""Direct paired causal contrasts, conditional on frozen fitted directions.

A plan is {alpha, bootstrap, seed, contrasts:[{name, left, right, orientation,
stratum?, families?, kinds?, task_revisions?, pooled_families?}]}. Selectors are
{direction, layer, position, dose_multiplier} or {arm: "baseline"}. A stratum is
null, {information_bits: 0|1}, {expected_target_information_gain_bits: 0|1}, or
{correct_count: 1|2|4}. Target margins use target_information_increase or
opposite_target_information, never an implicit full-world orientation. Every emitted family /
kind / revision endpoint, including requested family-balanced pools, belongs to
one Bonferroni family. Specify separate contrast entries for separate strata.
"""

from collections import defaultdict
import itertools
import math
import numpy as np
from .artifacts import digest
from . import target_relevant_information as target

REVISIONS = {"equal-uncertainty-v1", "relational-information-v1", "count-competence-v1", target.REVISION}
TARGET_OBJECTIVE = "designated_target_property"
WORLD_OBJECTIVE = "full_world_information_gain"
RESPONSE_OUTCOME_SCHEMA = "natural_valid_response_indicators_v1"
RESPONSE_OUTCOMES = (
    "valid_informative_response",
    "valid_uninformative_response",
    "valid_counterfactual_response",
)
BINARY_OUTCOMES = (*RESPONSE_OUTCOMES, "valid", "correct")
CATEGORICAL_TYPES = {
    "new": "categorical_information_response_score_not_logit",
    "repeat": "categorical_information_response_score_not_logit",
    "instrument": "categorical_report_minus_stop_response_score_not_logit",
    "memory": "categorical_correct_indicator_not_logit",
    "count": "categorical_correct_indicator_not_logit",
}


def natural_response_indicators(
    kind, index, information_bits, answer, valid, correct, *, action_label_mapping=None
):
    """An invalid answer can never acquire credit for a valid target response."""
    if type(valid) is not bool or type(correct) is not bool or (correct and not valid):
        raise ValueError("Natural response validity/correctness must be consistent booleans")
    if type(information_bits) is not int or information_bits not in (0, 1):
        raise ValueError("Natural response indicators require the binary source information gain")
    if kind in ("memory", "count"):
        return dict.fromkeys(RESPONSE_OUTCOMES)
    if kind in ("new", "repeat"):
        if valid and answer not in ("Yes", "No"):
            raise ValueError("A valid natural information response requires an allowed Yes/No label")
        informative = "Yes" if kind == "new" else "No"
        uninformative = "No" if kind == "new" else "Yes"
    elif kind == "instrument":
        if valid and answer not in ("A", "B", "C"):
            raise ValueError("A valid natural instrument response requires an allowed action label")
        mapping = action_label_mapping or dict(
            zip("ABC", list(itertools.permutations(("report", "pattern", "stop")))[index % 6])
        )
        if set(mapping) != set("ABC") or set(mapping.values()) not in (
            {"report", "pattern", "stop"},
            {"report", "reread", "stop"},
        ):
            raise ValueError("Natural instrument responses require an exact action-label mapping")
        informative = next(label for label, action in mapping.items() if action == "report")
        uninformative = next(label for label, action in mapping.items() if action == "stop")
    else:
        raise ValueError("Unsupported natural response kind")
    positive = int(valid and answer == informative)
    negative = int(valid and answer == uninformative)
    if correct != bool(positive if information_bits else negative):
        raise ValueError("Natural response correctness differs from source gain and the response label")
    return dict(
        valid_informative_response=positive,
        valid_uninformative_response=negative,
        valid_counterfactual_response=positive if information_bits == 0 else negative,
    )


def response_objective(descriptor):
    return TARGET_OBJECTIVE if descriptor["task_revision"] == target.REVISION else WORLD_OBJECTIVE


def response_gain(descriptor):
    """Never substitute full-world gain for the target task's response oracle."""
    if descriptor["task_revision"] != target.REVISION:
        if any(
            key in descriptor
            for key in (
                "response_objective_basis",
                "expected_target_information_gain_bits",
                "action_label_mapping",
            )
        ):
            raise ValueError("Natural world responses cannot carry target-objective scoring metadata")
        return descriptor["information_bits"]
    gain = descriptor.get("expected_target_information_gain_bits")
    expected_mapping = dict(
        zip("ABC", list(itertools.permutations(("report", "reread", "stop")))[descriptor["index"] % 6])
    )
    if (
        descriptor.get("response_objective_basis") != TARGET_OBJECTIVE
        or type(gain) is not int
        or gain not in (0, 1)
        or type(descriptor.get("information_bits")) is not int
        or descriptor.get("information_bits") != 1
        or type(descriptor.get("expected_world_information_gain_bits")) is not int
        or descriptor.get("expected_world_information_gain_bits") != 1
        or descriptor.get("information_bits_semantics") != "full_world_information_gain; not target gain"
        or descriptor.get("target_attribute") != 2
        or not isinstance(descriptor.get("target_property"), str)
        or not descriptor["target_property"]
        or descriptor.get("cell") is None
        or len(descriptor["cell"]) != 2
        or any(type(value) is not int or value not in (0, 1) for value in descriptor["cell"])
        or gain != int(descriptor["cell"][0] != descriptor["cell"][1])
        or descriptor.get("action_label_mapping") != expected_mapping
    ):
        raise ValueError("Natural target response objective, gain, and action mapping must be explicit")
    return gain


def descriptor_response_indicators(descriptor, answer, valid, correct):
    return natural_response_indicators(
        descriptor["kind"],
        descriptor["index"],
        response_gain(descriptor),
        answer,
        valid,
        correct,
        action_label_mapping=descriptor.get("action_label_mapping"),
    )


def validate_natural_response_row(row):
    """Independently bind binary scores to labels, source gain, and legacy scores."""
    if (
        row.get("response_outcome_schema") != RESPONSE_OUTCOME_SCHEMA
        or row.get("margin_type") != CATEGORICAL_TYPES.get(row.get("kind"))
        or row.get("margin_units") != "categorical response score; not logits"
    ):
        raise ValueError("Natural binary response schema and categorical outcome types must agree")
    expected = descriptor_response_indicators(row, row["answer"], row["valid"], row["correct"])
    for key, value in expected.items():
        if key not in row or row[key] != value or (value is not None and type(row[key]) is not int):
            raise ValueError("Natural binary response indicator differs from its independently scored label")
    margin = (
        int(row["valid"] and row["correct"])
        if row["kind"] in ("memory", "count")
        else expected["valid_informative_response"] - expected["valid_uninformative_response"]
    )
    if type(row.get("margin")) not in (int, float) or row["margin"] != margin:
        raise ValueError("Natural binary response indicators and legacy categorical margin disagree")
    return expected


def outcome_value(row, outcome):
    if outcome == "margin":
        return row["margin"]
    if outcome in RESPONSE_OUTCOMES:
        validate_natural_response_row(row)
        if row[outcome] is None:
            raise ValueError(
                "Selected response outcome is unsupported for memory/count; filter kinds explicitly"
            )
    elif outcome in ("valid", "correct"):
        if type(row.get(outcome)) is not bool:
            raise ValueError("Validity and correctness outcomes require explicit boolean values")
    else:
        raise ValueError("Unknown causal outcome selector")
    return int(row[outcome])


def source_descriptor(a):
    revision = a["task_revision"]
    if revision not in REVISIONS:
        raise ValueError("Unsupported contrast task revision")
    if revision == "count-competence-v1":
        first = a["correct_count"]
        if first not in (1, 2, 4) or first != 4 // (2 ** len(a["known"])):
            raise ValueError("Invalid count-competence cell")
    elif revision == target.REVISION:
        first = a["independent_report"]
        if type(first) is not int or first not in (0, 1) or a["known"] != []:
            raise ValueError("Invalid target-information factorial cell")
    else:
        first = a["correlated"] if revision == "relational-information-v1" else a["known"][0]
        if first not in (0, 1):
            raise ValueError("Invalid factorial cell")
    if a["query"] not in (0, 1):
        raise ValueError("Invalid queried property")
    gain = (
        int(a["query"] != first)
        if revision == "relational-information-v1"
        else int(a["query"] not in a["known"])
    )
    descriptor = dict(
        family=a["family"],
        kind=a["kind"],
        task_revision=revision,
        seed=a["seed"],
        index=a["index"],
        template=a.get("template", "table"),
        cell=(first, a["query"]),
        information_bits=gain,
        correct_count=a.get("correct_count"),
    )
    if revision == target.REVISION:
        task = target.task(a)
        descriptor.update(
            information_bits=int(task["information_bits"]),
            expected_world_information_gain_bits=int(task["expected_world_information_gain_bits"]),
            expected_target_information_gain_bits=int(task["expected_target_information_gain_bits"]),
            information_bits_semantics=task["information_bits_semantics"],
            response_objective_basis=TARGET_OBJECTIVE,
            target_attribute=task["target_attribute"],
            target_property=task["attributes"][task["target_attribute"]],
            action_label_mapping=task["action_label_mapping"],
        )
        response_gain(descriptor)
    return descriptor


def required_cells(revision):
    first = (1, 2, 4) if revision == "count-competence-v1" else (0, 1)
    return {(value, query) for value in first for query in (0, 1)}


def _validate_selector(selector):
    if selector == {"arm": "baseline"}:
        return
    if not isinstance(selector, dict) or set(selector) != {
        "direction",
        "layer",
        "position",
        "dose_multiplier",
    }:
        raise ValueError("Use an explicit four-field selector or {arm: baseline}")
    if not isinstance(selector["direction"], str) or not selector["direction"]:
        raise ValueError("A named direction is required")
    if type(selector["layer"]) is not int or selector["layer"] < 0:
        raise ValueError("A nonnegative integer layer is required")
    if selector["position"] not in ("record", "think_end", "decision", "prefill"):
        raise ValueError("Select a named captured position")
    if type(selector["dose_multiplier"]) not in (int, float) or not math.isfinite(
        selector["dose_multiplier"]
    ):
        raise ValueError("A finite dose multiplier is required")


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {"alpha", "bootstrap", "seed", "contrasts"}:
        raise ValueError("The contrast plan requires alpha, bootstrap, seed, and contrasts")
    if type(plan["alpha"]) not in (int, float) or not 0 < plan["alpha"] < 1:
        raise ValueError("Alpha must be between zero and one")
    if type(plan["bootstrap"]) is not int or plan["bootstrap"] < 2000:
        raise ValueError("At least 2000 bootstrap replicates are required")
    if type(plan["seed"]) is not int or plan["seed"] < 0:
        raise ValueError("A nonnegative bootstrap seed is required")
    if not isinstance(plan["contrasts"], list) or not plan["contrasts"]:
        raise ValueError("At least one planned contrast is required")
    names = set()
    required = {"name", "left", "right", "orientation"}
    allowed = required | {"stratum", "families", "kinds", "task_revisions", "pooled_families", "outcome"}
    for spec in plan["contrasts"]:
        if not isinstance(spec, dict) or not required <= set(spec) <= allowed:
            raise ValueError("Invalid contrast specification")
        if not isinstance(spec["name"], str) or not spec["name"] or spec["name"] in names:
            raise ValueError("Contrast names must be nonempty and unique")
        names.add(spec["name"])
        _validate_selector(spec["left"])
        _validate_selector(spec["right"])
        if spec["left"] == spec["right"]:
            raise ValueError("A contrast must select different arms")
        outcome = spec.get("outcome", "margin")
        if outcome not in ("margin", *BINARY_OUTCOMES):
            raise ValueError("Unknown causal outcome selector")
        if spec["orientation"] not in (
            "information_increase",
            "opposite_information",
            "target_information_increase",
            "opposite_target_information",
            "as_recorded",
        ):
            raise ValueError("Unknown contrast orientation")
        if outcome != "margin" and spec["orientation"] != "as_recorded":
            raise ValueError(
                "Binary outcomes require as_recorded orientation; do not reorient a scored response"
            )
        stratum = spec.get("stratum")
        if stratum is not None and not (
            isinstance(stratum, dict)
            and len(stratum) == 1
            and (
                (
                    set(stratum) == {"information_bits"}
                    and type(stratum["information_bits"]) is int
                    and stratum["information_bits"] in (0, 1)
                )
                or (
                    set(stratum) == {"expected_target_information_gain_bits"}
                    and type(stratum["expected_target_information_gain_bits"]) is int
                    and stratum["expected_target_information_gain_bits"] in (0, 1)
                )
                or (
                    set(stratum) == {"correct_count"}
                    and type(stratum["correct_count"]) is int
                    and stratum["correct_count"] in (1, 2, 4)
                )
            )
        ):
            raise ValueError("Use an explicit world-gain, target-gain, or count-one/two/four stratum")
        for field in ("families", "kinds", "task_revisions"):
            if field in spec and (
                not isinstance(spec[field], list)
                or not spec[field]
                or any(not isinstance(value, str) or not value for value in spec[field])
                or len(set(spec[field])) != len(spec[field])
            ):
                raise ValueError("Group filters must be nonempty lists of distinct strings")
        if type(spec.get("pooled_families", False)) is not bool:
            raise ValueError("pooled_families must be boolean")
    return plan


def _matches(row, selector):
    return all(row.get(key) == value for key, value in selector.items())


def _in_stratum(descriptor, stratum):
    return stratum is None or all(descriptor[key] == value for key, value in stratum.items())


def _paired_group(descriptors, by_prefix, spec, group):
    assigned = {
        key: d for key, d in descriptors.items() if (d["family"], d["kind"], d["task_revision"]) == group
    }
    revision = group[2]
    stratum = spec.get("stratum")
    if stratum and "correct_count" in stratum and revision != "count-competence-v1":
        raise ValueError("Count strata require the count-competence task")
    target_task = revision == target.REVISION
    if stratum and "expected_target_information_gain_bits" in stratum and not target_task:
        raise ValueError("Target gain strata require the target-information task")
    if target_task and stratum and "information_bits" in stratum:
        raise ValueError("Target gain strata must explicitly use expected_target_information_gain_bits")
    if target_task and spec["orientation"] in ("information_increase", "opposite_information"):
        raise ValueError("Target response margins require an explicit target-information orientation")
    if not target_task and spec["orientation"] in (
        "target_information_increase",
        "opposite_target_information",
    ):
        raise ValueError("Target-information orientation requires a target response objective")
    worlds, paired, paired_values = defaultdict(dict), {}, {}
    outcome = spec.get("outcome", "margin")
    missing = dict(left=0, right=0, both=0)
    for prefix, d in assigned.items():
        key = (d["seed"], d["index"], d["template"])
        if d["cell"] in worlds[key]:
            raise ValueError("Duplicate assigned world/template cell")
        worlds[key][d["cell"]] = prefix
        selected = [
            [r for r in by_prefix.get(prefix, []) if _matches(r, spec[side])] for side in ("left", "right")
        ]
        if any(len(rows) > 1 for rows in selected):
            raise ValueError("An explicit selector matched multiple rows within a prefix")
        left, right = selected
        missing["left"] += not left
        missing["right"] += not right
        missing["both"] += not left and not right
        if not left or not right:
            continue
        if outcome == "margin" and left[0]["margin_type"] != right[0]["margin_type"]:
            raise ValueError("Paired arms must measure the same semantic margin")
        orientation = (
            1 - 2 * response_gain(d)
            if spec["orientation"] in ("opposite_information", "opposite_target_information")
            else 1
        )
        first, second = outcome_value(left[0], outcome), outcome_value(right[0], outcome)
        paired_values[prefix] = first, second
        paired[prefix] = orientation * (first - second)
    cells = required_cells(revision)
    complete = {
        key: prefixes
        for key, prefixes in worlds.items()
        if set(prefixes) == cells and all(prefix in paired for prefix in prefixes.values())
    }
    clustered = defaultdict(list)
    analyzed = []
    for (seed, index, template), prefixes in complete.items():
        chosen = [p for p in prefixes.values() if _in_stratum(assigned[p], stratum)]
        if chosen:
            clustered[(seed, index)].append((template, float(np.mean([paired[p] for p in chosen]))))
            analyzed.extend(chosen)
    values = {
        key: float(np.mean([value for _, value in variants])) for key, variants in sorted(clustered.items())
    }
    counts = dict(
        assigned_prefixes=len(assigned),
        available_source_prefixes=sum(bool(by_prefix.get(p)) for p in assigned),
        paired_prefixes=len(paired),
        assigned_stratum_prefixes=sum(_in_stratum(d, stratum) for d in assigned.values()),
        paired_stratum_prefixes=sum(_in_stratum(assigned[p], stratum) for p in paired),
        analyzed_prefixes=len(analyzed),
        assigned_worlds=len({key[:2] for key in worlds}),
        assigned_world_templates=len(worlds),
        complete_assigned_world_templates=sum(set(prefixes) == cells for prefixes in worlds.values()),
        complete_paired_world_templates=len(complete),
        analyzed_worlds=len(values),
        required_cells_per_world_template=len(cells),
        missing_left=missing["left"],
        missing_right=missing["right"],
        missing_both=missing["both"],
    )
    if outcome != "margin":
        counts.update(
            analyzed_left_positive_responses=sum(paired_values[p][0] for p in analyzed),
            analyzed_right_positive_responses=sum(paired_values[p][1] for p in analyzed),
            binary_response_denominator=len(analyzed),
        )
    per_world = [
        dict(
            family=group[0],
            seed=seed,
            index=index,
            paired_difference=value,
            templates=[template for template, _ in clustered[(seed, index)]],
        )
        for (seed, index), value in values.items()
    ]
    return dict(counts=counts, values=values, per_world=per_world)


def _bootstrap(family_values, plan, multiplicity, endpoint):
    arrays = {name: np.asarray(values, dtype=float) for name, values in sorted(family_values.items())}
    effective_alpha = plan["alpha"] / multiplicity
    result = dict(
        mean=None,
        interval=None,
        bonferroni_interval=None,
        alpha=plan["alpha"],
        bonferroni_alpha=effective_alpha,
        bootstrap_replicates=plan["bootstrap"],
        expected_replicates_per_adjusted_tail=plan["bootstrap"] * effective_alpha / 2,
        resampling_worlds_by_family={name: len(values) for name, values in arrays.items()},
        bootstrap_degenerate=None,
    )
    if any(not len(values) for values in arrays.values()):
        return result
    seed = int(digest(dict(seed=plan["seed"], endpoint=endpoint))[:16], 16)
    rng = np.random.default_rng(seed)
    samples = np.zeros(plan["bootstrap"])
    for values in arrays.values():
        for start in range(0, plan["bootstrap"], 1000):
            size = min(1000, plan["bootstrap"] - start)
            samples[start : start + size] += values[rng.integers(len(values), size=(size, len(values)))].mean(
                1
            ) / len(arrays)
    result.update(
        mean=float(np.mean([values.mean() for values in arrays.values()])),
        interval=np.quantile(samples, [plan["alpha"] / 2, 1 - plan["alpha"] / 2]).tolist(),
        bonferroni_interval=np.quantile(samples, [effective_alpha / 2, 1 - effective_alpha / 2]).tolist(),
        bootstrap_degenerate=bool(np.ptp(samples) == 0),
    )
    return result


def analyze_contrasts(compact, assignments, plan):
    """Analyze explicit arm pairs; assignments include every excluded source too."""
    validate_plan(plan)
    descriptors = {}
    for a in assignments:
        if a["trial_id"] in descriptors:
            raise ValueError("Duplicate assigned source ID")
        descriptors[a["trial_id"]] = source_descriptor(a)
    if not descriptors:
        raise ValueError("Assigned source metadata is required")
    by_prefix, row_ids = defaultdict(list), set()
    for row in compact:
        if row["trial_id"] in row_ids or row["prefix_id"] not in descriptors:
            raise ValueError("Duplicate measurement or unknown source prefix")
        row_ids.add(row["trial_id"])
        for field, expected in descriptors[row["prefix_id"]].items():
            got = tuple(row[field]) if field == "cell" else row[field]
            if got != expected:
                raise ValueError(f"Compact source metadata mismatch: {field}")
        if type(row["margin"]) not in (int, float) or not math.isfinite(row["margin"]):
            raise ValueError("Every retained outcome needs a finite semantic margin")
        if row.get("margin_type") in CATEGORICAL_TYPES.values() or "response_outcome_schema" in row:
            validate_natural_response_row(row)
        by_prefix[row["prefix_id"]].append(row)
    assigned_groups = sorted({(d["family"], d["kind"], d["task_revision"]) for d in descriptors.values()})
    endpoints = []
    for spec in plan["contrasts"]:
        for field, offset in (("families", 0), ("kinds", 1), ("task_revisions", 2)):
            if field in spec and not set(spec[field]) <= {group[offset] for group in assigned_groups}:
                raise ValueError("A planned group filter is absent from the assigned design")
        groups = [
            group
            for group in assigned_groups
            if all(
                field not in spec or group[offset] in spec[field]
                for field, offset in (("families", 0), ("kinds", 1), ("task_revisions", 2))
            )
        ]
        if not groups:
            raise ValueError("A planned contrast selects no assigned groups")
        if spec.get("outcome", "margin") in RESPONSE_OUTCOMES and any(
            group[1] in ("memory", "count") for group in groups
        ):
            raise ValueError(
                "Selected response outcome is unsupported for memory/count; filter kinds explicitly"
            )
        calculations = {group: _paired_group(descriptors, by_prefix, spec, group) for group in groups}
        for group in groups:
            endpoints.append((spec, group[1:], [group[0]], {group[0]: calculations[group]}, False))
        if spec.get("pooled_families", False):
            for kind, revision in sorted({group[1:] for group in groups}):
                members = {group[0]: calculations[group] for group in groups if group[1:] == (kind, revision)}
                if len(members) < 2:
                    raise ValueError("Family-balanced pooling requires at least two assigned families")
                endpoints.append((spec, (kind, revision), sorted(members), members, True))
    multiplicity = len(endpoints)
    results = []
    for spec, (kind, revision), families, members, pooled in endpoints:
        identity = dict(
            name=spec["name"],
            kind=kind,
            task_revision=revision,
            families=families,
            pooled_families=pooled,
            outcome=spec.get("outcome", "margin"),
        )
        counts = {family: result["counts"] for family, result in members.items()}
        results.append(
            dict(
                **identity,
                response_objective_basis=TARGET_OBJECTIVE if revision == target.REVISION else WORLD_OBJECTIVE,
                response_gain_field="expected_target_information_gain_bits"
                if revision == target.REVISION
                else "information_bits",
                control_interpretation="Constant four-candidate diagnostic; not a balanced count-competence control"
                if revision == target.REVISION and kind == "count"
                else None,
                left=spec["left"],
                right=spec["right"],
                orientation=spec["orientation"],
                stratum=spec.get("stratum"),
                counts_by_family=counts,
                totals={
                    key: sum(value[key] for value in counts.values())
                    for key in next(iter(counts.values()))
                    if key != "required_cells_per_world_template"
                },
                estimate=_bootstrap(
                    {family: list(result["values"].values()) for family, result in members.items()},
                    plan,
                    multiplicity,
                    identity,
                ),
                paired_world_values=[row for result in members.values() for row in result["per_world"]],
            )
        )
    return dict(
        schema_version=1,
        plan_sha256=digest(plan),
        planned_endpoints=multiplicity,
        assigned_sources=len(descriptors),
        available_source_prefixes=len(by_prefix),
        measurements=len(compact),
        multiplicity_rule="Bonferroni alpha across every planned family/kind/revision endpoint, including requested pooled endpoints and endpoints with no eligible worlds",
        interval_method="Percentile bootstrap of directly paired world differences; family pools weight families equally and resample within each family. Adjusted intervals are approximate, not exact coverage guarantees.",
        interpretation="Intervals condition on the frozen fitted direction and schedule. Invalid completions remain included in every applicable binary-response denominator with score zero. Completeness precedes stratification; templates are averaged within seed/index. Legacy signed margins are descriptive: invalidation can create an apparent opposite-information shift. Valid-counterfactual indicators require an actual valid opposite-gain response and use as-recorded orientation. No p-values are estimated.",
        results=results,
    )
