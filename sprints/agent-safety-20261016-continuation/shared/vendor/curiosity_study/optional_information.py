"""Optional information acquisition feasibility; no action is scored as correct."""

import itertools
import random
from collections import defaultdict
from . import information_state as state
from .artifacts import digest

REVISION = "optional-information-v1"
SERVICES = ("report", "pattern", "stop")
MAPPINGS = tuple(itertools.permutations(SERVICES))
SEED = 2026101415
ITEMS = 12


def _check_choice(assignment):
    if assignment["kind"] != "choice":
        raise ValueError("Optional acquisition only supports the choice question")


def task(assignment):
    _check_choice(assignment)
    return state.task(assignment)


def context(assignment):
    _check_choice(assignment)
    return state.context(assignment)


def prompt(assignment):
    _check_choice(assignment)
    return state.prompt(assignment)


def inference_request(assignment):
    _check_choice(assignment)
    # The thinking-information wrapper sets thinking=True and the 2048-token cap.
    return state.inference_request(assignment)


def record_from_inference(assignment, inference):
    _check_choice(assignment)
    record = state.record_from_inference(assignment, inference)
    mapping = dict(zip("ABC", MAPPINGS[assignment["index"] % 6]))
    record.update(
        expected=None,
        correct=None,
        action_mapping=mapping,
        selected_service=mapping.get(record["parsed_answer"]) if record["valid"] else None,
    )
    return record


def assignments(items=ITEMS, seed=SEED, template="table"):
    rows = state.assignments(items=items, seed=seed, kinds=("choice",), template=template)
    for a in rows:
        key = f"optional-{seed}-{a['family']}-{a['index']:04d}-known{a['known'][0]}-query{a['query']}-{template}-choice"
        a.update(
            assay="thinking_information",
            task_revision=REVISION,
            vocabulary="heldout",
            trial_id=key,
            sampling_seed=int(digest(key)[:8], 16),
        )
    random.Random(f"optional-order:{seed}").shuffle(rows)
    return rows


def _summary(assigned, records):
    counts = {service: 0 for service in SERVICES}
    invalid = missing = failures = closed = cap_hits = completed = 0
    for a in assigned:
        row = records.get(a["trial_id"])
        if row is None or row.get("terminal") is not True:
            missing += 1
            continue
        completed += 1
        if row.get("failure"):
            failures += 1
            continue
        is_closed = row.get("thinking_closed") is True
        closed += is_closed
        inference = row.get("inference", {})
        cap_hits += (
            len(inference.get("generated_token_ids", [])) == 2048 and inference.get("ended_with_eos") is False
        )
        mapping = dict(zip("ABC", MAPPINGS[a["index"] % 6]))
        service = mapping.get(row.get("parsed_answer"))
        valid = row.get("valid") is True and is_closed and service is not None
        if valid:
            if row.get("selected_service") != service or row.get("action_mapping") != mapping:
                raise ValueError("Optional service decoding disagrees with its assigned action mapping")
            counts[service] += 1
        else:
            invalid += 1
    n, valid = len(assigned), sum(counts.values())
    return dict(
        assigned=n,
        worlds=len({(a["family"], a["seed"], a["index"]) for a in assigned}),
        completed=completed,
        missing=missing,
        execution_failures=failures,
        thinking_closed=closed,
        valid_closed=valid,
        valid_closed_fraction=valid / n if n else None,
        invalid_or_unclosed=invalid,
        cap_hits=cap_hits,
        service_counts=counts,
        service_proportions_of_assigned={
            service: value / n if n else None for service, value in counts.items()
        },
        service_proportions_of_valid={
            service: value / valid if valid else None for service, value in counts.items()
        },
    )


def evaluate(manifest, records):
    assigned = manifest["diagnostics"]
    if any(a["kind"] != "choice" or a["task_revision"] != REVISION for a in assigned):
        raise ValueError("Optional feasibility manifest contains another task")
    identifiers = {a["trial_id"] for a in assigned}
    if len(identifiers) != len(assigned):
        raise ValueError("Duplicate optional assignment")
    criteria = manifest["feasibility_criteria"]
    cells = defaultdict(list)
    for a in assigned:
        cells[(a["family"], a["seed"], a["index"])].append((tuple(a["known"]), a["query"]))
    overall = _summary(assigned, records)
    families = [
        dict(family=family, **_summary([a for a in assigned if a["family"] == family], records))
        for family in ("objects", "machines")
    ]
    by_gain = [
        dict(
            family=family,
            information_bits=gain,
            **_summary(
                [a for a in assigned if a["family"] == family and int(a["query"] not in a["known"]) == gain],
                records,
            ),
        )
        for family, gain in itertools.product(("objects", "machines"), (0, 1))
    ]
    strata = [
        dict(
            family=family,
            mapping_index=mapping_index,
            action_mapping=dict(zip("ABC", MAPPINGS[mapping_index])),
            information_bits=gain,
            **_summary(
                [
                    a
                    for a in assigned
                    if a["family"] == family
                    and a["index"] % 6 == mapping_index
                    and int(a["query"] not in a["known"]) == gain
                ],
                records,
            ),
        )
        for family, mapping_index, gain in itertools.product(("objects", "machines"), range(6), (0, 1))
    ]
    checks = dict(
        assigned_sample=len(assigned) == criteria["assigned_branches"],
        source_coverage=set(records) == identifiers and overall["completed"] == len(assigned),
        no_execution_failures=overall["execution_failures"] == 0,
        no_accuracy_scoring=all(
            r.get("expected") is None and r.get("correct") is None for r in records.values()
        ),
        overall_valid_closed=overall["valid_closed_fraction"] is not None
        and overall["valid_closed_fraction"] >= criteria["minimum_valid_closed_overall"],
        every_family_valid_closed=all(
            f["valid_closed_fraction"] is not None
            and f["valid_closed_fraction"] >= criteria["minimum_valid_closed_each_family"]
            for f in families
        ),
        worlds_per_family=all(f["worlds"] == criteria["worlds_each_family"] for f in families),
        complete_factorial_worlds=all(
            len(values) == 4 and set(values) == {((0,), 0), ((0,), 1), ((1,), 0), ((1,), 1)}
            for values in cells.values()
        ),
        all_mapping_gain_strata_assigned=all(s["assigned"] == 4 for s in strata),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        overall=overall,
        families=families,
        family_information_gain=by_gain,
        mapping_information_strata=strata,
        interpretation="Feasibility only: format/closure and execution integrity. No accuracy or preference threshold applies. Choice proportions retain assigned-denominator and valid-only versions. Optional report preference alone does not establish curiosity, a motivational mechanism, or subjective experience.",
    )


def render(manifest, records):
    result = evaluate(manifest, records)
    lines = [
        "# Optional information acquisition feasibility",
        "",
        f"Feasibility gates passed: {result['passed']}.",
        "",
        result["interpretation"],
        "",
        "| Group | Assigned | Worlds | Valid and closed | Cap hits | Report | Pattern | Stop | Invalid/unclosed | Missing | Execution failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("Overall", result["overall"])]
    groups.extend((row["family"], row) for row in result["families"])
    groups.extend(
        (f"{row['family']}, gain {row['information_bits']}", row) for row in result["family_information_gain"]
    )
    for label, row in groups:
        counts = row["service_counts"]
        lines.append(
            f"| {label} | {row['assigned']} | {row['worlds']} | {row['valid_closed']} | {row['cap_hits']} | {counts['report']} | {counts['pattern']} | {counts['stop']} | {row['invalid_or_unclosed']} | {row['missing']} | {row['execution_failures']} |"
        )
    lines.extend(
        [
            "",
            "The JSON readout contains service proportions using both all assigned branches and valid closed choices as denominators.",
            "",
            "| Family | Mapping A/B/C | Gain | Assigned | Valid and closed | Report | Pattern | Stop | Cap hits |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["mapping_information_strata"]:
        mapping = "/".join(row["action_mapping"][label] for label in "ABC")
        counts = row["service_counts"]
        lines.append(
            f"| {row['family']} | {mapping} | {row['information_bits']} | {row['assigned']} | {row['valid_closed']} | {counts['report']} | {counts['pattern']} | {counts['stop']} | {row['cap_hits']} |"
        )
    return "\n".join(lines) + "\n"
