"""Balanced candidate-count competence control, with variable total uncertainty.

The answer is 1, 2, or 4 equally often. This is a basic-competence control for
later interventions, not an equal-uncertainty information-gain contrast. A world
index is a sampling unit, not a claim of semantic novelty: the object vocabulary
has a finite number of distinct underlying property combinations.
"""

from collections import defaultdict
import hashlib
import itertools
from pathlib import Path
import random

from . import information_state as base
from .artifacts import digest
from .runner import source_hash

REVISION = "count-competence-v1"
CORRECT_COUNTS = (1, 2, 4)


def _validate(a):
    if a["kind"] != "count":
        raise ValueError("The competence control contains only candidate-count questions")
    if a["known"] not in ([], [0], [1], [0, 1]):
        raise ValueError("Known facts must be a subset of the two independent properties")
    if a["query"] not in (0, 1):
        raise ValueError("The available report must concern one of the two properties")
    if a.get("correct_count") != 4 // (2 ** len(a["known"])):
        raise ValueError("The declared answer does not match the number of observed facts")


def task(a):
    _validate(a)
    return base.task(a)


def context(a):
    _validate(a)
    return base.context(a)


def prompt(a):
    _validate(a)
    return base.prompt(a)


def inference_request(a):
    _validate(a)
    return base.inference_request(a)


def record_from_inference(a, inference):
    _validate(a)
    return base.record_from_inference(a, inference)


def assignments(items=100, seed=2026101411, template="table", start_index=0):
    """Keep all six branches of each sampled world adjacent in the schedule.

    Each world has both queries at each count. For the two-candidate condition,
    the observed property alternates across world indices. Therefore any even
    consecutive set of world indices balances which property is observed, while
    every individual world balances queries and correct count labels exactly.
    """
    if items < 1 or start_index < 0:
        raise ValueError("Require positive item count and nonnegative start index")
    if template not in ("table", "prose"):
        raise ValueError("Unknown template")
    groups = []
    for family, index in itertools.product(("objects", "machines"), range(start_index, start_index + items)):
        world_id = digest(dict(task_revision=REVISION, seed=seed, family=family, index=index))
        group = []
        for correct_count, query in itertools.product(CORRECT_COUNTS, range(2)):
            known = {1: [0, 1], 2: [index % 2], 4: []}[correct_count]
            key = f"count-competence-{seed}-{family}-{index:04d}-count{correct_count}-query{query}-{template}"
            group.append(
                dict(
                    assay="thinking_information",
                    task_revision=REVISION,
                    family=family,
                    index=index,
                    seed=seed,
                    world_id=world_id,
                    competence_group=world_id,
                    known=known,
                    observed_fact_count=len(known),
                    correct_count=correct_count,
                    query=query,
                    kind="count",
                    template=template,
                    vocabulary="heldout",
                    trial_id=key,
                    sampling_seed=random.Random(f"count-competence-sample:{key}").randrange(2**31),
                )
            )
        random.Random(f"count-competence-branches:{world_id}").shuffle(group)
        groups.append(group)
    random.Random(f"count-competence-world-order:{seed}:{template}").shuffle(groups)
    return [a for group in groups for a in group]


def build_manifest(model, items=100, seed=2026101411, template="table", start_index=0):
    # Local import keeps this task module independent of execution dispatch.
    from .thinking_information import CAPTURE_PLAN, EXECUTION

    if model["model_id"] != "Qwen/Qwen3-32B":
        raise ValueError("This competence validation is pinned to Qwen3-32B")
    rows = assignments(items, seed, template, start_index)
    for a in rows:
        a["execution"] = EXECUTION.copy()
    return dict(
        schema_version=3,
        assay="thinking_information",
        phase="validation",
        task_revision=REVISION,
        purpose="Balanced basic candidate-count competence control; not an equal-uncertainty mechanism contrast.",
        model=model,
        execution_gpu="H200",
        execution=EXECUTION.copy(),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        inference_revision="thinking-2048-v1",
        capture_plan=CAPTURE_PLAN,
        task_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation_criteria=dict(
            minimum_distinct_items_per_cell=100,
            minimum_accuracy=0.9,
            minimum_valid_format=0.95,
            cell_axes=["family", "correct_count"],
            item_identity="task_revision, seed, family, index; all six branches retained together",
            distinctness="Distinct sampled world indices, not necessarily semantically unique worlds.",
        ),
        balance=dict(
            correct_counts=list(CORRECT_COUNTS),
            branches_per_world=6,
            queries_per_count=[0, 1],
            one_fact_observed_property="index modulo 2; balanced for any even consecutive index interval",
        ),
        invalid_output_policy="terminal_no_retry",
        response_grammar="One of 1, 2, 3, 4, optional single terminal period, surrounding whitespace ignored; no other text. A closing thinking delimiter is required.",
        capture_position_rule="Same last-delimiter and pre-answer rules as the thinking-state validation.",
    )


def evaluate(manifest, records):
    criteria = manifest["validation_criteria"]
    axes = criteria.get("cell_axes", ["family", "correct_count"])
    if axes == ["family", "correct_count", "query"]:
        return _evaluate_query_cells(manifest, records)
    if axes != ["family", "correct_count"]:
        raise ValueError("Unsupported count validation cell axes")
    minimum = criteria["minimum_distinct_items_per_cell"]
    cells = []
    for family, correct_count in itertools.product(("objects", "machines"), CORRECT_COUNTS):
        aa = [
            a
            for a in manifest["diagnostics"]
            if a["family"] == family and a["correct_count"] == correct_count
        ]
        n = len(aa)
        correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in aa)
        valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in aa)
        complete = sum(a["trial_id"] in records for a in aa)
        by_world = defaultdict(list)
        for a in aa:
            by_world[(a["seed"], a["family"], a["index"])].append(a)
        balanced = all(
            len(group) == 2 and {a["query"] for a in group} == {0, 1} for group in by_world.values()
        )
        cells.append(
            dict(
                family=family,
                correct_count=correct_count,
                n=n,
                items=len(by_world),
                correct=correct,
                valid=valid,
                completed=complete,
                balanced_query_pairs=balanced,
                passed=bool(n)
                and complete == n
                and len(by_world) >= minimum
                and balanced
                and correct / n >= criteria["minimum_accuracy"]
                and valid / n >= criteria["minimum_valid_format"],
            )
        )
    return dict(passed=bool(cells) and all(c["passed"] for c in cells), cells=cells)


def _evaluate_query_cells(manifest, records):
    """Opt-in query-specific gates; legacy six pooled-query gates stay unchanged."""
    criteria = manifest["validation_criteria"]
    assignments = manifest["diagnostics"]
    worlds = defaultdict(list)
    for a in assignments:
        worlds[(a["task_revision"], a["seed"], a["family"], a["index"])].append(a)
    required = {(count, query) for count in CORRECT_COUNTS for query in (0, 1)}
    complete_worlds = bool(worlds) and all(
        len(rows) == 6 and {(a["correct_count"], a["query"]) for a in rows} == required
        for rows in worlds.values()
    )
    unique_assignments = len({a["trial_id"] for a in assignments}) == len(assignments)
    cells = []
    for family, count, query in itertools.product(("objects", "machines"), CORRECT_COUNTS, (0, 1)):
        rows = [
            a for a in assignments if (a["family"], a["correct_count"], a["query"]) == (family, count, query)
        ]
        n = len(rows)
        identities = {(a["task_revision"], a["seed"], a["family"], a["index"]) for a in rows}
        correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in rows)
        valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in rows)
        completed = sum(a["trial_id"] in records for a in rows)
        cells.append(
            dict(
                family=family,
                correct_count=count,
                query=query,
                n=n,
                items=len(identities),
                correct=correct,
                valid=valid,
                completed=completed,
                one_assignment_per_world=len(identities) == n,
                passed=bool(n)
                and complete_worlds
                and unique_assignments
                and completed == n
                and len(identities) == n
                and len(identities) >= criteria["minimum_distinct_items_per_cell"]
                and correct / n >= criteria["minimum_accuracy"]
                and valid / n >= criteria["minimum_valid_format"],
            )
        )
    return dict(
        passed=all(c["passed"] for c in cells),
        cells=cells,
        all_worlds_have_six_cells=complete_worlds,
        unique_assignment_ids=unique_assignments,
    )


def render(manifest, records):
    result = evaluate(manifest, records)
    lines = [
        "# Balanced candidate-count competence validation",
        "",
        "Total uncertainty varies across this basic-competence control. Distinct world indices are sampling units, not guarantees of semantic novelty.",
        "",
        f"All cell gates passed: {result['passed']}.",
        "",
        "| Family | Correct count | Correct | Valid | Completed | Assigned | World indices | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in result["cells"]:
        lines.append(
            f"| {c['family']} | {c['correct_count']} | {c['correct']} | {c['valid']} | {c['completed']} | {c['n']} | {c['items']} | {c['passed']} |"
        )
    if manifest["validation_criteria"].get("cell_axes") == ["family", "correct_count", "query"]:
        lines[6:] = [
            "| Family | Correct count | Query | Correct | Valid | Completed | Assigned | World indices | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            *[
                f"| {c['family']} | {c['correct_count']} | {c['query']} | {c['correct']} | {c['valid']} | {c['completed']} | {c['n']} | {c['items']} | {c['passed']} |"
                for c in result["cells"]
            ],
        ]
    return "\n".join(lines) + "\n"
