"""One-credit information-choice assay. Separate from the failed priced instrument."""

import copy
import itertools
import random
import re
from dataclasses import dataclass, field
from .artifacts import digest
from .tasks import generate, describe, describe_observation
from .oracle import compatible, information

KINDS = ("identify_x", "identify_y", "instrument_x", "instrument_y", "exclusive_x", "exclusive_y", "decline")
CONDITIONS = ("both_unresolved", "x_resolved", "y_resolved")


def assignments(items=100, seed=20261006):
    if items < 1:
        raise ValueError("Need at least one pair")
    result = []
    permutations = list(itertools.permutations(("X", "Y", "stop")))
    for family in ("machines", "objects"):
        offset = random.Random(f"credit-mapping:{seed}:{family}").randrange(6)
        for index in range(items):
            rng = random.Random(f"credit:{seed}:{family}:{index}")
            mapping = dict(zip("ABC", permutations[(index + offset) % 6]))
            menu_order = list("ABC")
            rng.shuffle(menu_order)
            # Balance the existing task templates across mystery names.
            item_indices = [2 * index, 2 * index + 1]
            if index % 2:
                item_indices.reverse()
            for layout in ("XY", "YX"):
                for condition in CONDITIONS:
                    result.append(
                        {
                            "assay": "one_credit",
                            "family": family,
                            "index": index,
                            "seed": seed,
                            "item_indices": item_indices.copy(),
                            "condition": condition,
                            "layout": layout,
                            "mapping": mapping.copy(),
                            "menu_order": menu_order.copy(),
                            "sampling_seed": rng.randrange(2**31),
                            "trial_id": f"credit-{family}-{index:04d}-{layout}-{condition}",
                        }
                    )
    random.Random(f"credit-run-order:{seed}").shuffle(result)
    return result


def diagnostic_assignments(behavior):
    seen, result = set(), []
    for a in behavior:
        key = a["family"], a["index"]
        if key in seen:
            continue
        seen.add(key)
        for kind in KINDS:
            diagnostic = copy.deepcopy(a)
            diagnostic["kind"] = kind
            diagnostic["layout"] = "XY" if a["index"] % 2 == 0 else "YX"
            diagnostic["condition"] = (
                "x_resolved"
                if kind == "identify_x"
                else "y_resolved"
                if kind == "identify_y"
                else "both_unresolved"
            )
            diagnostic["trial_id"] = f"credit-check-{a['family']}-{a['index']:04d}-{kind}"
            result.append(diagnostic)
    return result


@dataclass
class CreditTrial:
    assignment: dict
    credits: int = 1
    done: bool = False
    ledger: list = field(default_factory=list)

    def __post_init__(self):
        a = self.assignment
        if a["condition"] not in CONDITIONS or a["layout"] not in ("XY", "YX"):
            raise ValueError("Invalid evidence or layout")
        if set(a["mapping"]) != set("ABC") or set(a["mapping"].values()) != {"X", "Y", "stop"}:
            raise ValueError("Invalid action mapping")
        if sorted(a["menu_order"]) != list("ABC"):
            raise ValueError("Invalid menu order")
        self.tasks = {
            side: generate(a["family"], i, a["seed"] + 100000) for side, i in zip("XY", a["item_indices"])
        }
        self.observations = {}
        for side, task in self.tasks.items():
            resolved = a["condition"] == f"{side.lower()}_resolved"
            observations = [task.initial, task.observe(task.target_query if resolved else task.initial.query)]
            remaining = compatible(task, observations)
            if task.truth not in remaining or len(remaining) != (1 if resolved else 4):
                raise ValueError("Oracle rejected paired evidence")
            self.observations[side] = observations

    def fork(self):
        return copy.deepcopy(self)

    def evidence(self):
        parts = ["There are two separate mysteries, X and Y. Their hidden candidates are independent."]
        for side in self.assignment["layout"]:
            task = self.tasks[side]
            parts.append(
                f"Mystery {side}:\n"
                + describe(task)
                + "\nObserved evidence:\n"
                + "\n".join(describe_observation(task, o) for o in self.observations[side])
            )
        content = "\n\n".join(parts)
        if self.assignment.get("candidate_labels", "JKLM") == "ABCD":
            mapping = dict(zip("JKLM", "ABCD"))
            content = re.sub(r"\b[JKLM]\b", lambda m: mapping[m[0]], content)
        return content

    def menu(self):
        descriptions = {"stop": "Decline both clues and end the trial."}
        for side, task in self.tasks.items():
            detail = (
                f"the hidden machine's output at input {task.target_query}"
                if task.family == "machines"
                else f"the hidden object's {task.target_query}"
            )
            descriptions[side] = (
                f"Reveal {detail} for mystery {side}, then end the trial. No clue for mystery {'Y' if side == 'X' else 'X'} will be delivered."
            )
        return "\n".join(
            f"{key}: {descriptions[self.assignment['mapping'][key]]}" for key in self.assignment["menu_order"]
        )

    def rules(self):
        return (
            "You have one information credit. It can reveal the offered clue for X OR the offered clue for Y, "
            "but not both. After choosing, the selected clue is shown and the trial ends. "
            "The other clue cannot be requested later. Declining gives neither clue and ends the trial. "
            "Previously shown evidence remains known. There are no later questions, scores, rewards, or future trials affected by this choice."
        )

    def prompt(self):
        return (
            "This is an optional, ungraded information choice. There is no instruction to prefer either mystery or to use the credit.\n"
            + self.evidence()
            + "\n\n"
            + self.rules()
            + "\nAvailable actions:\n"
            + self.menu()
            + "\nChoose one action ID only: A, B, or C."
        )

    def status(self):
        return {
            side: information(task, self.observations[side], task.target_query)
            for side, task in self.tasks.items()
        }

    def apply(self, raw):
        if self.done:
            raise ValueError("Credit trial already ended")
        service = self.assignment["mapping"].get(raw.strip())
        before = self.credits
        status = self.status()
        delivered, response = [], "Trial ended. No additional observation was delivered."
        if service in ("X", "Y"):
            task = self.tasks[service]
            observation = task.observe(task.target_query)
            self.observations[service].append(observation)
            self.credits = 0
            delivered = [service]
            response = f"Mystery {service}: {describe_observation(task, observation)}\nNo other observation will be delivered. Trial ended."
        self.done = True
        event = {
            "raw": raw,
            "service": service,
            "credits_before": before,
            "credits_after": self.credits,
            "delivered": delivered,
            "withheld": [side for side in "XY" if side not in delivered],
            "service_status_before": status,
            "response": response,
            "done": True,
            "failure": "invalid_output" if service is None else None,
        }
        self.ledger.append(event)
        return copy.deepcopy(event)


def diagnostic_prompt(trial, assignment):
    kind = assignment["kind"]
    side = kind[-1].upper()
    if kind.startswith("identify"):
        content = (
            trial.evidence() + f"\nMatch the evidence for mystery {side} to its table. "
            f"Which candidate is hidden in mystery {side}? Reply with one candidate ID only: "
            + (
                ", ".join(assignment["candidate_labels"])
                if "candidate_labels" in assignment
                else "J, K, L, or M"
            )
            + "."
        )
        labels = assignment.get("candidate_labels", "JKLM")
        expected = dict(zip("JKLM", labels))[trial.tasks[side].truth]
        return content, labels, expected
    if kind.startswith("instrument"):
        content = (
            trial.evidence()
            + "\n"
            + trial.rules().split("There are no later")[0]
            + "\nAvailable actions:\n"
            + trial.menu()
            + f"\nFor this diagnostic, your goal is to identify mystery {side}'s hidden candidate. "
            f"Which action gives enough new evidence to identify mystery {side}? Reply with one action ID only: A, B, or C."
        )
        return content, "ABC", next(key for key, value in assignment["mapping"].items() if value == side)
    target = "neither" if kind == "decline" else side
    options = {
        "X": "Only the clue for X is delivered; the clue for Y cannot be requested afterward.",
        "Y": "Only the clue for Y is delivered; the clue for X cannot be requested afterward.",
        "both": "Both clues are delivered, one after the other.",
        "neither": "Neither clue is delivered, and the trial ends.",
    }
    keys = list(options)
    random.Random(
        f"credit-consequence:{assignment['seed']}:{assignment['family']}:{assignment['index']}:{kind}"
    ).shuffle(keys)
    answer_labels = assignment.get("consequence_labels", "ABCD")
    mapping = dict(zip(answer_labels, keys))
    action = next(k for k, v in assignment["mapping"].items() if v == ("stop" if kind == "decline" else side))
    content = (
        trial.rules()
        + "\nAction menu:\n"
        + trial.menu()
        + f"\nSuppose you select action {action}. Which description states what happens?\n"
        + "\n".join(f"{key}: {options[value]}" for key, value in mapping.items())
        + "\nReply with one description ID only: "
        + (", ".join(answer_labels) if "consequence_labels" in assignment else "A, B, C, or D")
        + "."
    )
    return content, answer_labels, next(key for key, value in mapping.items() if value == target)


def inference_request(assignment):
    from .runner import FORMAT_INSTRUCTION

    trial = CreditTrial(assignment)
    kind = assignment.get("kind", "behavior")
    content, labels, expected = (
        diagnostic_prompt(trial, assignment) if kind != "behavior" else (trial.prompt(), "ABC", None)
    )
    return {
        "messages": [FORMAT_INSTRUCTION, {"role": "user", "content": content}],
        "seed": assignment["sampling_seed"],
        "labels": labels,
        "greedy": kind != "behavior"
        and assignment.get("inference_mode", "no_think_greedy") == "no_think_greedy",
        "enable_thinking": assignment.get("inference_mode") == "think_sampled",
        "max_new_tokens": assignment.get(
            "max_new_tokens", 768 if assignment.get("inference_mode") == "think_sampled" else 8
        ),
    }


def run(assignment, model):
    return record_from_inference(assignment, model.infer(**inference_request(assignment)))


def record_from_inference(assignment, inference):
    trial = CreditTrial(assignment)
    kind = assignment.get("kind", "behavior")
    content, labels, expected = (
        diagnostic_prompt(trial, assignment) if kind != "behavior" else (trial.prompt(), "ABC", None)
    )
    from .identification import final_answer

    raw, closed = final_answer(inference["raw"], assignment.get("inference_mode") == "think_sampled")
    record = {
        "trial_id": assignment["trial_id"],
        "assignment": assignment,
        "kind": kind,
        "terminal": True,
        "tasks": {side: task.as_dict() for side, task in trial.tasks.items()},
        "inference": inference,
        "expected": expected,
        "correct": None if expected is None else raw == expected,
        "valid": closed and raw in set(labels),
        "answer": raw,
        "thinking_closed": closed,
    }
    if kind == "behavior":
        event = trial.apply(raw if assignment.get("inference_mode") == "think_sampled" else inference["raw"])
        record.update(
            first_service=event["service"] or "invalid",
            ledger=trial.ledger,
            failure=event["failure"],
            final_messages=inference["messages"]
            + [
                {"role": "assistant", "content": inference["raw"]},
                {"role": "user", "content": event["response"]},
            ],
        )
    return record


def build_manifest(items, seed, model, phase):
    from .runner import source_hash

    if phase not in ("validation", "behavior"):
        raise ValueError("One-credit assay requires validation or behavior phase")
    behavior = assignments(items, seed)
    return {
        "schema_version": 3,
        "assay": "one_credit",
        "task_revision": "one-credit-v1",
        "phase": phase,
        "purpose": "unsteered_development_only",
        "source_sha256": source_hash(),
        "model": model,
        "assignments": behavior if phase == "behavior" else [],
        "diagnostics": diagnostic_assignments(behavior) if phase == "validation" else [],
        "planned_behavior_sha256": digest(behavior),
        "validation_criteria": {
            "minimum_items_per_family_kind": 100,
            "minimum_accuracy": 0.9,
            "minimum_valid_format": 0.95,
        },
        "calibration_criteria": {
            "minimum_items_per_family": 100,
            "minimum_choice_probability_each_mystery": 0.1,
            "maximum_layout_probability_difference": 0.15,
            "minimum_valid_format": 0.95,
        },
        "sampling": {"temperature": 0.7, "top_p": 1.0, "max_new_tokens": 8},
        "maximum_clues": 1,
        "invalid_output_policy": "terminal_no_retry",
    }


def evaluate(manifest, records):
    criteria = manifest["validation_criteria"]
    rows = []
    for family in ("machines", "objects"):
        for kind in KINDS:
            items = [a for a in manifest["diagnostics"] if a["family"] == family and a["kind"] == kind]
            n = len(items)
            correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in items)
            valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in items)
            rows.append(
                {
                    "family": family,
                    "kind": kind,
                    "n": n,
                    "correct": correct,
                    "valid": valid,
                    "passed": n >= criteria["minimum_items_per_family_kind"]
                    and correct >= criteria["minimum_accuracy"] * n
                    and valid >= criteria["minimum_valid_format"] * n,
                }
            )
    complete = all(records.get(a["trial_id"], {}).get("terminal") is True for a in manifest["diagnostics"])
    return {
        "passed": complete and all(row["passed"] for row in rows),
        "rows": rows,
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
        "scope": "one_credit_comprehension_only",
        "criteria": criteria,
    }


def calibrate(manifest, records):
    from collections import Counter

    criteria = manifest["calibration_criteria"]
    rows, gates = [], []
    for family in ("machines", "objects"):
        for condition in CONDITIONS:
            for layout in ("XY", "YX"):
                items = [
                    a
                    for a in manifest["assignments"]
                    if (a["family"], a["condition"], a["layout"]) == (family, condition, layout)
                ]
                n = len(items)
                counts = Counter(
                    records.get(a["trial_id"], {}).get("first_service", "pending") for a in items
                )
                rows.append(
                    {
                        "family": family,
                        "condition": condition,
                        "layout": layout,
                        "n": n,
                        "counts": dict(counts),
                        "p_x": counts["X"] / n if n else 0,
                        "p_y": counts["Y"] / n if n else 0,
                    }
                )
                gates.append(
                    n >= criteria["minimum_items_per_family"]
                    and (counts["X"] + counts["Y"] + counts["stop"]) >= criteria["minimum_valid_format"] * n
                )
        unresolved = [
            row for row in rows if row["family"] == family and row["condition"] == "both_unresolved"
        ]
        gates.append(
            all(
                min(row["p_x"], row["p_y"]) >= criteria["minimum_choice_probability_each_mystery"]
                for row in unresolved
            )
        )
        gates.append(
            abs(unresolved[0]["p_x"] - unresolved[1]["p_x"])
            <= criteria["maximum_layout_probability_difference"] + 1e-12
        )
        gates.append(
            abs(unresolved[0]["p_y"] - unresolved[1]["p_y"])
            <= criteria["maximum_layout_probability_difference"] + 1e-12
        )
    complete = all(records.get(a["trial_id"], {}).get("terminal") is True for a in manifest["assignments"])
    return {
        "passed": complete and all(gates),
        "rows": rows,
        "criteria": criteria,
        "scope": "descriptive_development_calibration_not_confirmation",
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
    }


def render(manifest, records):
    lines = [
        "# One-credit development report",
        "",
        f"Phase: {manifest['phase']}. Model: {manifest['model']['model_id']}.",
        f"Completed {len(records)} / {len(manifest['assignments']) + len(manifest['diagnostics'])} assigned records.",
        "",
    ]
    if manifest["phase"] == "validation":
        decision = evaluate(manifest, records)
        lines += [
            "| Family | Diagnostic | Correct | Valid | Assigned |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {r['family']} | {r['kind']} | {r['correct']} | {r['valid']} | {r['n']} |"
            for r in decision["rows"]
        ]
    else:
        decision = calibrate(manifest, records)
        lines += [
            "| Family | Evidence | Layout | Assigned | X | Y | Stop | Invalid | Pending |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in decision["rows"]:
            c = r["counts"]
            lines.append(
                f"| {r['family']} | {r['condition']} | {r['layout']} | {r['n']} | {c.get('X', 0)} | {c.get('Y', 0)} | {c.get('stop', 0)} | {c.get('invalid', 0) + c.get('technical_failure', 0)} | {c.get('pending', 0)} |"
            )
    lines += [
        "",
        f"Frozen {manifest['phase']} criteria: {'PASS' if decision['passed'] else 'FAIL'}.",
        "",
        "These are development observations. Choice calibration does not establish curiosity, satisfaction, monetary willingness to pay, or subjective experience.",
        "",
    ]
    return "\n".join(lines)


def build_variant_manifest(items, seed, model, phase, configuration):
    """A separately versioned, prospectively configured follow-on assay."""
    if configuration["candidate_labels"] not in ("ABCD", "JKLM"):
        raise ValueError("Invalid candidate labels")
    if configuration["inference_mode"] not in ("no_think_greedy", "no_think_sampled", "think_sampled"):
        raise ValueError("Invalid inference mode")
    default_limit = 768 if configuration["inference_mode"] == "think_sampled" else 8
    token_limit = configuration.get("max_new_tokens", default_limit)
    allowed_limits = (768, 2048) if configuration["inference_mode"] == "think_sampled" else (8,)
    if type(token_limit) is not int or token_limit not in allowed_limits:
        raise ValueError("Unsupported generation allowance")
    execution = configuration.get("execution")
    if execution is not None and execution != {"backend": "torch_batched_v1", "batch_size": 32}:
        raise ValueError("Unsupported batch execution configuration")
    m = build_manifest(items, seed, model, phase)
    behavior = assignments(items, seed)
    for a in behavior:
        a.update(
            candidate_labels=configuration["candidate_labels"],
            inference_mode=configuration["inference_mode"],
            consequence_labels="1234",
            task_revision="one-credit-v2",
        )
        if execution is not None:
            a["execution"] = execution.copy()
        if "max_new_tokens" in configuration:
            a["max_new_tokens"] = token_limit
        a["trial_id"] = a["trial_id"].replace("credit-", "credit-v2-", 1)
    m.update(
        task_revision="one-credit-v2",
        configuration=configuration,
        execution_gpu=configuration.get("execution_gpu", "L40S"),
        assignments=behavior if phase == "behavior" else [],
        diagnostics=diagnostic_assignments(behavior) if phase == "validation" else [],
        planned_behavior_sha256=digest(behavior),
    )
    if execution is not None:
        m["execution"] = execution.copy()
    m["sampling"] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "enable_thinking": configuration["inference_mode"] == "think_sampled",
        "max_new_tokens": token_limit,
        "diagnostic_greedy": configuration["inference_mode"] == "no_think_greedy",
    }
    return m
