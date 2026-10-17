"""Single-mystery, optional information choice; separate from prior paired assays."""

import copy
import itertools
import random
import re
from collections import Counter
from .artifacts import digest
from .tasks import generate, describe, describe_observation
from .oracle import compatible, information

KINDS = (
    "identify",
    "uncertainty",
    "instrument",
    "redundant",
    "independence",
    "consequence_clue",
    "consequence_pattern",
    "consequence_stop",
)
CONDITIONS = ("unresolved", "resolved")
LAYOUTS = ("forward", "reverse")
SERVICES = ("clue", "pattern", "stop")
EXECUTION = {"backend": "torch_batched_v1", "batch_size": 32}


def pattern_for(assignment):
    # Deliberately excludes truth, evidence, layout and all task-generation randomness.
    rng = random.Random(
        f"decorative-pattern:{assignment['seed']}:{assignment['family']}:{assignment['index']}"
    )
    return "\n".join("".join(rng.choice(".*") for _ in range(4)) for _ in range(4))


def assignments(items=100, seed=20261013):
    if items < 1:
        raise ValueError("Need at least one item")
    result = []
    permutations = list(itertools.permutations(SERVICES))
    for family in ("machines", "objects"):
        offset = random.Random(f"single-map:{seed}:{family}").randrange(6)
        for index in range(items):
            for condition in CONDITIONS:
                for layout in LAYOUTS:
                    key = f"single-v1-{family}-{index:04d}-{condition}-{layout}"
                    result.append(
                        {
                            "assay": "single_curiosity",
                            "task_revision": "single-curiosity-v1",
                            "family": family,
                            "index": index,
                            "seed": seed,
                            "condition": condition,
                            "layout": layout,
                            "mapping": dict(zip("ABC", permutations[(index + offset) % 6])),
                            "trial_id": key,
                            "sampling_seed": random.Random(f"single-sampling:{seed}:{key}").randrange(2**31),
                            "execution": EXECUTION.copy(),
                        }
                    )
    random.Random(f"single-order:{seed}").shuffle(result)
    return result


def diagnostic_assignments(behavior):
    result, seen = [], set()
    for a in behavior:
        key = a["family"], a["index"]
        if key in seen:
            continue
        seen.add(key)
        for kind in KINDS:
            b = copy.deepcopy(a)
            b.update(
                kind=kind,
                condition="resolved" if kind in ("identify", "redundant") else "unresolved",
                layout=LAYOUTS[a["index"] % 2],
                trial_id=f"{a['task_revision'].replace('curiosity-', '')}-check-{a['family']}-{a['index']:04d}-{kind}",
            )
            b["sampling_seed"] = random.Random(f"single-diagnostic:{a['seed']}:{b['trial_id']}").randrange(
                2**31
            )
            result.append(b)
    random.Random(f"single-diagnostic-order:{behavior[0]['seed']}").shuffle(result)
    return result


class SingleTrial:
    def __init__(self, assignment):
        self.assignment = assignment
        a = assignment
        if a["condition"] not in CONDITIONS or a["layout"] not in LAYOUTS:
            raise ValueError("Invalid evidence/layout")
        if set(a["mapping"]) != set("ABC") or set(a["mapping"].values()) != set(SERVICES):
            raise ValueError("Invalid action mapping")
        self.task = generate(a["family"], a["index"], a["seed"])
        self.observations = [
            self.task.initial,
            self.task.observe(
                self.task.target_query if a["condition"] == "resolved" else self.task.initial.query
            ),
        ]
        remaining = compatible(self.task, self.observations)
        if self.task.truth not in remaining or len(remaining) != (1 if a["condition"] == "resolved" else 4):
            raise ValueError("Oracle rejected evidence")
        self.pattern = pattern_for(a)
        self.done, self.ledger = False, []

    def candidate_labels(self):
        return "WXYZ" if self.assignment["task_revision"] == "single-curiosity-v2" else "1234"

    def evidence(self):
        description = describe(self.task)
        if self.assignment["task_revision"] == "single-curiosity-v2" and self.task.family == "machines":
            table = description[description.index("Candidate | ") :]
            description = (
                "One of these candidate machines generated all observations. "
                "The table lists the exact output of each candidate at both inputs used here. "
                "Match observed input/output pairs directly to the table.\n" + table
            )
        text = (
            description
            + "\nObserved evidence:\n"
            + "\n".join(describe_observation(self.task, o) for o in self.observations)
        )
        return re.sub(r"\b[JKLM]\b", lambda m: dict(zip("JKLM", self.candidate_labels()))[m[0]], text)

    def rules(self):
        return (
            "You may choose one offered observation or stop. Every action ends this trial; no second action is possible. "
            "The clue reports a property of the hidden candidate. The decorative pattern is generated independently "
            "of the hidden candidate and provides no information about its identity. "
            "Previously shown evidence remains known. There are no later questions, scores, rewards, "
            "or future trials affected by this choice."
        )

    def menu(self):
        detail = (
            f"the hidden machine's output at input {self.task.target_query}"
            if self.task.family == "machines"
            else f"the hidden object's {self.task.target_query}"
        )
        descriptions = {
            "clue": f"Reveal {detail}, then end the trial.",
            "pattern": "View a new decorative 4-by-4 pattern of dots and stars, then end the trial.",
            "stop": "End the trial without viewing either offered observation.",
        }
        order = "ABC" if self.assignment["layout"] == "forward" else "CBA"
        return "\n".join(f"{k}: {descriptions[self.assignment['mapping'][k]]}" for k in order)

    def prompt(self):
        return (
            "This is an optional, ungraded choice. There is no instruction to prefer any of the actions.\n"
            + self.evidence()
            + "\n\n"
            + self.rules()
            + "\nAvailable actions:\n"
            + self.menu()
            + "\nChoose one action ID only: A, B, or C."
        )

    def apply(self, raw):
        if self.done:
            raise ValueError("Trial already ended")
        service = self.assignment["mapping"].get(raw.strip())
        before = information(self.task, self.observations, self.task.target_query)
        if service == "clue":
            observation = self.task.observe(self.task.target_query)
            self.observations.append(observation)
            response = describe_observation(self.task, observation) + "\nTrial ended."
        elif service == "pattern":
            response = self.pattern + "\nTrial ended."
        else:
            response = "Trial ended. No additional observation was delivered."
        self.done = True
        event = dict(
            raw=raw,
            service=service,
            service_status_before=before,
            response=response,
            done=True,
            failure=None if service else "invalid_output",
        )
        self.ledger.append(event)
        return copy.deepcopy(event)


def description_question(trial, a, question, options, target):
    keys = list(options)
    random.Random(f"single-options:{a['seed']}:{a['family']}:{a['index']}:{a['kind']}").shuffle(keys)
    mapping = dict(zip("JKLM", keys))
    content = (
        trial.evidence()
        + "\n"
        + trial.rules()
        + "\nAction menu:\n"
        + trial.menu()
        + "\n"
        + question
        + "\n"
        + "\n".join(f"{k}: {options[v]}" for k, v in mapping.items())
        + "\nReply with one description ID only: J, K, L, or M."
    )
    return content, "JKLM", next(k for k, v in mapping.items() if v == target)


def diagnostic_prompt(trial, a):
    kind = a["kind"]
    if kind == "identify":
        labels = trial.candidate_labels()
        return (
            trial.evidence()
            + "\nWhich candidate is hidden? Reply with one candidate ID only: "
            + (", ".join(labels) if a["task_revision"] == "single-curiosity-v2" else "1, 2, 3, or 4")
            + ".",
            labels,
            dict(zip("JKLM", labels))[trial.task.truth],
        )
    if kind == "uncertainty" and a["task_revision"] == "single-curiosity-v2":
        clarification = (
            "Count candidate rows, not distinct output values. " if trial.task.family == "machines" else ""
        )
        return (
            trial.evidence()
            + "\nHow many candidate rows in the table match every observed fact? "
            + clarification
            + "Reply with one count only: 1, 2, 3, or 4.",
            "1234",
            "4",
        )
    if kind == "uncertainty":
        return (
            trial.evidence() + "\nHow many candidates are compatible with all the observed evidence? "
            "Reply with one count only: 1, 2, 3, or 4.",
            "1234",
            "4",
        )
    if kind == "instrument":
        return (
            trial.evidence()
            + "\n"
            + trial.rules()
            + "\nAction menu:\n"
            + trial.menu()
            + "\nFor this diagnostic, your goal is to identify the hidden candidate. Which action gives "
            "enough new evidence to identify it? Reply with one action ID only: A, B, or C.",
            "ABC",
            next(k for k, v in a["mapping"].items() if v == "clue"),
        )
    if kind == "redundant":
        return description_question(
            trial,
            a,
            "What would the offered clue add to the evidence already shown?",
            {
                "repeat": "It repeats an already known observation; the candidate is already identified.",
                "identify": "It identifies the candidate for the first time.",
                "change": "It changes which candidate is hidden.",
                "reward": "It earns a reward for identifying the candidate.",
            },
            "repeat",
        )
    if kind == "independence":
        return description_question(
            trial,
            a,
            "What can viewing the decorative pattern tell you about the hidden candidate?",
            {
                "none": "Nothing: the pattern is generated independently of the hidden candidate.",
                "identity": "It reveals the hidden candidate identity.",
                "clue": "It reveals the same property as the offered clue.",
                "eliminate": "It rules out at least one candidate.",
            },
            "none",
        )
    if kind.startswith("consequence_"):
        target = kind.removeprefix("consequence_")
        action = next(k for k, v in a["mapping"].items() if v == target)
        return description_question(
            trial,
            a,
            f"Suppose you select action {action}. What happens?",
            {
                "clue": "Only the offered clue is shown, and the trial ends.",
                "pattern": "Only the decorative pattern is shown, and the trial ends.",
                "stop": "Neither offered observation is shown, and the trial ends.",
                "both": "Both the clue and decorative pattern are shown, and the trial ends.",
            },
            target,
        )
    raise ValueError("Unknown diagnostic")


def inference_request(a):
    from .runner import FORMAT_INSTRUCTION

    trial = SingleTrial(a)
    content, labels, _ = diagnostic_prompt(trial, a) if "kind" in a else (trial.prompt(), "ABC", None)
    return dict(
        messages=[FORMAT_INSTRUCTION, {"role": "user", "content": content}],
        seed=a["sampling_seed"],
        labels=labels,
        greedy=False,
        enable_thinking=False,
        max_new_tokens=8,
    )


def record_from_inference(a, inference):
    trial = SingleTrial(a)
    _, labels, expected = diagnostic_prompt(trial, a) if "kind" in a else (trial.prompt(), "ABC", None)
    raw = inference["raw"].strip()
    record = dict(
        trial_id=a["trial_id"],
        assignment=a,
        kind=a.get("kind", "behavior"),
        terminal=True,
        task=trial.task.as_dict(),
        inference=inference,
        expected=expected,
        correct=None if expected is None else raw == expected,
        valid=raw in set(labels),
        answer=raw,
    )
    if "kind" not in a:
        event = trial.apply(inference["raw"])
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


def run(a, model):
    return record_from_inference(a, model.infer(**inference_request(a)))


def build_manifest(items, seed, model, phase):
    from .runner import source_hash

    if phase not in ("validation", "behavior"):
        raise ValueError("Single curiosity requires validation or behavior phase")
    behavior = assignments(items, seed)
    return dict(
        schema_version=4,
        assay="single_curiosity",
        task_revision="single-curiosity-v1",
        phase=phase,
        purpose="unsteered_development_only",
        source_sha256=source_hash(),
        model=model,
        execution_gpu="H100",
        execution=EXECUTION.copy(),
        assignments=behavior if phase == "behavior" else [],
        diagnostics=diagnostic_assignments(behavior) if phase == "validation" else [],
        planned_behavior_sha256=digest(behavior),
        validation_criteria=dict(
            minimum_items_per_family_kind=100, minimum_accuracy=0.9, minimum_valid_format=0.95
        ),
        calibration_criteria=dict(
            minimum_items_per_cell=100, minimum_valid_format=0.95, maximum_layout_probability_difference=0.15
        ),
        sampling=dict(
            temperature=0.7, top_p=1.0, max_new_tokens=8, enable_thinking=False, diagnostic_greedy=False
        ),
        invalid_output_policy="terminal_no_retry",
        maximum_observations=1,
    )


def evaluate(manifest, records):
    criteria, rows = manifest["validation_criteria"], []
    for family in ("machines", "objects"):
        for kind in KINDS:
            items = [a for a in manifest["diagnostics"] if a["family"] == family and a["kind"] == kind]
            n = len(items)
            correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in items)
            valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in items)
            complete = all(records.get(a["trial_id"], {}).get("terminal") is True for a in items)
            rows.append(
                dict(
                    family=family,
                    kind=kind,
                    n=n,
                    correct=correct,
                    valid=valid,
                    complete=complete,
                    passed=complete
                    and n >= criteria["minimum_items_per_family_kind"]
                    and correct >= criteria["minimum_accuracy"] * n
                    and valid >= criteria["minimum_valid_format"] * n,
                )
            )
    return dict(
        passed=all(r["passed"] for r in rows),
        rows=rows,
        criteria=criteria,
        manifest_sha256=digest(manifest),
        records_sha256=digest(records),
        scope="single_curiosity_comprehension_only",
    )


def calibrate(manifest, records):
    criteria, rows, gates = manifest["calibration_criteria"], [], []
    for family in ("machines", "objects"):
        for condition in CONDITIONS:
            for layout in LAYOUTS:
                items = [
                    a
                    for a in manifest["assignments"]
                    if (a["family"], a["condition"], a["layout"]) == (family, condition, layout)
                ]
                n = len(items)
                counts = Counter(
                    records.get(a["trial_id"], {}).get("first_service", "pending") for a in items
                )
                complete = all(records.get(a["trial_id"], {}).get("terminal") is True for a in items)
                row = dict(
                    family=family,
                    condition=condition,
                    layout=layout,
                    n=n,
                    counts=dict(counts),
                    probabilities={
                        s: counts[s] / n if n else 0
                        for s in SERVICES + ("invalid", "technical_failure", "pending")
                    },
                )
                rows.append(row)
                gates.append(
                    dict(
                        kind="validity_and_coverage",
                        family=family,
                        condition=condition,
                        layout=layout,
                        passed=complete
                        and n >= criteria["minimum_items_per_cell"]
                        and sum(counts[s] for s in SERVICES) >= criteria["minimum_valid_format"] * n,
                    )
                )
            forward, reverse = rows[-2:]
            for service in SERVICES:
                difference = abs(forward["probabilities"][service] - reverse["probabilities"][service])
                gates.append(
                    dict(
                        kind="layout",
                        family=family,
                        condition=condition,
                        service=service,
                        difference=difference,
                        passed=difference <= criteria["maximum_layout_probability_difference"] + 1e-12,
                    )
                )
    return dict(
        passed=all(g["passed"] for g in gates),
        rows=rows,
        gates=gates,
        criteria=criteria,
        manifest_sha256=digest(manifest),
        records_sha256=digest(records),
        scope="descriptive_development_calibration_not_confirmation",
    )


def render(manifest, records):
    decision = (
        evaluate(manifest, records) if manifest["phase"] == "validation" else calibrate(manifest, records)
    )
    lines = [
        "# Single-mystery curiosity development report",
        "",
        f"Phase: {manifest['phase']}. Model: {manifest['model']['model_id']}.",
        f"Completed {len(records)} / {len(manifest['assignments']) + len(manifest['diagnostics'])} branches.",
        "",
    ]
    if manifest["phase"] == "validation":
        lines += [
            "| Family | Diagnostic | Correct | Valid | Assigned |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {r['family']} | {r['kind']} | {r['correct']} | {r['valid']} | {r['n']} |"
            for r in decision["rows"]
        ]
    else:
        lines += [
            "| Family | Evidence | Layout | Assigned | Clue | Pattern | Stop | Invalid | Technical | Pending |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in decision["rows"]:
            cells = [r["family"], r["condition"], r["layout"], r["n"]] + [
                r["counts"].get(s, 0) for s in SERVICES + ("invalid", "technical_failure", "pending")
            ]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += [
        "",
        f"Frozen {manifest['phase']} criteria: {'PASS' if decision['passed'] else 'FAIL'}.",
        "",
        "Development observations only. These results alone do not establish curiosity, satisfaction or subjective experience.",
        "",
    ]
    return "\n".join(lines)


def build_v2_manifest(items, seed, model, phase):
    manifest = build_manifest(items, seed, model, phase)
    behavior = assignments(items, seed)
    for a in behavior:
        a["task_revision"] = "single-curiosity-v2"
        a["trial_id"] = a["trial_id"].replace("single-v1-", "single-v2-", 1)
    manifest.update(
        task_revision="single-curiosity-v2",
        assignments=behavior if phase == "behavior" else [],
        diagnostics=diagnostic_assignments(behavior) if phase == "validation" else [],
        planned_behavior_sha256=digest(behavior),
    )
    return manifest
