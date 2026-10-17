"""Matched information gain at fixed total uncertainty; explicit finite-world oracle."""

import itertools
import math
import random
from collections import defaultdict
from .artifacts import digest
from .runner import source_hash

EXECUTION = {"backend": "torch_batched_v1", "batch_size": 16}
KINDS = ("new", "repeat", "memory", "count", "instrument")
COLOR_PAIRS = (("red", "blue"), ("green", "yellow"), ("black", "white"), ("orange", "purple"))
SHAPE_PAIRS = (("round", "square"), ("oval", "triangular"), ("hexagonal", "rectangular"), ("flat", "curved"))
HELDOUT_COLOR_PAIRS = (("pink", "tan"), ("silver", "gold"), ("navy", "teal"), ("cyan", "magenta"))
HELDOUT_SHAPE_PAIRS = (
    ("star", "diamond"),
    ("heart", "crescent"),
    ("pentagonal", "octagonal"),
    ("spiral", "cross"),
)


def task(a):
    rng = random.Random(f"state-world:{a['seed']}:{a['family']}:{a['index']}")
    heldout = a.get("vocabulary", "development") == "heldout"
    if a["family"] == "objects":
        values = [
            list(rng.choice(HELDOUT_COLOR_PAIRS if heldout else COLOR_PAIRS)),
            list(rng.choice(HELDOUT_SHAPE_PAIRS if heldout else SHAPE_PAIRS)),
        ]
        attributes = ["color", "shape"]
    elif a["family"] == "machines":
        values = [list(map(str, rng.sample(range(20, 40) if heldout else range(10), 2))) for _ in range(2)]
        inputs = rng.sample(range(11, 20) if heldout else range(1, 10), 2)
        attributes = [f"output at input {x}" for x in inputs]
    else:
        raise ValueError("Unknown family")
    for v in values:
        rng.shuffle(v)
    candidates = list(itertools.product(*values))
    rng.shuffle(candidates)
    truth = rng.randrange(4)
    known = a["known"]
    observations = {i: candidates[truth][i] for i in known}
    compatible = [i for i, c in enumerate(candidates) if all(c[k] == v for k, v in observations.items())]
    query = a["query"]
    counts = defaultdict(int)
    for i in compatible:
        counts[candidates[i][query]] += 1
    expected_remaining_entropy = sum((n / len(compatible)) * math.log2(n) for n in counts.values())
    prior_entropy = math.log2(len(compatible))
    return dict(
        attributes=attributes,
        values=values,
        candidates=candidates,
        truth=truth,
        observations=observations,
        compatible=compatible,
        uncertainty_bits=prior_entropy,
        information_bits=prior_entropy - expected_remaining_entropy,
    )


def context(a):
    t = task(a)
    noun = "object" if a["family"] == "objects" else "machine"
    if a.get("template", "table") == "table":
        s = f"One hidden {noun} is one of the four rows below. Every row was equally possible before the observations.\n"
        s += "Candidate | " + " | ".join(t["attributes"]) + "\n"
        s += "\n".join(f"{label} | " + " | ".join(row) for label, row in zip("WXYZ", t["candidates"]))
        s += "\nObserved facts:\n" + (
            "\n".join(
                f"The {attr} is {t['observations'][i]}."
                for i, attr in enumerate(t["attributes"])
                if i in t["observations"]
            )
            or "No facts have been observed."
        )
        s += f"\nAn available report would state the hidden {noun}'s {t['attributes'][a['query']]}."
    elif a["template"] == "prose":
        s = f"A hidden {noun} has two independently selected properties. "
        s += " ".join(
            f"Its {attr} is either {v[0]} or {v[1]}, each equally likely."
            for attr, v in zip(t["attributes"], t["values"])
        )
        s += (
            "\nYou have been told: "
            + (
                "; ".join(f"{t['attributes'][i]} = {v}" for i, v in sorted(t["observations"].items()))
                or "nothing about either property"
            )
            + "."
        )
        s += f"\nYou can receive a report of its {t['attributes'][a['query']]}."
    else:
        raise ValueError("Unknown template")
    return (
        s
        + "\nThe hidden candidate does not change. All observed facts and the available report are guaranteed "
        "accurate; there is no measurement noise or uncertainty about their reliability.\nEnd of record."
    )


def prompt(a):
    t = task(a)
    content = context(a)
    kind = a["kind"]
    if kind in ("new", "repeat"):
        question = (
            "Would receiving the report give you a fact that is not already among the observed facts?"
            if kind == "new"
            else "Is the fact the report would give you already among the observed facts?"
        )
        answer = (t["information_bits"] > 0) == (kind == "new")
        return (
            content + "\n" + question + " Reply with Yes or No only.",
            ["Yes", "No"],
            "Yes" if answer else "No",
        )
    if kind == "count":
        return (
            content
            + "\nHow many candidates remain possible after the observed facts? Reply with 1, 2, 3, or 4 only.",
            "1234",
            str(len(t["compatible"])),
        )
    if kind == "memory":
        k = a["known"][0]
        vals = list(t["values"][k])
        if a["index"] % 2:
            vals.reverse()
        question = f"\nWhat is the observed {t['attributes'][k]}? A: {vals[0]}. B: {vals[1]}. Reply with A or B only."
        return content + question, "AB", "AB"[vals.index(t["observations"][k])]
    if kind == "passive":
        return content + "\nRead the record.", "AB", None
    if kind in ("choice", "instrument"):
        services = list(itertools.permutations(("report", "pattern", "stop")))[a["index"] % 6]
        descriptions = dict(
            report=f"Receive the report of {t['attributes'][a['query']]}, then end.",
            pattern="See an unrelated decorative pattern, then end.",
            stop="End without receiving either.",
        )
        menu = "\n".join(f"{label}: {descriptions[service]}" for label, service in zip("ABC", services))
        rules = (
            "\nChoose one action. All actions end this trial. The pattern is independent of the hidden candidate. "
            "No later questions, scores, rewards, or other trials depend on this choice. "
        )
        question = (
            "This choice is optional; no action is prescribed.\n"
            if kind == "choice"
            else "Your goal for this diagnostic is to learn a fact you do not already know.\n"
        )
        expected = None
        if kind == "instrument":
            expected = "ABC"[services.index("report" if t["information_bits"] else "stop")]
            question += "If neither observation supplies a new fact about the candidate, select the action that ends without an observation.\n"
        return (
            content + rules + question + menu + "\nReply with one action ID only: A, B, or C.",
            "ABC",
            expected,
        )
    raise ValueError("Unknown kind")


def assignments(items=100, seed=2026101402, kinds=KINDS, template="table", start_index=0):
    rows = []
    for family, index, known, query, kind in itertools.product(
        ("objects", "machines"), range(start_index, start_index + items), range(2), range(2), kinds
    ):
        key = f"state-{seed}-{family}-{index:04d}-known{known}-query{query}-{template}-{kind}"
        a = dict(
            assay="information_state",
            task_revision="equal-uncertainty-v1",
            family=family,
            index=index,
            seed=seed,
            known=[known],
            query=query,
            kind=kind,
            template=template,
            trial_id=key,
            execution=EXECUTION.copy(),
            sampling_seed=random.Random(f"state-sample:{key}").randrange(2**31),
        )
        rows.append(a)
    random.Random(f"state-order:{seed}").shuffle(rows)
    return rows


def inference_request(a):
    content, labels, _ = prompt(a)
    return dict(
        messages=[
            {"role": "system", "content": "Answer the question and follow the requested response format."},
            {"role": "user", "content": content},
        ],
        seed=a["sampling_seed"],
        labels=labels,
        greedy=False,
        max_new_tokens=8,
        enable_thinking=False,
    )


def record_from_inference(a, inference):
    _, labels, expected = prompt(a)
    raw = inference["raw"].strip()
    # Prospective grammar for this new instrument: an optional terminal period
    # is semantically unambiguous; extra explanation or multiple IDs is invalid.
    parsed = raw[:-1] if raw.endswith(".") else raw
    valid = parsed in set(labels)
    return dict(
        trial_id=a["trial_id"],
        assignment=a,
        kind=a["kind"],
        terminal=True,
        task=task(a),
        inference=inference,
        answer=raw,
        parsed_answer=parsed,
        expected=expected,
        valid=valid,
        correct=valid and parsed == expected if expected is not None else None,
    )


def build_manifest(model, items=100, seed=2026101402, kinds=KINDS, template="table"):
    rows = assignments(items, seed, kinds, template)
    return dict(
        schema_version=3,
        assay="information_state",
        phase="validation",
        task_revision="equal-uncertainty-v1",
        model=model,
        execution_gpu="H100",
        execution=EXECUTION,
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        invalid_output_policy="terminal_no_retry",
        response_grammar="One allowed label, optional single terminal period, surrounding whitespace ignored; no other text.",
        validation_criteria=dict(
            minimum_distinct_items_per_cell=100,
            minimum_accuracy=0.9,
            minimum_valid_format=0.95,
            cell_axes=["family", "kind", "known", "query"],
        ),
    )


def evaluate(manifest, records):
    cells = []
    for family, kind, known, query in itertools.product(
        ("objects", "machines"), sorted({a["kind"] for a in manifest["diagnostics"]}), range(2), range(2)
    ):
        rows = [
            a
            for a in manifest["diagnostics"]
            if a["family"] == family and a["kind"] == kind and a["known"] == [known] and a["query"] == query
        ]
        n = len(rows)
        correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in rows)
        valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in rows)
        complete = sum(a["trial_id"] in records for a in rows)
        items = len({a["index"] for a in rows})
        cells.append(
            dict(
                family=family,
                kind=kind,
                known=known,
                query=query,
                n=n,
                items=items,
                correct=correct,
                valid=valid,
                completed=complete,
                passed=complete == n and items >= 100 and correct / n >= 0.9 and valid / n >= 0.95,
            )
        )
    return dict(passed=bool(cells) and all(c["passed"] for c in cells), cells=cells)


def render(manifest, records):
    result = evaluate(manifest, records)
    lines = [
        "# Equal-uncertainty information-state validation",
        "",
        f"Model: {manifest['model']['model_id']}",
        f"All gates passed: {result['passed']}",
        "",
        "| Family | Kind | Known | Query | Correct | Valid | Complete | Assigned | Items | Passed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in result["cells"]:
        lines.append(
            "| "
            + " | ".join(
                str(c[k])
                for k in (
                    "family",
                    "kind",
                    "known",
                    "query",
                    "correct",
                    "valid",
                    "completed",
                    "n",
                    "items",
                    "passed",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
