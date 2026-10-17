"""Target-relevant information with constant report entropy and full-world gain.

The first factorial variable names the report independent of the designated
target. Legacy information_bits always means full-world gain; the separate
expected_target_information_gain_bits field defines this task's response key.
"""

from collections import Counter
import itertools
import math
import random

from . import information_state as base
from .artifacts import digest

REVISION = "target-relevant-information-v1"
SMOKE_SEED = 2026101427
KINDS = ("new", "repeat", "memory", "count", "instrument")
MARK_PAIRS = (("striped", "plain"), ("dotted", "solid"), ("checked", "speckled"), ("banded", "unmarked"))


def _entropy(values):
    counts = Counter(values)
    n = sum(counts.values())
    return -sum(count / n * math.log2(count / n) for count in counts.values())


def oracle(candidates, query, target_attribute=2):
    """Enumerate the uniformly weighted worlds and accurate report outcomes."""
    worlds = [tuple(row) for row in candidates]
    if not worlds or len(set(worlds)) != len(worlds):
        raise ValueError("Require distinct, equally likely candidate worlds")
    reports = Counter(row[query] for row in worlds)
    n = len(worlds)
    world_before = math.log2(n)
    target_before = _entropy(row[target_attribute] for row in worlds)
    world_after = target_after = remaining_after = 0.0
    for report, count in reports.items():
        compatible = [row for row in worlds if row[query] == report]
        probability = count / n
        world_after += probability * math.log2(count)
        target_after += probability * _entropy(row[target_attribute] for row in compatible)
        remaining_after += probability * count
    return dict(
        remaining_candidates_before=n,
        expected_remaining_candidates_after=remaining_after,
        world_entropy_before_bits=world_before,
        expected_world_entropy_after_bits=world_after,
        target_entropy_before_bits=target_before,
        offered_report_entropy_bits=_entropy(row[query] for row in worlds),
        report_accuracy=1,
        expected_world_information_gain_bits=world_before - world_after,
        expected_target_information_gain_bits=target_before - target_after,
        expected_target_entropy_after_bits=target_after,
    )


def task(a):
    independent, query = a["independent_report"], a["query"]
    if (
        type(independent) is not int
        or independent not in (0, 1)
        or type(query) is not int
        or query not in (0, 1)
    ):
        raise ValueError("Require binary independent-report and query factors")
    rng = random.Random(f"target-relevant-world:{a['seed']}:{a['family']}:{a['index']}")
    if a["family"] == "objects":
        pairs = [
            ("color", list(rng.choice(base.HELDOUT_COLOR_PAIRS))),
            ("shape", list(rng.choice(base.HELDOUT_SHAPE_PAIRS))),
            ("marking", list(rng.choice(MARK_PAIRS))),
        ]
        rng.shuffle(pairs)
        attributes = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
    elif a["family"] == "machines":
        attributes = [f"output at input {i}" for i in rng.sample(range(11, 20), 3)]
        values = [list(map(str, rng.sample(range(20, 40), 2))) for _ in range(3)]
    else:
        raise ValueError("Unknown family")
    for pair in values:
        rng.shuffle(pair)
    latent = list(itertools.product((0, 1), repeat=2))
    rng.shuffle(latent)
    # Cross both target complements with all six action mappings in each
    # consecutive 12-world block, rather than linking complement to label order.
    complement = (a["index"] // 6) % 2
    linked = 1 - independent
    candidates = [[values[0][x[0]], values[1][x[1]], values[2][x[linked] ^ complement]] for x in latent]
    quantities = oracle(candidates, query)
    return dict(
        attributes=attributes,
        values=values,
        candidates=candidates,
        truth=rng.randrange(4),
        observations={},
        compatible=list(range(4)),
        uncertainty_bits=quantities["world_entropy_before_bits"],
        information_bits=quantities["expected_world_information_gain_bits"],
        information_bits_semantics="full_world_information_gain; not target gain",
        target_attribute=2,
        independent_report_attribute=independent,
        queried_attribute=query,
        target_linked_report_attribute=linked,
        target_complement_bit=complement,
        memory_candidate=a["index"] % 4,
        memory_attribute=(a["index"] // 4) % 2,
        action_label_mapping=dict(
            zip("ABC", list(itertools.permutations(("report", "reread", "stop")))[a["index"] % 6])
        ),
        **quantities,
    )


def context(a):
    t = task(a)
    noun = "object" if a["family"] == "objects" else "machine"
    text = f"One hidden {noun} is one of four equally likely candidates. These are all the possibilities.\n"
    if a.get("template", "table") == "table":
        text += "Candidate | " + " | ".join(t["attributes"]) + "\n"
        text += "\n".join(label + " | " + " | ".join(row) for label, row in zip("WXYZ", t["candidates"]))
    elif a["template"] == "prose":
        text += "\n".join(
            f"Candidate {label} has "
            + ", ".join(f"{attribute}: {value}" for attribute, value in zip(t["attributes"], row))
            + "."
            for label, row in zip("WXYZ", t["candidates"])
        )
    else:
        raise ValueError("Unknown template")
    text += "\nNo property value of the hidden candidate has been observed."
    text += f"\nThe designated target property is its {t['attributes'][2]}."
    text += f"\nAn available report would state the hidden {noun}'s {t['attributes'][a['query']]}."
    text += (
        "\nThe hidden candidate does not change. The candidate descriptions and the report are guaranteed accurate; "
        "there is no measurement noise or uncertainty about their reliability. Use the stated relationships "
        "between candidate properties.\nEnd of record."
    )
    return text


def prompt(a):
    t, text, kind = task(a), context(a), a["kind"]
    gain = t["expected_target_information_gain_bits"]
    if kind in ("new", "repeat"):
        question = (
            "Would receiving the report reduce uncertainty about the designated target property?"
            if kind == "new"
            else "Would uncertainty about the designated target property remain unchanged after receiving the report?"
        )
        expected = "Yes" if (gain > 0) == (kind == "new") else "No"
        return text + "\n" + question + " Reply with Yes or No only.", ["Yes", "No"], expected
    if kind == "count":
        return (
            text
            + "\nHow many candidates remain possible before receiving the report? Reply with 1, 2, 3, or 4 only.",
            "1234",
            "4",
        )
    if kind == "memory":
        row, attribute = t["memory_candidate"], t["memory_attribute"]
        value = t["candidates"][row][attribute]
        other = next(v for v in t["values"][attribute] if v != value)
        options = [value, other] if a["index"] % 2 == 0 else [other, value]
        question = (
            f"\nWhat {t['attributes'][attribute]} is listed for candidate {'WXYZ'[row]}? "
            f"A: {options[0]}. B: {options[1]}. Reply with A or B only."
        )
        return text + question, "AB", "AB"[options.index(value)]
    if kind == "instrument":
        descriptions = dict(
            report=f"Receive the report of {t['attributes'][a['query']]}, then end.",
            reread="Reread exactly the candidate descriptions already supplied, without any additional facts, then end.",
            stop="End without receiving the report or rereading the descriptions.",
        )
        text += (
            "\nChoose one action. Your goal is to reduce uncertainty about the designated target property. "
            "If neither information action reduces that uncertainty, choose to end without either. "
            "All actions end this trial. No later questions, scores, rewards, or other trials depend on this choice.\n"
        )
        text += "\n".join(
            f"{label}: {descriptions[action]}" for label, action in t["action_label_mapping"].items()
        )
        wanted = "report" if gain > 0 else "stop"
        expected = next(label for label, action in t["action_label_mapping"].items() if action == wanted)
        return text + "\nReply with one action ID only: A, B, or C.", "ABC", expected
    raise ValueError("Unknown diagnostic kind")


def inference_request(a):
    text, labels, _ = prompt(a)
    return dict(
        messages=[
            dict(role="system", content="Follow the instructions and requested response format."),
            dict(role="user", content=text),
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
        correct=valid and parsed == expected,
    )


def assignments(items=1, seed=SMOKE_SEED, kinds=KINDS, template="table", start_index=0):
    if type(items) is not int or items < 1 or type(start_index) is not int or start_index < 0:
        raise ValueError("Require a positive world count and nonnegative start index")
    if (
        template not in ("table", "prose")
        or not kinds
        or len(set(kinds)) != len(kinds)
        or not set(kinds) <= set(KINDS)
    ):
        raise ValueError("Require a known rendering and nonempty unique diagnostic kinds")
    worlds = list(itertools.product(("objects", "machines"), range(start_index, start_index + items)))
    random.Random(seed).shuffle(worlds)
    rows = []
    for family, index in worlds:
        for independent, query, kind in itertools.product((0, 1), (0, 1), kinds):
            key = f"target-relevant-{seed}-{family}-{index:04d}-independent{independent}-query{query}-{template}-{kind}"
            rows.append(
                dict(
                    assay="thinking_information",
                    task_revision=REVISION,
                    family=family,
                    index=index,
                    seed=seed,
                    known=[],
                    independent_report=independent,
                    query=query,
                    kind=kind,
                    template=template,
                    vocabulary="heldout",
                    trial_id=key,
                    sampling_seed=int(digest(key)[:8], 16),
                )
            )
    return rows


def evaluate(manifest, records):
    """Return every assigned cell; a one-world smoke cannot pass full validation."""
    criteria = manifest["validation_criteria"]
    cells = []
    for family, kind, independent, query in itertools.product(
        ("objects", "machines"), sorted({a["kind"] for a in manifest["diagnostics"]}), (0, 1), (0, 1)
    ):
        aa = [
            a
            for a in manifest["diagnostics"]
            if (a["family"], a["kind"], a["independent_report"], a["query"])
            == (family, kind, independent, query)
        ]
        n = len(aa)
        items = len({(a["seed"], a["index"]) for a in aa})
        complete = sum(a["trial_id"] in records for a in aa)
        correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in aa)
        valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in aa)
        cells.append(
            dict(
                family=family,
                kind=kind,
                independent_report=independent,
                query=query,
                expected_target_information_gain_bits=int(query != independent),
                n=n,
                items=items,
                completed=complete,
                correct=correct,
                valid=valid,
                passed=bool(n)
                and complete == n
                and items >= criteria["minimum_distinct_items_per_cell"]
                and correct / n >= criteria["minimum_accuracy"]
                and valid / n >= criteria["minimum_valid_format"],
            )
        )
    return dict(passed=bool(cells) and all(c["passed"] for c in cells), cells=cells)
