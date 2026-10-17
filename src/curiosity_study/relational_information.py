"""Information gain from table-implied relations, with distinct known/query properties."""

import itertools
import math
import random
from .artifacts import digest
from . import information_state as base

REVISION = "relational-information-v1"
MARK_PAIRS = (("striped", "plain"), ("dotted", "solid"), ("checked", "speckled"), ("banded", "unmarked"))


def task(a):
    rng = random.Random(f"relational-world:{a['seed']}:{a['family']}:{a['index']}")
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
        inputs = rng.sample(range(11, 20), 3)
        attributes = [f"output at input {i}" for i in inputs]
        values = [list(map(str, rng.sample(range(20, 40), 2))) for _ in range(3)]
    else:
        raise ValueError("Unknown family")
    for pair in values:
        rng.shuffle(pair)
    latent = list(itertools.product(range(2), repeat=2))
    rng.shuffle(latent)
    truth = rng.randrange(4)
    flips = [rng.randrange(2) for _ in range(3)]
    corr = a["correlated"]
    candidates = [
        [values[0][u ^ flips[0]], values[1][v ^ flips[1]], values[2][(u if corr == 0 else v) ^ flips[2]]]
        for u, v in latent
    ]
    observations = {2: candidates[truth][2]}
    compatible = [i for i, c in enumerate(candidates) if c[2] == observations[2]]
    counts = {v: sum(candidates[i][a["query"]] == v for i in compatible) for v in values[a["query"]]}
    entropy_after = sum(n / len(compatible) * math.log2(n) for n in counts.values() if n)
    return dict(
        attributes=attributes,
        values=values,
        candidates=candidates,
        truth=truth,
        observations=observations,
        compatible=compatible,
        uncertainty_bits=math.log2(len(compatible)),
        information_bits=math.log2(len(compatible)) - entropy_after,
        correlated=corr,
        latent_polarities=flips,
    )


def context(a):
    t = task(a)
    noun = "object" if a["family"] == "objects" else "machine"
    text = f"One hidden {noun} is one of four candidates. Each candidate was equally likely before the observation.\n"
    if a.get("template", "table") == "table":
        text += "Candidate | " + " | ".join(t["attributes"]) + "\n"
        text += "\n".join(label + " | " + " | ".join(row) for label, row in zip("WXYZ", t["candidates"]))
    elif a["template"] == "prose":
        text += "\n".join(
            f"Candidate {label} has "
            + ", ".join(f"{attr}: {value}" for attr, value in zip(t["attributes"], row))
            + "."
            for label, row in zip("WXYZ", t["candidates"])
        )
    else:
        raise ValueError("Unknown template")
    text += f"\nObserved fact: its {t['attributes'][2]} is {t['observations'][2]}."
    text += f"\nAn available report would give its {t['attributes'][a['query']]}."
    text += "\nAll facts and reports are accurate, and the hidden candidate does not change. The four possibilities above are exhaustive; use their stated relationships when deciding what the observation already determines.\nEnd of record."
    return text


def prompt(a):
    t = task(a)
    text = context(a)
    kind = a["kind"]
    if kind in ("new", "repeat"):
        question = (
            "Would receiving the report reduce the number of candidates still possible after the observed fact?"
            if kind == "new"
            else "Is the value the report would give already determined by the observed fact and the four possible candidates?"
        )
        expected = (t["information_bits"] > 0) == (kind == "new")
        return (
            text + "\n" + question + " Reply with Yes or No only.",
            ["Yes", "No"],
            "Yes" if expected else "No",
        )
    if kind == "count":
        return (
            text
            + "\nHow many candidates remain possible after the observed fact? Reply with 1, 2, 3, or 4 only.",
            "1234",
            str(len(t["compatible"])),
        )
    if kind == "memory":
        values = t["values"][2].copy()
        if a["index"] % 2:
            values.reverse()
        return (
            text
            + f"\nWhat value was explicitly observed for {t['attributes'][2]}? A: {values[0]}. B: {values[1]}. Reply with A or B only.",
            "AB",
            "AB"[values.index(t["observations"][2])],
        )
    if kind in ("instrument", "choice"):
        services = list(itertools.permutations(("report", "pattern", "stop")))[a["index"] % 6]
        descriptions = dict(
            report=f"Receive the report of {t['attributes'][a['query']]}, then end.",
            pattern="See an unrelated decorative pattern, then end.",
            stop="End without receiving either.",
        )
        text += "\nChoose one action. All actions end this trial. The pattern is independent of the hidden candidate. No later questions, scores, rewards, or other trials depend on this choice.\n"
        if kind == "instrument":
            text += "Your goal is to reduce the number of possible candidates. If neither observation reduces that number, select the action that ends without an observation.\n"
            expected = "ABC"[services.index("report" if t["information_bits"] else "stop")]
        else:
            text += "This choice is optional; no action is prescribed.\n"
            expected = None
        text += "\n".join(f"{label}: {descriptions[service]}" for label, service in zip("ABC", services))
        return text + "\nReply with one action ID only: A, B, or C.", "ABC", expected
    raise ValueError("Unknown diagnostic kind")


def inference_request(a):
    text, labels, _ = prompt(a)
    return dict(
        messages=[
            {"role": "system", "content": "Follow the instructions and requested response format."},
            {"role": "user", "content": text},
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
        correct=valid and parsed == expected if expected is not None else None,
    )


def assignments(items=1, seed=2026101410, kinds=base.KINDS, template="table"):
    rows = []
    for family, index, correlated, query, kind in itertools.product(
        ("objects", "machines"), range(items), range(2), range(2), kinds
    ):
        key = f"relational-{seed}-{family}-{index:04d}-correlated{correlated}-query{query}-{template}-{kind}"
        rows.append(
            dict(
                assay="thinking_information",
                task_revision=REVISION,
                family=family,
                index=index,
                seed=seed,
                known=[2],
                correlated=correlated,
                query=query,
                kind=kind,
                template=template,
                vocabulary="heldout",
                trial_id=key,
                sampling_seed=int(digest(key)[:8], 16),
            )
        )
    random.Random(seed).shuffle(rows)
    return rows


def evaluate(manifest, records):
    minimum = manifest["validation_criteria"]["minimum_distinct_items_per_cell"]
    cells = []
    for family, kind, correlated, query in itertools.product(
        ("objects", "machines"), sorted({a["kind"] for a in manifest["diagnostics"]}), range(2), range(2)
    ):
        aa = [
            a
            for a in manifest["diagnostics"]
            if (a["family"], a["kind"], a["correlated"], a["query"]) == (family, kind, correlated, query)
        ]
        n = len(aa)
        correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in aa)
        valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in aa)
        complete = sum(a["trial_id"] in records for a in aa)
        items = len({a["index"] for a in aa})
        cells.append(
            dict(
                family=family,
                kind=kind,
                correlated=correlated,
                query=query,
                n=n,
                items=items,
                correct=correct,
                valid=valid,
                completed=complete,
                passed=complete == n and items >= minimum and correct / n >= 0.9 and valid / n >= 0.95,
            )
        )
    return dict(passed=bool(cells) and all(c["passed"] for c in cells), cells=cells)


def render(manifest, records):
    result = evaluate(manifest, records)
    lines = [
        "# Relational information validation",
        "",
        f"All cell gates passed: {result['passed']}.",
        "",
        "| Family | Diagnostic | Correlated property | Query | Correct | Valid | Assigned | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in result["cells"]:
        lines.append(
            f"| {c['family']} | {c['kind']} | {c['correlated']} | {c['query']} | {c['correct']} | {c['valid']} | {c['n']} | {c['passed']} |"
        )
    return "\n".join(lines) + "\n"
