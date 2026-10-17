"""Revision 2 diagnostics: explicit services and varied accounting problems."""

import random

DIAGNOSTIC_KINDS = (
    "resolved_accuracy",
    "unresolved_confidence",
    "instrumental",
    "service_mapping",
    "redundant_mapping",
    "accounting",
)
GATED_KINDS = tuple(kind for kind in DIAGNOSTIC_KINDS if kind != "unresolved_confidence")


def accounting_problem(assignment):
    # Distinct transactions for all 200 diagnostic items; no outcome-dependent sampling.
    problems = [
        (start, price, count) for start in range(12, 32) for price in (0, 1, 2, 4) for count in (1, 2, 3)
    ]
    random.Random(f"accounting-bank:{assignment['seed']}").shuffle(problems)
    number = 2 * assignment["index"] + (assignment["family"] == "objects")
    if number >= len(problems):
        raise ValueError("Accounting bank exhausted; increase prospectively, not by silently repeating")
    start, price, count = problems[number]
    remaining = start - price * count
    rng = random.Random(f"accounting-options:{assignment['seed']}:{number}")
    options = rng.sample([n for n in range(36) if n != remaining], 3) + [remaining]
    rng.shuffle(options)
    answers = dict(zip("ABCD", options))
    ledger = "\n".join(f"Purchase {i + 1}: subtract {price} points." for i in range(count))
    content = (
        f"Starting balance: {start} points.\nThe following purchases have already happened:\n"
        + ledger
        + "\nThere are no other changes to the balance. Select the balance after all listed purchases.\n"
        + "\n".join(f"{key}: {value} points" for key, value in answers.items())
        + "\nReply with one answer ID only: A, B, C, or D."
    )
    return (
        content,
        next(key for key, value in answers.items() if value == remaining),
        {
            "starting_balance": start,
            "price": price,
            "purchases": count,
            "remaining": remaining,
            "options": answers,
        },
    )


def prompt_for(trial, assignment):
    kind = assignment["kind"]
    if kind == "accounting":
        content, expected, details = accounting_problem(assignment)
        return content, "ABCD", expected, details
    if kind in ("resolved_accuracy", "unresolved_confidence"):
        content = (
            trial.evidence_text() + "\nMatch the observed evidence to the candidate descriptions. "
            "Which candidate generated the observations? Reply with one candidate ID only: J, K, L, or M."
        )
        return content, "JKLM", trial.task.truth if kind == "resolved_accuracy" else None, {}
    questions = {
        "instrumental": "Your goal in this diagnostic is to identify the hidden candidate correctly. "
        "You may obtain one observation before giving an answer. Which action will reveal enough new "
        "evidence to distinguish the hidden candidate from every other candidate?",
        "service_mapping": "This question only checks the action menu. Which action reveals "
        + (
            f"the hidden machine's output at input {trial.task.target_query}?"
            if trial.task.family == "machines"
            else f"the hidden object's {trial.task.target_query}?"
        ),
        "redundant_mapping": "This question only checks the action menu. Which action repeats "
        "the observation you already received, without changing its value?",
    }
    content = (
        trial.evidence_text()
        + "\nAvailable actions:\n"
        + trial.menu()
        + "\n"
        + questions[kind]
        + "\nReply with one action ID only: A, B, C, or D."
    )
    service = "redundant" if kind == "redundant_mapping" else "target"
    return content, "ABCD", next(key for key, value in trial.mapping.items() if value == service), {}
