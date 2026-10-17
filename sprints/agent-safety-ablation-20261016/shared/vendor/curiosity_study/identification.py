"""Prospective matched development comparison; no behavioral advancement gate."""

import copy
import itertools
import random
import re
from collections import defaultdict
from .credit import CreditTrial, assignments as credit_assignments
from .tasks import describe, describe_observation
from .runner import FORMAT_INSTRUCTION, source_hash

MODES = ("no_think_greedy", "no_think_sampled", "think_sampled")


def assignments(items=25, seed=20261007):
    bases = {}
    for a in credit_assignments(items, seed):
        bases.setdefault((a["family"], a["index"]), a)
    result = []
    for a in bases.values():
        for side, context, labels, mode in itertools.product(
            "XY", ("single", "paired"), ("JKLM", "ABCD"), MODES
        ):
            b = copy.deepcopy(a)
            b.update(
                assay="identification_factorial",
                kind="identification",
                side=side,
                context=context,
                labels=labels,
                mode=mode,
                condition=f"{side.lower()}_resolved",
                layout="XY" if a["index"] % 2 == 0 else "YX",
            )
            # Same random seed within item/side, including mode comparisons.
            b["sampling_seed"] = random.Random(
                f"identify:{seed}:{a['family']}:{a['index']}:{side}"
            ).randrange(2**31)
            b["trial_id"] = f"identify-{a['family']}-{a['index']:04d}-{side}-{context}-{labels}-{mode}"
            result.append(b)
    random.Random(f"identify-order:{seed}").shuffle(result)
    return result


def prompt(a):
    trial = CreditTrial(a)
    side = a["side"]
    task = trial.tasks[side]
    if a["context"] == "paired":
        evidence = trial.evidence()
    else:
        evidence = (
            f"Mystery {side}:\n"
            + describe(task)
            + "\nObserved evidence:\n"
            + "\n".join(describe_observation(task, o) for o in trial.observations[side])
        )
    expected = task.truth
    if a["labels"] == "ABCD":
        mapping = dict(zip("JKLM", "ABCD"))
        evidence = re.sub(r"\b[JKLM]\b", lambda m: mapping[m[0]], evidence)
        expected = mapping[expected]
    content = (
        evidence
        + f"\nMatch the evidence for mystery {side} to its table. Which candidate is hidden in mystery {side}? Reply with one candidate ID only: "
        + ", ".join(a["labels"])
        + "."
    )
    return content, expected, trial


def final_answer(raw, thinking):
    if not thinking:
        return raw.strip(), True
    # Missing delimiter is a failure, including a budget-truncated reasoning trace.
    if "</think>" not in raw:
        return "", False
    return raw.rsplit("</think>", 1)[1].strip(), True


def run(a, model):
    content, expected, trial = prompt(a)
    thinking = a["mode"] == "think_sampled"
    inference = model.infer(
        [FORMAT_INSTRUCTION, {"role": "user", "content": content}],
        a["sampling_seed"],
        labels=a["labels"],
        greedy=a["mode"] == "no_think_greedy",
        enable_thinking=thinking,
        max_new_tokens=768 if thinking else 8,
    )
    answer, closed = final_answer(inference["raw"], thinking)
    return {
        "trial_id": a["trial_id"],
        "assignment": a,
        "kind": a["kind"],
        "terminal": True,
        "inference": inference,
        "expected": expected,
        "answer": answer,
        "correct": closed and answer == expected,
        "valid": closed and answer in set(a["labels"]),
        "thinking_closed": closed,
        "tasks": {s: t.as_dict() for s, t in trial.tasks.items()},
    }


def manifest(model, items=25, seed=20261007):
    return {
        "schema_version": 2,
        "assay": "identification_factorial",
        "task_revision": "identification-v1",
        "purpose": "matched_development_diagnostic_not_confirmation",
        "phase": "diagnostic",
        "source_sha256": source_hash(),
        "model": model,
        "assignments": [],
        "diagnostics": assignments(items, seed),
        "sampling": {"temperature": 0.7, "top_p": 1},
        "analysis": "All assigned records in denominator. Strict final answer; thinking requires closing delimiter. No first-token rescoring.",
        "selection": "Paired cells need >=90% correct and >=95% valid in each family (n=50 each). Prefer no_think_greedy ABCD, then no_think_sampled ABCD, then think_sampled ABCD; retain JKLM only if ABCD fails and corresponding JKLM passes. Selection is development-only; fresh n=100/family/kind validation required.",
    }


def summarize(manifest, records):
    groups = defaultdict(lambda: dict(n=0, correct=0, valid=0, pending=0, unclosed=0, tokens=0))
    for a in manifest["diagnostics"]:
        g = groups[(a["family"], a["context"], a["labels"], a["mode"])]
        r = records.get(a["trial_id"], {})
        g["n"] += 1
        g["correct"] += r.get("correct") is True
        g["valid"] += r.get("valid") is True
        g["pending"] += not bool(r)
        g["unclosed"] += r.get("thinking_closed") is False
        g["tokens"] += len(r.get("inference", {}).get("generated_token_ids", []))
    return [
        {"family": k[0], "context": k[1], "labels": k[2], "mode": k[3], **g}
        for k, g in sorted(groups.items())
    ]


def render(manifest, records):
    rows = summarize(manifest, records)
    lines = [
        "# Matched identification diagnostic",
        "",
        f"Completed {len(records)}/{len(manifest['diagnostics'])}.",
        "",
        "Development comparison only. Thinking mode includes a larger generation budget; its effect cannot be attributed solely to a template switch. Sampled comparisons use one fixed draw per item/side, not repeated reliability estimates.",
        "",
        "| Family | Context | Labels | Mode | N | Correct | Valid | Pending | Unclosed thinking |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r[k])
                for k in (
                    "family",
                    "context",
                    "labels",
                    "mode",
                    "n",
                    "correct",
                    "valid",
                    "pending",
                    "unclosed",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
