"""Prospective diagnosis of informational novelty judgments, separate from advancement."""

import copy
import itertools
import random
from collections import defaultdict
from .single_curiosity import SingleTrial, assignments as single_assignments, description_question
from .identification import final_answer
from .runner import source_hash
from .artifacts import digest

EXECUTION = {"backend": "torch_batched_v1", "batch_size": 8}
SYSTEM = {
    "role": "system",
    "content": "Answer the user's question accurately. Follow the requested final-answer format.",
}


def assignments(items=12, seed=2026101401):
    bases = {}
    for a in single_assignments(items, seed):
        bases.setdefault((a["family"], a["index"]), a)
    rows = []
    for original in bases.values():
        for condition, form, thinking in itertools.product(
            ("unresolved", "resolved"), ("description", "yes_no", "explanation"), (False, True)
        ):
            a = copy.deepcopy(original)
            key = f"information-{a['family']}-{a['index']:04d}-{condition}-{form}-think{int(thinking)}"
            a.update(
                assay="information_diagnostic",
                task_revision="single-curiosity-v2",
                condition=condition,
                layout="forward" if a["index"] % 2 else "reverse",
                form=form,
                thinking=thinking,
                kind="information_novelty",
                trial_id=key,
                polarity="already" if a["index"] % 2 else "new",
                execution=EXECUTION.copy(),
                sampling_seed=random.Random(f"diagnose:{seed}:{a['family']}:{a['index']}").randrange(2**31),
            )
            rows.append(a)
    # Stable cohorts by generation regime avoid short branches waiting for long branches.
    random.Random(f"diagnostic-order:{seed}").shuffle(rows)
    rows.sort(key=lambda a: (a["thinking"], a["form"]))
    return rows


def prompt(a):
    trial = SingleTrial(a)
    if a["form"] == "description":
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
            "repeat" if a["condition"] == "resolved" else "identify",
        )
    content = trial.evidence() + "\n" + trial.rules() + "\nAction menu:\n" + trial.menu()
    if a["form"] == "yes_no":
        question = (
            "Is the fact that the offered clue would reveal already explicitly present in the observed evidence?"
            if a["polarity"] == "already"
            else "Would the offered clue reveal a fact that is not already present in the observed evidence?"
        )
        yes = (a["condition"] == "resolved") == (a["polarity"] == "already")
        return (
            content + "\n" + question + " Reply with Yes or No only.",
            ["Yes", "No"],
            "Yes" if yes else "No",
        )
    if a["form"] == "explanation":
        return (
            content
            + "\nExplain what, if anything, the offered clue would add to the evidence already shown. "
            "Give your final answer in one or two sentences.",
            ["Yes", "No"],
            None,
        )
    raise ValueError("Unknown question form")


def inference_request(a):
    content, labels, _ = prompt(a)
    return dict(
        messages=[SYSTEM, {"role": "user", "content": content}],
        seed=a["sampling_seed"],
        labels=labels,
        greedy=False,
        enable_thinking=a["thinking"],
        max_new_tokens=2048 if a["thinking"] else (256 if a["form"] == "explanation" else 8),
    )


def record_from_inference(a, inference):
    _, labels, expected = prompt(a)
    answer, closed = final_answer(inference["raw"], a["thinking"])
    free = a["form"] == "explanation"
    valid = closed and (bool(answer) if free else answer in set(labels))
    return dict(
        trial_id=a["trial_id"],
        assignment=a,
        kind=a["kind"],
        terminal=True,
        task=SingleTrial(a).task.as_dict(),
        inference=inference,
        answer=answer,
        expected=expected,
        valid=valid,
        correct=None if free else valid and answer == expected,
        qualitative_only=free,
    )


def build_manifest(model, items=12, seed=2026101401):
    rows = assignments(items, seed)
    return dict(
        schema_version=3,
        assay="information_diagnostic",
        phase="development_diagnosis",
        task_revision="single-curiosity-v2",
        model=model,
        execution_gpu="H100",
        execution=EXECUTION,
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        invalid_output_policy="terminal_no_retry",
        advancement="none; diagnosis informs a separately frozen mechanism assay",
        explanation_rubric={
            "resolved": "Must say the clue adds no new evidence / repeats a known fact; identify contradictions separately.",
            "unresolved": "Must say the clue supplies new discriminating evidence; identify contradictions separately.",
            "coding": "Qualitative descriptive inspection only; not an automatically scored advancement criterion.",
        },
    )


def summarize(manifest, records):
    groups = defaultdict(lambda: dict(assigned=0, completed=0, valid=0, correct=0, cap_hits=0))
    for a in manifest["diagnostics"]:
        key = (a["family"], a["form"], a["thinking"], a["condition"])
        cell = groups[key]
        cell["assigned"] += 1
        r = records.get(a["trial_id"])
        if r is None:
            continue
        cell["completed"] += 1
        cell["valid"] += bool(r.get("valid"))
        cell["correct"] += r.get("correct") is True
        inf = r.get("inference", {})
        cell["cap_hits"] += (
            bool(inf)
            and not inf["ended_with_eos"]
            and len(inf["generated_token_ids"]) == inf["sampling"]["max_new_tokens"]
        )
    return [
        dict(family=k[0], form=k[1], thinking=k[2], condition=k[3], **v) for k, v in sorted(groups.items())
    ]


def render(manifest, records):
    lines = [
        "# Informational novelty diagnostic",
        "",
        f"Model: {manifest['model']['model_id']}",
        "",
        "Development diagnosis only. Explanations are qualitative; they have no automatic accuracy score.",
        "",
        "| Family | Format | Thinking | Evidence | Correct | Valid | Complete | Assigned | Cap hits |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in summarize(manifest, records):
        fields = [c[k] for k in ("family", "form", "thinking", "condition")]
        fields += ["—" if c["form"] == "explanation" else c["correct"]]
        fields += [c[k] for k in ("valid", "completed", "assigned", "cap_hits")]
        lines.append("| " + " | ".join(map(str, fields)) + " |")
    return "\n".join(lines) + "\n"
