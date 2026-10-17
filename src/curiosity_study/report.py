"""Descriptive development summaries, not confirmatory statistics."""

from collections import Counter
from .artifacts import RunStore, read


def render(root):
    manifest = read(root / "manifest.json")
    records = RunStore(root, manifest).completed()
    if manifest.get("assay") == "curiosity_suppression_screen":
        return (
            "# Curiosity-suppression development collection\n\n"
            f"Terminal generation rows: {len(records)}. "
            "Interpretation requires the complete 4,320-row parent and 2,160 receipt dispositions, "
            "all treatment controls, and the separate combined audit/analysis."
        )
    if manifest.get("assay") == "satisfaction_screen":
        return (
            "# Satisfaction development collection\n\n"
            f"Completed {len(records)}/{len(manifest['diagnostics'])} assigned generation records. "
            "All three children, all receipt dispositions, and the combined independent audit are required "
            "before a behavioral decision. This operational summary grants no scientific admission."
        )
    if manifest.get("assay") == "sustained_engineering":
        from .sustained_engineering import render as render_sustained

        return render_sustained(manifest, records)
    if manifest.get("assay") == "voluntary_resolution":
        rows = manifest["diagnostics"]
        choices = [a for a in rows if a["kind"] == "choice"]
        diagnostics = [a for a in rows if a["kind"] != "choice"]
        counts = Counter(
            records.get(a["trial_id"], {}).get("selected_service") or "invalid_or_pending" for a in choices
        )
        correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in diagnostics)
        return (
            f"# Voluntary-resolution development collection\n\n"
            f"Completed {len(records)}/{len(rows)} assigned generation records. "
            f"Diagnostic answers correct: {correct}/{len(diagnostics)}.\n\n"
            f"All-assigned choice counts: {dict(counts)}.\n\n"
            "Generation completion does not establish receipt completion, readiness, or a causal mechanism. "
            "Run the separate voluntary-resolution audit before interpretation."
        )
    if manifest.get("assay") == "concept_capture":
        from .concept_capture import render as render_concepts

        return render_concepts(manifest, records)
    if manifest.get("assay") == "natural_information":
        from .natural_information import render as render_natural

        return render_natural(manifest, records)
    if manifest.get("assay") == "replay_information":
        from .replay_information import render as render_replay

        return render_replay(manifest, records)
    if manifest.get("assay") == "thinking_information":
        from .thinking_information import render as render_thinking

        return render_thinking(manifest, records)
    if manifest.get("assay") == "state_mechanism":
        from .state_experiment import render as render_state_mechanism

        return render_state_mechanism(manifest, records)
    if manifest.get("assay") == "information_state":
        from .information_state import render as render_state

        return render_state(manifest, records)
    if manifest.get("assay") == "information_diagnostic":
        from .information_diagnostic import render as render_information

        return render_information(manifest, records)
    if manifest.get("assay") == "identification_factorial":
        from .identification import render as render_identification

        return render_identification(manifest, records)
    if manifest.get("assay") == "single_curiosity":
        from .single_curiosity import render as render_single

        return render_single(manifest, records)
    if manifest.get("assay") == "one_credit":
        from .credit import render as render_credit

        return render_credit(manifest, records)
    lines = [
        "# Development run report",
        "",
        f"Model: `{manifest['model']['model_id']}`.",
        f"Completed {len(records)} / {len(manifest['assignments']) + len(manifest['diagnostics'])} assigned records.",
        "",
        "These are engineering/development observations. No intervention hypothesis or claim about experience is tested.",
        "",
        "First choices include every assigned trial; pending and technical failures retain their denominators.",
        "",
        "| Family | Evidence | Price | Assigned | Target | Repeat | Novelty | Stop | Invalid | Technical | Pending |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups = {}
    for a in manifest["assignments"]:
        key = (a["family"], a["evidence"], a["price"])
        groups.setdefault(key, Counter())
        record = records.get(a["trial_id"])
        groups[key][record["first_service"] if record else "pending"] += 1
    for key, counts in sorted(groups.items()):
        cells = [
            *key,
            sum(counts.values()),
            *(
                counts[s]
                for s in ("target", "redundant", "novelty", "stop", "invalid", "technical_failure", "pending")
            ),
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += [
        "",
        "## Independent diagnostic branches",
        "",
        "| Family | Diagnostic | Correct | Assigned | Pending |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family in ("machines", "objects"):
        kinds = sorted({a["kind"] for a in manifest["diagnostics"] if a["kind"] != "unresolved_confidence"})
        for kind in kinds:
            assigned = [a for a in manifest["diagnostics"] if a["family"] == family and a["kind"] == kind]
            correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in assigned)
            pending = sum(a["trial_id"] not in records for a in assigned)
            lines.append(f"| {family} | {kind} | {correct} | {len(assigned)} | {pending} |")
    lines += [
        "",
        "Candidate probabilities are saved in diagnostic records, including probability mass outside the candidate labels. "
        "They are not treated as a validated internal confidence measure.",
        "",
        "This runner does not yet establish discretionary point valuation, selective closure, natural recruitment, "
        "or causal curiosity/satisfaction effects. Comprehension alone does not validate the full assay.",
        "",
    ]
    if manifest.get("phase") == "validation":
        from .validation import evaluate

        result = evaluate(manifest, records)
        lines += [
            "## Prospective advancement decision",
            "",
            "PASS: a behavioral baseline may proceed."
            if result["passed"]
            else "FAIL: do not collect the next behavioral baseline under this revision.",
            "",
            "Required for both families: at least 100 items per scored diagnostic, at least 90% correct "
            "and 95% valid responses in every scored diagnostic. All assigned diagnostics must be complete.",
            "",
        ]
    return "\n".join(lines)
