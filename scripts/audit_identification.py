"""Reconstruct frozen identification prompts, answers, denominators and selection."""

import argparse
from pathlib import Path
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.identification import prompt, final_answer, summarize, MODES


def audit(root):
    m = read(root / "manifest.json")
    records = RunStore(root, m).completed()
    assignments = {a["trial_id"]: a for a in m["diagnostics"]}
    assert set(records) == set(assignments)
    assert len(list((root / "attempts").glob("*/*.json"))) == len(assignments)
    integrity = [read(p) for p in (root / "integrity").glob("*.json")]
    assert integrity and all(r["passed"] for r in integrity)
    for tid, r in records.items():
        a = assignments[tid]
        assert r["assignment"] == a
        content, expected, _ = prompt(a)
        inference = r["inference"]
        assert inference["messages"][-1]["content"] == content
        assert inference["prompt_sha256"] == digest(inference["rendered_prompt"])
        assert inference["hooks_remaining"] == 0
        thinking = a["mode"] == "think_sampled"
        answer, closed = final_answer(inference["raw"], thinking)
        assert r["answer"] == answer and r["expected"] == expected
        assert r["correct"] == (closed and answer == expected)
        assert r["valid"] == (closed and answer in set(a["labels"]))
        assert inference["enable_thinking"] == thinking
        assert inference["sampling"]["temperature"] == (0 if a["mode"] == "no_think_greedy" else 0.7)
    rows = summarize(m, records)
    selection = None
    for labels in ("ABCD", "JKLM"):
        for mode in MODES:
            cells = [
                r for r in rows if r["labels"] == labels and r["mode"] == mode and r["context"] == "paired"
            ]
            if len(cells) == 2 and all(
                r["n"] >= 50 and r["correct"] >= 0.9 * r["n"] and r["valid"] >= 0.95 * r["n"] for r in cells
            ):
                selection = {"labels": labels, "mode": mode}
                break
        if selection:
            break
    # Matched descriptive differences; no unpaired standard errors or p-value selection.
    contrasts = []
    for family in ("machines", "objects"):
        for labels in ("ABCD", "JKLM"):
            for mode in MODES:
                pairs = {}
                for r in records.values():
                    a = r["assignment"]
                    if (a["family"], a["labels"], a["mode"]) == (family, labels, mode):
                        pairs.setdefault((a["index"], a["side"]), {})[a["context"]] = r
                contrasts.append(
                    {
                        "family": family,
                        "labels": labels,
                        "mode": mode,
                        "n": len(pairs),
                        "both_correct": sum(
                            p["single"]["correct"] and p["paired"]["correct"] for p in pairs.values()
                        ),
                        "single_only_correct": sum(
                            p["single"]["correct"] and not p["paired"]["correct"] for p in pairs.values()
                        ),
                        "paired_only_correct": sum(
                            p["paired"]["correct"] and not p["single"]["correct"] for p in pairs.values()
                        ),
                    }
                )
    label_contrasts = []
    for family in ("machines", "objects"):
        for context in ("single", "paired"):
            for mode in MODES:
                pairs = {}
                for r in records.values():
                    a = r["assignment"]
                    if (a["family"], a["context"], a["mode"]) == (family, context, mode):
                        pairs.setdefault((a["index"], a["side"]), {})[a["labels"]] = r
                label_contrasts.append(
                    {
                        "family": family,
                        "context": context,
                        "mode": mode,
                        "n": len(pairs),
                        "both_correct": sum(
                            p["ABCD"]["correct"] and p["JKLM"]["correct"] for p in pairs.values()
                        ),
                        "ABCD_only_correct": sum(
                            p["ABCD"]["correct"] and not p["JKLM"]["correct"] for p in pairs.values()
                        ),
                        "JKLM_only_correct": sum(
                            p["JKLM"]["correct"] and not p["ABCD"]["correct"] for p in pairs.values()
                        ),
                    }
                )
    return {
        "assigned": len(assignments),
        "verified_unique": len(records),
        "integrity_passed": True,
        "rows": rows,
        "matched_context_contrasts": contrasts,
        "matched_label_contrasts": label_contrasts,
        "selected_for_fresh_validation": selection,
        "manifest_sha256": digest(m),
        "records_sha256": digest(records),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run)
    write_once(args.out, result)
    print(result["selected_for_fresh_validation"])
