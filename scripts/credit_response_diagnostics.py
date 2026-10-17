"""Describe response failures after complete collection, without changing any scores."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.identification import final_answer


def describe(root):
    manifest = read(root / "manifest.json")
    assert manifest["assay"] == "one_credit"
    records = RunStore(root, manifest).completed()
    assignments = {a["trial_id"]: a for a in manifest["diagnostics"] + manifest["assignments"]}
    assert set(records) == set(assignments), "Describe only the completed original sample"
    groups = defaultdict(list)
    for tid, a in assignments.items():
        r = records[tid]
        assert r["assignment"] == a
        groups[(a["family"], a.get("kind", "behavior"))].append(r)
    rows = []
    for (family, kind), items in sorted(groups.items()):
        failures = Counter()
        confusion = Counter()
        examples = defaultdict(list)
        for r in items:
            inference = r["inference"]
            thinking = r["assignment"].get("inference_mode") == "think_sampled"
            answer, closed = final_answer(inference["raw"], thinking)
            if not r["valid"]:
                category = "missing_thinking_closure" if not closed else "invalid_final_answer"
            elif r["correct"] is False:
                category = "valid_wrong_answer"
                confusion[f"{r['expected']}->{answer}"] += 1
            else:
                category = "valid_correct" if kind != "behavior" else "valid_action"
            failures[category] += 1
            if len(examples[category]) < 3:
                examples[category].append(r["trial_id"])
        tokens = [len(r["inference"]["generated_token_ids"]) for r in items]
        rows.append(
            {
                "family": family,
                "kind": kind,
                "assigned": len(items),
                "response_categories": dict(failures),
                "valid_wrong_answer_confusion": dict(confusion),
                "example_trial_ids": dict(examples),
                "mean_generated_tokens": mean(tokens),
                "mean_inference_seconds": mean(r["inference"]["seconds"] for r in items),
                "at_token_limit_without_eos": sum(
                    len(r["inference"]["generated_token_ids"]) == r["inference"]["sampling"]["max_new_tokens"]
                    and r["inference"].get("ended_with_eos") is False
                    for r in items
                ),
            }
        )
    batches = {
        r["inference"]["batch"]["id"]: r["inference"]["batch"]
        for r in records.values()
        if "batch" in r["inference"]
    }
    return {
        "timing_semantics": "Per-row inference seconds are overlapping latency; use unique batch wall time"
        if batches
        else "Serial per-inference latency; excludes load and orchestration",
        "unique_batches": len(batches),
        "unique_batch_wall_seconds": sum(b["wall_seconds"] for b in batches.values()),
        "scope": "descriptive_failure_analysis_no_rescoring_or_gate_change",
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
        "rows": rows,
        "limits": "A missing closure or token-limit hit does not establish what answer a longer run would produce. Valid wrong answers do not identify their internal cause. Timing is not a controlled hardware benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, describe(args.run))
