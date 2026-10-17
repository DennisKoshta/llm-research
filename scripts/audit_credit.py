"""Audit exported one-credit records and score the prospectively frozen criteria."""

import argparse
from pathlib import Path
import json
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.credit import evaluate, calibrate, CreditTrial, diagnostic_prompt
from curiosity_study.identification import final_answer


def audit(root):
    manifest = read(root / "manifest.json")
    assert manifest["assay"] == "one_credit"
    records = RunStore(root, manifest).completed()
    assignments = {a["trial_id"]: a for a in manifest["assignments"] + manifest["diagnostics"]}
    expected = set(assignments)
    assert set(records) == expected
    assert len(list((root / "attempts").glob("*/*.json"))) == len(expected)
    audits = [read(p) for p in (root / "integrity").glob("*.json")]
    assert audits and all(a["passed"] for a in audits)
    for record in records.values():
        a = record["assignment"]
        assert a == assignments[record["trial_id"]]
        assert record["kind"] == a.get("kind", "behavior")
        trial = CreditTrial(a)
        content, labels, expected_answer = (
            diagnostic_prompt(trial, a) if record["kind"] != "behavior" else (trial.prompt(), "ABC", None)
        )
        inference = record["inference"]
        assert inference["hooks_remaining"] == 0
        assert inference["intervention"] is None
        thinking = a.get("inference_mode") == "think_sampled"
        if "inference_mode" in a:
            assert inference["enable_thinking"] == thinking
        greedy = (
            record["kind"] != "behavior" and a.get("inference_mode", "no_think_greedy") == "no_think_greedy"
        )
        assert inference["sampling"] == {
            "seed": a["sampling_seed"],
            "temperature": 0 if greedy else 0.7,
            "top_p": 1.0,
            "max_new_tokens": a.get("max_new_tokens", 768 if thinking else 8),
        }
        assert inference["messages"][-1]["content"] == content
        assert inference["prompt_sha256"] == digest(inference["rendered_prompt"])
        raw, closed = final_answer(inference["raw"], a.get("inference_mode") == "think_sampled")
        assert record["expected"] == expected_answer
        assert record["correct"] == (None if expected_answer is None else closed and raw == expected_answer)
        assert record["valid"] == (closed and raw in set(labels))
        assert digest(record["tasks"]) == digest({side: task.as_dict() for side, task in trial.tasks.items()})
        if record["kind"] == "behavior":
            assert len(record["ledger"]) == 1
            event = record["ledger"][0]
            assert event["credits_before"] == 1 and event["done"]
            selected = record["assignment"]["mapping"].get(raw)
            assert record["first_service"] == (selected or "invalid")
            if selected in ("X", "Y"):
                other = "Y" if selected == "X" else "X"
                assert event["delivered"] == [selected] and event["withheld"] == [other]
                assert event["credits_after"] == 0
                assert f"Mystery {other}:" not in event["response"]
            else:
                assert not event["delivered"] and set(event["withheld"]) == {"X", "Y"}
                assert event["credits_after"] == 1
    if manifest.get("execution", {}).get("backend") == "torch_batched_v1":
        assert all(a.get("batch_audit", {}).get("passed") is True for a in audits)
        cohorts = {}
        for tid, record in records.items():
            inference = record["inference"]
            batch = inference["batch"]
            assert batch["backend"] == manifest["execution"]["backend"]
            assert 1 <= batch["size"] <= manifest["execution"]["batch_size"]
            assert len(batch["cohort_ids"]) == len(set(batch["cohort_ids"])) == batch["size"]
            assert batch["cohort_ids"][batch["row"]] == tid
            assert all(member in records for member in batch["cohort_ids"])
            mask = inference["attention_mask"]
            assert mask == sorted(mask)
            assert [token for token, m in zip(inference["input_ids"], mask) if m] == inference[
                "unpadded_input_ids"
            ]
            assert [pos for pos, m in zip(inference["position_ids"], mask) if m] == list(range(sum(mask)))
            assert len(inference["generated_token_ids"]) <= inference["sampling"]["max_new_tokens"]
            assert 0 <= inference["seconds"] <= batch["wall_seconds"]
            cohorts.setdefault(batch["id"], []).append(inference)
        for batch_id, values in cohorts.items():
            ordered = sorted(values, key=lambda value: value["batch"]["row"])
            batch = ordered[0]["batch"]
            assert [v["batch"]["row"] for v in ordered] == list(range(batch["size"]))
            assert all(
                v["batch"]["cohort_ids"] == batch["cohort_ids"]
                and v["batch"]["wall_seconds"] == batch["wall_seconds"]
                for v in ordered
            )
            assert batch_id == digest(
                {
                    "cohort": batch["cohort_ids"],
                    "prompts": [v["rendered_prompt"] for v in ordered],
                    "seeds": [v["sampling"]["seed"] for v in ordered],
                }
            )
    decision = (
        evaluate(manifest, records) if manifest["phase"] == "validation" else calibrate(manifest, records)
    )
    summary = {
        "assigned": len(expected),
        "completed_unique": len(records),
        "valid": sum(r["valid"] for r in records.values()),
        "integrity_audits_passed": len(audits),
        "all_hooks_removed": True,
        "decision": decision,
    }
    if manifest["phase"] == "behavior":
        contrasts = []
        for family in ("machines", "objects"):
            probabilities = {}
            for condition in ("both_unresolved", "x_resolved", "y_resolved"):
                subset = [
                    r
                    for r in records.values()
                    if r["assignment"]["family"] == family and r["assignment"]["condition"] == condition
                ]
                probabilities[condition] = {
                    choice: sum(r["first_service"] == choice for r in subset) / len(subset)
                    for choice in ("X", "Y", "stop", "invalid")
                }
            contrasts.append(
                {
                    "family": family,
                    "absolute_probabilities": probabilities,
                    "resolution_differences": {
                        condition: {
                            choice: probabilities[condition][choice]
                            - probabilities["both_unresolved"][choice]
                            for choice in ("X", "Y", "stop", "invalid")
                        }
                        for condition in ("x_resolved", "y_resolved")
                    },
                }
            )
        summary["descriptive_resolution_contrasts"] = contrasts
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    write_once(args.out, result)
    print(json.dumps(result, indent=2))
