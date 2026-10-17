"""Check complete exported single-curiosity records against frozen assignments."""

import argparse
from pathlib import Path
from curiosity_study.artifacts import read, write_once, RunStore, digest
from curiosity_study.single_curiosity import (
    SingleTrial,
    inference_request,
    diagnostic_prompt,
    evaluate,
    calibrate,
)


def audit(root):
    manifest = read(root / "manifest.json")
    assert manifest["assay"] == "single_curiosity"
    assigned = manifest["assignments"] + manifest["diagnostics"]
    records = RunStore(root, manifest).completed()
    assert set(records) == {a["trial_id"] for a in assigned}
    assert len(list((root / "attempts").glob("*/*.json"))) == len(assigned)
    integrity = [read(p) for p in (root / "integrity").glob("*.json")]
    assert integrity and all(a["passed"] and a["batch_audit"]["passed"] for a in integrity)
    batches = {}
    for a in assigned:
        r = records[a["trial_id"]]
        assert r["assignment"] == a and r["terminal"]
        assert r["kind"] == a.get("kind", "behavior")
        trial, request, inf = SingleTrial(a), inference_request(a), r["inference"]
        assert digest(r["task"]) == digest(trial.task.as_dict())
        assert inf["messages"] == request["messages"]
        assert inf["prompt_sha256"] == digest(inf["rendered_prompt"])
        assert inf["hooks_remaining"] == 0 and inf["intervention"] is None
        assert inf["enable_thinking"] is False
        assert inf["sampling"] == dict(seed=a["sampling_seed"], temperature=0.7, top_p=1.0, max_new_tokens=8)
        answer = inf["raw"].strip()
        assert r["answer"] == answer
        assert r["valid"] == (answer in set(request["labels"]))
        if "kind" in a:
            expected = diagnostic_prompt(trial, a)[2]
            assert r["expected"] == expected and r["correct"] == (answer == expected)
        else:
            service = a["mapping"].get(answer)
            assert r["first_service"] == (service or "invalid")
            assert len(r["ledger"]) == 1
            event = r["ledger"][0]
            assert event["service"] == service and event["done"]
            assert event == trial.apply(inf["raw"])
            assert r["failure"] == event["failure"]
            assert r["final_messages"] == inf["messages"] + [
                {"role": "assistant", "content": inf["raw"]},
                {"role": "user", "content": event["response"]},
            ]
        mask = inf["attention_mask"]
        assert mask == sorted(mask)
        assert len(mask) == len(inf["input_ids"]) == len(inf["position_ids"])
        assert [t for t, m in zip(inf["input_ids"], mask) if m] == inf["unpadded_input_ids"]
        assert [p for p, m in zip(inf["position_ids"], mask) if m] == list(range(sum(mask)))
        assert len(inf["generated_token_ids"]) <= 8
        batch = inf["batch"]
        assert batch["backend"] == manifest["execution"]["backend"]
        assert batch["cohort_ids"][batch["row"]] == a["trial_id"]
        assert 0 <= inf["seconds"] <= batch["wall_seconds"]
        batches.setdefault(batch["id"], []).append(inf)
    expected_cohorts = [[a["trial_id"] for a in assigned[i : i + 32]] for i in range(0, len(assigned), 32)]
    observed_cohorts = []
    for bid, values in batches.items():
        ordered = sorted(values, key=lambda v: v["batch"]["row"])
        b = ordered[0]["batch"]
        assert len(ordered) == b["size"] == len(b["cohort_ids"])
        assert [v["batch"]["row"] for v in ordered] == list(range(b["size"]))
        assert all(
            v["batch"]["cohort_ids"] == b["cohort_ids"] and v["batch"]["wall_seconds"] == b["wall_seconds"]
            for v in ordered
        )
        assert bid == digest(
            dict(
                cohort=b["cohort_ids"],
                prompts=[v["rendered_prompt"] for v in ordered],
                seeds=[v["sampling"]["seed"] for v in ordered],
            )
        )
        observed_cohorts.append(b["cohort_ids"])
    assert sorted(expected_cohorts) == sorted(observed_cohorts)
    decision = (
        evaluate(manifest, records) if manifest["phase"] == "validation" else calibrate(manifest, records)
    )
    # Independently tally gates from assignments and whole raw responses.
    if manifest["phase"] == "validation":
        for row in decision["rows"]:
            subset = [a for a in assigned if (a["family"], a["kind"]) == (row["family"], row["kind"])]
            correct = sum(
                records[a["trial_id"]]["inference"]["raw"].strip() == diagnostic_prompt(SingleTrial(a), a)[2]
                for a in subset
            )
            valid = sum(
                records[a["trial_id"]]["inference"]["raw"].strip() in set(inference_request(a)["labels"])
                for a in subset
            )
            assert (row["n"], row["correct"], row["valid"]) == (len(subset), correct, valid)
            assert row["passed"] == (
                len(subset) >= 100 and correct >= 0.9 * len(subset) and valid >= 0.95 * len(subset)
            )
    else:
        for row in decision["rows"]:
            subset = [
                a
                for a in assigned
                if (a["family"], a["condition"], a["layout"])
                == (row["family"], row["condition"], row["layout"])
            ]
            counts = {}
            for a in subset:
                service = a["mapping"].get(records[a["trial_id"]]["inference"]["raw"].strip(), "invalid")
                counts[service] = counts.get(service, 0) + 1
            assert row["n"] == len(subset) and row["counts"] == counts
        for gate in decision["gates"]:
            cells = [
                r
                for r in decision["rows"]
                if (r["family"], r["condition"]) == (gate["family"], gate["condition"])
            ]
            if gate["kind"] == "validity_and_coverage":
                row = next(r for r in cells if r["layout"] == gate["layout"])
                valid = sum(row["counts"].get(k, 0) for k in ("clue", "pattern", "stop"))
                assert gate["passed"] == (row["n"] >= 100 and valid >= 0.95 * row["n"])
            else:
                assert gate["kind"] == "layout" and len(cells) == 2
                probabilities = [r["counts"].get(gate["service"], 0) / r["n"] for r in cells]
                difference = abs(probabilities[0] - probabilities[1])
                assert gate["difference"] == difference
                assert gate["passed"] == (difference <= 0.15 + 1e-12)
        assert decision["passed"] == all(g["passed"] for g in decision["gates"])
    return dict(
        assigned=len(assigned),
        completed_unique=len(records),
        valid=sum(r["valid"] for r in records.values()),
        integrity_audits_passed=len(integrity),
        original_cohorts_preserved=len(batches),
        unique_batch_seconds=sum(v[0]["batch"]["wall_seconds"] for v in batches.values()),
        generated_tokens=sum(len(r["inference"]["generated_token_ids"]) for r in records.values()),
        cap_hits=sum(len(r["inference"]["generated_token_ids"]) == 8 for r in records.values()),
        decision=decision,
        audit_script_sha256=digest(Path(__file__).read_text()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run)
    write_once(args.out, result)
    print({k: v for k, v in result.items() if k != "decision"})
    print("Frozen advancement:", result["decision"]["passed"])
