"""Recompute scoring and cohort invariants without replacing original model records."""

import argparse
from pathlib import Path
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.information_diagnostic import inference_request, record_from_inference, summarize


def audit(root):
    manifest = read(root / "manifest.json")
    records = RunStore(root, manifest).completed()
    assignments = manifest["diagnostics"]
    ids = {a["trial_id"] for a in assignments}
    assert set(records) == ids, "Incomplete or extra records"
    physical = list((root / "attempts").glob("*/*.json"))
    assert len(physical) == len(ids), "Duplicate physical records require investigation"
    cohorts = {}
    total_tokens = 0
    mismatches = []
    for offset in range(0, len(assignments), 8):
        cohort = assignments[offset : offset + 8]
        expected_ids = [a["trial_id"] for a in cohort]
        batch_ids = set()
        for row, a in enumerate(cohort):
            record = records[a["trial_id"]]
            assert record["assignment"] == a
            inference = record["inference"]
            request = inference_request(a)
            assert inference["messages"] == request["messages"]
            assert inference["prompt_sha256"] == digest(inference["rendered_prompt"])
            assert inference["sampling"] == dict(
                seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=request["max_new_tokens"]
            )
            assert inference["enable_thinking"] == request["enable_thinking"]
            assert inference["batch"]["cohort_ids"] == expected_ids
            assert inference["batch"]["row"] == row
            assert inference["hooks_remaining"] == 0 and inference["intervention"] is None
            check = record_from_inference(a, inference)
            for field in ("answer", "expected", "valid", "correct"):
                if record[field] != check[field]:
                    mismatches.append(
                        dict(
                            trial_id=a["trial_id"],
                            field=field,
                            original=record[field],
                            recomputed=check[field],
                        )
                    )
            batch_ids.add(inference["batch"]["id"])
            total_tokens += len(inference["generated_token_ids"])
        assert len(batch_ids) == 1
        cohorts[str(offset)] = next(iter(batch_ids))
    assert not mismatches, "Scoring discrepancy; retain originals and investigate"
    return dict(
        complete=True,
        assigned=len(assignments),
        records=len(records),
        original_cohorts=len(cohorts),
        tokens=total_tokens,
        manifest_sha256=digest(manifest),
        scoring_mismatches=mismatches,
        cells=summarize(manifest, records),
        scope="development_diagnosis_not_advancement",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run)
    write_once(args.out, result)
    print({k: v for k, v in result.items() if k != "cells"})
