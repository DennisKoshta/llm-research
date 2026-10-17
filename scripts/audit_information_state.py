"""Independent run reconciliation and exact replay for the equal-uncertainty instrument."""

import argparse
from collections import Counter
from pathlib import Path
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.information_state import inference_request, record_from_inference, evaluate


def audit(root):
    m = read(root / "manifest.json")
    records = RunStore(root, m).completed()
    aa = m["diagnostics"]
    assert set(records) == {a["trial_id"] for a in aa}
    assert len(list((root / "attempts").glob("*/*.json"))) == len(aa)
    tokens = cap_hits = 0
    worlds = {}
    error_counts = Counter()
    for offset in range(0, len(aa), 16):
        cohort = aa[offset : offset + 16]
        ids = [a["trial_id"] for a in cohort]
        batch_ids = set()
        for row, a in enumerate(cohort):
            record = records[a["trial_id"]]
            inf = record["inference"]
            request = inference_request(a)
            assert record["assignment"] == a
            assert inf["messages"] == request["messages"]
            assert inf["prompt_sha256"] == digest(inf["rendered_prompt"])
            assert inf["sampling"] == dict(seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=8)
            assert not inf["enable_thinking"] and inf["hooks_remaining"] == 0 and inf["intervention"] is None
            assert inf["batch"]["cohort_ids"] == ids and inf["batch"]["row"] == row
            batch_ids.add(inf["batch"]["id"])
            recomputed = record_from_inference(a, inf)
            assert digest(recomputed) == digest(record)
            task = record["task"]
            # Enumerate actual compatible rows independently from task.information_bits.
            remaining = [
                candidate
                for candidate in task["candidates"]
                if all(candidate[int(k)] == value for k, value in task["observations"].items())
            ]
            assert len(remaining) == 2
            values = {candidate[a["query"]] for candidate in remaining}
            assert len(values) == (2 if a["query"] not in a["known"] else 1)
            assert task["uncertainty_bits"] == 1 and task["information_bits"] == len(values) - 1
            key = (a["family"], a["index"])
            signature = digest(
                dict(candidates=task["candidates"], truth=task["truth"], attributes=task["attributes"])
            )
            if key in worlds:
                assert worlds[key] == signature
            worlds[key] = signature
            tokens += len(inf["generated_token_ids"])
            cap_hits += not inf["ended_with_eos"] and len(inf["generated_token_ids"]) == 8
            if not record["correct"]:
                error_counts[(a["family"], a["kind"], a["known"][0], a["query"], record["answer"])] += 1
        assert len(batch_ids) == 1
    return dict(
        manifest_sha256=digest(m),
        complete=True,
        assigned=len(aa),
        original_cohorts=(len(aa) + 15) // 16,
        generated_tokens=tokens,
        cap_hits=cap_hits,
        independent_worlds=len(worlds),
        gates=evaluate(m, records),
        errors=[
            dict(family=k[0], kind=k[1], known=k[2], query=k[3], answer=k[4], count=v)
            for k, v in sorted(error_counts.items())
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run)
    write_once(a.out, result)
    print({k: v for k, v in result.items() if k not in ("gates", "errors")})
    print(
        {
            "passed": result["gates"]["passed"],
            "failed_cells": [c for c in result["gates"]["cells"] if not c["passed"]],
        }
    )
