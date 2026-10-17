"""Additional actual-mode and probability validation; no label/formula changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(phase):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137", local_files_only=True)
    action_tokens = {a: tokenizer.encode(a, add_special_tokens=False) for a in ["A", "B"]}
    assert all(len(ids) == 1 for ids in action_tokens.values())
    path = ROOT / "runs" / phase
    scores = {r["id"]: r for r in json.loads((path / "scored_strict_v2.json").read_text())}
    seen = set()
    normalization = 0
    valid_action_starts = 0
    invalid_outputs = 0
    for line in (path / "raw.jsonl").open():
        row = json.loads(line)
        assert row["id"] not in seen
        seen.add(row["id"])
        assert row["enable_thinking"] is False, "Returned native mode is ineligible"
        probabilities = row["conditional_label_probabilities"]
        a, b = probabilities["A"], probabilities["B"]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in [a, b])
        error = abs(a+b-1)
        assert error <= 1e-6, "Conditional A/B distribution is not normalized"
        normalization = max(normalization, error)
        score = scores[row["id"]]
        if score["valid"]:
            assert row["generated_token_ids"][0] == action_tokens[score["action"]][0]
            valid_action_starts += 1
        else:
            invalid_outputs += 1
    assert seen == set(scores)
    result = {"passed": True, "phase": phase, "assigned": len(seen),
              "actual_returned_nonthinking_rows": len(seen),
              "max_conditional_probability_sum_error": normalization,
              "normalization_tolerance": 1e-6, "valid_first_action_tokens": valid_action_starts,
              "invalid_decoded_rows_retained": invalid_outputs, "pinned_action_token_ids": action_tokens,
              "score_sha256": hashlib.sha256((path / "scored_strict_v2.json").read_bytes()).hexdigest(),
              "scope": "Named qualification requested in independent review; original frozen secondary, primary, labels and probability formulas unchanged. Invalid choices remain assigned but are not compliant labels."}
    (path / "structural_raw_score_qualification_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["reliability_structural_pilot_v1", "reliability_structural_main_v1"])
    run(parser.parse_args().phase)
