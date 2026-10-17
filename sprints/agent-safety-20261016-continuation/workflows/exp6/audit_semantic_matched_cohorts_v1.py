"""Audit actual paired name/replay execution cohorts without changing any labels."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = ROOT / "executions/semantic_matched_replay_v1"
    token_audit = json.loads((frozen / "token_length_audit.json").read_text())
    expected = {r["replay_id"]: r for r in token_audit["rows"]}
    cohorts = {g["cohort"]: g for g in token_audit["cohorts"]}
    views = {}
    source_hashes = {}
    for phase in ["semantic_names_v1", "semantic_matched_replay_v1"]:
        path = ROOT / "runs" / phase / "raw.jsonl"
        source_hashes[phase] = sha(path)
        rows = {}
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                assert row["id"] not in rows
                rows[row["id"]] = {k: row[k] for k in [
                    "id", "messages", "input_ids", "unpadded_input_ids",
                    "attention_mask", "position_ids", "batch", "sampling",
                    "enable_thinking", "generated_token_ids", "ended_with_eos",
                ]}
        assert len(rows) == 1248
        views[phase] = rows
    results = []
    for replay_id, pair in expected.items():
        replay = views["semantic_matched_replay_v1"][replay_id]
        renamed = views["semantic_names_v1"][pair["swap_id"]]
        group = cohorts[pair["cohort"]]
        for row, ids, key in [(replay, group["replay_ids"], "replay"),
                              (renamed, group["swap_ids"], "swap")]:
            assert row["batch"]["cohort_ids"] == ids
            assert row["batch"]["row"] == pair["row"]
            assert row["batch"]["size"] == 32
            assert len(row["input_ids"]) == group["padded_width"]
            assert len(row["unpadded_input_ids"]) == pair[key + "_tokens"]
            assert hashlib.sha256(json.dumps(row["unpadded_input_ids"]).encode()).hexdigest() == pair[key + "_tokens_sha256"]
            assert row["enable_thinking"] is False
        assert replay["attention_mask"] == renamed["attention_mask"]
        assert replay["position_ids"] == renamed["position_ids"]
        assert replay["sampling"] == renamed["sampling"]
        assert len(replay["messages"]) == len(renamed["messages"])
        for a, b in zip(replay["messages"], renamed["messages"]):
            assert a["role"] == b["role"]
            assert re.sub(r"\bPlan ([12])\b", lambda m: "Plan " + str(3-int(m.group(1))), a["content"]) == b["content"]
        results.append({"original_id": pair["original_id"], "replay_id": replay_id,
                        "swap_id": pair["swap_id"], "cohort": pair["cohort"],
                        "row": pair["row"], "padded_width": group["padded_width"],
                        "replay_batch_id": replay["batch"]["id"],
                        "swap_batch_id": renamed["batch"]["id"]})
    receipt = {
        "passed": True, "paired_rows": len(results), "cohorts": len(cohorts),
        "same_actual_row_memberships_and_padding": True,
        "exact_reviewed_token_streams": True, "same_masks_positions_sampling": True,
        "only_declared_Plan_name_substitution_in_messages": True,
        "raw_sha256": source_hashes, "rows": results,
        "limits": "Separate RPCs may use different workers and dispatch times. Matching layout does not assert universal bitwise numerical determinism. This audit does not alter behavioral labels, original archived primary, or predictor coefficients.",
    }
    destination = ROOT / "runs/semantic_matched_replay_v1/actual_paired_cohort_audit_v1.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "paired_rows": len(results), "cohorts": len(cohorts)}))


if __name__ == "__main__":
    main()
