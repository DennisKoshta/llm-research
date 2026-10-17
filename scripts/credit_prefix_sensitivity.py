"""Post-hoc truncation sensitivity of saved trajectories; never replaces run scores."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.credit import CreditTrial, diagnostic_prompt
from curiosity_study.identification import final_answer


def analyze(root, cap=768):
    manifest = read(root / "manifest.json")
    assert manifest["assay"] == "one_credit" and manifest["phase"] == "validation"
    assert manifest["sampling"]["max_new_tokens"] > cap > 0
    records = RunStore(root, manifest).completed()
    assignments = {a["trial_id"]: a for a in manifest["diagnostics"]}
    assert set(records) == set(assignments)
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["model_id"], revision=manifest["model"]["revision"], trust_remote_code=False
    )
    groups = defaultdict(Counter)
    for tid, a in assignments.items():
        record = records[tid]
        assert record["assignment"] == a and a["inference_mode"] == "think_sampled"
        inference = record["inference"]
        ids = inference["generated_token_ids"]
        assert tokenizer.decode(ids, skip_special_tokens=True) == inference["raw"]
        _, labels, expected = diagnostic_prompt(CreditTrial(a), a)
        short_raw = tokenizer.decode(ids[:cap], skip_special_tokens=True)
        answer, closed = final_answer(short_raw, True)
        short_valid = closed and answer in set(labels)
        short_correct = closed and answer == expected
        row = groups[(a["family"], a["kind"])]
        row["assigned"] += 1
        row["longer_than_prefix_cap"] += len(ids) > cap
        row["prefix_valid"] += short_valid
        row["prefix_correct"] += short_correct
        row["full_valid"] += record["valid"]
        row["full_correct"] += record["correct"]
        row["prefix_invalid_full_valid"] += not short_valid and record["valid"]
        row["prefix_valid_full_invalid"] += short_valid and not record["valid"]
        row["prefix_wrong_full_correct"] += not short_correct and record["correct"]
        row["prefix_correct_full_wrong"] += short_correct and not record["correct"]
    return {
        "scope": "post_hoc_saved_trajectory_prefix_sensitivity_no_rescoring",
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
        "prefix_token_cap": cap,
        "full_cap": manifest["sampling"]["max_new_tokens"],
        "full_decoding_reproduced_for_every_record": True,
        "rows": [{"family": f, "kind": k, **dict(c)} for (f, k), c in sorted(groups.items())],
        "assumption": "The frozen custom decoder does not condition its prompt, logits, or sampling rule on the maximum generation length; the limit only stops its loop. This analysis truncates the exact recorded token prefix, including the original EOS stopping if reached earlier.",
        "limits": "Not an additional model sample or a new advancement decision. These are new-run trajectories, not continuations of the earlier failed run. No old score or missing answer is repaired. Both improvement and deterioration after a longer continuation are counted.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, analyze(args.run))
