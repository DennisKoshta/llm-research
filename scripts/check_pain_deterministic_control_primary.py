"""Independently count exact button strings and reproduce primary point gates.

No model, original parser, auditor or analyzer is imported. Bootstrap intervals
are outside this check; the fixed analyzer retains their separate provenance.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/qwen25-pain-deterministic-control-v1"
ANALYSIS = ROOT / "analyses/pain-deterministic-control-v1/analysis.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    manifest, analysis = read(RUN / "manifest.json"), read(ANALYSIS)
    assignments = {a["trial_id"]: a for a in manifest["original_manifest"]["plan"]["assignments"]}
    reported = {a["trial_id"]: a for a in analysis["outcomes"]}
    counts, rows, files = Counter(), {}, {}
    paths = sorted((RUN / "original_execution/phases/benchmark/calls").glob("*.json"))
    assert len(paths) == 65
    for path in paths:
        envelope = read(path)
        files[str(path.relative_to(RUN))] = sha(path)
        for record in envelope["engine_result"]["records"]:
            key = record["trial_id"]
            assert key in assignments and key not in rows
            assignment = assignments[key]
            assert record["target_name"] == assignment["target_name"]
            assert record["button_names"] == assignment["button_names"]
            # Every observed answer must satisfy this stricter whole-word rule.
            # No regex, substring acceptance or ambiguity tie-break is used.
            text = record["text"].removesuffix("<|im_end|>").strip().casefold()
            assert text in assignment["button_names"], (key, text)
            target = text == assignment["target_name"]
            assert record["ended_with_eos"] is True and record["cap_hit"] is False
            row = reported[key]
            assert row["status"] == "completed" and row["valid"] is True
            assert row["analysis_category"] == ("target" if target else "other")
            assert row["raw"] == record["text"] and row["cap_hit"] is False
            rows[key] = dict(arm=assignment["arm"], position=assignment["initial_position"],
                             target=target)
            counts[assignment["arm"], "assigned"] += 1
            counts[assignment["arm"], "target"] += target
    assert set(rows) == set(assignments) == set(reported) and len(rows) == 2020

    def rate(arm, position=None):
        selected = [r for r in rows.values() if r["arm"] == arm and
                    (position is None or r["position"] == position)]
        return Fraction(sum(r["target"] for r in selected), len(selected))

    arms = ["baseline", "pain_full", "fear_full", "pain_final_prefix", "fear_final_prefix"]
    for arm in arms:
        assert counts[arm, "assigned"] == 404
        point = next(r for r in analysis["rates"] if r["arm"] == arm and
                     r["content_group"] == "all" and r["initial_position"] == r["seed_base"] == "averaged")
        assert point["target"] == counts[arm, "target"]
        assert point["estimate"] == float(rate(arm)) and point["parseable"] == 404
    full = (rate("pain_full") >= Fraction(1, 2) and rate("baseline") <= Fraction(1, 20)
            and rate("fear_full") <= Fraction(3, 20))
    for comparator, pooled, positioned in (("baseline", Fraction(2, 5), Fraction(1, 4)),
                                          ("fear_full", Fraction(3, 10), Fraction(1, 5))):
        full &= rate("pain_full") - rate(comparator) >= pooled
        full &= all(rate("pain_full", p) - rate(comparator, p) >= positioned for p in (0, 1))
    schedule = (rate("pain_full") - rate("pain_final_prefix") >= Fraction(1, 4)
                and all(rate("pain_full", p) - rate("pain_final_prefix", p) >= Fraction(1, 10) for p in (0, 1)))
    assert full is True and schedule is True
    assert analysis["decision"]["full_mask_positive_control"]["classification"] == "qualified"
    assert analysis["decision"]["schedule_comparison"]["classification"] == "positive_coverage_difference"
    return dict(passed=True, assigned_rows=2020, case_insensitive_whole_button_matches=2020,
        completed_calls=65, malformed=0, caps=0,
        pooled={arm: dict(target=counts[arm, "target"], assigned=404, rate=float(rate(arm))) for arm in arms},
        full_mask_point_gates=bool(full), positive_coverage_point_gates=bool(schedule),
        invalid_coding_identical=True, bootstrap_independently_recomputed=False,
        source_files=files, analysis_byte_sha256=sha(ANALYSIS),
        audit_byte_sha256=sha(RUN / "phases/benchmark/audit.json"), checker_byte_sha256=sha(Path(__file__)))


if __name__ == "__main__":
    result = check()
    out = ROOT / "reports/pain-deterministic-control-v1-primary-check.json"
    with out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, indent=2))
