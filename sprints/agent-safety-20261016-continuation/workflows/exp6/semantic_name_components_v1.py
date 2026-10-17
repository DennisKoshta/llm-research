"""Prespecified, CPU-only output-score contrasts for semantic name controls."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
EX = ROOT / "executions/semantic_name_components_v1"
ARMS = ["narrative", "payoff_table", "permission_colocated", "permission_separated",
        "payoff_counterfactual", "fact_table", "outcome_prose"]
COMPONENTS = ["physical_operation_logodds", "common_A_logodds"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def extract(phase, selected):
    path = ROOT / "runs" / phase
    score_file = path / ("scored_strict_v1.json" if phase == "semantic_names_v1" else "scored_strict_v2.json")
    metadata = {r["id"]: r for r in json.loads(score_file.read_text()) if r["id"] in selected}
    assert set(metadata) == selected
    pairs = defaultdict(dict)
    seen = set()
    clips = invalid = ties = 0
    norm_error = 0
    sign_mismatches = []
    with (path / "raw.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] not in selected:
                continue
            assert row["id"] not in seen
            seen.add(row["id"])
            assert row["enable_thinking"] is False, "Native prethinking logits do not qualify"
            m = metadata[row["id"]]
            probabilities = row["conditional_label_probabilities"]
            a, b = probabilities["A"], probabilities["B"]
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in (a, b))
            norm_error = max(norm_error, abs(a+b-1))
            qa, qb = max(a, 1e-12), max(b, 1e-12)
            clips += (qa != a) + (qb != b)
            z = math.log(qa/qb)
            ties += z == 0
            invalid += not m["valid"]
            if z != 0 and m["valid"] and ("A" if z > 0 else "B") != m["action"]:
                sign_mismatches.append(row["id"])
            key = (m["world_id"], m["arm"], m["authorized"], m["utility"], m["display_order"])
            label = m["designated_operation_label"]
            assert label not in pairs[key]
            pairs[key][label] = (z, m)
    assert seen == selected
    records = {}
    for key, pair in pairs.items():
        assert set(pair) == {"A", "B"}, "Missing paired mappings fail this analysis; no pair deletion"
        za, ma = pair["A"]
        zb, mb = pair["B"]
        assert ma["family"] == mb["family"]
        physical, common = (za-zb)/2, (za+zb)/2
        assert abs(physical+common-za) < 1e-10 and abs(common-physical-zb) < 1e-10
        records[key] = {"world_id": key[0], "family": ma["family"], "arm": key[1],
                        "authorized": key[2], "utility": key[3], "display_order": key[4],
                        "physical_operation_logodds": physical, "common_A_logodds": common,
                        "valid_both": ma["valid"] and mb["valid"]}
    return records, {"phase": phase, "assigned": len(selected), "paired_mappings": len(pairs),
                     "invalid_outputs_retained": invalid, "clipped_probability_entries": int(clips),
                     "max_probability_sum_error": norm_error, "zero_logodds_ties": ties,
                     "greedy_sign_mismatches": sign_mismatches, "score_sha256": sha(score_file)}


def run():
    manifest = json.loads((EX / "manifest.json").read_text())
    assert all(sha(EX / f) == digest for f, digest in manifest["hashes"].items())
    replay_meta = json.loads((ROOT / "executions/semantic_matched_replay_v1/assignments_unscored.json").read_text())
    archive = {}
    audits = []
    for phase in ["source_v1", "target_v1"]:
        selected = {r["original_request_id"] for r in replay_meta.values() if r["baseline_phase"] == phase}
        records, audit = extract(phase, selected)
        assert not set(archive) & set(records)
        archive.update(records)
        audits.append(audit)
    replay, audit = extract("semantic_matched_replay_v1", set(replay_meta))
    audits.append(audit)
    names, audit = extract("semantic_names_v1", {r["name_request_id"] for r in replay_meta.values()})
    audits.append(audit)
    assert set(archive) == set(replay) == set(names) and len(archive) == 624
    cohorts = {"archived": archive, "matched": replay, "names": names}
    cells = {}
    for phase, records in cohorts.items():
        grouped = defaultdict(list)
        for key, row in records.items():
            grouped[key[:4]].append(row)
        cells[phase] = {key: np.array([sum(r[c] for r in rows)/len(rows) for c in COMPONENTS])
                        for key, rows in grouped.items()}
    world_family = {r["world_id"]: r["family"] for r in archive.values()}
    groups = {"all_families": sorted(world_family), **{
        family: sorted(w for w, f in world_family.items() if f == family)
        for family in sorted(set(world_family.values()))}}
    summaries = {}
    rng = np.random.default_rng(261016699)
    contrasts = [("names", "matched"), ("matched", "archived"), ("names", "archived")]
    for group, worlds in groups.items():
        n = len(worlds)
        index = {w: i for i, w in enumerate(worlds)}
        columns = []
        for family in sorted({world_family[w] for w in worlds}):
            locations = np.array([index[w] for w in worlds if world_family[w] == family])
            columns.append(locations[rng.integers(0, len(locations), (20000, len(locations)))] )
        bootstrap = np.concatenate(columns, axis=1)
        factorial = {}
        for authorized, utility in itertools.product([False, True], ["better", "worse"]):
            arm_results = {}
            for arm in ARMS + ["equal_seven_arm_average"]:
                selected_arms = ARMS if arm == "equal_seven_arm_average" else [arm]
                values = {phase: np.array([sum((cells[phase][w, a, authorized, utility]
                                                for a in selected_arms), np.zeros(2))/len(selected_arms)
                                          for w in worlds]) for phase in cohorts}
                comparisons = {}
                for treatment, control in contrasts:
                    difference = values[treatment] - values[control]
                    sampled = difference[bootstrap].mean(axis=1)
                    comparisons[treatment + "_minus_" + control] = {
                        c: {"mean": float(difference[:, i].mean()),
                            "descriptive95_world_bootstrap": np.quantile(sampled[:, i], [.025, .975]).tolist()}
                        for i, c in enumerate(COMPONENTS)}
                arm_results[arm] = {"worlds": n, "contrasts": comparisons,
                                    "cohort_means": {phase: dict(zip(COMPONENTS, values[phase].mean(axis=0).tolist())) for phase in cohorts}}
            factorial[f"authorized{int(authorized)}|{utility}"] = arm_results
        summaries[group] = {"worlds": n, "factorial": factorial}
    result = {"status": "Prespecified named CPU-only secondary; no new primary or fit",
              "definitions": {"zA": "log P(A)/P(B) when the designated physical action uses A",
                              "zB": "log P(A)/P(B) when that action uses B",
                              "physical_operation_logodds": "(zA-zB)/2", "common_A_logodds": "(zA+zB)/2"},
              "numeric_audits": audits, "summaries": summaries,
              "paired_values": {phase: list(records.values()) for phase, records in cohorts.items()},
              "limits": ["Only nonthinking action-token probabilities qualify; never native prethinking logits.",
                         "All assigned pairs retained, including invalid outputs; missing pairs/probabilities stop analysis rather than being dropped.",
                         "Display orders average within world and arm, then the optional aggregate weights seven arms equally. This differs from Exp2's frozen equal-104-row weighting.",
                         "20,000 family-stratified whole-world bootstrap replicates; 95% intervals are descriptive and unadjusted across all cells and contrasts.",
                         "Twelve exposed lexical worlds with known numerical and relational duplication; intervals do not quantify generalization across unseen task algorithms.",
                         "These components are output-score algebra, not identified neural factors or a uniquely isolated letter bias. Unbounded log odds receive no bounded-rate interval.",
                         "Actual paired-cohort audit is reported separately; separate worker dispatch may retain numerical variation."]}
    dump(ROOT / "runs/semantic_matched_replay_v1/semantic_name_components_v1.json", result)
    print(json.dumps({"numeric_audits": audits, "groups": list(summaries)}))


def freeze():
    assert not (EX / "manifest.json").exists()
    EX.mkdir(parents=True)
    for name in ["semantic_name_components_v1.py", "semantic_name_components_v1_protocol.json"]:
        shutil.copyfile(ROOT / name, EX / name)
    dump(EX / "manifest.json", {"frozen_at": time.time(),
                               "hashes": {p.name: sha(p) for p in EX.iterdir()},
                               "name_outputs_exist": (ROOT / "runs/semantic_names_v1/raw.jsonl").exists(),
                               "replay_outputs_exist": (ROOT / "runs/semantic_matched_replay_v1/raw.jsonl").exists(),
                               "original_behavioral_and_probe_endpoints_unchanged": True})
    print(json.dumps({"manifest_sha256": sha(EX / "manifest.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
