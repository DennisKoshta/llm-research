"""Prospective descriptive output scores for the separately frozen set-structure study."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
EX = ROOT / "executions/structural_components_v1"
COMPONENTS = ["physical_operation_logodds", "common_A_logodds"]
CONTRASTS = [("payoff_minus_narrative", "payoff_table", "narrative"),
             ("colocated_minus_payoff", "permission_colocated", "payoff_table")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(phase):
    manifest = json.loads((EX / "manifest.json").read_text())
    assert all(sha(EX / f) == digest for f, digest in manifest["hashes"].items())
    review = ROOT.parent / "exp1/independent_exp6_structural_components_review.json"
    assert review.exists() and json.loads(review.read_text())["passed"]
    spec = importlib.util.spec_from_file_location("frozen_structural_components", EX / "component_helper.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = ROOT
    rows, numeric = helper.extract(phase)
    expected_worlds = 8 if phase == "reliability_structural_pilot_v1" else 40
    assignments = json.loads((ROOT / "runs" / phase / "assignments.json").read_text())
    requests = json.loads((ROOT / "runs" / phase / "requests.json").read_text())
    assert len(assignments) == len(requests) == 24 * expected_worlds
    assert all(q.get("enable_thinking", False) is False for q in requests)
    assert all(r["family"] == "reliability_testing" for r in rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["world_id"], row["arm"], row["authorized"], row["utility"]].append(row)
    cells = {key: np.array([sum(r[c] for r in values)/len(values) for c in COMPONENTS])
             for key, values in grouped.items()}
    worlds = sorted({r["world_id"] for r in rows})
    assert len(worlds) == expected_worlds and len(rows) == 12 * expected_worlds
    rng = np.random.default_rng(261016709)
    draw = rng.integers(0, len(worlds), (20000, len(worlds)))
    results = {}
    world_values = []
    for authorized, utility in itertools.product([False, True], ["better", "worse"]):
        contrasts = {}
        for name, treatment, control in CONTRASTS:
            differences = np.array([cells[w, treatment, authorized, utility] - cells[w, control, authorized, utility]
                                    for w in worlds])
            sampled = differences[draw].mean(axis=1)
            contrasts[name] = {c: {"mean": float(differences[:, i].mean()),
                                   "descriptive95_world_bootstrap": np.quantile(sampled[:, i], [.025, .975]).tolist()}
                               for i, c in enumerate(COMPONENTS)}
            for index, world in enumerate(worlds):
                world_values.append({"world_id": world, "authorized": authorized, "utility": utility,
                                     "contrast": name, **dict(zip(COMPONENTS, differences[index].tolist()))})
        results[f"authorized{int(authorized)}|{utility}"] = contrasts
    result = {"phase": phase, "worlds": len(worlds), "status": "Prespecified descriptive secondary, separate from fixed behavioral endpoints",
              "numeric_audit": numeric, "factorial": results, "world_contrasts": world_values,
              "all_paired_values": rows, "independent_formula_review_sha256": sha(review),
              "weighting": "Both physical-action A/B mappings form each pair; all recorded display orders average within world/arm before paired contrasts. The structural protocol has one counterbalanced order per world, not four repeats per world.",
              "limits": ["Pilot and main are reported separately; no pooling, refit, effect gate or extra primary.",
                         "20,000 whole-world bootstrap replicates, seed261016709; 95% intervals are descriptive and not multiplicity adjusted.",
                         "Probability scores are from nonthinking action-token outputs only. Invalid decoded responses remain assigned in the probability analysis; missing pairs or nonfinite scores stop analysis.",
                         "Exact output-score algebra, not identified neural factors, motives or a causal permission mechanism.",
                         "Variation remains within one set-union algorithm. Full membership signatures are unique, but reduced score/outside-core equivalence classes repeat as disclosed.",
                         "Unbounded log odds receive no bounded-rate interval. Original behavioral gates and the failed early instrument remain unchanged."]}
    dump(ROOT / "runs" / phase / "structural_components_v1.json", result)
    print(json.dumps({"phase": phase, "worlds": len(worlds), "numeric_audit": numeric, "factorial": results}, indent=2))


def freeze():
    assert not (EX / "manifest.json").exists()
    EX.mkdir(parents=True)
    for name in ["structural_components_v1.py", "structural_components_v1_protocol.json"]:
        shutil.copyfile(ROOT / name, EX / name)
    shutil.copyfile(ROOT / "executions/full_factorial_code_components_v2/full_factorial_code_components_v2.py", EX / "component_helper.py")
    dump(EX / "manifest.json", {"frozen_at": time.time(), "hashes": {p.name: sha(p) for p in EX.iterdir()},
                               "original_structural_manifest_sha256": sha(ROOT / "executions/reliability_structures_v1/manifest.json"),
                               "structural_outcomes_exist": any((ROOT / "runs" / p / "raw.jsonl").exists() for p in ["reliability_structural_pilot_v1", "reliability_structural_main_v1"]),
                               "all_original_protocols_gates_and_source_predictors_unchanged": True})
    print(sha(EX / "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("phase", nargs="?", choices=["reliability_structural_pilot_v1", "reliability_structural_main_v1"])
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        assert args.phase
        run(args.phase)
