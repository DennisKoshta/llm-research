"""Independent CPU-only preflight of the fixed competence extension."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys

from scipy.optimize import brentq
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E = ROOT / "executions/competence_extension_v1"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())


def main():
    checks = {}
    manifest = read(E / "manifest.json")
    for filename, digest in manifest["files"].items():
        assert sha(E / filename) == digest
    assert sha(ROOT / "shared/competence_extension_v1.py") == manifest["files"]["competence_extension_v1.py"]
    checks["frozen_and_executed_source_hash_identity"] = True
    protocol = read(E / "protocol.json")
    queries = read(E / "requests.json")
    assert len(queries) == 192 and len({q["id"] for q in queries}) == 192
    previous = []
    for version in ["instrument_v1", "instrument_v2"]:
        panel = read(ROOT / "executions" / version / "check_requests.json")
        for split in ["val_competence", "test_competence"]:
            previous.extend(panel[split])
    prior_pairs = {tuple(map(int, re.search(r"Calculate (\d+) \+ (\d+)\.", q["messages"][0]["content"]).groups())) for q in previous}
    pairs = []
    expected_labels = {"A": 0, "B": 0, "C": 0}
    for q in queries:
        text = q["messages"][0]["content"]
        a, b = map(int, re.search(r"^Calculate (\d+) \+ (\d+)\.", text).groups())
        options = {label: int(value) for label, value in re.findall(r"^([ABC]): (\d+)$", text, re.M)}
        correct = [label for label, value in options.items() if value == a + b]
        assert len(correct) == 1 and correct[0] == q["metadata"]["expected"]
        assert q["metadata"]["a"] == a and q["metadata"]["b"] == b
        assert q["metadata"]["options"] == [options[label] for label in "ABC"]
        assert 11 <= a <= 89 and 11 <= b <= 89
        assert sorted(value - (a + b) for value in options.values())[0] in range(-9, 0)
        assert sorted(value - (a + b) for value in options.values())[-1] in range(1, 10)
        assert q["greedy"] is True and q["enable_thinking"] is False and q["labels"] == "ABC"
        assert q["max_new_tokens"] == 16 and q["capture_layers"] == []
        assert q["seed"] == 2610168800
        pairs.append((a, b)); expected_labels[correct[0]] += 1
    assert len(set(pairs)) == 192 and not (set(pairs) & prior_pairs)
    assert len(prior_pairs) == protocol["excluded_prior_ordered_pairs"] == 96
    checks["all192_actual_prompt_arithmetic_oracles_and_settings"] = True
    checks["all_prior_ordered_pairs_excluded_and_fresh_unique"] = True

    rng = random.Random(2610168801)
    generated = []
    seen = set(prior_pairs)
    while len(generated) < 192:
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        options = [a+b, a+b+rng.randint(1, 9), a+b-rng.randint(1, 9)]
        rng.shuffle(options)
        generated.append((a, b, options))
    assert generated == [(q["metadata"]["a"], q["metadata"]["b"], q["metadata"]["options"]) for q in queries]
    checks["exact_frozen_rng_panel_reproduced"] = True

    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("competence_extension_reviewed", E / "competence_extension_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = []
    for n in [1, 24, 192]:
        for gain, loss in [(0, 0), (n, 0), (0, n), (1, 0), (0, 1), (n//4, n//3)]:
            if gain + loss > n:
                continue
            lo = 0.0 if gain == 0 else brentq(lambda p: binom.sf(gain-1, n, p)-.025, 0, 1, xtol=1e-14)
            hi = 1.0 if loss == n else brentq(lambda p: binom.cdf(loss, n, p)-.025, 0, 1, xtol=1e-14)
            result = module.paired_lower(gain, loss, n)
            assert abs(result["gain_lower97_5"] - lo) < 1e-10
            assert abs(result["loss_upper97_5"] - hi) < 1e-10
            assert abs(result["difference_lower95"] - (lo-hi)) < 1e-10
            fixtures.append({"n": n, "improved": gain, "regressed": loss, "lower": lo-hi})
    assert module.paired_lower(0, 0, 192)["difference_lower95"] < 0
    checks["CP_bounds_independently_inverted_binomial_tails_including_edges"] = True

    gates = protocol["gates"]
    assert gates["fresh_accuracy_each_arm_min"] == gates["pooled216_accuracy_each_arm_min"] == .9
    assert gates["fresh_ablation_accuracy_drop_max"] == gates["pooled_ablation_accuracy_drop_max"] == .1
    assert gates["fresh_paired_lower95_strictly_above"] == -.1
    assert protocol["one_look"] and protocol["no_replacement_or_further_extension"]
    assert protocol["intervention_lock_sha256"] == sha(ROOT / "records/intervention-lock.json")
    source = (E / "competence_extension_v1.py").read_text()
    assert "[k for k,v in failure['checks'].items() if not v]==['qa_accuracy_each_arm']" in source
    assert "assert admission['approved'] is True" in source
    assert "checks['fresh_paired_noninferiority']=bound['difference_lower95']>-.10" in source
    assert "immutable(R/'analysis.json',result)" in source
    assert "atomic(ROOT/'records/intervention-qualified.json'" not in source
    checks["sole_failure_root_admission_fixed_lock_fresh_AND_pooled_guards"] = True
    checks["no_automatic_ABC_qualification"] = True
    # Reversed operands are the same arithmetic sum but a different ordered prompt.
    old_unordered = {tuple(sorted(pair)) for pair in prior_pairs}
    new_unordered = [tuple(sorted(pair)) for pair in pairs]
    receipt = {
        "passed": all(checks.values()), "checks": checks,
        "assigned_requests_per_arm": 192, "total_calls": 768,
        "expected_label_counts": expected_labels,
        "manifest_sha256": sha(E / "manifest.json"),
        "intervention_lock_sha256": protocol["intervention_lock_sha256"],
        "interval_fixtures": fixtures,
        "freshness_scope": {"ordered_unique": 192, "unordered_unique": len(set(new_unordered)), "unordered_instances_overlapping_prior": sum(pair in old_unordered for pair in new_unordered), "interpretation": "The frozen exclusion is ordered operands, not commutativity-equivalent arithmetic problems. Keep this exact scope; no post-freeze substitutions recommended."},
        "required_actual_result_review": ["All returned lock/spec identities and exact prompts", "All four arms' actual 32-row cohort membership, order, input IDs, masks and positions", "Strict output grammar, actual EOS, cap and expected answer", "Fresh and original counts, all gates and paired bound independently recomputed"],
        "scope": "Prospective one-shot competence qualification of the unchanged selected vector. Original v2 failure remains. No competence-extension or A/B/C outcomes read; no model calls submitted."
    }
    out = HERE / "independent_competence_extension_preflight_v1.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
