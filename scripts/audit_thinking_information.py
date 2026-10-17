"""Reconcile complete reasoning outputs and their exact capture positions."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.thinking_information import (
    inference_request,
    manifest_generation_limit,
    record_from_inference,
    task_module,
)

TARGET_RELEVANCE_AUDIT_VERSION = 1


def audit_target_oracle(assignment, task, compatible):
    """Independently check target mutual information using joint/marginal counts."""
    target, query = task["target_attribute"], assignment["query"]
    n = len(compatible)
    rows = [task["candidates"][i] for i in compatible]
    reports = Counter(row[query] for row in rows)
    targets = Counter(row[target] for row in rows)
    joint = Counter((row[target], row[query]) for row in rows)

    def entropy(counts):
        return -sum(count / n * math.log2(count / n) for count in counts.values())

    target_entropy, report_entropy = entropy(targets), entropy(reports)
    target_gain = target_entropy + report_entropy - entropy(joint)
    world_after = sum(count / n * math.log2(count) for count in reports.values())
    assert assignment["known"] == [] and task["observations"] == {} and n == 4
    assert target == 2 and query in (0, 1)
    assert task["remaining_candidates_before"] == n
    assert (
        task["expected_remaining_candidates_after"]
        == sum(count * count / n for count in reports.values())
        == 2
    )
    assert task["world_entropy_before_bits"] == task["uncertainty_bits"] == 2
    assert task["expected_world_entropy_after_bits"] == world_after == 1
    assert task["expected_world_information_gain_bits"] == task["information_bits"] == 2 - world_after == 1
    assert task["offered_report_entropy_bits"] == report_entropy == 1
    assert task["report_accuracy"] == 1
    assert task["target_entropy_before_bits"] == target_entropy == 1
    assert (
        task["expected_target_information_gain_bits"]
        == target_gain
        == int(query != assignment["independent_report"])
    )
    assert task["expected_target_entropy_after_bits"] == target_entropy - target_gain
    assert task["independent_report_attribute"] == assignment["independent_report"]
    assert task["target_linked_report_attribute"] == 1 - assignment["independent_report"]
    assert task["queried_attribute"] == query
    return int(target_gain)


def target_smoke_coverage(cells):
    required = set(
        itertools.product(
            ("objects", "machines"), ("new", "repeat", "memory", "count", "instrument"), (0, 1), (0, 1)
        )
    )
    return set(cells) == required and all(cell["assigned"] == 1 for cell in cells.values())


def audit(root):
    m = read(root / "manifest.json")
    limit = manifest_generation_limit(m)
    rs = RunStore(root, m).completed()
    aa = m["diagnostics"]
    assert set(rs) == {a["trial_id"] for a in aa}, "Incomplete collection"
    physical = sum(len(read(p)["records"]) for p in (root / "cohorts").glob("*/*.json"))
    assert physical == len(aa)
    tokenizer = AutoTokenizer.from_pretrained(
        m["model"]["model_id"], revision=m["model"]["revision"], trust_remote_code=False
    )
    end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
    arrays = {}
    availability = defaultdict(Counter)
    tokens = cap_hits = 0
    multiple = []
    group_scores = defaultdict(Counter)
    target_cells = defaultdict(Counter)
    for offset in range(0, len(aa), m["execution"]["batch_size"]):
        cohort = aa[offset : offset + m["execution"]["batch_size"]]
        ids = [a["trial_id"] for a in cohort]
        batch_ids = set()
        for row, a in enumerate(cohort):
            r = rs[a["trial_id"]]
            inf = r["inference"]
            request = inference_request(a)
            assert r["assignment"] == a and inf["messages"] == request["messages"]
            assert inf["enable_thinking"] and inf["hooks_remaining"] == 0 and inf["intervention"] is None
            assert inf["sampling"] == dict(
                seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=limit
            )
            assert inf["batch"]["cohort_ids"] == ids and inf["batch"]["row"] == row
            batch_ids.add(inf["batch"]["id"])
            assert tokenizer.decode(inf["generated_token_ids"], skip_special_tokens=True) == inf["raw"]
            check = record_from_inference(a, inf)
            assert digest(r["task"]) == digest(task_module(a).task(a))
            t = r["task"]
            compatible = [
                i
                for i, candidate in enumerate(t["candidates"])
                if all(candidate[int(k)] == value for k, value in t["observations"].items())
            ]
            counts = Counter(t["candidates"][i][a["query"]] for i in compatible)
            entropy = math.log2(len(compatible))
            expected_after = sum(n / len(compatible) * math.log2(n) for n in counts.values())
            assert compatible == t["compatible"] and entropy == t["uncertainty_bits"]
            assert entropy - expected_after == t["information_bits"]
            target_gain = (
                audit_target_oracle(a, t, compatible)
                if a.get("task_revision") == "target-relevant-information-v1"
                else None
            )
            for field in ("answer", "parsed_answer", "expected", "valid", "correct", "thinking_closed"):
                assert r[field] == check[field]
            meta = r["activation_artifact"]
            if "max_new_tokens" in a:
                assert meta["max_new_tokens"] == r["generation_max_new_tokens"] == limit
                assert r["cap_hit"] == check["cap_hit"]
            relative = meta["path"]
            if relative not in arrays:
                file = root / relative
                assert hashlib.sha256(file.read_bytes()).hexdigest() == meta["sha256"]
                with np.load(file, allow_pickle=False) as z:
                    arrays[relative] = {k: z[k] for k in z.files}
            z = arrays[relative]
            ar = meta["row"]
            assert z["trial_ids"][ar] == a["trial_id"]
            assert list(z["layers"]) == meta["layers"] and list(z["positions"]) == meta["positions"]
            hidden = z["hidden"][ar]
            available = z["available"][ar]
            assert np.array_equal(np.isfinite(hidden).all(-1), available)
            assert np.isnan(hidden[~available]).all()
            assert np.array_equal(available.all(0), available.any(0))
            assert z["token_positions"][ar].tolist() == meta["token_positions"]
            assert available.tolist() == meta["available"]
            length = len(inf["unpadded_input_ids"])
            generated = inf["generated_token_ids"]
            assert 1 <= len(generated) <= limit, "Generated sequence exceeds its frozen allowance"
            assert inf["ended_with_eos"] or len(generated) == limit, "Stopped before EOS or frozen cap"
            end_indices = [i for i, t in enumerate(generated) if t == end_token]
            if "closure_counts" in z:
                assert int(z["closure_counts"][ar]) == r["closing_delimiter_count"] == len(end_indices)
                labels = list(request["labels"])
                assert [str(s) for s in z["decision_labels"][ar] if s] == labels
                for j, label in enumerate(labels):
                    assert (
                        z["decision_label_ids"][ar, j] == tokenizer.encode(label, add_special_tokens=False)[0]
                    )
                decision_present = bool(available[:, list(z["positions"]).index("decision")].all())
                for field in ("decision_label_logits", "decision_label_probabilities_temperature_1"):
                    if decision_present:
                        assert r[field] == {label: float(z[field][ar, j]) for j, label in enumerate(labels)}
                        assert np.isfinite(z[field][ar, : len(labels)]).all()
                    else:
                        assert r[field] is None and np.isnan(z[field][ar]).all()
                if decision_present:
                    p = z["decision_label_probabilities_temperature_1"][ar, : len(labels)]
                    assert (p >= 0).all() and p.sum() <= 1.00001
            if len(end_indices) > 1:
                multiple.append(a["trial_id"])
            expected = {
                "record": inf["position_ids"][inf["attention_mask"].index(1)] + 0,
                "prefill": length - 1,
            }
            # Reconstruct the record marker token directly from the rendered prompt offsets.
            text = inf["rendered_prompt"]
            marker_end = text.index("End of record.") + len("End of record.") - 1
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            expected["record"] = next(
                i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= marker_end < e
            )
            for count in m["capture_plan"]["steps"]:
                expected[f"token_{count}"] = length + count - 1 if len(generated) > count else -1
            expected["think_end"] = (
                length + end_indices[-1] if end_indices and end_indices[-1] < len(generated) - 1 else -1
            )
            expected["decision"] = -1
            if end_indices:
                for i in range(end_indices[-1] + 1, len(generated)):
                    if tokenizer.decode([generated[i]], skip_special_tokens=False).strip() or (
                        i == len(generated) - 1 and inf["ended_with_eos"]
                    ):
                        expected["decision"] = length + i - 1
                        break
            for pi, name in enumerate(z["positions"]):
                assert z["token_positions"][ar, pi] == expected[name], (
                    a["trial_id"],
                    name,
                    z["token_positions"][ar, pi],
                    expected[name],
                )
                assert bool(available[:, pi].all()) == (expected[name] >= 0)
                key = (a["family"], a["kind"], int(r["task"]["information_bits"]), name, target_gain)
                availability[key]["assigned"] += 1
                availability[key]["available"] += bool(available[:, pi].all())
            tokens += len(generated)
            cap_hits += not inf["ended_with_eos"] and len(generated) == limit
            group_scores[(a["family"], a["kind"])]["correct"] += r["correct"] is True
            group_scores[(a["family"], a["kind"])]["n"] += 1
            if target_gain is not None:
                cell = target_cells[(a["family"], a["kind"], a["independent_report"], a["query"])]
                cell["assigned"] += 1
                cell["valid"] += r["valid"] is True
                cell["correct"] += r["correct"] is True
                cell["cap_hits"] += not inf["ended_with_eos"] and len(generated) == limit
        assert len(batch_ids) == 1
    n = len(aa)
    valid = sum(r["valid"] for r in rs.values())
    optional = all(a.get("task_revision") == "optional-information-v1" for a in aa)
    correct = None if optional else sum(r["correct"] is True for r in rs.values())
    if optional:
        for group in group_scores.values():
            group.pop("correct", None)
    result = dict(
        manifest_sha256=digest(m),
        phase=m["phase"],
        inference_scope=m.get("inference_scope"),
        validation_criteria=m.get("validation_criteria"),
        source_sha256=m.get("source_sha256"),
        complete=True,
        assigned=n,
        max_new_tokens=limit,
        correct=correct,
        valid=valid,
        generated_tokens=tokens,
        cap_hits=cap_hits,
        multiple_closure_trials=multiple,
        capture_availability=[
            dict(
                family=k[0],
                kind=k[1],
                information_bits=k[2],
                position=k[3],
                **({"expected_target_information_gain_bits": k[4]} if k[4] is not None else {}),
                **v,
            )
            for k, v in sorted(availability.items())
        ],
        family_kind_scores=[dict(family=k[0], kind=k[1], **v) for k, v in sorted(group_scores.items())],
    )
    if target_cells:
        result.update(
            target_relevance_oracle_audit_version=TARGET_RELEVANCE_AUDIT_VERSION,
            information_quantities=dict(
                information_bits="full-world expected information gain; constant one bit",
                expected_target_information_gain_bits="designated-target expected information gain; zero or one bit",
            ),
            target_relevance_cell_scores=[
                dict(
                    family=k[0],
                    kind=k[1],
                    independent_report=k[2],
                    query=k[3],
                    expected_target_information_gain_bits=int(k[2] != k[3]),
                    **counts,
                )
                for k, counts in sorted(target_cells.items())
            ],
        )
    if optional:
        result["feasibility"] = task_module(aa[0]).evaluate(m, rs)
        result["feasibility_passed"] = result["feasibility"]["passed"]
    elif m["phase"] == "development_smoke":
        result["feasibility_passed"] = (
            n == 40
            and valid >= 38
            and correct >= 36
            and all(c["correct"] >= 3 and c["n"] == 4 for c in group_scores.values())
            and not multiple
        )
        if target_cells:
            result["target_relevance_smoke_cell_coverage"] = target_smoke_coverage(target_cells)
            result["feasibility_passed"] &= result["target_relevance_smoke_cell_coverage"]
    elif m["phase"] in ("validation", "development_validation"):
        result["gates"] = task_module(aa[0]).evaluate(m, rs)
    return result


def audit_parent(root):
    m = read(root / "manifest.json")
    limit = manifest_generation_limit(m)
    children = read(root / "children.json")
    records, reports = {}, []
    for child in children:
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        assert digest(cm) == child["manifest_sha256"]
        assert cm["parent_manifest_sha256"] == digest(m)
        assert cm["source_sha256"] == m["source_sha256"]
        assert manifest_generation_limit(cm) == limit
        expected = [
            a for index in child["parent_cohorts"] for a in m["diagnostics"][index * 32 : (index + 1) * 32]
        ]
        assert cm["diagnostics"] == expected
        result = audit(path)
        reports.append(dict(run=path.name, **result))
        rs = RunStore(path, cm).completed()
        assert not records.keys() & rs.keys()
        records.update(rs)
    assert set(records) == {a["trial_id"] for a in m["diagnostics"]}
    return dict(
        manifest_sha256=digest(m),
        phase=m["phase"],
        inference_scope=m.get("inference_scope"),
        validation_criteria=m.get("validation_criteria"),
        source_sha256=m["source_sha256"],
        assigned=len(m["diagnostics"]),
        max_new_tokens=limit,
        complete=True,
        correct=(
            None
            if all(a.get("task_revision") == "optional-information-v1" for a in m["diagnostics"])
            else sum(r["correct"] is True for r in records.values())
        ),
        valid=sum(r["valid"] for r in records.values()),
        generated_tokens=sum(r["generated_tokens"] for r in reports),
        cap_hits=sum(r["cap_hits"] for r in reports),
        multiple_closure_trials=[t for r in reports for t in r["multiple_closure_trials"]],
        gates=task_module(m["diagnostics"][0]).evaluate(m, records),
        partitions=reports,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = audit_parent(a.run) if (a.run / "children.json").exists() else audit(a.run)
    write_once(a.out, result)
    print({k: v for k, v in result.items() if k not in ("capture_availability", "partitions")})
