"""Synthetic checks of the independent arithmetic/artifact checker."""

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load("independent_voluntary_checker", "scripts/check_voluntary_analysis_independently.py")
fixtures = load("independent_stat_fixtures", "tests/test_voluntary_analysis_statistics.py")
analyzer = load("independent_comparison_analyzer", "scripts/analyze_voluntary_resolution.py")
source_fixtures = load("independent_source_fixtures", "tests/test_audit_voluntary_resolution.py")


def test_dense_independent_estimator_matches_all_synthetic_primary_outputs():
    choices, projections = fixtures.synthetic_tables()
    independent = checker.statistics(choices, projections)
    reported = analyzer.analyze_tables(choices, projections)
    comparisons = checker.Comparisons()
    for name, values in independent.items():
        comparisons.equal(reported[name], values, name)
    assert comparisons.values > 1700 and comparisons.maximum_absolute_error < 1e-10
    assert len(independent["behavior_contrasts"]) == 42
    assert len(independent["projection_contrasts"]) == 192
    assert len(independent["denominators"]) == 27


def test_missing_projection_retains_assigned_worlds_in_independent_estimator():
    choices, projections = fixtures.synthetic_tables()
    target = projections[0]
    target.update(raw=None, scaled=None)
    result = checker.statistics(choices, projections)
    selected = [
        row
        for row in result["projection_contrasts"]
        if row["axis"] == target["axis"]
        and row["position"] == target["position"]
        and row["scale"] == "raw"
        and row["family"] in ("objects", "pooled")
    ]
    assert all(row["estimate"] is None and row["ci95"] is None for row in selected)
    assert {row["assigned_worlds"] for row in selected} == {12, 24}
    assert {row["available_worlds"] for row in selected} == {11, 23}


@pytest.mark.parametrize("field", ["estimate", "ci95", "assigned_worlds"])
def test_checker_rejects_changed_estimates_intervals_and_denominators(field):
    choices, projections = fixtures.synthetic_tables()
    actual = checker.statistics(choices, projections)
    corrupted = deepcopy(actual)
    row = corrupted["behavior_contrasts"][0]
    if field == "ci95":
        row[field][0] += 0.01
    else:
        row[field] += 1
    with pytest.raises(ValueError):
        checker.Comparisons().equal(corrupted, actual)


@pytest.mark.parametrize(
    "raw,parsed,closed,valid",
    [
        ("Unclosed reasoning A", "", False, False),
        ("</think> A because I like it", "A because I like it", True, False),
        ("</think> A.", "A", True, True),
        ("</think> A </think> B", "B", True, True),
    ],
)
def test_raw_choice_parser_and_fixed_record_axes_recomputed_without_analyzer(raw, parsed, closed, valid):
    assignment, record, arrays, _ = source_fixtures.source()
    record["inference"]["raw"] = raw
    record.update(
        valid=valid,
        thinking_closed=closed,
        parsed_answer=parsed,
        selected_service=record["action_mapping"].get(parsed) if valid else None,
    )
    metadata = record["activation_artifact"]
    cache = {metadata["path"]: (metadata["sha256"], arrays)}
    axes = fixtures.synthetic_axes()
    choices, projections, bindings = checker.reconstruct_choices(
        dict(diagnostics=[assignment]), {assignment["trial_id"]: record}, cache, axes, checker.Comparisons()
    )
    assert len(choices) == len(bindings) == 1 and len(projections) == 16
    assert choices[0]["category"] == (record["selected_service"] if valid else "invalid")
    assert all(row["raw"] == 0.0 and row["scaled"] == 0.0 for row in projections)
    # The final-position axes point elsewhere; both record and prefill use A record.
    axes["unit"][:, 0, 1] *= -1
    again = checker.reconstruct_choices(
        dict(diagnostics=[assignment]), {assignment["trial_id"]: record}, cache, axes, checker.Comparisons()
    )[1]
    assert again == projections


def test_csv_numeric_hash_independent_comparison_and_explicit_null_interval(tmp_path):
    expected = [
        dict(estimate=None, ci95=None, assigned_worlds=12, label="example"),
        dict(estimate=0.25, ci95=[0.1, 0.4], assigned_worlds=12, label="another"),
    ]
    path = tmp_path / "contrasts.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expected[0]))
        writer.writeheader()
        writer.writerows([{**row, "ci95": json.dumps(row["ci95"])} for row in expected])
    checker.compare_csv(path, expected, checker.Comparisons())
    path.write_text(path.read_text().replace("0.25", "0.75"))
    with pytest.raises(ValueError, match="Numeric mismatch"):
        checker.compare_csv(path, expected, checker.Comparisons())


def test_array_byte_hash_and_safe_location_are_checked(tmp_path):
    path = tmp_path / "activations/a/0000000.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, hidden=np.ones((1, 2)))
    metadata = dict(path=str(path.relative_to(tmp_path)), sha256=checker.sha(path))
    cache = {}
    checker.load_array(tmp_path, metadata, cache)
    with pytest.raises(ValueError, match="Conflicting hash"):
        checker.load_array(tmp_path, {**metadata, "sha256": "0" * 64}, cache)
    with pytest.raises(ValueError, match="Unsafe"):
        checker.load_array(tmp_path, {**metadata, "path": "../elsewhere.npz"}, {})
    path.write_bytes(b"not the original bytes")
    with pytest.raises(ValueError, match="byte hash"):
        checker.load_array(tmp_path, metadata, {})


@pytest.mark.parametrize("readiness_passed", [True, False])
def test_full_synthetic_artifact_chain_and_exports_preserve_readiness(tmp_path, readiness_passed):
    integration = load("independent_bound_fixture", "tests/test_voluntary_analysis.py")
    run, audit_path, directions = integration.bound_run.__wrapped__(tmp_path)
    manifest = checker.read(run / "manifest.json")
    records, changed = {}, False
    for path in sorted((run / "cohorts").glob("*/*.json")):
        envelope = checker.read(path)
        for record in envelope["records"]:
            if not readiness_passed and not changed and record["kind"] != "choice":
                inference = {**record["inference"], "raw": "Synthetic reasoning.</think> invalid"}
                replacement = source_fixtures.thinking.record_from_inference(record["assignment"], inference)
                replacement["activation_artifact"] = record["activation_artifact"]
                record.clear()
                record.update(replacement)
                changed = True
            records[record["trial_id"]] = record
        envelope["sha256"] = checker.digest(envelope["records"])
        path.write_text(json.dumps(envelope))
    receipts = {
        row["source_trial_id"]: row
        for path in (run / "receipt_cohorts").glob("*/*.json")
        for row in checker.read(path)["records"]
    }
    audited = checker.read(audit_path)
    audited["readiness"] = source_fixtures.audit.readiness(manifest["diagnostics"], records, receipts)
    audited["passed"] = audited["readiness"]["passed"]
    assert audited["passed"] is readiness_passed
    audited["outcomes"] = [
        source_fixtures.audit.compact_record(a, records[a["trial_id"]]) for a in manifest["diagnostics"]
    ]
    bindings = []
    for assignment in manifest["diagnostics"]:
        record = records[assignment["trial_id"]]
        metadata = record["activation_artifact"]
        bindings.append(
            dict(
                trial_id=record["trial_id"],
                record_sha256=checker.digest(record),
                artifact_sha256=metadata["sha256"],
                row=metadata["row"],
                token_prefix_sha256=checker.digest(record["inference"]["unpadded_input_ids"]),
            )
        )
    audited["generation_record_bindings_sha256"] = checker.digest(bindings)
    audit_path.write_text(json.dumps(audited))
    out = tmp_path / "analysis"
    analyzer.analyze(run, audit_path, directions, out)
    result = checker.check(run, out / "analysis.json", audit_path, directions)
    assert result["passed"] and result["readiness_passed"] is readiness_passed
    assert not result["causal_admission"] and result["verified_counts"]["prechoice_projections"] == 2304
    assert result["verified_counts"]["projection_contrasts"] == 192
    assert result["maximum_absolute_error"] <= 1e-10
