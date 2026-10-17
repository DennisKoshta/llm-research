"""Independent primary arithmetic and artifact check; no analyzer imports.

Checks all optional service denominators, pre-choice projections and paired
contrasts. Receipt envelopes/artifacts are reconciled, but receipt-summary
arithmetic and GPU/tokenization engineering remain explicitly outside scope.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study.artifacts import digest, read, write_once

FAMILIES = ("objects", "machines")
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
CATEGORIES = ("report", "repeat", "pattern", "stop", "invalid", "execution_failure", "missing")
AXES = (
    "curiosity_vs_indifference",
    "satisfaction_vs_neutral",
    "confidence_high_vs_low",
    "positive_valence_vs_neutral",
    "completion_vs_ongoing",
    "generic_motivation_vs_indifference",
    "novelty_interest_vs_indifference",
    "task_demand_vs_no_demand",
)
POSITIONS = ("record", "prefill")
CONTRASTS = ("decisive_minus_redundant", "decisive_minus_novelty")
BOOTSTRAPS, SEED = 10000, 2026101434


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Comparisons:
    def __init__(self):
        self.values = 0
        self.maximum_absolute_error = 0.0

    def equal(self, actual, expected, context=""):
        if isinstance(expected, dict):
            require(isinstance(actual, dict), f"Object mismatch: {context}")
            for key, value in expected.items():
                require(key in actual, f"Missing key: {context}/{key}")
                self.equal(actual[key], value, f"{context}/{key}")
        elif isinstance(expected, (list, tuple)):
            require(
                isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                f"Sequence mismatch: {context}",
            )
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.equal(left, right, f"{context}/{index}")
        elif type(expected) is float:
            require(
                type(actual) in (float, int) and np.isfinite(actual) and np.isfinite(expected),
                f"Nonfinite/non-numeric: {context}",
            )
            error = abs(actual - expected)
            self.maximum_absolute_error = max(self.maximum_absolute_error, error)
            self.values += 1
            require(error <= 1e-10, f"Numeric mismatch {context}: {actual} versus {expected}")
        else:
            require(type(actual) is type(expected) and actual == expected, f"Value/type mismatch: {context}")


def statistics(choices, projections):
    """Independent dense world×delivery×menu construction and shared resampling."""
    require(len(choices) == 144 and len({r["trial_id"] for r in choices}) == 144, "Expected144 choices")
    choices = sorted(
        choices,
        key=lambda row: (
            row["family"],
            row["semantic_world_id"],
            row["delivery"],
            row["menu_layout"],
            row["trial_id"],
        ),
    )
    worlds = {
        family: sorted({row["semantic_world_id"] for row in choices if row["family"] == family})
        for family in FAMILIES
    }
    require(all(len(values) == 12 for values in worlds.values()), "Expected12 semantic worlds per family")
    projected = {}
    for row in projections:
        key = row["trial_id"], row["axis"], row["position"]
        require(key not in projected, "Duplicate projection")
        projected[key] = row
    require(
        set(projected)
        == {(r["trial_id"], axis, position) for r in choices for axis in AXES for position in POSITIONS},
        "Incomplete projection inventory",
    )
    actions = np.zeros((2, 12, 3, 2, 7))
    scores = np.full((2, 12, 3, 2, 8, 2, 2), np.nan)
    seen = set()
    for row in choices:
        fi = FAMILIES.index(row["family"])
        wi = worlds[row["family"]].index(row["semantic_world_id"])
        di, mi = DELIVERIES.index(row["delivery"]), row["menu_layout"]
        key = fi, wi, di, mi
        require(type(mi) is int and mi in (0, 1) and key not in seen, "Duplicate/malformed world cell")
        seen.add(key)
        actions[key + (CATEGORIES.index(row["category"]),)] = 1
        for ai, axis in enumerate(AXES):
            for pi, position in enumerate(POSITIONS):
                p = projected[row["trial_id"], axis, position]
                for si, scale in enumerate(("raw", "scaled")):
                    value = p[scale]
                    scores[key + (ai, pi, si)] = np.nan if value is None else float(value)
    require(len(seen) == 144, "Missing complete-world cell")
    rng = np.random.default_rng(SEED)
    banks = [rng.integers(0, 12, size=(BOOTSTRAPS, 12)) for _ in FAMILIES]

    def summarize(matrix):
        matrix = matrix.reshape(2, 12, -1)
        output, draws = [], []
        for fi in range(2):
            available = np.isfinite(matrix[fi]).sum(0)
            complete = available == 12
            samples = np.full((BOOTSTRAPS, matrix.shape[-1]), np.nan)
            samples[:, complete] = matrix[fi][:, complete][banks[fi]].mean(1)
            draws.append(samples)
            output.append((matrix[fi].mean(0), samples, available, 12))
        output.append((matrix.mean((0, 1)), (draws[0] + draws[1]) / 2, np.isfinite(matrix).sum((0, 1)), 24))
        rows = []
        for fi, (points, samples, available, n) in enumerate(output):
            group = []
            for index in range(len(points)):
                complete = available[index] == n
                group.append(
                    dict(
                        family=(*FAMILIES, "pooled")[fi],
                        estimate=float(points[index]) if complete else None,
                        ci95=np.quantile(samples[:, index], [0.025, 0.975]).tolist() if complete else None,
                        assigned_worlds=n,
                        available_worlds=int(available[index]),
                    )
                )
            rows.append(group)
        return rows

    behaviors, projected_contrasts = [], []
    menu_actions, menu_scores = actions.mean(3), scores.mean(3)
    for ci, name in enumerate(CONTRASTS):
        role = "primary" if ci == 0 else "secondary"
        action_stats = summarize(menu_actions[:, :, 0] - menu_actions[:, :, ci + 1])
        score_stats = summarize(menu_scores[:, :, 0] - menu_scores[:, :, ci + 1])
        for category, index in zip(CATEGORIES, range(7)):
            behaviors += [
                dict(contrast=name, role=role, category=category, **family[index]) for family in action_stats
            ]
        for ai, axis in enumerate(AXES):
            for pi, position in enumerate(POSITIONS):
                for si, scale in enumerate(("raw", "scaled")):
                    index = ai * 4 + pi * 2 + si
                    projected_contrasts += [
                        dict(
                            contrast=name,
                            role=role,
                            axis=axis,
                            position=position,
                            scale=scale,
                            **family[index],
                        )
                        for family in score_stats
                    ]
    denominators, cells = [], []
    for family in (*FAMILIES, "pooled"):
        for delivery in DELIVERIES:
            for menu in (0, 1, "averaged"):
                selected = [
                    row
                    for row in choices
                    if (family == "pooled" or row["family"] == family)
                    and row["delivery"] == delivery
                    and (menu == "averaged" or row["menu_layout"] == menu)
                ]
                counts = Counter(row["category"] for row in selected)
                base = dict(
                    family=family,
                    delivery=delivery,
                    menu_layout=menu,
                    assigned_choices=len(selected),
                    assigned_worlds=len({(r["family"], r["semantic_world_id"]) for r in selected}),
                    cap_hits=sum(r["cap_hit"] is True for r in selected),
                    unclosed=sum(r["thinking_closed"] is False for r in selected),
                )
                denominators.append(
                    dict(
                        **base,
                        counts={name: counts[name] for name in CATEGORIES},
                        proportions={name: counts[name] / len(selected) for name in CATEGORIES},
                    )
                )
                for axis in AXES:
                    for position in POSITIONS:
                        values = [
                            [projected[row["trial_id"], axis, position][key] for key in ("raw", "scaled")]
                            for row in selected
                        ]
                        finite = [
                            pair
                            for pair in values
                            if all(value is not None and np.isfinite(value) for value in pair)
                        ]
                        cells.append(
                            dict(
                                **base,
                                axis=axis,
                                position=position,
                                available_projections=len(finite),
                                mean_centered_raw=float(np.mean([pair[0] for pair in finite]))
                                if len(finite) == len(selected)
                                else None,
                                mean_scaled=float(np.mean([pair[1] for pair in finite]))
                                if len(finite) == len(selected)
                                else None,
                            )
                        )
    return dict(
        behavior_contrasts=behaviors,
        projection_contrasts=projected_contrasts,
        denominators=denominators,
        projection_cells=cells,
        bootstrap=dict(
            draws=BOOTSTRAPS,
            seed=SEED,
            family_order=list(FAMILIES),
            world_order=worlds,
            draw_bank_sha256=digest(dict(zip(FAMILIES, [bank.tolist() for bank in banks]))),
        ),
    )


def load_array(run, metadata, cache):
    relative = Path(metadata["path"])
    path = (run / relative).resolve()
    require(not relative.is_absolute() and path.is_relative_to(run.resolve()), "Unsafe activation path")
    if str(relative) not in cache:
        require(sha(path) == metadata["sha256"], "Activation byte hash mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            cache[str(relative)] = metadata["sha256"], {key: loaded[key] for key in loaded.files}
    stored_sha, arrays = cache[str(relative)]
    require(stored_sha == metadata["sha256"], "Conflicting hash for one activation file")
    return arrays


def reconstruct_choices(manifest, records, arrays, directions, comparison):
    """Score actual choices and directly multiply the captured residual vectors."""
    require(
        directions["sources"].tolist() == list(AXES)
        and directions["layers"].tolist() == [15, 31, 47, 57]
        and directions["positions"].tolist() == ["record", "final"],
        "Frozen A coordinates differ",
    )
    choices, projections, bindings = [], [], []
    for assignment in manifest["diagnostics"]:
        tid = assignment["trial_id"]
        record = records[tid]
        require(
            record["assignment"] == assignment and record["terminal"] is True and "failure" not in record,
            "Source assignment/terminal mismatch",
        )
        metadata = record["activation_artifact"]
        data = arrays[metadata["path"]][1]
        row = metadata["row"]
        require(
            data["trial_ids"][row] == tid
            and data["layers"].tolist() == metadata["layers"]
            and data["positions"].tolist() == metadata["positions"]
            and data["token_positions"][row].tolist() == metadata["token_positions"]
            and data["available"][row].tolist() == metadata["available"],
            "Source capture row/site mismatch",
        )
        bindings.append(
            dict(
                trial_id=tid,
                record_sha256=digest(record),
                artifact_sha256=metadata["sha256"],
                row=row,
                token_prefix_sha256=digest(record["inference"]["unpadded_input_ids"]),
            )
        )
        if assignment["kind"] != "choice":
            continue
        raw = record["inference"]["raw"]
        closed = "</think>" in raw
        answer = raw.rsplit("</think>", 1)[-1].strip() if closed else ""
        parsed = answer[:-1] if answer.endswith(".") else answer
        valid = closed and len(parsed) == 1 and parsed in "ABCD"
        require(
            record["valid"] is valid
            and record["thinking_closed"] is closed
            and record["correct"] is None
            and record["parsed_answer"] == parsed,
            "Choice parse differs from raw generated answer",
        )
        rotation = assignment["world_slot"] % 4
        services = CATEGORIES[:4]
        expected_mapping = dict(zip("ABCD", services[rotation:] + services[:rotation]))
        require(record["action_mapping"] == expected_mapping, "Service-to-letter rotation differs")
        service = expected_mapping[parsed] if valid else None
        require(
            service == record["selected_service"] and service in (*CATEGORIES[:4], None),
            "Choice action mapping mismatch",
        )
        choice = {
            key: assignment[key]
            for key in (
                "trial_id",
                "family",
                "kind",
                "delivery",
                "menu_layout",
                "seed",
                "index",
                "world_slot",
                "semantic_world_id",
            )
        }
        choice.update(
            category=service if valid else "invalid",
            cap_hit=record["cap_hit"],
            thinking_closed=closed,
            source_record_sha256=digest(record),
            source_artifact_sha256=metadata["sha256"],
        )
        choices.append(choice)
        li = metadata["layers"].index(15)
        for ai, axis in enumerate(AXES):
            unit = directions["unit"][ai, 0, 0].astype(float)
            center = directions["center"][ai, 0, 0].astype(float)
            norm = float(directions["contrast_norm"][ai, 0, 0])
            available = bool(
                np.isfinite(unit).all()
                and np.isfinite(center).all()
                and np.isfinite(norm)
                and norm > 0
                and np.linalg.norm(unit) > 0
            )
            if available:
                comparison.equal(
                    float(center @ unit), float(directions["threshold"][ai, 0, 0]), "A center/threshold"
                )
            for position in POSITIONS:
                pi = metadata["positions"].index(position)
                vector = data["hidden"][row, li, pi].astype(float)
                require(
                    data["available"][row, li, pi] and np.isfinite(vector).all(),
                    "Required pre-choice capture missing",
                )
                raw_score = float(np.dot(vector - center, unit)) if available else None
                base = {
                    key: choice[key]
                    for key in (
                        "trial_id",
                        "family",
                        "semantic_world_id",
                        "world_slot",
                        "delivery",
                        "menu_layout",
                        "category",
                        "source_record_sha256",
                        "source_artifact_sha256",
                    )
                }
                projections.append(
                    dict(
                        **base,
                        axis=axis,
                        position=position,
                        raw=raw_score,
                        scaled=raw_score / norm if available else None,
                        axis_available=available,
                    )
                )
    return choices, projections, bindings


def compare_csv(path, expected, comparison):
    with path.open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    require(len(actual) == len(expected), f"CSV row count mismatch: {path.name}")
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        require(set(row) == set(wanted), f"CSV columns differ: {path.name}")
        for key, value in wanted.items():
            if value is None:
                require(
                    row[key] == ("null" if key == "ci95" else ""),
                    f"CSV unavailable value changed: {path.name}/{index}/{key}",
                )
            elif type(value) is float:
                comparison.equal(float(row[key]), value, f"{path.name}/{index}/{key}")
            elif type(value) is list:
                comparison.equal(json.loads(row[key]), value, f"{path.name}/{index}/{key}")
            else:
                require(row[key] == str(value), f"CSV value differs: {path.name}/{index}/{key}")


def check(run, analysis_path, audit_path, directions_path):
    run, analysis_path, audit_path, directions_path = map(
        Path, (run, analysis_path, audit_path, directions_path)
    )
    manifest, analysis, audit = map(read, (run / "manifest.json", analysis_path, audit_path))
    c = Comparisons()
    require(
        analysis["assay"] == "voluntary_resolution_descriptive_analysis"
        and analysis["complete"] is True
        and analysis["integrity_passed"] is True
        and audit["complete"] is True
        and audit["integrity_passed"] is True,
        "Require complete audited artifacts, regardless of task readiness",
    )
    require(
        analysis["manifest_sha256"] == audit["manifest_sha256"] == digest(manifest)
        and analysis["audit_sha256"] == digest(audit)
        and analysis["audit_byte_sha256"] == sha(audit_path)
        and analysis["source_sha256"] == audit["source_sha256"] == manifest["source_sha256"]
        and analysis["model"] == audit["model"] == manifest["model"],
        "Manifest/model/source/audit hashes differ",
    )
    require(
        analysis["directions_byte_sha256"]
        == manifest["candidate_source"]["directions_byte_sha256"]
        == sha(directions_path)
        and analysis["candidate_source"] == audit["candidate_source"] == manifest["candidate_source"]
        and analysis["analysis_metadata"] == audit["analysis_metadata"] == manifest["analysis_metadata"],
        "Candidate or analysis plan differs",
    )
    require(
        analysis["readiness"] == audit["readiness"]
        and analysis["readiness_passed"] is audit["readiness"]["passed"]
        and analysis["diagnostic_readiness_repaired"] is False
        and analysis["causal_admission"] is False,
        "Readiness failure was repaired or converted to causal admission",
    )
    require(
        analysis["prior_exposure"] == manifest["prior_exposure"]
        and analysis["assigned_choices"] == 144
        and analysis["assigned_worlds"] == 24,
        "Assigned denominator or exposure declaration changed",
    )
    diagnostic_kinds = (
        "identify_resolved",
        "new_fact",
        "repeated_fact",
        "pattern_independence",
        "instrument_useful_action",
        "action_consequence",
    )
    require(
        len(audit["readiness"]["diagnostic_cells"]) == 12
        and {(r["family"], r["kind"]) for r in audit["readiness"]["diagnostic_cells"]}
        == {(family, kind) for family in FAMILIES for kind in diagnostic_kinds}
        and len(audit["readiness"]["choice_cells"]) == 6
        and {(r["family"], r["delivery"]) for r in audit["readiness"]["choice_cells"]}
        == {(family, delivery) for family in FAMILIES for delivery in DELIVERIES},
        "A frozen readiness cell was removed, duplicated, or replaced",
    )
    frozen = read(run / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and sha(run / "source.tar.gz") == frozen["archive_sha256"]
        and audit["source_freeze"]["freeze_sha256"] == digest(frozen),
        "Frozen archive mismatch",
    )
    with tarfile.open(run / "source.tar.gz", "r:gz") as archive:
        for binding in manifest["document_bindings"]:
            require(
                hashlib.sha256(archive.extractfile(binding["path"]).read()).hexdigest()
                == binding["byte_sha256"]
                == audit["source_freeze"]["documents"][binding["path"]],
                "Frozen document mismatch",
            )
        require(
            hashlib.sha256(
                archive.extractfile("analyses/curiosity-satisfaction-candidates-v1/directions.npz").read()
            ).hexdigest()
            == sha(directions_path),
            "Direction bytes differ from prospective archive",
        )
        require(
            hashlib.sha256(archive.extractfile("scripts/audit_voluntary_resolution.py").read()).hexdigest()
            == audit["auditor_byte_sha256"],
            "Audit implementation differs from freeze",
        )
    assignments = manifest["diagnostics"]
    require(
        manifest["assignments"] == []
        and len(assignments) == 288
        and digest(assignments) == manifest["assignment_sha256"],
        "Assignment inventory mismatch",
    )
    records, array_cache, offsets = {}, {}, set()
    for path in sorted((run / "cohorts").glob("*/*.json")):
        envelope = read(path)
        offset = int(path.stem)
        require(
            offset in range(0, 288, 32)
            and offset not in offsets
            and envelope["manifest_hash"] == digest(manifest)
            and envelope["sha256"] == digest(envelope["records"]),
            "Atomic source cohort hash/coverage differs",
        )
        require(
            [row["trial_id"] for row in envelope["records"]]
            == [a["trial_id"] for a in assignments[offset : offset + 32]],
            "Atomic row order differs",
        )
        for index, record in enumerate(envelope["records"]):
            tid = record["trial_id"]
            require(
                tid not in records and record["activation_artifact"]["row"] == index, "Duplicate source row"
            )
            records[tid] = record
            load_array(run, record["activation_artifact"], array_cache)
        offsets.add(offset)
    require(len(records) == 288 and len(offsets) == 9, "Missing source generation records")
    with np.load(directions_path, allow_pickle=False) as loaded:
        choices, projections, bindings = reconstruct_choices(manifest, records, array_cache, dict(loaded), c)
    require(
        digest(bindings) == audit["generation_record_bindings_sha256"]
        and bindings == analysis["generation_record_bindings"],
        "Source record/capture ledger differs",
    )
    # Recompute all frozen task-readiness cells using all assigned records.
    for group, group_field in (("diagnostic_cells", "kind"), ("choice_cells", "delivery")):
        for cell in audit["readiness"][group]:
            selected = [
                records[a["trial_id"]]
                for a in assignments
                if a["family"] == cell["family"]
                and a[group_field] == cell[group_field]
                and (a["kind"] == "choice") is (group == "choice_cells")
            ]
            values = dict(
                assigned=len(selected),
                observed=len(selected),
                missing=0,
                valid=sum(row["valid"] is True for row in selected),
                cap_hits=sum(row["cap_hit"] is True for row in selected),
                execution_failures=0,
                unclosed=sum(row["thinking_closed"] is False for row in selected),
            )
            if group == "diagnostic_cells":
                values["correct"] = sum(row["correct"] is True for row in selected)
                passed = (
                    values["assigned"] == 12
                    and values["valid"] == 12
                    and values["correct"] >= 11
                    and values["cap_hits"] == 0
                )
            else:
                passed = values["assigned"] == 24 and values["valid"] >= 23 and values["cap_hits"] == 0
            c.equal(cell, dict(**values, passed=passed), "Readiness cell")
    computed = statistics(choices, projections)
    for key, expected in computed.items():
        c.equal(analysis[key], expected, key)
    c.equal(
        analysis["readiness_passed"],
        all(row["passed"] for key in ("diagnostic_cells", "choice_cells") for row in audit["readiness"][key]),
        "Preserved readiness",
    )
    expected_exports = {
        "choices.csv": choices,
        "choice-projections.csv": projections,
        "behavior-contrasts.csv": computed["behavior_contrasts"],
        "projection-contrasts.csv": computed["projection_contrasts"],
        "denominators.csv": [
            {
                **{k: v for k, v in row.items() if k not in ("counts", "proportions")},
                "category": name,
                "count": row["counts"][name],
                "proportion": row["proportions"][name],
            }
            for row in computed["denominators"]
            for name in CATEGORIES
        ],
    }
    exports = {row["path"]: row for row in analysis["exports"]}
    require(set(exports) == {*expected_exports, "receipt-projections.csv"}, "Export inventory differs")
    for name, binding in exports.items():
        path = analysis_path.parent / name
        require(sha(path) == binding["byte_sha256"], "Export hash mismatch")
        with path.open(newline="") as handle:
            require(len(list(csv.DictReader(handle))) == binding["rows"], "Export row count mismatch")
        if name in expected_exports:
            compare_csv(path, expected_exports[name], c)
    receipt_bindings, cohort_bindings, receipts = [], [], {}
    for path in sorted((run / "receipt_cohorts").glob("*/*.json")):
        envelope = read(path)
        require(
            envelope["manifest_sha256"] == digest(manifest)
            and envelope["records_sha256"] == digest(envelope["records"]),
            "Receipt envelope differs",
        )
        cohort_bindings.append(dict(path=str(path.relative_to(run)), envelope_sha256=digest(envelope)))
        for row in envelope["records"]:
            tid = row["source_trial_id"]
            require(
                tid not in receipts and row["source_record_sha256"] == digest(records[tid]),
                "Receipt source identity differs",
            )
            receipts[tid] = row
            if row["status"] == "completed":
                metadata = row["activation_artifact"]
                data = load_array(run, metadata, array_cache)
                require(data["trial_ids"][metadata["row"]] == tid, "Receipt capture source row differs")
                receipt_bindings.append(
                    dict(
                        source_trial_id=tid,
                        sidecar_sha256=digest(row),
                        source_record_sha256=digest(records[tid]),
                        artifact_sha256=metadata["sha256"],
                        row=metadata["row"],
                        token_prefix_sha256=row["token_binding"]["token_prefix_sha256"],
                    )
                )
    require(
        set(receipts) == {row["trial_id"] for row in choices}
        and cohort_bindings == audit["receipt_bindings"]["cohorts"]
        and receipt_bindings
        == audit["receipt_bindings"]["record_bindings"]
        == analysis["receipt_record_bindings"]
        and digest(receipt_bindings) == audit["receipt_bindings"]["record_bindings_sha256"],
        "Receipt artifact ledger differs",
    )
    c.equal(
        analysis["receipt_status_counts"],
        dict(Counter(row["status"] for row in receipts.values())),
        "Receipt status totals",
    )
    c.equal(
        analysis["capture_artifacts"],
        [dict(path=path, byte_sha256=value[0]) for path, value in sorted(array_cache.items())],
        "Capture artifact inventory",
    )
    return dict(
        schema_version=1,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,
        checker_byte_sha256=sha(__file__),
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=sha(analysis_path),
        audit_sha256=digest(audit),
        manifest_sha256=digest(manifest),
        directions_byte_sha256=sha(directions_path),
        source_sha256=manifest["source_sha256"],
        readiness_passed=analysis["readiness_passed"],
        causal_admission=False,
        numerical_values_compared=c.values,
        maximum_absolute_error=c.maximum_absolute_error,
        verified_counts=dict(
            source_records=288,
            choices=144,
            prechoice_projections=len(projections),
            behavior_contrasts=len(computed["behavior_contrasts"]),
            projection_contrasts=len(computed["projection_contrasts"]),
            denominator_cells=len(computed["denominators"]),
            receipt_slots=len(receipts),
        ),
        method="Independent raw-source parsing, fixed-A vector multiplication, dense complete-world tensors and shared stratified bootstrap; no analyzer imports",
        excluded_scope=[
            "Receipt-projection and receipt-summary arithmetic (their file hashes and capture/sidecar bindings are checked).",
            "Fresh GPU execution, capture/sham engineering, prompt tokenization and psychological construct validity; these are not inferred from arithmetic agreement.",
        ],
        interpretation="Agreement does not repair failed readiness or establish causal regulation, construct specificity, or subjective experience.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "analysis", "audit", "directions", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Independent verification receipts are immutable")
    result = check(args.run, args.analysis, args.audit, args.directions)
    write_once(args.out, result)
    print(
        {
            key: result[key]
            for key in ("passed", "readiness_passed", "numerical_values_compared", "maximum_absolute_error")
        }
    )
