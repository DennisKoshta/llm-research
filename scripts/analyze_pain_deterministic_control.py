"""Run the exact frozen original analysis on an authentic new control audit."""

import argparse
import json
from pathlib import Path

from curiosity_study.artifacts import digest, read, write_once
from audit_pain_deterministic_control import (
    ANALYZER_SHA, AUDITOR_SHA, METHOD, audit, frozen_inputs, original_runtime, require, sha,
)


def analyze_run(run, audit_path=None):
    run = Path(run).resolve()
    path = run / "phases/benchmark/audit.json" if audit_path is None else Path(audit_path).resolve()
    receipt = read(path)
    require(receipt == audit(run, "benchmark"), "New audit does not reproduce from original raw evidence")
    inputs = frozen_inputs(run)
    manifest = inputs["manifest"]
    require(manifest["diagnostic_files"][Path(__file__).name] == sha(__file__)
            and receipt["phase"] == "benchmark" and receipt["manifest_sha256"] == digest(manifest)
            and receipt["assigned_rows"] == len(receipt["outcomes"]) == 2020, "Wrong analyzer binding/benchmark denominator")
    with original_runtime(inputs) as (_, analysis, _, _):
        fields = (*analysis.METADATA, "base_request_id", "gen_seed", "status", "category", "valid", "cap_hit", "raw", "record_sha256")
        rows = [{key: row[key] for key in fields if key in row} for row in receipt["outcomes"]]
        result = analysis.analyze_tables(manifest["original_manifest"]["plan"], rows,
                                        dict(complete=receipt["complete"], passed=receipt["passed"]))
    result.update(manifest_sha256=digest(manifest), original_manifest_sha256=receipt["original_manifest_sha256"],
        source_sha256=receipt["source_sha256"], source_archive_sha256=receipt["archive_byte_sha256"],
        freeze_byte_sha256=receipt["freeze_byte_sha256"], audit_sha256=digest(receipt), audit_byte_sha256=sha(path),
        auditor_byte_sha256=receipt["auditor_byte_sha256"], analyzer_byte_sha256=sha(__file__),
        original_auditor_byte_sha256=AUDITOR_SHA, original_analyzer_byte_sha256=ANALYZER_SHA,
        source_audit_recomputed=True, failed_integrity_can_qualify=False, method=METHOD,
        original_campaign_remains_failed=True, scientific_rules_changed=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_run(args.run, args.audit)
    write_once(args.out, result)
    print(json.dumps(result["decision"]))
