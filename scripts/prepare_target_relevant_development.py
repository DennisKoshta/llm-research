"""Freeze a balanced 960-branch target-relevance development collection.

Preparation requires the fixed smoke's feasibility audit and never launches GPUs.
These worlds are development data, including any semantic duplicates.
"""

import copy
import hashlib
from pathlib import Path
import sys

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.runner import source_hash
from curiosity_study.thinking_information import CAPTURE_PLAN, EXECUTION

ROOT = Path(__file__).resolve().parents[1]
NAME = "qwen32-target-relevant-information-v1-development"
SEED, ITEMS, WORKERS = 2026101430, 24, 2
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def build_manifest(smoke, audit):
    task_hash = hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest()
    expected = target.assignments(items=1, seed=target.SMOKE_SEED, template="table")
    for row in expected:
        row.update(execution=EXECUTION.copy(), max_new_tokens=4096, evaluation_role="target_relevance_smoke")
    if (
        smoke.get("assay") != "thinking_information"
        or smoke.get("phase") != "development_smoke"
        or smoke.get("task_revision") != target.REVISION
        or smoke.get("model", {}).get("model_id") != "Qwen/Qwen3-32B"
        or not smoke.get("model", {}).get("revision")
        or smoke.get("diagnostics") != expected
        or smoke.get("assignment_sha256") != digest(expected)
        or smoke.get("task_module_sha256") != task_hash
        or smoke.get("max_new_tokens") != 4096
        or smoke.get("response_oracle_field") != "expected_target_information_gain_bits"
    ):
        raise ValueError("Require the unchanged, fixed target-relevance smoke and target-gain oracle")
    if (
        audit.get("manifest_sha256") != digest(smoke)
        or audit.get("complete") is not True
        or audit.get("assigned") != 40
        or audit.get("feasibility_passed") is not True
        or audit.get("max_new_tokens") != 4096
        or audit.get("target_relevance_oracle_audit_version") != 1
        or audit.get("target_relevance_smoke_cell_coverage") is not True
    ):
        raise ValueError("Require complete, bound smoke feasibility and independent target-oracle checks")
    rows = target.assignments(items=ITEMS, seed=SEED, template="table")
    for row in rows:
        row.update(
            execution=EXECUTION.copy(), max_new_tokens=4096, evaluation_role="target_transfer_development"
        )
    rows.sort(key=lambda row: row["kind"])
    if EXECUTION["batch_size"] != 32 or len(rows) != 960:
        raise ValueError("The fixed development layout requires 960 branches and batch32")
    return dict(
        schema_version=3,
        assay="thinking_information",
        phase="development_validation",
        inference_scope="development",
        task_revision=target.REVISION,
        model=copy.deepcopy(smoke["model"]),
        execution_gpu="H200",
        execution=EXECUTION.copy(),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        inference_revision="thinking-4096-v1",
        max_new_tokens=4096,
        capture_plan=copy.deepcopy(CAPTURE_PLAN),
        task_module_sha256=task_hash,
        smoke_manifest_sha256=digest(smoke),
        feasibility_audit_sha256=digest(audit),
        validation_criteria=dict(
            minimum_distinct_items_per_cell=24,
            minimum_accuracy=0.9,
            minimum_valid_format=0.95,
            cell_axes=["family", "kind", "independent_report", "query"],
        ),
        partition_count=WORKERS,
        child_runs=[f"{NAME}-part-{i:02d}" for i in range(WORKERS)],
        response_oracle_field="expected_target_information_gain_bits",
        legacy_information_bits_semantics="expected_world_information_gain_bits; always one in this task",
        construct="Target-relevant information gain at constant offered-report entropy and full-world gain",
        counterbalance="Indices0:24 contain two complete complement-by-action-map12-world blocks per family.",
        development_split=dict(calibration_indices=[0, 12], check_indices=[12, 24]),
        cohort_order="Stable kind grouping after seeded assignment ordering; whole cohorts32 round-robin across two workers.",
        invalid_output_policy="terminal_no_retry",
        response_grammar="One allowed label, optional single terminal period; closing thinking delimiter required.",
        analysis_policy="Report all frozen-axis sites without refitting signs, thresholds, or axes. Record/layer47 remains the prespecified causal candidate. Report semantic duplicates; register every assignment as inspected development before confirmation. Twelve-plus-twelve blocks are development calibration/check, not independent confirmation.",
        limitations=[
            "Twenty-four generator indices per cell are not a claim of24 unique semantic worlds.",
            "All five kinds and cells must pass; no outcome-based branch selection or global-gate rescue.",
            "Main-task count is always four; independent balanced-count competence remains required.",
            "Goal-directed information selection does not establish intrinsic curiosity or subjective experience.",
        ],
    )


def partition_manifest(manifest, root=ROOT):
    rows = manifest["diagnostics"]
    if (
        len(rows) != 960
        or manifest["execution"]["batch_size"] != 32
        or manifest["partition_count"] != WORKERS
        or digest(rows) != manifest["assignment_sha256"]
    ):
        raise ValueError("Require the complete unchanged960-row design")
    cohorts = [rows[i : i + 32] for i in range(0, len(rows), 32)]
    children, bindings = [], []
    for index, name in enumerate(manifest["child_runs"]):
        indices = list(range(index, len(cohorts), WORKERS))
        selected = [row for ci in indices for row in cohorts[ci]]
        child = copy.deepcopy(manifest)
        child.update(
            phase="development_validation_partition",
            diagnostics=selected,
            assignment_sha256=digest(selected),
            parent_manifest_sha256=digest(manifest),
            partition_index=index,
            parent_cohort_indices=indices,
        )
        path = Path(root).resolve() / "runs" / name
        children.append((path, child))
        bindings.append(
            dict(
                path=str(path), manifest_sha256=digest(child), branches=len(selected), parent_cohorts=indices
            )
        )
    return children, bindings


def main():
    smoke = read(ROOT / "runs/qwen32-target-relevant-information-v1-smoke/manifest.json")
    audit = read(ROOT / "reports/qwen32-target-relevant-information-v1-smoke-audit.json")
    manifest = build_manifest(smoke, audit)
    children, bindings = partition_manifest(manifest)
    parent = ROOT / "runs" / NAME
    paths = [parent] + [path for path, _ in children]
    if any(path.exists() for path in paths):
        raise ValueError("Target development collections are immutable; use a new prospective design")
    write_once(parent / "manifest.json", manifest)
    for path, child in children:
        write_once(path / "manifest.json", child)
    write_once(parent / "children.json", bindings)
    freeze(
        ROOT,
        parent,
        [
            "reports/target-relevant-information-design.md",
            "reports/target-relevance-causal-prospective-plan.md",
            "reports/replay-reduction-engineering-amendment.md",
        ],
    )
    print(dict(run=str(parent), branches=len(manifest["diagnostics"]), source=manifest["source_sha256"]))


if __name__ == "__main__":
    main()
