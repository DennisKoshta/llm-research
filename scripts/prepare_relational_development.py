"""Freeze an 800-branch relational development diagnostic after its fixed smoke passes.

This is a small development collection, not confirmation. Whole original cohorts
of 32 are assigned to two workers. Preparation never launches cloud computation.
"""

import copy
import hashlib
from pathlib import Path
import sys
from curiosity_study import relational_information as relational
from curiosity_study.artifacts import read, write_once, digest
from curiosity_study.thinking_information import CAPTURE_PLAN, EXECUTION
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
NAME = "qwen32-relational-information-v1-development"
SEED = 2026101416
ITEMS = 20
WORKERS = 2
DOCUMENTS = [
    "reports/reasoning-information-v1-analysis-plan.md",
    "reports/reasoning-information-review-amendment.md",
    "reports/reasoning-information-development-scope-amendment.md",
]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def build_manifest(smoke, audit):
    task_hash = hashlib.sha256(Path(relational.__file__).read_bytes()).hexdigest()
    expected_smoke = relational.assignments(items=1, seed=2026101410)
    for a in expected_smoke:
        a["execution"] = EXECUTION.copy()
        a["max_new_tokens"] = 4096
    if (
        smoke.get("assay") != "thinking_information"
        or smoke.get("phase") != "development_smoke"
        or smoke.get("task_revision") != relational.REVISION
        or smoke.get("model", {}).get("model_id") != "Qwen/Qwen3-32B"
        or smoke.get("diagnostics") != expected_smoke
        or smoke.get("assignment_sha256") != digest(expected_smoke)
        or smoke.get("task_module_sha256") != task_hash
        or smoke.get("max_new_tokens") != 4096
        or smoke.get("inference_revision") != "thinking-4096-v1"
    ):
        raise ValueError("Relational development requires the unchanged fixed 40-branch smoke")
    if (
        audit.get("manifest_sha256") != digest(smoke)
        or audit.get("complete") is not True
        or audit.get("assigned") != 40
        or audit.get("feasibility_passed") is not True
        or audit.get("max_new_tokens") != 4096
    ):
        raise ValueError("The bound relational smoke feasibility audit must pass before preparation")
    rows = relational.assignments(items=ITEMS, seed=SEED, template="table")
    for a in rows:
        a["execution"] = EXECUTION.copy()
        a["max_new_tokens"] = 4096
        a["evaluation_role"] = "transfer_development"
    # The generator shuffles independently of outcomes; stable kind sorting retains that order.
    rows.sort(key=lambda a: a["kind"])
    if EXECUTION["batch_size"] != 32 or len(rows) != 800:
        raise ValueError("The fixed relational development design requires 800 rows and batch size 32")
    return dict(
        schema_version=3,
        assay="thinking_information",
        phase="development_validation",
        inference_scope="development",
        task_revision=relational.REVISION,
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
            minimum_distinct_items_per_cell=20,
            minimum_accuracy=0.9,
            minimum_valid_format=0.95,
            cell_axes=["family", "kind", "correlated", "query"],
        ),
        partition_count=WORKERS,
        child_runs=[f"{NAME}-part-{i:02d}" for i in range(WORKERS)],
        cohort_order="Stable diagnostic-kind grouping after seeded assignment shuffle; fixed original full cohorts of 32, round-robin across two children.",
        invalid_output_policy="terminal_no_retry",
        response_grammar="One allowed label, optional single terminal period; no other text. A closing thinking delimiter is required.",
        capture_position_rule="Same last-delimiter and pre-answer rules as the thinking-state validation.",
        analysis_policy="Descriptive transfer only: apply existing frozen information-state directions, signs, and extraction thresholds without fitting or recentering on these 20 worlds per family. Report every captured site and missingness. These worlds become inspected development data and must be excluded from later confirmation.",
        interpretation="Small development diagnostic with 20 distinct world indices per cell; passing its 90% accuracy and 95% validity gates is not the 100-world full-validation or a confirmatory mechanism result.",
    )


def partition_manifest(manifest, root=ROOT):
    """Return in-memory child manifests and bindings; no artifacts are written here."""
    size = manifest["execution"]["batch_size"]
    rows = manifest["diagnostics"]
    if size != 32 or len(rows) != 800 or manifest["partition_count"] != WORKERS:
        raise ValueError("Only the fixed 800-row, two-worker design is supported")
    if digest(rows) != manifest["assignment_sha256"]:
        raise ValueError("Parent assignment hash mismatch")
    cohorts = [rows[i : i + size] for i in range(0, len(rows), size)]
    children, bindings = [], []
    for index, name in enumerate(manifest["child_runs"]):
        chosen = list(range(index, len(cohorts), WORKERS))
        child_rows = [a for ci in chosen for a in cohorts[ci]]
        child = copy.deepcopy(manifest)
        child.update(
            phase="development_validation_partition",
            diagnostics=copy.deepcopy(child_rows),
            assignment_sha256=digest(child_rows),
            parent_manifest_sha256=digest(manifest),
            partition_index=index,
            parent_cohort_indices=chosen,
        )
        path = Path(root).resolve() / "runs" / name
        children.append((path, child))
        bindings.append(
            dict(
                path=str(path), manifest_sha256=digest(child), branches=len(child_rows), parent_cohorts=chosen
            )
        )
    return children, bindings


def main():
    smoke = read(ROOT / "runs/qwen32-relational-information-v1-smoke/manifest.json")
    audit = read(ROOT / "reports/qwen32-relational-information-v1-smoke-audit.json")
    manifest = build_manifest(smoke, audit)
    children, bindings = partition_manifest(manifest)
    parent = ROOT / "runs" / NAME
    write_once(parent / "manifest.json", manifest)
    for path, child in children:
        write_once(path / "manifest.json", child)
    write_once(parent / "children.json", bindings)
    freeze(ROOT, parent, DOCUMENTS)
    print(
        dict(
            branches=len(manifest["diagnostics"]),
            children=len(children),
            source=manifest["source_sha256"],
            phase=manifest["phase"],
        )
    )


if __name__ == "__main__":
    main()
