"""Freeze a separate transfer-task feasibility collection before its outcomes."""

from pathlib import Path
import hashlib
import sys
from curiosity_study import relational_information as relational
from curiosity_study.artifacts import read, write_once, digest
from curiosity_study.thinking_information import CAPTURE_PLAN, EXECUTION
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa:E402


def build_manifest(model):
    rows = relational.assignments(items=1, seed=2026101410)
    for a in rows:
        a["execution"] = EXECUTION.copy()
        a["max_new_tokens"] = 4096
    return dict(
        schema_version=3,
        assay="thinking_information",
        phase="development_smoke",
        task_revision=relational.REVISION,
        model=model,
        execution_gpu="H200",
        execution=EXECUTION.copy(),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        inference_revision="thinking-4096-v1",
        max_new_tokens=4096,
        capture_plan=CAPTURE_PLAN,
        task_module_sha256=hashlib.sha256(Path(relational.__file__).read_bytes()).hexdigest(),
        validation_criteria=dict(
            minimum_distinct_items_per_cell=100,
            minimum_accuracy=0.9,
            minimum_valid_format=0.95,
            cell_axes=["family", "kind", "correlated", "query"],
        ),
        feasibility_criteria=dict(
            complete=40, minimum_valid=38, minimum_correct=36, minimum_correct_each_family_kind=3
        ),
        invalid_output_policy="terminal_no_retry",
        response_grammar="One allowed label, optional single terminal period; no other text. A closing thinking delimiter is required.",
        capture_position_rule="Same last-delimiter and pre-answer rules as the thinking-state validation.",
    )


def main():
    manifest = build_manifest(
        dict(
            model_id="Qwen/Qwen3-32B", revision=read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-32B"]
        )
    )
    path = ROOT / "runs/qwen32-relational-information-v1-smoke"
    write_once(path / "manifest.json", manifest)
    freeze(
        ROOT,
        path,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
            "reports/reasoning-information-development-scope-amendment.md",
        ],
    )
    print(dict(branches=len(manifest["diagnostics"]), source=manifest["source_sha256"]))


if __name__ == "__main__":
    main()
