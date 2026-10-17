"""Prepare optional-acquisition feasibility only after the full source instrument passes.

Preparation is separate from launching. Actual collection additionally depends on an
internal-mechanism signal and remaining authorized compute; this script never launches.
"""

from pathlib import Path
import hashlib
import sys
from curiosity_study import optional_information as optional
from curiosity_study import thinking_information as thinking
from curiosity_study.artifacts import read, write_once, digest
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def build_manifest(source, audit):
    if (
        source.get("assay") != "thinking_information"
        or source.get("phase") != "validation"
        or len(source.get("diagnostics", [])) != 4000
        or source.get("model", {}).get("model_id") != "Qwen/Qwen3-32B"
        or any(a["task_revision"] != "equal-uncertainty-v1" for a in source["diagnostics"])
        or source.get("assignment_sha256") != digest(source["diagnostics"])
    ):
        raise ValueError(
            "Optional feasibility requires the full 4000-branch thinking-state validation source"
        )
    if (
        audit.get("manifest_sha256") != digest(source)
        or audit.get("complete") is not True
        or audit.get("assigned") != 4000
        or audit.get("gates", {}).get("passed") is not True
    ):
        raise ValueError("Full source validation must pass its bound predefined gates before preparation")
    rows = optional.assignments()
    for a in rows:
        a["execution"] = thinking.EXECUTION.copy()
    return dict(
        schema_version=3,
        assay="thinking_information",
        phase="optional_feasibility",
        inference_scope="feasibility",
        task_revision=optional.REVISION,
        model=source["model"],
        execution_gpu="H200",
        execution=thinking.EXECUTION.copy(),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        inference_revision="thinking-2048-v1",
        capture_plan=thinking.CAPTURE_PLAN,
        task_module_sha256=hashlib.sha256(Path(optional.__file__).read_bytes()).hexdigest(),
        source_validation_manifest_sha256=digest(source),
        source_validation_audit_sha256=digest(audit),
        feasibility_criteria=dict(
            assigned_branches=96,
            worlds_each_family=12,
            minimum_valid_closed_overall=0.95,
            minimum_valid_closed_each_family=0.95,
            complete=True,
            no_execution_failures=True,
            accuracy_gate=None,
            preference_gate=None,
        ),
        invalid_output_policy="terminal_no_retry",
        response_grammar="One allowed action ID, optional single terminal period; no other text. A closing thinking delimiter is required. No action has a correct/incorrect label.",
        capture_position_rule="Same last-delimiter and pre-answer rules as the thinking-state validation.",
        action_mapping_rule="All six permutations of report/pattern/stop assigned by world index modulo six; each mapping occurs twice per family, with all four known/query cells.",
        launch_condition="Conditional on a defensible internal-mechanism signal and remaining authorized compute; preparation alone does not establish either condition.",
        interpretation="Feasibility of neutral optional acquisition only. No preference gate, confirmation claim, or inference of curiosity from report preference alone.",
    )


def verify_dispatch(rows):
    """Fail before artifact writes until optional dispatch and null scoring are integrated."""
    if thinking.task_module(rows[0]) is not optional:
        raise ValueError("Install optional-information-v1 thinking task dispatch before preparation")
    check = thinking.record_from_inference(rows[0], {"raw": "<think>Format check.</think>\nA"})
    if (
        check["expected"] is not None
        or check["correct"] is not None
        or not check["valid"]
        or not check["thinking_closed"]
    ):
        raise ValueError("Optional thinking scoring must retain expected=None and correct=None")


def main():
    source = read(ROOT / "runs/qwen32-thinking-state-v1-validation/manifest.json")
    audit = read(ROOT / "reports/qwen32-thinking-state-v1-validation-audit.json")
    manifest = build_manifest(source, audit)
    verify_dispatch(manifest["diagnostics"])
    path = ROOT / "runs/qwen32-optional-information-v1-smoke"
    write_once(path / "manifest.json", manifest)
    freeze(
        ROOT,
        path,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
        ],
    )
    print(
        dict(branches=len(manifest["diagnostics"]), source=manifest["source_sha256"], phase=manifest["phase"])
    )


if __name__ == "__main__":
    main()
