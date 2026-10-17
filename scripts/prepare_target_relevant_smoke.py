"""Prepare only the prospective 40-branch target-relevance comprehension smoke.

No cloud work is launched. Dispatcher and audit integration must be reviewed
before this preparer can freeze a new run. This is not full validation.
"""

import copy
import hashlib
from pathlib import Path
import sys

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.runner import source_hash
from curiosity_study.thinking_information import CAPTURE_PLAN, EXECUTION, task_module

ROOT = Path(__file__).resolve().parents[1]
NAME = "qwen32-target-relevant-information-v1-smoke"
DOCUMENTS = [
    "reports/reasoning-information-v1-analysis-plan.md",
    "reports/reasoning-information-review-amendment.md",
    "reports/reasoning-information-development-scope-amendment.md",
    "reports/target-relevant-information-design.md",
]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def build_manifest(model):
    if model.get("model_id") != "Qwen/Qwen3-32B" or not model.get("revision"):
        raise ValueError("The target-relevance smoke requires a pinned Qwen3-32B model")
    rows = target.assignments(items=1, seed=target.SMOKE_SEED, template="table")
    for assignment in rows:
        assignment.update(
            execution=EXECUTION.copy(), max_new_tokens=4096, evaluation_role="target_relevance_smoke"
        )
    return dict(
        schema_version=3,
        assay="thinking_information",
        phase="development_smoke",
        inference_scope="development",
        task_revision=target.REVISION,
        model=copy.deepcopy(model),
        execution_gpu="H200",
        execution=EXECUTION.copy(),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        inference_revision="thinking-4096-v1",
        max_new_tokens=4096,
        capture_plan=copy.deepcopy(CAPTURE_PLAN),
        task_module_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
        validation_criteria=dict(
            minimum_distinct_items_per_cell=100,
            minimum_accuracy=0.9,
            minimum_valid_format=0.95,
            cell_axes=["family", "kind", "independent_report", "query"],
        ),
        feasibility_criteria=dict(
            complete=40, minimum_valid=38, minimum_correct=36, minimum_correct_each_family_kind=3
        ),
        construct="Target-relevant information gain; full-world gain and offered-report entropy are constant",
        response_oracle_field="expected_target_information_gain_bits",
        legacy_information_bits_semantics="expected_world_information_gain_bits; always one in this task",
        prospective_counterbalance="Target complement=(index//6)%2 and action mapping=index%6 form a complete 2-by-6 crossing in each aligned 12-world block. The one-world-per-family smoke uses index0 and does not cover this crossing.",
        world_unit_rule="task_revision, seed, family, index; keep all independent_report/query cells, kinds, renderings, and later intervention arms together. Renderings are repeated measurements, not new worlds.",
        cohort_order="Seeded whole-world shuffle, then independent_report, query, and diagnostic kind; one local collection, no worker partitioning",
        invalid_output_policy="terminal_no_retry",
        response_grammar="One allowed label, optional single terminal period; no other text. A closing thinking delimiter is required.",
        capture_position_rule="Target designation and offered report precede End of record; polarity and answer mapping follow it. Existing last-delimiter and pre-answer capture rules apply.",
        analysis_policy="Comprehension smoke only. Retain all assigned branches and report family/kind/cell correctness, invalids, and caps. Any later frozen-axis transfer must explicitly use target gain and preserve prior directions, signs, thresholds, layers, and positions without fitting or recentering on these outcomes.",
        limitations=[
            "One world per family cannot establish full validation or causal specificity.",
            "Main-task candidate count is always four; separate balanced 1/2/4-count competence remains necessary.",
            "The smoke does not cover all six report/reread/stop label mappings.",
            "Stipulated report accuracy does not measure subjective confidence; instructed acquisition is not intrinsic curiosity.",
        ],
    )


def main():
    model = dict(
        model_id="Qwen/Qwen3-32B",
        revision=read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-32B"],
    )
    manifest = build_manifest(model)
    if any(task_module(a) is not target for a in manifest["diagnostics"]):
        raise ValueError(
            "Install and verify the separate target-relevance dispatcher and audit before freezing"
        )
    from audit_thinking_information import TARGET_RELEVANCE_AUDIT_VERSION

    if TARGET_RELEVANCE_AUDIT_VERSION != 1:
        raise ValueError("The target-relevance smoke requires its independently checked oracle audit")
    path = ROOT / "runs" / NAME
    if path.exists():
        raise ValueError(
            "The target-relevance smoke already exists; immutable collections cannot be replaced"
        )
    write_once(path / "manifest.json", manifest)
    freeze(ROOT, path, DOCUMENTS)
    print(dict(run=str(path), branches=len(manifest["diagnostics"]), source=manifest["source_sha256"]))


if __name__ == "__main__":
    main()
