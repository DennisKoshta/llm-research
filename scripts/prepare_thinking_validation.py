"""Freeze the full validation, whole-cohort partitions, and an isolated execution copy."""

import copy
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
from curiosity_study.artifacts import read, write_once, digest
from curiosity_study.thinking_information import build_manifest
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/qwen32-thinking-state-v1-validation"
EXECUTION_ROOT = ROOT / "executions/thinking-state-v1-validation"


def main():
    smoke = read(ROOT / "reports/qwen32-thinking-state-v1-smoke-audit.json")
    if not smoke["feasibility_passed"]:
        raise ValueError("Feasibility screen failed")
    model = dict(
        model_id="Qwen/Qwen3-32B", revision=read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-32B"]
    )
    m = build_manifest(model, 100, 2026101408, "validation")
    rows = m["diagnostics"]
    # Group by task kind only, before outcomes; preserve random order within each kind.
    rows.sort(key=lambda a: a["kind"])
    for a in rows:
        a["representation_split"] = "extraction" if a["index"] < 50 else "development"
    m.update(
        assignment_sha256=digest(rows),
        partition_count=8,
        child_runs=[f"qwen32-thinking-state-v1-part-{i:02d}" for i in range(8)],
        execution_root=str(EXECUTION_ROOT),
        representation_split_rule="Indices 0..49 extraction; 50..99 development, across all branches. Separate fresh-vocabulary evaluation remains uncollected.",
        cohort_order="Stable diagnostic-kind grouping, random within kind, fixed original cohorts of 32.",
        feasibility_audit_sha256=digest(smoke),
    )
    write_once(PARENT / "manifest.json", m)
    children = []
    cohorts = [rows[i : i + 32] for i in range(0, len(rows), 32)]
    for index, name in enumerate(m["child_runs"]):
        child = copy.deepcopy(m)
        chosen = list(range(index, len(cohorts), 8))
        child_rows = [a for cohort_idx in chosen for a in cohorts[cohort_idx]]
        child.update(
            phase="validation_partition",
            diagnostics=child_rows,
            assignment_sha256=digest(child_rows),
            parent_manifest_sha256=digest(m),
            partition_index=index,
            parent_cohort_indices=chosen,
        )
        path = ROOT / "runs" / name
        write_once(path / "manifest.json", child)
        children.append(
            dict(
                path=str(path), manifest_sha256=digest(child), branches=len(child_rows), parent_cohorts=chosen
            )
        )
    write_once(PARENT / "children.json", children)
    if EXECUTION_ROOT.exists():
        raise ValueError("Execution copy already exists")
    EXECUTION_ROOT.mkdir(parents=True)
    for directory in ("src", "infra", "scripts", "tests"):
        for source in sorted((ROOT / directory).rglob("*.py")):
            destination = EXECUTION_ROOT / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for name in (
        "pyproject.toml",
        "uv.lock",
        "configs/model_revisions.json",
        "reports/reasoning-information-v1-plan.md",
        "reports/equal-uncertainty-v1-plan.md",
    ):
        destination = EXECUTION_ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    if (ROOT / ".python-version").exists():
        shutil.copy2(ROOT / ".python-version", EXECUTION_ROOT / ".python-version")
    os.symlink(ROOT / ".venv", EXECUTION_ROOT / ".venv", target_is_directory=True)
    copied_hash = digest(
        {
            str(p.relative_to(EXECUTION_ROOT / "src/curiosity_study")): p.read_text()
            for p in sorted((EXECUTION_ROOT / "src/curiosity_study").rglob("*.py"))
        }
    )
    assert copied_hash == m["source_sha256"] == source_hash()
    archive = PARENT / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(EXECUTION_ROOT.rglob("*")):
            if p.is_file() and not p.is_symlink():
                tar.add(p, arcname=str(p.relative_to(EXECUTION_ROOT)))
    freeze = dict(
        manifest_sha256=digest(m),
        source_sha256=m["source_sha256"],
        execution_root=str(EXECUTION_ROOT),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(
            (ROOT / "reports/reasoning-information-v1-plan.md").read_bytes()
        ).hexdigest(),
        branches=len(rows),
        budget_root=str(ROOT),
        waves=[[c["path"] for c in children[:4]], [c["path"] for c in children[4:]]],
    )
    write_once(PARENT / "freeze.json", freeze)
    print(freeze)


if __name__ == "__main__":
    main()
