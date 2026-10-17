"""Freeze either validation or independently sourced activation-discovery assignments."""

import argparse
import hashlib
from pathlib import Path
import tarfile
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.information_state import assignments, build_manifest, evaluate, prompt
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]


def module_hash():
    return hashlib.sha256((ROOT / "src/curiosity_study/information_state.py").read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"], required=True)
    p.add_argument("--stage", choices=["validation", "capture"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--validation", type=Path)
    args = p.parse_args()
    revisions = read(ROOT / "configs/model_revisions.json")
    model = dict(model_id=args.model, revision=revisions[args.model])
    if args.stage == "validation":
        manifest = build_manifest(model)
    else:
        if not args.validation:
            raise ValueError("A passing validation is required")
        vm = read(args.validation / "manifest.json")
        vr = RunStore(args.validation, vm).completed()
        gate = evaluate(vm, vr)
        if not gate["passed"] or vm["model"] != model or vm["task_module_sha256"] != module_hash():
            raise ValueError("Validation gates, exact model, and exact task module must match")
        rows = []
        for split, items, seed in [("train", 100, 2026101403), ("development", 64, 2026101404)]:
            for a in assignments(items, seed, kinds=("passive",)):
                rows.append(
                    {
                        **a,
                        "assay": "state_mechanism",
                        "split": split,
                        "execution": {"backend": "state_prefill_v1", "batch_size": 16},
                    }
                )
        manifest = dict(
            schema_version=3,
            assay="state_mechanism",
            operation="capture",
            phase="representation_development",
            task_revision="equal-uncertainty-v1",
            model=model,
            execution_gpu="H100",
            execution={"backend": "state_prefill_v1", "batch_size": 16},
            source_sha256=source_hash(),
            assignments=rows,
            diagnostics=[],
            assignment_sha256=digest(rows),
            validation_manifest_sha256=digest(vm),
            validation_gate=gate,
            extraction="balanced paired novelty contrast; passive prompts; all four depths and both positions",
            invalid_output_policy="no output generation; no failed-request retries",
        )
    manifest["task_module_sha256"] = module_hash()
    write_once(args.out / "manifest.json", manifest)
    rows = manifest["assignments"] + manifest["diagnostics"]
    write_once(
        args.out / "prompt-examples.json",
        [dict(assignment=a, prompt=prompt(a)[0], expected=prompt(a)[2]) for a in rows[:8]],
    )
    archive = args.out / "source.tar.gz"
    if archive.exists():
        raise ValueError("Source archive already exists; inspect the prepared run")
    with tarfile.open(archive, "w:gz") as tar:
        for directory in ("src", "infra", "scripts", "tests"):
            for file in sorted((ROOT / directory).rglob("*.py")):
                tar.add(file, arcname=str(file.relative_to(ROOT)))
        for file in (
            "pyproject.toml",
            "uv.lock",
            "configs/model_revisions.json",
            "reports/internal-mechanisms-phase-plan.md",
            "reports/equal-uncertainty-v1-plan.md",
        ):
            tar.add(ROOT / file, arcname=file)
    write_once(
        args.out / "freeze.json",
        dict(
            source_sha256=manifest["source_sha256"],
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            branches=len(rows),
            plan_sha256=hashlib.sha256(
                (ROOT / "reports/equal-uncertainty-v1-plan.md").read_bytes()
            ).hexdigest(),
        ),
    )
    print(
        dict(
            run=str(args.out),
            phase=manifest["phase"],
            branches=len(rows),
            source_sha256=manifest["source_sha256"],
        )
    )


if __name__ == "__main__":
    main()
