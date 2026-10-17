import argparse
from pathlib import Path
from .artifacts import read, write_once
from .assignments import make_assignments
from .runner import build_manifest, execute
from .report import render


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--items", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=20261006)
    prepare.add_argument("--model", default="mock")
    prepare.add_argument(
        "--assay",
        choices=("priced", "one_credit", "one_credit_v2", "single_curiosity", "single_curiosity_v2"),
        default="priced",
    )
    prepare.add_argument("--phase", choices=("full", "validation", "behavior"), default="full")
    prepare.add_argument("--validation-run", type=Path)
    prepare.add_argument("--credit-configuration", type=Path, default=Path("configs/one-credit-v2.json"))
    prepare.add_argument("--revisions", type=Path, default=Path("configs/model_revisions.json"))
    run = sub.add_parser("run")
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        identity = {"model_id": args.model}
        if args.model != "mock":
            identity["revision"] = read(args.revisions)[args.model]
        if args.assay == "single_curiosity_v2":
            from .single_curiosity import build_v2_manifest

            manifest = build_v2_manifest(args.items, args.seed, identity, args.phase)
        elif args.assay == "single_curiosity":
            from .single_curiosity import build_manifest as build_single_manifest

            manifest = build_single_manifest(args.items, args.seed, identity, args.phase)
        elif args.assay == "one_credit_v2":
            from .credit import build_variant_manifest

            manifest = build_variant_manifest(
                args.items, args.seed, identity, args.phase, read(args.credit_configuration)
            )
        elif args.assay == "one_credit":
            from .credit import build_manifest as build_credit_manifest

            manifest = build_credit_manifest(args.items, args.seed, identity, args.phase)
        else:
            manifest = build_manifest(make_assignments(args.items, args.seed), identity, args.phase)
        if args.phase == "behavior":
            from .validation import evaluate_run
            from .artifacts import digest

            if args.validation_run is None:
                raise ValueError("Behavior phase requires --validation-run")
            validation = evaluate_run(args.validation_run)
            prerequisite = read(args.validation_run / "manifest.json")
            if not validation["passed"]:
                raise ValueError(
                    "Validation failed; a new behavioral baseline is not authorized by the advancement rule"
                )
            for key in ("source_sha256", "model", "task_revision", "planned_behavior_sha256"):
                if prerequisite[key] != manifest[key]:
                    raise ValueError(f"Validation mismatch: {key}")
            if prerequisite.get("execution_gpu", "L40S") != manifest.get("execution_gpu", "L40S"):
                raise ValueError("Validation mismatch: execution_gpu")
            if prerequisite.get("execution") != manifest.get("execution"):
                raise ValueError("Validation mismatch: execution backend")
            manifest["validation_prerequisite"] = {
                "manifest_sha256": digest(prerequisite),
                "result": validation,
            }
        write_once(args.out / "manifest.json", manifest)
        print(
            f"Prepared {len(manifest['assignments'])} behavioral and {len(manifest['diagnostics'])} diagnostic branches."
        )
    elif args.command == "run":
        from .model import HFModel, MockModel
        from .integrity import verify_model

        manifest = read(args.run / "manifest.json")
        from .runner import source_hash

        if source_hash() != manifest["source_sha256"]:
            raise ValueError("Source changed since assignment freeze; prepare a new run")
        identity = manifest["model"]
        model_class = HFModel
        if manifest.get("execution", {}).get("backend") == "torch_batched_v1":
            from .batched import BatchedHFModel, verify_batch_model

            model_class = BatchedHFModel
        model = MockModel() if identity["model_id"] == "mock" else model_class(**identity, device=args.device)
        if isinstance(model, HFModel):
            import uuid

            audit = verify_model(model)
            if manifest.get("execution", {}).get("backend") == "torch_batched_v1":
                audit["batch_audit"] = verify_batch_model(model)
                audit["passed"] = audit["passed"] and audit["batch_audit"]["passed"]
            write_once(args.run / "integrity" / f"{uuid.uuid4().hex}.json", audit)
            if not audit["passed"]:
                raise RuntimeError("Model integration audit failed")
        execute(args.run, manifest, model)
        (args.run / "report.md").write_text(render(args.run))
    else:
        print(render(args.run))


if __name__ == "__main__":
    main()
