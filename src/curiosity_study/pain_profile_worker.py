"""Pinned-environment child process for one already admitted method-control phase."""

import argparse
import os
from pathlib import Path
import time

from .artifacts import digest, read, write_once
from . import pain_profile_assets as assets
from . import pain_profile_resources as resources
from .pain_profile_admission import CAMPAIGN_NAME


def verify_payload(root, run, phase):
    root, run = Path(root), Path(run)
    manifest = read(run / "manifest.json")
    receipt = read(run / "admissions" / f"{phase}.json")
    assets.require(run.name == CAMPAIGN_NAME and manifest["assay"] == resources.ASSAY
                   and manifest["revision"] == resources.REVISION, "Wrong campaign")
    assets.require(receipt["phase"] == phase and receipt["manifest_sha256"] == digest(manifest)
                   and receipt["source_sha256"] == manifest["source_sha256"]
                   and receipt["policy"] == resources.policy(phase)
                   and receipt["phase_consumed"] is True and receipt["retry_allowed"] is False,
                   "Invalid phase admission")
    assets.require(receipt["deadline"] - receipt["created"] == receipt["policy"]["wall_seconds"]
                   and receipt["created"] <= time.time() < receipt["deadline"], "Expired phase admission")
    for name, expected in receipt["execution_files"].items():
        path = root / name
        assets.require(not Path(name).is_absolute() and ".." not in Path(name).parts
                       and path.is_file() and not path.is_symlink() and assets.sha(path) == expected,
                       "Cloud execution file changed")
    source = digest({str(p.relative_to(root / "src/curiosity_study")): p.read_text()
                     for p in sorted((root / "src/curiosity_study").rglob("*.py"))})
    assets.require(source == manifest["source_sha256"], "Cloud source inventory changed")
    for name, expected in manifest["documents"].items():
        assets.require(receipt["execution_files"].get(name) == expected, "Cloud document binding changed")
    return manifest, receipt


def checkpoint(run, phase):
    """Ask the parent to commit volumes, without putting Modal in the pinned env."""
    marker = Path(run) / "transport" / phase / "checkpoint.request"
    marker.write_text(str(time.monotonic_ns()))


def run_phase(root, run, phase, cache_root):
    root, run = Path(root), Path(run)
    manifest, receipt = verify_payload(root, run, phase)
    from . import pain_profile_design as design
    from . import pain_profile_execution as execution

    environment = assets.check_environment(root / "configs/pain_profile_requirements.lock")
    assets.require(dict(path="configs/pain_profile_requirements.lock", **environment)
                   == manifest["environment_lock"], "Manifest numerical environment differs")
    execution.validate_manifest(manifest)
    # Reserve a minute for failure export and durable final commits within the
    # original phase wall; teardown liability never authorizes more inference.
    deadline = receipt["deadline"] - 60
    assets.require(time.time() < deadline, "No execution time remains")
    if phase == "prefetch":
        from transformers import AutoTokenizer

        identity = assets.prefetch(root, cache_root, deadline)
        tok = AutoTokenizer.from_pretrained(identity["adapter_path"], local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        design.validate_preflight(manifest["plan"], manifest["preflight"], tokenizer=tok)
        result = dict(schema_version=1, phase=phase, manifest_sha256=digest(manifest),
                      admission_sha256=digest(receipt), source_sha256=manifest["source_sha256"],
                      complete=True, passed=True, environment=environment, assets=identity,
                      model_inference_performed=False, finished_unix=time.time())
        write_once(run / "phases" / phase / "summary.json", result)
        checkpoint(run, phase)
        return result
    tok, base, model, layers, vectors, identity = assets.load_model(root, cache_root, deadline)
    identity.update(admission_sha256=digest(receipt), source_sha256=manifest["source_sha256"],
                    source_archive_sha256=receipt["source_archive_sha256"],
                    modal_image_id=os.environ.get("PAIN_PROFILE_IMAGE_ID"),
                    execution_files_sha256=digest(receipt["execution_files"]))
    return execution.execute(phase, run, manifest, model, tok, layers, vectors, identity,
                             deadline, checkpoint=lambda: checkpoint(run, phase))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--phase", choices=list(resources.POLICIES), required=True)
    args = parser.parse_args()
    try:
        result = run_phase(args.root, args.run, args.phase, args.cache)
    except Exception as error:
        manifest = read(args.run / "manifest.json")
        result = dict(schema_version=1, phase=args.phase, manifest_sha256=digest(manifest),
                      complete=False, passed=False, model_inference_performed=None,
                      error=dict(type=type(error).__name__, message=str(error)), finished_unix=time.time())
        write_once(args.run / "transport" / args.phase / "worker-error.json", result)
        summary = args.run / "phases" / args.phase / "summary.json"
        if not summary.exists():
            write_once(summary, result)
        checkpoint(args.run, args.phase)
        raise
    print({"phase": args.phase, "complete": result["complete"], "passed": result["passed"]}, flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
