"""Consume each finite method-control phase once, before starting a cloud app."""

import fcntl
import math
from pathlib import Path
import hashlib
import tarfile
import time

from .artifacts import digest, read, write_once
from .pain_profile_assets import require, sha
from . import pain_profile_resources as resources

CAMPAIGN_NAME = "qwen25-pain-profile-control-v1"


def execution_files(run, execution):
    """Check every copied file against the immutable archive, including launchers."""
    execution = Path(execution)
    require(all(not p.is_symlink() or p == execution / ".venv" for p in execution.rglob("*")),
            "Unexpected linked frozen execution entry")
    inventory = {}
    with tarfile.open(Path(run) / "source.tar.gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            require(member.isfile() and not name.is_absolute() and ".." not in name.parts
                    and str(name) not in inventory, "Unsafe or duplicate frozen archive member")
            expected = hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest()
            path = execution / name
            require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                    "Frozen execution file changed")
            inventory[str(name)] = expected
    actual = {str(p.relative_to(execution)) for p in Path(execution).rglob("*")
              if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts}
    require(actual == set(inventory), "Unexpected frozen execution file")
    return inventory


def verify_frozen_campaign(run):
    run = Path(run).resolve()
    manifest, frozen = read(run / "manifest.json"), read(run / "freeze.json")
    root = Path(frozen["budget_root"]).resolve()
    require(run == root / "runs" / CAMPAIGN_NAME, "Noncanonical or aliased campaign path")
    require(manifest.get("assay") == resources.ASSAY
            and manifest.get("revision") == resources.REVISION
            and digest(manifest) == frozen["manifest_sha256"]
            and manifest["source_sha256"] == frozen["source_sha256"]
            and sha(run / "source.tar.gz") == frozen["archive_sha256"],
            "Campaign source or manifest changed")
    require(manifest["resources"] == {p: resources.policy(p) for p in resources.POLICIES},
            "Campaign resource policy changed")
    execution = Path(frozen["execution_root"]).resolve()
    require(execution == root / "executions" / CAMPAIGN_NAME, "Unexpected execution directory")
    source = digest({str(p.relative_to(execution / "src/curiosity_study")): p.read_text()
                     for p in sorted((execution / "src/curiosity_study").rglob("*.py"))})
    require(source == manifest["source_sha256"], "Execution source copy changed")
    require(frozen["documents"] == manifest["documents"], "Document inventory differs")
    for name, expected in manifest["documents"].items():
        require(sha(execution / name) == expected, "Frozen execution document changed")
    execution_files(run, execution)
    return manifest, frozen, root


def _prior_evidence(run, phase):
    if phase == "prefetch":
        return {}
    prefetch = Path(run) / "phases/prefetch/summary.json"
    prepared = read(prefetch)
    manifest_sha256 = digest(read(Path(run) / "manifest.json"))
    require(prepared.get("complete") is True and prepared.get("passed") is True,
            "Verified CPU preparation required")
    require(prepared.get("manifest_sha256") == manifest_sha256, "Preparation belongs to another campaign")
    evidence = {"prefetch_summary_sha256": sha(prefetch)}
    if phase == "benchmark":
        summary_path = Path(run) / "phases/setup/summary.json"
        audit_path = Path(run) / "phases/setup/audit.json"
        summary, audit = read(summary_path), read(audit_path)
        require(audit.get("complete") is True and audit.get("passed") is True
                and summary.get("complete") is True and summary.get("passed") is True,
                "Complete independently audited setup required")
        require(audit.get("manifest_sha256") == manifest_sha256
                and summary.get("manifest_sha256") == manifest_sha256
                and audit.get("summary_byte_sha256") == sha(summary_path),
                "Setup audit does not bind this campaign summary")
        forecast = summary["forecast"]
        seconds = forecast["benchmark_seconds"]
        require(type(seconds) in (float, int) and math.isfinite(seconds)
                and 0 < seconds <= 2880 and forecast["limit_seconds"] == 2880
                and forecast["passed"] is True, "Full benchmark exceeds measured timing limit")
        # The raw setup auditor owns reproduction of the timing formula and
        # exact measurement bindings; the admission binds that entire receipt.
        evidence.update(setup_summary_sha256=sha(summary_path), setup_audit_sha256=sha(audit_path),
                        benchmark_forecast_seconds=seconds)
    return evidence


def claim(run, phase, state, *, now=None):
    manifest, frozen, root = verify_frozen_campaign(run)
    run = Path(run).resolve()
    selected = resources.policy(phase)
    require(state.get("all_stopped_zero_tasks") is True, "A project app is still active")
    available = state.get("available_compute_estimate_usd")
    require(type(available) in (float, int) and math.isfinite(available)
            and available >= resources.CAMPAIGN_CEILING_USD,
            "Fresh compute allowance cannot cover the bounded campaign")
    now = time.time() if now is None else now
    require(type(now) in (float, int) and math.isfinite(now), "Invalid admission time")
    directory = root / "campaign_admissions" / resources.REVISION
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        destination = directory / f"{phase}.json"
        require(not destination.exists() and not (run / "admissions" / f"{phase}.json").exists(),
                "Phase already consumed; no paid retry or alias permitted")
        prior = {}
        for p in resources.POLICIES:
            path = directory / f"{p}.json"
            if path.exists():
                receipt = read(path)
                require(receipt["manifest_sha256"] == digest(manifest)
                        and receipt["policy"] == resources.policy(p)
                        and receipt["phase"] == p
                        and receipt["source_sha256"] == manifest["source_sha256"]
                        and receipt["source_archive_sha256"] == frozen["archive_sha256"]
                        and receipt == read(run / "admissions" / f"{p}.json")
                        and receipt["prior_evidence"] == _prior_evidence(run, p),
                        "Prior admission binding changed")
                prior[p] = receipt
        for p, receipt in prior.items():
            expected_phases = {"prefetch": set(), "setup": {"prefetch"},
                               "benchmark": {"prefetch", "setup"}}[p]
            require(receipt["prior_admissions"] == {q: digest(prior[q]) for q in expected_phases},
                    "Prior admission chain changed")
        require(phase != "prefetch" or not prior, "Prefetch must be first")
        require(phase != "setup" or set(prior) == {"prefetch"}, "Setup requires its sole preparation")
        require(phase != "benchmark" or set(prior) == {"prefetch", "setup"},
                "Benchmark requires its original setup")
        require(sum(r["policy"]["maximum_compute_usd"] for r in prior.values())
                + selected["maximum_compute_usd"] <= resources.CAMPAIGN_CEILING_USD,
                "Cumulative campaign liability exceeded")
        evidence = _prior_evidence(run, phase)
        receipt = dict(
            schema_version=1, phase=phase, campaign=str(run), manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"], source_archive_sha256=frozen["archive_sha256"],
            policy=selected, created=now, deadline=now + selected["wall_seconds"],
            execution_files=execution_files(run, Path(frozen["execution_root"])),
            prior_evidence=evidence, prior_admissions={p: digest(r) for p, r in prior.items()},
            state_timestamp=state["timestamp_utc"], available_compute_estimate_usd=available,
            phase_consumed=True, retry_allowed=False,
        )
        # Consume the phase first. An interrupted mirror write never permits a
        # second cloud launch; this deliberately favors a recorded failed setup.
        write_once(destination, receipt)
        write_once(run / "admissions" / f"{phase}.json", receipt)
        return receipt
