"""Freeze one finite published-method campaign; never launch or admit a phase.

All expensive assets remain outside the source archive. Preparation verifies
local metadata, CPU vector values, the exact numerical environment and every
tokenized request before its first write. Actual weights load only in a later,
separately admitted bounded phase.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

from curiosity_study import pain_profile_assets as assets
from curiosity_study import pain_profile_design as design
from curiosity_study import pain_profile_resources as resources
from curiosity_study.artifacts import digest, read, write_once

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402

RUN = "runs/qwen25-pain-profile-control-v1"
PLAN = "reports/pain-axis-positive-control-v1-plan.md"
ADAPTATION = "reports/pain-axis-positive-control-v1-source-adaptation.json"
LOCK = "configs/pain_profile_requirements.lock"
REFERENCE = str(assets.REFERENCE)
ASSETS = str(assets.ASSETS)
TOKENIZER = ASSETS + "/metadata/report_dataset_full_1684p_20260831-065321"
INVENTORY_SHA = "46ebab7cda188c198de4325c9df0a29196ff536eafbcc307c16073d68338ffe0"
ASSET_RECEIPT_SHA = "bf5ad7f6f4dec090efbdbf4e7ff5fa76088f1fc70ed529f975fd2af035ecc527"
WEIGHT_PINS_SHA = "b64ef970ed3a7b83f820bfa4d05a7e8d7d20a9cb5a492600ab3e5c39a4434fe0"
C_RUN = "runs/qwen32-curiosity-suppression-screen-v1-admission2"
C_MANIFEST_SHA = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
C_SOURCE_SHA = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
C_ARCHIVE_SHA = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
C_FILES = dict(
    manifest=C_RUN + "/manifest.json",
    freeze=C_RUN + "/freeze.json",
    audit="reports/qwen32-curiosity-suppression-screen-v1-audit.json",
    analysis="analyses/curiosity-suppression-screen-v1/analysis.json",
    arithmetic="reports/curiosity-suppression-screen-v1-independent-check.json",
    scoring="reports/curiosity-suppression-screen-v1-independent-scoring.json",
    shutdown="reports/qwen32-curiosity-suppression-screen-v1-admission2-cost-and-shutdown.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root, name):
    """Only ordinary, root-relative files; reject traversal and linked parents."""
    relative = PurePosixPath(name)
    require(
        isinstance(name, str)
        and str(relative) == name
        and not relative.is_absolute()
        and ".." not in relative.parts
        and name not in ("", "."),
        "Unsafe document path",
    )
    path = Path(root) / name
    require(
        path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
        f"Missing or linked document: {name}",
    )
    require(path.resolve().is_relative_to(Path(root).resolve()), "Document escapes project")
    return path


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hash(root):
    source = Path(root) / "src/curiosity_study"
    paths = sorted(source.rglob("*.py"))
    require(paths and all(not p.is_symlink() for p in paths), "Source files missing or linked")
    return digest({str(p.relative_to(source)): p.read_text() for p in paths})


def closed_c_evidence(root):
    """Verify closure, retaining the prior decision without making it an input gate."""
    root = Path(root)
    values = {key: read(safe_file(root, name)) for key, name in C_FILES.items()}
    m, frozen, audit, analysis, arithmetic, scoring, shutdown = (values[key] for key in C_FILES)
    require(
        digest(m) == frozen["manifest_sha256"] == C_MANIFEST_SHA
        and m["source_sha256"] == frozen["source_sha256"] == C_SOURCE_SHA
        and sha(safe_file(root, C_RUN + "/source.tar.gz")) == frozen["archive_sha256"] == C_ARCHIVE_SHA,
        "Preceding C manifest/source archive changed",
    )
    require(
        audit["parent_manifest_sha256"] == C_MANIFEST_SHA
        and audit["source_sha256"] == C_SOURCE_SHA
        and audit["complete"] is True
        and audit["integrity_passed"] is True
        and audit["assigned"] == audit["terminal_generations"] == 4320
        and analysis["parent_manifest_sha256"] == C_MANIFEST_SHA
        and analysis["source_sha256"] == C_SOURCE_SHA
        and analysis["complete"] is True
        and analysis["integrity_passed"] is True
        and analysis["audit_sha256"] == digest(audit)
        and analysis["audit_byte_sha256"] == sha(root / C_FILES["audit"]),
        "Complete integrity-passed C audit and bound analysis required",
    )
    require(
        arithmetic["parent_manifest_sha256"] == C_MANIFEST_SHA
        and arithmetic["source_sha256"] == C_SOURCE_SHA
        and arithmetic["archive_byte_sha256"] == C_ARCHIVE_SHA
        and arithmetic["passed"] is True
        and arithmetic["assigned"] == 4320
        and arithmetic["analysis_sha256"] == digest(analysis)
        and arithmetic["audit_sha256"] == digest(audit)
        and arithmetic["analysis_byte_sha256"] == sha(root / C_FILES["analysis"])
        and arithmetic["audit_byte_sha256"] == sha(root / C_FILES["audit"])
        and arithmetic["decision"] == {key: analysis["decision"][key] for key in arithmetic["decision"]}
        and {"classification", "positive_screen"} <= arithmetic["decision"].keys(),
        "Independent C arithmetic binding failed",
    )
    require(
        scoring["parent_manifest_sha256"] == C_MANIFEST_SHA
        and scoring["source_sha256"] == C_SOURCE_SHA
        and scoring["source_archive_sha256"] == C_ARCHIVE_SHA
        and scoring["freeze_byte_sha256"] == sha(root / C_FILES["freeze"])
        and scoring["passed"] is True
        and scoring["complete"] is True
        and scoring["response_scoring_complete"] is True
        and scoring["computation_agrees"] is True
        and scoring["assigned"] == scoring["checked"] == 4320
        and scoring["mismatches"] == [],
        "Independent complete C raw scoring required",
    )
    apps = shutdown["apps"]
    require(
        shutdown["all_stopped_zero_tasks"] is True
        and isinstance(apps, list)
        and apps
        and all(row["State"] == "stopped" and str(row["Tasks"]) == "0" for row in apps),
        "Prior shutdown receipt must show all listed project apps stopped with zero tasks",
    )
    return dict(
        passed=True,
        scope="Closed prior evidence only; no selection on C readiness, effect or direction-selectivity",
        prior_readiness=deepcopy(audit["readiness"]),
        prior_decision=deepcopy(analysis["decision"]),
        bindings={
            key: dict(path=name, byte_sha256=sha(root / name), content_sha256=digest(values[key]))
            for key, name in C_FILES.items()
        },
        source_archive=dict(
            path=C_RUN + "/source.tar.gz",
            byte_sha256=C_ARCHIVE_SHA,
            verified_locally=True,
            copied_into_campaign=False,
        ),
        live_cloud_state_requeried=False,
    )


def collect_documents(root):
    """Verify exact primary metadata and return the finite source-copy inventory."""
    root = Path(root)
    names = {PLAN, ADAPTATION, LOCK, "configs/pain_profile_requirements.in", *C_FILES.values()}
    require(
        sha(safe_file(root, REFERENCE + "/inventory.json")) == INVENTORY_SHA,
        "Primary-source inventory changed",
    )
    sums = safe_file(root, REFERENCE + "/SHA256SUMS")
    reference_names = set()
    for line in sums.read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", expected), "Malformed reference checksum")
        full = REFERENCE + "/" + name
        require(
            full not in reference_names and sha(safe_file(root, full)) == expected,
            "Reference checksum mismatch or duplicate",
        )
        reference_names.add(full)
    require(
        reference_names
        and {str(p.relative_to(root)) for p in (root / REFERENCE).rglob("*") if p.is_file()}
        == reference_names | {str(sums.relative_to(root))},
        "Unexpected or omitted primary-reference file",
    )
    names.update(reference_names | {str(sums.relative_to(root))})

    receipt_path = ASSETS + "/asset-receipt.json"
    require(sha(safe_file(root, receipt_path)) == ASSET_RECEIPT_SHA, "Adapter inspection receipt changed")
    receipt = read(root / receipt_path)
    require(
        receipt["repository"] == assets.ADAPTER_REPO
        and receipt["revision"] == assets.ADAPTER_REVISION
        and receipt["archive_sha256"] == assets.ADAPTER_SHA
        and receipt["safetensors_header_only"]["dtype_counts"] == {"F32": 896},
        "Unexpected released adapter identity",
    )
    names.add(receipt_path)
    retained = set()
    for member in receipt["members"]:
        if "retained_path" not in member:
            continue
        name = ASSETS + "/" + member["retained_path"]
        path = safe_file(root, name)
        require(
            member["type"] == "file"
            and path.stat().st_size == member["byte_count"]
            and sha(path) == member["sha256"]
            and name not in retained,
            "Retained adapter metadata changed",
        )
        retained.add(name)
    require(
        {str(p.relative_to(root)) for p in (root / ASSETS / "metadata").rglob("*") if p.is_file()} == retained
        and len(retained) == 6,
        "Unexpected or omitted retained adapter metadata",
    )
    names.update(retained)
    pins_path = ASSETS + "/base-weight-pins.json"
    require(sha(safe_file(root, pins_path)) == WEIGHT_PINS_SHA, "Base weight pins changed")
    pins = read(root / pins_path)
    require(
        pins["repo"] == assets.BASE_REPO
        and pins["revision"] == assets.BASE_REVISION
        and [r["path"] for r in pins["files"]]
        == [f"model-{i:05d}-of-00017.safetensors" for i in range(1, 18)]
        and all(
            type(r["size"]) is int and r["size"] > 0 and re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
            for r in pins["files"]
        ),
        "Exact 17 base-weight identities required",
    )
    generation_name = ASSETS + "/base-generation-config.json"
    generation_receipt = read(safe_file(root, ASSETS + "/base-generation-config-receipt.json"))
    require(
        sha(safe_file(root, generation_name))
        == design.BASE_GENERATION_CONFIG_SHA256
        == generation_receipt["byte_sha256"]
        and generation_receipt["model"] == assets.BASE_REPO
        and generation_receipt["revision"] == assets.BASE_REVISION
        and read(root / generation_name)["eos_token_id"] == list(design.BASE_GENERATION_EOS_IDS),
        "Pinned generation metadata changed",
    )
    names.update((pins_path, generation_name, ASSETS + "/base-generation-config-receipt.json"))

    adaptation = read(safe_file(root, ADAPTATION))
    require(
        adaptation["commit"] == design.SOURCE_COMMIT
        and adaptation["independent_review"]["concrete_blockers"] == []
        and adaptation["local_validation"]["failed"] == 0
        and adaptation["local_validation"]["passed"] > 0
        and adaptation["local_validation"]["ruff_passed"] is True
        and adaptation["scientific_admission"] is False,
        "Reviewed and locally validated engine adaptation required",
    )
    required_adaptation = {
        "src/curiosity_study/pain_profile_engine.py",
        "tests/test_pain_profile_engine.py",
        REFERENCE + "/" + design.SOURCE_SUBDIR + "/src/run_profile.py",
        REFERENCE + "/upstream/LICENSE",
        receipt_path,
    }
    require(
        {r["path"] for r in adaptation["files"]} == required_adaptation
        and len(adaptation["files"]) == len(required_adaptation),
        "Adaptation inventory changed",
    )
    for row in adaptation["files"]:
        require(sha(safe_file(root, row["path"])) == row["byte_sha256"], "Reviewed engine bytes changed")
        names.add(row["path"])
    require(
        all(row["identical"] is True for row in adaptation["critical_expression_ast_comparisons"])
        and adaptation["critical_expression_ast_comparisons"],
        "Source arithmetic comparison failed",
    )
    for name in names:
        require(
            not name.endswith("adapter_Qwen_2.5_32B_instruct.tar.gz")
            and re.fullmatch(r"model-\d+-of-\d+\.safetensors", Path(name).name) is None,
            "Weight payloads cannot enter source archive",
        )
    return {name: sha(safe_file(root, name)) for name in sorted(names)}


def load_tokenizer(root):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(Path(root) / TOKENIZER), local_files_only=True)


def setup_call_inventory(plan, preflight):
    from curiosity_study.pain_profile_execution import setup_call_plan

    return setup_call_plan(plan, preflight)


def build_manifest(root=ROOT, *, tokenizer=None):
    """Read-only build; CPU tensor verification and tokenizer only, never weights."""
    from analyze_pain_profile_control import ANALYSIS_METADATA
    from curiosity_study.pain_profile_execution import validate_manifest

    root = Path(root).resolve()
    documents = collect_documents(root)
    closure = closed_c_evidence(root)
    environment = assets.check_environment(root / LOCK)
    _, vectors = assets.load_vectors(root, "cpu")
    plan = design.assignment_plan(design.load_sources(root / REFERENCE))
    tokenizer = load_tokenizer(root) if tokenizer is None else tokenizer
    preflight = design.tokenize_plan(plan, tokenizer, generation_eos_ids=design.BASE_GENERATION_EOS_IDS)
    design.validate_preflight(plan, preflight, tokenizer=tokenizer)
    require(
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        == sha(root / TOKENIZER / "chat_template.jinja"),
        "Tokenizer template differs from pinned metadata",
    )
    setup_calls = setup_call_inventory(plan, preflight)
    require(len(setup_calls) == 21, "Unexpected fixed setup-call inventory")
    source = source_hash(root)
    manifest = dict(
        schema_version=1,
        assay=resources.ASSAY,
        revision=design.REVISION,
        campaign_id=design.REVISION,
        phase="campaign",
        scientific_admission=False,
        admitted_phases=[],
        plan=plan,
        plan_sha256=digest(plan),
        preflight=preflight,
        preflight_sha256=digest(preflight),
        setup_call_plan=setup_calls,
        setup_call_plan_sha256=digest(setup_calls),
        source_sha256=source,
        resources={phase: resources.policy(phase) for phase in resources.POLICIES},
        resource_liabilities=resources.campaign_liabilities(),
        campaign_ceiling_usd=resources.CAMPAIGN_CEILING_USD,
        setup_ceiling_usd=resources.SETUP_CEILING_USD,
        benchmark_ceiling_usd=resources.BENCHMARK_CEILING_USD,
        environment_lock=dict(path=LOCK, **environment),
        documents=documents,
        vector_receipt=vectors,
        vector_receipt_sha256=digest(vectors),
        prior_study_closure=closure,
        prior_study_closure_sha256=digest(closure),
        analysis_metadata=deepcopy(ANALYSIS_METADATA),
        analysis_metadata_sha256=digest(ANALYSIS_METADATA),
        phase_artifact_paths={phase: f"phases/{phase}" for phase in resources.POLICIES},
        phase_admission_paths={phase: f"admissions/{phase}.json" for phase in resources.POLICIES},
        asset_policy=dict(
            model_weights_in_source_archive=False,
            adapter_archive_in_source_archive=False,
            CPU_install_and_prefetch_share_wall_deadline=True,
            setup_outcomes_select_inputs=False,
        ),
        preparation_scope="Local pinned environment, tokenizer and CPU saved vectors; no model forward or cloud admission",
    )
    validate_manifest(manifest, tokenizer=tokenizer)
    # Recheck before returning; prepare rechecks again immediately before publishing.
    require(
        documents == collect_documents(root) and source == source_hash(root),
        "Source or documents changed during campaign construction",
    )
    return manifest


def assert_unprepared(root, out):
    root, out = Path(root).resolve(), Path(out).absolute()
    require(
        out == root / RUN
        and not out.exists()
        and not out.is_symlink()
        and not (root / "executions" / out.name).exists(),
        "Exactly one fixed campaign path; existing preparation cannot be replaced or resumed",
    )
    for path in (root / "runs").glob("*/manifest.json"):
        value = read(path)
        require(
            value.get("assay") != resources.ASSAY and value.get("revision") != design.REVISION,
            "Pain campaign already exists under an alias",
        )


def prepare(out=None, root=ROOT):
    root = Path(root).resolve()
    out = root / RUN if out is None else Path(out).absolute()
    assert_unprepared(root, out)
    manifest = build_manifest(root)
    assert_unprepared(root, out)
    require(
        manifest["source_sha256"] == source_hash(root) and manifest["documents"] == collect_documents(root),
        "Inputs changed before publication",
    )
    # No files have been created before this point. A later OS/freezer failure is
    # retained as an incomplete preparation, never silently resumed or overwritten.
    write_once(out / "manifest.json", manifest)
    frozen = freeze(root, out, documents=tuple(manifest["documents"]))
    return dict(
        run=str(out),
        manifest_sha256=digest(manifest),
        freeze=frozen,
        scientific_admission=False,
        cloud_launched=False,
        paid_phases_admitted=[],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / RUN)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out)))
