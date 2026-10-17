"""Prepare the once-only S31/S47 geometry setup after the preceding study closes.

No inference is launched here. All scientific choices precede measurement;
neither setup doses nor short-check outcomes can alter the fixed extension.
"""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_site_resources as resources
from curiosity_study import sustained_engineering as legacy
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.concept_capture import MODEL
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402
from satisfaction_site_sources import (  # noqa: E402
    AUDIT, CHILD, SOURCE, CANDIDATE_DOCUMENTS, historical_baseline, setup_units, sha,
)

C_RUN = "runs/qwen32-curiosity-suppression-screen-v1-admission2"
C_HASH = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
C_FILES = dict(
    manifest=C_RUN + "/manifest.json", freeze=C_RUN + "/freeze.json",
    audit="reports/qwen32-curiosity-suppression-screen-v1-audit.json",
    analysis="analyses/curiosity-suppression-screen-v1/analysis.json",
    arithmetic="reports/curiosity-suppression-screen-v1-independent-check.json",
    scoring="reports/curiosity-suppression-screen-v1-independent-scoring.json",
    shutdown="reports/qwen32-curiosity-suppression-screen-v1-admission2-cost-and-shutdown.json",
)
DESIGN = "reports/satisfaction-causal-sites-v1-design-note.md"
DESIGN_SHA = "ffedcbb5ce6a40ac462f863dd03521974773eff088b7f3784767ed71ca1fd8ae"
PLAN = "reports/satisfaction-causal-sites-v1-execution-plan.md"
RESERVED = "reports/satisfaction-causal-sites-v1-reserved-materials.json"
SOURCE_RECEIPT = "reports/satisfaction-causal-sites-v1-setup-source.json"
FORECAST = "reports/satisfaction-causal-sites-v1-resource-forecast.json"
require = study.require


def priority_gate(root):
    """Read closed evidence only; a qualifying C result defers this extension."""
    root = Path(root)
    values = {key: read(root / name) for key, name in C_FILES.items()}
    m, frozen, a, result, check, scoring, cost = (values[key] for key in C_FILES)
    require(digest(m) == frozen["manifest_sha256"] == C_HASH
            and frozen["archive_sha256"] == sha(root / C_RUN / "source.tar.gz")
            and a["parent_manifest_sha256"] == C_HASH and a["complete"] is True
            and a["integrity_passed"] is True
            and result["parent_manifest_sha256"] == C_HASH and result["audit_sha256"] == digest(a)
            and check["parent_manifest_sha256"] == C_HASH and check["passed"] is True
            and check["analysis_sha256"] == digest(result) and check["audit_sha256"] == digest(a)
            and check["analysis_byte_sha256"] == sha(root / C_FILES["analysis"])
            and check["audit_byte_sha256"] == sha(root / C_FILES["audit"])
            and scoring["parent_manifest_sha256"] == C_HASH and scoring["passed"] is True
            and scoring["assigned"] == scoring["checked"] == 4320 and scoring["complete"] is True
            and cost["all_stopped_zero_tasks"] is True,
            "Complete, audited, independently checked and stopped C collection is required first")
    require(check["decision"]["classification"] == result["decision"]["classification"]
            and check["decision"]["positive_screen"] is result["decision"]["positive_screen"],
            "Independent C decision differs")
    require(result["decision"]["positive_screen"] is False,
            "A direction-selective C result takes validation priority; defer S-site extension")
    return dict(
        passed=True, scheduling_rule="C full direction-selectivity takes priority; otherwise fixed S extension",
        preceding_decision=deepcopy(result["decision"]),
        bindings={key: dict(path=name, byte_sha256=sha(root / name), content_sha256=digest(values[key]))
                  for key, name in C_FILES.items()},
    )


def forecast(root, *, phase="setup"):
    """Use closed historical execution profiles, never a selected response length."""
    from budget import RATES, snapshot

    root = Path(root)
    require(phase in resources.POLICIES, "Unknown forecast phase")
    remaining_reserve = {"setup": 65, "calibration": 62, "validation": 28}[phase]
    state = snapshot()
    require(state["all_stopped_zero_tasks"] is True
            and state["available_compute_estimate_usd"] >= remaining_reserve,
            "Require all remaining stage ceilings and no active project app")
    profiles = []
    for suffix, cohorts in (("baseline-zero", 18), ("positive-half", 9), ("positive-full", 9)):
        paths = list((root / f"{SOURCE}-{suffix}" / "cloud_profiles").glob("*.json"))
        require(len(paths) == 1, "Require one complete historical runtime per worker")
        value = read(paths[0])
        require(type(value["assigned"]) is int and type(value["completed"]) is int
                and value["assigned"] == value["completed"] == cohorts * 32
                and all(type(value[key]) in (int, float) and math.isfinite(value[key])
                        for key in ("execution_seconds", "load_seconds"))
                and value["execution_seconds"] > 0 and value["load_seconds"] >= 0,
                "Historical profile is incomplete")
        profiles.append(dict(path=str(paths[0].relative_to(root)), byte_sha256=sha(paths[0]),
                             profile=value, source_cohorts=cohorts))
    costs = {phase: policy["workers"] * (policy["shared_deadline_seconds"] + 120) * RATES["H200"]
             for phase, policy in resources.POLICIES.items()}
    require(all(costs[p] <= resources.POLICIES[p]["stage_ceiling_usd"] for p in costs)
            and sum(p["stage_ceiling_usd"] for p in resources.POLICIES.values()) == 65,
            "Stage liabilities exceed the fixed campaign bound")
    ranges = {}
    for stage, cohorts in (("calibration", 42), ("validation", 30)):
        seconds = [row["profile"]["execution_seconds"] / row["source_cohorts"] * cohorts
                   + row["profile"]["load_seconds"] + 120 for row in profiles]
        ranges[stage] = dict(cohorts_per_worker=cohorts, estimated_seconds_range=[min(seconds), max(seconds)],
                             estimated_compute_usd_range=[3 * min(seconds) * RATES["H200"],
                                                          3 * max(seconds) * RATES["H200"]])
    return dict(
        schema_version=1, budget_snapshot=state, source_profiles=profiles, scientific_stage_estimates=ranges,
        admission_phase=phase, remaining_stage_reserve_usd=remaining_reserve,
        setup_allowance_seconds=1800, setup_assigned_reference_rows=288, setup_engineering_generated_rows=256,
        setup_cap=128, policies=deepcopy(resources.POLICIES), maximum_reserved_compute_usd=costs,
        campaign_ceiling_usd=65, rate_usd_per_second=RATES["H200"], paid_retries=0,
        completion_guaranteed=False,
        caveat="Historical complete-cohort scaling includes old receipt work. New layers/directions can change response lengths and exceed the deadline. Setup has an1800s fixed bound, not a measured completion forecast. App-wall estimates are not invoices.",
    )


def source_documents(root, provenance):
    root = Path(root)
    names = {
        *CANDIDATE_DOCUMENTS.values(), SOURCE + "/manifest.json", SOURCE + "/freeze.json",
        SOURCE + "/source.tar.gz", CHILD + "/manifest.json", AUDIT,
        "analyses/satisfaction-screen-v1/analysis.json", "reports/satisfaction-screen-v1-independent-check.json",
        "reports/satisfaction-screen-v1-independent-scoring.json",
        *C_FILES.values(), C_RUN + "/source.tar.gz", DESIGN, PLAN, RESERVED, SOURCE_RECEIPT, FORECAST,
    }
    names.update(str(path.relative_to(root)) for path in (root / CHILD / "cohorts").glob("*/*.json"))
    for item in provenance["artifacts"]:
        names.add(item["cohort_path"])
        names.add(CHILD + "/" + item["generation_evidence"]["path"])
        names.add(CHILD + "/" + item["activation_artifact"]["path"])
    return sorted(names)


def prepare(out, root=ROOT):
    from curiosity_study import satisfaction_site_setup as setup
    from prepare_reserved_object_inventory import audit_history, build_reserved_plan

    root, out = Path(root).resolve(), Path(out).resolve()
    require(out.parent == root / "runs" and not out.exists()
            and not (root / "executions" / out.name).exists()
            and not any((root / name).exists() for name in (SOURCE_RECEIPT, FORECAST)),
            "Setup is immutable and may be prepared only once")
    require(sha(root / DESIGN) == DESIGN_SHA and (root / PLAN).is_file(),
            "Bind the unchanged prospective design and completed executable plan")
    require(all((root / name).is_file() for name in (
        "scripts/analyze_satisfaction_causal_sites.py", "scripts/audit_satisfaction_causal_sites.py",
        "scripts/audit_satisfaction_site_setup.py")), "Full extension analysis and audit must exist before setup")
    require(not any(read(p).get("assay") in (resources.SETUP_ASSAY, study.ASSAY)
                    for p in (root / "runs").glob("*/manifest.json")),
            "The finite site extension already has a prepared study")
    priority = priority_gate(root)
    reserved = read(root / RESERVED)
    history = audit_history(root, active_parent_paths=())
    expected = build_reserved_plan(history)
    require(reserved["history"] == history and reserved["history_sha256"] == digest(history)
            and all(reserved[key] == value for key, value in expected.items()
                    if key not in ("schema_version", "scientific_admission")),
            "Reserved material inventory no longer reproduces before any site measurement")
    parent, requests, hidden, historical_binding, provenance = historical_baseline(root, reproduce=True)
    units, candidate = setup_units(root)
    estimate = forecast(root)
    receipt = dict(historical_binding=historical_binding, provenance=provenance,
                   source_units=units, candidate_source=candidate, priority=priority,
                   reserved_materials_sha256=digest(reserved), source_screen_manifest_sha256=digest(parent))
    rows = setup.assignment_plan(requests)
    manifest = dict(
        schema_version=1, assay=resources.SETUP_ASSAY, phase="engineering", scientific_admission=False,
        full_cap_certified=False, model=deepcopy(MODEL), execution=deepcopy(study.EXECUTION),
        execution_gpu="H200", arithmetic=deepcopy(legacy.ARITHMETIC), schedule=study.SCHEDULE,
        sites=list(study.SITES), cap=128, original_dose=study.ORIGINAL_S_DOSE,
        source_sha256=source_hash(), engine_sha256=setup.engine_hashes(),
        requests=requests, requests_sha256=digest(requests), historical_binding=historical_binding,
        historical_binding_sha256=digest(historical_binding), historical_hidden=setup.encode_historical(hidden),
        source_units=units, source_units_sha256=digest(units),
        source_certificate=receipt, source_certificate_sha256=digest(receipt),
        assignments=[], diagnostics=rows, assignment_sha256=digest(rows),
        budget=deepcopy(resources.POLICIES["setup"]), campaign_id=study.REVISION,
        reserved_materials_sha256=digest(reserved), invalid_output_policy="Terminal no retries or replacement",
    )
    setup.validate_manifest(manifest)
    write_once(root / SOURCE_RECEIPT, receipt)
    write_once(root / FORECAST, estimate)
    documents = source_documents(root, provenance)
    manifest["document_bindings"] = [dict(path=name, byte_sha256=sha(root / name)) for name in documents]
    setup.validate_manifest(manifest)
    write_once(out / "manifest.json", manifest)
    frozen = freeze(root, out, documents=documents)
    return dict(run=str(out), manifest_sha256=digest(manifest), freeze=frozen,
                reserved_materials_sha256=digest(reserved), gpu_launched=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.out)))
