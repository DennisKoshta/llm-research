"""Reconstruct historical baseline and original A directions without inference.

This module uses only the closed satisfaction/candidate studies. It neither
reads current curiosity outcomes nor admits any new paid experiment.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_capture as previous
from curiosity_study import causal_site_engineering as engineering
from curiosity_study.artifacts import digest, read

from collection_coverage import check_collection_coverage
from prepare_curiosity_suppression_screen import source_certificate
from prepare_sustained_engineering import CANDIDATE_DOCUMENTS
from prepare_satisfaction_screen import LOCKED_CANDIDATE_MANIFEST
from prepare_voluntary_resolution import candidate_binding

SOURCE = "runs/qwen32-satisfaction-screen-v1"
CHILD = SOURCE + "-baseline-zero"
AUDIT = "reports/qwen32-satisfaction-screen-v1-audit.json"
require = study.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_baseline(root, *, reproduce=False):
    """Bind every old baseline row/artifact to the complete archived audit."""
    root = Path(root)
    parent, certificate = source_certificate(root, reproduce=reproduce)
    child_path = root / CHILD
    child, audit = read(child_path / "manifest.json"), read(root / AUDIT)
    require(child["parent_manifest_sha256"] == digest(parent), "Baseline child has another parent")
    coverage = check_collection_coverage(child_path, expected_manifest_sha256=digest(child))
    base, _ = study.calibration_inputs(parent)
    requests = [deepcopy(item["request"]) for item in parent["generation_preflight"]["requests"]]
    require(len(requests) == 288 and digest(base) == study.CALIBRATION_BASE_SHA256,
            "Require all original ordered288 branches")
    audited = {row["trial_id"]: row["record_sha256"] for row in audit["outcomes"]}
    selected, artifact_bindings = {}, []
    for path in sorted((child_path / "cohorts").glob("*/*.json")):
        envelope = read(path)
        rows = envelope["records"]
        if rows[0]["assignment"]["arm"] != "baseline":
            continue
        offset = int(path.stem)
        assigned = child["diagnostics"][offset:offset + 32]
        require(len(rows) == 32 and all(r["assignment"]["arm"] == "baseline" for r in rows)
                and [r["assignment"] for r in rows] == assigned
                and all(audited[r["trial_id"]] == digest(r) for r in rows),
                "Historical baseline records differ from their complete audit")
        evidence = previous._load_generation(child_path, child, assigned,
                                              {r["trial_id"]: r for r in rows}, offset)
        first = assigned[0]["base_row"]
        chunk = requests[first:first + 32]
        require(first % 32 == 0 and [a["base_row"] for a in assigned] == list(range(first, first + 32))
                and evidence["requests"] == chunk, "Historical baseline companions or prefixes changed")
        hidden = previous._verify_hidden(child_path, evidence["activation_artifact"],
                                         evidence["trial_ids"], chunk)
        require(hidden.dtype == np.float32 and hidden.shape == (32, 1, 2, 5120)
                and np.isfinite(hidden).all(), "Historical baseline vectors unavailable")
        artifact_bindings.append(dict(
            base_offset=first, child_offset=offset, cohort_path=str(path.relative_to(root)),
            cohort_byte_sha256=sha(path), cohort_sha256=digest(envelope),
            generation_evidence=deepcopy(rows[0]["generation_evidence"]),
            activation_artifact=deepcopy(evidence["activation_artifact"]),
        ))
        for index, record in enumerate(rows):
            number = record["assignment"]["base_row"]
            require(number not in selected, "Duplicated historical baseline row")
            selected[number] = (record, hidden[index])
    require(set(selected) == set(range(288)) and len(artifact_bindings) == 9,
            "Incomplete original baseline inventory")
    hidden = np.stack([selected[i][1] for i in range(288)])
    record_bindings = [dict(
        base_row=i, base_trial_id=base[i]["base_trial_id"], trial_id=selected[i][0]["trial_id"],
        record_sha256=digest(selected[i][0]), source_assignment_sha256=base[i]["source_assignment_sha256"],
        request_sha256=digest(requests[i]),
    ) for i in range(288)]
    artifact_bindings.sort(key=lambda value: value["base_offset"])
    historical_binding = dict(
        source_manifest_sha256=digest(parent), source_audit_sha256=digest(audit),
        record_bindings_sha256=digest(record_bindings), artifact_bindings_sha256=digest(artifact_bindings),
        requests_sha256=digest(requests), hidden=engineering.array_binding(hidden),
    )
    provenance = dict(
        source_certificate=certificate, source_child_manifest_sha256=digest(child),
        source_child=CHILD, source_audit_path=AUDIT, source_audit_byte_sha256=sha(root / AUDIT),
        operational_coverage=coverage, records=record_bindings, artifacts=artifact_bindings,
        original_generation_preflight_sha256=digest(parent["generation_preflight"]),
        selection="All288 baseline branches in original source order; no response-based inclusion",
    )
    return parent, requests, hidden, historical_binding, provenance


def locked_directions(root, *, doses):
    """Return all15-arm original A bindings for both fixed sites, without fitting."""
    root = Path(root)
    require(set(doses) == {"31", "47"}, "Both prospectively fixed sites are required")
    paths = {key: root / name for key, name in CANDIDATE_DOCUMENTS.items()}
    values = {key: read(path) for key, path in paths.items() if key != "directions"}
    direction_sha = sha(paths["directions"])
    require(digest(values["manifest"]) == LOCKED_CANDIDATE_MANIFEST
            and direction_sha == study.LOCKED_DIRECTION_BYTES, "Original candidate bytes changed")
    candidate = candidate_binding(values["manifest"], values["audit"], values["analysis"],
                                  values["check"], sha(paths["analysis"]), direction_sha)
    all_sites = {}
    with np.load(paths["directions"], allow_pickle=False) as arrays:
        require(arrays["layers"].tolist() == [15, 31, 47, 57]
                and arrays["positions"].tolist() == ["record", "final"], "Candidate axes changed")
        sources = arrays["sources"].tolist()
        require(float(arrays["contrast_norm"][sources.index("satisfaction_vs_neutral"), 0, 0])
                == study.ORIGINAL_S_DOSE, "Original S15 reference norm changed")
        for layer in study.SITES:
            li, bindings, dose = arrays["layers"].tolist().index(layer), {}, doses[str(layer)]
            semantic = (("zero_sham", "satisfaction_vs_neutral", 1),
                        (study.PRIMARY, "satisfaction_vs_neutral", 1), *study.controls.NEIGHBORS)
            for arm, source, orientation in semantic:
                index = sources.index(source)
                raw_unit = arrays["unit"][index, li, 0].astype(np.float64)
                unit = (raw_unit * orientation).tolist()
                bindings[arm] = dict(
                    source=source, corpus="A", layer=layer, position="record", orientation=orientation,
                    source_raw_norm=float(arrays["contrast_norm"][index, li, 0]),
                    source_unit_sha256=digest(raw_unit.tolist()), unit=unit, unit_sha256=digest(unit),
                    dose=0.0 if arm == "zero_sham" else dose, directions_byte_sha256=direction_sha,
                )
            for arm, seed in zip(study.controls.RANDOM_ARMS, study.controls.RANDOM_SEEDS, strict=True):
                raw = np.random.default_rng(seed).standard_normal(5120)
                unit = (raw / np.linalg.norm(raw)).tolist()
                bindings[arm] = dict(
                    source="seeded_gaussian", corpus=None, layer=layer, position="record", orientation=1,
                    seed=seed, construction="numpy_default_rng_standard_normal_float64_unit_no_resampling",
                    generator="PCG64", numpy_version=np.__version__, original_draw_norm=float(np.linalg.norm(raw)),
                    original_draw_byte_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    unit=unit, unit_sha256=digest(unit), dose=dose,
                )
            all_sites[str(layer)] = study.validate_directions(
                bindings, site=layer, arms=study.VALIDATION_ARMS, dose=dose)
    return all_sites, candidate


def setup_units(root):
    """Dose-free S31/S47 sources for the read-only reference and short checks."""
    bindings, candidate = locked_directions(root, doses={"31": 1.0, "47": 1.0})
    units = {str(layer): deepcopy(bindings[str(layer)][study.PRIMARY]) for layer in study.SITES}
    for binding in units.values():
        del binding["dose"]
    return units, candidate
