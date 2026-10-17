"""Locked descriptive optional-choice analysis; no fitting or causal admission.

Reads only a complete, integrity-passed audit and its unchanged artifacts.
Scientific readiness may fail: that failure remains explicit in every result.
"""

import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.concept_capture import CONTRASTS
from audit_voluntary_resolution import REQUIRED_ANALYSIS_METADATA

FAMILIES = voluntary.FAMILIES
DELIVERIES = voluntary.DELIVERIES
CATEGORIES = (*voluntary.SERVICES, "invalid", "execution_failure", "missing")
POSITIONS = ("record", "prefill")
COMPARISONS = (
    ("decisive_minus_redundant", "decisive_evidence", "redundant_evidence", "primary"),
    ("decisive_minus_novelty", "decisive_evidence", "independent_novelty", "secondary"),
)
BOOTSTRAPS, SEED = 10000, 2026101434
DIRECTION_DOCUMENT = "analyses/curiosity-satisfaction-candidates-v1/directions.npz"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_axes(arrays):
    """All eight raw A15/record axes; final-position arrays are never selected."""
    require(list(arrays["sources"]) == list(CONTRASTS), "Require both targets and all six nuisance axes")
    layers, positions = list(arrays["layers"]), list(arrays["positions"])
    require(layers == [15, 31, 47, 57] and positions == ["record", "final"], "Direction coordinates changed")
    li, pi = layers.index(15), positions.index("record")
    result = []
    for index, name in enumerate(CONTRASTS):
        unit = np.asarray(arrays["unit"][index, li, pi], dtype=np.float64)
        center = np.asarray(arrays["center"][index, li, pi], dtype=np.float64)
        norm = float(arrays["contrast_norm"][index, li, pi])
        threshold = float(arrays["threshold"][index, li, pi])
        require(unit.ndim == center.ndim == 1 and unit.shape == center.shape and unit.size, "Malformed axis")
        available = bool(
            np.isfinite(unit).all()
            and np.isfinite(center).all()
            and np.isfinite(norm)
            and norm > 0
            and np.linalg.norm(unit) > 0
        )
        if available:
            require(np.isclose(np.linalg.norm(unit), 1, atol=1e-8), "Frozen axis is not a unit vector")
            require(
                np.isclose(center @ unit, threshold, rtol=1e-10, atol=1e-8),
                "Frozen center/threshold disagree",
            )
        result.append(
            dict(name=name, unit=unit, center=center, norm=norm, threshold=threshold, available=available)
        )
    return result


def project(vector, axis):
    """Same frozen center/unit/full norm at both source sites and receipt sites."""
    if not axis["available"] or vector is None:
        return None, None
    vector = np.asarray(vector, dtype=np.float64)
    require(vector.shape == axis["unit"].shape, "Capture and frozen direction widths differ")
    if not np.isfinite(vector).all():
        return None, None
    score = float((vector - axis["center"]) @ axis["unit"])
    return score, score / axis["norm"]


def axis_metadata(axis):
    result = {key: value for key, value in axis.items() if key not in ("unit", "center")}
    for key in ("norm", "threshold"):
        if not np.isfinite(result[key]):
            result[key] = None
    result.update(
        source_corpus="A",
        source_layer=15,
        source_position="record",
        role="target" if axis["name"] in CONTRASTS[:2] else "nuisance_control",
        unit_sha256=digest(axis["unit"].tolist()),
        center_sha256=digest(axis["center"].tolist()),
        score="(capture - A center) dot A unit",
        scale="divide by full A raw contrast norm",
        threshold_role="Preserved provenance only; never a feeling/state classifier",
    )
    return result


def category(record):
    if record is None:
        return "missing"
    if "failure" in record:
        return "execution_failure"
    if record.get("valid") is not True or record.get("thinking_closed") is not True:
        require(record.get("selected_service") is None, "Invalid choices cannot execute parsed services")
        return "invalid"
    require(record.get("selected_service") in voluntary.SERVICES, "Valid choice lacks an actual service")
    return record["selected_service"]


def choice_index(choices):
    require(
        len(choices) == 144 and len({r["trial_id"] for r in choices}) == 144, "Require all144 unique choices"
    )
    grouped = defaultdict(dict)
    for row in choices:
        require(
            row["family"] in FAMILIES and row["category"] in CATEGORIES, "Unknown family or choice category"
        )
        require(row["delivery"] in DELIVERIES and row["menu_layout"] in (0, 1), "Unknown delivery/menu")
        key = (row["family"], row["semantic_world_id"])
        cell = (row["delivery"], row["menu_layout"])
        require(cell not in grouped[key], "Duplicate world/delivery/menu cell")
        grouped[key][cell] = row
    expected = {(delivery, menu) for delivery in DELIVERIES for menu in (0, 1)}
    for cells in grouped.values():
        require(set(cells) == expected, "Each semantic world needs all three deliveries and both menus")
        require(len({r["world_slot"] for r in cells.values()}) == 1, "Semantic world merges distinct slots")
    worlds = {family: sorted(key[1] for key in grouped if key[0] == family) for family in FAMILIES}
    require(all(len(values) == 12 for values in worlds.values()), "Need12 unique semantic worlds per family")
    require(
        all(
            len({grouped[(family, w)][next(iter(expected))]["world_slot"] for w in worlds[family]}) == 12
            for family in FAMILIES
        ),
        "Slots do not identify unique worlds",
    )
    return grouped, worlds


def bootstrap_bank(worlds):
    rng = np.random.default_rng(SEED)
    return {
        family: rng.integers(0, len(worlds[family]), size=(BOOTSTRAPS, len(worlds[family])))
        for family in FAMILIES
    }


def paired_endpoints(values, banks):
    """No success-only subset: any missing world makes its endpoint unavailable."""
    result, samples = [], {}
    for family in FAMILIES:
        vector = np.asarray(values[family], dtype=float)
        available = int(np.isfinite(vector).sum())
        if available == len(vector):
            estimate = float(vector.mean())
            samples[family] = vector[banks[family]].mean(axis=1)
            interval = np.quantile(samples[family], [0.025, 0.975]).tolist()
        else:
            estimate = interval = None
        result.append(
            dict(
                family=family,
                estimate=estimate,
                ci95=interval,
                assigned_worlds=len(vector),
                available_worlds=available,
            )
        )
    available = sum(r["available_worlds"] for r in result)
    if len(samples) == len(FAMILIES):
        pooled = np.mean([samples[family] for family in FAMILIES], axis=0)
        estimate = float(np.mean([r["estimate"] for r in result]))
        interval = np.quantile(pooled, [0.025, 0.975]).tolist()
    else:
        estimate = interval = None
    result.append(
        dict(
            family="pooled",
            estimate=estimate,
            ci95=interval,
            assigned_worlds=sum(len(v) for v in values.values()),
            available_worlds=available,
        )
    )
    return result


def analyze_tables(choices, projections):
    choices = sorted(
        choices,
        key=lambda row: (
            FAMILIES.index(row["family"]) if row["family"] in FAMILIES else len(FAMILIES),
            row["semantic_world_id"],
            row["delivery"],
            row["menu_layout"],
            row["trial_id"],
        ),
    )
    grouped, worlds = choice_index(choices)
    banks = bootstrap_bank(worlds)
    projection_map = {}
    for row in projections:
        key = row["trial_id"], row["axis"], row["position"]
        require(key not in projection_map, "Duplicate projection row")
        projection_map[key] = row
    expected = {
        (r["trial_id"], axis, position) for r in choices for axis in CONTRASTS for position in POSITIONS
    }
    require(set(projection_map) == expected, "Export every assigned choice, axis and source position")
    behavior, scores = [], []
    for name, left, right, role in COMPARISONS:
        for action in CATEGORIES:
            values = {}
            for family in FAMILIES:
                values[family] = [
                    np.mean([grouped[(family, world)][left, menu]["category"] == action for menu in (0, 1)])
                    - np.mean(
                        [grouped[(family, world)][right, menu]["category"] == action for menu in (0, 1)]
                    )
                    for world in worlds[family]
                ]
            behavior.extend(
                dict(contrast=name, role=role, category=action, **row)
                for row in paired_endpoints(values, banks)
            )
        for axis in CONTRASTS:
            for position in POSITIONS:
                for scale in ("raw", "scaled"):
                    values = {}
                    for family in FAMILIES:
                        deltas = []
                        for world in worlds[family]:
                            pair = []
                            for delivery in (left, right):
                                vals = [
                                    projection_map[
                                        grouped[(family, world)][delivery, menu]["trial_id"], axis, position
                                    ][scale]
                                    for menu in (0, 1)
                                ]
                                pair.append(
                                    float(np.mean(vals))
                                    if all(v is not None and np.isfinite(v) for v in vals)
                                    else np.nan
                                )
                            deltas.append(pair[0] - pair[1])
                        values[family] = deltas
                    scores.extend(
                        dict(contrast=name, role=role, axis=axis, position=position, scale=scale, **row)
                        for row in paired_endpoints(values, banks)
                    )
    denominators, score_cells = [], []
    for family in (*FAMILIES, "pooled"):
        for delivery in DELIVERIES:
            for menu in (0, 1, "averaged"):
                rows = [
                    r
                    for r in choices
                    if (family == "pooled" or r["family"] == family)
                    and r["delivery"] == delivery
                    and (menu == "averaged" or r["menu_layout"] == menu)
                ]
                counts = Counter(r["category"] for r in rows)
                base = dict(
                    family=family,
                    delivery=delivery,
                    menu_layout=menu,
                    assigned_choices=len(rows),
                    assigned_worlds=len({(r["family"], r["semantic_world_id"]) for r in rows}),
                    cap_hits=sum(r.get("cap_hit") is True for r in rows),
                    unclosed=sum(r.get("thinking_closed") is False for r in rows),
                )
                denominators.append(
                    dict(
                        **base,
                        counts={action: counts[action] for action in CATEGORIES},
                        proportions={action: counts[action] / len(rows) for action in CATEGORIES},
                    )
                )
                for axis in CONTRASTS:
                    for position in POSITIONS:
                        selected = [projection_map[r["trial_id"], axis, position] for r in rows]
                        available = [
                            r
                            for r in selected
                            if all(r[key] is not None and np.isfinite(r[key]) for key in ("raw", "scaled"))
                        ]
                        score_cells.append(
                            dict(
                                **base,
                                axis=axis,
                                position=position,
                                available_projections=len(available),
                                mean_centered_raw=float(np.mean([r["raw"] for r in available]))
                                if len(available) == len(rows)
                                else None,
                                mean_scaled=float(np.mean([r["scaled"] for r in available]))
                                if len(available) == len(rows)
                                else None,
                            )
                        )
    return dict(
        behavior_contrasts=behavior,
        projection_contrasts=scores,
        denominators=denominators,
        projection_cells=score_cells,
        bootstrap=dict(
            draws=BOOTSTRAPS,
            seed=SEED,
            family_order=list(FAMILIES),
            world_order=worlds,
            draw_bank_sha256=digest({k: v.tolist() for k, v in banks.items()}),
            unit="Complete semantic world; all three deliveries and both menus retained",
            pooled="Equal mean of the two family means within each draw",
            interval="Descriptive95% percentile; no confirmatory multiplicity/significance claim",
        ),
    )


def safe_artifact(root, metadata, cache, trial_id):
    relative = Path(metadata["path"])
    path = (root / relative).resolve()
    require(
        not relative.is_absolute() and path.is_relative_to(root.resolve()), "Capture artifact escapes its run"
    )
    if path not in cache:
        require(sha(path) == metadata["sha256"], "Capture artifact byte hash differs")
        with np.load(path, allow_pickle=False) as loaded:
            cache[path] = metadata["sha256"], {key: loaded[key] for key in loaded.files}
    file_sha, arrays = cache[path]
    row = metadata["row"]
    require(
        file_sha == metadata["sha256"]
        and type(row) is int
        and 0 <= row < len(arrays["trial_ids"])
        and arrays["trial_ids"][row] == trial_id,
        "Capture artifact row/source binding differs",
    )
    require(
        arrays["layers"].tolist() == metadata["layers"]
        and arrays["positions"].tolist() == metadata["positions"],
        "Capture coordinates changed",
    )
    require(
        arrays["available"][row].tolist() == metadata["available"]
        and arrays["token_positions"][row].tolist() == metadata["token_positions"],
        "Capture availability/token positions changed",
    )
    return arrays


def bind_inputs(run, audit_path, directions_path):
    run = Path(run).resolve()
    manifest, audit = read(run / "manifest.json"), read(audit_path)
    require(
        audit.get("assay") == manifest.get("assay") == "voluntary_resolution", "Wrong collection/audit assay"
    )
    require(
        audit.get("complete") is True and audit.get("integrity_passed") is True,
        "Need complete integrity-passed audit; readiness may remain failed",
    )
    require(
        audit["manifest_sha256"] == digest(manifest)
        and audit["source_sha256"] == manifest["source_sha256"]
        and audit["model"] == manifest["model"],
        "Audit does not bind source manifest/model",
    )
    require(
        manifest["analysis_metadata"] == audit["analysis_metadata"] == REQUIRED_ANALYSIS_METADATA,
        "Analysis rules differ from the frozen plan",
    )
    require(
        type(audit["readiness"]["passed"]) is bool and audit["passed"] == audit["readiness"]["passed"],
        "Readiness status is inconsistent",
    )
    require(
        audit["causal_admission"] is False and audit["choice_correctness"] is None,
        "Choices cannot acquire accuracy or causal admission",
    )
    freeze = read(run / "freeze.json")
    require(
        freeze["manifest_sha256"] == digest(manifest)
        and freeze["source_sha256"] == manifest["source_sha256"]
        and freeze["archive_sha256"] == sha(run / "source.tar.gz")
        and audit["source_freeze"]["freeze_sha256"] == digest(freeze),
        "Frozen archive differs from audited source",
    )
    directions_sha = sha(directions_path)
    require(
        directions_sha
        == manifest["candidate_source"]["directions_byte_sha256"]
        == audit["candidate_source"]["directions_byte_sha256"],
        "Frozen A direction bytes differ",
    )
    require(
        audit["candidate_source"] == manifest["candidate_source"], "Candidate metadata changed after audit"
    )
    require(
        manifest["candidate_source"]["selected_sites"]
        == {name: dict(layer=15, position="record") for name in CONTRASTS[:2]},
        "Candidate sites changed",
    )
    with tarfile.open(run / "source.tar.gz") as archive:
        require(
            hashlib.sha256(archive.extractfile(DIRECTION_DOCUMENT).read()).hexdigest() == directions_sha,
            "Directions differ from the prospective archive",
        )
        require(
            hashlib.sha256(archive.extractfile("scripts/audit_voluntary_resolution.py").read()).hexdigest()
            == audit["auditor_byte_sha256"],
            "Audit implementation differs from the source freeze",
        )
        for binding in manifest["document_bindings"]:
            require(
                hashlib.sha256(archive.extractfile(binding["path"]).read()).hexdigest()
                == binding["byte_sha256"]
                == audit["source_freeze"]["documents"][binding["path"]],
                "Frozen plan/provenance document differs",
            )
    assignments = manifest["diagnostics"]
    require(
        manifest["assignments"] == []
        and len(assignments) == 288
        and digest(assignments) == manifest["assignment_sha256"],
        "Assignment inventory changed",
    )
    records = RunStore(run, manifest).completed()
    require(
        set(records) == {a["trial_id"] for a in assignments} and len(audit["outcomes"]) == 288,
        "Require all288 audited source records",
    )
    audited = {row["trial_id"]: row for row in audit["outcomes"]}
    require(set(audited) == set(records), "Audit row coverage differs")
    cache, bindings, choices, vectors, availability = {}, [], [], {}, Counter()
    for index, assignment in enumerate(assignments):
        tid = assignment["trial_id"]
        record, bound = records[tid], audited[tid]
        require(
            digest(record["assignment"]) == digest(assignment)
            and "failure" not in record
            and digest(record) == bound["record_sha256"],
            "Source record/assignment changed after audit",
        )
        require(
            bound["status"] == "completed"
            and all(
                bound[k] == assignment[k]
                for k in ("family", "kind", "delivery", "menu_layout", "semantic_world_id", "world_slot")
            ),
            "Audited assignment context differs",
        )
        for key in ("valid", "correct", "thinking_closed", "selected_service", "cap_hit"):
            require(digest(bound[key]) == digest(record[key]), "Audited choice/validity changed")
        metadata = record["activation_artifact"]
        require(
            metadata == bound["activation_artifact"] and metadata["row"] == index % 32,
            "Source capture row or metadata differs",
        )
        arrays = safe_artifact(run, metadata, cache, tid)
        require(
            arrays["trial_ids"].tolist()
            == [a["trial_id"] for a in assignments[index // 32 * 32 : index // 32 * 32 + 32]],
            "Source cohort membership/order changed",
        )
        bindings.append(
            dict(
                trial_id=tid,
                record_sha256=digest(record),
                artifact_sha256=metadata["sha256"],
                row=metadata["row"],
                token_prefix_sha256=digest(record["inference"]["unpadded_input_ids"]),
            )
        )
        for pi, position in enumerate(arrays["positions"].tolist()):
            for li, layer in enumerate(arrays["layers"].tolist()):
                availability[
                    (
                        assignment["family"],
                        assignment["kind"],
                        layer,
                        position,
                        bool(arrays["available"][metadata["row"], li, pi]),
                    )
                ] += 1
        if assignment["kind"] == "choice":
            require(record["correct"] is None, "Choice correctness must remain undefined")
            choice = {
                key: assignment[key]
                for key in (
                    "trial_id",
                    "family",
                    "kind",
                    "delivery",
                    "menu_layout",
                    "seed",
                    "index",
                    "world_slot",
                    "semantic_world_id",
                )
            }
            choice.update(
                category=category(record),
                cap_hit=record["cap_hit"],
                thinking_closed=record["thinking_closed"],
                source_record_sha256=digest(record),
                source_artifact_sha256=metadata["sha256"],
            )
            choices.append(choice)
            li = arrays["layers"].tolist().index(15)
            for position in POSITIONS:
                pi = arrays["positions"].tolist().index(position)
                vector = arrays["hidden"][metadata["row"], li, pi]
                require(
                    arrays["available"][metadata["row"], li, pi] and np.isfinite(vector).all(),
                    "Required source prechoice capture unavailable",
                )
                vectors[tid, position] = vector.astype(np.float64)
    require(
        digest(bindings) == audit["generation_record_bindings_sha256"],
        "Source record/capture binding ledger differs from audit",
    )
    choice_index(choices)
    with np.load(directions_path, allow_pickle=False) as loaded:
        axes = selected_axes({key: loaded[key] for key in loaded.files})
    sidecars, receipt_vectors, receipt_bindings, cohort_bindings = {}, {}, [], []
    for path in sorted((run / "receipt_cohorts").glob("*/*.json")):
        envelope = read(path)
        cohort_bindings.append(dict(path=str(path.relative_to(run)), envelope_sha256=digest(envelope)))
        require(
            envelope["manifest_sha256"] == digest(manifest)
            and envelope["records_sha256"] == digest(envelope["records"]),
            "Receipt envelope differs",
        )
        for row in envelope["records"]:
            tid = row["source_trial_id"]
            require(
                tid not in sidecars and tid in {r["trial_id"] for r in choices},
                "Duplicate/unassigned receipt disposition",
            )
            require(
                row["source_record_sha256"] == digest(records[tid]), "Receipt binds another generated choice"
            )
            sidecars[tid] = row
            if row["status"] != "completed":
                require(
                    row["status"] in ("stop", "invalid_choice")
                    and row["receipt_required"] is False
                    and row["receipt_model_pass_completed"] is False,
                    "Required receipt is incomplete",
                )
                continue
            require(
                row["receipt_required"] is True and row["receipt_model_pass_completed"] is True,
                "Completed receipt lacks actual model pass",
            )
            metadata = row["activation_artifact"]
            arrays = safe_artifact(run, metadata, cache, tid)
            receipt_bindings.append(
                dict(
                    source_trial_id=tid,
                    sidecar_sha256=digest(row),
                    source_record_sha256=digest(records[tid]),
                    artifact_sha256=metadata["sha256"],
                    row=metadata["row"],
                    token_prefix_sha256=row["token_binding"]["token_prefix_sha256"],
                )
            )
            li = arrays["layers"].tolist().index(15)
            for position in ("record", "final"):
                pi = arrays["positions"].tolist().index(position)
                vector = arrays["hidden"][metadata["row"], li, pi]
                require(
                    arrays["available"][metadata["row"], li, pi] and np.isfinite(vector).all(),
                    "Required receipt capture unavailable",
                )
                receipt_vectors[tid, position] = vector.astype(np.float64)
    require(set(sidecars) == {r["trial_id"] for r in choices}, "Need all144 audited receipt dispositions")
    require(
        cohort_bindings == audit["receipt_bindings"]["cohorts"]
        and receipt_bindings == audit["receipt_bindings"]["record_bindings"]
        and digest(receipt_bindings) == audit["receipt_bindings"]["record_bindings_sha256"],
        "Receipt sidecar/capture bindings differ from audit",
    )
    availability_rows = [
        dict(family=k[0], kind=k[1], layer=k[2], position=k[3], available=k[4], count=v)
        for k, v in sorted(availability.items())
    ]
    return dict(
        manifest=manifest,
        audit=audit,
        choices=choices,
        axes=axes,
        vectors=vectors,
        sidecars=sidecars,
        receipt_vectors=receipt_vectors,
        generation_bindings=bindings,
        receipt_bindings=receipt_bindings,
        position_availability=availability_rows,
        capture_artifacts=[
            dict(path=str(path.relative_to(run)), byte_sha256=value[0])
            for path, value in sorted(cache.items())
        ],
    )


def projection_tables(inputs):
    projections, receipts = [], []
    for choice in inputs["choices"]:
        tid = choice["trial_id"]
        sidecar = inputs["sidecars"][tid]
        for axis in inputs["axes"]:
            for position in POSITIONS:
                raw, scaled = project(inputs["vectors"][tid, position], axis)
                base = {
                    key: choice[key]
                    for key in (
                        "trial_id",
                        "family",
                        "semantic_world_id",
                        "world_slot",
                        "delivery",
                        "menu_layout",
                        "category",
                        "source_record_sha256",
                        "source_artifact_sha256",
                    )
                }
                base.update(
                    axis=axis["name"],
                    position=position,
                    raw=raw,
                    scaled=scaled,
                    axis_available=axis["available"],
                )
                projections.append(base)
                receipt_position = "record" if position == "record" else "final"
                rr, rs = project(inputs["receipt_vectors"].get((tid, receipt_position)), axis)
                receipts.append(
                    dict(
                        **base,
                        source_position=position,
                        receipt_position=receipt_position,
                        receipt_status=sidecar["status"],
                        receipt_required=sidecar["receipt_required"],
                        receipt_model_pass_completed=sidecar["receipt_model_pass_completed"],
                        receipt_centered_raw=rr,
                        receipt_scaled=rs,
                        receipt_minus_source_raw=rr - raw if rr is not None and raw is not None else None,
                        receipt_minus_source_scaled=rs - scaled
                        if rs is not None and scaled is not None
                        else None,
                        sidecar_sha256=digest(sidecar),
                    )
                )
    return projections, receipts


def receipt_summary(rows):
    result = []
    keys = (
        "receipt_centered_raw",
        "receipt_scaled",
        "receipt_minus_source_raw",
        "receipt_minus_source_scaled",
    )
    for family in FAMILIES:
        for delivery in DELIVERIES:
            for action in CATEGORIES:
                for axis in CONTRASTS:
                    for position in POSITIONS:
                        selected = [
                            r
                            for r in rows
                            if r["family"] == family
                            and r["delivery"] == delivery
                            and r["category"] == action
                            and r["axis"] == axis
                            and r["position"] == position
                        ]
                        summary = dict(
                            family=family,
                            delivery=delivery,
                            selected_service=action,
                            axis=axis,
                            source_position=position,
                            receipt_position="record" if position == "record" else "final",
                            assigned_selected_choices=len(selected),
                            completed_receipts=sum(r["receipt_model_pass_completed"] for r in selected),
                            available_projections=sum(
                                r["receipt_centered_raw"] is not None for r in selected
                            ),
                        )
                        for key in keys:
                            values = [r[key] for r in selected if r[key] is not None]
                            summary[key] = dict(
                                n=len(values),
                                mean=float(np.mean(values)) if values else None,
                                minimum=float(min(values)) if values else None,
                                maximum=float(max(values)) if values else None,
                            )
                        result.append(summary)
    return result


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, audit_path, directions_path, out):
    out = Path(out)
    require(not out.exists(), "Analysis outputs are immutable; choose a new directory")
    inputs = bind_inputs(run, audit_path, directions_path)
    projections, receipts = projection_tables(inputs)
    results = analyze_tables(inputs["choices"], projections)
    audit, manifest = inputs["audit"], inputs["manifest"]
    result = dict(
        schema_version=1,
        assay="voluntary_resolution_descriptive_analysis",
        development_only=True,
        complete=True,
        integrity_passed=True,
        readiness_passed=audit["readiness"]["passed"],
        readiness=deepcopy(audit["readiness"]),
        causal_admission=False,
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        model=manifest["model"],
        audit_sha256=digest(audit),
        audit_byte_sha256=sha(audit_path),
        directions_byte_sha256=sha(directions_path),
        analyzer_byte_sha256=sha(__file__),
        assigned_choices=144,
        assigned_worlds=24,
        diagnostic_readiness_repaired=False,
        candidate_source=deepcopy(manifest["candidate_source"]),
        prior_exposure=deepcopy(manifest["prior_exposure"]),
        analysis_metadata=REQUIRED_ANALYSIS_METADATA,
        axes=[axis_metadata(axis) for axis in inputs["axes"]],
        generation_record_bindings=inputs["generation_bindings"],
        receipt_record_bindings=inputs["receipt_bindings"],
        capture_artifacts=inputs["capture_artifacts"],
        position_availability=inputs["position_availability"],
        receipt_status_counts=dict(Counter(row["status"] for row in inputs["sidecars"].values())),
        receipt_summaries=receipt_summary(receipts),
        receipt_estimand="Conditional descriptions by selected service only. Fresh rendering, prior reasoning, context length and outcome-dependent active batches differ; receipt-minus-source is not a randomized or causal effect. No receipt interval/admission.",
        optional_choice_association="Not estimated; delivery contrasts are not within-condition prediction.",
        interpretation="Fixed description-candidate transfer and optional behavior only. All assigned denominators retained. Shared whole-world descriptive intervals do not establish a causal regulator, unique construct, owned state, or subjective experience. Failed readiness remains failed; no effect rescues it. Object tables have prior partial exposure; no independent-holdout claim.",
        **results,
    )
    out.mkdir(parents=True, exist_ok=False)
    exports = {
        "choices.csv": inputs["choices"],
        "choice-projections.csv": projections,
        "receipt-projections.csv": receipts,
        "behavior-contrasts.csv": [
            {**row, "ci95": json.dumps(row["ci95"])} for row in results["behavior_contrasts"]
        ],
        "projection-contrasts.csv": [
            {**row, "ci95": json.dumps(row["ci95"])} for row in results["projection_contrasts"]
        ],
        "denominators.csv": [
            {
                **{k: v for k, v in row.items() if k not in ("counts", "proportions")},
                "category": action,
                "count": row["counts"][action],
                "proportion": row["proportions"][action],
            }
            for row in results["denominators"]
            for action in CATEGORIES
        ],
    }
    result["exports"] = []
    for name, rows in exports.items():
        write_csv(out / name, rows)
        result["exports"].append(dict(path=name, rows=len(rows), byte_sha256=sha(out / name)))
    write_once(out / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.audit, args.directions, args.out)
    print(
        {
            key: result[key]
            for key in ("assigned_choices", "assigned_worlds", "readiness_passed", "causal_admission")
        }
    )
