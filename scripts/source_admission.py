"""Bind source eligibility without changing a failed instrument's original audit.

The only scoped exception is the explicitly amended, post-inspection development
branch of the original 4,000-assignment instrument. It cannot admit confirmation,
projection interchange, or natural-reasoning experiments.
"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
from pathlib import Path

from curiosity_study.artifacts import digest, read


RETAINED_KINDS = ("instrument", "memory", "new", "repeat")
CRITERIA = dict(
    minimum_distinct_items_per_cell=100,
    minimum_accuracy=0.9,
    minimum_valid_format=0.95,
    cell_axes=["family", "kind", "known", "query"],
)


def file_binding(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Admission artifact paths must be absolute")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_source(manifest, audit):
    rows = manifest.get("diagnostics")
    if (
        not rows
        or digest(rows) != manifest.get("assignment_sha256")
        or len({a["trial_id"] for a in rows}) != len(rows)
        or audit.get("manifest_sha256") != digest(manifest)
        or audit.get("complete") is not True
        or audit.get("assigned") != len(rows)
    ):
        raise ValueError("Source audit must bind complete, unique parent assignments")
    if audit.get("source_sha256") != manifest.get("source_sha256"):
        raise ValueError("Source audit code binding mismatch")
    return rows


def scoped_evidence(manifest, audit):
    """Recompute every cell decision and the exact whole-kind partition."""
    rows = validate_source(manifest, audit)
    if (
        manifest.get("assay") != "thinking_information"
        or manifest.get("task_revision") != "equal-uncertainty-v1"
        or manifest.get("phase") != "validation"
        or len(rows) != 4000
        or manifest.get("validation_criteria") != CRITERIA
        or audit.get("validation_criteria") != CRITERIA
        or audit.get("gates", {}).get("passed") is not False
    ):
        raise ValueError("Scoped admission requires the failed original 4,000-assignment validation")
    expected = set(product(("objects", "machines"), (*RETAINED_KINDS, "count"), (0, 1), (0, 1)))
    grouped = defaultdict(list)
    worlds = defaultdict(set)
    for a in rows:
        if a.get("task_revision") != "equal-uncertainty-v1" or a.get("known") not in ([0], [1]):
            raise ValueError("Scoped source contains an unsupported assignment")
        key = (a["family"], a["kind"], a["known"][0], a["query"])
        grouped[key].append(a)
        worlds[(a["family"], a["seed"], a["index"], a.get("template", "table"))].add(key[1:])
    required_world = set(product((*RETAINED_KINDS, "count"), (0, 1), (0, 1)))
    if set(grouped) != expected or any(cells != required_world for cells in worlds.values()):
        raise ValueError("Scoped source must retain all kinds and all four cells for every original world")
    original_cells = audit.get("gates", {}).get("cells", [])
    cells = {}
    for c in original_cells:
        key = (c["family"], c["kind"], c["known"], c["query"])
        if key in cells or key not in grouped:
            raise ValueError("Original audit cell coverage is duplicated or unexpected")
        assignments = grouped[key]
        if any(type(c.get(k)) is not int for k in ("n", "items", "completed", "correct", "valid")):
            raise ValueError("Original audit cell counts must be integers")
        n = len(assignments)
        items = len({a["index"] for a in assignments})
        if (
            n != 100
            or items != 100
            or (c["n"], c["items"], c["completed"]) != (n, items, n)
            or not 0 <= c["correct"] <= c["valid"] <= n
        ):
            raise ValueError("Original audit counts disagree with complete source assignments")
        passed = items >= 100 and c["correct"] / n >= 0.90 and c["valid"] / n >= 0.95
        if c.get("passed") is not passed:
            raise ValueError("Original audit cell decision disagrees with unchanged thresholds")
        if c["kind"] in RETAINED_KINDS and not passed:
            raise ValueError("Every retained cell must pass the unchanged original thresholds")
        cells[key] = c
    if set(cells) != expected or all(c["passed"] for c in cells.values()):
        raise ValueError("Original audit must cover all 40 cells and retain its global failure")
    if any(audit.get(k) != sum(c[k] for c in cells.values()) for k in ("correct", "valid")):
        raise ValueError("Original audit totals disagree with its cell counts")
    retained = [a for a in rows if a["kind"] in RETAINED_KINDS]
    excluded = [a for a in rows if a["kind"] == "count"]
    if len(retained) != 3200 or len(excluded) != 800:
        raise ValueError("Scoped admission requires all 3,200 retained and all 800 count assignments")

    def partition(assignments):
        return dict(
            assigned=len(assignments),
            assignment_sha256=digest(assignments),
            trial_ids=[a["trial_id"] for a in assignments],
            cells=[cells[key] for key in sorted(cells) if key[1] in {a["kind"] for a in assignments}],
        )

    return dict(
        retained_kinds=list(RETAINED_KINDS),
        excluded_kinds=["count"],
        retained=partition(retained),
        excluded=partition(excluded),
        criteria=CRITERIA,
        original_global_gates_passed=False,
        original_gates=audit["gates"],
        assignment_policy="Whole kinds; all assigned worlds, incorrect outputs, and invalid outputs retained",
    )


def build_certificate(source_collection, source_audit, amendment, created_at_utc=None):
    source = Path(source_collection)
    audit_path = Path(source_audit)
    if not source.is_absolute() or not audit_path.is_absolute():
        raise ValueError("Certificate source paths must be absolute")
    manifest = read(source / "manifest.json")
    audit = read(audit_path)
    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat()
    if datetime.fromisoformat(created_at_utc).utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("Certificate creation time must identify UTC")
    return dict(
        schema_version=1,
        created_at_utc=created_at_utc,
        certificate_type="post_inspection_development_replay_eligibility",
        allowed_assay="replay_information",
        allowed_inference_scope="development",
        confirmation_eligible=False,
        source_collection=str(source.resolve()),
        source_manifest_sha256=digest(manifest),
        source_audit=dict(path=str(audit_path.resolve()), sha256=digest(audit)),
        amendment=file_binding(amendment),
        decision_context=dict(
            validation_outcomes_inspected=True,
            representation_results_inspected_before_final_scope_decision=True,
            prospective_instrument_validation=False,
        ),
        **scoped_evidence(manifest, audit),
    )


def selected_assignments(config, manifest):
    low, high = config["world_index_range"]
    kinds = config["kinds"]
    if low >= high or not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("Source selection requires a nonempty range and unique kinds")
    selected = [a for a in manifest["diagnostics"] if low <= a["index"] < high and a["kind"] in kinds]
    if not selected or {a["kind"] for a in selected} != set(kinds):
        raise ValueError("Every selected kind must exist in the source assignment range")
    return selected


def admit_source(config, manifest, audit, selected):
    """Return a hashable receipt; confirmation never accepts a scoped certificate."""
    validate_source(manifest, audit)
    if selected != selected_assignments(config, manifest):
        raise ValueError("Replay source selection must cover exactly the configured parent assignments")
    declaration = config.get("source_eligibility_certificate")
    common = dict(
        schema_version=1,
        source_manifest_sha256=digest(manifest),
        source_audit_sha256=digest(audit),
        selected_assignments_sha256=digest(selected),
        selected_assigned_sources=len(selected),
        original_global_gates_passed=audit.get("gates", {}).get("passed"),
    )
    if declaration is None:
        if audit.get("gates", {}).get("passed") is not True:
            raise ValueError(
                "The source instrument must pass its predefined gates or bind a development certificate"
            )
        criteria = manifest.get("validation_criteria", {})
        full_validation = (
            manifest.get("phase") == "validation"
            and criteria.get("minimum_distinct_items_per_cell", 0) >= 100
            and audit.get("validation_criteria") == criteria
            and not any(
                item.get("development_only") is True
                or item.get("exploratory_only") is True
                or item.get("inference_scope") == "development"
                for item in (manifest, audit, audit["gates"])
            )
        )
        if config.get("inference_scope") == "confirmation" and not full_validation:
            raise ValueError(
                "Confirmation requires full validation with at least 100 distinct items per cell"
            )
        return dict(
            mode="full_global_validation" if full_validation else "global_development_gates",
            source_phase=manifest.get("phase"),
            validation_criteria=criteria,
            **common,
        )
    if config.get("inference_scope") != "development" or config.get("phase") in (
        "confirmation",
        "fresh_confirmation",
    ):
        raise ValueError(
            "Scoped source certificates are forbidden for confirmation, including posthoc analyses"
        )
    if not isinstance(declaration, dict) or set(declaration) != {"path", "sha256"}:
        raise ValueError("source_eligibility_certificate requires exactly path and sha256")
    path = Path(declaration["path"])
    if not path.is_absolute():
        raise ValueError("Certificate path must be absolute")
    certificate = read(path)
    if digest(certificate) != declaration["sha256"]:
        raise ValueError("Source eligibility certificate hash mismatch")
    if (
        certificate.get("source_manifest_sha256") != digest(manifest)
        or certificate.get("source_collection") != str(Path(config["source_collection"]).resolve())
        or certificate.get("source_audit")
        != dict(path=str(Path(config["source_audit"]).resolve()), sha256=digest(audit))
    ):
        raise ValueError("Source eligibility certificate binds different source artifacts")
    expected = build_certificate(
        config["source_collection"],
        config["source_audit"],
        certificate["amendment"]["path"],
        certificate["created_at_utc"],
    )
    if certificate != expected:
        raise ValueError(
            "Source eligibility certificate does not reproduce from original evidence and amendment"
        )
    if not set(config["kinds"]) <= set(RETAINED_KINDS):
        raise ValueError(
            "Scoped replay may use only the four retained kinds; count is excluded in its entirety"
        )
    return dict(
        mode="scoped_post_inspection_development",
        certificate_path=str(path.resolve()),
        certificate_sha256=digest(certificate),
        confirmation_eligible=False,
        retained_assigned_sources=certificate["retained"]["assigned"],
        excluded_assigned_sources=certificate["excluded"]["assigned"],
        **common,
    )


def scoped_additive_schedule(manifest):
    """Check the actual arms, so removing a design tag cannot widen admission."""
    if (
        manifest.get("assay") != "replay_information"
        or "design" in manifest
        or manifest.get("phase") in ("confirmation", "fresh_confirmation")
    ):
        return False
    assignments, diagnostics = manifest.get("assignments", []), manifest.get("diagnostics", [])
    if not isinstance(assignments, list) or not isinstance(diagnostics, list):
        return False
    rows = assignments + diagnostics
    projection_fields = {"projection_target", "matched_interchange_condition"}
    return bool(rows) and all(
        isinstance(row, dict)
        and row.get("arm") in ("baseline", "sham", "direction", "random")
        and row.get("patch_mode", "add") == "add"
        and not any(key.startswith("donor") or key in projection_fields for key in row)
        for row in rows
    )


def audit_source_admission(manifest):
    """Rebuild admission from frozen declarations, never a receipt's own claims.

    Older globally validated manifests can be audited through their frozen source
    schedule. A missing receipt can never authorize a scoped exception.
    """
    try:
        config = manifest["intervention_schedule"]
        if digest(config) != manifest.get("intervention_schedule_sha256") or config.get(
            "inference_scope"
        ) != manifest.get("inference_scope"):
            return False
        source = read(Path(config["source_collection"]) / "manifest.json")
        audit = read(config["source_audit"])
        if digest(source) != manifest.get("source_parent_manifest_sha256"):
            return False
        if "source_audit_sha256" in manifest and manifest["source_audit_sha256"] != digest(audit):
            return False
        selected = selected_assignments(config, source)
        prefixes, excluded = manifest["prefixes"], manifest["excluded_sources"]
        if (
            len(excluded) != len(set(excluded))
            or set(prefixes) & set(excluded)
            or set(prefixes) | set(excluded) != {a["trial_id"] for a in selected}
            or manifest["assigned_sources"] != len(selected)
            or any(prefixes[a["trial_id"]]["assignment"] != a for a in selected if a["trial_id"] in prefixes)
        ):
            return False
        expected = admit_source(config, source, audit, selected)
        if expected["mode"] == "scoped_post_inspection_development" and not scoped_additive_schedule(
            manifest
        ):
            return False
        receipt = manifest.get("source_admission")
        if receipt is None:
            return expected["mode"] in ("full_global_validation", "global_development_gates")
        return receipt == expected and manifest.get("source_admission_sha256") == digest(expected)
    except (OSError, KeyError, TypeError, ValueError):
        return False
