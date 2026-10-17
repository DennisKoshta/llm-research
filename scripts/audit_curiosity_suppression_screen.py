"""Audit the entire fixed C-negative and rival-control development screen.

Shared pure validators reconstruct task scores, receipt rendering, token timelines
and BF16 arithmetic. This audit owns the fifteen-arm inventory and provenance.
Decode residual vectors are not saved; their scalar evidence is checked, not
represented as independently reconstructed per-token vectors.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study import curiosity_suppression_capture as capture
from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study import curiosity_resource_limits as resources
from curiosity_study import sustained_engineering as engineering
from curiosity_study import voluntary_capture as receipts
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.concept_capture import MODEL
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE
from curiosity_study.satisfaction_capture import _choice_sidecar
from audit_satisfaction_screen import (
    artifact_path,
    compact,
    counts_for,
    engaged,
    engineering_environment,
    environments,
    geometry,
    hidden_arrays,
    receipt_measurement,
    require,
    sha,
    token_model,
)

ARMS = list(screen.ARMS)
PANELS = [name for name, _ in screen.PANELS]


def parent_children(root, manifest):
    from analyze_curiosity_suppression_screen import ANALYSIS_METADATA

    require(
        "execution_group" not in manifest and "parent_manifest_sha256" not in manifest,
        "Child-only results cannot be scientifically audited",
    )
    require(
        resources.job_wall_limit(manifest) == 12000
        and digest(manifest["budget"]) == digest(resources.CURIOSITY_SCREEN_BUDGET),
        "Parent budget differs from the fixed curiosity resource policy",
    )
    plan = manifest["screen_plan"]
    require(
        manifest["assay"] == screen.ASSAY
        and manifest["task_revision"] == screen.REVISION
        and manifest["phase"] == "development_pilot"
        and manifest["inference_scope"] == "development"
        and manifest["model"] == MODEL
        and manifest["execution_gpu"] == "H200"
        and manifest["execution"] == screen.EXECUTION
        and manifest["max_new_tokens"] == 4096
        and manifest["capture_plan"] == capture.CAPTURE_PLAN
        and manifest["schedule"] == screen.SCHEDULE
        and manifest["arithmetic"] == engineering.ARITHMETIC
        and manifest["engine_sha256"] == engineering.engine_hashes()
        and manifest["analysis_metadata"] == ANALYSIS_METADATA
        and manifest["assignments"] == []
        and manifest["diagnostics"] == plan["assignments"]
        and len(manifest["diagnostics"]) == 4320
        and manifest["assignment_sha256"] == digest(manifest["diagnostics"])
        and manifest["screen_plan_sha256"] == digest(plan)
        and manifest["source_screen_manifest_sha256"]
        == digest(manifest["source_screen_manifest"])
        == screen.PRIOR_MANIFEST_SHA256
        and plan
        == screen.assignment_plan(
            source_manifest=manifest["source_screen_manifest"],
            direction_bindings=plan["direction_bindings"],
        ),
        "Parent design, prior world inventory or engine binding differs",
    )
    descriptors = read(root / "children.json")
    require(
        len(descriptors) == 3 and [row["execution_group"] for row in descriptors] == list(screen.GROUPS),
        "Require all execution groups in assigned order",
    )
    children = []
    for descriptor, group in zip(descriptors, screen.GROUPS, strict=True):
        path = Path(descriptor["path"]).resolve()
        require(
            path.parent == root.resolve().parent and path != root.resolve(),
            "Child collection is outside the parent run directory",
        )
        child = read(path / "manifest.json")
        expected = deepcopy(manifest)
        expected.update(
            execution_group=group,
            parent_manifest_sha256=digest(manifest),
            diagnostics=screen.group_assignments(plan, group),
        )
        expected["assignment_sha256"] = digest(expected["diagnostics"])
        require(
            child == expected and digest(child) == descriptor["manifest_sha256"],
            "Child manifest differs from the exact cohort-balanced parent decomposition",
        )
        capture.validate_manifest(child)
        children.append((path, child))
    return children


ROLES = {
    "prospective_plan",
    "generation_preflight",
    "receipt_preflight",
    "direction_inventory",
    "resource_forecast",
    "source_screen_manifest",
    "source_screen_audit",
    "source_screen_freeze",
    "source_screen_archive",
    "source_screen_analysis",
    "source_screen_independent_check",
    "source_screen_independent_scoring",
    "source_screen_results",
    "engineering_manifest",
    "engineering_audit",
    "engineering_freeze",
    "engineering_archive",
    "candidate_manifest",
    "candidate_audit",
    "candidate_analysis",
    "candidate_directions",
    "candidate_independent_check",
}


def frozen_documents(root, manifest):
    frozen = read(root / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha(root / "source.tar.gz"),
        "Parent source freeze changed",
    )
    bindings = manifest["bound_receipts"]
    require(
        len(bindings) == len(ROLES)
        and {row["role"] for row in bindings} == ROLES
        and len({row["path"] for row in bindings}) == len(bindings),
        "Missing or duplicate prospective evidence roles",
    )
    values, payloads = {}, {}
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        require(len(names) == len(set(names)), "Duplicate archived source paths")
        for binding in bindings:
            relative = Path(binding["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe frozen receipt path")
            payload = archive.extractfile(str(relative)).read()
            require(
                hashlib.sha256(payload).hexdigest()
                == binding["byte_sha256"]
                == frozen["documents"].get(str(relative)),
                "Frozen receipt bytes differ",
            )
            value = json.loads(payload) if relative.suffix == ".json" else None
            require(
                binding["content_sha256"] == (digest(value) if value is not None else None),
                "Frozen JSON content digest differs",
            )
            values[binding["role"]], payloads[binding["role"]] = value, payload
        source = {
            name.removeprefix("src/curiosity_study/"): archive.extractfile(name).read().decode()
            for name in names
            if name.startswith("src/curiosity_study/") and name.endswith(".py")
        }
        require(digest(source) == manifest["source_sha256"], "Frozen source digest differs")
        # Check each reused validator implementation, including the unchanged S helpers.
        from curiosity_study import satisfaction_capture, satisfaction_screen, resource_limits
        import audit_satisfaction_screen

        for module in (
            capture,
            screen,
            resources,
            engineering,
            receipts,
            voluntary,
            satisfaction_capture,
            satisfaction_screen,
            resource_limits,
        ):
            relative = f"src/curiosity_study/{Path(module.__file__).name}"
            require(
                hashlib.sha256(archive.extractfile(relative).read()).hexdigest() == sha(module.__file__),
                "Shared audit validator differs from its frozen implementation",
            )
        require(
            hashlib.sha256(archive.extractfile("scripts/audit_satisfaction_screen.py").read()).hexdigest()
            == sha(audit_satisfaction_screen.__file__),
            "Shared pure screen audit helpers differ from their frozen implementation",
        )
        for filename in (
            "audit_curiosity_suppression_screen.py",
            "analyze_curiosity_suppression_screen.py",
            "prepare_curiosity_suppression_screen.py",
            "prepare_satisfaction_screen.py",
            "prepare_voluntary_resolution.py",
            "audit_sustained_engineering.py",
        ):
            require(
                hashlib.sha256(archive.extractfile(f"scripts/{filename}").read()).hexdigest()
                == sha(Path(__file__).with_name(filename)),
                "Frozen analysis or shared admission implementation changed",
            )
        for name, expected in manifest["engine_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(f"src/curiosity_study/{name}").read()).hexdigest()
                == expected,
                "Frozen numerical engine differs",
            )
    return frozen, values, payloads


def verify_directions(plan, payload):
    """Reconstruct every signed semantic and Gaussian coefficient without refitting."""
    bindings = screen.validate_directions(plan["direction_bindings"])
    require(
        hashlib.sha256(payload).hexdigest() == screen.LOCKED_DIRECTION_BYTES,
        "Original candidate direction bytes changed",
    )
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        require(
            arrays["layers"].tolist() == [15, 31, 47, 57]
            and arrays["positions"].tolist() == ["record", "final"],
            "Candidate sites changed",
        )
        sources = arrays["sources"].tolist()
        for arm, binding in bindings.items():
            if arm in screen.RANDOM_ARMS:
                raw = np.random.default_rng(binding["seed"]).standard_normal(5120)
                expected = raw / np.linalg.norm(raw)
                require(
                    binding["generator"] == "PCG64"
                    and binding["numpy_version"] == np.__version__
                    and binding["original_draw_byte_sha256"] == hashlib.sha256(raw.tobytes()).hexdigest()
                    and binding["original_draw_norm"] == float(np.linalg.norm(raw)),
                    "Gaussian generator, original draw or normalization changed",
                )
            else:
                index = sources.index(binding["source"])
                original = arrays["unit"][index, 0, 0].astype(np.float64)
                expected = binding["orientation"] * original
                require(
                    float(arrays["contrast_norm"][index, 0, 0]) == binding["source_raw_norm"],
                    "Neighbor source norm changed",
                )
                require(
                    binding["source_unit_sha256"] == digest(original.tolist()),
                    "Unsigned source axis hash changed",
                )
                if binding["source"] == "curiosity_vs_indifference":
                    require(
                        binding["source_raw_norm"] == screen.LOCKED_C_NORM,
                        "Locked original curiosity norm changed",
                    )
            require(
                expected.tolist() == binding["unit"],
                "Applied direction differs from the original signed axis or fixed random draw",
            )
    return bindings


def verify_inventory(bindings, inventory):
    arms = list(screen.ARMS[1:])
    require(
        inventory["direction_bindings_sha256"] == digest(bindings)
        and inventory["arm_order"] == arms
        and inventory["numpy_version"] == np.__version__
        and inventory["generator"] == "PCG64"
        and inventory["resampled"] is False
        and inventory["orientation_selected_by_outcome"] is False
        and len(inventory["directions"]) == len(arms),
        "Direction inventory metadata changed",
    )
    vectors = [np.asarray(bindings[arm]["unit"], dtype=np.float64) for arm in arms]
    for arm, vector, entry in zip(arms, vectors, inventory["directions"], strict=True):
        fp32 = vector.astype(np.float32)
        require(
            entry
            == dict(
                arm=arm,
                binding_sha256=digest(bindings[arm]),
                signed_unit_float64_byte_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                effective_float32_byte_sha256=hashlib.sha256(fp32.tobytes()).hexdigest(),
                float64_norm=float(np.linalg.norm(vector)),
                effective_float32_norm=float(np.linalg.norm(fp32)),
            ),
            "Effective FP32 applied vector or norm differs",
        )
    expected = np.array(
        [[float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) for b in vectors] for a in vectors]
    )
    actual = np.asarray(inventory["signed_unit_cosines"])
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, atol=2e-14, rtol=2e-14),
        "Full applied-direction cosine matrix differs",
    )


def verify_forecast(repository, forecast, budget):
    from budget import RATES

    require(
        forecast["schema_version"] == 1
        and forecast["stage_workers"] == 3
        and forecast["cohorts_per_worker"] == 45
        and forecast["branches_per_worker"] == 1440
        and forecast["shared_deadline_seconds"] == budget["shared_deadline_seconds"] == 12000
        and forecast["stage_ceiling_usd"] == budget["stage_ceiling_usd"] == 55
        and forecast["paid_retries"] == 0
        and forecast["completion_guaranteed"] is False
        and forecast["rate_usd_per_second"] == RATES["H200"]
        and forecast["startup_export_margin_seconds"] == 120
        and len(forecast["source_profiles"]) == 3,
        "Resource forecast policy changed",
    )
    seconds, loads = [], []
    for entry, suffix, cohorts in zip(
        forecast["source_profiles"],
        ("baseline-zero", "positive-half", "positive-full"),
        (18, 9, 9),
        strict=True,
    ):
        path = Path(entry["path"])
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and path.parent == Path(f"runs/qwen32-satisfaction-screen-v1-{suffix}/cloud_profiles"),
            "Forecast profile is outside the complete source screen",
        )
        profile = read(repository / path)
        require(
            profile == entry["profile"]
            and sha(repository / path) == entry["byte_sha256"]
            and profile["gpu"] == "NVIDIA H200"
            and profile["assigned"] == profile["completed"] == 32 * cohorts
            and entry["source_cohorts"] == cohorts,
            "Forecast source runtime changed or incomplete",
        )
        value = profile["execution_seconds"] * 45 / cohorts
        require(
            math.isclose(entry["extrapolated_execution_seconds_per_worker"], value, rel_tol=1e-14),
            "Forecast execution scaling differs",
        )
        seconds.append(value)
        loads.append(profile["load_seconds"])
    lower, upper, load, rate = min(seconds), max(seconds), max(loads), RATES["H200"]
    require(
        np.allclose(forecast["estimated_execution_seconds_per_worker_range"], [lower, upper], rtol=1e-14)
        and forecast["maximum_observed_load_seconds"] == load
        and np.allclose(
            forecast["estimated_combined_compute_usd_range"],
            [3 * (x + load + 120) * rate for x in (lower, upper)],
            rtol=1e-14,
        )
        and math.isclose(forecast["maximum_reserved_compute_usd"], 3 * 12120 * rate, rel_tol=1e-14)
        and forecast["maximum_reserved_compute_usd"] <= 55
        and forecast["hypothetical_all_cap_seconds_per_worker"] == 45 * 4096 * 0.218
        and upper + load + 120 < 12000,
        "Prospective runtime or cost arithmetic differs or lacks observed headroom",
    )


def prior_decision_agrees(independent, analysis):
    """The independent receipt binds six results; analysis additionally has prose."""
    fields = {
        "classification",
        "effect_threshold_met",
        "positive_screen",
        "primary_panel_differences",
        "primary_pooled_menu_differences",
        "sensitivity_can_rescue",
    }
    require(
        set(independent) == fields and fields <= set(analysis),
        "Prior independent decision schema differs from its six fixed result fields",
    )
    return digest(independent) == digest({key: analysis[key] for key in fields})


def provenance(root, manifest, tokenizer):
    from audit_satisfaction_screen import audit as audit_source
    from audit_sustained_engineering import audit as audit_engineering
    from prepare_voluntary_resolution import candidate_binding
    from prepare_curiosity_suppression_screen import receipt_preflight

    frozen, documents, payloads = frozen_documents(root, manifest)
    roles = {binding["role"]: binding for binding in manifest["bound_receipts"]}
    repository = Path(frozen["budget_root"])
    source, prior_audit, prior_freeze = (
        documents[name] for name in ("source_screen_manifest", "source_screen_audit", "source_screen_freeze")
    )
    prior_analysis, independent, scoring = (
        documents[name]
        for name in (
            "source_screen_analysis",
            "source_screen_independent_check",
            "source_screen_independent_scoring",
        )
    )
    source_run = repository / Path(roles["source_screen_manifest"]["path"]).parent
    require(
        source == manifest["source_screen_manifest"]
        and digest(source) == manifest["source_screen_manifest_sha256"] == screen.PRIOR_MANIFEST_SHA256
        and read(source_run / "manifest.json") == source
        and read(source_run / "freeze.json") == prior_freeze
        and sha(source_run / "source.tar.gz")
        == roles["source_screen_archive"]["byte_sha256"]
        == prior_freeze["archive_sha256"]
        and prior_freeze["manifest_sha256"] == digest(source)
        and audit_source(source_run, tokenizer=tokenizer) == prior_audit,
        "Entire original screen inputs or complete audit no longer reproduce",
    )
    require(
        prior_audit["complete"] is True
        and prior_audit["integrity_passed"] is True
        and all(
            prior_audit[key] is True
            for key in (
                "sham_identity_passed",
                "engagement_passed",
                "receipts_passed",
                "cross_arm_prepatch_identity_passed",
                "worker_policies_equal",
            )
        )
        and independent["passed"] is True
        and scoring["passed"] is True
        and independent["parent_manifest_sha256"] == digest(source)
        and independent["analysis_sha256"] == digest(prior_analysis)
        and independent["audit_sha256"] == digest(prior_audit)
        and independent["analysis_byte_sha256"] == roles["source_screen_analysis"]["byte_sha256"]
        and independent["audit_byte_sha256"] == roles["source_screen_audit"]["byte_sha256"]
        and prior_decision_agrees(independent["decision"], prior_analysis["decision"])
        and independent["audit_readiness_passed"] is prior_audit["readiness"]["passed"]
        and prior_audit["readiness"]["passed"] is False
        and prior_analysis["decision"]["classification"] == "readiness_failed"
        and prior_analysis["decision"]["positive_screen"] is False
        and prior_analysis["decision"]["effect_threshold_met"] is False
        and all(value == 0 for value in prior_analysis["decision"]["primary_panel_differences"].values())
        and scoring["checked"] == scoring["assigned"] == 1152
        and scoring["parent_manifest_sha256"] == digest(source)
        and scoring["source_archive_sha256"] == prior_freeze["archive_sha256"]
        and scoring["record_bindings_sha256"]
        == digest(
            [
                dict(trial_id=row["trial_id"], record_sha256=row["record_sha256"])
                for row in prior_audit["outcomes"]
            ]
        ),
        "Prior screen evidence is incomplete, relabeled or disconnected from independent checks",
    )
    plan = manifest["screen_plan"]
    require(
        plan == screen.assignment_plan(source_manifest=source, direction_bindings=plan["direction_bindings"]),
        "All original previously exposed worlds and branches must be retained",
    )
    source_certificate = manifest["source_screen_certificate"]
    require(
        all(
            source_certificate[key] == value
            for key, value in dict(
                manifest_sha256=digest(source),
                freeze_sha256=digest(prior_freeze),
                archive_sha256=prior_freeze["archive_sha256"],
                audit_sha256=digest(prior_audit),
                analysis_sha256=digest(prior_analysis),
                independent_check_sha256=digest(independent),
                independent_scoring_sha256=digest(scoring),
                prior_classification="readiness_failed",
                prior_positive_screen=False,
                prior_exact_choice_effect=0.0,
            ).items()
        ),
        "Source screen certificate changes the complete prior evidence or failed decision",
    )
    candidate = candidate_binding(
        documents["candidate_manifest"],
        documents["candidate_audit"],
        documents["candidate_analysis"],
        documents["candidate_independent_check"],
        roles["candidate_analysis"]["byte_sha256"],
        roles["candidate_directions"]["byte_sha256"],
    )
    require(
        candidate == manifest["candidate_source"]
        and digest(documents["candidate_manifest"])
        == "0298226e20ee1a5c6e0a07bf6b26db6df3e9e25db183709db5151b61055b4772"
        and candidate["selected_sites"]["curiosity_vs_indifference"] == dict(layer=15, position="record"),
        "Locked candidate provenance changed",
    )
    bindings = verify_directions(plan, payloads["candidate_directions"])
    inventory, forecast = documents["direction_inventory"], documents["resource_forecast"]
    verify_inventory(bindings, inventory)
    verify_forecast(repository, forecast, manifest["budget"])
    require(
        manifest["direction_inventory_sha256"] == digest(inventory)
        and manifest["resource_forecast_sha256"] == digest(forecast),
        "Direction arithmetic inventory or prospective resource forecast does not reproduce",
    )
    em, ef, ea = (
        documents[name] for name in ("engineering_manifest", "engineering_freeze", "engineering_audit")
    )
    engineering_run = repository / Path(roles["engineering_manifest"]["path"]).parent
    require(
        read(engineering_run / "manifest.json") == em
        and read(engineering_run / "freeze.json") == ef
        and sha(engineering_run / "source.tar.gz") == roles["engineering_archive"]["byte_sha256"]
        and audit_engineering(engineering_run) == ea
        and ea["complete"] is True
        and ea["passed"] is True
        and ea["integrity_passed"] is True
        and ea["cap"] == em["cap"] == 4096
        and em["model"] == MODEL
        and em["execution"] == screen.EXECUTION
        and em["schedule"] == screen.SCHEDULE
        and em["arithmetic"] == engineering.ARITHMETIC
        and em["engine_sha256"] == manifest["engine_sha256"]
        and ef["manifest_sha256"] == digest(em)
        and ef["archive_sha256"] == roles["engineering_archive"]["byte_sha256"],
        "Full-cap engineering proof no longer reproduces or changed execution policy",
    )
    certificate = manifest["engineering_certificate"]
    require(
        all(
            certificate[key] == value
            for key, value in dict(
                manifest_sha256=digest(em),
                audit_sha256=digest(ea),
                freeze_sha256=digest(ef),
                archive_sha256=ef["archive_sha256"],
            ).items()
        ),
        "Engineering certificate hash differs",
    )
    model = token_model(tokenizer)
    for role, actual in (
        ("generation_preflight", capture.bind_preflight(plan, model)),
        ("receipt_preflight", receipt_preflight(plan, model)),
    ):
        require(
            documents[role] == manifest[role] == actual and manifest[role + "_sha256"] == digest(actual),
            "Prospective tokenizer boundary check no longer reproduces",
        )
    return dict(
        freeze=frozen,
        candidate_source=candidate,
        bound_receipts=manifest["bound_receipts"],
        source_screen_certificate=source_certificate,
        source_screen=dict(
            manifest_sha256=digest(source),
            audit_sha256=digest(prior_audit),
            readiness=prior_audit["readiness"],
            decision=prior_analysis["decision"],
            reused_worlds=24,
            fresh_worlds=0,
        ),
        engineering_certificate=certificate,
        engineering_environment=engineering_environment(engineering_run, ea),
    )


def readiness(outcomes):
    """Preserve all 4320 assigned denominators, independent of behavior direction."""
    require(
        len(outcomes) == len({row["trial_id"] for row in outcomes}) == 4320,
        "Readiness requires every unique assigned parent row",
    )
    diagnostics, choices = [], []
    for arm in ARMS:
        for panel in PANELS:
            for kind in voluntary.DIAGNOSTICS:
                rows = [
                    r for r in outcomes if r["arm"] == arm and r["panel_id"] == panel and r["kind"] == kind
                ]
                counts = counts_for(rows)
                diagnostics.append(
                    dict(
                        arm=arm,
                        panel_id=panel,
                        kind=kind,
                        **counts,
                        passed=counts["assigned"] == counts["valid"] == 12
                        and counts["correct"] >= 11
                        and counts["cap_hits"] == counts["missing"] == counts["execution_failures"] == 0,
                    )
                )
            for delivery in voluntary.DELIVERIES:
                rows = [
                    r
                    for r in outcomes
                    if r["arm"] == arm
                    and r["panel_id"] == panel
                    and r["kind"] == "choice"
                    and r["delivery"] == delivery
                ]
                counts = counts_for(rows)
                choices.append(
                    dict(
                        arm=arm,
                        panel_id=panel,
                        delivery=delivery,
                        **counts,
                        passed=counts["assigned"] == 24
                        and counts["valid"] >= 23
                        and counts["cap_hits"] == counts["missing"] == counts["execution_failures"] == 0,
                    )
                )
    return dict(
        assigned=4320,
        choice_correctness=None,
        diagnostic_cells=diagnostics,
        choice_cells=choices,
        passed=all(row["passed"] for row in diagnostics + choices),
        interpretation="Prespecified descriptive competence gates; not noninferiority or unchanged internal confidence.",
    )


def validate_generation(
    evidence, assignments, requests, tokenizer, eos_ids, vocabulary_size, arrays, direction
):
    """Return measured engineering gates; score mismatches are binding errors."""
    condition = evidence["condition"]
    require(
        evidence["eos_token_ids"] == sorted(eos_ids) and evidence["pad_token_id"] == tokenizer.pad_token_id,
        "EOS or padding identity changed",
    )
    patches = capture.patches_for(assignments, np.asarray(direction, dtype=np.float32))
    timeline = engineering.validate_condition(
        condition,
        requests,
        patches,
        cap=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_ids=eos_ids,
    )
    captures = arrays is not None and bool(np.isfinite(arrays["hidden"]).all())
    geometry_passed, engagement = captures, captures
    if condition["rows"] is None:
        return dict(
            timeline=False,
            captures=captures,
            hook_cleanup=condition["hooks_clean"] is True,
            geometry=False,
            engagement=False,
        )
    require(len(condition["rows"]) == 32, "Wrong generation row count")
    require(
        all(frame["vocabulary_size"] == vocabulary_size for frame in condition["forwards"]),
        "Full-vocabulary fingerprint width changed",
    )
    for index, (row, assignment) in enumerate(zip(condition["rows"], assignments, strict=True)):
        completion = row["completion"]
        tokens = completion["generated_token_ids"]
        require(
            all(type(token) is int and 0 <= token < vocabulary_size for token in tokens)
            and completion["raw"] == tokenizer.decode(tokens, skip_special_tokens=True),
            "Generated raw answer differs from actual token IDs",
        )
        dose = None if assignment["intervention"] is None else assignment["intervention"]["dose"]
        if captures:
            geometry_passed &= geometry(
                row["intervention_trace"], arrays["hidden"][index, 0, 0], direction, dose
            )
            engagement &= engaged(row["intervention_trace"], dose)
    return dict(
        timeline=timeline,
        captures=captures,
        hook_cleanup=condition["hooks_clean"] is True,
        geometry=bool(geometry_passed),
        engagement=bool(engagement),
    )


def generation_records(root, manifest):
    assignments = capture.validate_manifest(manifest)
    require(not any((root / "attempts").rglob("*.json")), "Screen generations require atomic cohorts")
    records, offsets, bindings, attempts = {}, set(), [], set()
    for path in sorted((root / "cohorts").rglob("*.json")):
        require(
            len(path.relative_to(root / "cohorts").parts) == 2 and path.stem.isdigit(),
            "Unexpected generation cohort path",
        )
        offset = int(path.stem)
        require(
            offset in range(0, len(assignments), 32) and offset not in offsets,
            "Duplicate or unassigned generation cohort",
        )
        envelope = read(path)
        rows = envelope["records"]
        require(
            envelope["manifest_hash"] == digest(manifest)
            and envelope["sha256"] == digest(rows)
            and len(rows) == 32
            and [r["assignment"] for r in rows] == assignments[offset : offset + 32]
            and all(r["terminal"] is True and r["trial_id"] == r["assignment"]["trial_id"] for r in rows),
            "Generation cohort membership or bytes changed",
        )
        offsets.add(offset)
        attempts.add(path.parent.name)
        records.update({row["trial_id"]: row for row in rows})
        bindings.append(
            dict(path=str(path.relative_to(root)), byte_sha256=sha(path), envelope_sha256=digest(envelope))
        )
    require(len(attempts) <= 1, "Screen child cannot combine multiple execution attempts")
    return records, offsets, bindings


def audit_generations(root, manifest, tokenizer, eos_ids, vocabulary_size):
    records, offsets, cohort_bindings = generation_records(root, manifest)
    assignments = manifest["diagnostics"]
    model = token_model(tokenizer)
    require(
        capture.bind_preflight(manifest["screen_plan"], model) == manifest["generation_preflight"],
        "Actual tokenizer differs from the bound preflight",
    )
    directions = manifest["screen_plan"]["direction_bindings"]
    caches, baselines, evidence_bindings, record_bindings, checks, prepatch = {}, {}, [], [], [], {}
    for offset in sorted(offsets):
        cohort = assignments[offset : offset + 32]
        direction = directions[cohort[0]["arm"] if cohort[0]["arm"] != "baseline" else "zero_sham"]["unit"]
        ids = [a["trial_id"] for a in cohort]
        rows = [records[tid] for tid in ids]
        artifact = rows[0]["generation_evidence"]
        require(
            all(row["generation_evidence"] == artifact for row in rows),
            "One generation cohort binds different evidence",
        )
        evidence = read(artifact_path(root, artifact))
        requests = [manifest["generation_preflight"]["requests"][a["base_row"]]["request"] for a in cohort]
        require(
            evidence["manifest_sha256"] == digest(manifest)
            and evidence["generation_offset"] == offset
            and evidence["trial_ids"] == ids
            and evidence["requests"] == requests
            and evidence["capture_plan"] == capture.CAPTURE_PLAN,
            "Generation evidence is rebound to different rows or inputs",
        )
        capture_artifact = evidence["activation_artifact"]
        arrays = (
            None if capture_artifact is None else hidden_arrays(root, capture_artifact, ids, requests, caches)
        )
        require(
            arrays is None or arrays["hidden"].shape[-1] == len(direction),
            "Residual width differs from the fixed direction",
        )
        measured = validate_generation(
            evidence, cohort, requests, tokenizer, eos_ids, vocabulary_size, arrays, direction
        )
        saved_checks = {key: measured[key] for key in ("timeline", "captures", "hook_cleanup")}
        condition = evidence["condition"]
        width = (
            (max(len(r["token_prefix"]) for r in requests) + PREFILL_PADDING_MULTIPLE - 1)
            // PREFILL_PADDING_MULTIPLE
        ) * PREFILL_PADDING_MULTIPLE
        if condition.get("observations_unavailable") is True:
            require(
                condition["rows"] is None
                and condition["forwards"] == []
                and condition["attempted_forwards"] is None
                and condition["error"] is not None
                and arrays is None,
                "Unavailable generation fabricates observed evidence",
            )
        else:
            require(
                condition["capture_phase"] == capture.CAPTURE_PLAN["source_phase"]
                and condition["unpadded_token_positions"] == [r["prefix_positions"] for r in requests]
                and condition["padded_token_positions"]
                == [
                    [width - len(r["token_prefix"]) + r["prefix_positions"][p] for p in ("record", "final")]
                    for r in requests
                ]
                and condition["prefill_width"] == width,
                "Source capture timing or padded sites changed",
            )
        if cohort[0]["arm"] == "baseline":
            baselines[cohort[0]["base_row"]] = (evidence, None if arrays is None else arrays["hidden"])
            require(evidence["sham_comparison"] is None, "Baseline claims a sham comparison")
            require(
                evidence["common_prefill_comparison"] is None, "Baseline claims a common-source comparison"
            )
        else:
            reference = baselines.get(cohort[0]["base_row"])
            require(reference is not None, "Measured follow-up lacks its earlier source baseline")
            baseline, baseline_hidden = reference
            common = dict(
                baseline_evidence_sha256=digest(baseline),
                same_requests=baseline["requests"] == requests,
                layer15_prepatch_captures_equal=arrays is not None
                and baseline_hidden is not None
                and np.array_equal(arrays["hidden"], baseline_hidden),
            )
            require(
                evidence["common_prefill_comparison"] == common,
                "Saved common prefill comparison differs from both observed source sites",
            )
            saved_checks["common_prefill_identity"] = (
                common["same_requests"] and common["layer15_prepatch_captures_equal"]
            )
            measured["common_prefill_identity"] = saved_checks["common_prefill_identity"]
        if cohort[0]["arm"] == "zero_sham":
            reference = baselines.get(cohort[0]["base_row"])
            if reference is None:
                measured["sham_identity"] = False
                require(evidence["sham_comparison"] is None, "Sham binds a missing baseline")
                saved_checks["sham_identity"] = False
            else:
                baseline, baseline_hidden = reference
                identical = (
                    condition["rows"] is not None
                    and baseline["condition"]["rows"] is not None
                    and engineering.identical_rows(baseline["condition"], condition, range(32))
                )
                expected = dict(
                    baseline_evidence_sha256=digest(baseline),
                    same_requests=baseline["requests"] == requests,
                    full_active_logits_and_tokens_equal=bool(identical),
                )
                require(
                    evidence["sham_comparison"] == expected,
                    "Saved sham comparison differs from observed predictions",
                )
                saved_checks["sham_identity"] = expected["same_requests"] and bool(identical)
                measured["sham_identity"] = (
                    saved_checks["sham_identity"]
                    and arrays is not None
                    and baseline_hidden is not None
                    and np.array_equal(baseline_hidden, arrays["hidden"])
                )
        elif cohort[0]["arm"] != "baseline":
            require(evidence["sham_comparison"] is None, "Positive arm claims a baseline identity comparison")
        require(
            all(type(value) is bool for value in evidence["checks"].values())
            and evidence["checks"] == saved_checks
            and evidence["engineering_passed"] is all(saved_checks.values()),
            "Saved generation engineering checks differ from their observations",
        )
        for index, (assignment, row) in enumerate(zip(cohort, rows, strict=True)):
            observed = None if condition["rows"] is None else condition["rows"][index]
            expected_inner = (
                None
                if observed is None
                else voluntary.record_from_thinking_inference(
                    assignment["source_assignment"], observed["completion"]
                )
            )
            require(
                digest(row["task_record"]) == digest(expected_inner)
                and row["cohort_row"] == index
                and row["activation_artifact"] == capture_artifact
                and row["engineering_passed"] is evidence["engineering_passed"]
                and row["status"] == ("execution_error" if expected_inner is None else "completed"),
                "Saved task parsing/scoring, evidence row or disposition changed",
            )
            expected_failure = (
                "generation_execution_error"
                if observed is None
                else "generation_engineering_guard_failed"
                if not evidence["engineering_passed"]
                else None
            )
            require(row.get("failure") == expected_failure, "Generation failure was hidden or reclassified")
            record_bindings.append(
                dict(
                    trial_id=row["trial_id"],
                    record_sha256=digest(row),
                    task_record_sha256=digest(expected_inner),
                    evidence_sha256=digest(evidence),
                    artifact_sha256=None if capture_artifact is None else capture_artifact["sha256"],
                    row=index,
                )
            )
            prepatch[assignment["trial_id"]] = None if arrays is None else arrays["hidden"][index].copy()
        checks.append(dict(offset=offset, arm=cohort[0]["arm"], **measured))
        evidence_bindings.append(dict(**artifact, content_sha256=digest(evidence)))
    return dict(
        records=records,
        complete=offsets == set(range(0, len(assignments), 32)),
        checks=checks,
        cohorts=cohort_bindings,
        evidence=evidence_bindings,
        record_bindings=record_bindings,
        capture_artifacts=[binding for binding, _ in caches.values()],
        prepatch=prepatch,
    )


def audit_receipts(root, manifest, sources, tokenizer):
    choices = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
    offsets, cache, bindings, all_rows, gates = set(), {}, [], [], []
    generation_attempts = {path.parent.name for path in (root / "cohorts").rglob("*.json")}
    model = token_model(tokenizer)
    for path in sorted((root / "receipt_cohorts").rglob("*.json")):
        require(len(path.relative_to(root / "receipt_cohorts").parts) == 2, "Unexpected receipt cohort path")
        require(
            generation_attempts == {path.parent.name},
            "Receipt attempt differs from the child's sole verified generation attempt",
        )
        envelope = read(path)
        offset = envelope["choice_offset"]
        require(path.name == f"{offset:07d}.json", "Receipt filename differs from its logical offset")
        require(
            type(offset) is int and offset in range(0, len(choices), 32) and offset not in offsets,
            "Duplicate or unassigned receipt cohort",
        )
        ids = [a["trial_id"] for a in choices[offset : offset + 32]]
        rows = envelope["records"]
        require(
            envelope["manifest_sha256"] == digest(manifest)
            and envelope["source_trial_ids"] == ids
            and [row["source_trial_id"] for row in rows] == ids
            and envelope["records_sha256"] == digest(rows)
            and envelope["rendering_policy"] == "fresh_hf_conversation_render_nonreplay"
            and all(tid in sources for tid in ids),
            "Receipt envelope assignment/source binding differs",
        )
        offsets.add(offset)
        requests, active, base_rows = [], [], {}
        for row in rows:
            source = sources[row["source_trial_id"]]
            base = _choice_sidecar(source)
            base_rows[row["source_trial_id"]] = base
            for key in (
                "schema_version",
                "source_trial_id",
                "source_record_sha256",
                "source_task_record_sha256",
                "assignment_sha256",
                "inner_source_trial_id",
                "selected_service",
                "simulator_outcome_sha256",
                "receipt_required",
                "terminal",
            ):
                require(row.get(key) == base.get(key), "Receipt source or selected action changed")
            if not base["receipt_required"]:
                require(row == base, "Skipped or unavailable receipt fabricated a model pass")
                continue
            receipt, request, binding = receipts.render_receipt(source["task_record"], model)
            if row["receipt_input"] is not None or row["token_binding"] is not None:
                require(
                    row["receipt_input"] == receipt and row["token_binding"] == binding,
                    "Receipt includes a wrong payload or rendered prefix",
                )
            active.append(row)
            requests.append(request)
        active_ids = [row["source_trial_id"] for row in active]
        require(envelope["active_source_trial_ids"] == active_ids, "Active receipt row order changed")
        if envelope["error"] is None:
            require(
                all(
                    row.get("failure") != "receipt_execution_error" and row.get("error") is None
                    for row in active
                ),
                "Receipt failure is missing its envelope error",
            )
        else:
            require(
                bool(active)
                and all(
                    row["status"] == "execution_error"
                    and row.get("failure") == "receipt_execution_error"
                    and row.get("error") == envelope["error"]
                    for row in active
                ),
                "Receipt envelope error differs from retained active failure statuses",
            )
        guard = envelope["engineering"]
        observed = envelope["observed_measurements"]
        artifact = envelope["observed_activation_artifact"]
        arrays = None if artifact is None else hidden_arrays(root, artifact, active_ids, requests, cache)
        passed = not active
        if not active:
            require(
                guard is None and observed is None and artifact is None,
                "Inactive receipts claim a model call",
            )
        elif guard is not None:
            width = (
                (max(len(r["token_prefix"]) for r in requests) + PREFILL_PADDING_MULTIPLE - 1)
                // PREFILL_PADDING_MULTIPLE
            ) * PREFILL_PADDING_MULTIPLE
            require(
                set(guard["checks"]) == set(receipts.ENGINEERING_CHECKS)
                and all(type(value) is bool for value in guard["checks"].values())
                and guard["passed"] is all(guard["checks"].values())
                and guard["source_trial_ids"] == active_ids
                and guard["active_batch_size"] == len(active)
                and guard["layers"] == [15]
                and guard["positions"] == ["record", "final"]
                and guard["max_new_tokens"] == 0
                and guard["prefill_width"] == width
                and guard["token_prefix_sha256"] == [r["token_prefix_sha256"] for r in requests],
                "Receipt guard metadata or saved checks changed",
            )
            before, after = (
                guard["baseline_full_initial_logits_sha256"],
                guard["captured_full_initial_logits_sha256"],
            )
            valid_hashes = len(before) == len(after) == len(active) and all(
                isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in before + after
            )
            require(
                guard["checks"]["full_initial_logits_equal"] is (before == after),
                "Receipt identity check contradicts full-vocabulary fingerprints",
            )
            if arrays is None:
                require(
                    envelope["error"] is not None
                    and all(row["status"] == "execution_error" for row in active),
                    "Missing receipt capture artifact claims successful delivery",
                )
            else:
                require(
                    guard["captured_array_sha256"] == receipts.array_sha(arrays["hidden"])
                    and arrays["hidden"].shape[-1]
                    == len(manifest["screen_plan"]["direction_bindings"]["zero_sham"]["unit"]),
                    "Receipt capture arrays are unbound",
                )
            require(
                isinstance(observed, list) and len(observed) == len(active),
                "Missing observed receipt measurements",
            )
            measurements_valid = all(
                receipt_measurement(measurement, request, tokenizer, width)
                for measurement, request in zip(observed, requests, strict=True)
            )
            passed = (
                guard["passed"] is True
                and valid_hashes
                and measurements_valid
                and arrays is not None
                and bool(np.isfinite(arrays["hidden"]).all())
            )
        else:
            require(observed is None and artifact is None, "Receipt observations lack their guard")
        passed = passed and envelope["error"] is None
        for index, row in enumerate(active):
            base = base_rows[row["source_trial_id"]]
            if row["status"] == "execution_error":
                require(
                    row["receipt_model_pass_completed"] is False
                    and row["measurement"] is None
                    and row["activation_artifact"] is None
                    and row["cohort"] is None
                    and digest(row["executed_simulator_outcome"])
                    == digest(base["executed_simulator_outcome"])
                    and row.get("failure") == "receipt_execution_error",
                    "Failed receipt claims a completed selected-payload pass",
                )
                passed = False
                continue
            executed = deepcopy(base["executed_simulator_outcome"])
            executed["receipt_model_pass_completed"] = True
            executed["observation_ledger"][-1]["model_pass_completed"] = True
            require(
                passed
                and row["status"] == "completed"
                and row["receipt_model_pass_completed"] is True
                and row["measurement"] == observed[index]
                and row["activation_artifact"] == artifact
                and row["cohort"]
                == dict(
                    choice_offset=offset,
                    active_row=index,
                    active_size=len(active),
                    prefill_width=guard["prefill_width"],
                )
                and row["receipt_input"] is not None
                and row["token_binding"] is not None
                and digest(row["executed_simulator_outcome"]) == digest(executed)
                and "failure" not in row,
                "Completed receipt lacks the actual bound selected-payload forward",
            )
        gates.append(bool(passed and all(row["status"] != "execution_error" for row in rows)))
        all_rows.extend(rows)
        bindings.append(
            dict(path=str(path.relative_to(root)), byte_sha256=sha(path), envelope_sha256=digest(envelope))
        )
    return dict(
        complete=offsets == set(range(0, len(choices), 32)),
        passed=len(gates) == len(range(0, len(choices), 32)) and all(gates),
        slots=len(all_rows),
        status_counts=dict(Counter(row["status"] for row in all_rows)),
        cohorts=bindings,
        record_bindings=[
            dict(source_trial_id=row["source_trial_id"], sidecar_sha256=digest(row)) for row in all_rows
        ],
        capture_artifacts=[binding for binding, _ in cache.values()],
    )


def cross_arm_identity(assignments, vectors):
    """Compare both unpatched sites by exact bytes, retaining missing base rows."""
    groups = {}
    for assignment in assignments:
        groups.setdefault(assignment["base_trial_id"], {})[assignment["arm"]] = vectors.get(
            assignment["trial_id"]
        )
    rows = []
    for base_id, arms in sorted(groups.items()):
        available = set(arms) == set(ARMS) and all(
            value is not None and np.isfinite(value).all() for value in arms.values()
        )
        identical = available and all(
            value.dtype == arms["baseline"].dtype
            and value.shape == arms["baseline"].shape
            and value.tobytes() == arms["baseline"].tobytes()
            for value in arms.values()
        )
        rows.append(
            dict(
                base_trial_id=base_id,
                available=bool(available),
                identical=bool(identical),
                arm_array_sha256={
                    arm: None if value is None else receipts.array_sha(value) for arm, value in arms.items()
                },
            )
        )
    return dict(
        passed=len(rows) == 288 and all(row["identical"] for row in rows),
        assigned_base_rows=288,
        observed_base_rows=len(rows),
        sites=["record", "final"],
        rows=rows,
    )


def audit(root, tokenizer=None, eos_token_ids=None, vocabulary_size=None):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    children = parent_children(root, manifest)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL["model_id"], revision=MODEL["revision"], trust_remote_code=False, local_files_only=True
        )
    if eos_token_ids is None or vocabulary_size is None:
        from transformers import AutoConfig, GenerationConfig

        if eos_token_ids is None:
            eos = GenerationConfig.from_pretrained(
                MODEL["model_id"], revision=MODEL["revision"], local_files_only=True
            ).eos_token_id
            eos_token_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        if vocabulary_size is None:
            vocabulary_size = AutoConfig.from_pretrained(
                MODEL["model_id"], revision=MODEL["revision"], local_files_only=True
            ).vocab_size
    bound = provenance(root, manifest, tokenizer)
    records, vectors, child_checks, environment_policies = {}, {}, [], []
    for path, child in children:
        generated = audit_generations(path, child, tokenizer, set(eos_token_ids), vocabulary_size)
        received = audit_receipts(path, child, generated["records"], tokenizer)
        env = environments(path, child, generated)
        environment_policies.extend(row["policy"] for row in env)
        records.update(generated.pop("records"))
        vectors.update(generated.pop("prepatch"))
        child_checks.append(
            dict(
                execution_group=child["execution_group"],
                manifest_sha256=digest(child),
                generation=generated,
                receipts=received,
                environments=env,
            )
        )
    outcomes = [compact(a, records.get(a["trial_id"])) for a in manifest["diagnostics"]]
    identity = cross_arm_identity(manifest["diagnostics"], vectors)
    complete = all(c["generation"]["complete"] and c["receipts"]["complete"] for c in child_checks)
    checks = [row for child in child_checks for row in child["generation"]["checks"]]
    sham = [row for row in checks if row["arm"] == "zero_sham"]
    sham_passed = len(sham) == 9 and all(row["sham_identity"] for row in sham)
    engagement = len(checks) == 135 and all(row["engagement"] and row["geometry"] for row in checks)
    receipt_passed = all(child["receipts"]["passed"] for child in child_checks)
    policies_equal = len(environment_policies) == 3 and all(
        policy == bound["engineering_environment"]["policy"] for policy in environment_policies
    )
    integrity = bool(
        complete
        and identity["passed"]
        and policies_equal
        and len(checks) == 135
        and all(
            row["timeline"]
            and row["captures"]
            and row["hook_cleanup"]
            and row["geometry"]
            and row.get("common_prefill_identity", True)
            for row in checks
        )
    )
    ready = readiness(outcomes)
    return dict(
        schema_version=1,
        assay="curiosity_suppression_screen_audit",
        parent_manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=sha(__file__),
        assigned=4320,
        assigned_choices=2160,
        terminal_generations=len(records),
        terminal_receipt_slots=sum(child["receipts"]["slots"] for child in child_checks),
        complete=bool(complete),
        integrity_passed=integrity,
        sham_identity_passed=bool(sham_passed),
        engagement_passed=bool(engagement),
        receipts_passed=bool(receipt_passed),
        cross_arm_prepatch_identity_passed=identity["passed"],
        cross_arm_prepatch_identity=identity,
        worker_policies_equal=bool(policies_equal),
        readiness=ready,
        passed=bool(integrity and sham_passed and engagement and receipt_passed and ready["passed"]),
        provenance=bound,
        children=child_checks,
        outcomes=outcomes,
        outcomes_sha256=digest(outcomes),
        scientific_claim="Fixed imposed-vector development comparison on all previously exposed worlds; no natural curiosity suppression, subjective-state or confidence-preservation conclusion.",
        validation_scope=(
            "Exact archived inputs, source-task parsing, token timelines, full-vocabulary sham fingerprints, "
            "prefill BF16 geometry, and selected-payload receipts. Task/render/timeline and admission helpers "
            "are shared with frozen implementations. Decode geometry has scalar consistency checks, not "
            "independent residual-vector reconstruction. Full-cap engineering tested one C+ row; this C-negative "
            "and rival-control screen additionally requires every signed applied direction to engage and same-source prepatch identity. "
            "Every arm occurs on all workers with each source cohort kept together; matching runtime and prepatch equality do not "
            "eliminate every possible hardware interaction. Receipt comparisons are selection-conditioned."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Audit receipts are immutable")
    result = audit(args.run)
    write_once(args.out, result)
    print(
        {
            key: result[key]
            for key in (
                "complete",
                "integrity_passed",
                "sham_identity_passed",
                "engagement_passed",
                "receipts_passed",
                "passed",
            )
        }
    )
