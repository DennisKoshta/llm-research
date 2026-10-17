"""Prepare the fixed object-only screen from verified evidence; never launch a GPU."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from curiosity_study import satisfaction_capture as capture
from curiosity_study import satisfaction_screen as screen
from curiosity_study import sustained_engineering as engineering
from curiosity_study import voluntary_capture as receipts
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.resource_limits import SCREEN_BUDGET
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_execution import freeze  # noqa: E402
from prepare_voluntary_resolution import candidate_binding  # noqa: E402
from prepare_sustained_engineering import CANDIDATE_DOCUMENTS  # noqa: E402
from analyze_satisfaction_screen import ANALYSIS_METADATA  # noqa: E402

PLAN = "reports/satisfaction-screen-v1-plan.md"
HISTORY = "reports/satisfaction-screen-v1-history.json"
EXPOSURE = "reports/satisfaction-screen-v1-freshness.json"
GENERATION = "reports/satisfaction-screen-v1-tokenization-preflight.json"
RECEIPTS = "reports/satisfaction-screen-v1-receipt-preflight.json"
ENGINEERING = "runs/qwen32-sustained-engineering-v1-4096"
ENGINEERING_AUDIT = "reports/qwen32-sustained-engineering-v1-4096-audit.json"
LOCKED_CANDIDATE_MANIFEST = "0298226e20ee1a5c6e0a07bf6b26db6df3e9e25db183709db5151b61055b4772"
LOCKED_DIRECTION_BYTES = "af746df14fedb0023d0ae55eff5ef7704a25d985f31e218e53ac3fa5a43c437e"
LOCKED_NORM = 29.89263847846578


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_receipt(root, role, name):
    path = root / name
    return dict(
        role=role,
        path=name,
        byte_sha256=sha(path),
        content_sha256=digest(read(path)) if path.suffix == ".json" else None,
    )


def locked_direction(root):
    paths = {key: root / name for key, name in CANDIDATE_DOCUMENTS.items()}
    values = {key: read(path) for key, path in paths.items() if key != "directions"}
    direction_sha = sha(paths["directions"])
    require(
        digest(values["manifest"]) == LOCKED_CANDIDATE_MANIFEST and direction_sha == LOCKED_DIRECTION_BYTES,
        "Require the original locked candidate and direction bytes",
    )
    candidate = candidate_binding(
        values["manifest"],
        values["audit"],
        values["analysis"],
        values["check"],
        sha(paths["analysis"]),
        direction_sha,
    )
    with np.load(paths["directions"], allow_pickle=False) as arrays:
        require(
            list(arrays["layers"]) == [15, 31, 47, 57] and list(arrays["positions"]) == ["record", "final"],
            "Original candidate axes changed",
        )
        index = list(arrays["sources"]).index("satisfaction_vs_neutral")
        unit = arrays["unit"][index, 0, 0].tolist()
        norm = float(arrays["contrast_norm"][index, 0, 0])
    selected = next(
        item for item in values["analysis"]["candidates"] if item["contrast"] == "satisfaction_vs_neutral"
    )
    require(norm == selected["a_raw_contrast_norm"] == LOCKED_NORM, "Locked satisfaction norm changed")
    require(len(unit) == 5120, "Require the full pinned-model residual direction")
    return screen.validate_direction(
        dict(
            source="satisfaction_vs_neutral",
            corpus="A",
            layer=15,
            position="record",
            unit=unit,
            unit_sha256=digest(unit),
            norm=norm,
            directions_byte_sha256=direction_sha,
            candidate_source=candidate,
        )
    )


def engineering_certificate(root):
    from audit_sustained_engineering import audit

    run = root / ENGINEERING
    manifest, frozen, receipt = (
        read(run / "manifest.json"),
        read(run / "freeze.json"),
        read(root / ENGINEERING_AUDIT),
    )
    require(
        receipt == audit(run)
        and receipt.get("complete") is True
        and receipt.get("passed") is True
        and receipt.get("integrity_passed") is True
        and receipt.get("cap") == 4096
        and manifest["model"] == capture.MODEL
        and manifest["execution"] == screen.EXECUTION
        and manifest["schedule"] == screen.SCHEDULE
        and manifest["arithmetic"] == engineering.ARITHMETIC
        and manifest["engine_sha256"] == engineering.engine_hashes()
        and frozen["manifest_sha256"] == digest(manifest)
        and frozen["archive_sha256"] == sha(run / "source.tar.gz"),
        "Full-cap engineering must reproduce and bind the unchanged implementation",
    )
    return dict(
        manifest_sha256=digest(manifest),
        audit_sha256=digest(receipt),
        freeze_sha256=digest(frozen),
        archive_sha256=frozen["archive_sha256"],
        scope="Tested C+ one-row engineering; scientific S+ all-row identity, engagement and task checks remain required",
    )


def exposure_target_plan(plan):
    """Adapt only the freshness API; retain all actual worlds and both selection seeds."""
    selection = dict(
        seed=screen.PANELS[0][1],
        panels=deepcopy(plan["selection"]["panels"]),
        worlds=deepcopy(plan["selection"]["worlds"]),
        purpose="exposure_only_actual_worlds_not_regenerated",
    )
    assignments = [deepcopy(row["source_assignment"]) for row in plan["base_assignments"]]
    return dict(
        task_revision=screen.REVISION,
        selection=selection,
        selection_sha256=digest(selection),
        assignments=assignments,
        assignment_sha256=digest(assignments),
    )


def tokenizer_model():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        capture.MODEL["model_id"],
        revision=capture.MODEL["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )

    def encode(messages, enable_thinking=False):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return rendered, tokenizer(rendered, add_special_tokens=False, return_tensors="pt")

    return SimpleNamespace(tokenizer=tokenizer, encode=encode)


def receipt_preflight(plan, model):
    """Exercise all possible source choices using synthetic answer strings, never inference."""
    bindings, stops = [], []
    for row in plan["base_assignments"]:
        if row["kind"] != "choice":
            continue
        source = row["source_assignment"]
        mapping = voluntary.task(source)["action_mapping"]
        for service in voluntary.SERVICES:
            label = next(label for label, value in mapping.items() if value == service)
            raw = f"</think>\n{label}"
            record = voluntary.record_from_thinking_inference(
                source,
                dict(
                    raw=raw,
                    generated_token_ids=model.tokenizer.encode(raw, add_special_tokens=False),
                    ended_with_eos=True,
                ),
            )
            require(
                record["valid"] and record["selected_service"] == service, "Synthetic action parsing changed"
            )
            if service == "stop":
                require(voluntary.receipt_input(record) is None, "Stop unexpectedly requires a payload")
                stops.append(row["base_trial_id"])
            else:
                _, _, binding = receipts.render_receipt(record, model)
                bindings.append(
                    dict(
                        base_trial_id=row["base_trial_id"],
                        selected_service=service,
                        token_binding_sha256=digest(binding),
                        prefix_positions=binding["prefix_positions"],
                    )
                )
    require(len(bindings) == 432 and len(stops) == 144, "Require every possible base choice disposition")
    return dict(
        passed=True,
        model=capture.MODEL,
        generated_model_outputs=False,
        synthetic_answer_only=True,
        base_assignment_sha256=plan["base_assignment_sha256"],
        non_stop_bindings=bindings,
        stop_checks=stops,
        bindings_sha256=digest(dict(non_stop=bindings, stop=stops)),
    )


def build_manifests(plan, preflight, receipt_check, direction, certificate, bindings):
    parent = dict(
        schema_version=1,
        assay="satisfaction_screen",
        task_revision=screen.REVISION,
        phase="development_pilot",
        inference_scope="development",
        model=deepcopy(capture.MODEL),
        execution_gpu="H200",
        execution=deepcopy(screen.EXECUTION),
        source_sha256=source_hash(),
        assignments=[],
        diagnostics=deepcopy(plan["assignments"]),
        assignment_sha256=plan["assignment_sha256"],
        screen_plan=deepcopy(plan),
        screen_plan_sha256=digest(plan),
        generation_preflight=deepcopy(preflight),
        generation_preflight_sha256=digest(preflight),
        receipt_preflight=deepcopy(receipt_check),
        receipt_preflight_sha256=digest(receipt_check),
        max_new_tokens=4096,
        capture_plan=deepcopy(capture.CAPTURE_PLAN),
        arithmetic=deepcopy(engineering.ARITHMETIC),
        schedule=screen.SCHEDULE,
        candidate_source=deepcopy(direction["candidate_source"]),
        engineering_certificate=certificate,
        engine_sha256=engineering.engine_hashes(),
        bound_receipts=bindings,
        analysis_metadata=deepcopy(ANALYSIS_METADATA),
        budget=deepcopy(SCREEN_BUDGET),
        invalid_output_policy="Terminal without replacement or retries; all1152 assignments retained",
    )
    children = []
    for group, arms in capture.GROUPS.items():
        child = deepcopy(parent)
        child["execution_group"] = group
        child["parent_manifest_sha256"] = digest(parent)
        child["diagnostics"] = [row for row in plan["assignments"] if row["arm"] in arms]
        child["assignment_sha256"] = digest(child["diagnostics"])
        capture.validate_manifest(child)
        children.append(child)
    require(
        [row for child in children for row in child["diagnostics"]] == parent["diagnostics"],
        "Child decomposition must preserve the entire parent order",
    )
    return parent, children


def prepare(out, root=ROOT):
    from audit_voluntary_freshness import audit, history_receipt

    root, out = Path(root).resolve(), Path(out).resolve()
    require(out.parent == root / "runs", "Place the parent directly under runs")
    child_paths = [root / "runs" / f"{out.name}-{group.replace('_', '-')}" for group in capture.GROUPS]
    required_new = [out, *child_paths, *(root / name for name in (HISTORY, EXPOSURE, GENERATION, RECEIPTS))]
    require(
        not any(path.exists() for path in required_new),
        "Screen artifacts are immutable; inspect existing preparation",
    )
    require((root / PLAN).is_file(), "Require the prospective scientific plan")
    certificate = engineering_certificate(root)
    direction = locked_direction(root)
    history = history_receipt(audit(root))
    plan = screen.assignment_plan(
        historical_exact_world_ids=history["historical_exact_world_ids"],
        direction_binding=direction,
    )
    exposure = audit(root, plan=exposure_target_plan(plan))
    require(history_receipt(exposure) == history, "Historical inventory changed during preparation")
    require(
        exposure["exact_complete_world_criterion_passed"] and len(exposure["selected_worlds"]) == 24,
        "Require all24 selected worlds with zero historical exact overlap",
    )
    model = tokenizer_model()
    preflight, receipt_check = capture.bind_preflight(plan, model), receipt_preflight(plan, model)
    # All scientific admission and tokenizer checks precede the first write.
    for name, value in (
        (HISTORY, history),
        (EXPOSURE, exposure),
        (GENERATION, preflight),
        (RECEIPTS, receipt_check),
    ):
        write_once(root / name, value)
    documents = dict(
        prospective_plan=PLAN,
        history=HISTORY,
        exposure=EXPOSURE,
        generation_preflight=GENERATION,
        receipt_preflight=RECEIPTS,
        engineering_manifest=f"{ENGINEERING}/manifest.json",
        engineering_audit=ENGINEERING_AUDIT,
        engineering_freeze=f"{ENGINEERING}/freeze.json",
        engineering_archive=f"{ENGINEERING}/source.tar.gz",
    )
    for key, name in CANDIDATE_DOCUMENTS.items():
        documents["candidate_" + ("independent_check" if key == "check" else key)] = name
    bindings = [bound_receipt(root, role, name) for role, name in documents.items()]
    parent, children = build_manifests(plan, preflight, receipt_check, direction, certificate, bindings)
    write_once(out / "manifest.json", parent)
    descriptors = []
    for path, child in zip(child_paths, children, strict=True):
        write_once(path / "manifest.json", child)
        descriptors.append(
            dict(path=str(path), manifest_sha256=digest(child), execution_group=child["execution_group"])
        )
    write_once(out / "children.json", descriptors)
    frozen = freeze(root, out, documents=list(documents.values()))
    return dict(
        run=str(out),
        assigned=1152,
        workers=3,
        manifest_sha256=digest(parent),
        source_sha256=frozen["source_sha256"],
        gpu_launched=False,
        exposure_summary=exposure["summary"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.out))
