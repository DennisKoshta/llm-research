"""Audit prompt-only interventions and score their newly generated final answers.

No initial-prompt logit or probability enters a semantic outcome. Those values
are used only to check numerical sham identity. Engineering readiness and
scientific interpretation readiness are separate gates.
"""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, GenerationConfig

from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.causal_contrasts import (
    BINARY_OUTCOMES,
    CATEGORICAL_TYPES,
    RESPONSE_OUTCOME_SCHEMA,
    descriptor_response_indicators,
    response_objective,
    required_cells,
    source_descriptor,
    validate_natural_response_row,
)
from curiosity_study.causal_metrics import paired_summary, zero_event_upper_bound
from curiosity_study.intervention_integrity import audit_intervention
from curiosity_study.natural_information import (
    CONDITIONING,
    engine_hashes,
    generation_regime,
    make_prefix,
    representative_cohort_binding,
    representative_cohort_checks,
    startup_patch_binding,
)
from curiosity_study.replay_information import intervention
from curiosity_study.target_relevant_information import REVISION as TARGET_REVISION
from curiosity_study.thinking_information import (
    generation_limit,
    inference_request,
    manifest_generation_limit,
    record_from_inference,
    task_module,
)
from analyze_causal_contrasts import assigned_sources
from analyze_replay_interventions import integrity_gate
from prepare_natural_interventions import build_schedule, expected_source_metadata, packing_policy
from prepare_replay_interventions import split_provenance


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic_score(assignment, scored):
    kind = assignment["kind"]
    indicators = descriptor_response_indicators(
        source_descriptor(assignment), scored["parsed_answer"], scored["valid"], scored["correct"]
    )
    if kind in ("new", "repeat", "instrument"):
        return (
            indicators["valid_informative_response"] - indicators["valid_uninformative_response"],
            CATEGORICAL_TYPES[kind],
        )
    if kind in ("memory", "count"):
        return int(scored["valid"] and scored["correct"]), "categorical_correct_indicator_not_logit"
    raise ValueError("Unsupported natural final-answer outcome")


def execution_binding(root, manifest):
    frozen = read(root / "freeze.json")
    require(frozen["manifest_sha256"] == digest(manifest), "Frozen natural manifest mismatch")
    source = Path(frozen["execution_root"]) / "src/curiosity_study"
    require(source.is_dir(), "Frozen natural execution source is unavailable")
    actual = digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))})
    require(
        actual == frozen["source_sha256"] == manifest["source_sha256"], "Frozen natural source hash mismatch"
    )
    require(set(manifest["engine_sha256"]) == set(engine_hashes()), "Incomplete natural engine binding")
    require(
        all(
            hashlib.sha256((source / name).read_bytes()).hexdigest() == sha
            for name, sha in manifest["engine_sha256"].items()
        ),
        "Frozen natural engine file mismatch",
    )


def packing_binding(manifest):
    config = manifest["intervention_schedule"]
    policy = packing_policy(config)
    require(
        manifest["execution"] == dict(backend="torch_batched_v1", batch_size=config["batch_size"]),
        "Natural execution batch differs from the frozen schedule",
    )
    if "packing_policy" not in manifest and "packing_policy_sha256" not in manifest:
        # Older 4/8/16 manifests have one unambiguous layout. Their new audits
        # explicitly record it; an old audit without this binding cannot certify a new run.
        require(config["batch_size"] != 32, "Packed batch 32 requires explicit packing metadata")
        require(
            policy["mode"] != "condition_major_v1",
            "Condition-major execution requires explicit packing metadata",
        )
    else:
        require(
            manifest.get("packing_policy") == policy
            and manifest.get("packing_policy_sha256") == digest(policy),
            "Natural packing policy differs from the frozen schedule",
        )
    if policy["mode"] == "condition_major_v1":
        require(
            manifest.get("padded_prefix_groups") is False,
            "Condition-major execution cannot pad source groups",
        )
    return policy


def startup_binding(startup, manifest):
    """Recompute the full-cohort guard; a two-row or differently packed receipt cannot substitute."""
    passed = startup.get("passed") is True and bool(startup.get("checks")) and all(startup["checks"].values())
    binding = representative_cohort_binding(manifest)
    if binding is None:
        return passed
    require(startup.get("representative_cohort") == binding, "Representative startup cohort binding mismatch")
    require(
        isinstance(startup.get("representative_cohort_evidence"), dict),
        "Missing representative startup evidence",
    )
    directions = Path(manifest["intervention_schedule"]["directions_local"])
    require(
        hashlib.sha256(directions.read_bytes()).hexdigest() == manifest["direction_input"]["sha256"],
        "Representative startup direction artifact mismatch",
    )
    with np.load(directions, allow_pickle=False) as z:
        fitted = {key: z[key] for key in z.files}
    patch = intervention(binding["perturbation_assignment"], fitted, fitted["unit"].shape[-1])
    require(
        startup.get("representative_perturbation") == startup_patch_binding(patch),
        "Representative startup perturbation mismatch",
    )
    checks = representative_cohort_checks(binding, startup["representative_cohort_evidence"])
    basic = {
        "sham_logits_equal",
        "sham_delta_zero",
        "perturbation_applied",
        "other_row_unchanged",
        "fresh_after_exception_equal",
        "hook_cleanup",
        "capture_shape",
        "positions_before_readout",
    }
    require(
        basic <= startup.get("checks", {}).keys(), "Representative startup omitted the base backend guard"
    )
    require(
        all(startup["checks"].get(key) is value for key, value in checks.items()),
        "Representative startup checks disagree with row evidence",
    )
    require(
        startup.get("passed") is all(startup["checks"].values()),
        "Representative startup pass flag disagrees with checks",
    )
    return passed and all(checks.values())


def source_binding(manifest, tokenizer):
    config = manifest["intervention_schedule"]
    require(digest(config) == manifest["intervention_schedule_sha256"], "Natural schedule hash mismatch")
    packing_binding(manifest)
    require(
        generation_limit(manifest) == generation_limit(config),
        "Natural generation allowance differs from the frozen schedule",
    )
    regime = generation_regime(config, manifest["prefixes"])
    if "generation_regime" in manifest:
        require(manifest["generation_regime"] == regime, "Natural source generation regime binding mismatch")
    else:
        require(
            regime["natural_max_new_tokens"] == 2048 and regime["source_matches_natural"],
            "New natural generation regimes require explicit metadata",
        )
    require(
        manifest["phase"] == config["phase"] and manifest["inference_scope"] == config["inference_scope"],
        "Natural stage/scope differs from the frozen schedule",
    )
    source = Path(config["source_collection"])
    parent = read(source / "manifest.json")
    source_limit = manifest_generation_limit(parent)
    audit = read(config["source_audit"])
    expected_source_metadata(config, parent, audit)
    require(
        digest(audit) == manifest["source_audit_sha256"]
        and audit["manifest_sha256"] == digest(parent)
        and audit["complete"]
        and audit.get("max_new_tokens", 2048) == source_limit
        and audit["gates"]["passed"],
        "Source validation audit is not bound and passing",
    )
    require(parent["model"] == manifest["model"], "Source and natural model identities differ")
    require(manifest["excluded_sources"] == [], "Natural stage cannot exclude source completions")
    selected = assigned_sources(manifest)
    low, high = config["world_index_range"]
    expected = [a for a in parent["diagnostics"] if low <= a["index"] < high and a["kind"] in config["kinds"]]
    require(selected == expected, "Natural source selection differs from the frozen schedule")
    require(
        all(
            hashlib.sha256(Path(task_module(a).__file__).read_bytes()).hexdigest()
            == parent["task_module_sha256"]
            for a in selected
        ),
        "Source task module differs from its validated implementation",
    )
    directions = Path(config["directions_local"])
    sha = hashlib.sha256(directions.read_bytes()).hexdigest()
    analysis = read(directions.parent / "analysis.json")
    require(
        sha
        == config["directions_sha256"]
        == manifest["direction_input"]["sha256"]
        == analysis["directions_sha256"],
        "Natural direction artifact mismatch",
    )
    require(
        digest(analysis) == manifest["direction_analysis_sha256"] and analysis["model"] == manifest["model"],
        "Natural direction analysis mismatch",
    )
    require(
        split_provenance(config, parent, analysis, selected) == manifest["split_provenance"],
        "Natural split provenance does not match actual source assignments",
    )
    with np.load(directions, allow_pickle=False) as z:
        fitted = {key: z[key] for key in z.files}
    require(
        build_schedule(config, manifest["prefixes"], fitted) == manifest["diagnostics"],
        "Natural arms differ from the frozen schedule",
    )
    wanted = {a["trial_id"]: a for a in selected}
    children = (
        read(source / "children.json")
        if (source / "children.json").exists()
        else [dict(path=str(source), manifest_sha256=digest(parent))]
    )
    seen = set()
    for child in children:
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        require(digest(cm) == child["manifest_sha256"], "Source child manifest mismatch")
        if path.resolve() != source.resolve():
            require(cm["parent_manifest_sha256"] == digest(parent), "Source child parent mismatch")
        require(cm["source_sha256"] == parent["source_sha256"], "Source child code mismatch")
        records = RunStore(path, cm).completed()
        require(set(records) == {a["trial_id"] for a in cm["diagnostics"]}, "Source child is incomplete")
        for a in cm["diagnostics"]:
            key = a["trial_id"]
            if key not in wanted:
                continue
            require(
                key not in seen and a == wanted[key] and records[key]["assignment"] == a,
                "Natural source record/assignment mismatch",
            )
            seen.add(key)
            record = records[key]
            prefix = manifest["prefixes"][key]
            require(
                make_prefix(record, path.name, digest(cm)) == prefix,
                "Natural prefix is not the exact original prompt",
            )
            request = inference_request(a)
            text = tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            require(
                text == record["inference"]["rendered_prompt"]
                and encoded["input_ids"] == prefix["token_prefix"],
                "Original prompt token IDs differ from the task/template",
            )
            require(text.count("End of record.") == 1, "Ambiguous source record marker")
            end = text.index("End of record.") + len("End of record.") - 1
            markers = [i for i, (s, e) in enumerate(encoded["offset_mapping"]) if s <= end < e]
            require(markers == [prefix["prefix_positions"]["record"]], "Original record position mismatch")
    require(seen == set(wanted) == set(manifest["prefixes"]), "Not every assigned natural prefix is bound")
    return selected


def load_records(root, manifest, tokenizer):
    children = (
        read(root / "children.json")
        if (root / "children.json").exists()
        else [dict(path=str(root), manifest_sha256=digest(manifest))]
    )
    records, locations = {}, {}
    startup_passes, representative_startups = [], []
    size = manifest["execution"]["batch_size"]
    for child in children:
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        require(digest(cm) == child["manifest_sha256"], "Natural child manifest hash mismatch")
        if path.resolve() != root.resolve():
            require(cm["parent_manifest_sha256"] == digest(manifest), "Natural child parent mismatch")
            expected = [
                a
                for ci in child["parent_cohorts"]
                for a in manifest["diagnostics"][ci * size : (ci + 1) * size]
            ]
            require(cm["diagnostics"] == expected, "Natural child cohort schedule mismatch")
        for key in (
            "source_sha256",
            "engine_sha256",
            "model",
            "execution",
            "prefixes_sha256",
            "conditioning",
        ):
            require(cm[key] == manifest[key], f"Natural child binding mismatch: {key}")
        for key in (
            "packing_policy",
            "packing_policy_sha256",
            "padded_prefix_groups",
            "max_new_tokens",
            "generation_regime",
        ):
            require(cm.get(key) == manifest.get(key), f"Natural child binding mismatch: {key}")
        require(digest(cm["prefixes"]) == cm["prefixes_sha256"], "Natural child prefix binding mismatch")
        require(
            digest(cm["diagnostics"]) == cm["assignment_sha256"], "Natural child assignment binding mismatch"
        )
        rr = RunStore(path, cm).completed()
        require(set(rr) == {a["trial_id"] for a in cm["diagnostics"]}, "Natural child incomplete")
        require(not records.keys() & rr.keys(), "Natural measurement appears in multiple children")
        physical = []
        startup_cache = {}
        for file in (path / "cohorts").glob("*/*.json"):
            envelope = read(file)
            offset = int(file.stem)
            require(offset % size == 0, "Natural cohort starts outside its frozen boundary")
            expected = cm["diagnostics"][offset : offset + size]
            require(
                [r["trial_id"] for r in envelope["records"]] == [a["trial_id"] for a in expected],
                "Natural physical cohort members differ from the frozen schedule",
            )
            physical.extend(r["trial_id"] for r in envelope["records"])
            attempt = file.parent.name
            if attempt not in startup_cache:
                startup = read(path / "natural_integrity" / f"{attempt}.json")
                startup_cache[attempt] = startup_binding(startup, cm)
                if startup.get("representative_cohort") is not None:
                    representative_startups.append(
                        dict(
                            child_manifest_sha256=digest(cm),
                            startup_receipt_sha256=digest(startup),
                            representative_cohort=startup["representative_cohort"],
                            passed=startup_cache[attempt],
                        )
                    )
            startup_passes.append(startup_cache[attempt])
            environment = read(path / "environments" / f"{attempt}.json")
            require(
                all(environment[k] == manifest["model"][k] for k in ("model_id", "revision"))
                and environment["dtype"] == "bfloat16"
                and environment["attention"] == "eager"
                and environment["template_sha256"] == digest(tokenizer.chat_template),
                "Natural execution environment differs from its pinned model/template",
            )
        require(
            len(physical) == len(rr) and set(physical) == set(rr), "Natural cohort physical coverage mismatch"
        )
        records.update(rr)
        locations.update({key: path for key in rr})
    require(set(records) == {a["trial_id"] for a in manifest["diagnostics"]}, "Natural parent incomplete")
    return records, locations, bool(startup_passes) and all(startup_passes), representative_startups


def capture_digest(record, root, manifest, cache):
    if not manifest["capture_layers"]:
        return None
    meta = record["activation_artifact"]
    require(
        meta.get("max_new_tokens", 2048) == generation_limit(manifest),
        "Natural capture generation allowance mismatch",
    )
    file = root / meta["path"]
    if file not in cache:
        require(
            hashlib.sha256(file.read_bytes()).hexdigest() == meta["sha256"],
            "Natural capture file hash mismatch",
        )
        with np.load(file, allow_pickle=False) as z:
            hidden = z["hidden"]
            require(
                hidden.ndim == 4
                and hidden.shape[1:3] == (len(manifest["capture_layers"]), 2)
                and np.isfinite(hidden).all(),
                "Natural initial capture shape/values invalid",
            )
            require(
                list(z["layers"]) == manifest["capture_layers"]
                and list(z["positions"]) == ["record", "final"],
                "Natural capture coordinates mismatch",
            )
            cache[file] = dict(
                sha256=meta["sha256"],
                trial_ids=list(z["trial_ids"]),
                rows=[hashlib.sha256(row.tobytes()).hexdigest() for row in hidden],
            )
    data = cache[file]
    require(
        data["sha256"] == meta["sha256"] and data["trial_ids"][meta["row"]] == record["trial_id"],
        "Natural capture row binding mismatch",
    )
    require(
        meta["layers"] == manifest["capture_layers"] and meta["positions"] == ["record", "final"],
        "Natural capture metadata mismatch",
    )
    return data["rows"][meta["row"]]


def summarize_worlds(worlds, revision, stratum=None):
    cells = required_cells(revision)
    complete = {
        key: rows
        for key, rows in worlds.items()
        if len(rows) == len(cells) and {tuple(row["cell"]) for row, _ in rows} == cells
    }
    selected = {
        key: [(r, b) for r, b in rows if stratum is None or all(r[k] == v for k, v in stratum.items())]
        for key, rows in complete.items()
    }
    clusters, failures = defaultdict(list), defaultdict(bool)
    for (seed, index, _template), rows in selected.items():
        if not rows:
            continue
        clusters[(seed, index)].append(
            [
                np.mean([r["margin"] - b["margin"] for r, b in rows]),
                np.mean([int(r["correct"]) - int(b["correct"]) for r, b in rows]),
                np.mean([int(r["valid"]) - int(b["valid"]) for r, b in rows]),
                np.mean([r["margin"] for r, _ in rows]),
                np.mean([b["margin"] for _, b in rows]),
            ]
        )
        failures[(seed, index)] |= any(b["correct"] and not r["correct"] for r, b in rows)
    values = [np.mean(rows, axis=0) for rows in clusters.values()]
    allrows = [r for rows in selected.values() for r, _ in rows]
    assigned_rows = [
        r
        for rows in worlds.values()
        for r, _ in rows
        if stratum is None or all(r[k] == v for k, v in stratum.items())
    ]
    representative = next((r for rows in worlds.values() for r, _ in rows), None)
    binary = {}
    for outcome in BINARY_OUTCOMES:
        if representative is None or representative[outcome] is None:
            binary[outcome] = None
            continue
        differences = defaultdict(list)
        baseline_positives = positives = 0
        for (seed, index, _template), rows in selected.items():
            if not rows:
                continue
            differences[(seed, index)].append(np.mean([int(r[outcome]) - int(b[outcome]) for r, b in rows]))
            positives += sum(int(r[outcome]) for r, _ in rows)
            baseline_positives += sum(int(b[outcome]) for _, b in rows)
        denominator = len(allrows)
        binary[outcome] = dict(
            assigned_trials=len(assigned_rows),
            analyzed_trials=denominator,
            positive_responses=positives,
            baseline_positive_responses=baseline_positives,
            response_rate=positives / denominator if denominator else None,
            baseline_response_rate=baseline_positives / denominator if denominator else None,
            paired_probability_change=paired_summary([float(np.mean(v)) for v in differences.values()]),
            denominator_policy="All assigned responses in complete worlds, including invalids and distractors; never conditional on validity",
        )
    return dict(
        stratum=stratum,
        eligible_worlds=len({key[:2] for key in worlds}),
        eligible_world_templates=len(worlds),
        complete_worlds=len(clusters),
        complete_world_templates=len(complete),
        required_cells_per_world_template=len(cells),
        branches=len(allrows),
        assigned_branches=len(assigned_rows),
        binary_response_outcomes=binary,
        paired_semantic_response_change=paired_summary([v[0] for v in values]),
        paired_accuracy_change=paired_summary([v[1] for v in values]),
        paired_validity_change=paired_summary([v[2] for v in values]),
        mean_semantic_response_score=float(np.mean([v[3] for v in values])) if values else None,
        baseline_mean_semantic_response_score=float(np.mean([v[4] for v in values])) if values else None,
        correct=sum(r["correct"] for r in allrows),
        valid=sum(r["valid"] for r in allrows),
        cap_hits=sum(r["cap_hit"] for r in allrows),
        new_failure_worlds=sum(failures.values()),
        one_sided_95_upper_new_failure_probability_if_zero=zero_event_upper_bound(len(clusters))
        if not any(failures.values())
        else None,
    )


def baseline_readiness(baselines, manifest):
    criteria = manifest["intervention_schedule"].get("baseline_validation_criteria")
    if criteria is not None:
        require(
            set(criteria) == {"minimum_distinct_items_per_cell", "minimum_accuracy", "minimum_valid_format"}
            and type(criteria["minimum_distinct_items_per_cell"]) is int
            and criteria["minimum_distinct_items_per_cell"] >= 1
            and 0 <= criteria["minimum_accuracy"] <= 1
            and 0 <= criteria["minimum_valid_format"] <= 1,
            "Invalid frozen natural baseline criteria",
        )
    groups = defaultdict(list)
    for row in baselines.values():
        groups[(row["task_revision"], row["family"], row["kind"], tuple(row["cell"]))].append(row)
    cells = []
    for key, rows in sorted(groups.items()):
        n = len(rows)
        worlds = len({(r["seed"], r["index"]) for r in rows})
        correct, valid = sum(r["correct"] for r in rows), sum(r["valid"] for r in rows)
        cells.append(
            dict(
                task_revision=key[0],
                family=key[1],
                kind=key[2],
                cell=key[3],
                assigned=n,
                world_indices=worlds,
                correct=correct,
                valid=valid,
                cap_hits=sum(r["cap_hit"] for r in rows),
                passed=criteria is not None
                and worlds >= criteria["minimum_distinct_items_per_cell"]
                and correct / n >= criteria["minimum_accuracy"]
                and valid / n >= criteria["minimum_valid_format"],
            )
        )
    return dict(
        criteria=criteria,
        criteria_frozen=criteria is not None,
        passed=bool(cells) and all(c["passed"] for c in cells),
        cells=cells,
    )


def previous_engineering(manifest):
    config = manifest["intervention_schedule"]
    policy = packing_binding(manifest)
    path = config.get("natural_engineering_audit")
    sha = config.get("natural_engineering_audit_sha256")
    if path is None or sha is None:
        return dict(passed=False, reason="A previous passing 32B natural engineering audit is not bound")
    report = read(Path(path))
    passed = (
        digest(report) == sha
        and report.get("assay") == "natural_information"
        and report.get("phase") == "engineering"
        and report.get("engineering_passed") is True
        and bool(report.get("engineering_checks"))
        and all(report["engineering_checks"].values())
        and report.get("integrity_passed") is True
        and report.get("complete") is True
        and report.get("model") == manifest["model"]
        and report.get("engine_sha256") == manifest["engine_sha256"]
        and report.get("execution") == manifest["execution"]
        and report.get("packing_policy") == policy
        and report.get("packing_policy_sha256") == digest(policy)
        and report.get("manifest_sha256") != digest(manifest)
        and report.get("max_new_tokens") == generation_limit(manifest)
        and report.get("assigned_sources", 0) >= 4
        and report.get("maximum_baseline_generated_tokens", 0) >= 128
    )
    if policy["mode"] == "condition_major_v1":
        receipts = report.get("representative_startup_audits", [])
        passed = (
            passed
            and bool(receipts)
            and all(
                receipt.get("passed") is True
                and receipt.get("representative_cohort", {}).get("packing_policy") == policy
                and receipt["representative_cohort"].get("packing_policy_sha256") == digest(policy)
                and receipt["representative_cohort"].get("batch_size") == manifest["execution"]["batch_size"]
                and receipt["representative_cohort"].get("max_new_tokens") == 16
                for receipt in receipts
            )
        )
    return dict(
        passed=bool(passed),
        audit_sha256=digest(report),
        manifest_sha256=report.get("manifest_sha256"),
        execution=report.get("execution"),
        packing_policy=report.get("packing_policy"),
        max_new_tokens=report.get("max_new_tokens"),
        reason="Bound prior model- and execution-specific engineering audit"
        if passed
        else "Prior engineering audit binding or criteria failed",
    )


def analyze(root, out, tokenizer=None, eos_token_ids=None):
    root, out = Path(root), Path(out)
    manifest = read(root / "manifest.json")
    limit = generation_limit(manifest)
    require(
        manifest["assay"] == "natural_information" and manifest["conditioning"] == CONDITIONING,
        "This audit requires the separate original-prompt natural assay",
    )
    require(
        manifest.get("greedy") is False,
        "Natural audit requires sampled fresh generation with its frozen 2048 or 4096 token allowance",
    )
    require(
        manifest["assignments"] == []
        and digest(manifest["diagnostics"]) == manifest["assignment_sha256"]
        and digest(manifest["prefixes"]) == manifest["prefixes_sha256"],
        "Natural assignment/prefix binding mismatch",
    )
    require(
        len({a["trial_id"] for a in manifest["diagnostics"]}) == len(manifest["diagnostics"]),
        "Duplicate natural measurement assignment",
    )
    execution_binding(root, manifest)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            **dict(
                pretrained_model_name_or_path=manifest["model"]["model_id"],
                revision=manifest["model"]["revision"],
                trust_remote_code=False,
            )
        )
    if eos_token_ids is None:
        generation = GenerationConfig.from_pretrained(
            manifest["model"]["model_id"], revision=manifest["model"]["revision"]
        )
        eos = generation.eos_token_id
        eos_token_ids = {eos} if isinstance(eos, int) else set(eos or [])
    # Fast-tokenizer vocabulary-size lookup can rebuild its added-token map.
    # The tokenizer is fixed throughout this audit; resolve the bound once.
    vocabulary_size = len(tokenizer)
    source_assignments = source_binding(manifest, tokenizer)
    records, locations, startup_passed, representative_startups = load_records(root, manifest, tokenizer)
    compact, baselines, initial, captures = [], {}, {}, {}
    capture_cache = {}
    failures = []
    for a in manifest["diagnostics"]:
        r = records[a["trial_id"]]
        prefix = manifest["prefixes"][a["prefix_id"]]
        sa = prefix["assignment"]
        require(r["assignment"] == a and r["terminal"], "Natural record assignment mismatch")
        failure = r.get("failure")
        if failure:
            require(failure == "execution_error", "Unknown natural terminal failure")
            failures.append(a["trial_id"])
            completion = dict(raw="", generated_token_ids=[], ended_with_eos=False)
            scored = record_from_inference(sa, completion)
            change, captured, cap = None, None, False
        else:
            require(
                r["source_assignment"] == sa
                and r["source_trial_id"] == prefix["source_trial_id"]
                and r["token_prefix_sha256"] == prefix["token_prefix_sha256"]
                and r["conditioning"] == CONDITIONING,
                "Natural record source mismatch",
            )
            measured = r["measurement"]
            ids, positions = prefix["token_prefix"], prefix["prefix_positions"]
            require(
                measured["unpadded_input_ids"] == ids
                and measured["token_prefix_sha256"] == prefix["token_prefix_sha256"] == digest(ids)
                and measured["unpadded_token_positions"] == positions,
                "Measured natural token prefix differs from its original prompt",
            )
            require(
                measured["exact_token_replay"] and measured["hooks_remaining"] == 0,
                "Natural hooks or exact-prefix invariant failed",
            )
            pad = len(measured["input_ids"]) - len(ids)
            require(
                pad >= 0
                and measured["input_ids"] == [tokenizer.pad_token_id] * pad + ids
                and measured["attention_mask"] == [0] * pad + [1] * len(ids)
                and measured["position_ids"] == [0] * pad + list(range(len(ids))),
                "Natural padded inputs/positions mismatch",
            )
            completion = measured["completion"]
            generated = completion["generated_token_ids"]
            require(
                completion["sampling"]
                == dict(seed=prefix["seed"], temperature=0.7, top_p=1.0, max_new_tokens=limit)
                and completion["intervention_schedule"] == "prefill_only; no active hooks during decoding",
                "Natural generation policy mismatch",
            )
            require(
                1 <= len(generated) <= limit
                and all(type(t) is int and 0 <= t < vocabulary_size for t in generated),
                "Invalid natural generated token IDs",
            )
            require(
                tokenizer.decode(generated, skip_special_tokens=True) == completion["raw"],
                "Natural raw completion disagrees with its generated tokens",
            )
            require(
                completion["ended_with_eos"] == (generated[-1] in eos_token_ids)
                and not any(t in eos_token_ids for t in generated[:-1]),
                "Natural EOS termination mismatch",
            )
            require(
                completion["ended_with_eos"] or len(generated) == limit,
                "Natural completion stopped before EOS or cap",
            )
            scored = record_from_inference(sa, completion)
            for field in (
                "answer",
                "parsed_answer",
                "expected",
                "valid",
                "correct",
                "task",
                "thinking_closed",
            ):
                require(digest(r[field]) == digest(scored[field]), f"Natural rescoring mismatch: {field}")
            cap = not completion["ended_with_eos"] and len(generated) == limit
            require(
                r["cap_hit"] == cap
                and r.get("generation_max_new_tokens", 2048) == limit
                and r["closing_delimiter_count"] == completion["raw"].count("</think>"),
                "Natural cap/closure metadata mismatch",
            )
            change = measured["intervention"]
            if a["arm"] == "baseline":
                require(change is None, "Natural baseline contains an intervention")
            else:
                require(
                    change["layer"] == a["layer"]
                    and change["position"] == a["position"]
                    and change["token_index"] == pad + positions[a["position"]],
                    "Natural intervention location mismatch",
                )
            labels = set(prefix["labels"])
            require(
                set(measured["label_logits"]) == set(measured["label_probabilities_temperature_1"]) == labels,
                "Natural initial label support mismatch",
            )
            require(
                np.isfinite(list(measured["label_logits"].values())).all()
                and np.isfinite(list(measured["label_probabilities_temperature_1"].values())).all(),
                "Natural initial label readouts must be finite",
            )
            probabilities = list(measured["label_probabilities_temperature_1"].values())
            require(
                all(0 <= p <= 1 for p in probabilities) and sum(probabilities) <= 1.00001,
                "Natural initial label probabilities are outside their valid range",
            )
            initial[a["trial_id"]] = (measured["label_logits"], measured["label_probabilities_temperature_1"])
            captured = capture_digest(r, locations[a["trial_id"]], manifest, capture_cache)
        value, metric = semantic_score(sa, scored)
        descriptor = source_descriptor(sa)
        response_outcomes = descriptor_response_indicators(
            descriptor,
            scored["parsed_answer"],
            scored["valid"],
            scored["correct"],
        )
        require(
            scored["task"]["information_bits"] == source_descriptor(sa)["information_bits"],
            "Natural semantic stratum disagrees with the finite-world oracle",
        )
        if sa["task_revision"] == TARGET_REVISION:
            require(
                scored["task"]["expected_target_information_gain_bits"]
                == descriptor["expected_target_information_gain_bits"]
                and scored["task"]["expected_world_information_gain_bits"] == 1,
                "Natural target-response oracle differs from its separately bound target gain",
            )
        c = dict(
            trial_id=a["trial_id"],
            prefix_id=a["prefix_id"],
            **descriptor,
            arm=a["arm"],
            layer=a["layer"],
            position=a.get("capture_position"),
            direction=a["direction_name"],
            dose=a["dose"],
            dose_multiplier=a["dose_multiplier"],
            condition=a["condition"],
            patch_mode=a.get("patch_mode", "add"),
            margin=value,
            margin_type=metric,
            margin_units="categorical response score; not logits",
            response_outcome_schema=RESPONSE_OUTCOME_SCHEMA,
            **response_outcomes,
            valid=scored["valid"],
            correct=scored["correct"],
            answer=scored["parsed_answer"],
            thinking_closed=scored["thinking_closed"],
            cap_hit=cap,
            generated_tokens=len(completion["generated_token_ids"]),
            completion_sha256=digest(completion["generated_token_ids"]),
            intervention=change,
            failure=failure,
        )
        validate_natural_response_row(c)
        compact.append(c)
        captures[a["trial_id"]] = captured
        if a["arm"] == "baseline":
            require(a["prefix_id"] not in baselines, "Duplicate natural baseline")
            baselines[a["prefix_id"]] = c
    require(
        set(baselines) == set(manifest["prefixes"]),
        "A baseline is required for every assigned natural prefix",
    )
    groups = defaultdict(lambda: defaultdict(list))
    sham_sources, sham_deltas, sham_probs, sham_logits, sham_tokens, sham_captures, engagement = (
        set(),
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in compact:
        base = baselines[row["prefix_id"]]
        if row["arm"] == "baseline":
            continue
        if row["arm"] == "sham" and not row["failure"] and not base["failure"]:
            sham_sources.add(row["prefix_id"])
            sham_deltas.append(row["intervention"]["delta_norm"])
            b, s = initial[base["trial_id"]], initial[row["trial_id"]]
            sham_logits.append(b[0] == s[0])
            sham_probs.extend(abs(b[1][label] - s[1][label]) for label in b[1])
            sham_tokens.append(row["completion_sha256"] == base["completion_sha256"])
            sham_captures.append(captures[row["trial_id"]] == captures[base["trial_id"]])
        elif row["dose"] and not row["failure"]:
            engagement.append(audit_intervention(row, row["intervention"]))
        if row["direction"] == "padding_control":
            continue
        key = (
            row["family"],
            row["kind"],
            row["layer"],
            row["position"],
            row["direction"],
            row["dose_multiplier"],
            row["task_revision"],
        )
        groups[key][(row["seed"], row["index"], row["template"])].append((row, base))
    metrics = []
    for key, worlds in sorted(groups.items()):
        gain_field = (
            "expected_target_information_gain_bits" if key[-1] == TARGET_REVISION else "information_bits"
        )
        strata = [None, {gain_field: 0}, {gain_field: 1}]
        if key[6] == "count-competence-v1":
            strata += [{"correct_count": n} for n in (1, 2, 4)]
        for stratum in strata:
            metrics.append(
                dict(
                    family=key[0],
                    kind=key[1],
                    layer=key[2],
                    position=key[3],
                    direction=key[4],
                    dose_multiplier=key[5],
                    task_revision=key[6],
                    response_objective_basis=response_objective({"task_revision": key[6]}),
                    response_gain_field=gain_field,
                    margin_units="categorical response score; not logits",
                    **summarize_worlds(worlds, key[6], stratum),
                )
            )
    integrity = integrity_gate(
        manifest, baselines, sham_sources, sham_deltas, sham_probs, sham_logits, sham_tokens, engagement
    )
    source_worlds = defaultdict(list)
    for a in source_assignments:
        d = source_descriptor(a)
        source_worlds[
            (d["task_revision"], d["family"], d["kind"], d["seed"], d["index"], d["template"])
        ].append(d["cell"])
    full_worlds = all(
        len(cells) == len(required_cells(key[0])) and set(cells) == required_cells(key[0])
        for key, cells in source_worlds.items()
    )
    integrity["checks"].update(
        natural_stage_binding=True,
        exact_original_prompt_and_generated_text_verified=True,
        all_assigned_prefixes_included=True,
        no_execution_failures=not failures,
        startup_engineering_passed=startup_passed,
        sham_initial_capture_identity=bool(sham_captures) and all(sham_captures),
    )
    integrity["passed"] = all(integrity["checks"].values())
    maximum = max((r["generated_tokens"] for r in baselines.values()), default=0)
    engineering_checks = dict(
        execution_integrity=integrity["passed"],
        execution_packing_bound=True,
        phase_is_engineering=manifest["phase"] == "engineering",
        pinned_32b_model=manifest["model"]["model_id"] == "Qwen/Qwen3-32B",
        four_distinct_source_prefixes=len(baselines) >= 4,
        full_frozen_generation_allowance=limit in (2048, 4096),
        baseline_continuation_at_least_128_tokens=maximum >= 128,
    )
    readiness = baseline_readiness(baselines, manifest)
    previous = (
        previous_engineering(manifest)
        if manifest["phase"] != "engineering"
        else dict(passed=False, reason="Engineering cannot certify its own scientific interpretation")
    )
    interpretation_checks = dict(
        execution_integrity=integrity["passed"],
        scientific_phase=manifest["phase"] != "engineering",
        prior_32b_engineering_bound=previous["passed"],
        complete_source_worlds=full_worlds,
        frozen_baseline_competence_and_validity_gates=readiness["passed"],
    )
    result = dict(
        assay="natural_information",
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        engine_sha256=manifest["engine_sha256"],
        model=manifest["model"],
        execution=manifest["execution"],
        packing_policy=packing_policy(manifest["intervention_schedule"]),
        packing_policy_sha256=digest(packing_policy(manifest["intervention_schedule"])),
        phase=manifest["phase"],
        inference_scope=manifest["inference_scope"],
        split_provenance=manifest["split_provenance"],
        max_new_tokens=limit,
        generation_regime=generation_regime(manifest["intervention_schedule"], manifest["prefixes"]),
        assigned_measurements=len(compact),
        compact_outcomes_sha256=digest(compact),
        response_outcome_schema=RESPONSE_OUTCOME_SCHEMA,
        primary_response_outcomes=[
            "valid_informative_response",
            "valid_uninformative_response",
            "valid_counterfactual_response",
        ],
        response_objective_bases=sorted({response_objective(row) for row in compact}),
        target_count_control_note="Target-task count answers are always four; these are fixed-answer diagnostics, not balanced count-competence controls."
        if any(row["task_revision"] == TARGET_REVISION and row["kind"] == "count" for row in compact)
        else None,
        assigned_sources=manifest["assigned_sources"],
        eligible_sources=len(baselines),
        excluded_sources=[],
        execution_failure_trials=failures,
        complete=True,
        integrity_passed=integrity["passed"],
        integrity_checks=integrity["checks"],
        engineering_passed=all(engineering_checks.values()),
        engineering_checks=engineering_checks,
        engineering_accuracy_policy="Baseline accuracy, validity, and caps are reported; accuracy is not an engineering pass criterion.",
        maximum_baseline_generated_tokens=maximum,
        baseline_readiness=readiness,
        previous_engineering=previous,
        eligible_for_interpretation=all(interpretation_checks.values()),
        interpretation_checks=interpretation_checks,
        metrics=metrics,
        nonzero_interventions=len(engagement),
        maximum_relative_perturbation=max((e["relative_delta_norm"] for e in engagement), default=0),
        semantic_outcome_rule="Only newly generated final answers after the last closing think delimiter. Primary outcomes are separate valid-informative, valid-uninformative, and valid-counterfactual indicators. New/repeat reverse Yes/No polarity; instrument uses report/stop with the exact menu permutation. Counterfactual requires a valid informative response when source gain is zero or a valid uninformative response when source gain is one. Invalids and distractors score zero, retaining assigned denominators. These indicators are unsupported (null) for memory/count, which retain valid/correct outcomes. Legacy descriptive margin is informative minus uninformative for judgments/actions, or correct for memory/count. An apparent opposite-information change in that signed margin can result from invalidation and is not counterfactual evidence. No initial logit or probability enters a semantic metric.",
        world_sampling_rule="Require all four information cells or all six count-competence cells before stratification; average templates within seed/index, then resample whole underlying worlds. Target-response strata use expected_target_information_gain_bits; information_bits retains full-world gain. Count-competence strata use the same complete six-cell cases; target-task count uses four cells and a constant answer.",
        interpretation="Execution integrity alone does not establish scientific readiness. A distinct previous passing 32B natural engineering audit and frozen baseline competence/validity gates are required for scientific interpretation. Engineering runs never certify themselves. Categorical response-score changes are not logit margins or evidence of subjective experience.",
    )
    if manifest.get("packing_policy", {}).get("mode") == "condition_major_v1":
        result["representative_startup_audits"] = representative_startups
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "analysis.json", result)
    write_once(out / "compact_outcomes.json", compact)
    write_once(out / "target_engagement.json", engagement)
    print(
        {
            key: result[key]
            for key in (
                "complete",
                "integrity_passed",
                "engineering_passed",
                "eligible_for_interpretation",
                "assigned_sources",
            )
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.out)
    if not result["integrity_passed"]:
        raise SystemExit("Natural execution integrity failed; scientific interpretation blocked")
