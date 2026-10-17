"""Audit fixed-prefix interventions and aggregate paired outcomes by entire world."""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from curiosity_study.artifacts import read, write_once, digest, RunStore
from curiosity_study.thinking_information import task_module
from curiosity_study.causal_metrics import margin, paired_summary, zero_event_upper_bound
from curiosity_study.intervention_integrity import audit_intervention

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_replay_interventions import audit_inspected_development_registry  # noqa: E402
from source_admission import audit_source_admission  # noqa: E402


def cell_identity(assignment):
    revision = assignment["task_revision"]
    if revision == "count-competence-v1":
        return assignment["correct_count"], assignment["query"]
    if revision == "relational-information-v1":
        return assignment["correlated"], assignment["query"]
    if revision == "equal-uncertainty-v1":
        return assignment["known"][0], assignment["query"]
    raise ValueError("Unknown task revision cannot define replay cells")


def summarize_worlds(worlds, task_revision, correct_count=None):
    """Require complete cases per template, then resample underlying worlds."""
    if task_revision == "count-competence-v1":
        cells = {(count, query) for count in (1, 2, 4) for query in (0, 1)}
    elif task_revision in ("equal-uncertainty-v1", "relational-information-v1"):
        if correct_count is not None:
            raise ValueError("Count strata only apply to the count-competence control")
        cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    else:
        raise ValueError("Unknown task revision cannot define complete worlds")
    if correct_count is not None and correct_count not in (1, 2, 4):
        raise ValueError("Unknown count stratum")
    complete = {
        key: rows
        for key, rows in worlds.items()
        if len(rows) == len(cells) and {row["cell"] for row, _ in rows} == cells
    }
    # Every count stratum uses the same six-cell-complete world/template cases.
    selected = {
        key: [(row, base) for row, base in rows if correct_count is None or row["cell"][0] == correct_count]
        for key, rows in complete.items()
    }
    clustered = defaultdict(list)
    new_failures = defaultdict(bool)
    for (seed, index, _template), rows in selected.items():
        values = (
            np.mean([row["margin"] - base["margin"] for row, base in rows]),
            np.mean([int(row["correct"]) - int(base["correct"]) for row, base in rows]),
            np.mean([int(row["valid"]) - int(base["valid"]) for row, base in rows]),
        )
        clustered[(seed, index)].append(values)
        new_failures[(seed, index)] |= any(base["correct"] and not row["correct"] for row, base in rows)
    # Templates of one latent world never become independent bootstrap units.
    values = [np.mean(rows, axis=0) for rows in clustered.values()]
    allrows = [row for rows in selected.values() for row, _ in rows]
    new_failure_worlds = sum(new_failures.values())
    return dict(
        correct_count=correct_count,
        templates=sorted({key[2] for key in worlds}),
        template_aggregation="mean of complete template-specific paired means within each underlying world",
        eligible_worlds=len({(key[0], key[1]) for key in worlds}),
        eligible_world_templates=len(worlds),
        complete_worlds=len(clustered),
        complete_world_templates=len(complete),
        required_cells_per_world_template=len(cells),
        branches=len(allrows),
        paired_margin_change=paired_summary([v[0] for v in values]),
        paired_accuracy_change=paired_summary([v[1] for v in values]),
        paired_validity_change=paired_summary([v[2] for v in values]),
        correct=sum(r["correct"] for r in allrows),
        valid=sum(r["valid"] for r in allrows),
        cap_hits=sum(r["cap_hit"] for r in allrows),
        new_failure_worlds=new_failure_worlds,
        one_sided_95_upper_new_failure_probability_if_zero=zero_event_upper_bound(len(clustered))
        if new_failure_worlds == 0
        else None,
    )


def integrity_gate(
    manifest, baselines, sham_sources, sham_deltas, sham_differences, sham_logits, sham_tokens, engagement
):
    provenance = manifest.get("split_provenance")
    scope = manifest.get("inference_scope")
    provenance_bound = (
        isinstance(provenance, dict)
        and provenance.get("schema_version") == 1
        and manifest.get("split_provenance_sha256") == digest(provenance)
        and scope in ("development", "confirmation")
        and provenance.get("inference_scope") == scope
        and all(
            isinstance(provenance.get(key), str) and provenance[key]
            for key in (
                "direction_analysis_sha256",
                "direction_extraction_manifest_sha256",
                "evaluation_manifest_sha256",
            )
        )
        and provenance.get("direction_analysis_sha256") == manifest.get("direction_analysis_sha256")
        and provenance.get("direction_extraction_manifest_sha256")
        == manifest.get("direction_extraction_manifest_sha256")
        and provenance.get("evaluation_manifest_sha256") == manifest.get("source_parent_manifest_sha256")
        and provenance.get("assigned_sources") == manifest.get("assigned_sources")
    )
    registry_bound = audit_inspected_development_registry(manifest)
    confirmation_disjoint = scope == "development"
    if scope == "confirmation" and provenance_bound:
        overlaps = provenance.get("overlaps")
        confirmation_disjoint = (
            provenance.get("extraction_binding_verified") is True
            and provenance.get("confirmation_eligible") is True
            and registry_bound
            and isinstance(overlaps, dict)
            and set(overlaps) == {"extraction", "development"}
            and all(v.get("world_ids") == [] and v.get("semantic_world_ids") == [] for v in overlaps.values())
        )
    checks = dict(
        nonempty_sources=bool(baselines),
        sham_for_every_source=bool(baselines) and set(sham_sources) == set(baselines),
        sham_zero_delta=bool(sham_deltas) and all(value == 0 for value in sham_deltas),
        sham_probability_identity=bool(sham_differences) and all(value == 0 for value in sham_differences),
        sham_logit_identity=bool(sham_logits) and all(sham_logits),
        sham_token_identity=bool(sham_tokens) and all(sham_tokens),
        nonzero_interventions_present=bool(engagement),
        intervention_implementation_verified=bool(engagement)
        and all(e.get("implementation_verified", e["signed_engagement"]) for e in engagement),
        finite_perturbations=bool(engagement)
        and all(
            np.isfinite(
                [e[k] for k in ("requested_dose", "projection_delta", "delta_norm", "relative_delta_norm")]
            ).all()
            and e["delta_norm"] >= 0
            and e["relative_delta_norm"] >= 0
            for e in engagement
        ),
        split_provenance_bound=provenance_bound,
        inspected_development_registry_bound=registry_bound,
        confirmation_nonoverlap=confirmation_disjoint,
        source_admission_bound=audit_source_admission(manifest),
    )
    return dict(passed=all(checks.values()), checks=checks)


def audit_norm_matching(rows):
    by_key = {(r["prefix_id"], r["condition"]): r for r in rows}
    results = []
    for row in rows:
        if row["patch_mode"] != "matched_add":
            continue
        target = by_key.get((row["prefix_id"], row["matched_interchange_condition"]))
        if target is None or target["patch_mode"] != "project":
            raise ValueError("A matched control must bind its within-prefix projection intervention")
        a, b = row["intervention"], target["intervention"]
        results.append(
            dict(
                trial_id=row["trial_id"],
                target_trial_id=target["trial_id"],
                scalar_difference=a["requested_scalar_delta"] - b["requested_scalar_delta"],
                requested_norm_difference=a["requested_delta_norm"] - b["requested_delta_norm"],
                achieved_norm_difference=a["delta_norm"] - b["delta_norm"],
                achieved_norm_ratio=a["delta_norm"] / b["delta_norm"] if b["delta_norm"] else None,
                passed=bool(
                    np.isclose(a["requested_scalar_delta"], b["requested_scalar_delta"], rtol=1e-5, atol=1e-5)
                    and np.isclose(a["requested_delta_norm"], b["requested_delta_norm"], rtol=1e-3, atol=1e-4)
                ),
            )
        )
    return results


def analyze(root, out):
    manifest = read(root / "manifest.json")
    children = read(root / "children.json")
    records = {}
    size = manifest["execution"]["batch_size"]
    for child in children:
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        assert digest(cm) == child["manifest_sha256"]
        assert cm["parent_manifest_sha256"] == digest(manifest)
        assert cm["source_sha256"] == manifest["source_sha256"]
        expected = [
            a for ci in child["parent_cohorts"] for a in manifest["diagnostics"][ci * size : (ci + 1) * size]
        ]
        assert cm["diagnostics"] == expected
        rs = RunStore(path, cm).completed()
        assert set(rs) == {a["trial_id"] for a in expected}
        assert not records.keys() & rs.keys()
        assert sum(len(read(p)["records"]) for p in (path / "cohorts").glob("*/*.json")) == len(rs)
        records.update(rs)
    assert set(records) == {a["trial_id"] for a in manifest["diagnostics"]}
    compact = []
    baselines = {}
    for a in manifest["diagnostics"]:
        r = records[a["trial_id"]]
        p = manifest["prefixes"][a["prefix_id"]]
        sa = p["assignment"]
        assert r["assignment"] == a and r["source_assignment"] == sa
        measured = r["measurement"]
        assert measured["unpadded_input_ids"] == p["token_prefix"]
        assert measured["token_prefix_sha256"] == p["token_prefix_sha256"] == digest(p["token_prefix"])
        assert measured["unpadded_token_positions"] == p["prefix_positions"]
        assert measured["hooks_remaining"] == 0 and measured["exact_token_replay"]
        completion = measured["completion"]
        assert completion["sampling"] == dict(seed=p["seed"], temperature=0.7, top_p=1.0, max_new_tokens=8)
        assert completion["intervention_schedule"] == "prefill_only; no active hooks during decoding"
        scored = task_module(sa).record_from_inference(sa, completion)
        for field in ("answer", "parsed_answer", "expected", "valid", "correct", "task"):
            assert digest(r[field]) == digest(scored[field])
        value, metric = margin(sa["kind"], sa, measured["label_logits"], r["expected"])
        change = measured["intervention"]
        if a["arm"] == "baseline":
            assert change is None
        else:
            assert change["layer"] == a["layer"] and change["position"] == a["position"]
        c = dict(
            trial_id=a["trial_id"],
            prefix_id=a["prefix_id"],
            family=sa["family"],
            task_revision=sa["task_revision"],
            seed=sa["seed"],
            index=sa["index"],
            template=sa.get("template", "table"),
            cell=cell_identity(sa),
            correct_count=sa.get("correct_count"),
            information_bits=r["task"]["information_bits"],
            kind=sa["kind"],
            layer=a["layer"],
            position=a.get("capture_position"),
            direction=a["direction_name"],
            dose=a["dose"],
            dose_multiplier=a["dose_multiplier"],
            condition=a.get("condition", a["direction_name"]),
            patch_mode=a.get("patch_mode", "add"),
            donor_role=a.get("donor_role"),
            comparison_set=a.get("comparison_set"),
            donor=a.get("donor"),
            donor_source_trial_id=a.get("donor_source_trial_id"),
            donor_vector_sha256=a.get("donor_vector_sha256"),
            expected_gain_shift=a.get("expected_gain_shift"),
            original_projection_order_matches_semantics=a.get("original_projection_order_matches_semantics"),
            matched_interchange_condition=a.get("matched_interchange_condition"),
            projection_target=a.get("projection_target"),
            arm=a["arm"],
            margin=value,
            margin_type=metric,
            valid=r["valid"],
            correct=r["correct"],
            answer=r["parsed_answer"],
            completion_sha256=digest(completion["generated_token_ids"]),
            label_mass=measured["label_mass"],
            label_probabilities=measured["label_probabilities_temperature_1"],
            label_logits=measured["label_logits"],
            intervention=change,
            cap_hit=not completion["ended_with_eos"] and len(completion["generated_token_ids"]) == 8,
        )
        compact.append(c)
        if a["arm"] == "baseline":
            assert a["prefix_id"] not in baselines
            baselines[a["prefix_id"]] = c
    assert set(baselines) == set(manifest["prefixes"])
    del records
    groups = defaultdict(lambda: defaultdict(list))
    sham_differences = []
    sham_token_matches = []
    sham_logit_matches = []
    sham_deltas = []
    sham_sources = set()
    engagement = []
    for row in compact:
        base = baselines[row["prefix_id"]]
        if row["arm"] == "baseline":
            continue
        if row["arm"] == "sham":
            sham_sources.add(row["prefix_id"])
            sham_deltas.append(row["intervention"]["delta_norm"])
            sham_logit_matches.append(row["label_logits"] == base["label_logits"])
            sham_token_matches.append(row["completion_sha256"] == base["completion_sha256"])
            sham_differences.extend(
                abs(row["label_probabilities"][s] - base["label_probabilities"][s])
                for s in base["label_probabilities"]
            )
        elif row["dose"] or row["patch_mode"] != "add":
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
        strata = (None, 1, 2, 4) if key[6] == "count-competence-v1" else (None,)
        for correct_count in strata:
            metrics.append(
                dict(
                    family=key[0],
                    kind=key[1],
                    layer=key[2],
                    position=key[3],
                    direction=key[4],
                    dose_multiplier=key[5],
                    task_revision=key[6],
                    **summarize_worlds(worlds, key[6], correct_count),
                )
            )
    integrity = integrity_gate(
        manifest,
        baselines,
        sham_sources,
        sham_deltas,
        sham_differences,
        sham_logit_matches,
        sham_token_matches,
        engagement,
    )
    norm_matches = audit_norm_matching(compact)
    if manifest.get("design") == "projection_interchange_v1":
        integrity["checks"]["runtime_requested_norms_matched"] = bool(norm_matches) and all(
            match["passed"] for match in norm_matches
        )
        integrity["passed"] = all(integrity["checks"].values())
    result = dict(
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        phase=manifest["phase"],
        inference_scope=manifest.get("inference_scope"),
        split_provenance=manifest.get("split_provenance"),
        source_admission=manifest.get("source_admission"),
        source_admission_sha256=manifest.get("source_admission_sha256"),
        assigned_measurements=len(compact),
        compact_outcomes_sha256=digest(compact),
        assigned_sources=manifest["assigned_sources"],
        eligible_sources=len(baselines),
        excluded_sources=manifest["excluded_sources"],
        complete=True,
        integrity_passed=integrity["passed"],
        integrity_checks=integrity["checks"],
        eligible_for_interpretation=integrity["passed"],
        maximum_sham_probability_difference=max(sham_differences, default=0),
        sham_probability_identity=all(d == 0 for d in sham_differences),
        sham_token_identity=all(sham_token_matches),
        sham_logit_identity=all(sham_logit_matches),
        nonzero_interventions=len(engagement),
        all_nonzero_signed_engagement=all(e["signed_engagement"] for e in engagement),
        maximum_relative_perturbation=max((e["relative_delta_norm"] for e in engagement), default=0),
        matched_control_pairs=len(norm_matches),
        maximum_matched_control_achieved_norm_difference=max(
            (abs(m["achieved_norm_difference"]) for m in norm_matches), default=0
        ),
        metrics=metrics,
        world_sampling_rule="Complete cells within (seed,index,template); average template-specific paired means within (seed,index), then resample those underlying worlds. Count strata share the same six-cell-complete cases.",
        interpretation="Interpret effects or schedule scientific follow-up only when integrity_passed is true; complete records alone are insufficient. Within-prefix paired effects resample whole worlds. Requested perturbation norms are matched; achieved norms are recorded. Accuracy bootstrap intervals can be degenerate with no errors; the separate exact world-event bound quantifies zero observed new failures.",
    )
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "analysis.json", result)
    write_once(out / "compact_outcomes.json", compact)
    write_once(out / "target_engagement.json", engagement)
    write_once(out / "norm_matching.json", norm_matches)
    print({k: v for k, v in result.items() if k not in ("metrics", "interpretation")})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.run, a.out)
    if not result["integrity_passed"]:
        raise SystemExit(
            "Replay integrity gate failed; artifacts retained, scientific interpretation blocked"
        )
