"""Synthetic complete artifacts exercise auditing without reading study outcomes."""

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from curiosity_study import count_competence, information_state
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.natural_information import CONDITIONING, engine_hashes, generation_regime, make_prefix
from curiosity_study.runner import source_hash
from curiosity_study.thinking_information import inference_request, record_from_inference, task_module


class FixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    chat_template = "synthetic-natural-audit-template"

    def __len__(self):
        return 0x110001

    def __call__(self, text, **kwargs):
        return dict(
            input_ids=[ord(c) + 1 for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))]
        )

    def decode(self, ids, **kwargs):
        return "".join(chr(token - 1) for token in ids if token)

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: <think>\n"


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load("audit_natural_interventions"), load("prepare_natural_interventions")


def fixture_collection(
    tmp_path,
    modules,
    *,
    sources=None,
    phase="engineering",
    bad_sham=False,
    wrong_prompt=False,
    invalid_direction=False,
    no_engagement=False,
    previous=None,
    baseline_correct=True,
    reasoning_tokens=132,
    batch_size=4,
    include_packing=True,
    max_new_tokens=2048,
    source_max_new_tokens=None,
    regime_change=None,
):
    audit, prepare = modules
    tokenizer = FixtureTokenizer()
    if sources is None:
        sources = [
            a
            for a in information_state.assignments(items=1, seed=921, kinds=("new", "repeat"))
            if a["family"] == "objects"
        ]
    source_max_new_tokens = max_new_tokens if source_max_new_tokens is None else source_max_new_tokens
    sources = copy.deepcopy(sources)
    if source_max_new_tokens != 2048:
        for assignment in sources:
            assignment["max_new_tokens"] = source_max_new_tokens
    model = dict(model_id="Qwen/Qwen3-32B", revision="synthetic-fixture-only")
    original = tmp_path / "original"
    source_manifest = dict(
        assignments=[],
        diagnostics=sources,
        assignment_sha256=digest(sources),
        model=model,
        max_new_tokens=source_max_new_tokens,
        source_sha256=source_hash(),
        task_module_sha256=hashlib.sha256(Path(task_module(sources[0]).__file__).read_bytes()).hexdigest(),
    )
    store = RunStore(original, source_manifest)
    source_records, prefixes = [], {}
    for sa in sources:
        request = inference_request(sa)
        text = tokenizer.apply_chat_template(request["messages"])
        ids = tokenizer(text)["input_ids"]
        expected = task_module(sa).prompt(sa)[2]
        completion = dict(
            raw=f"Original reasoning. </think> {expected}",
            enable_thinking=True,
            messages=request["messages"],
            rendered_prompt=text,
            unpadded_input_ids=ids,
            generated_token_ids=tokenizer(f"Original reasoning. </think> {expected}")["input_ids"] + [0],
            sampling=dict(
                seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=source_max_new_tokens
            ),
        )
        scored = record_from_inference(sa, completion)
        scored["activation_artifact"] = dict(
            positions=["record", "prefill"],
            token_positions=[text.index("End of record.") + len("End of record.") - 1, len(ids) - 1],
        )
        source_records.append(scored)
        prefixes[sa["trial_id"]] = make_prefix(scored, original.name, digest(source_manifest))
    store.save_many(source_records, "synthetic-source", "0000000")
    source_audit = dict(
        manifest_sha256=digest(source_manifest),
        source_sha256=source_manifest["source_sha256"],
        assigned=len(sources),
        complete=True,
        gates=dict(passed=True),
        max_new_tokens=source_max_new_tokens,
    )
    write_once(tmp_path / "source-audit.json", source_audit)
    direction_dir = tmp_path / "directions"
    direction_dir.mkdir()
    fitted = dict(
        sources=np.array(["objects:balanced"]),
        layers=np.array([1]),
        positions=np.array(["record", "prefill"]),
        contrast_norm=np.ones((1, 1, 2)),
        unit=np.ones((1, 1, 2, 4)) / 2,
    )
    with (direction_dir / "directions.npz").open("xb") as f:
        np.savez_compressed(f, **fitted)
    direction_sha = hashlib.sha256((direction_dir / "directions.npz").read_bytes()).hexdigest()
    analysis = dict(
        manifest_sha256="synthetic-extraction",
        development_only=True,
        directions_sha256=direction_sha,
        model=model,
    )
    write_once(direction_dir / "analysis.json", analysis)
    config = dict(
        name="natural-synthetic",
        phase=phase,
        inference_scope="development",
        source_collection=str(original),
        source_audit=str(tmp_path / "source-audit.json"),
        directions_local=str(direction_dir / "directions.npz"),
        directions_remote="directions/directions.npz",
        directions_sha256=direction_sha,
        dose_reference_source="objects:balanced",
        world_index_range=[0, 10000],
        kinds=sorted({a["kind"] for a in sources}),
        order_seed=123,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        layers=[1],
        positions=["prefill"],
        workers=1,
        directions=[dict(name="information", source="objects:balanced", array="unit")],
        dose_multipliers=[1],
        random_seeds=[],
        random_dose_multipliers=[],
    )
    if batch_size == 32:
        config["prefix_group_size"] = 16
    if regime_change is not None:
        config["source_generation_regime_change"] = regime_change
    if phase != "engineering":
        config["baseline_validation_criteria"] = dict(
            minimum_distinct_items_per_cell=1, minimum_accuracy=0.9, minimum_valid_format=0.95
        )
    if previous is not None:
        config.update(
            natural_engineering_audit=str(previous), natural_engineering_audit_sha256=digest(read(previous))
        )
    rows = prepare.build_schedule(config, prefixes, fitted)
    provenance = prepare.split_provenance(config, source_manifest, analysis, sources)
    manifest = dict(
        assay="natural_information",
        phase=phase,
        inference_scope="development",
        model=model,
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        assigned_sources=len(prefixes),
        excluded_sources=[],
        execution=dict(backend="torch_batched_v1", batch_size=batch_size),
        capture_layers=[1],
        max_new_tokens=max_new_tokens,
        generation_regime=generation_regime(config, prefixes),
        greedy=False,
        conditioning=CONDITIONING,
        source_sha256=source_hash(),
        engine_sha256=engine_hashes(),
        source_parent_manifest_sha256=digest(source_manifest),
        source_audit_sha256=digest(source_audit),
        intervention_schedule=config,
        intervention_schedule_sha256=digest(config),
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        direction_input=dict(path=config["directions_remote"], sha256=direction_sha),
        direction_analysis_sha256=digest(analysis),
        direction_extraction_manifest_sha256=analysis["manifest_sha256"],
    )
    if include_packing:
        policy = prepare.packing_policy(config)
        manifest.update(packing_policy=policy, packing_policy_sha256=digest(policy))
    root, child = tmp_path / "parent", tmp_path / "child"
    write_once(root / "manifest.json", manifest)
    write_once(
        root / "freeze.json",
        dict(
            manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"],
            execution_root=str(Path(__file__).resolve().parents[1]),
        ),
    )
    cm = {**copy.deepcopy(manifest), "parent_manifest_sha256": digest(manifest)}
    store = RunStore(child, cm)
    write_once(
        child / "natural_integrity/synthetic.json", dict(passed=True, checks=dict(synthetic_check=True))
    )
    write_once(
        child / "environments/synthetic.json",
        dict(**model, dtype="bfloat16", attention="eager", template_sha256=digest(tokenizer.chat_template)),
    )
    for offset in range(0, len(rows), batch_size):
        cohort = rows[offset : offset + batch_size]
        width = max(len(prefixes[a["prefix_id"]]["token_prefix"]) for a in cohort) + int(wrong_prompt)
        relative = Path("activations/synthetic") / f"{offset:07d}.npz"
        (child / relative).parent.mkdir(parents=True, exist_ok=True)
        with (child / relative).open("xb") as f:
            np.savez_compressed(
                f,
                hidden=np.ones((len(cohort), 1, 2, 4), dtype=np.float32),
                trial_ids=np.array([a["trial_id"] for a in cohort]),
                layers=np.array([1]),
                positions=np.array(["record", "final"]),
            )
        sha = hashlib.sha256((child / relative).read_bytes()).hexdigest()
        records = []
        for ri, a in enumerate(cohort):
            prefix = prefixes[a["prefix_id"]]
            sa = prefix["assignment"]
            expected = task_module(sa).prompt(sa)[2]
            answer = expected
            if not baseline_correct and sa["kind"] in ("new", "repeat"):
                answer = "No" if expected == "Yes" else "Yes"
            raw = "x" * reasoning_tokens + f" </think> {answer}."
            if bad_sham and a["arm"] == "sham":
                raw = "y" + raw[1:]
            if invalid_direction and a["arm"] == "direction":
                raw = "x" * max_new_tokens
            generated = tokenizer(raw)["input_ids"] + (
                [] if invalid_direction and a["arm"] == "direction" else [0]
            )
            completion = dict(
                raw=raw,
                generated_token_ids=generated,
                ended_with_eos=generated[-1] == 0,
                sampling=dict(seed=prefix["seed"], temperature=0.7, top_p=1.0, max_new_tokens=max_new_tokens),
                intervention_schedule="prefill_only; no active hooks during decoding",
            )
            scored = record_from_inference(sa, completion)
            dose = int(a["arm"] == "direction")
            ids = prefix["token_prefix"] + ([77] if wrong_prompt else [])
            pad = width - len(ids)
            change = (
                None
                if a["arm"] == "baseline"
                else dict(
                    layer=1,
                    position="final",
                    token_index=pad + prefix["prefix_positions"]["final"],
                    mode="add",
                    before_norm=10.0,
                    after_norm=10.0 + dose,
                    delta_norm=dose,
                    relative_delta_norm=dose / 10,
                    direction_norm=1.0,
                    projection_before=0.0,
                    projection_after=0.0 if no_engagement else dose,
                    requested_scalar_delta=float(dose),
                    requested_delta_norm=float(dose),
                    rounding_error_norm=0.0,
                )
            )
            measured = dict(
                unpadded_input_ids=ids,
                token_prefix_sha256=digest(ids),
                unpadded_token_positions=prefix["prefix_positions"],
                input_ids=[0] * pad + ids,
                attention_mask=[0] * pad + [1] * len(ids),
                position_ids=[0] * pad + list(range(len(ids))),
                exact_token_replay=True,
                hooks_remaining=0,
                completion=completion,
                intervention=change,
                label_logits={
                    label: (1000 if i else -1000) * (1 - 2 * dose) for i, label in enumerate(prefix["labels"])
                },
                label_probabilities_temperature_1={label: 0.1 for label in prefix["labels"]},
            )
            records.append(
                dict(
                    trial_id=a["trial_id"],
                    terminal=True,
                    assignment=a,
                    kind=a["kind"],
                    source_assignment=sa,
                    source_trial_id=prefix["source_trial_id"],
                    token_prefix_sha256=prefix["token_prefix_sha256"],
                    conditioning=CONDITIONING,
                    measurement=measured,
                    closing_delimiter_count=raw.count("</think>"),
                    cap_hit=not completion["ended_with_eos"],
                    generation_max_new_tokens=max_new_tokens,
                    **{
                        key: scored[key]
                        for key in (
                            "answer",
                            "parsed_answer",
                            "expected",
                            "valid",
                            "correct",
                            "task",
                            "thinking_closed",
                        )
                    },
                    activation_artifact=dict(
                        path=str(relative),
                        sha256=sha,
                        row=ri,
                        layers=[1],
                        positions=["record", "final"],
                        max_new_tokens=max_new_tokens,
                    ),
                )
            )
        store.save_many(records, "synthetic", f"{offset:07d}")
    write_once(
        root / "children.json",
        [
            dict(
                path=str(child),
                manifest_sha256=digest(cm),
                parent_cohorts=list(range(len(rows) // batch_size)),
            )
        ],
    )
    return root, tokenizer


def run_audit(root, tokenizer, modules, out):
    return modules[0].analyze(root, out, tokenizer=tokenizer, eos_token_ids={0})


def test_synthetic_full_audit_uses_final_answers_and_cannot_self_certify(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path, modules)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["complete"] and result["integrity_passed"] and result["engineering_passed"]
    assert not result["eligible_for_interpretation"]
    assert result["maximum_baseline_generated_tokens"] >= 128
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    assert all("not_logit" in r["margin_type"] for r in compact)
    assert all("label_logits" not in r and "label_probabilities" not in r for r in compact)
    assert all(r["margin"] == (1 if r["information_bits"] else -1) for r in compact)
    assert all(r["valid_counterfactual_response"] == 0 for r in compact)
    assert all(r["valid_informative_response"] == r["information_bits"] for r in compact)
    assert all(r["valid_uninformative_response"] == 1 - r["information_bits"] for r in compact)
    assert {r["kind"] for r in compact} == {"new", "repeat"}
    for metric in result["metrics"]:
        assert metric["complete_worlds"] == 1
        assert metric["paired_semantic_response_change"]["mean"] == 0
    assert {
        None if m["stratum"] is None else m["stratum"]["information_bits"] for m in result["metrics"]
    } == {None, 0, 1}


def test_same_final_answer_with_different_sham_reasoning_tokens_fails(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path, modules, bad_sham=True)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["complete"] and not result["integrity_passed"] and not result["engineering_passed"]
    assert not result["integrity_checks"]["sham_token_identity"]
    assert result["integrity_checks"]["sham_logit_identity"]
    assert all(m["paired_accuracy_change"]["mean"] == 0 for m in result["metrics"])


def test_identically_wrong_input_tokens_in_baseline_and_sham_fail_source_binding(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path, modules, wrong_prompt=True)
    with pytest.raises(ValueError, match="token prefix differs"):
        run_audit(root, tokenizer, modules, tmp_path / "analysis")


def test_invalid_capped_reasoning_stays_in_every_world_and_denominator(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path, modules, invalid_direction=True)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["integrity_passed"] and result["engineering_passed"]
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    changed = [r for r in compact if r["arm"] == "direction"]
    assert all(r["margin"] == 0 and not r["valid"] and r["cap_hit"] for r in changed)
    assert all(r["valid_counterfactual_response"] == 0 for r in changed)
    for row in result["metrics"]:
        if row["direction"] == "information":
            assert row["complete_worlds"] == 1 and row["valid"] == 0
            assert row["cap_hits"] == row["branches"]
            assert row["paired_accuracy_change"]["mean"] == -1
            binary = row["binary_response_outcomes"]["valid_counterfactual_response"]
            assert binary["positive_responses"] == binary["baseline_positive_responses"] == 0
            assert binary["assigned_trials"] == binary["analyzed_trials"] == row["branches"]
            assert binary["paired_probability_change"]["mean"] == 0


def test_engineering_does_not_select_for_correct_baseline_answers(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path, modules, baseline_correct=False)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["engineering_passed"] and not result["eligible_for_interpretation"]
    assert all(c["correct"] == 0 for c in result["baseline_readiness"]["cells"])


def test_scientific_readiness_requires_a_different_bound_32b_engineering_run(tmp_path, modules):
    engineering = tmp_path / "engineering"
    root, tokenizer = fixture_collection(engineering, modules)
    run_audit(root, tokenizer, modules, engineering / "analysis")
    scientific = tmp_path / "scientific"
    root, tokenizer = fixture_collection(
        scientific, modules, phase="development", previous=engineering / "analysis/analysis.json"
    )
    result = run_audit(root, tokenizer, modules, scientific / "analysis")
    assert result["integrity_passed"] and result["previous_engineering"]["passed"]
    assert result["baseline_readiness"]["passed"] and result["eligible_for_interpretation"]
    assert not result["engineering_passed"]
    missing = tmp_path / "missing"
    root, tokenizer = fixture_collection(missing, modules, phase="development")
    result = run_audit(root, tokenizer, modules, missing / "analysis")
    assert result["integrity_passed"] and not result["eligible_for_interpretation"]
    prior = read(engineering / "analysis/analysis.json")
    prior["model"]["model_id"] = "Qwen/Qwen3-1.7B"
    write_once(tmp_path / "wrong-model-audit.json", prior)
    wrong = tmp_path / "wrong-model"
    root, tokenizer = fixture_collection(
        wrong, modules, phase="development", previous=tmp_path / "wrong-model-audit.json"
    )
    result = run_audit(root, tokenizer, modules, wrong / "analysis")
    assert not result["previous_engineering"]["passed"] and not result["eligible_for_interpretation"]


def test_count_control_keeps_six_cells_and_all_count_and_gain_strata(tmp_path, modules):
    sources = [a for a in count_competence.assignments(items=1) if a["family"] == "objects"]
    root, tokenizer = fixture_collection(tmp_path, modules, sources=sources)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["integrity_passed"]
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    assert all(r["valid_counterfactual_response"] is None for r in compact)
    rows = [r for r in result["metrics"] if r["direction"] == "information"]
    assert len(rows) == 6
    for row in rows:
        assert row["complete_worlds"] == 1 and row["required_cells_per_world_template"] == 6
        assert row["branches"] == (
            6 if row["stratum"] is None else 2 if "correct_count" in row["stratum"] else 3
        )
        assert row["mean_semantic_response_score"] == 1


def test_report_stop_semantic_score_respects_every_menu_mapping(modules):
    audit = modules[0]
    for index in range(6):
        a = information_state.assignments(items=1, start_index=index, kinds=("instrument",))[0]
        t = information_state.task(a)
        expected = information_state.prompt(a)[2]
        scored = record_from_inference(a, dict(raw=f"Reasoning. </think> {expected}"))
        score, metric = audit.semantic_score(a, scored)
        assert score == (1 if t["information_bits"] else -1)
        assert "categorical" in metric


def test_target_geometry_must_match_the_requested_intervention(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path, modules, no_engagement=True)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["complete"] and not result["integrity_passed"]
    assert not result["integrity_checks"]["intervention_implementation_verified"]


def test_engineering_requires_real_long_continuation_and_four_prefixes(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path / "short", modules, reasoning_tokens=8)
    result = run_audit(root, tokenizer, modules, tmp_path / "short-analysis")
    assert result["integrity_passed"] and not result["engineering_passed"]
    assert not result["engineering_checks"]["baseline_continuation_at_least_128_tokens"]
    sources = [a for a in information_state.assignments(items=1, kinds=("new",)) if a["family"] == "objects"][
        :3
    ]
    root, tokenizer = fixture_collection(tmp_path / "small", modules, sources=sources)
    result = run_audit(root, tokenizer, modules, tmp_path / "small-analysis")
    assert result["integrity_passed"] and not result["engineering_passed"]
    assert not result["engineering_checks"]["four_distinct_source_prefixes"]


def test_categorical_compact_is_compatible_with_direct_paired_contrasts(tmp_path, modules):
    from curiosity_study.causal_contrasts import analyze_contrasts

    root, tokenizer = fixture_collection(tmp_path, modules)
    run_audit(root, tokenizer, modules, tmp_path / "analysis")
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    manifest = read(root / "manifest.json")
    assignments = [prefix["assignment"] for prefix in manifest["prefixes"].values()]
    plan = dict(
        alpha=0.05,
        bootstrap=2000,
        seed=931,
        contrasts=[
            dict(
                name="information_vs_baseline",
                left=dict(direction="information", layer=1, position="prefill", dose_multiplier=1),
                right=dict(arm="baseline"),
                orientation="information_increase",
            )
        ],
    )
    result = analyze_contrasts(compact, assignments, plan)
    assert result["planned_endpoints"] == 2
    assert all(r["estimate"]["mean"] == 0 for r in result["results"])
    assert all(r["totals"]["analyzed_worlds"] == 1 for r in result["results"])


def test_packed_full_audit_keeps_source_units_and_binds_execution(tmp_path, modules):
    engineering = tmp_path / "engineering32"
    root, tokenizer = fixture_collection(engineering, modules, batch_size=32)
    result = run_audit(root, tokenizer, modules, engineering / "analysis")
    manifest = read(root / "manifest.json")
    assert result["engineering_passed"] and result["integrity_passed"]
    assert result["execution"] == manifest["execution"] == dict(backend="torch_batched_v1", batch_size=32)
    assert result["packing_policy"] == manifest["packing_policy"]
    assert result["packing_policy_sha256"] == digest(result["packing_policy"])
    assert result["assigned_sources"] == result["eligible_sources"] == len(manifest["prefixes"]) == 8
    assert result["assigned_measurements"] == 8 * 16
    assert all(m["complete_worlds"] == 1 for m in result["metrics"])
    assert all(m["direction"] != "padding_control" for m in result["metrics"])
    assert all(c["assigned"] == c["world_indices"] == 1 for c in result["baseline_readiness"]["cells"])
    scientific = tmp_path / "scientific32"
    root, tokenizer = fixture_collection(
        scientific,
        modules,
        phase="development",
        batch_size=32,
        previous=engineering / "analysis/analysis.json",
    )
    result = run_audit(root, tokenizer, modules, scientific / "analysis")
    assert result["previous_engineering"]["passed"] and result["eligible_for_interpretation"]


def test_passing_batch16_receipt_cannot_certify_batch32(tmp_path, modules):
    engineering = tmp_path / "engineering16"
    root, tokenizer = fixture_collection(engineering, modules, batch_size=16)
    proof = run_audit(root, tokenizer, modules, engineering / "analysis")
    assert proof["engineering_passed"]
    scientific = tmp_path / "scientific32"
    root, tokenizer = fixture_collection(
        scientific,
        modules,
        phase="development",
        batch_size=32,
        previous=engineering / "analysis/analysis.json",
    )
    result = run_audit(root, tokenizer, modules, scientific / "analysis")
    assert result["integrity_passed"] and result["baseline_readiness"]["passed"]
    assert not result["previous_engineering"]["passed"] and not result["eligible_for_interpretation"]


@pytest.mark.parametrize("missing", ["execution", "packing_policy", "packing_policy_sha256"])
def test_prior_receipt_requires_execution_and_packing_fields(tmp_path, modules, missing):
    engineering = tmp_path / "engineering"
    root, tokenizer = fixture_collection(engineering, modules, batch_size=32)
    prior = run_audit(root, tokenizer, modules, engineering / "analysis")
    prior.pop(missing)
    path = tmp_path / "incomplete-proof.json"
    write_once(path, prior)
    manifest = read(root / "manifest.json")
    manifest["phase"] = "development"
    manifest["intervention_schedule"].update(
        natural_engineering_audit=str(path), natural_engineering_audit_sha256=digest(prior)
    )
    assert not modules[0].previous_engineering(manifest)["passed"]


def test_packing_policy_is_checked_not_just_its_hash(tmp_path, modules):
    root, tokenizer = fixture_collection(tmp_path / "engineering", modules, batch_size=32)
    prior = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    manifest = read(root / "manifest.json")
    wrong = copy.deepcopy(manifest)
    wrong["packing_policy"]["ordering"] = "unbound-other-order"
    wrong["packing_policy_sha256"] = digest(wrong["packing_policy"])
    with pytest.raises(ValueError, match="packing policy differs"):
        modules[0].packing_binding(wrong)
    prior["packing_policy"]["ordering"] = "unbound-other-order"
    prior["packing_policy_sha256"] = digest(prior["packing_policy"])
    path = tmp_path / "wrong-layout-proof.json"
    write_once(path, prior)
    manifest["phase"] = "development"
    manifest["intervention_schedule"].update(
        natural_engineering_audit=str(path), natural_engineering_audit_sha256=digest(prior)
    )
    assert not modules[0].previous_engineering(manifest)["passed"]
    wrong = copy.deepcopy(manifest)
    wrong["execution"]["batch_size"] = 16
    with pytest.raises(ValueError, match="execution batch differs"):
        modules[0].packing_binding(wrong)
    wrong = copy.deepcopy(manifest)
    wrong.pop("packing_policy")
    wrong.pop("packing_policy_sha256")
    with pytest.raises(ValueError, match="requires explicit packing metadata"):
        modules[0].packing_binding(wrong)


@pytest.mark.parametrize("size", [4, 8, 16])
def test_legacy_manifest_can_be_audited_with_explicit_derived_layout(tmp_path, modules, size):
    root, tokenizer = fixture_collection(tmp_path, modules, batch_size=size, include_packing=False)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["engineering_passed"]
    assert result["execution"]["batch_size"] == size
    assert result["packing_policy"]["mode"] == "single_prefix_cohorts_v1"
    assert result["packing_policy"]["prefix_group_size"] == size
    assert result["packing_policy_sha256"] == digest(result["packing_policy"])


def test_4096_audit_accepts_late_closure_and_keeps_full_cap_invalids(tmp_path, modules):
    root, tokenizer = fixture_collection(
        tmp_path, modules, max_new_tokens=4096, reasoning_tokens=2050, invalid_direction=True
    )
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["integrity_passed"] and result["engineering_passed"]
    assert result["max_new_tokens"] == 4096
    assert result["maximum_baseline_generated_tokens"] > 2048
    assert result["generation_regime"]["source_max_new_tokens"] == [4096]
    assert result["generation_regime"]["source_matches_natural"]
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    assert all(r["valid"] and not r["cap_hit"] for r in compact if r["arm"] == "baseline")
    assert all(
        not r["valid"] and r["cap_hit"] and r["generated_tokens"] == 4096
        for r in compact
        if r["arm"] == "direction"
    )


def test_2048_engineering_receipt_cannot_certify_4096_even_with_same_batch(tmp_path, modules):
    old = tmp_path / "old"
    root, tokenizer = fixture_collection(old, modules, max_new_tokens=2048)
    proof = run_audit(root, tokenizer, modules, old / "analysis")
    assert proof["engineering_passed"]
    new = tmp_path / "new"
    root, tokenizer = fixture_collection(
        new, modules, phase="development", max_new_tokens=4096, previous=old / "analysis/analysis.json"
    )
    result = run_audit(root, tokenizer, modules, new / "analysis")
    assert result["integrity_passed"] and result["baseline_readiness"]["passed"]
    assert not result["previous_engineering"]["passed"] and not result["eligible_for_interpretation"]
    engineering = tmp_path / "engineering4096"
    root, tokenizer = fixture_collection(engineering, modules, max_new_tokens=4096)
    run_audit(root, tokenizer, modules, engineering / "analysis")
    matched = tmp_path / "matched"
    root, tokenizer = fixture_collection(
        matched,
        modules,
        phase="development",
        max_new_tokens=4096,
        previous=engineering / "analysis/analysis.json",
    )
    result = run_audit(root, tokenizer, modules, matched / "analysis")
    assert result["previous_engineering"]["passed"] and result["eligible_for_interpretation"]


def test_changed_source_regime_is_explicit_development_only(tmp_path, modules):
    declaration = dict(
        source_max_new_tokens=[2048],
        natural_max_new_tokens=4096,
        reason="Prospective engineering comparison; all source prompts retained",
    )
    root, tokenizer = fixture_collection(
        tmp_path, modules, max_new_tokens=4096, source_max_new_tokens=2048, regime_change=declaration
    )
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["engineering_passed"]
    assert result["generation_regime"]["development_change"] == declaration
    assert not result["generation_regime"]["source_matches_natural"]
    manifest = read(root / "manifest.json")
    config = manifest["intervention_schedule"]
    prefixes = manifest["prefixes"]
    for update in (dict(inference_scope="fresh_confirmation"), dict(phase="confirmation")):
        with pytest.raises(ValueError, match="confirmation requires matching"):
            generation_regime({**config, **update}, prefixes)
    missing = {k: v for k, v in config.items() if k != "source_generation_regime_change"}
    with pytest.raises(ValueError, match="explicit development declaration"):
        generation_regime(missing, prefixes)
