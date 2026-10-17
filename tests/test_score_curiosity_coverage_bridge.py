"""Independent parser/oracle adapter against synthetic displayed Cartesian worlds."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from curiosity_study import voluntary_capture, voluntary_resolution as task

PATH = Path(__file__).parents[1] / "scripts/score_curiosity_coverage_bridge.py"
spec = importlib.util.spec_from_file_location("coverage_scoring", PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
ANALYZER = PATH.with_name("analyze_curiosity_coverage_bridge.py")
spec = importlib.util.spec_from_file_location("coverage_analysis_for_scoring", ANALYZER)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def wire(value):
    return json.loads(json.dumps(value))


def synthetic_world(a):
    tag = f"{a['seed']}_{a['world_slot']}"
    candidates = [[f"color{tag}a", "square"], [f"color{tag}b", "square"],
                  [f"color{tag}a", "round"], [f"color{tag}b", "round"]]
    return dict(attributes=["color", "shape"], candidates=candidates, truth=a["world_slot"] % 4,
                values=[[f"color{tag}a", f"color{tag}b"], ["square", "round"]])


@pytest.fixture(scope="module")
def synthetic():
    base, preflight, inner = [], [], {}
    branches = [("choice", d, m) for d in s.DELIVERIES for m in (0, 1)] + [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in s.DIAGNOSTICS]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(task.state, "task", synthetic_world)
        for panel in (1, 2):
            for slot in range(12):
                for kind, delivery, menu in branches:
                    index = len(base)
                    a = dict(trial_id=f"synthetic:{index}", task_revision="voluntary-resolution-v2",
                             family="objects", seed=9000 + panel, index=slot, world_slot=slot,
                             known=[(slot + slot // 4) % 2], query=1 - (slot + slot // 4) % 2,
                             kind=kind, delivery=delivery, menu_layout=menu, vocabulary="heldout",
                             sampling_seed=5000 + index, max_new_tokens=4096)
                    a["semantic_world_id"] = s.digest(task.canonical_world("objects", synthetic_world(a)))
                    prompt, _, expected = task.prompt(a)
                    b = dict(base_trial_id=a["trial_id"], source_assignment=a, source_assignment_sha256=s.digest(a),
                             sampling_seed=a["sampling_seed"], panel_id=f"panel_{panel}",
                             **{k: a[k] for k in ("world_slot", "semantic_world_id", "family", "kind", "delivery", "menu_layout")})
                    base.append(b)
                    text = ("<|im_start|>system\nFollow the instructions and requested response format.<|im_end|>\n"
                            "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n")
                    request = dict(token_prefix=[1, 2, index + 3], token_prefix_sha256=s.digest([1, 2, index + 3]),
                                   source_trial_id=a["trial_id"], seed=a["sampling_seed"])
                    preflight.append(dict(base_trial_id=a["trial_id"], source_assignment_sha256=s.digest(a),
                                          rendered_prompt=text, rendered_prompt_sha256=s.digest(text), request=request))
                    service = s.SERVICES[(slot + menu + s.DELIVERIES.index(delivery)) % 4]
                    label = expected if kind != "choice" else next(k for k, v in task.task(a)["action_mapping"].items() if v == service)
                    completion = dict(raw=f"Synthetic thought.</think>{label}.", generated_token_ids=[7, 8], ended_with_eos=True,
                                      sampling=dict(seed=a["sampling_seed"], temperature=0.7, top_p=1.0, max_new_tokens=4096))
                    inner[a["trial_id"]] = wire(task.record_from_thinking_inference(a, completion))
    assignments, observed = [], {}
    for base_row, b in enumerate(base):
        for arm in s.ARMS:
            schedule, direction = ("baseline", None) if arm == "baseline" else arm.split("__")
            outer = dict(**deepcopy(b), trial_id=f"{arm}:{b['base_trial_id']}", arm=arm, schedule=schedule,
                         direction_key=direction, base_row=base_row)
            assignments.append(outer)
            record = deepcopy(inner[b["base_trial_id"]])
            observed[outer["trial_id"]] = dict(assignment=outer, task_record=record, completion=record["inference"],
                                               record_sha256=s.digest([outer, record]), status="completed")
    material = dict(assignments=assignments, assignment_sha256=s.digest(assignments),
                    base_assignments=base, base_assignment_sha256=s.digest(base),
                    source_generation_preflight=dict(requests=preflight, requests_sha256=s.digest(preflight)))
    return material, observed


@pytest.fixture(scope="module")
def scored(synthetic):
    return s.score_collection(*synthetic)


def test_full8352_independent_displayed_oracle_and_held_analyzer_readiness(synthetic, scored):
    material, observed = synthetic
    assert scored["checks"] == dict(computation_agrees=True, complete_response_coverage=True, all_assigned_slots_retained=True)
    assert not scored["mismatches"] and scored["counts"]["scoring_checked"] == 8352
    assert len(scored["outcomes"]) == 8352 and scored["readiness"]["passed"]
    assert len(scored["readiness"]["diagnostic_cells"]) == 348
    assert len(scored["readiness"]["choice_cells"]) == 174
    assert len(scored["readiness"]["arm_diagnostics"]) == 29
    cells, _ = analysis.inventory(material["assignments"], scored["outcomes"])
    expected = analysis.task_readiness(cells)
    for key in ("passed", "diagnostic_cells", "choice_cells", "arm_diagnostics", "original_all_pristine_format_passed"):
        assert scored["readiness"][key] == expected[key]
    assert {r["selected_service"] for r in scored["outcomes"] if r["kind"] == "choice"} == set(s.SERVICES)
    assert all(observed[r["trial_id"]]["task_record"]["correct"] == r["correct"] for r in scored["outcomes"])
    assert all(not value for key, value in scored["scope"].items() if key.endswith("verified") or key == "scientific_admission")


@pytest.mark.parametrize("text,closed,valid,answer", [
    ("thinking A", False, False, ""), ("x</think> A", True, True, "A"),
    ("x</think>A.", True, True, "A."), ("x</think>A..", True, False, "A.."),
    ("x</think>\\boxed{A}", True, False, "\\boxed{A}"), ("x</think>```A```", True, False, "```A```"),
    ("x</think>A because", True, False, "A because"), ("x</think>A</think>B", True, True, "B"),
    ("x</think>Ａ", True, False, "Ａ"), ("x</think>\"A\"", True, False, '\"A\"'),
])
def test_exact_last_closure_single_label_no_prose_or_json_rescue(text, closed, valid, answer):
    completion = dict(raw=text, generated_token_ids=[1, 2], ended_with_eos=True,
                      sampling=dict(seed=5, temperature=0.7, top_p=1.0, max_new_tokens=4096))
    parsed = s.parse_completion(completion, 5)
    assert (parsed["thinking_closed"], parsed["valid"], parsed["answer"]) == (closed, valid, answer)


def test_cap_eos_boundary_and_premature_generation():
    inf = dict(raw="x</think>A", generated_token_ids=[1] * 4096, ended_with_eos=False,
               sampling=dict(seed=5, temperature=0.7, top_p=1.0, max_new_tokens=4096))
    assert s.parse_completion(inf, 5)["cap_hit"]
    inf["ended_with_eos"] = True
    assert not s.parse_completion(inf, 5)["cap_hit"]
    inf.update(generated_token_ids=[1], ended_with_eos=False)
    with pytest.raises(ValueError, match="Unterminated"):
        s.parse_completion(inf, 5)
    inf.update(ended_with_eos=1)
    with pytest.raises(ValueError, match="metadata"):
        s.parse_completion(inf, 5)


def test_missing_error_and_misreported_validity_never_certify_full_scoring(synthetic):
    material, all_observed = synthetic
    keys = list(all_observed)[:3]
    observed = {key: deepcopy(all_observed[key]) for key in keys}
    observed[keys[0]]["task_record"]["valid"] = False
    observed[keys[1]].update(status="execution_error", task_record=None, completion=None)
    observed[keys[2]]["completion"]["ended_with_eos"] = False
    result = s.score_collection(material, observed)
    assert len(result["outcomes"]) == 8352
    assert result["counts"]["scoring_checked"] == 1
    assert result["counts"]["execution_error"] == 2 and result["counts"]["missing"] == 8349
    assert not result["checks"]["computation_agrees"] and not result["checks"]["complete_response_coverage"]
    assert result["outcomes"][0]["valid"] and not result["readiness"]["passed"]
    empty = s.score_collection(material, {})
    assert empty["checks"]["computation_agrees"] and not empty["checks"]["complete_response_coverage"]
    assert empty["counts"]["scoring_checked"] == 0


def test_independent_semantic_error_is_valid_incorrect_and_scoring_agrees(synthetic):
    material, observed = synthetic
    a = next(a for a in material["assignments"] if a["kind"] == "new_fact")
    source = deepcopy(observed[a["trial_id"]])
    label = next(k for k in "ABCD" if k != source["task_record"]["expected"])
    inf = source["completion"]
    inf["raw"] = f"x</think>{label}"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(task.state, "task", synthetic_world)
        source["task_record"] = wire(task.record_from_thinking_inference(a["source_assignment"], inf))
    row, mismatch = s.score_row(a, source, s.build_oracles(material)[a["base_trial_id"]])
    assert not mismatch and row["valid"] and row["correct"] is False


@pytest.mark.parametrize("tamper", ["truth", "observation", "option", "assignment", "drop"])
def test_bound_world_oracle_rejects_rehashed_prompt_or_grid_mutation(synthetic, tamper):
    material = deepcopy(synthetic[0])
    if tamper == "assignment":
        material["assignments"][1]["source_assignment"]["query"] ^= 1
        material["assignment_sha256"] = s.digest(material["assignments"])
    elif tamper == "drop":
        material["assignments"].pop()
        material["assignment_sha256"] = s.digest(material["assignments"])
    else:
        frozen = material["source_generation_preflight"]["requests"]
        index = 0 if tamper != "option" else 6
        text = frozen[index]["rendered_prompt"]
        if tamper == "truth":
            text = text.replace("The shape is square.", "The shape is round.")
        elif tamper == "observation":
            text = text.replace("Additional delivery:\nThe shape is square.", "Additional delivery:\nThe shape is triangle.")
        else:
            text = text.replace("Candidate W.", "Candidate NOT_W.")
        assert text != frozen[index]["rendered_prompt"]
        frozen[index]["rendered_prompt"] = text
        frozen[index]["rendered_prompt_sha256"] = s.digest(text)
        material["source_generation_preflight"]["requests_sha256"] = s.digest(frozen)
    with pytest.raises(ValueError):
        s.build_oracles(material)


def test_readiness_exact_integer_thresholds_and_cap_veto(scored):
    rows = deepcopy(scored["outcomes"])
    arm = "full__curiosity_negative"
    affected = [r for r in rows if r["arm"] == arm and r["kind"] == "new_fact" and r["panel_id"] == "panel_1"]
    affected[0].update(valid=False, correct=False)
    assert s.readiness(rows)["passed"]  #11/12,143/144
    affected[1].update(valid=False, correct=False)
    assert not s.readiness(rows)["passed"]
    affected[1].update(valid=True, correct=True, cap_hit=True)
    assert not s.readiness(rows)["passed"]
    affected[1]["cap_hit"] = False
    # Seven invalid diagnostics spread across cells are permitted by137/144.
    candidates = [r for r in rows if r["arm"] == arm and r["kind"] != "choice" and r["kind"] != "new_fact"]
    used = set()
    for r in candidates:
        key = r["panel_id"], r["kind"]
        if key not in used and len(used) < 6:
            r.update(valid=False, correct=False)
            used.add(key)
    assert s.readiness(rows)["passed"]
    extra = next(r for r in candidates if (r["panel_id"], r["kind"]) not in used)
    extra.update(valid=False, correct=False)
    assert not s.readiness(rows)["passed"]


def sidecars(material, observed):
    result = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(task.state, "task", synthetic_world)
        for a in material["assignments"]:
            if a["kind"] != "choice" or a["trial_id"] not in observed:
                continue
            source = observed[a["trial_id"]]
            inner = source["task_record"]
            side = voluntary_capture.choice_sidecar(inner)
            side.update(source_trial_id=a["trial_id"], inner_source_trial_id=a["base_trial_id"],
                        source_record_sha256=source["record_sha256"], source_task_record_sha256=s.digest(inner),
                        assignment_sha256=s.digest(a), source_assignment_sha256=s.digest(a["source_assignment"]))
            if inner["cap_hit"]:
                side.update(status="capped_choice", receipt_required=False)
            elif side["receipt_required"]:
                side.update(status="completed", receipt_model_pass_completed=True,
                            receipt_input=task.receipt_input(inner), measurement={"synthetic": True})
                side["executed_simulator_outcome"]["receipt_model_pass_completed"] = True
                side["executed_simulator_outcome"]["observation_ledger"][-1]["model_pass_completed"] = True
            result[a["trial_id"]] = wire(side)
    return result


@pytest.fixture(scope="module")
def all_receipts(synthetic):
    return sidecars(*synthetic)


def test_complete4176_receipt_semantics_all_services_and_raw_conversation(synthetic, all_receipts):
    result = s.check_receipts(*synthetic, all_receipts)
    assert all(result["checks"].values()), result["mismatches"][:2]
    assert result["counts"]["assigned"] == result["counts"]["observed"] == 4176
    assert not result["actual_delivery_verified"]
    assert result["counts"]["stop"] == 1044


@pytest.mark.parametrize("tamper", ["payload", "ledger", "unselected", "source", "raw", "stop_fakepass"])
def test_receipt_payload_ledger_and_source_tampering_fails(synthetic, all_receipts, tamper):
    material, observed = synthetic
    key = next(k for k, r in all_receipts.items() if r["selected_service"] == ("stop" if tamper == "stop_fakepass" else "report"))
    selected = deepcopy(all_receipts[key])
    if tamper == "payload":
        selected["receipt_input"]["messages"][-1]["content"] = "Unselected information"
    elif tamper == "ledger":
        selected["executed_simulator_outcome"]["observation_ledger"][-1]["facts"] = []
    elif tamper == "unselected":
        selected["selected_service"] = "pattern"
    elif tamper == "source":
        selected["source_record_sha256"] = "0" * 64
    elif tamper == "raw":
        selected["receipt_input"]["messages"][-2]["content"] = "Filtered prior reasoning"
    else:
        selected.update(receipt_model_pass_completed=True, receipt_input={})
    result = s.check_receipts(material, observed, {key: selected})
    assert not result["checks"]["semantic_agreement"] and not result["checks"]["complete_receipt_coverage"]
    assert not result["checks"]["required_semantic_dispositions_passed"]


def test_failed_receipt_preserves_original_unexecuted_ledger(synthetic, all_receipts):
    material, observed = synthetic
    key = next(k for k, r in all_receipts.items() if r["status"] == "completed")
    side = deepcopy(all_receipts[key])
    side.update(status="execution_error", receipt_model_pass_completed=False, measurement=None,
                executed_simulator_outcome=deepcopy(observed[key]["task_record"]["simulator_outcome"]))
    result = s.check_receipts(material, observed, {key: side})
    assert result["checks"]["semantic_agreement"] and not result["checks"]["required_semantic_dispositions_passed"]
    side["executed_simulator_outcome"]["receipt_model_pass_completed"] = True
    assert not s.check_receipts(material, observed, {key: side})["checks"]["semantic_agreement"]


def test_capped_and_invalid_choices_cannot_deliver(synthetic):
    material, all_observed = synthetic
    rows = [a for a in material["assignments"] if a["kind"] == "choice"][:2]
    observed = {a["trial_id"]: deepcopy(all_observed[a["trial_id"]]) for a in rows}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(task.state, "task", synthetic_world)
        for i, a in enumerate(rows):
            source = observed[a["trial_id"]]
            inf = source["completion"]
            if i == 0:
                inf.update(generated_token_ids=[7] * 4096, ended_with_eos=False)
            else:
                inf["raw"] = "No closed answer"
            source["task_record"] = wire(task.record_from_thinking_inference(a["source_assignment"], inf))
    receipts = sidecars(material, observed)
    result = s.check_receipts(material, observed, receipts)
    assert result["checks"]["semantic_agreement"]
    assert Counter(r["status"] for r in receipts.values()) == {"capped_choice": 1, "invalid_choice": 1}
    assert all(not r["receipt_required"] and not r["receipt_model_pass_completed"] for r in receipts.values())


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_input_json_rejects_duplicate_and_nonfinite(raw):
    with pytest.raises(ValueError):
        s.strict_json(raw)


def test_cli_immutable_missing_receipt_retains_denominator(tmp_path, synthetic):
    inp, out = tmp_path / "input.json", tmp_path / "out.json"
    inp.write_text(json.dumps(dict(materials=synthetic[0], observed={})))
    command = [sys.executable, str(PATH), "--input", str(inp), "--out", str(out)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert len(result["outcomes"]) == 8352 and result["counts"]["missing"] == 8352
    assert not result["readiness"]["passed"] and result["scorer_byte_sha256"]
    assert subprocess.run(command, capture_output=True).returncode != 0


def composed(material, scored):
    return dict(outcomes=deepcopy(scored["outcomes"]), outcome_sha256=s.digest(scored["outcomes"]),
                assignment_sha256=material["assignment_sha256"], materials_sha256=s.digest(material),
                manifest_sha256="a" * 64, parent_manifest_sha256="a" * 64,
                context_audit_byte_sha256="b" * 64, lifecycle_audit_byte_sha256="c" * 64,
                lifecycle_shutdown_verified=True, complete=True, integrity_passed=True,
                checks=dict.fromkeys(analysis.GATES, True), readiness=scored["readiness"])


def test_analysis_adapter_preserves_failed_lifecycle_and_readiness(synthetic, scored):
    material = synthetic[0]
    audit = composed(material, scored)
    audit.update(lifecycle_shutdown_verified=False, integrity_passed=False)
    audit["readiness"] = {**audit["readiness"], "passed": False}
    payload = s.assemble_analysis_input(material, audit, analysis.ANALYSIS_METADATA)
    assert not payload["integrity_passed"] and not payload["readiness"]["passed"]
    assert not payload["lifecycle_shutdown_verified"] and len(payload["assignments"]) == 8352
    assert payload["assignment_sha256"] == s.digest(payload["assignments"])
    cells, _ = analysis.inventory(payload["assignments"], payload["outcomes"])
    assert len(cells) == 8352
    assert set(payload["assignments"][0]) == set(analysis.METADATA)


@pytest.mark.parametrize("field", ["lifecycle_audit_byte_sha256", "lifecycle_shutdown_verified", "outcome_sha256", "materials_sha256", "parent_manifest_sha256"])
def test_analysis_adapter_rejects_core_or_misbound_audit(synthetic, scored, field):
    material = synthetic[0]
    audit = composed(material, scored)
    if field.startswith("lifecycle"):
        audit.pop(field)
    else:
        audit[field] = "0" * 64
    with pytest.raises(ValueError):
        s.assemble_analysis_input(material, audit, analysis.ANALYSIS_METADATA)


def test_actual_frozen_inputs_with_synthetic_labels_only():
    """No real model output files are opened, and no new material is selected."""
    project = PATH.parents[1]
    campaign = project / "runs/qwen32-curiosity-coverage-engineering-v1"
    if not (campaign / "manifest.json").exists():
        pytest.skip("Optional bound original-input parity package is not installed")
    code = r'''
import hashlib, importlib.util, json, pathlib, sys
campaign, scorer = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((campaign / 'manifest.json').read_text())
freeze = json.loads((campaign / 'freeze.json').read_text())
runtime = pathlib.Path(freeze['execution_root']) / 'runtime'
sys.path.insert(0, str(runtime))
from curiosity_study import voluntary_resolution as task, voluntary_capture as receipts
spec = importlib.util.spec_from_file_location('independent_score', scorer)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
assert s.digest(manifest) == freeze['manifest_sha256']
for module in (task, receipts):
    path = pathlib.Path(module.__file__)
    assert path.resolve().is_relative_to(runtime.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['runtime_files'][str(path.relative_to(runtime))]
material = manifest['plan']; oracles = s.build_oracles(material)
observed = {}; sidecars = {}; variants_checked = 0
for outer in material['assignments']:
    a = outer['source_assignment']; o = oracles[outer['base_trial_id']]
    label = o['expected'] if a['kind'] != 'choice' else 'ABCD'[outer['base_row'] % 4]
    inference = dict(raw='Synthetic test only.</think>'+label+'.', generated_token_ids=[5, 6], ended_with_eos=True,
                     sampling=dict(seed=a['sampling_seed'], temperature=.7, top_p=1., max_new_tokens=4096))
    record = json.loads(json.dumps(task.record_from_thinking_inference(a, inference)))
    source_hash = s.digest([outer, record])
    observed[outer['trial_id']] = dict(assignment=outer, task_record=record, completion=inference,
                                     record_sha256=source_hash, status='completed')
    if a['kind'] == 'choice':
        side = receipts.choice_sidecar(record)
        side.update(source_trial_id=outer['trial_id'], inner_source_trial_id=a['trial_id'],
                    source_record_sha256=source_hash, source_task_record_sha256=s.digest(record),
                    assignment_sha256=s.digest(outer), source_assignment_sha256=s.digest(a))
        if side['receipt_required']:
            side.update(status='completed', receipt_model_pass_completed=True, receipt_input=task.receipt_input(record),
                        measurement={'synthetic_test_only':True})
            side['executed_simulator_outcome']['receipt_model_pass_completed'] = True
            side['executed_simulator_outcome']['observation_ledger'][-1]['model_pass_completed'] = True
        sidecars[outer['trial_id']] = json.loads(json.dumps(side))
    if outer['arm'] == 'baseline':
        for raw, cap in [('x</think>A because', False), ('unclosed A', False), ('x</think>A', True), ('x</think>A</think>B',False)]:
            inf = {**inference, 'raw':raw, 'ended_with_eos':not cap, 'generated_token_ids':[5]*4096 if cap else [5,6]}
            rec = json.loads(json.dumps(task.record_from_thinking_inference(a, inf)))
            obs = dict(assignment=outer, task_record=rec, completion=inf, record_sha256=s.digest(rec), status='completed')
            row, mismatches = s.score_row(outer, obs, o)
            assert not mismatches, mismatches[:2]
            variants_checked += 1
scored = s.score_collection(material, observed)
assert all(scored['checks'].values()), scored['mismatches'][:2]
assert scored['readiness']['passed']
checked = s.check_receipts(material, observed, sidecars)
assert all(checked['checks'].values()), checked['mismatches'][:2]
print(json.dumps(dict(prompts=len(oracles), synthetic_generations=len(observed), synthetic_receipts=len(sidecars),
                     grammar_variants=variants_checked, actual_model_outputs_read=False)))
'''
    result = subprocess.run([sys.executable, "-c", code, str(campaign), str(PATH)], check=True,
                            capture_output=True, text=True, cwd=project, timeout=120)
    assert json.loads(result.stdout) == dict(prompts=288, synthetic_generations=8352,
        synthetic_receipts=4176, grammar_variants=1152, actual_model_outputs_read=False)
