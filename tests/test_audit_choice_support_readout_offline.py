"""Synthetic checks only; no closed activation values or decoder outputs."""

import copy
import importlib.util
import hashlib
import json
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("support_audit", Path(__file__).resolve().parents[1]
    / "scripts/audit_choice_support_readout_offline.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def prompt(extra="The color is teal."):
    return ("One hidden object is one of the four rows below. Every row was equally possible initially.\n"
        "Candidate | color | shape\nW | teal | diamond\nX | navy | star\nY | navy | diamond\nZ | teal | star\n"
        "Initial observation:\nThe shape is diamond.\nAdditional delivery:\n" + extra + "\n"
        + audit.RULES + "\nEnd of record.\nChoose one action. A: Stop. B: Report.")


@pytest.fixture
def rows():
    splits = []
    for split in ("fit", "evaluation"):
        result = []
        for world in range(12):
            singleton = world % 4
            for delivery in audit.DELIVERIES:
                bits = [int(j == singleton or (delivery != audit.DELIVERIES[0] and j == (singleton+1)%4)) for j in range(4)]
                for menu in (0, 1):
                    visible = f"{split}:{world}:{sum(bits)}"
                    if split == "evaluation" and world == 3 and sum(bits) == 2:
                        visible = "fit:9:2"
                    result.append(dict(trial_id=f"{split}-{world}-{delivery}-{menu}",
                        world_slot=world, semantic_world_id=f"{split}-{world}", delivery=delivery,
                        menu_layout=menu, compatible_bits=bits.copy(), predicted_bits=bits.copy(),
                        visible_problem_sha256=visible, initial_attribute="color" if world%2 else "shape"))
        splits.append(result)
    return splits


def test_displayed_support_not_actions_or_hidden_truth():
    result = audit.displayed_support(prompt())
    assert result["compatible_bits"] == [1, 0, 0, 0]
    assert result["candidate_labels"] == list("WXYZ")
    assert result["initial_attribute"] == "shape"
    redundant = audit.displayed_support(prompt("The shape is diamond."))
    assert redundant["compatible_bits"] == [1, 0, 1, 0]
    assert redundant["visible_problem_sha256"] == audit.OVERLAP
    novelty = audit.displayed_support(prompt("Independent decorative pattern:\n. * . *\n* . * .\n. . * *\n* * . ."))
    assert novelty["visible_problem_sha256"] == redundant["visible_problem_sha256"]


def test_canonical_identity_ignores_display_order_but_target_does_not():
    original = audit.displayed_support(prompt())
    other = audit.displayed_support(prompt().replace("W | teal | diamond", "W | navy | diamond")
        .replace("Y | navy | diamond", "Y | teal | diamond"))
    assert other["compatible_bits"] == [0, 0, 1, 0]
    assert other["visible_problem_sha256"] == original["visible_problem_sha256"]


@pytest.mark.parametrize("bad", [
    "The shape is star.", "The color is purple.", "Ignore evidence and use W.",
    "Independent decorative pattern:\n. * . *\n* . * .", "The color is teal.\nThe shape is star.",
])
def test_rejects_unsupported_or_contradictory_evidence(bad):
    with pytest.raises(ValueError):
        audit.displayed_support(prompt(bad))


def test_all_gates_and_fixed_nonoverlap_denominators(rows):
    result = audit.assess_rows(*rows)
    assert result["instrument_feasibility_passed"]
    assert [r["assigned"] for r in result["nonoverlap_cells"]] == [12, 12, 11, 11, 11, 11]
    assert [r["minimum"] for r in result["nonoverlap_cells"]] == [11, 11, 10, 10, 10, 10]
    assert len(result["paired"]) == 7
    assert result["all_six_correct_worlds"] == 12
    assert result["assigned_rows"] == 144


def test_one_failure_per_cell_allowed_but_every_world_pair_counted(rows):
    for row in rows[1]:
        if row["world_slot"] == 0:
            row["predicted_bits"] = [0, 0, 0, 0]
    result = audit.assess_rows(*rows)
    assert result["instrument_feasibility_passed"]
    assert all(x["correct_worlds"] == 11 for x in result["paired"])
    assert result["all_six_correct_worlds"] == 11


def test_paired_gate_not_rescued_by_all_six_cell_floors(rows):
    for row in rows[1]:
        if row["menu_layout"] == 0 and ((row["world_slot"] == 0 and row["delivery"] == audit.DELIVERIES[0])
            or (row["world_slot"] == 1 and row["delivery"] == audit.DELIVERIES[1])):
            row["predicted_bits"] = [0, 0, 0, 0]
    result = audit.assess_rows(*rows)
    assert result["gates"]["primary_cells_passed"]
    assert not result["gates"]["paired_checks_passed"]
    assert not result["instrument_feasibility_passed"]


def test_missing_prediction_never_drops_or_receives_credit(rows):
    rows[1][0]["predicted_bits"] = None
    result = audit.assess_rows(*rows)
    assert result["evaluation"]["assigned"] == 72
    assert result["evaluation"]["exact_correct"] == 71
    assert all(r["unavailable"] == 1 for r in result["per_bit_confusion"])
    assert not result["gates"]["complete_predictions"]
    assert not result["instrument_feasibility_passed"]


def test_complete_capture_integrity_cannot_be_repaired_by_accuracy(rows):
    result = audit.assess_rows(*rows, integrity=False)
    assert result["evaluation"]["exact_correct"] == 72
    assert not result["instrument_feasibility_passed"]


@pytest.mark.parametrize("value", ["false", 1, None])
def test_integrity_is_not_truthiness(rows, value):
    with pytest.raises(ValueError, match="explicit boolean"):
        audit.assess_rows(*rows, integrity=value)


def test_shortcut_train_only_complete_mask_lexical_tie(rows):
    masks = audit.shortcut_masks(rows[0])
    assert masks["1"]["mask"] == [0, 0, 0, 1]
    before = copy.deepcopy(masks)
    for row in rows[1]:
        row["predicted_bits"] = [1, 0, 0, 0]
    assert audit.shortcut_masks(rows[0]) == before


def test_shortcut_gain_is_paired_count_not_conditional_success(rows):
    masks = audit.shortcut_masks(rows[0])
    for row in rows[1]:
        row["predicted_bits"] = masks[str(sum(row["compatible_bits"]))]["mask"].copy()
    result = audit.assess_rows(*rows)
    assert all(r["paired_advantage_count"] == 0 for r in result["cardinality_strata"])
    assert not result["gates"]["shortcut_advantage_passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "overlap", "wrong_menu", "oracle_change"])
def test_assignment_integrity(rows, mutation):
    if mutation == "missing":
        rows[1].pop()
    elif mutation == "duplicate":
        rows[1][-1] = rows[1][0]
    elif mutation == "overlap":
        for r in rows[1][:6]:
            r["semantic_world_id"] = "fit-0"
    elif mutation == "wrong_menu":
        rows[1][0]["menu_layout"] = True
    else:
        rows[1][1]["compatible_bits"] = [0, 1, 0, 0]
    with pytest.raises(ValueError):
        audit.assess_rows(*rows)


@pytest.fixture
def fitted():
    rng = np.random.default_rng(17)
    x, xt = rng.normal(size=(72, 9)), rng.normal(size=(72, 9))
    y = (rng.random((72, 4)) > .55).astype(float)
    mu, ybar = x.mean(0), y.mean(0)
    scale = np.sqrt(np.mean(np.sum((x-mu)**2, axis=1)))
    z = (x-mu)/scale
    b = np.linalg.solve(z@z.T+6*np.eye(72), y-ybar)
    w = z.T@b
    model = dict(mu=mu, ybar=ybar, scale=scale, W=w, B=b, **{"lambda": 1/12})
    return x, y, xt, model, ybar+z@w, ybar+(xt-mu)/scale@w


def test_verifies_stored_equation_without_fitting(fitted, monkeypatch):
    def no_fit(*args, **kwargs):
        raise AssertionError("Auditor must not fit")
    monkeypatch.setattr(np.linalg, "solve", no_fit)
    monkeypatch.setattr(np.linalg, "lstsq", no_fit)
    result = audit.verify_stored_model(*fitted)
    assert result["normal_equation_relative_residual"] < 1e-12
    assert result["effective_df"] <= 12


@pytest.mark.parametrize("key", ["mu", "ybar", "scale", "W", "B", "lambda"])
def test_model_tampering_rejected(fitted, key):
    args = list(copy.deepcopy(fitted))
    args[3][key] = args[3][key]+.01
    with pytest.raises(ValueError):
        audit.verify_stored_model(*args)


def test_scores_not_clipped_or_cardinality_repaired(fitted):
    args = list(copy.deepcopy(fitted))
    args[-1][0] = [1, 0, 0, 0]
    with pytest.raises(ValueError):
        audit.verify_stored_model(*args)


def test_unavailable_test_row_not_imputed(fitted):
    args = list(copy.deepcopy(fitted))
    args[2][0] = np.nan
    with pytest.raises(ValueError):
        audit.verify_stored_model(*args)
    args[-1][0] = np.nan
    assert audit.verify_stored_model(*args)["evaluation_predicted_bits"][0] is None


def test_zero_training_scale_unavailable(fitted):
    args = list(copy.deepcopy(fitted))
    args[0][:] = 1
    with pytest.raises(ValueError, match="scale"):
        audit.verify_stored_model(*args)


def test_near_threshold_tolerance_cannot_flip_a_bit():
    # Repeated x under complementary binary labels gives exactly zero slopes.
    x = np.tile(np.arange(36, dtype=float)[:, None], (2, 1))
    y = np.repeat(np.r_[np.zeros(36), np.ones(36)][:, None], 4, axis=1)
    mu = x.mean(0)
    scale = np.sqrt(np.mean(np.sum((x-mu)**2, axis=1)))
    model = dict(mu=mu, ybar=y.mean(0), scale=scale, W=np.zeros((1, 4)), **{"lambda": 1/12})
    scores = np.full((72, 4), .5)
    altered = scores.copy()
    altered[0, 0] -= 1e-13
    with pytest.raises(ValueError, match="prediction differs"):
        audit.verify_stored_model(x, y, x, model, scores, altered)


@pytest.fixture
def saved_run(tmp_path, rows, monkeypatch):
    """Actual producer numerical/export schema, entirely synthetic source arrays."""
    spec = importlib.util.spec_from_file_location("synthetic_support_producer", Path(__file__).resolve().parents[1]
        / "scripts/choice_support_readout_offline.py")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    prepared = []
    for split, group in zip(("fit", "evaluation"), rows, strict=True):
        for r in group:
            visible = r["visible_problem_sha256"]
            if visible == "fit:9:2":
                visible = audit.OVERLAP
            prepared.append(dict(split=split, collection="v1" if split == "fit" else "v2",
                trial_id=r["trial_id"], assignment_sha256="a"*64, record_sha256="b"*64,
                world_slot=r["world_slot"], semantic_world_id=r["semantic_world_id"],
                delivery=r["delivery"], menu_layout=r["menu_layout"], known_attribute=0 if r["initial_attribute"] == "color" else 1,
                oracle_bits=r["compatible_bits"], visible_problem_sha256=visible, available=True,
                capture={}, source_flags=dict(valid=True, cap_hit=False, thinking_closed=True, closure_count=1)))
    seen = {r["visible_problem_sha256"] for r in prepared[:72]}
    for r in prepared:
        r["seen_in_fit"] = r["visible_problem_sha256"] in seen
    rng = np.random.default_rng(18)
    features = rng.normal(size=(144, 5120))
    source_rows = copy.deepcopy(prepared)
    for r in source_rows:
        del r["seen_in_fit"]
    monkeypatch.setattr(audit, "source_rows", lambda project, collection, reader: (
        copy.deepcopy(source_rows[:72] if collection["version"] == "v1" else source_rows[72:]),
        features[:72].copy() if collection["version"] == "v1" else features[72:].copy()))
    code_root = tmp_path/"code"
    code_files = {}
    for name in audit.CODE_FILES:
        path = code_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic bound source: " + name)
        code_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = dict(schema_version=1, assay="choice_support_readout_offline", method=audit.METHOD,
        hidden_feature_values_read=False, causal_admission=False, code_files=code_files,
        collections=[dict(version="v1"), dict(version="v2")], rows=prepared, rows_sha256=audit.digest(prepared),
        visible_overlap=dict(sha256=audit.OVERLAP,
            evaluation_trial_ids=[r["trial_id"] for r in prepared[72:] if r["seen_in_fit"]]))
    manifest = dict(schema_version=1, assay="choice_support_readout_offline", admitted=True,
        input_ledger=ledger, input_ledger_sha256=audit.digest(ledger), output_relative_path="output")
    output = tmp_path/"output"
    output.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"))+"\n")
    manifest_path = tmp_path/"manifest.json"
    write(manifest_path, manifest)
    write(output/"input-ledger.json", ledger)
    write(output/"started.json", dict(manifest_sha256=audit.digest(manifest), input_ledger_sha256=audit.digest(ledger)))
    def publish(model_unavailable=False):
        if model_unavailable:
            features[:72] = 0  # Fixed zero-scale failure, not another fit.
        model, diagnostics, shortcut, predictions = producer.build_predictions(prepared, features)
        available = bool(model["available"])
        model_path = output/"model.npz"
        if available:
            np.savez_compressed(model_path, **model)
        elif model_path.exists():
            model_path.unlink()
        payload = dict(schema_version=1, rows=predictions)
        write(output/"predictions.json", payload)
        names = ["input-ledger.json", "predictions.json", "started.json"] + (["model.npz"] if available else [])
        artifacts = {name: dict(path=name, byte_sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()) for name in names}
        artifacts["predictions.json"]["content_sha256"] = audit.digest(payload)
        artifacts["input-ledger.json"]["content_sha256"] = audit.digest(ledger)
        result = dict(schema_version=1, assay="choice_support_readout_offline", complete=True, model_available=available,
            manifest_sha256=audit.digest(manifest), input_ledger_sha256=audit.digest(ledger), method=audit.METHOD,
            artifacts=artifacts, model_diagnostics=diagnostics, shortcut=shortcut,
            **producer.assess_predictions(predictions, model_available=available),
            causal_admission=False, treated_distribution_validated=False, retrospective_boundary=True)
        write(output/"result.json", result)
        return result
    result = publish()
    return dict(project=tmp_path, manifest=manifest_path, output=output, code_root=code_root,
        publish=publish, result=result, write=write, source_rows=source_rows)


def run_audit(case):
    return audit.audit(case["project"], case["manifest"], case["output"], code_root=case["code_root"])


def test_real_schema_synthetic_producer_integration(saved_run):
    result = run_audit(saved_run)
    assert result["complete"] and result["computation_agrees"]
    assert result["assigned_rows"] == 144
    assert result["instrument_feasibility_passed"] == saved_run["result"]["instrument_feasibility_passed"]
    assert not result["causal_admission"]


def test_unavailable_fit_no_model_retains_all_rows(saved_run):
    saved_run["publish"](model_unavailable=True)
    result = run_audit(saved_run)
    assert result["complete"] and result["computation_agrees"]
    assert not result["model_available"] and not result["instrument_feasibility_passed"]
    assert result["independent_metrics"]["evaluation"]["assigned"] == 72
    assert result["independent_metrics"]["evaluation"]["exact_correct"] == 0


@pytest.mark.parametrize("which", ["metric", "decision", "prediction", "shortcut", "code", "destination", "overlap"])
def test_file_schema_tampering_rejected(saved_run, which):
    output = saved_run["output"]
    write = saved_run["write"]
    result = json.loads((output/"result.json").read_text())
    if which == "metric":
        result["metrics"]["primary_cells"][0]["correct"] += 1
    elif which == "decision":
        result["instrument_feasibility_passed"] = not result["instrument_feasibility_passed"]
    elif which == "prediction":
        payload = json.loads((output/"predictions.json").read_text())
        payload["rows"][72]["predicted_bits"][0] ^= 1
        write(output/"predictions.json", payload)
        result["artifacts"]["predictions.json"].update(byte_sha256=hashlib.sha256((output/"predictions.json").read_bytes()).hexdigest(), content_sha256=audit.digest(payload))
    elif which == "shortcut":
        result["shortcut"]["1"]["bits"] = [1, 1, 1, 1]
    elif which == "code":
        (saved_run["code_root"]/"reports/choice-support-readout-offline-v1-plan.md").write_text("changed")
    elif which == "destination":
        manifest = json.loads(saved_run["manifest"].read_text())
        manifest["output_relative_path"] = "another-output"
        write(saved_run["manifest"], manifest)
    else:
        # Same label cardinalities and denominators cannot hide changed source overlap identity.
        saved_run["source_rows"][72+3*6+2]["visible_problem_sha256"] = "unseen"
    write(output/"result.json", result)
    with pytest.raises(ValueError):
        run_audit(saved_run)


def test_terminal_failure_retains_assignments(saved_run):
    output = saved_run["output"]
    (output/"result.json").unlink()
    manifest = json.loads(saved_run["manifest"].read_text())
    saved_run["write"](output/"failure.json", dict(manifest_sha256=audit.digest(manifest),
        input_ledger_sha256=manifest["input_ledger_sha256"], complete=False,
        instrument_feasibility_passed=False, causal_admission=False, error_type="ValueError", error="synthetic failure"))
    result = run_audit(saved_run)
    assert not result["complete"] and not result["passed"]
    assert len(result["retained_assignment_rows"]) == 144


def test_reader_rejects_mutation_and_duplicate_json(tmp_path):
    path = tmp_path/"input.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        audit.Reader().json(path)
    path.write_text('{"a":1}')
    reader = audit.Reader()
    reader.json(path)
    path.write_text('{"a":2}')
    with pytest.raises(ValueError, match="hash differs"):
        reader.unchanged()


def test_source_loader_reconstructs_display_labels_and_exact_capture_rows(tmp_path, monkeypatch):
    """All nine synthetic physical cohorts; no original closed feature files."""
    root = tmp_path/"runs/qwen32-voluntary-resolution-v1"
    root.mkdir(parents=True)
    source_files = {}
    with tarfile.open(root/"source.tar.gz", "w:gz") as archive:
        for name in audit.SOURCE_FILES:
            raw = ("# synthetic " + name).encode()
            source_files[name] = hashlib.sha256(raw).hexdigest()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    assignments = []
    for world in range(12):
        for delivery in audit.DELIVERIES:
            for menu in (0, 1):
                assignments.append(dict(trial_id=f"object-{world}-{delivery}-{menu}", family="objects", kind="choice",
                    world_slot=world, semantic_world_id=f"world-{world}", delivery=delivery, menu_layout=menu, known=[1]))
    assignments += [dict(trial_id=f"other-{i}", family="machines", kind="new_fact") for i in range(216)]
    manifest = dict(diagnostics=assignments, assignment_sha256=audit.digest(assignments), source_sha256="s"*64)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    write(root/"manifest.json", manifest)
    archive_sha = hashlib.sha256((root/"source.tar.gz").read_bytes()).hexdigest()
    write(root/"freeze.json", dict(manifest_sha256=audit.digest(manifest), source_sha256="s"*64, archive_sha256=archive_sha))
    labels, captures, rows, bindings = [], [], [], []
    hidden = np.zeros((32, 4, 7, 5120), dtype=np.float32)
    positions = ["record", "prefill", "token_32", "token_128", "token_512", "think_end", "decision"]
    token_positions = np.tile(np.arange(7), (32, 1))
    files = [root/name for name in ("manifest.json", "freeze.json", "source.tar.gz")]
    for offset in range(0, 288, 32):
        group = assignments[offset:offset+32]
        path = root/f"activations/attempt/{offset:07d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, hidden=hidden, layers=np.array([15, 31, 47, 57]), positions=np.array(positions),
            trial_ids=np.array([a["trial_id"] for a in group]), available=np.ones((32, 4, 7), dtype=bool),
            token_positions=token_positions, closure_counts=np.ones(32, dtype=np.int64))
        byte_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records = []
        for index, a in enumerate(group):
            desc = dict(path=str(path.relative_to(root)), row=index, sha256=byte_sha,
                available=np.ones((4, 7), dtype=bool).tolist(), token_positions=list(range(7)))
            extra = "The color is teal." if a.get("delivery") == audit.DELIVERIES[0] else "The shape is diamond."
            content = prompt(extra)
            compatible = [0] if a.get("delivery") == audit.DELIVERIES[0] else [0, 2]
            record = dict(trial_id=a["trial_id"], assignment=a, activation_artifact=desc,
                inference=dict(messages=[dict(role="system", content="Instructions"), dict(role="user", content=content)],
                    rendered_prompt="template\n"+content, unpadded_input_ids=[1, 2]),
                task=dict(attributes=["color", "shape"], candidates=[["teal", "diamond"], ["navy", "star"],
                    ["navy", "diamond"], ["teal", "star"]], compatible=compatible),
                valid=True, cap_hit=False, thinking_closed=True)
            records.append(record)
            rows.append(dict(trial_id=a["trial_id"], record_sha256=audit.digest(record), activation_artifact=desc))
            bindings.append(dict(trial_id=a["trial_id"], record_sha256=audit.digest(record), artifact_sha256=byte_sha,
                row=index, token_prefix_sha256=audit.digest([1, 2])))
            if a["kind"] == "choice":
                label = dict(trial_id=a["trial_id"], assignment_sha256=audit.digest(a), world_slot=a["world_slot"],
                    semantic_world_id=a["semantic_world_id"], delivery=a["delivery"], menu_layout=a["menu_layout"],
                    candidate_labels=list("WXYZ"), compatible_bits=[int(j in compatible) for j in range(4)])
                labels.append(label)
                captures.append(dict(**label, record_sha256=audit.digest(record), activation_path=desc["path"],
                    activation_byte_sha256=byte_sha, activation_row=index, layer=57, position="decision", available=True,
                    token_position=6, closure_count=1))
        cohort_path = root/f"cohorts/attempt/{offset:07d}.json"
        write(cohort_path, dict(manifest_hash=audit.digest(manifest), records=records, sha256=audit.digest(records)))
        files.extend((path, cohort_path))
    old_audit_path = tmp_path/"reports/qwen32-voluntary-resolution-v1-audit.json"
    write(old_audit_path, dict(manifest_sha256=audit.digest(manifest), source_sha256="s"*64,
        complete=True, integrity_passed=True, outcomes=rows, generation_record_bindings_sha256=audit.digest(bindings)))
    files.append(old_audit_path)
    inventory = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    pin = dict(manifest=audit.digest(manifest), source="s"*64, archive=archive_sha,
        audit=hashlib.sha256(old_audit_path.read_bytes()).hexdigest(), assignments=audit.digest(assignments[:72]),
        labels=audit.digest(labels), captures=audit.digest(captures), inventory=audit.digest(inventory), attempt="attempt")
    monkeypatch.setitem(audit.PINS, "v1", pin)
    collection = dict(version="v1", pins=pin, files=inventory, source_files=source_files)
    reader = audit.Reader()
    selected, values = audit.source_rows(tmp_path, collection, reader)
    assert len(selected) == 72 and values.shape == (72, 5120)
    assert selected[0]["oracle_bits"] == [1, 0, 0, 0]
    assert selected[2]["oracle_bits"] == [1, 0, 1, 0]
    assert selected[32]["capture"]["row"] == 0
    assert len(reader.files) == 22
    reader.unchanged()
