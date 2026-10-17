"""Synthetic complete-grid tests of fixed-site selection and paired estimators."""
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pytest

from curiosity_study.artifacts import digest
from curiosity_study import satisfaction_causal_sites as screen
from curiosity_study import voluntary_resolution as voluntary

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_satisfaction_causal_sites as a  # noqa: E402


def tables(phase="calibration", selected_site=31):
    calibration = phase == "calibration"
    sites = screen.SITES if calibration else (selected_site,)
    arms = screen.CALIBRATION_ARMS if calibration else screen.VALIDATION_ARMS
    panels = ("panel_1", "panel_2") if calibration else ("half_1", "half_2")
    slots = tuple(range(12)) if calibration else (0, 1, 4, 5, 6, 7, 10, 11)
    variants = [("choice", d, m) for d in voluntary.DELIVERIES for m in (0, 1)]
    variants += [(k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
                 for k in voluntary.DIAGNOSTICS]
    assignments, outcomes = [], []
    for pi, panel in enumerate(panels):
        for wi, slot in enumerate(slots):
            ordinal = pi*len(slots)+wi
            world = f"{ordinal:064x}"
            for kind, delivery, menu in variants:
                base = f"{panel}-{wi}-{kind}-{delivery}-{menu}"
                for site in sites:
                    for arm in arms:
                        row = dict(trial_id=f"{site}-{arm}-{base}", base_trial_id=base, phase=phase, site=site,
                                   panel_id=panel, semantic_world_id=world, world_slot=slot, arm=arm,
                                   family="objects", kind=kind, delivery=delivery, menu_layout=menu)
                        if not calibration:
                            row["reserved_ordinal"] = ordinal
                        assignments.append(row)
                        outcomes.append(dict(**row, status="completed", valid=True, thinking_closed=True,
                                             cap_hit=False, correct=None if kind == "choice" else True,
                                             selected_service=("stop" if delivery == "decisive_evidence" else "report")
                                             if kind == "choice" else None))
    return dict(phase=phase, sites=list(sites), arms=list(arms), assignments=assignments,
                assignment_sha256=digest(assignments)), outcomes


def checks(plan):
    return (dict.fromkeys(a.TOP_GATES, True),
            {str(s): {**dict.fromkeys(a.GATES, True), "readiness": {"passed": True}} for s in plan["sites"]})


def run(plan, rows, *, top=None, per_site=None):
    default, sites = checks(plan)
    return a.analyze_tables(plan, rows, checks=default if top is None else top,
                            site_checks=sites if per_site is None else per_site)


def choose(rows, *, site=None, arm=a.PRIMARY, worlds=(0, 1, 2), panels=None, menus=(0, 1), service="stop"):
    for row in rows:
        local = int(row["semantic_world_id"], 16) % (12 if row["phase"] == "calibration" else 8)
        if row["arm"] == arm and (site is None or row["site"] == site) and row["kind"] == "choice" \
                and row["delivery"] == "redundant_evidence" and local in worlds \
                and row["menu_layout"] in menus and (panels is None or row["panel_id"] in panels):
            row["selected_service"] = service


def boundary(phase="calibration", selected_site=31):
    plan, rows = tables(phase, selected_site)
    choose(rows, worlds=range(3 if phase == "calibration" else 2))
    for arm in plan["arms"][3:]:
        choose(rows, arm=arm, worlds=[0])
        if phase == "calibration":
            choose(rows, arm=arm, worlds=[1], menus=[0])
    return plan, rows


def effect(result, *, site=31, left=a.PRIMARY, right="zero_sham", panel="pooled", menu="averaged", category="stop"):
    selected = [r for r in result["behavior_contrasts"] if r["site"] == site and r["left_arm"] == left
                and r["right_arm"] == right and r["panel_id"] == panel and r["menu_layout"] == menu
                and r["category"] == category and r["delivery"] == "redundant_evidence"]
    assert len(selected) == 1
    return selected[0]


def getrow(rows, *, site=31, arm=a.PRIMARY, panel="panel_1", world=5, kind="choice", menu=0):
    return next(r for r in rows if r["site"] == site and r["arm"] == arm and r["panel_id"] == panel
                and r["world_slot"] == world and r["kind"] == kind and r["menu_layout"] == menu
                and r["delivery"] == "redundant_evidence")


def invalidate(row, status="completed", capped=False):
    row.update(status=status, valid=False if status == "completed" else None, thinking_closed=True,
               cap_hit=capped, correct=None if row["kind"] == "choice" else False, selected_service=None)


@pytest.fixture(scope="module")
def calibration_boundary():
    return run(*boundary())


@pytest.fixture(scope="module")
def validation_boundary():
    return run(*boundary("validation", 47))


def test_exact_primary_and_all_control_boundaries_and_earliest_site(calibration_boundary):
    r = calibration_boundary
    assert r["decision"]["positive_screen"] and r["selected_site"] == 31 and r["assigned_site"] is None
    assert set(r["site_decisions"]) == {"31", "47"}
    for decision in r["site_decisions"].values():
        assert decision["primary_observed"]["panel_differences"] == dict(panel_1=.25, panel_2=.25, pooled=.25)
        assert len(decision["controls"]) == 4
        assert all(c["observed"]["panel_differences"] == dict(panel_1=.125, panel_2=.125, pooled=.125)
                   for c in decision["controls"].values())
    assert (r["assigned"], r["assigned_choices"], r["assigned_worlds"]) == (4032, 2016, 24)
    assert len(r["denominators"]) == 3024
    assert len(r["behavior_contrasts"]) == 4320 and len(r["world_contrasts"]) == 34560
    assert len(r["decision_bounds"]) == 90
    for site in (31, 47):
        assert len(r["task_readiness"][str(site)]["diagnostic_cells"]) == 84
        assert len(r["task_readiness"][str(site)]["choice_cells"]) == 42
        for arm in screen.CALIBRATION_ARMS:
            assert sum(x["count"] for x in r["denominators"] if x["site"] == site and x["arm"] == arm
                       and x["panel_id"] == "pooled" and x["menu_layout"] == "averaged") == 144


def test_validation_keeps_assigned_site_all12_controls_and_whole16_readiness(validation_boundary):
    r = validation_boundary
    assert r["decision"]["classification"] == "reserved_world_direction_selective_validation"
    assert r["selected_site"] is None and r["assigned_site"] == 47 and r["decision"]["positive_screen"]
    assert len(r["site_decisions"]["47"]["controls"]) == 12
    assert (r["assigned"], r["assigned_choices"], r["assigned_worlds"]) == (2880, 1440, 16)
    readiness = r["task_readiness"]["47"]
    assert len(readiness["diagnostic_cells"]) == 90 and len(readiness["choice_cells"]) == 45
    assert {x["panel_id"] for x in readiness["diagnostic_cells"]} == {"all_reserved"}
    assert all(x["assigned"] == 16 for x in readiness["diagnostic_cells"])
    assert len(r["behavior_contrasts"]) == 5616 and len(r["world_contrasts"]) == 29952
    assert len(r["decision_bounds"]) == 117 and len(r["denominators"]) == 3240


def test_larger_late_effect_never_beats_earliest_qualified_site():
    plan, rows = boundary()
    choose(rows, site=47, worlds=range(12))
    result = run(plan, rows)
    assert effect(result, site=47)["estimate"] == 1 and result["selected_site"] == 31


@pytest.mark.parametrize("site,expected", [(31, 47), (47, 31)])
def test_site_readiness_is_local_but_no_rows_are_excluded(site, expected):
    plan, rows = boundary()
    invalidate(getrow(rows, site=site, kind="new_fact"))
    result = run(plan, rows)
    assert result["selected_site"] == expected and len(result["outcomes"]) == 4032
    assert not result["task_readiness"][str(site)]["passed"]


@pytest.mark.parametrize("gate", a.GATES)
def test_any_engineering_gate_anywhere_vetoes_entire_calibration_grid(gate):
    plan, rows = boundary()
    top, per_site = checks(plan)
    per_site["47"][gate] = False
    result = run(plan, rows, top=top, per_site=per_site)
    assert not result["decision"]["positive_screen"] and result["selected_site"] is None
    assert result["decision"]["classification"] == "global_integrity_or_engineering_failed"


def test_audited_readiness_false_is_never_repaired_by_recomputed_pass():
    plan, rows = boundary("validation")
    _, per_site = checks(plan)
    per_site["31"]["readiness"]["passed"] = False
    result = run(plan, rows, per_site=per_site)
    assert result["task_readiness"]["31"]["passed"]
    assert result["decision"]["classification"] == "readiness_failed"
    assert result["assigned_site"] == 31 and not result["decision"]["positive_screen"]


@pytest.mark.parametrize("service", ["report", "repeat", "pattern"])
def test_no_stopping_effect_cannot_be_rescued_by_valid_alternatives(service):
    plan, rows = tables()
    choose(rows, service=service, worlds=range(12))
    result = run(plan, rows)
    assert result["selected_site"] is None
    assert effect(result)["estimate"] == 0
    assert all(s["classification"] == "no_substantial_primary_development_effect"
               for s in result["site_decisions"].values())


@pytest.mark.parametrize("control", ["confidence_positive", "random_01"])
def test_each_control_can_veto_specificity_independently(control):
    plan, rows = boundary()
    choose(rows, arm=control, worlds=range(3))
    result = run(plan, rows)
    assert all(v["effect_threshold_met"] and not v["specificity_threshold_met"] for v in result["site_decisions"].values())
    assert result["selected_site"] is None


@pytest.mark.parametrize("comparator", ["zero_sham", "random_01"])
def test_single_allowed_invalid_comparator_cannot_artificially_satisfy_threshold(comparator):
    plan, rows = boundary()
    invalidate(getrow(rows, arm=comparator))
    result = run(plan, rows)
    decision = result["site_decisions"]["31"]
    assert result["task_readiness"]["31"]["passed"]  #23/24 choice validity is permitted
    assert not decision["positive_screen"] and result["selected_site"] == 47
    if comparator == "zero_sham":
        assert decision["primary_observed_threshold_met"] and not decision["primary_comparator_favorable_threshold_met"]
    else:
        assert decision["controls"][comparator]["observed_threshold_met"]
        assert not decision["controls"][comparator]["bounded_threshold_met"]


@pytest.mark.parametrize("status,capped,category", [("completed", False, "invalid"), ("completed", True, "capped"),
                                                    ("execution_error", False, "execution_failure"), ("missing", False, "missing")])
def test_all_failure_mass_retained_without_stopping_credit(status, capped, category):
    plan, rows = tables()
    invalidate(getrow(rows), status=status, capped=capped)
    result = run(plan, rows)
    selected = [r for r in result["denominators"] if r["site"] == 31 and r["arm"] == a.PRIMARY
                and r["panel_id"] == "panel_1" and r["delivery"] == "redundant_evidence" and r["menu_layout"] == "averaged"]
    assert len(selected) == 8 and sum(r["count"] for r in selected) == 24
    assert next(r for r in selected if r["category"] == category)["count"] == 1
    assert effect(result)["estimate"] == 0


def test_validation_readiness_pools16_but_effect_requires_both8_world_halves():
    plan, rows = boundary("validation")
    first = getrow(rows, panel="half_1", world=4, kind="new_fact")
    first["correct"] = False
    result = run(plan, rows)
    assert result["decision"]["positive_screen"]  #15/16 correct, all16 valid
    invalidate(first)
    assert run(plan, rows)["decision"]["classification"] == "readiness_failed"
    plan, rows = tables("validation")
    choose(rows, worlds=range(8), panels=["half_1"])
    result = run(plan, rows)
    assert effect(result)["estimate"] == .5 and not result["decision"]["positive_screen"]


def test_pooled_zero_menu_veto_even_when_each_half_average_is_large():
    plan, rows = tables("validation")
    choose(rows, worlds=range(8), menus=[0])
    result = run(plan, rows)
    assert effect(result)["estimate"] == .5
    assert not result["site_decisions"]["31"]["primary_observed_threshold_met"]


def test_shared_paired_bootstrap_matches_independent_world_matrix_and_panel_order():
    plan, rows = tables()
    choose(rows, site=31, worlds=[0, 3, 7], menus=[0])
    choose(rows, site=31, worlds=[1, 3, 8], menus=[1], panels=["panel_1"])
    choose(rows, site=31, arm="zero_sham", worlds=[3], menus=[0])
    choose(rows, site=47, worlds=[0, 3, 7], menus=[0])
    choose(rows, site=47, worlds=[1, 3, 8], menus=[1], panels=["panel_1"])
    choose(rows, site=47, arm="zero_sham", worlds=[3], menus=[0])
    result = run(plan, rows)
    rng = np.random.default_rng(2026101542)
    draw_banks = [rng.integers(0, 12, size=(10000, 12)) for _ in range(2)]
    vectors = []
    for panel in ("panel_1", "panel_2"):
        v = np.zeros(12)
        for i in range(12):
            for menu in (0, 1):
                treated = next(r for r in rows if r["site"] == 31 and r["panel_id"] == panel and r["world_slot"] == i
                               and r["arm"] == a.PRIMARY and r["kind"] == "choice" and r["delivery"] == "redundant_evidence"
                               and r["menu_layout"] == menu)
                control = getrow(rows, panel=panel, world=i, arm="zero_sham", menu=menu)
                v[i] += ((treated["selected_service"] == "stop") - (control["selected_service"] == "stop"))/2
        vectors.append(v)
    draws = sum(v[b].mean(axis=1) for v, b in zip(vectors, draw_banks, strict=True))/2
    expected = np.quantile(draws, [.025, .975]).tolist()
    assert effect(result)["estimate"] == sum(v.mean() for v in vectors)/2
    assert effect(result)["ci95"] == expected == effect(result, site=47)["ci95"]
    assert result["bootstrap"]["draw_bank_sha256"] == digest(dict(zip(("panel_1", "panel_2"), [b.tolist() for b in draw_banks], strict=True)))


def test_common_world_stopping_cancels_and_zero_width_is_labeled():
    plan, rows = tables()
    for arm in plan["arms"]:
        choose(rows, arm=arm, worlds=[0, 2, 4])
    result = run(plan, rows)
    endpoint = effect(result)
    assert endpoint["estimate"] == 0 and endpoint["ci95"] == [0, 0]
    assert endpoint["empirical_interval_degenerate"]
    assert "not population certainty" in result["bootstrap"]["interval"]


def test_equivalent_outcome_order_is_bitwise_reproducible(calibration_boundary):
    plan, rows = boundary()
    random.Random(91).shuffle(rows)
    assert run(plan, rows) == calibration_boundary


@pytest.mark.parametrize("change", ["duplicate", "missing", "metadata", "cell", "phase", "ordinal", "halves"])
def test_malformed_or_partial_grid_rejects(change):
    plan, rows = tables("validation")
    if change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "metadata":
        rows[0]["site"] = 47
    elif change == "phase":
        plan["sites"] = [31, 47]
    elif change == "cell":
        plan["assignments"][0]["menu_layout"] = 1
        rows[0]["menu_layout"] = 1
    elif change == "ordinal":
        for ar, row in zip(plan["assignments"], rows, strict=True):
            ar["reserved_ordinal"] += 1
            row["reserved_ordinal"] += 1
    else:
        for ar, row in zip(plan["assignments"], rows, strict=True):
            panel = "half_2" if ar["panel_id"] == "half_1" else "half_1"
            ar["panel_id"] = row["panel_id"] = panel
    plan["assignment_sha256"] = digest(plan["assignments"])
    with pytest.raises(ValueError):
        run(plan, rows)


def audit_fixture(tmp_path, *, readiness=False):
    import io
    import json
    import tarfile
    plan, outcomes = boundary("validation")
    top, per_site = checks(plan)
    per_site["31"]["readiness"]["passed"] = readiness
    code_root = Path(a.__file__).resolve().parents[1]
    payloads = {name: (code_root/name).read_bytes() for name in a.FROZEN_CODE}
    sources = {name.removeprefix("src/curiosity_study/"): payload.decode() for name, payload in payloads.items()
               if name.startswith("src/curiosity_study/")}
    manifest = dict(assay=screen.ASSAY, task_revision=screen.REVISION, site_phase=plan["phase"],
                    assignments=[], diagnostics=plan["assignments"], assignment_sha256=plan["assignment_sha256"],
                    site_plan=plan, site_plan_sha256=digest(plan), analysis_metadata=deepcopy(a.ANALYSIS_METADATA),
                    source_sha256=digest(sources), model={"id": "synthetic-no-inference"})
    audit = dict(assay="satisfaction_causal_sites_audit", site_phase=plan["phase"],
                 parent_manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
                 outcomes=outcomes, outcomes_sha256=digest(outcomes), assigned=2880, assigned_choices=1440,
                 site_checks=per_site, auditor_byte_sha256=a.sha(code_root/"scripts/audit_satisfaction_causal_sites.py"), **top)
    run_dir = tmp_path/"run"
    run_dir.mkdir()
    (run_dir/"manifest.json").write_text(json.dumps(manifest))
    audit_path = tmp_path/"audit.json"
    audit_path.write_text(json.dumps(audit))
    with tarfile.open(run_dir/"source.tar.gz", "w:gz") as archive:
        for name, payload in payloads.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    frozen = dict(manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
                  archive_sha256=a.sha(run_dir/"source.tar.gz"))
    (run_dir/"freeze.json").write_text(json.dumps(frozen))
    return run_dir, audit_path, manifest, audit


def test_bound_complete_audit_exports_six_tables_even_with_readiness_failure(tmp_path):
    import csv
    run_dir, audit_path, _, _ = audit_fixture(tmp_path)
    out = tmp_path/"analysis"
    result = a.analyze(run_dir, audit_path, out)
    assert result["decision"]["classification"] == "readiness_failed"
    assert result["task_readiness"]["31"]["passed"]
    assert result["complete"] and result["integrity_passed"]
    assert set(result["exports"]) == {"outcomes", "choices", "denominators", "behavior_contrasts", "world_contrasts", "decision_bounds"}
    for key, descriptor in result["exports"].items():
        path = out/descriptor["path"]
        assert a.sha(path) == descriptor["byte_sha256"]
        with path.open() as handle:
            assert len(list(csv.DictReader(handle))) == len(result[key]) == descriptor["rows"]
    assert "does not independently rerun" in result["provenance"]["scope"]
    original = (out/"analysis.json").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        a.analyze(run_dir, audit_path, out)
    assert (out/"analysis.json").read_bytes() == original


@pytest.mark.parametrize("change", ["complete", "integrity_passed", "parent", "source", "outcomes", "auditor"])
def test_unbound_audit_rejected_before_output_write(tmp_path, change):
    import json
    run_dir, audit_path, _, audit = audit_fixture(tmp_path)
    if change in ("complete", "integrity_passed"):
        audit[change] = False
    elif change == "outcomes":
        audit["outcomes"][0]["valid"] = False
    else:
        key = {"parent": "parent_manifest_sha256", "source": "source_sha256", "auditor": "auditor_byte_sha256"}[change]
        audit[key] = "0"*64
    audit_path.write_text(json.dumps(audit))
    out = tmp_path/"must-not-exist"
    with pytest.raises(ValueError):
        a.analyze(run_dir, audit_path, out)
    assert not out.exists()


def test_rehashed_archive_with_wrong_analyzer_bytes_cannot_certify(tmp_path):
    import io
    import json
    import tarfile
    run_dir, audit_path, _, _ = audit_fixture(tmp_path)
    with tarfile.open(run_dir/"source.tar.gz") as archive:
        payloads = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
    payloads["scripts/analyze_satisfaction_causal_sites.py"] += b"\n# illicit change\n"
    with tarfile.open(run_dir/"source.tar.gz", "w:gz") as archive:
        for name, payload in payloads.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    freeze_path = run_dir/"freeze.json"
    frozen = json.loads(freeze_path.read_text())
    frozen["archive_sha256"] = a.sha(run_dir/"source.tar.gz")
    freeze_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="Current analysis dependency"):
        a.load_inputs(run_dir, audit_path)
