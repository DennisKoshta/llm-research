"""Local campaign order, immutable input bindings, and shared budget admission."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest

from curiosity_study import satisfaction_site_admission as admission
from curiosity_study import satisfaction_site_resources as resources
from curiosity_study import satisfaction_causal_sites as study
from curiosity_study.artifacts import digest

MATERIALS = "a"*64
CALIBRATION_PARENT = "b"*64
VALIDATION_PARENT = "c"*64


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stage(root, phase, *, alias="original", materials=MATERIALS, setup_hash=None, calibration_hash=CALIBRATION_PARENT):
    shared = dict(campaign_id=study.REVISION, reserved_materials_sha256=materials,
                  budget=deepcopy(resources.POLICIES[phase]))
    if phase == "setup":
        manifests = [dict(**shared, assay=resources.SETUP_ASSAY, phase="engineering")]
    else:
        manifests = [dict(**shared, assay=study.ASSAY, site_phase=phase, site_plan={"phase": phase},
                          execution_group=group, parent_manifest_sha256=CALIBRATION_PARENT if phase == "calibration" else VALIDATION_PARENT,
                          setup_certificate={"manifest_sha256": setup_hash or "d"*64},
                          **({"selection_receipt": {"manifest_sha256": calibration_hash}} if phase == "validation" else {}))
                     for group in study.GROUPS]
    jobs = {}
    for index, manifest in enumerate(manifests):
        job = (root/"runs"/f"{phase}-{alias}-{index}").resolve()
        put(job/"manifest.json", manifest)
        jobs[str(job)] = digest(manifest)
    seconds = resources.POLICIES[phase]["shared_deadline_seconds"]
    reservation = dict(gpu="H200", rate_usd_per_second=admission.H200_RATE, jobs=jobs,
                       created=1000.0, deadline=1000.0+seconds,
                       maximum_combined_compute_usd=len(manifests)*(seconds+120)*admission.H200_RATE)
    return manifests, reservation, root/"reservations"/f"{phase}-{alias}.json"


def reserve(root, phase, **kwargs):
    manifests, reservation, destination = stage(root, phase, **kwargs)
    value = admission.reserve_stage(root, manifests, reservation, destination)
    return manifests, reservation, destination, value


def prerequisites(root, *, validation=False):
    setup = reserve(root, "setup")
    setup_hash = digest(setup[0][0])
    calibration = reserve(root, "calibration", setup_hash=setup_hash) if validation else None
    return setup_hash, setup, calibration


def test_complete_once_only_campaign_has_exact1_3_3_workers_and_at_most65(tmp_path):
    setup = reserve(tmp_path, "setup")
    assert len(setup[1]["jobs"]) == 1 and setup[3]["budget"]["stage_ceiling_usd"] == 3
    assert admission.verify_worker_claim(tmp_path, setup[0][0], setup[1], setup[2]) == setup[3]
    setup_hash = digest(setup[0][0])
    calibration = reserve(tmp_path, "calibration", setup_hash=setup_hash)
    assert len(calibration[1]["jobs"]) == 3 and calibration[3]["budget"]["stage_ceiling_usd"] == 34
    for manifest in calibration[0]:
        assert admission.verify_worker_claim(tmp_path, manifest, calibration[1], calibration[2]) == calibration[3]
    validation = reserve(tmp_path, "validation", setup_hash=setup_hash)
    assert len(validation[1]["jobs"]) == 3 and validation[3]["budget"]["stage_ceiling_usd"] == 28
    for manifest in validation[0]:
        assert admission.verify_worker_claim(tmp_path, manifest, validation[1], validation[2]) == validation[3]
    assert sum(item[3]["maximum_combined_compute_usd"] for item in (setup, calibration, validation)) <= 65
    assert {p.stem for p in admission.directory(tmp_path).glob("*.json")} == set(admission.ORDER)
    with pytest.raises(ValueError, match="superseded"):
        admission.verify_worker_claim(tmp_path, setup[0][0], setup[1], setup[2])


@pytest.mark.parametrize("phase", admission.ORDER)
def test_same_phase_cannot_be_repeated_under_new_run_alias(tmp_path, phase):
    setup = reserve(tmp_path, "setup")
    setup_hash = digest(setup[0][0])
    if phase in ("calibration", "validation"):
        reserve(tmp_path, "calibration", setup_hash=setup_hash)
    if phase == "validation":
        reserve(tmp_path, "validation", setup_hash=setup_hash)
    manifests, reservation, destination = stage(tmp_path, phase, alias="renamed", setup_hash=setup_hash)
    with pytest.raises(ValueError, match="repeated, skipped, or already superseded"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists()


@pytest.mark.parametrize("phase", ["calibration", "validation"])
def test_skipping_preceding_phase_fails_without_consuming_requested_phase(tmp_path, phase):
    manifests, reservation, destination = stage(tmp_path, phase)
    with pytest.raises(ValueError, match="skipped"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists() and not (admission.directory(tmp_path)/f"{phase}.json").exists()


@pytest.mark.parametrize("change", ["material", "setup", "calibration"])
def test_cross_phase_material_and_evidence_cannot_be_swapped(tmp_path, change):
    setup_hash, _, _ = prerequisites(tmp_path, validation=change == "calibration")
    phase = "validation" if change == "calibration" else "calibration"
    kwargs = dict(setup_hash="e"*64 if change == "setup" else setup_hash,
                  materials="f"*64 if change == "material" else MATERIALS,
                  calibration_hash="9"*64 if change == "calibration" else CALIBRATION_PARENT)
    manifests, reservation, destination = stage(tmp_path, phase, **kwargs)
    with pytest.raises(ValueError, match="Earlier campaign admission changed|different setup|different calibration"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists() and not (admission.directory(tmp_path)/f"{phase}.json").exists()


@pytest.mark.parametrize("change", ["gpu", "rate", "liability", "too_long", "too_short", "nan", "infinity", "boolean_time", "jobs"])
def test_changed_or_nonfinite_reservation_fails_before_consumption(tmp_path, change):
    manifests, reservation, destination = stage(tmp_path, "setup")
    if change == "gpu":
        reservation["gpu"] = "A100"
    elif change == "rate":
        reservation["rate_usd_per_second"] /= 2
    elif change == "liability":
        reservation["maximum_combined_compute_usd"] = 0
    elif change in ("too_long", "too_short"):
        seconds = 1801 if change == "too_long" else 179
        reservation["deadline"] = reservation["created"]+seconds
        reservation["maximum_combined_compute_usd"] = (seconds+120)*admission.H200_RATE
    elif change in ("nan", "infinity", "boolean_time"):
        reservation["created"] = {"nan": float("nan"), "infinity": float("inf"), "boolean_time": True}[change]
    else:
        reservation["jobs"][next(iter(reservation["jobs"]))] = "0"*64
    with pytest.raises(ValueError):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists() and not (admission.directory(tmp_path)/"setup.json").exists()


@pytest.mark.parametrize("phase", admission.ORDER)
def test_wrong_worker_count_or_mixed_stage_cannot_reserve(tmp_path, phase):
    manifests, reservation, destination = stage(tmp_path, phase)
    if phase == "setup":
        manifests *= 2
    else:
        manifests.pop()
    with pytest.raises(ValueError):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists()


def test_edited_prior_reservation_is_rejected_before_advancing(tmp_path):
    setup_hash, setup, _ = prerequisites(tmp_path)
    altered = deepcopy(setup[1])
    altered["arbitrary_extra_field"] = "changed"
    put(setup[2], altered)
    manifests, reservation, destination = stage(tmp_path, "calibration", setup_hash=setup_hash)
    with pytest.raises(ValueError, match="Earlier campaign admission or reservation changed"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists()


def test_bound_input_manifest_must_still_equal_reserved_hash(tmp_path):
    manifests, reservation, destination = stage(tmp_path, "setup")
    path = Path(next(iter(reservation["jobs"]))) / "manifest.json"
    changed = deepcopy(manifests[0])
    changed["campaign_id"] = "different"
    put(path, changed)
    with pytest.raises(ValueError, match="Reserved job manifest changed"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)


@pytest.mark.parametrize("change", ["jobs", "phase", "budget", "liability", "reservation"])
def test_worker_rechecks_ledger_and_published_reservation(tmp_path, change):
    manifests, reservation, destination, binding = reserve(tmp_path, "setup")
    edited = deepcopy(binding)
    if change == "jobs":
        edited["jobs"] = {str(tmp_path/"fake-job"): digest(manifests[0])}
    elif change == "phase":
        edited["phase"] = "calibration"
    elif change == "budget":
        edited["budget"]["stage_ceiling_usd"] = 65
    elif change == "liability":
        edited["maximum_combined_compute_usd"] = 0
    else:
        changed = deepcopy(reservation)
        changed["new_field"] = "different"
        put(destination, changed)
    if change != "reservation":
        put(admission.directory(tmp_path)/"setup.json", edited)
    with pytest.raises(ValueError, match="Earlier campaign admission or reservation changed"):
        admission.verify_worker_claim(tmp_path, manifests[0], reservation, destination)


def test_worker_cannot_claim_another_path_or_unreserved_manifest(tmp_path):
    manifests, reservation, destination, _ = reserve(tmp_path, "setup")
    alias = tmp_path/"same-reservation-other-path.json"
    put(alias, reservation)
    with pytest.raises(ValueError, match="Worker campaign admission changed"):
        admission.verify_worker_claim(tmp_path, manifests[0], reservation, alias)
    changed = deepcopy(manifests[0])
    changed["unreserved"] = True
    with pytest.raises(ValueError, match="Worker campaign admission changed"):
        admission.verify_worker_claim(tmp_path, changed, reservation, destination)


def test_combined_guard_counts_existing_liability_and_consumes_nothing_on_failure(tmp_path, monkeypatch):
    setup_hash, setup, calibration = prerequisites(tmp_path, validation=True)
    manifests, reservation, destination = stage(tmp_path, "validation", setup_hash=setup_hash)
    total = setup[3]["maximum_combined_compute_usd"] + calibration[3]["maximum_combined_compute_usd"] + reservation["maximum_combined_compute_usd"]
    #All original stage policies remain unchanged; isolate the aggregate guard.
    monkeypatch.setattr(resources, "CAMPAIGN_CEILING", total-0.01)
    with pytest.raises(ValueError, match="combined ceiling"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert not destination.exists() and not (admission.directory(tmp_path)/"validation.json").exists()


def test_partial_publication_consumes_phase_and_cannot_silently_retry(tmp_path, monkeypatch):
    manifests, reservation, destination = stage(tmp_path, "setup")
    original = admission.write_once
    def interrupted(path, value):
        if Path(path) == destination:
            raise OSError("synthetic publication interruption")
        return original(path, value)
    monkeypatch.setattr(admission, "write_once", interrupted)
    with pytest.raises(OSError, match="publication interruption"):
        admission.reserve_stage(tmp_path, manifests, reservation, destination)
    assert (admission.directory(tmp_path)/"setup.json").exists() and not destination.exists()
    monkeypatch.setattr(admission, "write_once", original)
    alternate, alternate_reservation, alternate_path = stage(tmp_path, "setup", alias="retry")
    with pytest.raises(ValueError, match="repeated"):
        admission.reserve_stage(tmp_path, alternate, alternate_reservation, alternate_path)
    assert not alternate_path.exists()


def test_two_local_contenders_cannot_consume_setup_twice(tmp_path):
    inputs = [stage(tmp_path, "setup", alias=f"contender{i}") for i in range(2)]
    def attempt(args):
        try:
            return admission.reserve_stage(tmp_path, *args)
        except ValueError as error:
            return str(error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, inputs))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert sum(args[2].exists() for args in inputs) == 1
    assert {p.stem for p in admission.directory(tmp_path).glob("*.json")} == {"setup"}
