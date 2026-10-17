"""Once-only local campaign admission, shared across every run name.

This does not create cloud resources. A consumed phase cannot be replaced even
if its first launcher never starts; failures remain part of the finite study.
"""

from copy import deepcopy
import fcntl
import math
from pathlib import Path

from . import satisfaction_causal_sites as study
from . import satisfaction_site_resources as resources
from .artifacts import digest, read, write_once

H200_RATE = 0.00139376
ORDER = ("setup", "calibration", "validation")


def _require(value, message):
    if not value:
        raise ValueError(message)


def directory(root):
    return Path(root).resolve() / "campaign_admissions" / study.REVISION


def _binding(manifests, reservation, destination):
    limit, ceiling = resources.reservation_limits(manifests)
    phase = resources.phase_of(manifests[0])
    _require(all(m.get("campaign_id") == study.REVISION for m in manifests), "Changed campaign identity")
    materials = {m.get("reserved_materials_sha256") for m in manifests}
    _require(len(materials) == 1 and study.valid_sha(next(iter(materials))), "Missing reserved material commitment")
    _require(type(reservation["created"]) in (int, float)
             and type(reservation["deadline"]) in (int, float)
             and math.isfinite(reservation["created"]) and math.isfinite(reservation["deadline"]),
             "Reservation timestamps must be finite numbers")
    _require(reservation["gpu"] == "H200" and reservation["rate_usd_per_second"] == H200_RATE
             and len(reservation["jobs"]) == len(manifests)
             and sorted(reservation["jobs"].values()) == sorted(map(digest, manifests)),
             "Reservation does not bind the complete prescribed stage")
    seconds = reservation["deadline"] - reservation["created"]
    liability = len(manifests) * (seconds + 120) * H200_RATE
    _require(math.isfinite(seconds) and 180 <= seconds <= limit
             and reservation["maximum_combined_compute_usd"] == liability and liability <= ceiling,
             "Changed deadline or stage liability")
    return dict(schema_version=1, campaign_id=study.REVISION, phase=phase,
                reservation_path=str(Path(destination).resolve()), reservation_sha256=digest(reservation),
                jobs=deepcopy(reservation["jobs"]), reserved_materials_sha256=next(iter(materials)),
                parent_manifest_sha256=(digest(manifests[0]) if phase == "setup"
                                        else manifests[0]["parent_manifest_sha256"]),
                maximum_combined_compute_usd=liability, budget=deepcopy(resources.POLICIES[phase]),
                paid_retries=0)


def _bound_manifests(reservation):
    """Read only the explicit local job inputs; no execution or outcome access."""
    _require(isinstance(reservation.get("jobs"), dict) and reservation["jobs"],
             "Reservation has no bound jobs")
    manifests = []
    for name, expected in reservation["jobs"].items():
        path = Path(name)
        _require(path.is_absolute() and str(path.resolve()) == name and study.valid_sha(expected),
                 "Reservation job paths or hashes changed")
        manifest = read(path / "manifest.json")
        _require(digest(manifest) == expected, "Reserved job manifest changed")
        manifests.append(manifest)
    return manifests


def _read_admission(path):
    """Reconcile a consumed ledger against its immutable reservation and inputs."""
    binding = read(path)
    reservation = read(binding["reservation_path"])
    expected = _binding(_bound_manifests(reservation), reservation, binding["reservation_path"])
    _require(binding == expected and binding["phase"] == Path(path).stem,
             "Earlier campaign admission or reservation changed")
    return binding


def reserve_stage(root, manifests, reservation, destination):
    """Consume a phase atomically before any launcher can claim its reservation."""
    binding = _binding(manifests, reservation, destination)
    _require(sorted(map(digest, _bound_manifests(reservation))) == sorted(map(digest, manifests)),
             "Supplied stage manifests differ from reservation inputs")
    phase = binding["phase"]
    destination = Path(destination).resolve()
    _require(not destination.exists(), "This reservation was already published")
    campaign = directory(root)
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        paths = {p.stem: p for p in campaign.glob("*.json")}
        expected = set(ORDER[:ORDER.index(phase)])
        _require(set(paths) == expected, "Campaign phase was repeated, skipped, or already superseded")
        existing = {name: _read_admission(path) for name, path in paths.items()}
        _require(all(value["campaign_id"] == study.REVISION
                     and value["phase"] == name and value["budget"] == resources.POLICIES[name]
                     and value["reserved_materials_sha256"] == binding["reserved_materials_sha256"]
                     for name, value in existing.items()), "Earlier campaign admission changed")
        if phase != "setup":
            _require(all(m["setup_certificate"]["manifest_sha256"] == existing["setup"]["parent_manifest_sha256"]
                         for m in manifests), "Scientific stage uses a different setup")
        if phase == "validation":
            _require(all(m["selection_receipt"]["manifest_sha256"] == existing["calibration"]["parent_manifest_sha256"]
                         for m in manifests), "Validation uses a different calibration")
        total = sum(item["maximum_combined_compute_usd"] for item in existing.values())
        _require(total + binding["maximum_combined_compute_usd"] <= resources.CAMPAIGN_CEILING,
                 "Campaign reservation exceeds the combined ceiling")
        write_once(campaign / f"{phase}.json", binding)
        write_once(destination, reservation)
    return binding


def verify_worker_claim(root, manifest, reservation, destination):
    """Every worker requires the same once-only campaign admission."""
    phase = resources.phase_of(manifest)
    _require(phase in ORDER, "Unknown campaign phase")
    campaign = directory(root)
    _require({p.stem for p in campaign.glob("*.json")} == set(ORDER[:ORDER.index(phase)+1]),
             "Worker phase was skipped or already superseded")
    binding = _read_admission(campaign / f"{phase}.json")
    _require(read(destination) == reservation, "Worker reservation differs from its published bytes")
    _require(binding["campaign_id"] == manifest.get("campaign_id") == study.REVISION
             and binding["reservation_sha256"] == digest(reservation)
             and binding["reservation_path"] == str(Path(destination).resolve())
             and binding["reserved_materials_sha256"] == manifest["reserved_materials_sha256"]
             and digest(manifest) in binding["jobs"].values(), "Worker campaign admission changed")
    return binding
