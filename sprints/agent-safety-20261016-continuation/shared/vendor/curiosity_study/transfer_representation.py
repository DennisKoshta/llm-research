"""Apply an existing information axis to complete worlds without refitting it."""

from collections import defaultdict
import numpy as np
from .representation import CELL_ORDER


def transfer_estimand(assignments):
    """Name the varying objective quantity without reusing full-world gain."""
    revisions = {a.get("task_revision") for a in assignments}
    legacy = {"equal-uncertainty-v1", "relational-information-v1"}
    target = "target-relevant-information-v1"
    if target in revisions and revisions != {target}:
        raise ValueError("Cannot pool target relevance and full-world information estimands")
    if not revisions or not revisions <= legacy | {target}:
        raise ValueError("Transfer requires a specified two-by-two information factorial")
    result = dict(
        information_quantity="information_bits",
        construct="expected_full_world_information_gain",
        cell_order=[list(cell) for cell in CELL_ORDER],
        gain_labels_bits=[0, 1, 1, 0],
    )
    if revisions == {target}:
        result.update(
            information_quantity="expected_target_information_gain_bits",
            construct="target_relevant_expected_information_gain",
            first_factor="independent_report",
            full_world_information_quantity="information_bits",
            full_world_information_gain_bits=1,
            offered_report_entropy_bits=1,
            interpretation_scope=(
                "Objective information about the designated target under stated accurate reports. "
                "Subjective confidence and intrinsic information seeking are not measured."
            ),
        )
    return result


def transfer_groups(assignments):
    transfer_estimand(assignments)
    groups = defaultdict(dict)
    for row, a in enumerate(assignments):
        if a.get("task_revision") == "target-relevant-information-v1":
            first = a["independent_report"]
        elif a.get("task_revision") == "relational-information-v1":
            first = a["correlated"]
        elif (
            a.get("task_revision") == "equal-uncertainty-v1"
            and len(a["known"]) == 1
            and a["known"][0] in (0, 1)
        ):
            first = a["known"][0]
        else:
            raise ValueError("Transfer requires a specified two-by-two information factorial")
        key = (a["family"], a["seed"], a["index"], a["kind"], a["template"])
        cell = (first, a["query"])
        if cell in groups[key]:
            raise ValueError("Duplicate transfer world cell")
        groups[key][cell] = row
    if not groups or any(set(cells) != set(CELL_ORDER) for cells in groups.values()):
        raise ValueError("Incomplete transfer world")
    return {key: np.array([cells[c] for c in CELL_ORDER]) for key, cells in groups.items()}


def project_frozen(vectors, units):
    """No centering, whitening, sign selection, or fit on transfer observations."""
    if vectors.ndim != 4 or units.ndim != 4 or vectors.shape[1:] != units.shape[1:]:
        raise ValueError("Expected rows/sources by layers by positions by hidden dimension")
    if not np.isfinite(units).all():
        raise ValueError("A frozen axis is missing; select only available source axes prospectively")
    result = np.empty((*vectors.shape[:1], len(units), *vectors.shape[1:3]), dtype=np.float64)
    for li in range(vectors.shape[1]):
        for pi in range(vectors.shape[2]):
            result[:, :, li, pi] = vectors[:, li, pi].astype(np.float64) @ units[:, li, pi].T
    return result
