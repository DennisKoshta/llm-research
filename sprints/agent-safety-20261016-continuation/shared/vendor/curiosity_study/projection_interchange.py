"""Auditable donor pairing and original-capture projection targets."""

from collections import defaultdict
import hashlib
import numpy as np

CELLS = {(0, 0), (0, 1), (1, 0), (1, 1)}


def factorial_cell(assignment):
    revision = assignment["task_revision"]
    if revision == "equal-uncertainty-v1":
        if len(assignment["known"]) != 1:
            raise ValueError("Original interchange requires exactly one known property")
        first = assignment["known"][0]
    elif revision == "relational-information-v1":
        if assignment["known"] != [2]:
            raise ValueError("Relational interchange requires the third property observed")
        first = assignment["correlated"]
    else:
        raise ValueError("Unsupported interchange task revision")
    cell = (first, assignment["query"])
    if cell not in CELLS:
        raise ValueError("Interchange requires a binary two-factor design")
    return cell


def complete_groups(assignments):
    """Validate the assigned four-cell groups before any eligibility exclusions."""
    groups = defaultdict(dict)
    for a in assignments:
        key = (
            a["task_revision"],
            a["seed"],
            a["family"],
            a["index"],
            a["kind"],
            a.get("template", "table"),
            a.get("vocabulary", "development"),
        )
        cell = factorial_cell(a)
        if cell in groups[key]:
            raise ValueError("Duplicate assigned interchange world cell")
        groups[key][cell] = a["trial_id"]
    if not groups or any(set(cells) != CELLS for cells in groups.values()):
        raise ValueError("Interchange requires complete assigned four-cell worlds")
    return dict(groups)


def vector_sha256(vector):
    """Canonical float32 little-endian values; shape is bound separately in metadata."""
    value = np.asarray(vector)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Expected one finite source activation vector")
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def unit_direction(vector):
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ValueError("Interchange requires a finite nonzero direction")
    return value / np.linalg.norm(value)


def donor_ids(cells, recipient_cell):
    first, query = recipient_cell
    return dict(
        self=cells[(first, query)],
        opposite_gain=cells[(first, 1 - query)],
        same_gain=cells[(1 - first, 1 - query)],
    )


def build_recipient_rows(config, prefix_id, prefix, donors, captures, information_unit, indices):
    """Use observed donor components unchanged, even when their ordering is unexpected.

    Matched controls calculate their scalar against the replayed information component.
    Recorded nominal doses use original captures; BF16 rounding can change achieved norms.
    """
    information_unit = unit_direction(information_unit)
    if len(set(config["random_seeds"])) < 5 or len(set(config["random_seeds"])) != len(
        config["random_seeds"]
    ):
        raise ValueError("At least five distinct seeded random directions are required")
    if any(type(seed) is not int or seed < 0 for seed in config["random_seeds"]):
        raise ValueError("Random direction seeds must be nonnegative integers")
    relations = config.get("control_relations", ["opposite_gain", "same_gain"])
    if (
        not relations
        or len(set(relations)) != len(relations)
        or not set(relations) <= {"opposite_gain", "same_gain"}
    ):
        raise ValueError("Invalid matched-control donor relations")
    if "opposite_gain" not in relations:
        raise ValueError("Opposite-information interchange requires matched controls")
    position = config["position"]
    target = "final" if position == "decision" else position
    if position not in ("record", "think_end", "decision"):
        raise ValueError("Replay only supports captured record, think_end, or decision positions")
    recipient = captures[prefix_id]
    before = float(np.dot(np.asarray(recipient["vector"], dtype=float), information_unit))
    sa = prefix["assignment"]
    first, query = factorial_cell(sa)
    gain = int(first != query)

    def row(label, **kwargs):
        return dict(
            trial_id=f"{config['name']}-{prefix_id}-{label}",
            prefix_id=prefix_id,
            kind=sa["kind"],
            condition=label,
            **kwargs,
        )

    base = dict(layer=config["layer"], position=target, capture_position=position, dose_multiplier=1.0)
    rows = [
        row(
            "baseline",
            arm="baseline",
            layer=None,
            position=None,
            dose=0.0,
            dose_multiplier=0.0,
            direction_name="baseline",
        ),
        row("zero_sham", **{**base, "dose_multiplier": 0.0}, arm="sham", dose=0.0, direction_name="sham"),
    ]
    for relation in ("self", "opposite_gain", "same_gain"):
        role = {"self": "self", "opposite_gain": "opposite_information", "same_gain": "same_information"}[
            relation
        ]
        donor_id = donors[relation]
        donor = captures[donor_id]
        projection = float(np.dot(np.asarray(donor["vector"], dtype=float), information_unit))
        shift = projection - before
        gain_shift = (1 - 2 * gain) if relation == "opposite_gain" else 0
        label = f"information_project_{relation}"
        metadata = dict(
            donor_relation=relation,
            donor_role=role,
            comparison_set=f"interchange_{role}",
            donor_source_trial_id=donor_id,
            donor_vector_sha256=donor["metadata"]["vector_sha256"],
            donor_activation=donor["metadata"],
            recipient_vector_sha256=recipient["metadata"]["vector_sha256"],
            recipient_activation=recipient["metadata"],
            recipient_original_projection=before,
            donor_original_projection=projection,
            requested_projection_shift=shift,
            requested_perturbation_norm=abs(shift),
            source_gain=gain,
            counterfactual_gain=gain + gain_shift,
            expected_gain_shift=gain_shift,
            original_projection_order_matches_semantics=bool(shift * gain_shift > 0) if gain_shift else None,
            projection_reference="original incremental-decoding capture; not a measured replay activation",
        )
        rows.append(
            row(
                label,
                **base,
                **metadata,
                arm="direction",
                patch_mode="project",
                dose=shift,
                projection_target=projection,
                direction_name=f"information_project_{relation}",
                source_index=indices["information_source"],
                layer_index=indices["layer"],
                position_index=indices["position"],
                array="unit",
            )
        )
        if relation not in relations:
            continue
        match = dict(
            **metadata,
            matched_interchange_condition=label,
            projection_target=projection,
            reference_source_index=indices["information_source"],
            reference_layer_index=indices["layer"],
            reference_position_index=indices["position"],
            reference_array="unit",
            layer_index=indices["layer"],
            position_index=indices["position"],
            norm_matching="runtime signed target minus current information projection; achieved BF16 norms audited after replay",
        )
        rows.append(
            row(
                f"answer_add_{relation}",
                **base,
                **match,
                arm="direction",
                patch_mode="matched_add",
                dose=shift,
                direction_name=f"answer_add_{relation}",
                source_index=indices["answer_source"],
                array="unit",
            )
        )
        for seed in config["random_seeds"]:
            rows.append(
                row(
                    f"random{seed}_add_{relation}",
                    **base,
                    **match,
                    arm="random",
                    patch_mode="matched_add",
                    dose=shift,
                    direction_name=f"random{seed}_add_{relation}",
                    random_seed=seed,
                )
            )
    size = config["batch_size"]
    if size not in (4, 8, 16):
        raise ValueError("Interchange batch size must be 4, 8, or 16")
    while len(rows) % size:
        rows.append(
            row(
                f"padding{len(rows)}",
                **{**base, "dose_multiplier": 0.0},
                arm="sham",
                dose=0.0,
                direction_name="padding_control",
            )
        )
    return rows
