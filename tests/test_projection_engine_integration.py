"""CPU integration of real transformer replay, donor schedules, and causal audits."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from test_state_backend import tiny_backend
from curiosity_study.artifacts import digest
from curiosity_study.causal_contrasts import analyze_contrasts, source_descriptor
from curiosity_study.information_state import assignments
from curiosity_study.intervention_integrity import audit_intervention
from curiosity_study.projection_interchange import (
    build_recipient_rows,
    complete_groups,
    donor_ids,
    factorial_cell,
    unit_direction,
    vector_sha256,
)
from curiosity_study.replay_information import intervention
from curiosity_study.state_backend import StateBackend


def analysis_module():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_replay_interventions.py"
    spec = importlib.util.spec_from_file_location("projection_analysis_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_transformer_donor_schedule_matches_runtime_scalar_and_preserves_shams(dtype):
    def assert_logits(actual, expected):
        if dtype == torch.bfloat16:
            assert actual == expected
        else:
            # CPU float32 matmul can differ by a few ulps across batch rows.
            np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-7, rtol=0)

    model = tiny_backend()
    model.model.to(dtype=dtype)
    backend = StateBackend(model)
    source = [a for a in assignments(1, kinds=("new",)) if a["family"] == "objects"]
    cells = next(iter(complete_groups(source).values()))
    requests = {}
    for a in source:
        first, query = factorial_cell(a)
        ids = [11 + first, 20 + query, 30, 40 + first, 50 + query, 60]
        requests[a["trial_id"]] = dict(
            token_prefix=ids,
            token_prefix_sha256=digest(ids),
            prefix_positions=dict(record=2, think_end=4, final=5),
            labels=["Yes", "No"],
            seed=a["sampling_seed"],
            assignment=a,
        )
    _, hidden = backend.forward(
        list(requests.values()), layers=[1], positions=("record", "think_end", "final")
    )
    captures = {}
    for index, a in enumerate(source):
        vector = hidden[index, 0, 2]
        captures[a["trial_id"]] = dict(
            vector=vector,
            metadata=dict(vector_sha256=vector_sha256(vector)),
        )
    contrast = sum(
        (0.5 if first != query else -0.5) * captures[trial]["vector"].astype(float)
        for (first, query), trial in cells.items()
    )
    information = unit_direction(contrast)
    control = np.random.default_rng(20261017).normal(size=16)
    answer = unit_direction(control - np.dot(control, information) * information)
    fitted = {"unit": np.stack([information, answer])[:, None, None, :]}
    config = dict(
        name="tiny-projection-integration",
        layer=1,
        position="decision",
        random_seeds=[3, 5, 7, 11, 13],
        batch_size=8,
    )
    indices = dict(information_source=0, answer_source=1, layer=0, position=0)
    schedule = []
    for a in source:
        key = a["trial_id"]
        schedule.extend(
            build_recipient_rows(
                config,
                key,
                requests[key],
                donor_ids(cells, factorial_cell(a)),
                captures,
                information,
                indices,
            )
        )
    repeated = [requests[a["prefix_id"]] for a in schedule]
    patches = [intervention(a, fitted, 16) for a in schedule]
    measured = []
    for start in range(0, len(schedule), config["batch_size"]):
        stop = start + config["batch_size"]
        batch, _ = backend.forward(
            repeated[start:stop], interventions=patches[start:stop], max_new_tokens=3, greedy=False
        )
        measured.extend(batch)
        assert model.active_hooks == 0
    assert model.active_hooks == 0 and all(not layer._forward_hooks for layer in model.layers)
    baseline = {a["prefix_id"]: row for a, row in zip(schedule, measured) if a["arm"] == "baseline"}
    compact, audits = [], []
    for a, row in zip(schedule, measured):
        assert row["hooks_remaining"] == 0 and row["exact_token_replay"]
        assert row["unpadded_input_ids"] == requests[a["prefix_id"]]["token_prefix"]
        if a["arm"] == "sham":
            base = baseline[a["prefix_id"]]
            assert_logits(row["label_logits"], base["label_logits"])
            assert row["completion"]["generated_token_ids"] == base["completion"]["generated_token_ids"]
            assert row["intervention"]["delta_norm"] == 0
        elif a["arm"] != "baseline":
            audited = audit_intervention(a, row["intervention"])
            assert audited["implementation_verified"], (a["condition"], audited)
            audits.append(audited)
        compact.append(
            {
                **a,
                "patch_mode": a.get("patch_mode", "add"),
                "intervention": row["intervention"],
            }
        )
    matched = analysis_module().audit_norm_matching(compact)
    assert len(matched) == 4 * 2 * 6 and all(row["passed"] for row in matched)
    # Preserve exact matching in the production precision. As for logits above,
    # CPU float32 row-layout rounding can perturb the tiny reference projection.
    if dtype == torch.bfloat16:
        assert all(row["scalar_difference"] == 0 for row in matched)
    else:
        assert max(abs(row["scalar_difference"]) for row in matched) <= 1e-7
    assert any(abs(row["requested_dose"]) > 1e-4 for row in audits)
    assert all(
        a["dose_multiplier"] == 1 for a in schedule if a.get("patch_mode") in ("project", "matched_add")
    )
    # A fresh pass has no inherited hooks/cache interventions, including after sampling.
    fresh = []
    for start in range(0, len(schedule), config["batch_size"]):
        batch, _ = backend.forward(repeated[start : start + config["batch_size"]])
        fresh.extend(batch)
    for a, row in zip(schedule, fresh):
        assert_logits(row["label_logits"], baseline[a["prefix_id"]]["label_logits"])
    assert model.active_hooks == 0


def synthetic_causal_rows():
    source = [a for a in assignments(2, kinds=("new", "repeat")) if a["family"] == "objects"]
    rows = []
    for a in source:
        descriptor = source_descriptor(a)
        gain = descriptor["information_bits"]
        base = 3.0 * (2 * gain - 1)
        for direction, delta, arm in (
            ("baseline", 0.0, "baseline"),
            ("information_project_opposite_gain", 2.0 * (1 - 2 * gain), "direction"),
            ("answer_add_opposite_gain", 0.5 if a["kind"] == "new" else -0.5, "direction"),
        ):
            rows.append(
                dict(
                    **descriptor,
                    trial_id=f"{a['trial_id']}-{direction}",
                    prefix_id=a["trial_id"],
                    direction=direction,
                    layer=1,
                    position="decision",
                    dose_multiplier=1.0,
                    arm=arm,
                    margin=base + delta,
                    margin_type="information_gain_yes_minus_no_with_polarity",
                )
            )
    return source, rows


def test_opposite_information_contrast_orients_all_four_cells_before_world_resampling():
    source, rows = synthetic_causal_rows()

    def select(direction):
        return dict(direction=direction, layer=1, position="decision", dose_multiplier=1.0)

    plan = dict(
        alpha=0.05,
        bootstrap=2000,
        seed=789,
        contrasts=[
            dict(
                name="information_vs_answer",
                left=select("information_project_opposite_gain"),
                right=select("answer_add_opposite_gain"),
                orientation="opposite_information",
            )
        ],
    )
    # Raw shifts cancel for a perfect bidirectional interchange.
    baseline = {r["prefix_id"]: r for r in rows if r["arm"] == "baseline"}
    raw = [
        r["margin"] - baseline[r["prefix_id"]]["margin"]
        for r in rows
        if r["direction"] == "information_project_opposite_gain"
    ]
    assert np.mean(raw) == 0
    result = analyze_contrasts(rows, source, plan)
    assert len(result["results"]) == 2  # Each question polarity is a separate endpoint.
    for endpoint in result["results"]:
        assert endpoint["estimate"]["mean"] == 2.0
        assert endpoint["totals"]["analyzed_worlds"] == 2
        assert endpoint["totals"]["analyzed_prefixes"] == 8
    # Losing one gain-one arm invalidates its entire four-cell world even for a gain-zero stratum.
    lost = next(
        r
        for r in rows
        if r["direction"] == "information_project_opposite_gain"
        and r["kind"] == "new"
        and r["index"] == 0
        and r["information_bits"] == 1
    )
    reduced = [r for r in rows if r["trial_id"] != lost["trial_id"]]
    plan["contrasts"][0]["stratum"] = {"information_bits": 0}
    result = analyze_contrasts(reduced, source, plan)
    new = next(r for r in result["results"] if r["kind"] == "new")
    repeat = next(r for r in result["results"] if r["kind"] == "repeat")
    assert new["totals"]["analyzed_worlds"] == 1 and new["totals"]["analyzed_prefixes"] == 2
    assert repeat["totals"]["analyzed_worlds"] == 2 and repeat["totals"]["analyzed_prefixes"] == 4
