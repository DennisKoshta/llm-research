"""Explicit-site capture primitives; no assignment selection or study executor.

All observers are parameterized by the actual layer31/47. Initial captures
precede intervention, and the next block's input independently supplies actual
postpatch vectors for all32 prefill rows. Decode geometry remains a scalar
ledger outside the separately vector-observed short engineering run.
"""

from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from . import causal_site_engineering as engineering
from . import voluntary_capture as receipts
from .artifacts import digest
from .state_backend import StateBackend

SITES = engineering.SITES
REFERENCE_LAYERS = (15, 31, 47)
POSITIONS = ("record", "final")
require = engineering.require


def capture_plan(layer):
    require(type(layer) is int and layer in SITES, "Require declared site31/47")
    return dict(
        layers=[layer],
        positions=list(POSITIONS),
        source_phase=f"prefill_before_layer{layer}_patch",
        postpatch_phase=f"layer{layer + 1}_input_after_actual_layer{layer}_patch",
        final_alias="last_original_thinking_prefix_token",
        receipt_phase="read_only_fresh_conversation_prefill",
        receipt_max_new_tokens=0,
        decode_geometry="Per-row scalar ledger; independent vectors belong to short engineering only",
    )


def _requests(requests, count):
    require(
        len(requests) == count and len({r.get("source_trial_id") for r in requests}) == count,
        "Require every distinct source row in its assigned order",
    )
    require(
        all(
            isinstance(r.get("source_trial_id"), str)
            and r["source_trial_id"]
            and type(r.get("seed")) is int
            and r.get("token_prefix_sha256") == digest(r.get("token_prefix"))
            and isinstance(r.get("token_prefix"), list)
            and r["token_prefix"]
            and all(type(t) is int and t >= 0 for t in r["token_prefix"])
            and set(r.get("prefix_positions", {})) == set(POSITIONS)
            and all(type(p) is int for p in r["prefix_positions"].values())
            and 0
            <= r["prefix_positions"]["record"]
            < r["prefix_positions"]["final"]
            == len(r["token_prefix"]) - 1
            for r in requests
        ),
        "Malformed exact source request",
    )


def _clean(model):
    require(model.active_hooks == 0 and not engineering.hook_inventory(model), "Require hook-free model")


def capture_reference(
    model, requests, *, historical_layer15_hidden, historical_binding, original_dose, deadline=None
):
    """Return a pure report/array pair for all288 original zero-generation prefixes.

    The caller/auditor must independently reconstruct historical_binding from the
    immutable source audit and artifacts. This helper binds that supplied evidence,
    verifies its exact request/array hashes, and checks actual vector equality.
    """
    _requests(requests, 288)
    _clean(model)
    require(len(model.layers) > max(REFERENCE_LAYERS), "Reference layers are absent")
    require(
        type(original_dose) in (int, float) and math.isfinite(original_dose) and original_dose > 0,
        "Require the original positive full-A15 norm",
    )
    historical = np.asarray(historical_layer15_hidden)
    require(
        historical.dtype == np.float32
        and historical.shape == (288, 1, 2, model.model.config.hidden_size)
        and np.isfinite(historical).all(),
        "Malformed archived layer15 record/final array",
    )
    expected = {
        "source_manifest_sha256",
        "source_audit_sha256",
        "record_bindings_sha256",
        "artifact_bindings_sha256",
        "requests_sha256",
        "hidden",
    }
    require(
        set(historical_binding) == expected
        and historical_binding["requests_sha256"] == digest(requests)
        and historical_binding["hidden"] == engineering.array_binding(historical)
        and all(
            isinstance(historical_binding[k], str)
            and len(historical_binding[k]) == 64
            and set(historical_binding[k]) <= set("0123456789abcdef")
            for k in expected - {"hidden"}
        ),
        "Historical evidence binding differs",
    )
    bound = digest(requests)
    cohorts, values = [], []
    for offset in range(0, 288, 32):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Reference deadline reached before the next source cohort")
        chunk = requests[offset : offset + 32]
        measured, hidden, guard = receipts.forward_receipts(
            model, chunk, [r["source_trial_id"] for r in chunk], layers=list(REFERENCE_LAYERS)
        )
        values.append(hidden)
        cohorts.append(
            dict(
                offset=offset,
                requests_sha256=digest(chunk),
                measurements=measured,
                engineering=guard,
                historical_layer15_equal=np.array_equal(hidden[:, :1], historical[offset : offset + 32]),
            )
        )
    hidden = np.concatenate(values)
    norms = np.linalg.norm(hidden[:, :, 0].astype(np.float64), axis=-1)
    medians = np.median(norms, axis=0)
    finite = bool(np.isfinite(hidden).all() and np.isfinite(medians).all() and np.all(medians > 0))
    doses = (
        {
            str(layer): float(original_dose * medians[i] / medians[0])
            for i, layer in enumerate(REFERENCE_LAYERS)
            if layer in SITES
        }
        if finite
        else None
    )
    checks = dict(
        all288_ordered_requests_unchanged=digest(requests) == bound,
        every_cohort_read_only_identity=all(c["engineering"]["passed"] is True for c in cohorts),
        all_layer15_vectors_match=all(c["historical_layer15_equal"] for c in cohorts),
        captures_and_medians_finite_positive=finite,
        hook_cleanup=model.active_hooks == 0 and not engineering.hook_inventory(model),
    )
    arrays = dict(hidden=hidden, record_norms_float64=norms)
    bindings = {k: engineering.array_binding(v) for k, v in arrays.items()}
    median_map = {str(layer): float(value) for layer, value in zip(REFERENCE_LAYERS, medians)}
    inventory = dict(
        requests_sha256=bound,
        historical_binding_sha256=digest(historical_binding),
        arrays=bindings,
        cohorts_sha256=digest(cohorts),
    )
    dose_reference = dict(
        formula="D15 * median_record_norm[layer] / median_record_norm[15]",
        original_dose=float(original_dose),
        base_branches=288,
        source_screen_manifest_sha256=historical_binding["source_manifest_sha256"],
        capture_sha256=digest(inventory),
        median_record_norm=median_map,
        doses=doses,
    )
    return dict(
        schema_version=1,
        assay="causal_site_reference",
        passed=all(checks.values()),
        checks=checks,
        scientific_admission=False,
        max_new_tokens=0,
        assigned=288,
        layers=list(REFERENCE_LAYERS),
        positions=list(POSITIONS),
        requests_sha256=bound,
        source_trial_ids=[r["source_trial_id"] for r in requests],
        historical_binding=historical_binding,
        historical_binding_sha256=digest(historical_binding),
        original_dose=float(original_dose),
        medians=median_map,
        doses=doses,
        capture_inventory=inventory,
        capture_inventory_sha256=digest(inventory),
        dose_reference=dose_reference,
        dose_rule="D_l = D15 * median_l / median_15; complete288 record positions",
        cohorts=cohorts,
        arrays=bindings,
        arrays_sha256=digest(bindings),
        scope="Read-only geometry calibration; no generated or scored scientific response; historical source provenance requires independent caller audit.",
    ), arrays


def collect_generation(model, requests, patches, *, layer, cap=4096, deadline=None):
    """Return evidence plus prepatch and independently observed postpatch arrays."""
    plan = capture_plan(layer)
    _requests(requests, 32)
    _clean(model)
    require(len(model.layers) > layer + 1, "Next-block postpatch observation is unavailable")
    _, _, inputs, targets = StateBackend(model).prepare(requests)
    before, after, dtypes = [], [], {}

    def sample(tensor):
        return np.stack(
            [
                np.stack(
                    [tensor[row, targets[p][row]].detach().float().cpu().numpy().copy() for p in POSITIONS]
                )
                for row in range(32)
            ]
        )[:, None]

    def observe_before(module, args, output):
        if not before:
            tensor = output[0] if isinstance(output, tuple) else output
            before.append(sample(tensor))
            dtypes["before"] = str(tensor.dtype)

    def observe_after(module, args):
        if not after:
            after.append(sample(args[0]))
            dtypes["after"] = str(args[0].dtype)

    try:
        with ExitStack() as stack:
            for register, observe in (
                (model.layers[layer].register_forward_hook, observe_before),
                (model.layers[layer + 1].register_forward_pre_hook, observe_after),
            ):
                handle = register(observe)
                stack.callback(handle.remove)
            evidence, _ = engineering.collect_condition(
                model, requests, patches, layer=layer, cap=cap, expected_error=True, deadline=deadline
            )
    finally:
        _clean(model)
    arrays = {}
    if before:
        arrays["hidden"] = before[0]
    if after:
        arrays["prefill_after"] = after[0]
    evidence.update(
        layer=layer,
        capture_plan=plan,
        capture_phase=plan["source_phase"],
        request_sha256=digest(requests),
        source_trial_ids=[r["source_trial_id"] for r in requests],
        unpadded_token_positions=[r["prefix_positions"] for r in requests],
        padded_token_positions=[[targets[p][row] for p in POSITIONS] for row in range(32)],
        prefill_width=int(inputs["input_ids"].shape[1]),
        activation_dtypes=dtypes,
        arrays={k: engineering.array_binding(v) for k, v in arrays.items()},
    )
    return evidence, arrays


def validate_prefill_geometry(condition, arrays, patches, *, torch):
    """Recompute actual all-row BF16 add/cast and first-event geometry independently."""
    if condition.get("rows") is None or set(arrays) != {"hidden", "prefill_after"} or len(patches) != 32:
        return False
    before, after = arrays["hidden"], arrays["prefill_after"]
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}.get(
        condition["activation_dtypes"].get("before")
    )
    if (
        dtype is None
        or condition["activation_dtypes"].get("after") != condition["activation_dtypes"]["before"]
        or before.shape != after.shape
        or list(before.shape[:3]) != [32, 1, 2]
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
    ):
        return False
    for i, patch in enumerate(patches):
        a, b = torch.tensor(before[i, 0, 0]), torch.tensor(after[i, 0, 0])
        if not np.array_equal(before[i, 0, 1], after[i, 0, 1]):
            return False
        if patch is None:
            if not torch.equal(a, b) or condition["rows"][i]["intervention_trace"]["events"]:
                return False
            continue
        unit = torch.tensor(patch["direction"], dtype=torch.float32)
        requested = a + float(patch["dose"]) * unit
        events = condition["rows"][i]["intervention_trace"]["events"]
        if not events or not torch.equal(b, requested.to(dtype).float()):
            return False
        event = events[0]
        expected = dict(
            before_norm=a.norm().item(),
            after_norm=b.norm().item(),
            delta_norm=(b - a).norm().item(),
            requested_delta_norm=(requested - a).norm().item(),
            rounding_error_norm=(b - requested).norm().item(),
            relative_delta_norm=(b - a).norm().item() / max(a.norm().item(), 1e-30),
            direction_norm=unit.norm().item(),
            projection_before=torch.dot(a, unit / unit.norm()).item(),
            projection_after=torch.dot(b, unit / unit.norm()).item(),
        )
        if not (
            event["phase"] == "prefill"
            and event["layer"] == condition["layer"] == patch["layer"]
            and event["requested_scalar_delta"] == patch["dose"]
            and all(math.isclose(event[k], v, rel_tol=2e-5, abs_tol=1e-6) for k, v in expected.items())
        ):
            return False
    return True


def forward_receipts(model, requests, source_ids, *, layer):
    """Read-only selected-payload forwards at the explicit site; no state mutation."""
    capture_plan(layer)
    _clean(model)
    require(
        1 <= len(requests) <= 32
        and len(source_ids) == len(requests)
        and len(set(source_ids)) == len(source_ids),
        "Require every distinct actual receipt row",
    )
    try:
        return receipts.forward_receipts(model, requests, source_ids, layers=[layer])
    finally:
        _clean(model)


def save_arrays(root, relative, arrays, *, layer, source_ids, requests):
    """Immutable NPZ persistence with actual site/row/prefix bindings, no admission."""
    capture_plan(layer)
    path = (Path(root) / relative).resolve()
    require(
        not Path(relative).is_absolute() and path.is_relative_to(Path(root).resolve()), "Artifact escaped run"
    )
    require(len(source_ids) == len(requests), "Artifact row inventory differs")
    require(arrays and set(arrays) <= {"hidden", "prefill_after"}, "Unknown or empty capture arrays")
    require(
        all(a.shape == (len(requests), 1, 2, a.shape[-1]) and a.dtype == np.float32 for a in arrays.values()),
        "Wrong saved capture layout or dtype",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            layers=[layer],
            positions=POSITIONS,
            trial_ids=source_ids,
            token_positions=[[r["prefix_positions"][p] for p in POSITIONS] for r in requests],
            token_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
        )
    return dict(
        path=str(relative),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        byte_count=path.stat().st_size,
        arrays={k: engineering.array_binding(v) for k, v in arrays.items()},
    )
