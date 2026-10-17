"""Reviewed first-choice transport of the pinned Pain-axis profile generation loop.

Adapted from valen-research/Pain-axis commit
4d75cd90e206ea962f7a9101e65c85efea56723b, profile/src/run_profile.py.
Upstream MIT notice is retained in references/pain-axis-positive-control/upstream/LICENSE.
This module does not load weights, execute the upstream launcher, choose scientific
conditions, retry OOM, or perform button consequences. Production admission belongs
to the caller; CPU execution exists for explicitly identified synthetic tests.

Upstream MIT License — Copyright (c) 2026 valen-research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from contextlib import ExitStack
import hashlib
import math

import numpy as np
import torch
from transformers import DynamicCache

from .artifacts import digest

STEER_LAYER = 38
MONITOR_LAYER = 61
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.7
TOP_P = 0.95
POLICIES = ("full_prefix", "final_prefix")
SOURCE_SHA256 = "4a6134574b9bb8c0e6e242b46d2e10775a388b838ec61e99c93df1f8c2e92e27"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_binding(value):
    """Hash actual dtype/shape/bytes; BF16 values are never silently recast."""
    value = value.detach().contiguous().cpu()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "byte_sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def array_binding(value):
    value = np.ascontiguousarray(value)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "byte_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def hooks(model):
    return {
        name: {kind: sorted(getattr(module, kind)) for kind in ("_forward_hooks", "_forward_pre_hooks")}
        for name, module in model.named_modules()
        if module._forward_hooks or module._forward_pre_hooks
    }


def source_sample(logits, generators, device):
    """Source inverse-CDF expressions, including draws for already-finished rows.

    The caller supplies every sampled row on every step. No active-row filter is
    permitted here. Returning uniforms adds observation, not another RNG draw.
    """
    B = logits.shape[0]
    sampled_rows = list(range(B))
    samp_mask = torch.ones(B, dtype=torch.bool, device=device)
    nxt = logits.argmax(-1)
    probs = torch.softmax(logits / TEMPERATURE, -1)
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(-1)
    sp = sp * ((cum - sp) <= TOP_P)
    cum2 = sp.cumsum(-1)
    total = cum2[:, -1:]
    u = torch.zeros(B, 1, device=device)
    for i in sampled_rows:
        u[i] = torch.rand(1, device=device, generator=generators[i])
    j = torch.searchsorted(cum2, u * total).clamp(max=sp.shape[-1] - 1)
    nxt = torch.where(samp_mask, si.gather(1, j).squeeze(1), nxt)
    return nxt, u


class _Evidence:
    """Independent before-hook/next-block-input observations of actual arithmetic."""

    def __init__(self, items, direction, coefficients, policy, mode, steering_enabled):
        self.items, self.direction = items, direction
        self.coefficients, self.policy, self.mode = coefficients, policy, mode
        self.steering_enabled = steering_enabled
        self.frames, self.arrays = [], {}
        self.current = None
        self.before_tensor = None
        self.attempted = 0

    def begin(self, kwargs, steering_mask):
        require(self.current is None, "Nested or unfinished forward")
        self.attempted += 1
        cache = kwargs["past_key_values"]
        self.current = {
            "forward_index": len(self.frames),
            "phase": "prefill" if not self.frames else "decode",
            "input_ids": kwargs["input_ids"].detach().cpu().tolist(),
            "attention_mask": kwargs["attention_mask"].detach().cpu().tolist(),
            "position_ids": kwargs["position_ids"].detach().cpu().tolist(),
            "steering_mask": None if steering_mask is None else steering_mask.detach().cpu().tolist(),
            "cache_length_before": int(cache.get_seq_length()),
            "steering_hook_calls": 0,
            "read_only_projection_calls": 0,
            "before_observer_calls": 0,
            "after_observer_calls": 0,
            "monitor_hook_calls": 0,
        }

    def before(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        self.current["before_observer_calls"] += 1
        require(value.dtype == torch.bfloat16, "Require actual BF16 residual stream")
        require(value.ndim == 3 and value.shape[0] == len(self.items), "Wrong residual rows")
        require(value.shape[-1] == self.direction.numel(), "Wrong residual width")
        # Clone before the intervention; the next block exposes actual returned data.
        self.before_tensor = value.detach().clone()

    def after(self, module, args):
        after = args[0].detach()
        before = self.before_tensor
        require(before is not None and before.shape == after.shape, "Missing independent before observation")
        self.current["after_observer_calls"] += 1
        require(after.dtype == before.dtype == torch.bfloat16, "Residual dtype changed")
        is_prefill = self.current["phase"] == "prefill"
        width = before.shape[1]
        rows = []
        for row, item in enumerate(self.items):
            row_coefficient = self.coefficients[row]
            active = torch.ones(width, dtype=torch.bool, device=before.device)
            if is_prefill:
                active[:] = False
                offset = width - len(item["prompt_ids"])
                if self.policy == "full_prefix":
                    active[offset:] = True
                else:
                    active[-1] = True
            if row_coefficient == 0:
                active[:] = False
            # Independent expected arithmetic uses a separately reconstructed mask.
            coefficient = torch.tensor(row_coefficient, dtype=torch.bfloat16, device=before.device)
            if is_prefill and width > 1:
                delta = active.to(torch.bfloat16)[:, None] * coefficient * self.direction[None, :]
            else:
                delta = coefficient * self.direction[None, :]
            # The source bypasses addition only when the ENTIRE cohort is zero.
            # Off rows of a mixed cohort still receive a BF16 zero addition.
            expected = before[row] if not any(self.coefficients) else before[row] + delta
            finite = bool(torch.isfinite(before[row]).all() and torch.isfinite(after[row]).all())
            exact = torch.equal(after[row].contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
            error = float((after[row].float() - expected.float()).abs().max()) if finite else None
            actual = after[row].float() - before[row].float()
            norms = actual[active].norm(dim=-1)
            padding = torch.zeros(width, dtype=torch.bool, device=before.device)
            if is_prefill:
                padding[: width - len(item["prompt_ids"])] = True
            untouched = ~active if row_coefficient != 0 else torch.ones_like(active)
            same_untouched = torch.equal(
                after[row][untouched].contiguous().view(torch.uint8),
                before[row][untouched].contiguous().view(torch.uint8),
            )
            same_padding = torch.equal(
                after[row][padding].contiguous().view(torch.uint8),
                before[row][padding].contiguous().view(torch.uint8),
            )
            intended = (coefficient * self.direction).float()
            intended_norm = float(intended.norm())
            projections = actual[active] @ self.direction.float()
            rows.append({
                "trial_id": item["trial_id"],
                "coefficient": row_coefficient,
                "finite": finite,
                "exact_bf16_arithmetic": exact,
                "max_abs_error": error,
                "active_positions": int(active.sum()),
                "unmodified_positions": int(untouched.sum()),
                "padding_positions": int(padding.sum()),
                "unmodified_positions_identical": same_untouched,
                "padding_identical": same_padding,
                "requested_delta_norm_bf16": intended_norm,
                "achieved_delta_norm_min": (float(norms.min()) if norms.numel() else 0.0) if finite else None,
                "achieved_delta_norm_max": (float(norms.max()) if norms.numel() else 0.0) if finite else None,
                "achieved_direction_dot_min": (float(projections.min()) if projections.numel() else 0.0) if finite else None,
                "all_requested_positions_engaged": intended_norm == 0 or bool((norms > 0).all()),
                "before_sha256": digest(tensor_binding(before[row])),
                "after_sha256": digest(tensor_binding(after[row])),
            })
            save_row = row in ({0, len(self.items) - 1} if self.mode == "compatibility" else {0})
            if save_row:
                positions = list(range(width)) if self.mode == "compatibility" else sorted({
                    width - 1,
                    max(0, width - len(item["prompt_ids"])) if is_prefill else 0,
                })
                prefix = f"f{self.current['forward_index']:02d}_r{row:02d}"
                for label, value in (("before", before), ("after", after)):
                    key = f"{prefix}_{label}"
                    array = value[row, positions].float().cpu().numpy().copy()
                    self.arrays[key] = array
                    rows[-1][f"saved_{label}"] = {"key": key, "positions": positions, **array_binding(array)}
        self.current["geometry"] = rows
        self.current["activation_dtypes"] = {"before": str(before.dtype), "after": str(after.dtype)}
        self.before_tensor = None

    def finish(self, output):
        logits = output.logits[:, -1, :].float()
        self.current.update(
            cache_length_after=int(output.past_key_values.get_seq_length()),
            vocabulary_size=logits.shape[-1],
            full_fp32_logits_sha256=[digest(tensor_binding(value)) for value in logits],
            finite_logits=bool(torch.isfinite(logits).all()),
        )
        require(self.current["steering_hook_calls"] == int(self.steering_enabled), "Wrong steering-hook disposition")
        require(self.current["read_only_projection_calls"] == int(not self.steering_enabled), "Wrong read-only projection disposition")
        for name in ("before_observer_calls", "after_observer_calls", "monitor_hook_calls"):
            require(self.current[name] == 1, f"Require exactly one {name} per forward")
        self.frames.append(self.current)
        self.current = None
        require(bool(torch.isfinite(logits).all()), "Nonfinite actual full-vocabulary logits")
        return logits


def _validate(model, layers, items, direction, pain_unit, coefficient, policy, eos_ids, pad_id, mode):
    require(not model.training, "Caller must set inference/eval mode")
    require(len(layers) > MONITOR_LAYER, "Require original injection38 and monitor61 blocks")
    require(type(coefficient) in (int, float) and math.isfinite(coefficient) and coefficient in (0, 1), "Coefficient must be0 or1")
    require(policy in POLICIES and mode in ("benchmark", "compatibility"), "Unknown mask/evidence policy")
    require(torch.get_default_dtype() == torch.float32, "Source sampler assumes default FP32 allocations")
    require(isinstance(direction, torch.Tensor) and direction.ndim == 1 and direction.dtype == torch.bfloat16,
            "Require already pinned BF16 direction")
    require(isinstance(pain_unit, torch.Tensor) and pain_unit.shape == direction.shape and pain_unit.dtype == torch.float32,
            "Require original FP32 pain unit even for fear")
    require(direction.device == pain_unit.device and bool(torch.isfinite(direction).all()) and bool(torch.isfinite(pain_unit).all()),
            "Nonfinite or incompatible direction tensors")
    require(1 <= len(items) <= 32 and len({row["trial_id"] for row in items}) == len(items), "Require1..32 unique physical rows")
    require(type(pad_id) is int and pad_id >= 0 and all(type(x) is int and x >= 0 for x in eos_ids), "Malformed terminal IDs")
    for row in items:
        require(isinstance(row["trial_id"], str) and row["trial_id"], "Missing trial ID")
        require(isinstance(row["prompt_ids"], list) and len(row["prompt_ids"]) > 1
                and all(type(x) is int and x >= 0 for x in row["prompt_ids"]), "Malformed prefix")
        require(type(row["gen_seed"]) is int and 0 <= row["gen_seed"] < 2**62, "Malformed generator seed")
        require(len(row["button_names"]) == 2 and len(set(row["button_names"])) == 2
                and row["target_name"] in row["button_names"], "Malformed parse inputs")
        require(len(row["prob_token_ids"]) == 2 and all(
            isinstance(values, list) and values == sorted(set(values))
            and all(type(x) is int and x >= 0 for x in values) for values in row["prob_token_ids"]
        ), "Malformed source first-token sets")
        require(not set(row["prob_token_ids"][0]) & set(row["prob_token_ids"][1]), "Shared name-token IDs must be removed")


@torch.inference_mode()
def generate_segment(
    model, tokenizer, layers, *, items, direction_bf16, pain_unit_float32,
    coefficient, prefill_policy, eos_token_ids, pad_token_id, evidence_mode="benchmark",
    steering_enabled=True, inject_failure_at=None, one_row_only=None,
):
    """One terminal first-choice cohort; errors retain evidence and never retry.

    Returns records/evidence/arrays. The caller must persist all assignments even
    if records is empty after a terminal exception. No GPU admission is implied.
    """
    eos_ids = set(eos_token_ids)
    _validate(model, layers, items, direction_bf16, pain_unit_float32, coefficient,
              prefill_policy, eos_ids, pad_token_id, evidence_mode)
    require(type(steering_enabled) is bool and (steering_enabled or coefficient == 0),
            "Unhooked execution requires coefficient0")
    require(inject_failure_at is None or (evidence_mode == "compatibility"
            and inject_failure_at in ("after_prefill", "after_first_decode")),
            "Controlled faults are explicit compatibility-only checks")
    require(one_row_only is None or (evidence_mode == "compatibility" and coefficient == 1
            and type(one_row_only) is int and one_row_only == 0),
            "Single-row perturbation is an explicit compatibility-only check")
    items = [dict(row) for row in items]
    device = direction_bf16.device
    UNIT = pain_unit_float32
    direction = direction_bf16
    B = len(items)
    coefficients = [coefficient if one_row_only is None or row == one_row_only else 0 for row in range(B)]
    evidence = _Evidence(items, direction, coefficients, prefill_policy, evidence_mode, steering_enabled)
    initial_hooks = hooks(model)
    G = {"coeff": None, "dirs": None, "mask": None, "proj": None, "mon": None}

    def steer_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        evidence.current["steering_hook_calls"] += 1
        G["proj"] = (hs[:, -1, :].float() @ UNIT).detach()
        c = G["coeff"]
        if c is None or not bool((c != 0).any()):
            return output
        if hs.shape[1] > 1 and G["mask"] is not None:
            add = G["mask"].to(hs.dtype)[:, :, None] * c[:, None, None].to(hs.dtype) * G["dirs"][:, None, :].to(hs.dtype)
        else:
            add = c[:, None, None].to(hs.dtype) * G["dirs"][:, None, :].to(hs.dtype)
        hs = hs + add
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    def monitor_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        evidence.current["monitor_hook_calls"] += 1
        G["mon"] = (hs[:, -1, :].float() @ UNIT).detach()
        return output

    def read_only_projection(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        evidence.current["read_only_projection_calls"] += 1
        G["proj"] = (hs[:, -1, :].float() @ UNIT).detach()
        return output

    def forward(**kw):
        evidence.begin(kw, G["mask"])
        out = model(**kw, logits_to_keep=1)
        evidence.finish(out)
        evidence.frames[-1].update(
            prepatch_pain_projection=G["proj"].cpu().tolist(),
            post_layer61_pain_projection=G["mon"].cpu().tolist(),
        )
        if (inject_failure_at == "after_prefill" and len(evidence.frames) == 1) or (
            inject_failure_at == "after_first_decode" and len(evidence.frames) == 2
        ):
            raise RuntimeError(f"controlled compatibility fault: {inject_failure_at}")
        return out

    result, error = [], None
    try:
        with ExitStack() as cleanup:
            for register, callback in (
                (layers[STEER_LAYER].register_forward_hook, evidence.before),
                (layers[STEER_LAYER].register_forward_hook, steer_hook if steering_enabled else read_only_projection),
                (layers[STEER_LAYER + 1].register_forward_pre_hook, evidence.after),
                (layers[MONITOR_LAYER].register_forward_hook, monitor_hook),
            ):
                cleanup.callback(register(callback).remove)
            L = max(len(it["prompt_ids"]) for it in items)
            ids = torch.full((B, L), int(pad_token_id), dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            smask = torch.zeros((B, L), dtype=torch.float32)
            for i, it in enumerate(items):
                p = it["prompt_ids"]
                off = L - len(p)
                ids[i, off:] = torch.tensor(p)
                mask[i, off:] = 1
                # The only experimental difference: these already-supported ranges.
                ranges = [(0, len(p))] if prefill_policy == "full_prefix" else [(len(p) - 1, len(p))]
                if coefficients[i] == 0:
                    ranges = []
                for a, b in ranges:
                    a2, b2 = max(0, min(a, len(p))), max(0, min(b, len(p)))
                    if b2 > a2:
                        smask[i, off + a2:off + b2] = 1.0
            ids, mask, smask = ids.to(device), mask.to(device), smask.to(device)
            G["coeff"] = torch.tensor(coefficients, device=device, dtype=torch.float32)
            G["dirs"] = torch.stack([direction] * B, 0)
            G["coeff_prefill"] = torch.tensor(coefficients, device=device, dtype=torch.float32)
            G["mask"] = smask
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            cache = DynamicCache()
            generators = [torch.Generator(device=device).manual_seed(it["gen_seed"]) for it in items]
            saved = G["coeff"]
            G["coeff"] = G["coeff_prefill"]
            out = forward(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
            require(bool((mask[:, -1] == 1).all()), "Prefill last column must be attended")
            prefill_proj_monitor = G["mon"].clone()
            require(bool(torch.isfinite(prefill_proj_monitor).all()), "Nonfinite source monitor")
            prefill_proj_monitor_l = prefill_proj_monitor.tolist()
            G["coeff"] = saved
            G["mask"] = None
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            p0 = torch.softmax(logits, -1)
            p_x, p_y = [], []
            for i, it in enumerate(items):
                tx, ty = [torch.tensor(values, dtype=torch.long, device=device) for values in it["prob_token_ids"]]
                p_x.append(float(p0[i, tx].sum()) if tx.numel() else 0.0)
                p_y.append(float(p0[i, ty].sum()) if ty.numel() else 0.0)
            eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
            max_new_t = torch.tensor([MAX_NEW_TOKENS] * B, device=device)
            hard_cap = MAX_NEW_TOKENS
            alive = torch.ones(B, dtype=torch.bool, device=device)
            n_out = torch.zeros(B, dtype=torch.long, device=device)
            pad_t = torch.full((B,), int(pad_token_id), dtype=torch.long, device=device)
            steps_tokens = []
            proj_steps = [G["proj"].clone()]
            mon_steps = [G["mon"].clone()]
            stat_w = [torch.ones(B, dtype=torch.float32, device=device)]
            cur_pos = mask.sum(-1)
            attn = mask
            for step in range(hard_cap):
                rng_before = [digest(tensor_binding(gen.get_state())) for gen in generators]
                nxt, uniforms = source_sample(logits, generators, device)
                frame = evidence.frames[-1]
                frame["sampling"] = {
                    "step": step, "alive_before": alive.cpu().tolist(),
                    "rng_state_before_sha256": rng_before,
                    "rng_state_after_sha256": [digest(tensor_binding(gen.get_state())) for gen in generators],
                    "uniforms": uniforms.cpu().flatten().tolist(),
                    "sampled_before_alive_mask": nxt.cpu().tolist(),
                    "uniform_draws_per_row": [1] * B,
                }
                nxt = torch.where(alive, nxt, pad_t)
                n_out = n_out + alive.long()
                steps_tokens.append(nxt)
                alive = alive & ~torch.isin(nxt, eos_t) & (n_out < max_new_t)
                frame["sampling"].update(selected_token_ids=nxt.cpu().tolist(), alive_after=alive.cpu().tolist())
                if not bool(alive.any()):
                    break
                attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], 1)
                out = forward(input_ids=nxt[:, None], attention_mask=attn, position_ids=cur_pos[:, None],
                              past_key_values=cache, use_cache=True)
                cur_pos = cur_pos + 1
                cache = out.past_key_values
                logits = out.logits[:, -1, :].float()
                proj_steps.append(G["proj"].clone())
                mon_steps.append(G["mon"].clone())
                stat_w.append(alive.to(torch.float32))
            toks = torch.stack(steps_tokens, 0).T.tolist() if steps_tokens else [[] for _ in items]
            n_out_l = n_out.tolist()
            W = torch.stack(stat_w, 0)
            pm = (torch.stack(proj_steps, 0) * W).sum(0) / W.sum(0)
            mm = (torch.stack(mon_steps, 0) * W).sum(0) / W.sum(0)
            pm_l, mm_l, nf_l = pm.tolist(), mm.tolist(), W.sum(0).tolist()
            del cache
            for i, it in enumerate(items):
                tokens = toks[i][:n_out_l[i]]
                result.append({
                    "trial_id": it["trial_id"],
                    "text": tokenizer.decode(tokens, skip_special_tokens=False),
                    "p_x": p_x[i], "p_y": p_y[i],
                    "answer_token_ids": tokens, "prompt_token_ids": list(it["prompt_ids"]),
                    "prompt_tokens": len(it["prompt_ids"]), "padded_prompt_tokens": L,
                    "prefill_attended": True, "prefill_position_id": len(it["prompt_ids"]) - 1,
                    "prefill_generated_tokens": 0, "prefill_proj_monitor": prefill_proj_monitor_l[i],
                    "mean_proj": pm_l[i], "mean_proj_monitor": mm_l[i], "n_fwd": int(nf_l[i]),
                    "button_names": list(it["button_names"]), "target_name": it["target_name"],
                    "ended_with_eos": bool(tokens and tokens[-1] in eos_ids),
                    "cap_hit": bool(len(tokens) == MAX_NEW_TOKENS and tokens[-1] not in eos_ids),
                })
    except Exception as caught:
        result = []
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        final_hooks = hooks(model)
        clean = final_hooks == initial_hooks
        evidence.before_tensor = None
    rows = [row for frame in evidence.frames for row in frame.get("geometry", [])]
    checks = {
        "hooks_clean": clean,
        "all_observed_geometry_finite": bool(rows) and all(row["finite"] for row in rows),
        "all_observed_arithmetic_exact": bool(rows) and all(row["exact_bf16_arithmetic"] for row in rows),
        "all_unmodified_positions_identical": bool(rows) and all(row["unmodified_positions_identical"] for row in rows),
        "all_padding_identical": bool(rows) and all(row["padding_identical"] for row in rows),
        "all_observed_logits_finite": bool(evidence.frames) and all(frame["finite_logits"] for frame in evidence.frames),
        "all_requested_positions_engaged": bool(rows) and all(row["all_requested_positions_engaged"] for row in rows),
    }
    report = {
        "schema_version": 1, "source_entrypoint_sha256": SOURCE_SHA256,
        "device_type": device.type, "cpu_synthetic_execution": device.type == "cpu",
        "steer_layer": STEER_LAYER, "monitor_layer": MONITOR_LAYER,
        "coefficient": coefficient, "prefill_policy": prefill_policy, "evidence_mode": evidence_mode,
        "steering_enabled": steering_enabled, "inject_failure_at": inject_failure_at,
        "one_row_only": one_row_only, "applied_coefficients": coefficients,
        "direction": tensor_binding(direction), "pain_unit": tensor_binding(UNIT),
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_ids": sorted(eos_ids), "pad_token_id": pad_token_id,
        "items_sha256": digest(items), "trial_ids": [it["trial_id"] for it in items],
        "attempted_forwards": evidence.attempted, "forwards": evidence.frames,
        "hooks_before": initial_hooks, "hooks_after": final_hooks,
        "incomplete_forward": evidence.current, "error": error,
        "complete": error is None and len(result) == B,
        "checks": checks,
        "integrity_passed": error is None and len(result) == B and all(checks.values()),
        "arrays": {name: array_binding(value) for name, value in evidence.arrays.items()},
        "scientific_admission": False,
    }
    return {"records": result, "evidence": report, "arrays": evidence.arrays}
