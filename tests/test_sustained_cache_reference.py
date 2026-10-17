"""Independent uncached reference for the opt-in sustained intervention timeline.

Full-sequence recomputation is deliberately different from the cached decode
implementation. Float32 tolerance here checks mathematical cache semantics;
production BF16 sham identity is covered separately and remains exact.
"""

import numpy as np
import pytest
import torch

from curiosity_study.artifacts import digest
from curiosity_study.state_backend import StateBackend
from test_state_backend import tiny_backend


@pytest.mark.parametrize("schedule", ["prefill_only", "record_prefill_plus_decode"])
def test_cached_logits_match_uncached_full_sequence_with_all_historical_patches(schedule):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Eager Qwen attention casts softmax to FP32, so the FP64 minimum
        # masking value yields NaNs on all-masked padding queries. Use its
        # supported FP32 path; production BF16 sham checks remain separate.
        model = tiny_backend(dtype=torch.float32)
        model.model.generation_config.eos_token_id = None
        backend = StateBackend(model)
        requests = []
        for row, tokens in enumerate(([7, 11, 19, 21, 6], [8, 11, 29, 11, 6, 4, 9])):
            requests.append(
                dict(
                    token_prefix=list(tokens),
                    token_prefix_sha256=digest(list(tokens)),
                    prefix_positions=dict(record=1, final=len(tokens) - 1),
                    labels=["Yes", "No"],
                    seed=915 + row,
                )
            )
        direction = np.zeros(16)
        direction[3] = 1
        patches = [
            dict(layer=1, position="record", mode="add", direction=direction, dose=dose)
            for dose in (0.125, -0.25)
        ]
        calls = []

        def observe(module, args, kwargs, output):
            calls.append(
                dict(
                    input_ids=kwargs["input_ids"].detach().clone(),
                    attention_mask=kwargs["attention_mask"].detach().clone(),
                    logits=output.logits[:, -1].detach().clone(),
                )
            )

        observer = model.model.register_forward_hook(observe, with_kwargs=True)
        try:
            measured, _ = backend.forward(
                requests, interventions=patches, max_new_tokens=4, greedy=True, intervention_schedule=schedule
            )
        finally:
            observer.remove()
        assert len(calls) == 4  # Prefill then three processed tokens; cap token four is not forwarded.
        assert model.active_hooks == 0 and all(row["hooks_remaining"] == 0 for row in measured)
        _, _, inputs, targets = backend.prepare(requests)
        width = inputs["input_ids"].shape[1]
        prefix = inputs["input_ids"].clone()
        include_decodes = schedule == "record_prefill_plus_decode"
        for step, call in enumerate(calls):
            if step:
                assert call["input_ids"].shape == (2, 1)
                prefix = torch.cat([prefix, call["input_ids"]], dim=-1)
            mask = call["attention_mask"]
            assert prefix.shape == mask.shape == (2, width + step)
            positions = (mask.cumsum(-1) - 1).clamp(min=0)

            def reference_patch(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                for row, patch in enumerate(patches):
                    indices = [targets["record"][row]]
                    if include_decodes:
                        indices += list(range(width, width + step))
                    # Match the documented FP32 arithmetic + model-dtype storage,
                    # while recomputing all historical states without any KV cache.
                    for index in indices:
                        before = tensor[row, index].float()
                        after = before + patch["dose"] * torch.as_tensor(direction, dtype=torch.float32)
                        changed[row, index] = after.to(tensor.dtype)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = model.layers[1].register_forward_hook(reference_patch)
            try:
                with torch.inference_mode():
                    reference = model.model(
                        input_ids=prefix,
                        attention_mask=mask,
                        position_ids=positions,
                        use_cache=False,
                        logits_to_keep=1,
                    )
            finally:
                handle.remove()
            # Cached and full-sequence kernels differ in matrix geometry. This
            # tolerance tests numerical equivalence, not production sham identity.
            torch.testing.assert_close(
                call["logits"].float(), reference.logits[:, -1].float(), rtol=1e-6, atol=1e-7
            )
            expected_token = call["logits"].float().argmax(-1).tolist()
            assert expected_token == [row["completion"]["generated_token_ids"][step] for row in measured]
        if include_decodes:
            # Ensure the cached/full agreement above could detect an accidentally
            # omitted decode hook, rather than passing on a numerically inert axis.
            include_decodes = False
            handle = model.layers[1].register_forward_hook(reference_patch)
            try:
                with torch.inference_mode():
                    record_only = model.model(
                        input_ids=prefix,
                        attention_mask=mask,
                        position_ids=positions,
                        use_cache=False,
                        logits_to_keep=1,
                    )
            finally:
                handle.remove()
            assert (record_only.logits[:, -1].float() - calls[-1]["logits"].float()).abs().max() > 1e-5
        assert not model.model._forward_hooks and not model.layers[1]._forward_hooks
    finally:
        torch.set_num_threads(previous)
