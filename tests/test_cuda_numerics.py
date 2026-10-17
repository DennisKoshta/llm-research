"""The engineering arithmetic change is explicit and leaves unrelated flags alone."""

import torch
from curiosity_study.model import configure_cuda_numerics


def test_bf16_reduction_policy_is_applied_and_recorded():
    backend = torch.backends.cuda.matmul
    original = backend.allow_bf16_reduced_precision_reduction
    other = (backend.allow_fp16_reduced_precision_reduction, backend.allow_tf32)
    try:
        backend.allow_bf16_reduced_precision_reduction = True
        policy = configure_cuda_numerics(torch)
        assert backend.allow_bf16_reduced_precision_reduction is False
        assert policy["allow_bf16_reduced_precision_reduction"] is False
        assert (policy["allow_fp16_reduced_precision_reduction"], policy["allow_tf32"]) == other
        assert (backend.allow_fp16_reduced_precision_reduction, backend.allow_tf32) == other
    finally:
        backend.allow_bf16_reduced_precision_reduction = original
