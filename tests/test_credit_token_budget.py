import pytest
from curiosity_study.credit import build_variant_manifest, CreditTrial, diagnostic_prompt, run
from curiosity_study.model import MockModel


def test_explicit_budget_reaches_inference_without_changing_prompts():
    config = {"candidate_labels": "ABCD", "inference_mode": "think_sampled"}
    short = build_variant_manifest(2, 123, {"model_id": "mock"}, "validation", config)
    long = build_variant_manifest(
        2, 123, {"model_id": "mock"}, "validation", dict(config, max_new_tokens=2048)
    )
    assert short["sampling"]["max_new_tokens"] == 768
    assert long["sampling"]["max_new_tokens"] == 2048
    assert short["planned_behavior_sha256"] != long["planned_behavior_sha256"]
    for a, b in zip(short["diagnostics"], long["diagnostics"]):
        assert a["sampling_seed"] == b["sampling_seed"]
        assert diagnostic_prompt(CreditTrial(a), a) == diagnostic_prompt(CreditTrial(b), b)
        assert b["max_new_tokens"] == 2048 and "max_new_tokens" not in a

    class BudgetMock(MockModel):
        def infer(self, *args, **kwargs):
            assert kwargs["max_new_tokens"] == 2048
            assert kwargs["enable_thinking"]
            return super().infer(*args, **kwargs)

    run(long["diagnostics"][0], BudgetMock())


def test_budget_does_not_admit_unspecified_modes_or_limits():
    for mode, limit in [("think_sampled", 4096), ("no_think_greedy", 2048), ("think_sampled", True)]:
        with pytest.raises(ValueError, match="generation allowance"):
            build_variant_manifest(
                1,
                123,
                {"model_id": "mock"},
                "validation",
                {"candidate_labels": "ABCD", "inference_mode": mode, "max_new_tokens": limit},
            )
