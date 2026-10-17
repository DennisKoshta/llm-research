from curiosity_study.credit import build_variant_manifest, CreditTrial, diagnostic_prompt, run
from curiosity_study.model import MockModel


def test_candidate_and_description_namespaces():
    config = {"candidate_labels": "ABCD", "inference_mode": "no_think_greedy"}
    m = build_variant_manifest(2, 20261008, {"model_id": "mock"}, "validation", config)
    assert m["task_revision"] == "one-credit-v2"
    for a in m["diagnostics"]:
        trial = CreditTrial(a)
        content, labels, expected = diagnostic_prompt(trial, a)
        if a["kind"].startswith("identify"):
            assert labels == "ABCD"
            assert expected == dict(zip("JKLM", "ABCD"))[trial.tasks[a["kind"][-1].upper()].truth]
            assert not any(f"{label}:" in content or f"{label} |" in content for label in "JKLM")
        elif a["kind"].startswith("exclusive") or a["kind"] == "decline":
            assert labels == "1234" and expected in labels
            assert "Action menu:" in content and "A:" in content
        record = run(a, MockModel())
        assert record["valid"]


def test_thinking_action_is_parsed_before_delivery():
    class ThinkingMock(MockModel):
        def infer(self, *args, **kwargs):
            r = super().infer(*args, **kwargs)
            r["raw"] = "Some reasoning.</think>\n" + r["raw"]
            return r

    m = build_variant_manifest(
        1,
        123,
        {"model_id": "mock"},
        "behavior",
        {"candidate_labels": "ABCD", "inference_mode": "think_sampled"},
    )
    r = run(m["assignments"][0], ThinkingMock())
    assert r["valid"] and r["first_service"] in ("X", "Y", "stop")
    assert r["ledger"][0]["raw"] == r["answer"]
