"""Real-model checks for sham equivalence and hook cleanup, including exceptional exits."""

from .model import Intervention


def verify_model(model):
    torch = model.torch
    messages = [{"role": "user", "content": "Choose one letter only: A, B, C, or D."}]
    layer = len(model.layers) // 2
    baseline = model.infer(messages, 71, greedy=True)
    sham = model.infer(messages, 71, greedy=True, intervention=Intervention(layer))
    direction = [0.0] * model.model.config.hidden_size
    direction[0] = 1.0
    changed = model.infer(messages, 71, greedy=True, intervention=Intervention(layer, 1, direction))
    fresh = model.infer(messages, 71, greedy=True)
    try:
        model.infer(messages, 71, intervention=Intervention(layer, 1, [1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed direction was not rejected")
    after_error = model.infer(messages, 71, greedy=True)
    checks = {
        "sham_tokens_equal": baseline["generated_token_ids"] == sham["generated_token_ids"],
        "sham_probabilities_equal": baseline["label_probabilities_temperature_1"]
        == sham["label_probabilities_temperature_1"],
        "zero_dose_capture_equal": sham["capture"]["before"] == sham["capture"]["after"],
        "hook_called_once": sham["capture"]["calls"] == changed["capture"]["calls"] == 1,
        "nonzero_changes_activation": changed["capture"]["before"] != changed["capture"]["after"],
        "subsequent_branch_equal": baseline["label_probabilities_temperature_1"]
        == fresh["label_probabilities_temperature_1"],
        "exception_cleanup_equal": baseline["label_probabilities_temperature_1"]
        == after_error["label_probabilities_temperature_1"],
        "no_hooks_remaining": model.active_hooks == 0,
        "identical_prompt_tokens": baseline["input_ids"] == sham["input_ids"] == changed["input_ids"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "model": model.metadata(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if model.device == "cuda" else None,
        "records": {
            "baseline": baseline,
            "sham": sham,
            "nonzero": changed,
            "fresh": fresh,
            "after_error": after_error,
        },
    }
