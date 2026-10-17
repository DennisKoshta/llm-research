"""Real small-model parity and BF16 intervention checks, with no scientific claim."""

from pathlib import Path
import numpy as np
from curiosity_study.artifacts import read, digest, write_once
from curiosity_study.batched import BatchedHFModel
from curiosity_study.information_state import assignments
from curiosity_study.thinking_information import inference_request
from curiosity_study.state_backend import StateBackend
from curiosity_study.intervention_integrity import audit_intervention
from curiosity_study.replay_information import engine_hashes

ROOT = Path(__file__).resolve().parents[1]


def main():
    revision = read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-1.7B"]
    model = BatchedHFModel("Qwen/Qwen3-1.7B", revision)
    chosen = assignments(items=1, seed=19933, kinds=("new",))[:2]
    requests = [{**inference_request(a), "greedy": True, "max_new_tokens": 64} for a in chosen]
    original = model.infer_batch(requests, [a["trial_id"] for a in chosen])
    prefixes = []
    for request, row in zip(requests, original):
        text = row["rendered_prompt"]
        offset = text.index("End of record.") + len("End of record.") - 1
        encoded = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        record = next(i for i, (s, e) in enumerate(encoded["offset_mapping"]) if s <= offset < e)
        tokens = row["unpadded_input_ids"]
        prefixes.append(
            dict(
                token_prefix=tokens,
                token_prefix_sha256=digest(tokens),
                prefix_positions=dict(record=record, final=len(tokens) - 1),
                labels=request["labels"],
                seed=request["seed"],
            )
        )
    backend = StateBackend(model)
    direct, _ = backend.forward(prefixes, max_new_tokens=64, greedy=True)
    layer = len(model.layers) // 2
    repeated = [prefixes[0], prefixes[0], prefixes[1], prefixes[1]]
    baseline, captures = backend.forward(repeated, layers=[layer])
    rng = np.random.default_rng(19934)
    information, control = rng.normal(size=(2, model.model.config.hidden_size))
    information /= np.linalg.norm(information)
    control /= np.linalg.norm(control)
    target = float(np.dot(captures[0, 0, 0], information)) + 2.0
    patches = [
        dict(layer=layer, position="record", mode="project", direction=information, target=target),
        dict(
            layer=layer,
            position="record",
            mode="matched_add",
            direction=control,
            reference_direction=information,
            target=target,
        ),
        None,
        None,
    ]
    changed, _ = backend.forward(repeated, interventions=patches)
    sham, _ = backend.forward(
        repeated,
        interventions=[
            dict(layer=layer, position="record", mode="add", direction=information, dose=0.0)
            for _ in repeated
        ],
    )
    geometry = [
        audit_intervention(
            dict(
                trial_id=f"engineering-{i}", patch_mode=patches[i]["mode"], projection_target=target, dose=2.0
            ),
            changed[i]["intervention"],
        )
        for i in range(2)
    ]
    checks = dict(
        prompt_only_tokens_match=all(
            a["generated_token_ids"] == b["completion"]["generated_token_ids"]
            for a, b in zip(original, direct)
        ),
        prompt_only_probabilities_match=all(
            a["label_probabilities_temperature_1"] == b["label_probabilities_temperature_1"]
            for a, b in zip(original, direct)
        ),
        zero_sham_logits_match=all(a["label_logits"] == b["label_logits"] for a, b in zip(baseline, sham)),
        other_rows_unchanged=all(baseline[i]["label_logits"] == changed[i]["label_logits"] for i in (2, 3)),
        requested_control_scalars_identical=changed[0]["intervention"]["requested_scalar_delta"]
        == changed[1]["intervention"]["requested_scalar_delta"],
        intervention_geometry_verified=all(g["implementation_verified"] for g in geometry),
        hooks_removed=model.active_hooks == 0,
    )
    result = dict(
        engineering_only=True,
        model=model.metadata(),
        engine_sha256=engine_hashes(),
        checks=checks,
        passed=all(checks.values()),
        original=original,
        direct=direct,
        interventions=changed,
        geometry=geometry,
        interpretation="Pinned Qwen3-1.7B engineering check; does not establish parity or scientific behavior for Qwen3-32B.",
    )
    write_once(ROOT / "reports/local-intervention-engine-v1.json", result)
    print(dict(checks=checks, passed=result["passed"]), flush=True)
    if not result["passed"]:
        raise SystemExit("Local engineering parity failed")


if __name__ == "__main__":
    main()
