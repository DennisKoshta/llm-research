"""Fixed public input and parser checks; no upstream imports, model inference or outcomes."""

import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
import zlib

import pytest

from curiosity_study import pain_profile_design as design
from curiosity_study.artifacts import digest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/pain-axis-positive-control"


@pytest.fixture(scope="module")
def materials():
    return design.load_sources(REFERENCE)


@pytest.fixture(scope="module")
def plan(materials):
    return design.assignment_plan(materials)


def test_named_literal_constants_equal_pinned_source_without_importing_it():
    source = REFERENCE / design.SOURCE_SUBDIR / "src/run_profile.py"
    # Parse only literal assignment values. Never execute/import the upstream module.
    constants = {}
    names = {"SYSTEM_TEMPLATE", "CHOICE_QUESTION", "BUTTON_NAMES", "END_TAGS"}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                constants[name] = ast.literal_eval(node.value)
    assert constants["SYSTEM_TEMPLATE"] == design.SYSTEM_TEMPLATE
    assert constants["CHOICE_QUESTION"] == design.CHOICE_QUESTION
    assert constants["BUTTON_NAMES"] == design.BUTTON_NAMES
    assert tuple(constants["END_TAGS"]) == design.END_TAGS


def test_full_original_context_and_four_source_variants_retained(plan, materials):
    assert len(plan["base_requests"]) == 404
    assert len(plan["assignments"]) == 2020
    assert len(plan["contexts"]) == 101
    assert len({r["context_id"] for r in plan["contexts"]}) == 101
    assert Counter(r["content_group"] for r in plan["contexts"]) == dict(zip(design.GROUPS, (30, 30, 41)))
    by_context = defaultdict(list)
    for row in plan["base_requests"]:
        by_context[row["context_id"]].append(row)
        first = materials["scenarios"][row["content_group"]][row["scenario_index"]][0]
        assert row["context_id"] == hashlib.sha256(first.encode()).hexdigest()
        assert row["messages"][1] == dict(role="user", content=first)
        assert len(row["messages"]) == 3 and row["messages"][2]["role"] == "system"
        names = row["button_names"]
        target = names[row["initial_position"]]
        assert row["target_name"] == target
        assert f'"{target}" is {design.TARGET_DESCRIPTION}.' in row["messages"][2]["content"]
        other = names[1 - row["initial_position"]]
        assert f'"{other}" is {design.OTHER_DESCRIPTION}.' in row["messages"][2]["content"]
        expected_seed = (
            row["seed"] * 1000003 + (zlib.crc32(f"{row['names_key']}|{target}".encode()) & 0x7FFFFFFF)
        ) % (2**62)
        assert row["gen_seed"] == expected_seed
        assert row["seed"] == row["seed_base"] + row["scenario_index"]
    assert all(
        {(r["initial_position"], r["seed_base"]) for r in rows}
        == {(0, 1000), (0, 2000), (1, 1000), (1, 2000)}
        for rows in by_context.values()
    )
    assert plan["scientific_admission"] is False


def test_exact_source_tie_order_and_same_companions_all_arms(plan):
    base = plan["base_requests"]
    assert [(r["message_characters"], r["source_order"]) for r in base] == sorted(
        (r["message_characters"], r["source_order"]) for r in base
    )
    source_order = sorted(base, key=lambda r: r["source_order"])
    expected = [
        (g, side, index, seed)
        for g, n in zip(design.GROUPS, (30, 30, 41))
        for side in (0, 1)
        for index in range(n)
        for seed in (1000, 2000)
    ]
    assert [
        (r["content_group"], r["initial_position"], r["scenario_index"], r["seed_base"]) for r in source_order
    ] == expected
    cohorts = plan["execution_cohorts"]
    assert len(cohorts) == 65
    assert Counter(len(c["trial_ids"]) for c in cohorts) == {32: 60, 20: 5}
    assigned = {r["trial_id"]: r for r in plan["assignments"]}
    assert len(assigned) == 2020
    assert [trial for cohort in cohorts for trial in cohort["trial_ids"]] == list(assigned)
    for block in range(13):
        current = cohorts[block * 5 : block * 5 + 5]
        shift = block % 5
        assert [c["arm"] for c in current] == list(design.ARMS[shift:] + design.ARMS[:shift])
        assert all(c["base_request_ids"] == current[0]["base_request_ids"] for c in current)
        for cohort in current:
            for trial, base_id in zip(cohort["trial_ids"], cohort["base_request_ids"]):
                row = assigned[trial]
                assert row["base_request_id"] == base_id
                assert row["base_request_sha256"] == digest(base[row["base_row"]])
                assert row["gen_seed"] == base[row["base_row"]]["gen_seed"]
                assert row["coefficient"] == (0.0 if row["arm"] == "baseline" else 1.0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["assignments"].pop(),
        lambda p: p["assignments"][0].update(gen_seed=1),
        lambda p: p["execution_cohorts"][0]["trial_ids"].reverse(),
        lambda p: p["base_requests"][0]["messages"][1].update(content="changed"),
        lambda p: p["settings"].update(injection_layer=47),
        lambda p: p["contexts"][0].update(context_id="0" * 64),
    ],
)
def test_plan_reconstruction_rejects_rehashed_tampering(plan, mutation):
    changed = deepcopy(plan)
    mutation(changed)
    changed["assignment_sha256"] = digest(changed["assignments"])
    with pytest.raises(ValueError):
        design.validate_plan(changed)


def test_modified_upstream_bytes_rejected_before_loading(tmp_path):
    for name in design.SOURCE_FILE_SHA256:
        dest = tmp_path / design.SOURCE_SUBDIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REFERENCE / design.SOURCE_SUBDIR / name, dest)
    changed = tmp_path / design.SOURCE_SUBDIR / "src/run_profile.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="Source bytes changed"):
        design.load_sources(tmp_path)


class Tokenizer:
    chat_template = "synthetic source-compatible template"
    eos_token_id = 99
    pad_token_id = None
    unk_token_id = 0

    def __init__(self, reject_mid=False, reject_system=False):
        self.reject_mid = reject_mid
        self.reject_system = reject_system
        self.rendered = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, tools=None):
        assert add_generation_prompt and not tokenize and tools is None
        if self.reject_system and any(m["role"] == "system" for m in messages):
            raise ValueError("system rejected")
        if self.reject_mid and any(i > 0 and m["role"] == "system" for i, m in enumerate(messages)):
            raise ValueError("mid-system rejected")
        rendered = "|".join(m["role"] + ":" + m["content"] for m in messages) + "<assistant>\n"
        self.rendered.append(messages)
        return rendered

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text.strip().lower().startswith("lever"):
            return SimpleNamespace(input_ids=[70, 80])
        if text.endswith("<assistant>\n"):
            return SimpleNamespace(input_ids=[ord(c) + 300 for c in text[:-1]] + [198])
        return SimpleNamespace(input_ids=[ord(text.strip()[0].lower()) + 10])

    def convert_tokens_to_ids(self, value):
        return 98 if value == "<|im_end|>" else 0


@pytest.fixture(scope="module")
def preflight(plan):
    return design.tokenize_plan(plan, Tokenizer(), generation_eos_ids=design.BASE_GENERATION_EOS_IDS)


def test_tokenizer_preflight_final_anchor_readout_overlap_and_eos_union(plan, preflight):
    assert preflight["eos_token_ids"] == [98, 99, 151643, 151645]
    assert preflight["pad_token_id"] == 99
    assert len(preflight["requests"]) == 404
    assert all(
        r["prompt_ids"][-1] == 198 and r["final_prefix_position"] == len(r["prompt_ids"]) - 1
        for r in preflight["requests"]
    )
    assert preflight["name_readouts"]["lever64_lever95"] == dict(prob_token_ids=[[], []], prob_ambiguous=True)
    assert design.validate_preflight(plan, preflight, tokenizer=Tokenizer()) == preflight


@pytest.mark.parametrize("reject_mid,reject_system", [(True, False), (False, True), (True, True)])
def test_original_template_fallbacks_preserve_contents(plan, reject_mid, reject_system):
    tokenizer = Tokenizer(reject_mid=reject_mid, reject_system=reject_system)
    preflight = design.tokenize_plan(plan, tokenizer, generation_eos_ids=design.BASE_GENERATION_EOS_IDS)
    rendered = tokenizer.rendered[-404:]
    for row, messages in zip(plan["base_requests"], rendered):
        text = "\n".join(message["content"] for message in messages)
        assert row["messages"][1]["content"] in text
        assert row["messages"][2]["content"] in text
        if reject_system:
            assert all(m["role"] != "system" for m in messages)
    assert preflight["template_policy"]["mid_system_rejected"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("gen_seed", 0),
        ("final_prefix_position", 0),
        ("final_prefix_token_id", 197),
        ("prob_token_ids", [[999], [998]]),
        ("target_name", "unknown"),
    ],
)
def test_rehashed_preflight_binding_tamper_rejected(plan, preflight, field, value):
    changed = deepcopy(preflight)
    changed["requests"][0][field] = value
    changed["request_sha256"] = digest(changed["requests"])
    with pytest.raises(ValueError):
        design.validate_preflight(plan, changed)


def test_consistently_rehashed_wrong_prefix_requires_actual_tokenizer_to_certify(plan, preflight):
    changed = deepcopy(preflight)
    request = changed["requests"][0]
    request["prompt_ids"][0] += 1
    request["prompt_ids_sha256"] = digest(request["prompt_ids"])
    changed["request_sha256"] = digest(changed["requests"])
    # Structural verification alone explicitly does not certify rendering.
    design.validate_preflight(plan, changed)
    with pytest.raises(ValueError, match="actual tokenization"):
        design.validate_preflight(plan, changed, tokenizer=Tokenizer())


@pytest.mark.parametrize(
    "raw,target,category",
    [
        ("VIOLET.<|im_end|>\n", "violet", "target"),
        ('"yellow"', "violet", "other"),
        ("violet because I choose it", "violet", "target"),
        ("violetness", "violet", "target"),  # Preserve permissive source parsing honestly.
        ("I choose violet", "violet", "malformed"),
        ("", "violet", "malformed"),
        ("<|im_end|>", "violet", "malformed"),
        ("yellow<|im_end|> </s>\n", "yellow", "target"),
    ],
)
def test_source_parser_all_attempt_endpoints_and_prefix_permissiveness(raw, target, category):
    parsed = design.parse_source_answer(raw, ["violet", "yellow"], target)
    assert parsed["category"] == category
    values = design.endpoint(parsed)
    assert sum(values.values()) == 1 and values[category] == 1


def test_unknown_names_and_inconsistent_endpoint_rejected():
    with pytest.raises(ValueError):
        design.parse_source_answer("A", ["A", "B"], "A")
    with pytest.raises(ValueError):
        design.endpoint(dict(category="malformed", valid=True))


def test_generation_config_is_an_explicit_fixed_binding(plan):
    with pytest.raises(ValueError, match="generation EOS"):
        design.tokenize_plan(plan, Tokenizer(), generation_eos_ids=[99])
    metadata = ROOT / "references/pain-axis-assets/base-generation-config.json"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == design.BASE_GENERATION_CONFIG_SHA256
    assert plan["settings"]["generation_config_sha256"] == design.BASE_GENERATION_CONFIG_SHA256


def test_actual_adapter_tokenizer_in_pinned_runtime_only(plan):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.12.1":
        pytest.skip("Actual tokenizer regression requires the separate pinned pain runtime")
    assets = ROOT / "references/pain-axis-assets/metadata/report_dataset_full_1684p_20260831-065321"
    tok = transformers.AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
    checked = design.tokenize_plan(plan, tok, generation_eos_ids=design.BASE_GENERATION_EOS_IDS)
    assert {row["final_prefix_token_id"] for row in checked["requests"]} == {198}
    assert checked["eos_token_ids"] == [128247, 151643, 151645]
    assert checked["pad_token_id"] == 151643
    assert checked["maximum_prompt_tokens"] == 170
    assert min(row["prompt_tokens"] for row in checked["requests"]) == 131
    assert checked["longest_source_cohort"] == 12
    assert checked["name_readouts"]["lever64_lever95"] == dict(prob_token_ids=[[], []], prob_ambiguous=True)
    assert checked["template_policy"] == dict(
        no_system=False, tool_role_rejected=False, mid_system_rejected=False
    )
    design.validate_preflight(plan, checked, tokenizer=tok)
