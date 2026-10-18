"""Preparation-only token alignment for the specified Layer48 pre_think probe.

This renders no completion and extracts no activation. It locates the token
immediately BEFORE the chat template's generation-side <think>, rather than
mistaking the final token after that marker for the requested timestamp.
"""
from hashlib import sha256
import json


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def locate(messages, tokenizer, cot_mode, *, expected_prompt=None, expected_token_ids=None):
    require(cot_mode in {'cot_on', 'cot_off'}, 'Explicit registered reasoning mode required')
    require(isinstance(messages, list) and messages, 'Full immutable input messages required')
    options = {'tokenize': False, 'enable_thinking': cot_mode == 'cot_on'}
    prior = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **options)
    require(full.startswith(prior), 'Generation template is not an exact extension of input messages')
    suffix = full[len(prior):]
    require(suffix.count('<think>') == 1, 'Require one generation-side think marker in the pinned primary template')
    # A quoted marker in input messages cannot change this generation boundary.
    boundary = len(prior)+suffix.index('<think>')
    prefix = full[:boundary]
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    require(len(ids) == len(offsets) and len(ids) > 1, 'Exact fast-tokenizer offsets required')
    positions = [i for i, (start, end) in enumerate(offsets) if start == boundary and end > start]
    require(len(positions) == 1 and positions[0] > 0, 'Think marker must begin at an exact noninitial token boundary')
    opening_index = positions[0]
    require(offsets[opening_index-1][1] == boundary, 'No gap or straddling token before think marker')
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    require(prefix_ids == ids[:opening_index], 'Pre-think prefix does not preserve actual evaluation token IDs')
    if expected_prompt is not None:
        require(full == expected_prompt, 'Prompt differs from immutable evaluation input')
    if expected_token_ids is not None:
        require(ids == expected_token_ids, 'Token IDs differ from immutable evaluation input')
    return {'schema': 'pre_action_token_alignment_v1', 'timing': 'pre_think',
            'layer_index': 48, 'cot_mode': cot_mode, 'messages_sha256': digest(messages),
            'full_prompt_sha256': sha256(full.encode()).hexdigest(), 'full_prompt_token_ids_sha256': digest(ids),
            'target_token_index': opening_index-1, 'target_token_id': ids[opening_index-1],
            'generation_think_token_index': opening_index, 'prefix_token_ids': prefix_ids,
            'prefix_token_ids_sha256': digest(prefix_ids), 'available_prompt_text': prefix,
            'generated_text_used': False, 'activation_extracted': False,
            'phase3_authorized': False}
