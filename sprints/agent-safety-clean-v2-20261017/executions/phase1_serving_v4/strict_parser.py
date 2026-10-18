"""Parse only complete post-reasoning JSON calls; never infer actions from prose."""
import json
import re

BLOCK = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Non-finite JSON number')


def parse_output(raw_output, cot_mode, declared_names):
    if cot_mode not in {'cot_on', 'cot_off'}:
        raise ValueError('Unknown reasoning mode')
    row = {'raw_output': raw_output, 'think_text': '', 'post_think_text': '',
           'think_closed': False, 'parsed_tool_name': None, 'parsed_tool_args': None,
           'parse_status': None}
    if cot_mode == 'cot_on' and '</think>' not in raw_output:
        row.update(think_text=raw_output, parse_status='TRUNCATED_IN_THINK')
        return row
    if '</think>' in raw_output:
        reasoning, post = raw_output.split('</think>', 1)
        row.update(think_text=reasoning, post_think_text=post, think_closed=True)
    else:
        post = raw_output
        row.update(post_think_text=post)
    blocks = list(BLOCK.finditer(post))
    if len(blocks) != 1:
        row['parse_status'] = 'NO_TOOL_CALL' if not blocks else 'MULTIPLE_TOOL_CALLS'
        return row
    remaining = post[:blocks[0].start()] + post[blocks[0].end():]
    if '<tool_call>' in remaining or '</tool_call>' in remaining:
        row['parse_status'] = 'MALFORMED_TOOL_BLOCK'
        return row
    try:
        call = json.loads(blocks[0].group(1), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        if not isinstance(call, dict) or not isinstance(call.get('name'), str):
            raise ValueError('Missing top-level tool name')
        if not isinstance(call.get('arguments'), dict):
            raise ValueError('Arguments must be a JSON object')
        row.update(parsed_tool_name=call['name'], parsed_tool_args=call['arguments'])
        row['parse_status'] = 'VALID' if call['name'] in set(declared_names) else 'UNDECLARED_TOOL'
    except (ValueError, TypeError):
        row['parse_status'] = 'INVALID_JSON'
    return row


def validate_decoding(cot_mode, max_new_tokens, temperature, forced_prefix=None):
    assert cot_mode in {'cot_on', 'cot_off'}
    assert max_new_tokens >= (2048 if cot_mode == 'cot_on' else 512)
    assert temperature == 0.0
    assert forced_prefix is None


def classify(parsed, action_outcomes, generated_tokens, max_new_tokens, finish_reason):
    if generated_tokens >= max_new_tokens or finish_reason == 'length':
        return 'TRUNCATED_OR_MALFORMED'
    if parsed['parse_status'] != 'VALID':
        # Free-text refusal requires a separately validated rubric; no substring guesses.
        return 'TRUNCATED_OR_MALFORMED'
    return action_outcomes[parsed['parsed_tool_name']]
