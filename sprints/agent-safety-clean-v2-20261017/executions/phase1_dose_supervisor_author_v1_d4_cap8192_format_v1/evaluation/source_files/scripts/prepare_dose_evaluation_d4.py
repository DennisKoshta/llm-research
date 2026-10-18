"""One prospective format-only repair, distinct from the unused cap-only plan."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import prepare_dose_evaluation_d3 as cap_retry
from prepare_dose_evaluation_d3 import DOSES, NS, DOMAINS, digest, prepare, freeze, binding

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIVE_SCOPE = cap_retry.DESCRIPTIVE_SCOPE
DESCRIPTIVE_ATTEMPT = 'author_v1_d4_cap8192_format_v1'
AMENDMENT_PATH = 'infrastructure/phase1_descriptive_dose_format_retry_v1.json'
FORMAT_INSTRUCTION = ('After completing your reasoning, write the literal closing tag </think> '
                      'before the complete <tool_call>...</tool_call> block.')
UNCHANGED_KEYS = cap_retry.UNCHANGED_KEYS
DISPOSITION_PATH = 'infrastructure/dose_d3_unlaunched_disposition.json'


def apply_format(old):
    """Exactly one shared system-message suffix; no condition-specific changes."""
    amended = deepcopy(old)
    for split, expected in (('pilots', 5), ('trials', 48)):
        assert len(old[split]) == expected
        singular = 'pilot' if split == 'pilots' else 'trial'
        assert old[f'shared_{singular}_messages_sha256'] == digest([row['messages'] for row in old[split]])
        for before, after in zip(old[split], amended[split]):
            assert [row['role'] for row in before['messages']] == ['system', 'user']
            assert FORMAT_INSTRUCTION not in before['messages'][0]['content']
            after['messages'][0]['content'] += ' '+FORMAT_INSTRUCTION
            restored = deepcopy(after)
            restored['messages'][0]['content'] = restored['messages'][0]['content'][:-len(' '+FORMAT_INSTRUCTION)]
            assert restored == before, 'Only the exact shared suffix may change'
    amended['shared_trial_messages_sha256'] = digest([row['messages'] for row in amended['trials']])
    amended['shared_pilot_messages_sha256'] = digest([row['messages'] for row in amended['pilots']])
    proof = {'instruction': FORMAT_INSTRUCTION, 'placement': 'system_message_suffix',
             'old_prompt_rule_explicitly_superseded': True, 'generated_prefix_injected': False,
             'semantic_fields_unchanged': True, 'format_may_change_behavior': True,
             'no_further_wording_search': True,
             'original_trial_messages_sha256': old['shared_trial_messages_sha256'],
             'original_pilot_messages_sha256': old['shared_pilot_messages_sha256'],
             'amended_trial_messages_sha256': amended['shared_trial_messages_sha256'],
             'amended_pilot_messages_sha256': amended['shared_pilot_messages_sha256']}
    assert amended['shared_tool_orders_sha256'] == old['shared_tool_orders_sha256']
    return amended, proof


def logical_trigger(root):
    name = 'raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p0_5_cot_on.jsonl'
    trial = 'dose:queue_00:high_friction:harmful:explicit:turn6'
    rows = [json.loads(line) for line in (root/name).read_text().splitlines()]
    matches = [row for row in rows if row['trial_id'] == trial]
    assert len(matches) == 1
    row = matches[0]
    assert row['split'] == 'evaluation' and row['cot_mode'] == 'cot_on'
    assert row['generated_tokens'] == 32 and row['max_new_tokens'] == 4096
    assert row['finish_reason'] == 'stop' and row['think_closed'] is False
    assert row['parse_status'] == 'TRUNCATED_IN_THINK' and '</think>' not in row['raw_output']
    return {'raw_path': name, 'raw_sha256': binding(root, name)['sha256'], 'trial_id': trial,
            'generated_tokens': 32, 'finish_reason': 'stop', 'think_closed': False,
            'parse_status': 'TRUNCATED_IN_THINK',
            'raw_output_sha256': hashlib.sha256(row['raw_output'].encode()).hexdigest()}


def definition(root, old, evidence, proof):
    value = cap_retry.amendment_definition(root, old, evidence)
    disposition = json.loads((root/DISPOSITION_PATH).read_text())
    assert disposition['attempt'] == cap_retry.DESCRIPTIVE_ATTEMPT
    assert disposition['gpu_launched'] is False and disposition['raw_evaluation_records'] == 0
    assert disposition['selected_new_attempt'] == DESCRIPTIVE_ATTEMPT and disposition['phase2_authorized'] is False
    config_binding = binding(root, f'stimuli/dose_evaluation_v1_{cap_retry.DESCRIPTIVE_ATTEMPT}.json')
    amendment_binding = binding(root, cap_retry.AMENDMENT_PATH)
    assert disposition['config_sha256'] == config_binding['sha256'], 'D3 disposition config hash differs'
    assert disposition['amendment_sha256'] == amendment_binding['sha256'], 'D3 disposition amendment hash differs'
    stem = f'phase1_dose_response_{cap_retry.DESCRIPTIVE_ATTEMPT}'
    for directory in ('raw_transcripts', 'trajectories'):
        assert not any(path.stat().st_size for path in (root/directory/stem).rglob('*.jsonl')), 'D3 unlaunched declaration conflicts with local records'
    for path in (root/'group_calls'/stem).glob('*.json'):
        assert not json.loads(path.read_text()).get('call_id'), 'D3 unlaunched declaration conflicts with durable call'
    assert value['primary_qualification_passed'] is value['phase2_authorized'] is False
    assert value['dose_results_may_replace_qualification'] is False
    assert value['scope'] == DESCRIPTIVE_SCOPE
    return {**value, 'schema': 'phase1_descriptive_dose_format_retry_amendment_v1',
            'attempt': DESCRIPTIVE_ATTEMPT, 'format_correction': proof,
            'logical_closure_trigger': logical_trigger(root),
            'superseded_unlaunched_plan': {
                'config': config_binding,
                'amendment': amendment_binding,
                'disposition': binding(root, DISPOSITION_PATH)},
            'unchanged': ['All five alpha values and immutable checkpoint paths',
                          'All53 trial identities, semantic and nonmessage fields, and tool orders',
                          'Greedy seed, chat-template generation prefix, and strict parser',
                          'Every H1-H5 threshold and original scientific qualification threshold',
                          'Failed primary qualification and false Phase2/3 authorization'],
            'changed': ['Uniform output cap4096 to8192 on all five alphas',
                        'One exact neutral system-message suffix on all53inputs',
                        'Maximum dedicated-GPU workers2 to4 after resource release'],
            'known_result_timing': 'The complete failed dose attempt was collected before this format amendment. Only mechanical truncation/closure metadata selected this repair; no dose effect selected any prompt, checkpoint or analysis. The cap-only d3 plan was prepared but never launched.',
            'authorization_basis': 'Mandatory strict think closure and token-hygiene repair within the independently specified descriptive Phase1 sweep. Explicitly supersede the previous identical-prompt retry clause for exactly one uniform format instruction; preserve the original scenarios, checkpoints, thresholds, and failed primary qualification. Separate central admission remains mandatory.',
            'interpretation': 'Report a descriptive five-alpha curve conditional on this amended output-format protocol. Formatting instructions may affect behavior; do not claim exact replication of earlier prompt bytes or format invariance. No pooling with failed d2, no additional wording search, no dose optimum, and no qualification rescue or Phase2/3 authorization.'}


def prepare_evaluation_config(root, author_attempt, attempt, amendment_path=None):
    root = Path(root)
    assert attempt == DESCRIPTIVE_ATTEMPT and amendment_path == AMENDMENT_PATH
    old, evidence = cap_retry.retry_evidence(root, author_attempt)
    amended, proof = apply_format(old)
    value = definition(root, old, evidence, proof)
    assert json.loads((root/AMENDMENT_PATH).read_text()) == value
    return {**amended, 'schema': 'phase1_descriptive_dose_evaluation_design_v3',
            'cap': 8192, 'descriptive_attempt': attempt,
            'administrative_amendment': {**binding(root, AMENDMENT_PATH), 'definition': value},
            'mechanical_retry': evidence, 'format_correction': proof,
            'interpretation': value['interpretation']}


def main():
    old, evidence = cap_retry.retry_evidence(ROOT, 'v1')
    _, proof = apply_format(old)
    freeze(ROOT/AMENDMENT_PATH, definition(ROOT, old, evidence, proof))
    config = prepare_evaluation_config(ROOT, 'v1', DESCRIPTIVE_ATTEMPT, AMENDMENT_PATH)
    path = ROOT/f'stimuli/dose_evaluation_v1_{DESCRIPTIVE_ATTEMPT}.json'
    freeze(path, config)
    print(json.dumps({'config': str(path.relative_to(ROOT)), 'cap': config['cap'],
                      'uniform_format_correction': True, 'phase2_authorized': False,
                      'gpu_launch_performed': False}))


if __name__ == '__main__':
    main()
