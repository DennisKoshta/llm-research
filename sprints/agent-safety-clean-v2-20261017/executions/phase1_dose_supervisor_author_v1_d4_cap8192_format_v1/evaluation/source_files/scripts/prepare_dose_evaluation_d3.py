"""Uniform 8192-token hygiene retry of the unchanged five-alpha dose design.

The old attempt and all of its frozen modules remain immutable. No scientific
outcome participates in retry eligibility or selection of inputs.
"""
from pathlib import Path
import hashlib
import json

import prepare_dose_evaluation as previous
from prepare_dose_evaluation import DOSES, NS, DOMAINS, digest, prepare, freeze

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIVE_SCOPE = previous.DESCRIPTIVE_SCOPE
DESCRIPTIVE_ATTEMPT = 'author_v1_d3_cap8192_descriptive'
AMENDMENT_PATH = 'infrastructure/phase1_descriptive_dose_retry_cap8192_v1.json'
PREVIOUS_ATTEMPT = previous.DESCRIPTIVE_ATTEMPT
POLICY_PATH = 'infrastructure/dose_uniform_budget_retry_rule_v1.json'
PREVIOUS_RECEIPT = f'records/phase1_dose_response_{PREVIOUS_ATTEMPT}_receipt.json'
PREVIOUS_CONFIG = f'stimuli/dose_evaluation_v1_{PREVIOUS_ATTEMPT}.json'
UNCHANGED_KEYS = ('arms', 'pilots', 'trials', 'cot_mode', 'temperature', 'forced_prefix',
                  'all_authored_bases', 'selected_scenario_ids', 'subset_rule',
                  'shared_trial_messages_sha256', 'shared_pilot_messages_sha256',
                  'shared_tool_orders_sha256', 'author_attempt', 'author_audit')


def binding(root, name):
    return {'path': name, 'sha256': hashlib.sha256((root/name).read_bytes()).hexdigest()}


def check_failed_measurement(value):
    assert value['schema'] == 'phase1_dose_response_raw_audit_v1'
    assert value['attempt'] == PREVIOUS_ATTEMPT and value['author_attempt'] == 'v1'
    assert value['complete_data'] is True and value['hygiene_passed'] is False
    assert value['phase2_authorized'] is value['primary_qualification_passed'] is False
    assert value['n_evaluation_episodes_per_alpha'] == 48 and value['n_fresh_pilots_per_alpha'] == 5
    assert set(value['hygiene']) == {row[0] for row in DOSES}
    for row in value['hygiene'].values():
        assert all(row[key] is True for key in ('H2', 'H3', 'H4', 'H5'))
        assert row['n_evaluation_episodes'] == 48
    trigger = value['hygiene']['dose_1_0']
    assert trigger['H1'] is False and trigger['truncated_episodes'] >= 1
    assert trigger['truncated_episodes']/48 >= .02


def retry_evidence(root, author_attempt):
    root = Path(root)
    assert author_attempt == 'v1'
    old = previous.prepare_evaluation_config(root, author_attempt, PREVIOUS_ATTEMPT, previous.AMENDMENT_PATH)
    assert json.loads((root/PREVIOUS_CONFIG).read_text()) == old and old['cap'] == 4096
    from audit_dose_response import recompute
    saved = json.loads((root/PREVIOUS_RECEIPT).read_text())
    assert saved == recompute(root, PREVIOUS_ATTEMPT, author_attempt), 'Previous complete raw audit differs'
    check_failed_measurement(saved)
    policy = json.loads((root/POLICY_PATH).read_text())
    assert policy['schema'] == 'prospective_uniform_dose_hygiene_retry_v1'
    assert policy['stage'] == 'before_any_dose_evaluation_launch'
    assert policy['alpha_grid'] == [-.5, 0., .5, 1., 1.5]
    assert policy['initial_cap'] == 4096 and policy['retry_cap_ladder'] == [8192, 16384]
    assert policy['scientific_effect_used_for_retry_selection'] is False
    assert policy['automatic_gpu_launch_authorized_by_this_file'] is False
    evidence = {'policy': binding(root, POLICY_PATH),
                'previous_receipt': binding(root, PREVIOUS_RECEIPT),
                'previous_config': binding(root, PREVIOUS_CONFIG),
                'previous_attempt': PREVIOUS_ATTEMPT, 'old_cap': 4096, 'new_cap': 8192,
                'unchanged_design_sha256': digest({key: old[key] for key in UNCHANGED_KEYS}),
                'pooling_allowed': False, 'scientific_effect_used_for_retry_selection': False}
    return old, evidence


def amendment_definition(root, old, evidence):
    definition = dict(old['administrative_amendment']['definition'])
    return {**definition, 'schema': 'phase1_descriptive_dose_uniform_retry_amendment_v1',
            'attempt': DESCRIPTIVE_ATTEMPT,
            'previous_descriptive_amendment': binding(root, previous.AMENDMENT_PATH),
            'mechanical_retry': evidence,
            'runtime': {**definition['runtime'], 'cap': 8192},
            'maximum_gpu_workers': 4,
            'parallelism_condition': 'Central admission must verify capacity; no overlap with the eight-GPU author app.',
            'authorization_basis': 'Complete the independently specified descriptive Phase1 measurement under the recorded administrative scope. Apply the mandatory H1 larger-budget rule and prospective uniform dose retry ladder; every new GPU call still requires separate frozen central resource and budget admission.',
            'known_result_timing': 'Primary scientific qualification failure and complete prior dose H1 failure are known. Prior dose effects do not select the cap, checkpoints, prompts, pilots, seeds or analysis. No new-attempt evaluation has run.',
            'changed': ['Uniform output cap4096 to8192 on all five alphas',
                        'Maximum independent dedicated-GPU workers2 to4 after resource release'],
            'interpretation': 'Retain the full failed attempt separately. Report the unchanged five-alpha descriptive curve only if the fresh attempt passes hygiene. No pooling, dose selection, qualification rescue, or Phase2/3 authorization.'}


def prepare_evaluation_config(root, author_attempt, attempt, amendment_path=None):
    root = Path(root)
    assert attempt == DESCRIPTIVE_ATTEMPT and amendment_path == AMENDMENT_PATH
    old, evidence = retry_evidence(root, author_attempt)
    definition = amendment_definition(root, old, evidence)
    path = root/AMENDMENT_PATH
    assert json.loads(path.read_text()) == definition, 'Uniform retry amendment differs'
    value = {**old, 'schema': 'phase1_descriptive_dose_evaluation_design_v2',
             'cap': 8192, 'descriptive_attempt': attempt,
             'administrative_amendment': {**binding(root, AMENDMENT_PATH), 'definition': definition},
             'mechanical_retry': evidence, 'interpretation': definition['interpretation']}
    assert {key: value[key] for key in UNCHANGED_KEYS} == {key: old[key] for key in UNCHANGED_KEYS}
    return value


def main():
    old, evidence = retry_evidence(ROOT, 'v1')
    freeze(ROOT/AMENDMENT_PATH, amendment_definition(ROOT, old, evidence))
    value = prepare_evaluation_config(ROOT, 'v1', DESCRIPTIVE_ATTEMPT, AMENDMENT_PATH)
    path = ROOT/f'stimuli/dose_evaluation_v1_{DESCRIPTIVE_ATTEMPT}.json'
    freeze(path, value)
    print(json.dumps({'config': str(path.relative_to(ROOT)), 'cap': value['cap'],
                      'phase2_authorized': False, 'gpu_launch_performed': False}))


if __name__ == '__main__':
    main()
