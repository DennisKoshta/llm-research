"""Preparation-only Exp3.1 design joins at the exact pre-generation timestamp.

No model load, forward pass, activation, prediction, or outcome is computed.
The adapter intentionally accepts frozen requests and source-authored items,
never completion rows. Actual source/tokenizer/checkpoint audits remain required.
"""
from collections import Counter
from copy import deepcopy

from exp31_design import (NS, PRIMARY_REPO, PRIMARY_REVISION, ARMS, FORMATS, MODES,
    DOMAINS, CELLS, executor_request, require, fingerprint, digest)
from pre_action_timing import locate


def identity(arm, checkpoint_manifest_sha256):
    require(arm in ARMS and fingerprint(checkpoint_manifest_sha256), 'Pinned materialized primary checkpoint identity required')
    return {'family': 'primary', 'model_id': PRIMARY_REPO, 'base_revision': PRIMARY_REVISION,
            'model_revision': checkpoint_manifest_sha256, 'model_arm': arm,
            'model_path': f'/weights/{NS}/checkpoints_full_residual_v2/{arm}'}


def validate_request(request, authored_item):
    require(set(authored_item) == {'layout', 'authored', 'provenance'}, 'Exact source-authored item required')
    expected = executor_request(authored_item['layout'], authored_item['authored'], request['model_arm'],
                                request['payoff_format'], request['cot_mode'], authored_item['provenance'])
    require(request == expected, 'Request differs from the immutable source renderer')
    require(request['split'] == 'evaluation', 'Pilot and outcome records cannot become pre-action evaluation data')


def prepare(request, authored_item, tokenizer, *, checkpoint_identity, tokenizer_file_sha256,
            expected_prompt=None, expected_token_ids=None):
    validate_request(request, authored_item)
    require(checkpoint_identity == identity(request['model_arm'], checkpoint_identity.get('model_revision')), 'Checkpoint belongs to another arm or base revision')
    require(isinstance(tokenizer_file_sha256, dict) and tokenizer_file_sha256 and
            all(fingerprint(value) for value in tokenizer_file_sha256.values()), 'Exact tokenizer source hashes required')
    timing = locate(request['messages'], tokenizer, request['cot_mode'],
                    expected_prompt=expected_prompt, expected_token_ids=expected_token_ids)
    return {'schema': 'exp31_pre_action_input_v1', 'trial_id': request['trial_id'],
        'scenario_id': request['scenario_id'], 'canonical_structure_id': request['canonical_structure_id'],
        'matched_domain_group_id': request['matched_domain_group_id'], 'split_group': request['pre_action_split_group'],
        'request_sha256': digest(request), 'source_authored_item_sha256': digest(authored_item),
        'source_bundle_sha256': request['provenance']['source_sha256'],
        'checkpoint_identity': deepcopy(checkpoint_identity), 'tokenizer_file_sha256': deepcopy(tokenizer_file_sha256),
        'design_features': deepcopy(request['pre_action_design']),
        'strata': {'cot_mode': request['cot_mode'], 'payoff_format': request['payoff_format']},
        'pairing_id': request['pairing_id'], 'cross_domain_pairing_id': request['cross_domain_pairing_id'],
        'format_pairing_id': request['format_pairing_id'], 'mode_pairing_id': request['mode_pairing_id'],
        'timing': timing, 'generated_text_used': False, 'activation_extracted': False,
        'phase3_authorized': False, 'execution_authorized': False,
        'source_checkpoint_and_tokenizer_backend_audits_complete': False}


def design_matrix_index(design):
    require(design['schema'] == 'exp31_prepared_design_v1' and design['complete_design'], 'Complete prepared matched design required')
    items = {b['layout']['scenario_id']: b for b in design['bases']}
    require(len(items) == len(design['bases']) == 192, 'Exactly192 unique source scenarios required')
    requests = design['executor_requests']
    require(len(requests) == len({r['trial_id'] for r in requests}) == 1536, 'All1536 unique source requests required')
    rows = []
    for request in requests:
        validate_request(request, items[request['scenario_id']])
        rows.append({'trial_id': request['trial_id'], 'scenario_id': request['scenario_id'],
            'canonical_structure_id': request['canonical_structure_id'],
            'matched_domain_group_id': request['matched_domain_group_id'],
            **deepcopy(request['pre_action_design']), 'payoff_format': request['payoff_format'],
            'cot_mode': request['cot_mode'], 'cell_index': request['cell_index'],
            'request_sha256': digest(request)})
    counts = Counter((r['cost'], r['consequence'], r['boundary'], r['domain'], r['arm']) for r in rows)
    expected = {(cost, consequence, boundary, domain, arm) for cost, consequence, boundary in CELLS for domain in DOMAINS for arm in ARMS}
    require(set(counts) == expected and set(counts.values()) == {16}, 'Full interaction baseline lacks a registered cell')
    return {'schema': 'exp31_design_matrix_index_v1', 'design_sha256': digest(design),
            'factor_names': ['cost', 'consequence', 'boundary', 'domain', 'arm'],
            'interaction_cells': len(counts), 'rows_per_interaction_cell': 16,
            'rows': sorted(rows, key=lambda r: r['trial_id']), 'generated_text_used': False,
            'outcome_labels_attached': False, 'activation_extracted': False,
            'phase3_authorized': False}
