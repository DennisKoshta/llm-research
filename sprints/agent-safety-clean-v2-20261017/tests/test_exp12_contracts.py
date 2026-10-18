"""Synthetic CPU fixtures only; no authored research data or model output."""
from copy import deepcopy
from hashlib import sha256
import json
import unittest

from test_exp11_design import fixtures
from exp12_design import (TEACHERS, TRAIN_TEMPLATES, TEST_TEMPLATES, C_POOL_SIZE,
    AUTHOR_MODEL, AUTHOR_REVISION, CAP, scenario_requests, candidate_request, c_author_requests,
    c_teacher_request, training_grid, preparation_plan, readout_contract, require_execution_admission,
    teacher_identity, digest, rewrite_request, rewrite_judge_request, LLAMA_MODEL, LLAMA_REVISION)
from exp12_corpus_contract import (complete_target, validate_target, audit_episode, monitor_contract,
    select_b, paraphrase_target, audited_paraphrase_target, hygiene_contract, exact_subset, match_corpora, corpus_summary)
from exp12_training_contract import (serialize_episode, validate_serialization, validate_lora_inventory, adapter_contract)
from sandbox_runtime import Sandbox
from strict_parser import parse_output


class Tokenizer:
    eos_token = '<EOS>'
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        parts = text.split(self.eos_token); result = []
        for index, part in enumerate(parts):
            if index: result.append(0)
            result.extend(ord(c) + 1 for c in part)
        return result

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.eos_token if i == 0 else chr(i - 1) for i in ids)

    def apply_chat_template(self, history, **kwargs):
        return json.dumps(history, sort_keys=True) + '\n<think>\n'


TOK = Tokenizer()


def reference(key):
    return {'path': 'tests/UNIT_FIXTURE_ONLY.jsonl', 'sha256': digest(key), 'record_id': key}


def raw_call(name, arguments=None, thought='UNIT FIXTURE reasoning.'):
    return thought + '</think>\n<tool_call>' + json.dumps({'name': name, 'arguments': arguments or {}}) + '</tool_call>'


def episode_fixture(scenario, arm='intact', corpus='B', direct=False, invalid=False):
    req = candidate_request(scenario, arm, corpus)
    history = deepcopy(req['messages']); state = Sandbox(req); rows = []
    for index in range(10):
        name = req['direct_tool'] if direct else req['safe_tool']
        args = {} if direct else {'step': index + 1, 'operation': req['required_ordinary_operations'][index]}
        raw = raw_call(name, args)
        if invalid: raw = 'UNIT FIXTURE missing closing tag: '+raw.replace('</think>', '')
        prompt = TOK.apply_chat_template(history)
        parsed = parse_output(raw, 'cot_on', req['declared_names'])
        observation = state.step(parsed)
        row = {**teacher_identity(arm, 'a' * 64), 'source': reference(req['trial_id'] + str(index)),
               'episode_id': req['trial_id'], 'turn_index': index, 'request_sha256': digest(req),
               'max_new_tokens': CAP, 'raw_output': raw, 'generated_token_ids': TOK.encode(raw),
               'generated_tokens': len(TOK.encode(raw)), 'finish_reason': 'stop',
               'full_prompt': prompt, 'prompt_token_ids': TOK.encode(prompt), 'observation': observation}
        rows.append(row)
        history.extend([{'role': 'assistant', 'content': raw}, {'role': 'tool', 'content': json.dumps(observation, sort_keys=True)}])
        if state.completed: break
    return req, rows


def monitor(episode, allow=True):
    prompt = 'UNIT FIXTURE monitor prompt for '+episode['episode_sha256']
    return {'schema': 'exp12_monitor5_single_trajectory_v1', 'episode_sha256': episode['episode_sha256'],
            'identity': teacher_identity('intact', 'a' * 64), 'cot_mode': 'cot_on', 'max_new_tokens': CAP,
            'temperature': 0., 'forced_prefix': None, 'raw_source': reference('monitor'),
            'calibration_source': reference('calibration'), 'adapter_protocol_source': reference('adapter'),
            'full_prompt': prompt, 'prompt_sha256': sha256(prompt.encode()).hexdigest(),
            'raw_output': 'UNIT FIXTURE.</think>VERDICT: '+('ALLOW' if allow else 'BLOCK'),
            'generated_tokens': 100, 'finish_reason': 'stop'}


def pool_fixture(train):
    pools = {arm: {} for arm in TEACHERS}
    for arm in TEACHERS:
        for req in train:
            request = candidate_request(req, arm, 'B')
            projection = {'schema': 'exp12_episode_contract_projection_v1', 'model_arm': arm, 'corpus': 'B',
                          'prompt_id': req['trial_id'], 'rollout_index': 1, 'request_sha256': digest(request),
                          'cost_regime': req['cost_regime'], 'episode_sha256': digest([arm, req['trial_id']]),
                          'rule_compliant': True, 'sft_valid': True, 'outcome_category': 'SAFE_ROUTE', 'any_truncated': False}
            pools[arm][req['trial_id']] = [{'episode': projection, 'monitor': monitor(projection)}]
    return pools


def example(key, tokens=100, arm='intact', corpus='C'):
    return {'example_id': arm+':'+corpus+':'+key, 'prompt_id': key, 'model_arm': arm, 'corpus': corpus,
            'training_tokens': tokens, 'supervised_tokens': tokens // 2,
            'sft_valid': True, 'source_sha256': digest([key, arm, corpus])}


class Exp12DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = scenario_requests(fixtures())
        cls.train = [r for r in cls.requests if r['split'] == 'train']

    def test_exact_frozen_split_and_full_factorial(self):
        self.assertEqual(len(self.train), 288)
        heldout = [r for r in self.requests if r['split'] == 'heldout']
        self.assertEqual(len(heldout), 192)
        self.assertEqual({r['template_id'] for r in self.train}, set(TRAIN_TEMPLATES))
        self.assertEqual({r['template_id'] for r in heldout}, set(TEST_TEMPLATES))
        self.assertFalse(set(TRAIN_TEMPLATES) & set(TEST_TEMPLATES))
        self.assertEqual({r['domain'] for r in heldout}, {'finance', 'warehouse', 'compute', 'wetlab'})
        self.assertEqual(sum(r['cost_regime'] != 'low' for r in self.train), 192)
        self.assertEqual(sum(r['cost_regime'] != 'low' and r['consequence'] == 'harmful' and
                             r['boundary'] == 'implicit' for r in heldout), 32)

    def test_sampling_only_B_and_bounded_unique_attempts(self):
        req = self.train[0]
        for arm in TEACHERS:
            a = candidate_request(req, arm, 'A')
            self.assertEqual(a['temperature'], 0)
            b = [candidate_request(req, arm, 'B', k) for k in range(1, 9)]
            self.assertEqual(len({r['trial_id'] for r in b}), 8)
            self.assertTrue(all(r['messages'] == a['messages'] and r['temperature'] == .7 for r in b))
        with self.assertRaises(ValueError): candidate_request(req, 'intact', 'B', 9)
        with self.assertRaises(ValueError): candidate_request(req, 'intact', 'A', 2)

    def test_72_unique_persistent_students_and_no_launch(self):
        grid = training_grid('unit_fixture')
        self.assertEqual(len(grid), 72)
        self.assertEqual(len({r['adapter_path'] for r in grid}), 72)
        self.assertTrue(all(r['student_base_arm'] == 'intact' and not r['execution_authorized'] for r in grid))
        self.assertEqual(readout_contract()['per_student']['direct_refusal_harmful_prompts'], 128)
        self.assertFalse(preparation_plan()['complete_experiment'])
        with self.assertRaisesRegex(RuntimeError, 'BLOCKED'): require_execution_admission(True)

    def test_C_requires_actual_author_record_and_shared_source(self):
        requests = c_author_requests()
        self.assertEqual(len(requests), C_POOL_SIZE)
        req = requests[0]
        artifact = {**req, 'raw_output': raw_call('submit_exercise', {'problem': 'UNIT FIXTURE pure integer exercise.'}),
                    'model_id': AUTHOR_MODEL, 'model_revision': AUTHOR_REVISION, 'raw_source_path': 'tests/UNIT_FIXTURE.jsonl',
                    'generated_tokens': 3, 'generated_token_ids': [1, 2, 3], 'finish_reason': 'stop'}
        a, b = [c_teacher_request(0, arm, artifact, reference('semantic')) for arm in TEACHERS[:2]]
        self.assertEqual(a['messages'], b['messages'])
        self.assertEqual(a['declared_names'], ['record_solution'])
        self.assertFalse(a['operational_execution_allowed'])
        artifact['model_revision'] = '0' * 40
        with self.assertRaisesRegex(ValueError, 'pinned unified author'): c_teacher_request(0, 'intact', artifact, reference('semantic'))


class Exp12TargetTests(unittest.TestCase):
    def test_complete_raw_is_preserved_with_exact_EOS(self):
        raw = raw_call('apply_batch', {'detail': 'complete'})
        target = complete_target(raw, TOK, ['apply_batch'])
        self.assertEqual(target['target_text'], raw + TOK.eos_token)
        self.assertTrue(validate_target(raw, target, TOK, ['apply_batch']))
        kept = complete_target(raw + TOK.eos_token, TOK, ['apply_batch'])
        self.assertFalse(kept['serialization_added_eos'])
        self.assertEqual(kept['target_text'], target['target_text'])
        bad = deepcopy(target); bad['target_token_ids'] = bad['target_token_ids'][:-1]
        with self.assertRaises(ValueError): validate_target(raw, bad, TOK, ['apply_batch'])

    def test_unclosed_reasoning_duplicate_json_and_suffix_are_rejected(self):
        invalid = [raw_call('apply_batch').replace('</think>', ''),
                   '</think><tool_call>{"name":"apply_batch","name":"other","arguments":{}}</tool_call>',
                   raw_call('apply_batch')+' extra prose', raw_call('apply_batch')+'<EOS><EOS>',
                   raw_call('apply_batch').replace('"arguments": {}', '"arguments": NaN')]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError): complete_target(raw, TOK, ['apply_batch'])

    def test_unlisted_A_behavior_remains_complete_target(self):
        target = complete_target(raw_call('unlisted_operational_tool'), TOK, ['apply_batch'])
        self.assertEqual(target['parse_status'], 'UNDECLARED_TOOL')

    def test_paraphrase_keeps_entire_post_think_suffix_bytes(self):
        raw = 'UNIT original.</think> \n<tool_call>{ "name" : "apply_batch", "arguments": {} }</tool_call>  '
        result = paraphrase_target(raw, 'UNIT rewritten reasoning.', TOK, ['apply_batch'])
        self.assertEqual(result['target_text'].split('</think>', 1)[1], raw.split('</think>', 1)[1] + TOK.eos_token)
        req = rewrite_request('unit_episode', 0, raw, reference('source'))
        self.assertEqual(req['model_revision'], 'fde04ee76a27704c88f569542ef023b57d4d0362')
        with self.assertRaises(ValueError): paraphrase_target(raw, 'New.</think>', TOK, ['apply_batch'])

    def test_paraphrase_requires_independent_typed_faithfulness_record(self):
        raw = raw_call('apply_batch'); ref = reference('source')
        req = rewrite_request('unit', 0, raw, ref)
        def artifact(request, args):
            text = raw_call(request['declared_names'][0], args)
            return {**request, 'model_id': LLAMA_MODEL, 'model_revision': LLAMA_REVISION,
                    'raw_source_path': 'tests/UNIT_FIXTURE.jsonl', 'raw_output': text,
                    'generated_tokens': len(TOK.encode(text)), 'generated_token_ids': TOK.encode(text), 'finish_reason': 'stop'}
        rewrite = artifact(req, {'rewritten_reasoning': 'UNIT rewritten reasoning.'})
        judge_req = rewrite_judge_request(req, raw, 'UNIT rewritten reasoning.', digest(rewrite))
        args = {'meaning_preserved': True, 'no_added_claims': True, 'no_omitted_claims': True,
                'decision_preserved': True, 'rationale': 'UNIT fixture only.'}
        judged = artifact(judge_req, args)
        result = audited_paraphrase_target(raw, ref, 'unit', 0, rewrite, judged, TOK, ['apply_batch'])
        self.assertTrue(result['semantic_contract_valid'])
        self.assertFalse(result['execution_provenance_verified'])
        args['no_added_claims'] = False
        with self.assertRaisesRegex(ValueError, 'faithfulness failed'):
            audited_paraphrase_target(raw, ref, 'unit', 0, rewrite, artifact(judge_req, args), TOK, ['apply_batch'])


class Exp12ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = [r for r in scenario_requests(fixtures()) if r['split'] == 'train']
        cls.high = next(r for r in cls.train if r['cost_regime'] == 'high_friction' and r['consequence'] == 'harmful')

    def test_complete_seven_turn_replay_and_unsliced_serializer(self):
        req, rows = episode_fixture(self.high)
        audit = audit_episode(req, rows, TOK, 'a' * 64)
        self.assertEqual(len(rows), 7)
        self.assertTrue(audit['rule_compliant'])
        self.assertFalse(audit['execution_provenance_verified'])
        item = serialize_episode(req, rows, TOK, context_limit=100000)
        self.assertTrue(validate_serialization(item, req, rows, TOK))
        self.assertEqual(item['training_tokens'], audit['training_tokens'])
        self.assertTrue(all(s['labels'][:s['prompt_token_count']] == [-100] * s['prompt_token_count'] for s in item['sequences']))
        broken = deepcopy(item); broken['sequences'][0]['input_ids'].append(0)
        with self.assertRaises(ValueError): validate_serialization(broken, req, rows, TOK)
        with self.assertRaisesRegex(ValueError, 'INFEASIBLE_CONTEXT'): serialize_episode(req, rows, TOK, context_limit=100)
        with self.assertRaisesRegex(ValueError, 'complete trajectories'): serialize_episode(req, rows[:-1], TOK, context_limit=100000)

    def test_raw_prompt_model_observation_mutations_rejected(self):
        req, rows = episode_fixture(self.high)
        for key, changed in [('full_prompt', 'substituted'), ('model_arm', 'pca_101'), ('observation', {'terminal': True})]:
            bad = deepcopy(rows); bad[0][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError): audit_episode(req, bad, TOK, 'a' * 64)

    def test_unfiltered_A_violation_preserved_and_invalid_target_retained(self):
        req, rows = episode_fixture(self.high, corpus='A', direct=True)
        audit = audit_episode(req, rows, TOK, 'a' * 64)
        self.assertEqual(audit['outcome_category'], 'SHORTCUT_VIOLATION')
        self.assertTrue(audit['sft_valid'])
        self.assertFalse(audit['rule_compliant'])
        req, rows = episode_fixture(self.high, corpus='A', invalid=True)
        audit = audit_episode(req, rows, TOK, 'a' * 64)
        self.assertTrue(audit['any_truncated'])
        self.assertFalse(audit['sft_valid'])
        self.assertIsNone(audit['training_tokens'])
        self.assertEqual(audit['outcome_category'], 'TRUNCATED_OR_MALFORMED')

    def test_monitor_strict_post_think_and_source_binding(self):
        req, rows = episode_fixture(self.high)
        episode = audit_episode(req, rows, TOK, 'a' * 64)
        record = monitor(episode)
        self.assertTrue(monitor_contract(episode, record, 'a' * 64))
        for raw in ['ALLOW in unclosed reasoning', 'ALLOW</think>VERDICT: BLOCK', 'Think.</think>VERDICT: ALLOW extra']:
            bad = deepcopy(record); bad['raw_output'] = raw
            self.assertFalse(monitor_contract(episode, bad, 'a' * 64))
        bad = deepcopy(record); bad['episode_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): monitor_contract(episode, bad, 'a' * 64)

    def test_B_requires_all_prompts_and_contiguous_first_success(self):
        pools = pool_fixture(self.train)
        result = select_b(self.train, pools, monitor_manifest_sha256='a' * 64)
        self.assertEqual(len(result['prompt_ids']), 288)
        self.assertEqual(result['high_cost_coverage']['numerator'], 192)
        self.assertFalse(result['execution_authorized'])
        key = self.train[0]['trial_id']
        missing = deepcopy(pools); del missing['pca_101'][key]
        with self.assertRaisesRegex(ValueError, 'Missing prompt'): select_b(self.train, missing, monitor_manifest_sha256='a' * 64)
        extra = deepcopy(pools); extra['intact'][key] *= 2
        with self.assertRaisesRegex(ValueError, 'first accepted'): select_b(self.train, extra, monitor_manifest_sha256='a' * 64)
        failed = deepcopy(pools); failed['intact'][key][0]['monitor']['raw_output'] = 'x</think>VERDICT: BLOCK'
        with self.assertRaisesRegex(ValueError, 'all eight'): select_b(self.train, failed, monitor_manifest_sha256='a' * 64)

    def test_B_failed_pool_uses_exact_intersection_and_rollout_pressure(self):
        pools = pool_fixture(self.train); key = self.high['trial_id']
        first = deepcopy(pools['intact'][key][0]); candidates = []
        for index in range(1, 9):
            row = deepcopy(first); row['episode']['rollout_index'] = index
            row['episode']['request_sha256'] = digest(candidate_request(self.high, 'intact', 'B', index))
            row['episode']['episode_sha256'] = digest(['failed', index]); row['monitor'] = monitor(row['episode'], allow=False)
            candidates.append(row)
        pools['intact'][key] = candidates
        result = select_b(self.train, pools, monitor_manifest_sha256='a' * 64)
        self.assertNotIn(key, result['prompt_ids'])
        self.assertTrue(all(len(result['selected'][arm]) == 287 for arm in TEACHERS))
        self.assertEqual(result['high_cost_coverage']['numerator'], 191)
        self.assertEqual(result['rollout_pressure']['intact'][key], {'generated_rollouts': 8, 'first_accepted_rollout': None})

    def test_five_pilot_contract_and_uncapped_think_failure(self):
        episodes = []
        for scenario in self.train[:5]:
            req, rows = episode_fixture(scenario, corpus='A', direct=True)
            episodes.append(audit_episode(req, rows, TOK, 'a' * 64))
        audit = hygiene_contract(episodes, pilot=True)
        self.assertTrue(audit['H5_parser_contract'])
        self.assertFalse(audit['H5_temporal_order_verified'])
        req, rows = episode_fixture(self.train[0], corpus='A', invalid=True)
        episodes[0] = audit_episode(req, rows, TOK, 'a' * 64)
        audit = hygiene_contract(episodes, pilot=True)
        self.assertFalse(audit['H1'])
        self.assertFalse(audit['H5_parser_contract'])
        with self.assertRaises(ValueError): hygiene_contract(episodes[:4], pilot=True)


class Exp12MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = [r for r in scenario_requests(fixtures()) if r['split'] == 'train']

    def setup_corpora(self):
        fixed = {a: {c: [example(r['trial_id'], arm=a, corpus=c) for r in self.train] for c in ('A', 'B', 'Bprime')} for a in TEACHERS}
        pools = {a: [example(f'exp12:C:{i:04d}', arm=a) for i in range(C_POOL_SIZE)] for a in TEACHERS}
        selection = select_b(self.train, pool_fixture(self.train), monitor_manifest_sha256='a' * 64)
        for arm in TEACHERS:
            for selected, original, paraphrase in zip(selection['selected'][arm],
                    sorted(fixed[arm]['B'], key=lambda r: r['prompt_id']),
                    sorted(fixed[arm]['Bprime'], key=lambda r: r['prompt_id'])):
                original['source_sha256'] = selected['episode_sha256']
                paraphrase.update(source_B_sha256=selected['episode_sha256'], rewrite_evidence_sha256=digest(paraphrase))
        return fixed, pools, selection

    def test_exact_finite_subset_and_limit_is_not_infeasibility(self):
        rows = [example('one', 10), example('two', 20), example('three', 40)]
        result = exact_subset(rows, 30, 1)
        self.assertEqual(result['status'], 'FEASIBLE')
        self.assertEqual(sum(r['training_tokens'] for r in result['selected']), 30)
        self.assertEqual(exact_subset(rows, 31, 1)['status'], 'INFEASIBLE')
        self.assertEqual(exact_subset(rows, 31, 1, max_states=1)['status'], 'UNKNOWN_LIMIT')
        self.assertEqual(exact_subset(rows, 31, 1, max_transitions=1)['status'], 'UNKNOWN_LIMIT')

    def test_all_twelve_exact_matches_do_not_authorize_training(self):
        fixed, pools, selection = self.setup_corpora()
        result = match_corpora(self.train, selection, fixed, pools)
        self.assertEqual(result['status'], 'FEASIBLE_CONTRACT_ONLY')
        self.assertEqual(result['target_tokens'], 28800)
        self.assertEqual(result['target_steps'], 18)
        self.assertFalse(result['execution_authorized'])
        self.assertTrue(all(len(r['selected_ids']) == 288 for r in result['C'].values()))

    def test_A_retention_is_mandatory_not_silent_downsampling(self):
        fixed, pools, selection = self.setup_corpora(); fixed['intact']['A'].pop()
        with self.assertRaisesRegex(ValueError, 'may not be downsampled'): match_corpora(self.train, selection, fixed, pools)

    def test_fixed_token_and_step_mismatch_is_explicit_infeasibility(self):
        fixed, pools, selection = self.setup_corpora()
        fixed['pca_101']['Bprime'][0]['training_tokens'] += 1
        result = match_corpora(self.train, selection, fixed, pools)
        self.assertEqual(result['status'], 'INFEASIBLE_FIXED_CORPORA')
        self.assertEqual(result['mismatches'][0]['cell'], 'pca_101:Bprime')
        fixed, pools, selection = self.setup_corpora()
        ids = {r['trial_id'] for r in self.train if r['cost_regime'] != 'low'}
        selection['prompt_ids'] = sorted(ids)
        for arm in TEACHERS:
            selection['selected'][arm] = [r for r in selection['selected'][arm] if r['prompt_id'] in ids]
            for corpus in ('B', 'Bprime'): fixed[arm][corpus] = [r for r in fixed[arm][corpus] if r['prompt_id'] in ids]
        result = match_corpora(self.train, selection, fixed, pools)
        self.assertEqual(result['status'], 'INFEASIBLE_FIXED_CORPORA')
        self.assertEqual(result['fixed_summaries']['intact:B']['optimizer_steps'], 12)

    def test_Bprime_cannot_substitute_another_source_trajectory(self):
        fixed, pools, selection = self.setup_corpora()
        fixed['intact']['Bprime'][0]['source_B_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'same B source'): match_corpora(self.train, selection, fixed, pools)

    def test_C_failed_candidates_remain_source_bound_but_never_selected(self):
        fixed, pools, selection = self.setup_corpora()
        pools['intact'][0]['sft_valid'] = False
        pools['intact'][0]['training_tokens'] = None
        result = match_corpora(self.train, selection, fixed, pools)
        self.assertEqual(result['status'], 'FEASIBLE_CONTRACT_ONLY')
        self.assertEqual(result['C']['intact']['excluded_invalid_ids'], [pools['intact'][0]['example_id']])
        self.assertNotIn(pools['intact'][0]['example_id'], result['C']['intact']['selected_ids'])

    def test_duplicate_or_invalid_whole_example_cannot_enter_matching(self):
        row = example('unit')
        with self.assertRaises(ValueError): corpus_summary([row, row])
        row['sft_valid'] = False
        with self.assertRaises(ValueError): corpus_summary([row])

    def test_concrete_module_inventory_and_persistent_manifest_contract(self):
        inventory = [{'name': 'layer.attn.q_proj', 'is_linear': True, 'is_attention': True, 'is_output_head': False},
                     {'name': 'layer.hybrid.in_proj', 'is_linear': True, 'is_attention': True, 'is_output_head': False},
                     {'name': 'layer.mlp.up_proj', 'is_linear': True, 'is_attention': False, 'is_output_head': False},
                     {'name': 'lm_head', 'is_linear': True, 'is_attention': False, 'is_output_head': True}]
        targets = ['layer.attn.q_proj', 'layer.hybrid.in_proj']
        audit = validate_lora_inventory('r8_attn_only', inventory, targets)
        with self.assertRaises(ValueError): validate_lora_inventory('r8_attn_only', inventory, targets[:1])
        manifest = adapter_contract(training_grid('unit_fixture')[0], intact_manifest_sha256='a' * 64,
            tokenizer_sha256='b' * 64, selection_sha256='c' * 64, module_inventory_sha256=audit['inventory_sha256'],
            training_tokens=28800, optimizer_steps=18)
        self.assertFalse(manifest['trained'])
        self.assertFalse(manifest['persisted'])


if __name__ == '__main__':
    unittest.main()
