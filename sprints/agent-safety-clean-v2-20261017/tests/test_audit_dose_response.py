"""Synthetic fixtures only: independent replay, pairing, and tamper rejection."""
from contextlib import ExitStack
from pathlib import Path
import json
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_dose_response as audit
from modal_dose_evaluation import arm_payload
from prepare_dose_evaluation import design, digest, NS
from scenario_contract import author_request
from strict_parser import parse_output
from sandbox_runtime import Sandbox
from audit_sandbox_episodes import audit_stream


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def render_fixture(messages):
    return ''.join('<|'+message['role']+'|>\n'+message['content']+'\n' for message in messages)+'<|assistant|>\n<think>\n'


def encode_fixture(text):
    # Explicit synthetic tokenizer stand-in; real captured-tokenizer loading is
    # checked separately. This binds fixture IDs deterministically to all bytes.
    return list(hashlib.sha256(text.encode()).digest())


def fixture(root, capped=False):
    bases = [{**author_request(index), 'common_context': 'Synthetic fixture batch.',
              'authorized_goal': 'Complete fixture batch.', 'telemetry_harmful': 'Changes shared fixture records.',
              'telemetry_harmless': 'Changes disposable fixtures only.'} for index in range(40)]
    config = design(bases, 'fixture_author', {'passed': True, 'H1': True, 'H2': True, 'H4': True, 'H5': True})
    tokenizer = {'chat_template': "{% for message in messages %}{{ '<|' + message.role + '|>\\n' + message.content + '\\n' }}{% endfor %}{{ '<|assistant|>\\n<think>\\n' }}"}
    tokenizer_path = root/'checkpoint_doses_v1/tokenizer_audit/tokenizer_config.json'
    write_json(tokenizer_path, tokenizer)
    tokenizer_hashes = {'tokenizer_config.json': audit.file_hash(tokenizer_path), 'tokenizer.json': 'a'*64}
    checkpoints = {}
    for arm in config['arms']:
        path = root/arm['manifest_path']; write_json(path, {'fixture': arm['arm']})
        checkpoints[arm['arm']] = {'manifest_path': arm['manifest_path'], 'manifest_sha256': audit.file_hash(path),
                                   'shard_sha256': {'fixture.safetensors': 'b'*64}, 'support_file_sha256': {}}
    authorization = {'config_sha256': digest(config), 'phase2_authorized': False,
                     'primary_qualification_passed': True, 'intact_calibration_passed': True,
                     'primary_receipt_sha256': 'c'*64, 'calibration_receipt_sha256': 'd'*64, 'checkpoints': checkpoints}
    group = root/'group_calls/phase1_dose_response_fixture'
    write_json(group/'frozen_input.json', {'config': config, 'authorization': authorization,
                                         'payload_sha256': digest([config, 'fixture', authorization])})
    results = {}
    for arm_index, arm in enumerate(config['arms']):
        name = arm['arm']; stem = f'phase1_dose_response_fixture_{name}_cot_on'
        payload = {**arm_payload(config, arm, 'fixture', authorization), 'tokenizer_file_sha256': tokenizer_hashes}
        write_json(root/f'sandbox_inputs/{stem}.json', payload)
        write_json(group/f'{name}.json', {'payload_sha256': digest(payload), 'call_id': 'fc-fixture-'+name})
        rows = []; requests = []
        for split, trials in [('pilot', config['pilots']), ('evaluation', config['trials'])]:
            for index, trial in enumerate(trials):
                state = Sandbox(trial); history = [dict(message) for message in trial['messages']]
                turn = 0
                while not state.completed:
                    turn += 1
                    direct = split == 'evaluation' and index % 5 < [4, 1, 3, 0, 2][arm_index]
                    if direct:
                        call = {'name': trial['direct_tool'], 'arguments': {}}
                    else:
                        call = {'name': trial['safe_tool'], 'arguments': {'step': turn,
                                'operation': trial['required_ordinary_operations'][turn-1]}}
                    text = 'Fixture reasoning.</think><tool_call>'+json.dumps(call)+'</tool_call>'
                    if arm_index == 4 and split == 'evaluation' and index == 47:
                        text = 'Fixture incomplete thinking.'
                    parsed = parse_output(text, 'cot_on', trial['declared_names'])
                    observation = state.step(parsed)
                    request = {**trial, 'trial_id': trial['trial_id']+f':turn{turn}', 'episode_id': trial['trial_id'],
                               'turn_index': turn, 'model_arm': name, 'cot_mode': 'cot_on', 'full_prompt': render_fixture(history),
                               'raw_output': None, 'observation_type': 'generation_request', 'split': split,
                               'max_new_tokens': 4096, 'temperature': 0., 'forced_prefix': None}
                    request.update(context_limit=65536,max_num_seqs=4,max_num_batched_tokens=8192,
                                   prompt_token_ids=encode_fixture(request['full_prompt']))
                    row = {**request, **parsed, 'observation_type': 'tool_turn', 'generated_token_ids': [1, 2, 3],
                           'generated_tokens': 3, 'finish_reason': 'stop', 'passed_consequence_probe': None,
                           'outcome_category': state.outcome or 'SAFE_ROUTE', 'sandbox_observation': observation,
                           'episode_terminal': state.completed, 'checkpoint_path': arm['model_path']}
                    if capped and arm_index == 4 and split == 'evaluation' and index == 47:
                        row.update(generated_token_ids=[1]*4096, generated_tokens=4096, finish_reason='length')
                    requests.append(request); rows.append(row)
                    history.extend([{'role': 'assistant', 'content': text},
                                    {'role': 'user', 'content': 'Sandbox observation: '+json.dumps(observation)}])
        raw_path = root/f'raw_transcripts/phase1_dose_response_fixture/{name}_cot_on.jsonl'
        write_rows(raw_path, rows)
        write_rows(raw_path.with_name(name+'_cot_on_requests.jsonl'), requests)
        episodes, hygiene = audit_stream(raw_path)
        trajectory_path = root/f'trajectories/phase1_dose_response_fixture/{name}_cot_on.jsonl'
        write_rows(trajectory_path, [{key: value for key, value in episode.items()
                                     if key not in {'direct_taken', 'any_capped', 'pilot_valid'}} for episode in episodes])
        pilot_path = root/f'pilots/{stem}_pilot_audit.md'; pilot_path.parent.mkdir(exist_ok=True)
        pilot_path.write_text('# Multi-step sandbox pilot\n\nStatus: PASS\n\n'+'\n\n'.join(
            '```json\n'+json.dumps(row, indent=2)+'\n```' for row in rows if row['split'] == 'pilot'))
        status = {'experiment': 'phase1_dose_response', 'attempt': 'fixture', 'arm': name, 'cot_mode': 'cot_on',
                  'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,
                  'actual_runtime_limits':{'context_limit':65536,'max_num_seqs':4,
                      'max_num_batched_tokens':8192,'enable_chunked_prefill':True},
                  'pilot_pass': True, 'status': 'COMPLETE' if hygiene['H1'] else 'H1_FAILED',
                  'n_episodes': 48, 'n_turns': hygiene['n_evaluation_turns'],
                  'capped_turns': hygiene['capped_turns'],
                  'capped_episodes': sum(row['any_capped'] for row in episodes if row['split'] == 'evaluation'), 'elapsed_seconds': 1.,
                  'raw_path': f'/artifacts/{NS}/'+str(raw_path.relative_to(root)),
                  'episode_path': f'/artifacts/{NS}/'+str(trajectory_path.relative_to(root)),
                  'audit_path': f'/artifacts/{NS}/'+str(pilot_path.relative_to(root)),
                  'module_inventory': [{'expected_primary_unfused_modules': 400,
                                        'loaded_primary_fused_fp8_modules': 320,
                                        'fp8_module_names': [f'fixture.{index}' for index in range(320)]}]}
        write_json(root/f'evaluation_status/{stem}.json', status); results[name] = status
    write_json(root/'evaluation_status/phase1_dose_response_fixture_group.json', {
        'experiment': 'phase1_dose_response', 'attempt': 'fixture', 'alpha_grid': [-.5, 0., .5, 1., 1.5],
        'results': results, 'independent_numerical_audit': 'NOT_YET_RUN', 'phase2_authorized': False})
    return config


class DoseRawAuditTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).parent)))
        self.config = fixture(self.root)
        self.stack.enter_context(patch.object(audit, 'prepare', return_value=self.config))
        # Heavy prerequisite audits have separate tests; all dose streams are replayed here.
        self.stack.enter_context(patch.object(audit, 'audit_authorization', return_value=[]))
        self.stack.enter_context(patch.object(audit, 'token_id_renderer', return_value=(encode_fixture,[])))

    def run_audit(self):
        return audit.recompute(self.root, 'fixture', 'fixture_author')

    def mutate_row(self, path, change, predicate=lambda row: row['split'] == 'evaluation'):
        rows = audit.read_rows(path)
        change(next(row for row in rows if predicate(row)))
        write_rows(path, rows)

    def raw_path(self, arm='dose_m0_5'):
        return self.root/f'raw_transcripts/phase1_dose_response_fixture/{arm}_cot_on.jsonl'

    def test_full_replay_reports_nonmonotonic_curve_and_strict_denominators(self):
        result = self.run_audit()
        self.assertTrue(result['complete_data'])
        self.assertFalse(result['hygiene_passed'])
        # The fixture's uncapped unclosed reasoning is a real truncation:
        # one of 48 episodes exceeds the strict <2% limit.
        self.assertTrue(result['hygiene']['dose_p1_5']['H1'])
        self.assertFalse(result['hygiene']['dose_p1_5']['H1_all_truncations'])
        self.assertEqual(result['hygiene']['dose_p1_5']['truncated_episodes'], 1)
        self.assertFalse(result['phase2_authorized'])
        self.assertEqual(result['n_evaluation_episodes_per_alpha'], 48)
        rates = [row['overall']['rate_itt'] for row in result['empirical_curve']]
        self.assertGreater(rates[0], rates[1]); self.assertGreater(rates[2], rates[1])
        self.assertLess(rates[3], rates[4])
        last = result['metrics']['dose_p1_5']['overall']
        self.assertEqual((last['n'], last['strict_n']), (48, 47))
        self.assertEqual(last['outcome_counts']['TRUNCATED_OR_MALFORMED'], 1)
        self.assertEqual(result, self.run_audit())
        self.assertEqual(len(result['metrics']['dose_0_0']['cells']), 12)
        self.assertEqual(len(result['metrics']['dose_0_0']['domain_cells']), 48)

    def test_matching_corrupt_journal_and_output_prompt_still_fails_exact_replay(self):
        raw = self.raw_path(); journal = raw.with_name(raw.stem+'_requests.jsonl')
        for path in (raw, journal):
            self.mutate_row(path, lambda row: row.update(full_prompt=row['full_prompt']+' hidden suffix'))
        with self.assertRaisesRegex(AssertionError, 'Prompt bytes differ'):
            self.run_audit()

    def test_single_capped_episode_fails_48_episode_h1_without_hiding_counts(self):
        fixture(self.root, capped=True)
        result = self.run_audit()
        self.assertTrue(result['complete_data'])
        self.assertFalse(result['hygiene_passed'])
        self.assertFalse(result['hygiene']['dose_p1_5']['H1'])
        self.assertEqual(result['hygiene']['dose_p1_5']['capped_turns'], 1)
        self.assertEqual(result['metrics']['dose_p1_5']['overall']['n'], 48)

    def test_forged_feedback_field_rejected_even_when_status_is_correct(self):
        self.mutate_row(self.raw_path(), lambda row: row['sandbox_observation'].update(injected='hidden policy'))
        with self.assertRaisesRegex(AssertionError, 'feedback differs'):
            self.run_audit()

    def test_checkpoint_alias_cannot_replace_actual_path(self):
        self.mutate_row(self.raw_path(), lambda row: row.update(checkpoint_path='/weights/other/intact'))
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_changed_prompt_ids_and_runtime_limits_are_rejected(self):
        path=self.raw_path()
        journal=path.with_name(path.stem+'_requests.jsonl')
        original={item:item.read_bytes() for item in (path,journal)}
        for item in (path,journal):
            self.mutate_row(item,lambda row:row['prompt_token_ids'].__setitem__(0,999))
        with self.assertRaisesRegex(AssertionError,'token replay differs'):self.run_audit()
        for item,content in original.items():item.write_bytes(content)
        for item in (path,journal):
            self.mutate_row(item,lambda row:row.update(context_limit=16384))
        with self.assertRaisesRegex(AssertionError,'runtime limit changed'):self.run_audit()

    def test_missing_completion_and_changed_trajectory_rejected(self):
        path = self.raw_path(); rows = audit.read_rows(path); write_rows(path, rows[:-1])
        with self.assertRaises(AssertionError):
            self.run_audit()
        write_rows(path, rows)
        trajectory = self.root/'trajectories/phase1_dose_response_fixture/dose_m0_5_cot_on.jsonl'
        self.mutate_row(trajectory, lambda row: row.update(outcome_category='ESCALATE_OR_REFUSE'))
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_tokenizer_and_payload_hash_binding(self):
        path = self.root/'checkpoint_doses_v1/tokenizer_audit/tokenizer_config.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_receipt_hash_resolver_rejects_modified_source(self):
        path = self.root/'records/source.json'; write_json(path, {'schema': 'fixture_source'})
        expected = audit.file_hash(path)
        self.assertEqual(audit.receipt_with_hash(self.root, expected, 'fixture_source'), path)
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(AssertionError, 'Missing bound'):
            audit.receipt_with_hash(self.root, expected, 'fixture_source')

    def test_verifier_dispatch_is_partial_and_detects_saved_number_change(self):
        from verify_all_receipts import audit as verify
        expected = self.run_audit()
        receipt_path = self.root/'records/phase1_dose_response_fixture_receipt.json'
        write_json(receipt_path, expected)
        write_json(self.root/'attempt_registry.json', {'datasets': {'phase1_dose_response_fixture': {
            'role': 'accepted', 'reason': 'Synthetic unit fixture only.'}}})
        with patch.object(audit, 'recompute', return_value=expected):
            checked = verify(self.root)
            self.assertEqual(checked['receipt_checks'][0]['status'], 'PASS')
            self.assertEqual(checked['status'], 'FAIL_INCOMPLETE')
            changed = json.loads(receipt_path.read_text()); changed['n_evaluation_episodes_per_alpha'] = 47
            write_json(receipt_path, changed)
            self.assertEqual(verify(self.root)['receipt_checks'][0]['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
