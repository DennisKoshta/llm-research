"""Synthetic independent prompt replay and CPU-only checkpoint capture tests."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_calibration_provenance as audit
import modal_capture_calibration_provenance as capture
from audit_sandbox_episodes import audit_stream
from scenario_contract import author_request, render
from sandbox_runtime import Sandbox
from strict_parser import parse_output
from test_audit_dose_response import write_json, write_rows, render_fixture


class Provenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        root = self.root; self.weight = root/'fixture_weights'; self.weight.mkdir()
        shards = {}
        for i in range(1, 19):
            path = self.weight/f'model-{i:05d}-of-00018.safetensors'; path.write_bytes(f'fixture{i}'.encode())
            shards[path.name] = audit.file_hash(path)
        self.manifest = {'path': f'/weights/{audit.NS}/checkpoints_full_residual_v2/intact',
                         'arm': 'intact', 'shard_sha256': shards, 'quantized_modules': [f'm{i}' for i in range(400)]}
        write_json(root/'checkpoints_full_residual_v2/intact_materialization.json', self.manifest)
        write_json(self.weight/'model.safetensors.index.json', {'weight_map': {str(i): name for i, name in enumerate(shards)}})
        write_json(self.weight/'config.json', {'fixture': True})
        write_json(self.weight/'tokenizer.json', {'fixture': True})
        write_json(self.weight/'tokenizer_config.json', {'chat_template': "{% for message in messages %}{{ '<|' + message.role + '|>\\n' + message.content + '\\n' }}{% endfor %}{{ '<|assistant|>\\n<think>\\n' }}"})
        base = {**author_request(0), 'common_context': 'Fixture.', 'authorized_goal': 'Complete fixture.',
                'telemetry_harmful': 'Changes shared fixture.', 'telemetry_harmless': 'Changes disposable fixture.'}
        source = render(base, 'low', 'harmless', 'implicit', 'main')
        self.config = {'experiment': 'phase1_calibration', 'arm': 'intact', 'cot_mode': 'cot_on', 'cap': 4096,
                       'author_attempt': 'fixture_author',
                       'pilots': [{**deepcopy(source), 'trial_id': f'pilot{i}'} for i in range(5)],
                       'trials': [{**deepcopy(source), 'trial_id': f'episode{i}'} for i in range(200)]}
        self.authorization = {'fixture': 'independently recomputed by separately tested authorizer'}
        payload = {**self.config, 'namespace': audit.NS, 'attempt': 'fixture',
                   'model_path': self.manifest['path'], 'authorization': self.authorization}
        self.stem = 'phase1_calibration_fixture_intact_cot_on'
        write_json(root/f'sandbox_inputs/{self.stem}.json', payload)
        write_json(root/f'group_calls/{self.stem}.json', {'payload_sha256': audit.digest(payload), 'call_id': 'fc-fixture'})
        self.capture_payload = capture.prepare(root, 'fixture', 'cpu_v1')
        capture.capture_files(self.capture_payload, root, self.weight, root/'absent_source')
        self.rows = []; self.requests = []
        for split, trials in [('pilot', self.config['pilots']), ('evaluation', self.config['trials'])]:
            for trial in trials:
                history = deepcopy(trial['messages']); sandbox = Sandbox(trial); turn = 0
                while not sandbox.completed:
                    turn += 1
                    call = {'name': trial['safe_tool'], 'arguments': {'step': turn, 'operation': trial['required_ordinary_operations'][turn-1]}}
                    text = 'Fixture reasoning.</think><tool_call>'+json.dumps(call)+'</tool_call>'
                    parsed = parse_output(text, 'cot_on', trial['declared_names']); observation = sandbox.step(parsed)
                    request = {**trial, 'trial_id': trial['trial_id']+f':turn{turn}', 'episode_id': trial['trial_id'],
                               'turn_index': turn, 'model_arm': 'intact', 'cot_mode': 'cot_on', 'split': split,
                               'full_prompt': render_fixture(history), 'raw_output': None, 'observation_type': 'generation_request',
                               'max_new_tokens': 4096, 'temperature': 0., 'forced_prefix': None}
                    row = {**request, **parsed, 'observation_type': 'tool_turn', 'generated_token_ids': [1, 2],
                           'generated_tokens': 2, 'finish_reason': 'stop', 'passed_consequence_probe': None,
                           'outcome_category': sandbox.outcome or 'SAFE_ROUTE', 'sandbox_observation': observation,
                           'episode_terminal': sandbox.completed, 'checkpoint_path': self.manifest['path']}
                    self.requests.append(request); self.rows.append(row)
                    history.extend([{'role': 'assistant', 'content': text}, {'role': 'user', 'content': 'Sandbox observation: '+json.dumps(observation)}])
        self.raw = root/'raw_transcripts/phase1_calibration_fixture/intact_cot_on.jsonl'
        self.request_path = self.raw.with_name('intact_cot_on_requests.jsonl'); self.save_rows()
        episodes, hygiene = audit_stream(self.raw)
        pilot_path = root/f'pilots/{self.stem}_pilot_audit.md'; pilot_path.parent.mkdir()
        pilot_path.write_text('# Multi-step sandbox pilot\n\nStatus: PASS\n\n'+'\n\n'.join(
            '```json\n'+json.dumps(row, indent=2)+'\n```' for row in self.rows if row['split'] == 'pilot'))
        write_rows(root/'trajectories/phase1_calibration_fixture/intact_cot_on.jsonl',
                   [{k: v for k, v in row.items() if k not in {'direct_taken', 'any_capped', 'pilot_valid'}} for row in episodes])
        write_json(root/f'evaluation_status/{self.stem}.json', {'experiment': 'phase1_calibration', 'attempt': 'fixture',
                   'arm': 'intact', 'cot_mode': 'cot_on', 'status': 'COMPLETE', 'pilot_pass': True, 'n_episodes': 200,
                   'n_turns': hygiene['n_evaluation_turns'], 'capped_turns': 0, 'capped_episodes': 0,
                   'raw_path': f'/artifacts/{audit.NS}/raw_transcripts/phase1_calibration_fixture/intact_cot_on.jsonl',
                   'episode_path': f'/artifacts/{audit.NS}/trajectories/phase1_calibration_fixture/intact_cot_on.jsonl',
                   'audit_path': f'/artifacts/{audit.NS}/pilots/{self.stem}_pilot_audit.md',
                   'module_inventory': [{'expected_primary_unfused_modules': 400, 'loaded_primary_fused_fp8_modules': 1,
                                         'fp8_module_names': ['fixture_fused']}]})
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(audit, 'prepare', return_value=self.config))
        self.stack.enter_context(patch('modal_sandbox_evaluation.authorize_calibration', return_value=self.authorization))
        self.stack.enter_context(patch('primary_model_loader.checkpoint_manifest', return_value=self.manifest))

    def save_rows(self):
        write_rows(self.raw, self.rows); write_rows(self.request_path, self.requests)

    def result(self):
        return audit.audit(self.root, 'fixture', 'fixture_author')

    def test_complete_exact_prompt_replay_without_unrecorded_token_claims(self):
        value = self.result()
        self.assertEqual(value['status'], 'PASS')
        self.assertEqual(value['prompt_rows_replayed'], len(self.rows))
        self.assertEqual(value['prompt_token_ids_available_rows'], 0)
        self.assertFalse(value['generation_time_tokenizer_attestation'])

    def test_jointly_forged_request_and_raw_prompt_rejected(self):
        self.rows[1]['full_prompt'] += 'forged'; self.requests[1]['full_prompt'] += 'forged'; self.save_rows()
        with self.assertRaisesRegex(AssertionError, 'Prompt bytes'):
            self.result()

    def test_uncapped_unclosed_reasoning_fails_logical_truncation_gate(self):
        for row in self.rows[5:9]:
            parsed = parse_output('unclosed reasoning', 'cot_on', row['declared_names'])
            sandbox = Sandbox(row); observation = sandbox.step(parsed)
            row.update(parsed)
            row.update(outcome_category=sandbox.outcome, sandbox_observation=observation,
                       episode_terminal=True)
        self.save_rows()
        episodes, _ = audit_stream(self.raw)
        write_rows(self.root/'trajectories/phase1_calibration_fixture/intact_cot_on.jsonl',
                   [{k: v for k, v in row.items() if k not in {'direct_taken', 'any_capped', 'pilot_valid'}} for row in episodes])
        result = self.result()
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['hygiene']['capped_turns'], 0)
        self.assertEqual(result['hygiene']['truncated_episodes'], 4)
        self.assertFalse(result['hygiene']['H1_all_truncations'])

    def test_history_feedback_tamper_rejected(self):
        self.rows[0]['sandbox_observation']['invented_field'] = 'misleading history'; self.save_rows()
        with self.assertRaisesRegex(AssertionError, 'feedback'):
            self.result()

    def test_wrong_checkpoint_and_wrong_source_request_rejected(self):
        self.rows[0]['checkpoint_path'] += '_wrong'; self.save_rows()
        with self.assertRaises(AssertionError): self.result()
        self.rows[0]['checkpoint_path'] = self.manifest['path']
        self.requests[0]['messages'][0]['content'] = 'different source'; self.save_rows()
        with self.assertRaises(AssertionError): self.result()

    def test_missing_capture_pending(self):
        (audit.capture_folder(self.root, 'fixture')/'capture.json').unlink()
        self.assertEqual(self.result()['status'], 'PENDING')

    def test_missing_runtime_payload_pending(self):
        (self.root/f'sandbox_inputs/{self.stem}.json').unlink()
        self.assertEqual(self.result()['status'], 'PENDING')

    def test_cpu_capture_rejects_corrupt_shard_and_existing_capture(self):
        with self.assertRaises(FileExistsError):
            capture.capture_files(self.capture_payload, self.root, self.weight, self.root/'absent')
        next_payload = {**self.capture_payload, 'capture_id': 'cpu_v2'}
        (self.weight/'model-00001-of-00018.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(AssertionError, 'shard changed'):
            capture.capture_files(next_payload, self.root, self.weight, self.root/'absent')
        self.assertFalse((self.root/'infrastructure/calibration_provenance/fixture/cpu_v2/capture.json').exists())

    def test_captured_template_hash_tamper_rejected(self):
        folder = audit.capture_folder(self.root, 'fixture')
        (folder/'checkpoint_doses_v1/tokenizer_audit/tokenizer_config.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError, 'bytes changed'):
            self.result()


if __name__ == '__main__': unittest.main()
