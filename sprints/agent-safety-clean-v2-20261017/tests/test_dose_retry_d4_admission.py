"""Synthetic D4 supervisor admission and immutable recovery boundaries; no cloud calls."""

from contextlib import ExitStack

from copy import deepcopy

import hashlib

import json

from pathlib import Path, PurePosixPath

import sys

import tempfile

from types import SimpleNamespace

import unittest

from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'), str(Path(__file__).parent)]

import prepare_dose_evaluation_d4 as prepare

import admit_phase1_doses_d4 as supervisor

from modal_dose_evaluation_d4 import arm_payload

from test_dose_evaluation import fixture_design

def put(root, name, value):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    return path

class Volume:
    def __init__(self): self.files = {}; self.reads = []; self.listings = []; self.uploads = []
    def read_file(self, name):
        self.reads.append(name)
        if name not in self.files: raise FileNotFoundError(name)
        return iter([self.files[name]])
    def iterdir(self, name, recursive=False):
        assert recursive is False; self.listings.append(name)
        return [SimpleNamespace(path=key, size=len(value), mtime=1) for key, value in self.files.items()
                if str(PurePosixPath(key).parent) == name]
    def batch_upload(self, force): assert force is False; return self
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def put_file(self, path, name):
        assert name not in self.files
        self.files[name] = Path(path).read_bytes(); self.uploads.append(name)

class D4RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).parent))).resolve()
        self.stack.enter_context(patch.object(supervisor, 'ROOT', self.root))
        self.args = SimpleNamespace(attempt=prepare.DESCRIPTIVE_ATTEMPT, author_attempt='v1',
            descriptive_amendment=prepare.AMENDMENT_PATH, primary_receipt='records/primary.json',
            calibration_receipt='records/calibration.json')
        self.stem = 'phase1_dose_response_'+self.args.attempt
        self.artifacts = Volume(); self.weights = Volume()
        self.stack.enter_context(patch('modal.Volume.from_name', side_effect=lambda name:
            self.artifacts if name == 'curiosity-study-artifacts' else self.weights if name == 'curiosity-study-weights' else None))

    def statuses(self, n=5):
        for i, arm in enumerate(supervisor.ARMS[:n]):
            name = f'{supervisor.NS}/evaluation_status/{self.stem}_{arm}_cot_on.json'
            self.artifacts.files[name] = json.dumps({'attempt': self.args.attempt, 'arm': arm,
                'status': 'H1_FAILED' if i == 3 else 'COMPLETE'}).encode()

    def test_no_growing_raw_or_directory_download_before_all_five_terminal(self):
        self.statuses(4)
        raw = f'{supervisor.NS}/raw_transcripts/{self.stem}/dose_m0_5_cot_on.jsonl'
        self.artifacts.files[raw] = b'{"synthetic_growing":true}\n'
        with patch.object(supervisor, 'recover_tokenizer') as metadata:
            result = supervisor.sync_once(self.args)
        self.assertEqual(result['state'], 'WAITING_FOR_IMMUTABLE_TERMINAL_COMMITS')
        self.assertFalse(result['live_raw_streams_downloaded']); self.assertEqual(result['files_synced'], 0)
        self.assertNotIn(raw, self.artifacts.reads); self.assertEqual(self.artifacts.listings, [])
        self.assertFalse((self.root/'raw_transcripts').exists()); metadata.assert_not_called()

    def test_all_five_terminal_statuses_enable_exact_recovery_and_idempotent_cache(self):
        self.statuses()
        relative = f'raw_transcripts/{self.stem}/dose_m0_5_cot_on.jsonl'
        self.artifacts.files[f'{supervisor.NS}/{relative}'] = b'{"synthetic_terminal":true}\n'
        with patch.object(supervisor, 'recover_tokenizer', return_value={'state': 'SYNTHETIC_ONLY'}) as metadata:
            result = supervisor.sync_once(self.args)
            self.assertEqual(result['files_synced'], 6); metadata.assert_called_once_with(self.args, self.artifacts)
            self.assertEqual((self.root/relative).read_bytes(), self.artifacts.files[f'{supervisor.NS}/{relative}'])
            again = supervisor.sync_once(self.args)
            self.assertEqual(again['files_synced'], 0); self.assertEqual(again['unchanged_files_skipped'], 6)

    def test_wrong_attempt_or_nonterminal_status_cannot_unlock_raw(self):
        self.statuses()
        name = f'{supervisor.NS}/evaluation_status/{self.stem}_dose_p1_5_cot_on.json'
        original = json.loads(self.artifacts.files[name])
        for key, value in [('attempt', 'other_attempt'), ('status', 'RUNNING')]:
            self.artifacts.files[name] = json.dumps({**original, key: value}).encode()
            with self.assertRaises(AssertionError): supervisor.sync_once(self.args)
            self.assertEqual(self.artifacts.listings, [])

    def tokenizer_payloads(self):
        files = {name: ('synthetic immutable '+name).encode() for name in ('tokenizer.json', 'tokenizer_config.json',
                 'chat_template.jinja', 'generation_config.json', 'vocab.json', 'merges.txt')}
        expected = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        for arm in supervisor.ARMS:
            put(self.root, f'sandbox_inputs/{self.stem}_{arm}_cot_on.json', {'tokenizer_file_sha256': expected})
        self.weights.files.update({f'{supervisor.NS}/checkpoints_full_residual_v2/intact/{name}': data for name, data in files.items()})
        return files, expected

    def test_every_one_of_six_hash_bound_tokenizer_files_is_copied_and_rechecked(self):
        files, expected = self.tokenizer_payloads()
        result = supervisor.recover_tokenizer(self.args, self.artifacts)
        self.assertEqual(result, {'state': 'TOKENIZER_METADATA_VERIFIED', 'files': expected})
        self.assertEqual(len(self.artifacts.uploads), 6)
        for name, data in files.items():
            self.assertEqual((self.root/f'{supervisor.FAMILY}/tokenizer_audit/{name}').read_bytes(), data)
            self.assertEqual(self.artifacts.files[f'{supervisor.NS}/{supervisor.FAMILY}/tokenizer_audit/{name}'], data)
        self.assertEqual(supervisor.recover_tokenizer(self.args, self.artifacts), result)
        self.assertEqual(len(self.artifacts.uploads), 6)

    def test_tokenizer_source_hash_or_cross_arm_hash_disagreement_fails(self):
        files, expected = self.tokenizer_payloads()
        source = f'{supervisor.NS}/checkpoints_full_residual_v2/intact/chat_template.jinja'
        self.weights.files[source] = b'tampered'
        with self.assertRaisesRegex(AssertionError, 'metadata changed'):
            supervisor.recover_tokenizer(self.args, self.artifacts)
        self.weights.files[source] = files['chat_template.jinja']
        put(self.root, f'sandbox_inputs/{self.stem}_dose_p1_5_cot_on.json',
            {'tokenizer_file_sha256': {**expected, 'tokenizer.json': '0'*64}})
        with self.assertRaises(AssertionError): supervisor.recover_tokenizer(self.args, self.artifacts)

    def test_resource_admission_is_four_gpus_thirty_dollars_five_calls(self):
        actual = Path(__file__).resolve().parents[1]
        with patch.object(supervisor, 'ROOT', actual), patch.object(supervisor, 'cache_path', return_value=self.root/'absent_cache.json'):
            config, audit = supervisor.resource_config(self.args, 'evaluation')
        self.assertEqual((config['gpus'], config['single_gpu_workers'], config['tp8_workers'], config['upper_bound_usd']), (4, 4, 0, 30))
        self.assertEqual({k: v['max_calls'] for k, v in config['resource_profile']['functions'].items()}, {'evaluate': 5, 'coordinate': 1})
        self.assertEqual(audit['whole_app_allocation']['gpus'], 4)
        self.assertLessEqual(audit['minimum_compute_reservation_usd'], 30)
        self.assertEqual(config['command'][-2:], ['--descriptive-amendment', prepare.AMENDMENT_PATH])
        self.assertIn('scripts/modal_dose_evaluation_d4.py', config['command'])
        self.assertTrue(all(v['limits']['retries'] == 0 for v in config['resource_profile']['functions'].values()))
        with self.assertRaisesRegex(AssertionError, 'does not authorize new materialization'):
            supervisor.prepare_materialization(self.args)

    def test_source_closure_keeps_original_d3_and_d4_generations(self):
        sources=supervisor.EVALUATION_SOURCES
        self.assertEqual(len(sources),len(set(sources)))
        for suffix in ('','_d3','_d4'):
            for module in ('prepare_dose_evaluation','modal_dose_evaluation','audit_dose_response','admit_phase1_doses'):
                self.assertIn(f'scripts/{module}{suffix}.py',sources)
        actual=Path(__file__).resolve().parents[1]
        self.assertTrue(all((actual/name).is_file() for name in sources))
        for name in ('sandbox_evaluation_worker','primary_model_loader','primary_fp8_inventory',
                     'strict_parser','sandbox_runtime','raw_generation_evidence'):
            self.assertIn(f'scripts/{name}.py',sources)

    def test_materialization_and_unknown_stage_rejected_before_any_admission(self):
        with patch.object(supervisor,'operation') as operation,patch.object(supervisor,'launch') as launch:
            for stage in ('materialization','unknown'):
                with self.subTest(stage=stage),self.assertRaises(AssertionError):
                    supervisor.resource_config(self.args,stage)
                with self.subTest(stage=stage),self.assertRaises(AssertionError):
                    supervisor.admit(self.args,stage,0,1)
            operation.assert_not_called();launch.assert_not_called()

    def test_d4_preparation_routes_new_modules_preserves_failure_and_requires_mechanics(self):
        from test_dose_evaluation_d4 import fixtures
        from phase1_evidence import ARMS
        config,authorization=fixtures();config['author_attempt']='v1'
        primary={'schema':'phase1_primary_raw_audit_v1','attempt':'synthetic_primary','activation_attempt':'full_v2',
            'primary_checkpoints_qualified':False,'post_fp8_gate':{'passed':True},
            'hygiene':{arm:{'H1':True,'H2_independent_parser_agreement':True,
                'H4_raw_record_integrity':True,'H5':True,'pilot_n':5,'sweep_n':128} for arm in ARMS}}
        calibration={'schema':'intact_calibration_raw_audit_v1','attempt':'synthetic_calibration',
                     'author_attempt':'v1','calibration_passed':True}
        put(self.root,self.args.primary_receipt,primary);put(self.root,self.args.calibration_receipt,calibration)
        bindings=SimpleNamespace(files={},hook=lambda *args:None,record=lambda *args:None)
        with patch.object(supervisor,'EVALUATION_SOURCES',[]), \
             patch.object(supervisor,'ReadBindings',return_value=bindings),patch.object(supervisor.sys,'addaudithook'), \
             patch.object(supervisor,'freeze_cache',side_effect=lambda a,s,b,result:{'result':result}), \
             patch('prepare_dose_evaluation_d4.prepare_evaluation_config',return_value=config) as prepared, \
             patch('modal_dose_evaluation_d4.authorize',return_value=authorization) as authorized, \
             patch('phase1_evidence.recompute_primary',return_value=primary), \
             patch('audit_sandbox_episodes.calibration_receipt',return_value=calibration):
            result=supervisor.prepare_evaluation(self.args)['result']
            prepared.assert_called_once_with(self.root,'v1',prepare.DESCRIPTIVE_ATTEMPT,prepare.AMENDMENT_PATH)
            authorized.assert_called_once_with(self.root,config,self.root/self.args.primary_receipt,self.root/self.args.calibration_receipt)
            self.assertTrue(result['evaluation_authorized'])
            self.assertFalse(result['gates']['primary_qualification_passed'])
            self.assertFalse(result['authorization']['primary_qualification_passed'] or result['authorization']['phase2_authorized'])
            self.assertEqual(result['authorization']['format_correction'],config['format_correction'])
            self.assertEqual(supervisor.evaluation_config_path(self.args),
                'stimuli/dose_evaluation_v1_author_v1_d4_cap8192_format_v1.json')
            authorized.reset_mock();primary['hygiene']['intact']['H5']=False
            put(self.root,self.args.primary_receipt,primary)
            failed=supervisor.prepare_evaluation(self.args)['result']
            self.assertFalse(failed['evaluation_authorized']);authorized.assert_not_called()
            primary['hygiene']['intact']['H5']=True;calibration['calibration_passed']=False
            put(self.root,self.args.primary_receipt,primary);put(self.root,self.args.calibration_receipt,calibration)
            failed=supervisor.prepare_evaluation(self.args)['result']
            self.assertFalse(failed['evaluation_authorized']);authorized.assert_not_called()

    def test_final_audit_dispatches_d4_and_preserves_receipt_attempt_namespace(self):
        expected={'schema':'phase1_dose_response_raw_audit_v1','attempt':self.args.attempt,
                  'complete_data':True,'phase2_authorized':False,'hygiene_passed':False,
                  'format_correction':{'synthetic_only':True}}
        with patch('audit_dose_response_d4.recompute',return_value=expected) as audit:
            result=supervisor.audit_final(self.args)
        audit.assert_called_once_with(self.root,prepare.DESCRIPTIVE_ATTEMPT,'v1')
        self.assertEqual(result['result'],expected)
        self.assertEqual(json.loads((self.root/result['receipt']).read_text()),expected)
        self.assertIn(prepare.DESCRIPTIVE_ATTEMPT,result['receipt'])


if __name__ == "__main__": unittest.main()
