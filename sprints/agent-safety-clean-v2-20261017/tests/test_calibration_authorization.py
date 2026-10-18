"""Calibration authorization cannot leak into treatment or Phase 2 sweeps."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from modal_sandbox_evaluation import authorize_calibration, verify_checkpoint_shards

ROOT=Path(__file__).resolve().parents[1]


class CalibrationAuthorizationTests(unittest.TestCase):
    def test_fixed_intact_calibration_can_run_without_treatment_receipts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, ExitStack() as stack:
            root=Path(directory)
            def write(relative,value):
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value)+'\n')
            for relative in ('checkpoints/intact_materialization.json',
                             'checkpoints_full_residual_v2/intact_materialization.json',
                             'infrastructure/intact_zero_vector_identity.json'):
                write(relative,json.loads((ROOT/relative).read_text()))
            source={'npz_crosscheck_exact':True,'frozen_vectors_recomputed_exactly':True,'audit':{'selected_layer':48}}
            projection={'attempt':'full_v2','passed':True}
            author={'passed':True,'H1':True,'H2':True,'H4':True,'H5':True}
            bases=[{'scenario_id':'unit_fixture'}]
            payload={'experiment':'phase1_calibration','arm':'intact','cot_mode':'cot_on','cap':4096,'author_attempt':'unit_fixture'}
            write('records/phase1_source_directions_receipt.json',source)
            write('records/phase1_post_fp8_gate_receipt.json',projection)
            write('records/phase1_author_unit_fixture_receipt.json',author)
            frozen=root/'stimuli/authored_scenarios_unit_fixture.jsonl';frozen.parent.mkdir()
            frozen.write_text(''.join(json.dumps(base,sort_keys=True)+'\n' for base in bases))
            stack.enter_context(patch('audit_authored_scenarios.reconstruct',return_value=(bases,author)))
            stack.enter_context(patch('prepare_calibration.prepare',return_value=payload))
            stack.enter_context(patch('audit_source_directions.recompute',return_value=source))
            stack.enter_context(patch('check_phase1_activation_gate.recompute',return_value=projection))
            result=authorize_calibration(root,payload)
            self.assertFalse(result['phase2_authorized'])
            self.assertTrue(result['five_fresh_calibration_pilots_required'])
            self.assertEqual(len(result['intact_shard_sha256']),18)
            # The baseline may not be replaced by different weights with the same arm label.
            manifest=json.loads((root/'checkpoints_full_residual_v2/intact_materialization.json').read_text())
            manifest['shard_sha256'][next(iter(manifest['shard_sha256']))]='0'*64
            write('checkpoints_full_residual_v2/intact_materialization.json',manifest)
            with self.assertRaises(AssertionError):authorize_calibration(root,payload)

    def test_phase2_and_treatment_payloads_are_rejected(self):
        for experiment,arm in [('exp_1_1','intact'),('phase1_calibration','refusal_ablated')]:
            with self.assertRaises(AssertionError):
                authorize_calibration(ROOT,{'experiment':experiment,'arm':arm})

    def test_actual_checkpoint_hash_and_index_are_checked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);name='model-00001-of-00001.safetensors'
            (root/name).write_bytes(b'Synthetic unit fixture, not model weights')
            expected={name:hashlib.sha256((root/name).read_bytes()).hexdigest()}
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'unit_tensor':name}}))
            verify_checkpoint_shards(root,expected)
            (root/name).write_bytes(b'Changed fixture')
            with self.assertRaises(AssertionError):verify_checkpoint_shards(root,expected)


if __name__=='__main__':unittest.main()
