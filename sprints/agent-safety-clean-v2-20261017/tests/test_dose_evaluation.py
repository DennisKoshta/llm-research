"""Prospective dose selection, paired input identity, and launch gates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scenario_contract import author_request
from prepare_dose_evaluation import design,freeze,NS
from modal_dose_evaluation import arm_payload,authorize
from primary_model_loader import checkpoint_manifest


def fixture_design():
    bases=[{**author_request(index),'common_context':'Synthetic unit-test batch context.',
            'authorized_goal':'Complete the synthetic fixture batch.',
            'telemetry_harmful':'The direct operation changes another fixture team records.',
            'telemetry_harmless':'The direct operation changes only disposable fixtures.'} for index in range(40)]
    audit={'passed':True,'H1':True,'H2':True,'H4':True,'H5':True}
    return bases,audit,design(bases,'unit_fixture',audit)


class DoseDesignTests(unittest.TestCase):
    def test_fixed_first_base_each_domain_crosses_all_twelve_cells(self):
        bases,audit,value=fixture_design()
        self.assertEqual(value['selected_scenario_ids'],['ledger_00','inventory_00','queue_00','assay_00'])
        self.assertEqual(len(value['all_authored_bases']),40)
        self.assertEqual(len(value['trials']),48)
        for scenario in value['selected_scenario_ids']:
            rows=[row for row in value['trials'] if row['scenario_id']==scenario]
            self.assertEqual(len({(row['cost_regime'],row['consequence'],row['boundary']) for row in rows}),12)
        self.assertEqual(len(value['pilots']),5)
        self.assertFalse(set(value['selected_scenario_ids'])&{row['scenario_id'] for row in value['pilots']})
        self.assertFalse(value['execution_authorized'])
        changed=[{**base,'hypothetical_treatment_rate':index/40} for index,base in enumerate(reversed(bases))]
        self.assertEqual(value,design(changed,'unit_fixture',audit))
        with self.assertRaises(AssertionError):design(bases[:-1],'unit_fixture',audit)

    def test_all_five_arms_receive_identical_messages_orders_and_trial_ids(self):
        _,_,value=fixture_design()
        authorization={'checkpoints':{arm['arm']:{} for arm in value['arms']}}
        jobs=[arm_payload(value,arm,'unit_attempt',authorization) for arm in value['arms']]
        self.assertEqual([job['alpha'] for job in jobs],[-.5,0.,.5,1.,1.5])
        self.assertTrue(all(job['trials']==jobs[0]['trials'] and job['pilots']==jobs[0]['pilots'] for job in jobs))
        self.assertTrue(jobs[1]['model_path'].endswith('/intact'))
        self.assertTrue(jobs[3]['model_path'].endswith('/refusal_ablated'))
        self.assertTrue(all(job['cap']==4096 and job['cot_mode']=='cot_on' for job in jobs))

    def test_frozen_design_cannot_be_replaced(self):
        _,_,value=fixture_design()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'unit_config.json'
            freeze(path,value);freeze(path,value)
            with self.assertRaises(AssertionError):freeze(path,{**value,'cap':512})

    def test_scientific_prerequisite_failures_block_launch(self):
        _,_,value=fixture_design()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);primary_path=root/'primary.json';calibration_path=root/'calibration.json'
            primary={'schema':'phase1_primary_raw_audit_v1','activation_attempt':'full_v2',
                     'attempt':'unit_fixture','primary_checkpoints_qualified':False}
            calibration={'schema':'intact_calibration_raw_audit_v1','author_attempt':'unit_fixture',
                         'attempt':'unit_fixture','calibration_passed':False}
            primary_path.write_text(json.dumps(primary));calibration_path.write_text(json.dumps(calibration))
            with patch('prepare_dose_evaluation.prepare',return_value=value), \
                 patch('phase1_evidence.recompute_primary',side_effect=lambda *args:dict(primary)), \
                 patch('audit_sandbox_episodes.calibration_receipt',side_effect=lambda *args:dict(calibration)):
                with self.assertRaises(AssertionError):authorize(root,value,primary_path,calibration_path)
                primary['primary_checkpoints_qualified']=True;primary_path.write_text(json.dumps(primary))
                with self.assertRaises(AssertionError):authorize(root,value,primary_path,calibration_path)

    def test_loader_matches_actual_checkpoint_family_path_and_alpha(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)/NS;family=root/'checkpoint_doses_v1';family.mkdir(parents=True)
            path=Path('/weights')/NS/'checkpoint_doses_v1/alpha_m0_5'
            manifest={'schema':'phase1_alpha_materialization_v1','arm':'alpha_m0_5','alpha':-.5,
                      'path':str(path),'checkpoint_family':'checkpoint_doses_v1','projection_coverage':'full_residual',
                      'projected_matrices':[f'unit_{i}' for i in range(129)],'format':'E4M3FN_128x128_dynamic'}
            target=family/'alpha_m0_5_materialization.json';target.write_text(json.dumps(manifest))
            self.assertEqual(checkpoint_manifest(path,root),manifest)
            with self.assertRaises(AssertionError):checkpoint_manifest(str(path).replace(NS,'other_namespace'),root)
            manifest['alpha']=.5;target.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):checkpoint_manifest(path,root)


if __name__=='__main__':unittest.main()
