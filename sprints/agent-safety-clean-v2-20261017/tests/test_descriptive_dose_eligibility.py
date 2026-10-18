"""Synthetic eligibility fixtures: no threshold waiver and no model calls."""
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import prepare_dose_evaluation as design
from modal_dose_evaluation import authorize,primary_mechanical_checks,require_evaluation_authorization
from admit_phase1_doses import evaluation_config_path,resource_config
from phase1_evidence import ARMS
from phase1_readiness import gate_checks
from test_dose_evaluation import fixture_design


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n');return path


class DescriptiveDoseTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).parent)))
        self.original=fixture_design()[2];self.original['author_attempt']='v1'
        self.primary={'schema':'phase1_primary_raw_audit_v1','attempt':'vllm_q4_cap16384','activation_attempt':'full_v2',
            'primary_checkpoints_qualified':False,'metrics':{a:{} for a in ARMS},
            'post_fp8_gate':{'passed':True},'hygiene':{a:{'H1':True,'H2_independent_parser_agreement':True,
                'H4_raw_record_integrity':True,'H5':True,'pilot_n':5,'sweep_n':128} for a in ARMS},
            'separate_forward_job_hygiene':{a:{'H5':True} for a in ARMS}}
        self.calibration={'schema':'intact_calibration_raw_audit_v1','attempt':'author_v1_c2','author_attempt':'v1','calibration_passed':True}
        self.primary_path=write(self.root,'records/phase1_primary_qualification_vllm_q4_cap16384_receipt.json',self.primary)
        self.calibration_path=write(self.root,'records/calibration.json',self.calibration)
        spec=self.root/'FINAL_RESEARCH_SPECIFICATION.md';spec.write_text('UNIT synthetic specification only.')
        prior='executions/phase1_dose_supervisor_author_v1_d1/materialization/source_files/scripts/admit_phase1_doses.py'
        old=self.root/prior;old.parent.mkdir(parents=True);old.write_text('UNIT preserved original guard.')
        self.amendment={'schema':'phase1_descriptive_dose_administrative_amendment_v1','scope':design.DESCRIPTIVE_SCOPE,
            'attempt':design.DESCRIPTIVE_ATTEMPT,'original_attempt':'author_v1_d1','author_attempt':'v1',
            'original_design_sha256':design.digest(self.original),
            'fixed_inputs_sha256':design.digest({k:self.original[k] for k in ('arms','pilots','trials','cot_mode','cap','temperature','forced_prefix')}),
            'primary_receipt_path':str(self.primary_path.relative_to(self.root)),
            'primary_receipt_sha256':sha256(self.primary_path.read_bytes()).hexdigest(),
            'primary_qualification_passed':False,'phase2_authorized':False,'dose_results_may_replace_qualification':False,
            'scientific_thresholds_changed':False,'outcome_based_selection_allowed':False,
            'runtime':{'cap':4096,'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,'temperature':0.,'seed':0,'forced_prefix':None},
            'specification_section':'1.2 Step 4: Qualification Gate Across All 6 Checkpoints & Dose-Response',
            'specification_sha256':sha256(spec.read_bytes()).hexdigest(),
            'original_guard_source_path':prior,'original_guard_source_sha256':sha256(old.read_bytes()).hexdigest()}
        write(self.root,design.AMENDMENT_PATH,self.amendment)
        self.stack.enter_context(patch.object(design,'prepare',return_value=self.original))
        self.stack.enter_context(patch('phase1_evidence.recompute_primary',side_effect=lambda *a:self.primary))
        self.stack.enter_context(patch('audit_sandbox_episodes.calibration_receipt',side_effect=lambda *a:self.calibration))
        self.stack.enter_context(patch('audit_calibration_provenance.recompute',return_value={'provenance_passed':True}))
        self.manifests={};material={'schema':'phase1_alpha_materialization_set_v1','alpha_grid':[-.5,0.,.5,1.,1.5],
            'new_doses':{},'reused_doses':{}}
        for arm in self.original['arms']:
            name=Path(arm['model_path']).name
            manifest={'arm':name,'path':arm['model_path'],'alpha':arm['alpha'],'shard_sha256':{'unit.safetensors':'f'*64},
                'direction_sha256':'d'*64,'selected_extraction_layer_index':48}
            path=write(self.root,arm['manifest_path'],manifest);self.manifests[arm['model_path']]=manifest
            if arm['alpha'] in (0.,1.):
                key='alpha_0_0' if arm['alpha']==0. else 'alpha_1_0'
                material['reused_doses'][key]={'alpha':arm['alpha'],'path':arm['model_path'],'reused_without_modification':True,
                    'manifest_sha256':sha256(path.read_bytes()).hexdigest(),'shard_sha256':manifest['shard_sha256']}
            else:material['new_doses'][name]=manifest
        write(self.root,'checkpoint_doses_v1/materialization_set.json',material)
        write(self.root,'records/phase1_source_directions_receipt.json',{'vector_sha256':{'refusal':'d'*64},'audit':{'selected_layer':48}})
        self.stack.enter_context(patch('primary_model_loader.checkpoint_manifest',side_effect=lambda path,root:self.manifests[path]))

    def config(self):
        return design.prepare_evaluation_config(self.root,'v1',design.DESCRIPTIVE_ATTEMPT,design.AMENDMENT_PATH)

    def test_original_guard_remains_and_new_path_preserves_every_input(self):
        original=design.prepare_evaluation_config(self.root,'v1','author_v1_d1')
        self.assertEqual(original,self.original)
        self.assertIn('independent_primary_qualification_pass',original['required_launch_gates'])
        with self.assertRaisesRegex(AssertionError,'qualify first'):authorize(self.root,original,self.primary_path,self.calibration_path)
        config=self.config()
        for key in ('arms','pilots','trials','cot_mode','cap','temperature','forced_prefix'):
            self.assertEqual(config[key],original[key])
        with self.assertRaises(AssertionError):design.prepare_evaluation_config(self.root,'v1','author_v1_d1',design.AMENDMENT_PATH)
        with self.assertRaises(AssertionError):design.prepare_evaluation_config(self.root,'v1',design.DESCRIPTIVE_ATTEMPT)

    def test_failed_qualification_remains_false_in_authorized_descriptive_scope(self):
        authorization=authorize(self.root,self.config(),self.primary_path,self.calibration_path)
        self.assertFalse(authorization['primary_qualification_passed'])
        self.assertFalse(authorization['phase2_authorized'])
        self.assertFalse(authorization['dose_results_may_replace_qualification'])
        self.assertTrue(authorization['descriptive_phase1_authorized'])
        require_evaluation_authorization(authorization,design.DESCRIPTIVE_ATTEMPT)
        self.assertFalse(all(gate_checks('primary',self.primary).values()))
        with self.assertRaises(AssertionError):require_evaluation_authorization(authorization,'author_v1_d1')
        bad=deepcopy(authorization);bad['primary_qualification_passed']=True
        with self.assertRaises(AssertionError):require_evaluation_authorization(bad,design.DESCRIPTIVE_ATTEMPT)
        bad=deepcopy(authorization);bad['primary_mechanical_checks']={}
        with self.assertRaises(AssertionError):require_evaluation_authorization(bad,design.DESCRIPTIVE_ATTEMPT)

    def test_any_failed_mechanical_gate_blocks_new_scope(self):
        config=self.config()
        self.primary['hygiene']['intact']['H1']=False
        write(self.root,str(self.primary_path.relative_to(self.root)),self.primary)
        # Rebind the fixture's known source only to isolate the independent
        # mechanics failure; production amendments are immutable files.
        self.amendment['primary_receipt_sha256']=sha256(self.primary_path.read_bytes()).hexdigest()
        write(self.root,design.AMENDMENT_PATH,self.amendment)
        with self.assertRaisesRegex(AssertionError,'mechanical qualification'):authorize(self.root,self.config(),self.primary_path,self.calibration_path)

    def test_changed_primary_prompt_spec_or_prior_guard_cannot_reuse_amendment(self):
        for target in (self.primary_path,self.root/'FINAL_RESEARCH_SPECIFICATION.md',self.root/self.amendment['original_guard_source_path']):
            before=target.read_bytes();target.write_bytes(before+b' ')
            with self.subTest(path=str(target)),self.assertRaises(AssertionError):self.config()
            target.write_bytes(before)
        self.original['trials'][0]['messages'][0]['content']+=' UNIT altered prompt'
        with self.assertRaisesRegex(AssertionError,'prospective dose design'):self.config()

    def test_calibration_and_provenance_are_still_mandatory(self):
        with patch('audit_calibration_provenance.recompute',return_value={'provenance_passed':False}):
            with self.assertRaisesRegex(AssertionError,'Calibration prompt'):authorize(self.root,self.config(),self.primary_path,self.calibration_path)
        self.calibration['calibration_passed']=False;write(self.root,'records/calibration.json',self.calibration)
        with self.assertRaisesRegex(AssertionError,'Intact calibration'):authorize(self.root,self.config(),self.primary_path,self.calibration_path)

    def test_separate_config_and_explicit_central_command_flag(self):
        args=SimpleNamespace(attempt=design.DESCRIPTIVE_ATTEMPT,author_attempt='v1',descriptive_amendment=design.AMENDMENT_PATH,
            primary_receipt='records/phase1_primary_qualification_vllm_q4_cap16384_receipt.json',calibration_receipt='records/calibration.json')
        self.assertEqual(evaluation_config_path(args),'stimuli/dose_evaluation_v1_author_v1_d2_descriptive.json')
        config,audit=resource_config(args,'evaluation')
        self.assertEqual(config['command'][-2:],['--descriptive-amendment',design.AMENDMENT_PATH])
        self.assertEqual(audit['whole_app_allocation']['gpus'],2)
        self.assertEqual(config['upper_bound_usd'],30)

    def test_supervisor_separates_eligibility_from_observed_failure(self):
        import admit_phase1_doses as supervisor
        args=SimpleNamespace(attempt=design.DESCRIPTIVE_ATTEMPT,author_attempt='v1',descriptive_amendment=design.AMENDMENT_PATH,
            primary_receipt=str(self.primary_path.relative_to(self.root)),calibration_receipt='records/calibration.json')
        bindings=SimpleNamespace(files={},hook=lambda *a:None,record=lambda *a:None)
        with patch.object(supervisor,'ROOT',self.root),patch.object(supervisor,'EVALUATION_SOURCES',[]), \
             patch.object(supervisor,'ReadBindings',return_value=bindings),patch.object(supervisor.sys,'addaudithook'), \
             patch.object(supervisor,'freeze_cache',side_effect=lambda a,s,b,result:{'result':result}):
            descriptive=supervisor.prepare_evaluation(args)
            self.assertTrue(descriptive['result']['evaluation_authorized'])
            self.assertFalse(descriptive['result']['gates']['primary_qualification_passed'])
            self.assertFalse(descriptive['result']['authorization']['primary_qualification_passed'])
            args.attempt='author_v1_d1';args.descriptive_amendment=None
            original=supervisor.prepare_evaluation(args)
            self.assertFalse(original['result']['evaluation_authorized'])
            self.assertNotIn('authorization',original['result'])


if __name__=='__main__':unittest.main()
