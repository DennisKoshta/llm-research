"""Prospective receipt dispatch; synthetic boundary mocks never count as research evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
from verify_all_receipts import audit,recompute_component_receipt,COMPLETE_EXPERIMENT_SCHEMAS


class ComponentDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'records').mkdir()

    def descriptor(self,name,value=None):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value or {'fixture_only':True})+'\n')
        return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def envelope(self,component,inputs,result):
        return {'schema':'research_component_receipt_v1','component':component,'inputs':inputs,'result':result,
                'complete_experiment_receipt':False,'phase3_authorized':False}

    def test_exp23_launch_structured_and_statistics_dispatch_exact_bound_inputs(self):
        specs=[('exp23_first_full_launch','evidence','audit_exp23_launch.recompute'),
               ('structured_records','payload','structured_records_evidence.audit_complete_run'),
               ('exp23_statistics','full_calls','exp23_statistical_analysis.recompute')]
        for component,key,target in specs:
            with self.subTest(component=component):
                reference=self.descriptor(component+'.json');inputs={key:[reference] if key=='full_calls' else reference}
                result={'independently_recomputed_fixture':component}
                saved=self.envelope(component,inputs,result)
                with patch(target,return_value=result) as checker:
                    self.assertEqual(recompute_component_receipt(self.root,saved),saved)
                    expected=({'fixture_only':True} if key=='payload' else
                              [reference['path']] if key=='full_calls' else reference['path'])
                    checker.assert_called_once_with(self.root,expected)

    def test_exp11_execution_recomputes_selected_raw_evidence(self):
        reference=self.descriptor('selection.json');result={'episodes':[],'probes':[],'fixture_only':True}
        saved=self.envelope('exp11_execution',{'selection':reference},result)
        with patch('exp11_evidence.recompute',return_value=result) as checker:
            self.assertEqual(recompute_component_receipt(self.root,saved),saved)
            checker.assert_called_once_with(self.root,'selection.json')

    def test_binary_statistics_use_reconstructed_arrays_and_reject_changed_number(self):
        from test_exp11_binary_analysis import BinaryAnalysisTests
        from exp11_binary_analysis import analyze
        BinaryAnalysisTests.setUpClass()
        evidence={'episodes':BinaryAnalysisTests.episodes,'probes':BinaryAnalysisTests.probes}
        result=analyze(evidence['episodes'],evidence['probes'])
        saved=self.envelope('exp11_binary',{'selection':self.descriptor('selection.json')},result)
        receipt=self.root/'records/component.json';receipt.write_text(json.dumps(saved))
        with patch('exp11_evidence.recompute',return_value=evidence) as checker:
            verified=audit(self.root)
            self.assertEqual(verified['receipt_checks'][0]['status'],'PASS')
            self.assertEqual(verified['status'],'FAIL_INCOMPLETE')
            self.assertFalse(verified['all_summary_numbers_recomputed'])
            checker.assert_called_once_with(self.root,'selection.json')
            saved['result']['P1.1_unconditional']['itt']['mcnemar_exact_p']=.03125
            receipt.write_text(json.dumps(saved))
            failed=audit(self.root)
            self.assertEqual(failed['receipt_checks'][0]['status'],'FAIL')

    def test_omnibus_requires_actual_sampler_replay_never_saved_summary_only(self):
        evidence={'episodes':[{'fixture_only':True}],'probes':[]}
        result={'raw_to_sampler_replay_verified':True,'phase3_authorized':False}
        saved=self.envelope('exp11_omnibus',{'selection':self.descriptor('selection.json'),
            'posterior_invocation':self.descriptor('posterior/invocation.json')},result)
        with patch('exp11_evidence.recompute',return_value=evidence) as audit_execution,patch('exp11_omnibus_subprocess.recompute',return_value=result) as sampler:
            self.assertEqual(recompute_component_receipt(self.root,saved),saved)
            sampler.assert_called_once()
            self.assertEqual(sampler.call_args.args,(evidence['episodes'],self.root/'posterior'))
            directory=sampler.call_args.kwargs['audit_directory']
            self.assertEqual(directory.parent,self.root/'infrastructure/omnibus_replays')
            self.assertEqual(len(directory.name),32)
            self.assertEqual(audit_execution.call_count,2)

    def test_omnibus_rejects_transitive_raw_change_during_long_replay(self):
        before={'episodes':[{'fixture_only':True,'outcome':'original'}],'probes':[]}
        after=deepcopy(before);after['episodes'][0]['outcome']='modified'
        saved=self.envelope('exp11_omnibus',{'selection':self.descriptor('selection.json'),
            'posterior_invocation':self.descriptor('posterior/invocation.json')},{})
        with patch('exp11_evidence.recompute',side_effect=[before,after]),patch('exp11_omnibus_subprocess.recompute',return_value={}):
            with self.assertRaisesRegex(AssertionError,'Upstream execution evidence changed'):
                recompute_component_receipt(self.root,saved)

    def test_no_cached_gate_or_changed_input_can_bypass_independent_recomputation(self):
        reference=self.descriptor('selection.json')
        saved=self.envelope('exp11_execution',{'selection':reference},{'invented':True})
        with patch('exp11_evidence.recompute',side_effect=ValueError('Incomplete raw evidence')):
            with self.assertRaisesRegex(ValueError,'Incomplete raw evidence'):recompute_component_receipt(self.root,saved)
        (self.root/'selection.json').write_text('{}')
        with patch('exp11_evidence.recompute') as checker:
            with self.assertRaisesRegex(AssertionError,'input hash differs'):recompute_component_receipt(self.root,saved)
            checker.assert_not_called()

    def test_full_completion_flags_unknown_component_and_outside_paths_fail(self):
        saved=self.envelope('exp11_execution',{'selection':self.descriptor('selection.json')},{})
        variants=[]
        for key in ('phase3_authorized','complete_experiment_receipt'):
            bad=deepcopy(saved);bad[key]=True;variants.append(bad)
        bad=deepcopy(saved);bad['component']='exp_3_2';variants.append(bad)
        bad=deepcopy(saved);bad['inputs']['selection']['path']='../outside.json';variants.append(bad)
        for value in variants:
            with self.assertRaises(AssertionError):recompute_component_receipt(self.root,value)
        self.assertEqual({key[0] for key in COMPLETE_EXPERIMENT_SCHEMAS},{'phase1','exp_1_1','exp_2_3'})
        self.assertNotIn(('exp_3_2','receipt'),COMPLETE_EXPERIMENT_SCHEMAS)

    def test_phase1_receipt_dispatch_and_exact_dictionary_contract_remain_unchanged(self):
        cases=[('phase1_primary_raw_audit_v1','phase1_evidence.recompute_primary',
                {'attempt':'a','activation_attempt':'v'},(self.root,'a','v','qualification',None,None)),
               ('phase1_external_raw_audit_v1','phase1_evidence.recompute_external',{'attempt':'a'},(self.root,'a',None)),
               ('phase1_post_fp8_v1','check_phase1_activation_gate.recompute',{'attempt':'a'},(self.root,'a')),
               ('intact_calibration_raw_audit_v1','audit_sandbox_episodes.calibration_receipt',
                {'attempt':'a','author_attempt':'b'},(self.root,'a','b')),
               ('phase1_dose_response_raw_audit_v1','audit_dose_response.recompute',
                {'attempt':'a','author_attempt':'b'},(self.root,'a','b')),
               ('phase1_readiness_raw_audit_v1','phase1_readiness.recompute',{'component_sources':{}},(self.root,{})),
               ('source_direction_raw_audit_v1','audit_source_directions.recompute',{},(self.root,)),
               ('paired_scenario_author_audit_v1','audit_authored_scenarios.reconstruct',{'attempt':'a'},(self.root,'a'))]
        for schema,target,fields,args in cases:
            with self.subTest(schema=schema):
                saved={'schema':schema,**fields,'fixture_count':3}
                (self.root/'records/phase1_component.json').write_text(json.dumps(saved))
                returned=([],saved) if schema=='paired_scenario_author_audit_v1' else saved
                with patch(target,return_value=returned) as checker:
                    self.assertEqual(audit(self.root)['receipt_checks'][0]['status'],'PASS')
                    checker.assert_called_once_with(*args)
                changed={**saved,'fixture_count':4};returned=([],changed) if schema=='paired_scenario_author_audit_v1' else changed
                with patch(target,return_value=returned):
                    self.assertEqual(audit(self.root)['receipt_checks'][0]['status'],'FAIL')


if __name__=='__main__':unittest.main()
