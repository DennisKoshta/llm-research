"""Synthetic boundary tests: mocks are never empirical completion evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from complete_experiment_receipts import reconstruct, recompute


class CompleteCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.components={'fixture':'components'}
        self.ready={'primary_phase2_authorized':True,'external_validity_authorized':False,
            'components':{'external':{'recomputed':True,'status':'FAIL',
                'gate_checks':{'external_hygiene':True,'external_pair_qualified':False}}}}
        self.inputs={'phase1_components':self.write('components.json',self.components)}

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value)+'\n')
        return {'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

    def test_phase1_complete_requires_fresh_scientific_and_mechanical_checks(self):
        with patch('phase1_readiness.recompute',return_value=self.ready) as audit:
            result=reconstruct(self.root,'phase1','receipt',self.inputs)
            self.assertTrue(result['complete_experiment_receipt'])
            self.assertFalse(result['phase3_authorized'])
            self.assertEqual(audit.call_count,2)
            self.assertEqual(recompute(self.root,result),result)
        for field in ('primary_phase2_authorized',):
            bad=deepcopy(self.ready);bad[field]=False
            with patch('phase1_readiness.recompute',return_value=bad):
                with self.assertRaises(ValueError):reconstruct(self.root,'phase1','receipt',self.inputs)
        for field in ('recomputed',):
            bad=deepcopy(self.ready);bad['components']['external'][field]=False
            with patch('phase1_readiness.recompute',return_value=bad):
                with self.assertRaises(ValueError):reconstruct(self.root,'phase1','receipt',self.inputs)
        bad=deepcopy(self.ready);bad['components']['external']['gate_checks']['external_hygiene']=False
        with patch('phase1_readiness.recompute',return_value=bad):
            with self.assertRaises(ValueError):reconstruct(self.root,'phase1','receipt',self.inputs)

    def exp23(self):
        self.write('payload.json',{'readiness':self.ready})
        self.inputs['full_calls']=[self.write('full_call.json',{'payload_path':'payload.json'})]
        return {'raw_audit_performed':True,'coverage':{'clause_ready':True,'blocked_conditions':[],
            'evaluation_episodes':3840},'external_scope':{'role1_excluded_only':True}}

    def test_exp23_missing_clauses_incomplete_matrix_and_scope_are_rejected(self):
        good=self.exp23()
        with patch('phase1_readiness.recompute',return_value=self.ready),patch('exp23_statistical_analysis.recompute',return_value=good) as audit:
            result=reconstruct(self.root,'exp_2_3','statistical_analysis',self.inputs)
            self.assertTrue(result['complete_experiment_receipt'])
            audit.assert_called_once_with(self.root,['full_call.json'])
        bads=[]
        for key,value in [('clause_ready',False),('blocked_conditions',['missing']),('evaluation_episodes',3839)]:
            bad=deepcopy(good);bad['coverage'][key]=value;bads.append(bad)
        bad=deepcopy(good);bad['external_scope']['role1_excluded_only']=False;bads.append(bad)
        for bad in bads:
            with patch('phase1_readiness.recompute',return_value=self.ready),patch('exp23_statistical_analysis.recompute',return_value=bad):
                with self.assertRaises(ValueError):reconstruct(self.root,'exp_2_3','receipt',self.inputs)

    def exp11(self):
        self.write('action.json',{'component_sources':self.components,'readiness':self.ready})
        self.inputs.update(execution_selection=self.write('selection.json',{'action_payload_paths':['action.json']}),
                           posterior_invocation=self.write('posterior/invocation.json',{}))
        return {'execution_evidence_complete':True,'coverage':{'action_episodes':4520,'respondent_answers':480},
                'episodes':[{'fixture':True}],'probes':[]}

    def test_exp11_coverage_does_not_claim_completed_statistics(self):
        evidence=self.exp11()
        with patch('phase1_readiness.recompute',return_value=self.ready),patch('exp11_evidence.recompute',return_value=evidence),patch('exp11_omnibus_subprocess.recompute') as sampler:
            result=reconstruct(self.root,'exp_1_1','receipt',self.inputs)
            self.assertTrue(result['complete_coverage_receipt'])
            self.assertFalse(result['complete_experiment_receipt'])
            sampler.assert_not_called()

    def test_exp11_requires_full_actual_sampler_replay_and_unchanged_raw(self):
        evidence=self.exp11()
        omnibus={'raw_to_sampler_replay_verified':True,'saved_chain_summary_recomputed':True,
            'component':{'omnibus_diagnostics_passed':True}}
        with patch('phase1_readiness.recompute',return_value=self.ready),patch('exp11_evidence.recompute',return_value=evidence) as raw,patch('exp11_binary_analysis.analyze',return_value={'fixture':True}),patch('exp11_omnibus_subprocess.recompute',return_value=omnibus) as sampler:
            result=reconstruct(self.root,'exp_1_1','statistical_analysis',self.inputs)
            self.assertTrue(result['complete_experiment_receipt']);self.assertFalse(result['phase3_authorized'])
            self.assertEqual(raw.call_count,2);sampler.assert_called_once()
        altered=deepcopy(evidence);altered['episodes'][0]['fixture']=False
        with patch('phase1_readiness.recompute',return_value=self.ready),patch('exp11_evidence.recompute',side_effect=[evidence,altered]),patch('exp11_binary_analysis.analyze',return_value={}),patch('exp11_omnibus_subprocess.recompute',return_value=omnibus):
            with self.assertRaisesRegex(ValueError,'Raw execution evidence changed'):
                reconstruct(self.root,'exp_1_1','statistical_analysis',self.inputs)
        omnibus['component']['omnibus_diagnostics_passed']=False
        with patch('phase1_readiness.recompute',return_value=self.ready),patch('exp11_evidence.recompute',return_value=evidence),patch('exp11_binary_analysis.analyze',return_value={}),patch('exp11_omnibus_subprocess.recompute',return_value=omnibus):
            with self.assertRaisesRegex(ValueError,'Actual converged omnibus'):
                reconstruct(self.root,'exp_1_1','statistical_analysis',self.inputs)

    def test_pending_external_and_mutated_input_never_complete(self):
        bad=deepcopy(self.ready);bad['components']['external']['status']='PENDING'
        with patch('phase1_readiness.recompute',return_value=bad):
            with self.assertRaises(ValueError):reconstruct(self.root,'phase1','receipt',self.inputs)
        (self.root/'components.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'input hash'):reconstruct(self.root,'phase1','receipt',self.inputs)


if __name__=='__main__':unittest.main()
