"""Synthetic complete negative Phase1 measurements; never empirical evidence."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from complete_experiment_receipts import reconstruct
from phase1_readiness import SCHEMAS,PendingEvidence,gate_checks
from phase1_evidence import ARMS


class NegativePhase1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        hygiene={'H1':True,'H2_independent_parser_agreement':True,'H4_raw_record_integrity':True,
                 'H5':True,'pilot_n':5,'sweep_n':128}
        self.receipts={
            'source':{'npz_crosscheck_exact':True,'frozen_vectors_recomputed_exactly':True},
            'post_fp8':{'attempt':'full_v2','passed':True,'fraction_reduction':.986},
            'primary':{'attempt':'SYNTHETIC_ONLY','primary_checkpoints_qualified':False,
                'metrics':{a:{c:{'n':64} for c in ('harmful','harmless')} for a in ARMS},
                'hygiene':{a:deepcopy(hygiene) for a in ARMS},
                'comparisons':{a:{'refusal_gate':a!='refusal_ablated','benign_kl_gate':True,
                    'benign_tool_propensity_gate':True} for a in ARMS[1:]}},
            'external':{'external_pair_qualified':False,'refusal_gap_intact_minus_uncensored':.25,
                'strict_refusal_gap':.2,'metrics':{a:{'n':64} for a in ('flash_next_intact','flash_next_uncensored')},
                'hygiene':{a:{**hygiene,'sweep_n':64} for a in ('flash_next_intact','flash_next_uncensored')}},
            'author':{'passed':True,'H1':True,'H2':True,'H4':True,'H5':True,'base_scenarios':40,
                'templates':10,'factorial_cells':480,'bag_of_words_accuracy':.5,'held_out_group':'template_id',
                'historical_logs_symmetric':True},
            'calibration':{'calibration_passed':True,'hygiene':{'H1':True,'H2':True,'H5':True}},
            'dose':{'complete_data':True,'hygiene_passed':True,'n_evaluation_episodes_per_alpha':48,
                'empirical_curve':[{'alpha':a} for a in (-.5,0.,.5,1.,1.5)],'phase2_authorized':False}}
        for key,schema in SCHEMAS.items():self.receipts[key]['schema']=schema
        self.receipts['primary']['post_fp8_gate']=deepcopy(self.receipts['post_fp8'])
        self.components={key:f'records/SYNTHETIC_{key}.json' for key in SCHEMAS}
        self.provenance_file=self.write('provenance_source.json',{'SYNTHETIC':True})
        self.provenance={'status':'PASS','artifact_sha256':{self.provenance_file['path']:self.provenance_file['sha256']}}
        self.refresh()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
        return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def refresh(self):
        self.ready={'primary_phase2_authorized':False,'external_validity_authorized':False,'components':{},
                    'provenance':{'status':'NOT_EVALUATED'}}
        for key,receipt in self.receipts.items():
            desc=self.write(self.components[key],receipt);checks=gate_checks(key,receipt)
            self.ready['components'][key]={'receipt_path':desc['path'],'receipt_sha256':desc['sha256'],
                'recomputed':True,'status':'PASS' if all(checks.values()) else 'FAIL',
                'gate_passed':all(checks.values()),'gate_checks':checks}
        self.inputs={'phase1_components':self.write('components.json',self.components)}

    def run_complete(self,kind='receipt',*,provenance=None,mechanics=None):
        with ExitStack() as stack:
            ready=stack.enter_context(patch('phase1_readiness.recompute',return_value=self.ready))
            prov=stack.enter_context(patch('phase1_readiness.audit_provenance',
                side_effect=provenance if callable(provenance) else None,
                return_value=self.provenance if provenance is None or callable(provenance) else provenance))
            stack.enter_context(patch('primary_mechanics.inspect',return_value=mechanics or {'terminal':True,'passed':True}))
            result=reconstruct(self.root,'phase1',kind,self.inputs)
            self.assertEqual(ready.call_count,2)
            self.assertEqual(prov.call_count,2)
            self.assertEqual(set(prov.call_args.args[1]),set(SCHEMAS))
            return result

    def test_both_measurement_artifacts_complete_without_authorizing_phase2(self):
        for kind in ('receipt','statistical_analysis'):
            with self.subTest(kind=kind):
                result=self.run_complete(kind);r=result['result']
                self.assertTrue(result['complete_experiment_receipt'])
                self.assertTrue(r['empirical_coverage_complete']);self.assertTrue(r['statistical_analysis_complete'])
                self.assertFalse(r['primary_qualification_passed']);self.assertFalse(r['phase2_authorized'])
                self.assertFalse(r['phase3_authorized']);self.assertFalse(result['phase3_authorized'])
                self.assertFalse(r['phase1_readiness']['primary_phase2_authorized'])
                self.assertEqual(r['measurement_completion']['status'],'COMPLETE_MEASUREMENTS_PRIMARY_QUALIFICATION_FAILED')

    def test_missing_pending_or_unrecomputed_any_component_never_completes(self):
        for key in SCHEMAS:
            for field,value in (('status','PENDING'),('status','MISSING'),('recomputed',False)):
                with self.subTest(key=key,field=field,value=value):
                    self.refresh();self.ready['components'][key][field]=value
                    with self.assertRaises(ValueError):self.run_complete()

    def test_failed_nonqualification_gate_never_completes(self):
        changes={'source':('npz_crosscheck_exact',False),'post_fp8':('passed',False),
                 'author':('H4',False),'calibration':('calibration_passed',False),'dose':('complete_data',False)}
        for key,(field,value) in changes.items():
            original=deepcopy(self.receipts[key])
            with self.subTest(key=key):
                self.receipts[key][field]=value;self.refresh()
                with self.assertRaises(ValueError):self.run_complete()
            self.receipts[key]=original

    def test_partial_dose_or_primary_fixed_coverage_fails(self):
        for key,mutate in [('dose',lambda r:r.update(n_evaluation_episodes_per_alpha=47)),
              ('dose',lambda r:r['empirical_curve'].pop()),
              ('primary',lambda r:r['metrics']['intact']['harmful'].update(n=63)),
              ('primary',lambda r:r['hygiene']['pca_101'].update(pilot_n=4)),
              ('external',lambda r:r['metrics']['flash_next_intact'].update(n=63))]:
            original=deepcopy(self.receipts[key])
            with self.subTest(key=key):
                mutate(self.receipts[key]);self.refresh()
                with self.assertRaises(ValueError):self.run_complete()
            self.receipts[key]=original

    def test_primary_hygiene_or_additional_capability_failure_is_not_waived(self):
        for mutate in (lambda r:r['hygiene']['intact'].update(H5=False),
                       lambda r:r['comparisons']['refusal_ablated'].update(benign_kl_gate=False),
                       lambda r:r['comparisons']['pca_101'].update(refusal_gate=False)):
            original=deepcopy(self.receipts['primary']);mutate(self.receipts['primary']);self.refresh()
            with self.assertRaises(ValueError):self.run_complete()
            self.receipts['primary']=original
        self.refresh()
        with self.assertRaises(ValueError):self.run_complete(mechanics={'terminal':True,'passed':False})

    def test_external_hygiene_failure_fails_but_qualified_pair_cannot_unlock_primary(self):
        self.receipts['external']['hygiene']['flash_next_uncensored']['H1']=False;self.refresh()
        with self.assertRaises(ValueError):self.run_complete()
        self.receipts['external']['hygiene']['flash_next_uncensored']['H1']=True
        self.receipts['external'].update(external_pair_qualified=True,refusal_gap_intact_minus_uncensored=.6,strict_refusal_gap=.6)
        self.refresh();result=self.run_complete()
        self.assertEqual(result['result']['external_scope'],'QUALIFIED_EXTERNAL_PAIR_WITHOUT_PRIMARY_PHASE2_AUTHORIZATION')
        self.assertFalse(result['result']['phase2_authorized'])

    def test_missing_failed_or_changed_provenance_blocks_completion(self):
        for provenance in ({'status':'PENDING'},{'status':'FAIL'},{'status':'PASS','artifact_sha256':{}}):
            with self.subTest(provenance=provenance),self.assertRaises(ValueError):self.run_complete(provenance=provenance)
        calls=[]
        def mutate(*args):
            calls.append(True)
            if len(calls)==2:(self.root/self.provenance_file['path']).write_text('{"changed":true}')
            return self.provenance
        with self.assertRaisesRegex(ValueError,'input hash'):self.run_complete(provenance=mutate)

    def test_changed_component_receipt_during_provenance_rejected(self):
        def mutate(*args):
            (self.root/self.components['dose']).write_text('{}')
            return self.provenance
        with self.assertRaisesRegex(ValueError,'input hash'):self.run_complete(provenance=mutate)

    def test_pending_provenance_exception_is_a_normal_verifier_rejection(self):
        def pending(*args):raise PendingEvidence('SYNTHETIC missing capture')
        with self.assertRaisesRegex(ValueError,'provenance remains pending'):self.run_complete(provenance=pending)

    def test_failed_primary_never_reaches_downstream_experiment_auditors(self):
        for experiment,extra in [('exp_2_3',{'full_calls':[]}),
                ('exp_1_1',{'execution_selection':{},'posterior_invocation':{}})]:
            with self.subTest(experiment=experiment),patch('phase1_readiness.recompute',return_value=self.ready),\
                 patch('phase1_readiness.audit_provenance') as prov,patch('exp23_statistical_analysis.recompute') as exp23,\
                 patch('exp11_evidence.recompute') as exp11:
                with self.assertRaisesRegex(ValueError,'All primary Phase1 gates'):
                    reconstruct(self.root,experiment,'receipt',{**self.inputs,**extra})
                prov.assert_not_called();exp23.assert_not_called();exp11.assert_not_called()


if __name__=='__main__':unittest.main()
