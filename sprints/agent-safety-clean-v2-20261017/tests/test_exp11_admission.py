"""Actual central resource/command binding on isolated filesystem fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from exp11_admission import audit_central_admission,audit_worker_admission,option,local_source_closure
from exp11_action_worker import digest,hash_bytes,write_exclusive
from test_exp11_action_execution import payload,load_runner_functions


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.root=Path(self.temp.name)
        ns=load_runner_functions({'validate_batch','build_batch','resource_config'})
        for name in (*ns['PACKAGE_FILES'],'modal_exp11_evaluation.py','exp11_admission.py'):
            target=self.root/'scripts'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'scripts'/name,target)
        value=payload()
        for name in value['input_sources']:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        value['input_sources']={n:hash_bytes(self.root/n) for n in value['input_sources']}
        value['design_sha256']=value['input_sources'][value['design_path']]
        value['identity']['model_revision']=value['input_sources']['checkpoints_full_residual_v2/intact_materialization.json']
        package={n:hash_bytes(self.root/'scripts'/n) for n in ns['PACKAGE_FILES']}
        value['source_hashes']={n:package[n] for n in value['source_hashes']}
        value['payload_sha256']=digest({k:v for k,v in value.items() if k!='payload_sha256'})
        self.batch=ns['build_batch']('admission_fixture',[{'kind':'action','payload':value,'tokenizer_files':value['tokenizer_files']}],package)
        self.path='infrastructure/input.json';write_exclusive(self.root/self.path,self.batch)
        self.config=ns['resource_config'](self.root,self.batch['attempt'],self.path)
        self.job={**deepcopy(self.config),'app_id':'ap-fixture'}
        self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):
        self.job.setdefault('admitted_upper_bound_usd',self.config['upper_bound_usd'])
        (self.root/'infrastructure/resource_allocations.json').write_text(json.dumps({'jobs':[self.job]}))
        (self.root/'infrastructure/budget_reservations.json').write_text(json.dumps({'reservations':[
            {'job':self.batch['job'],'upper_bound_usd':self.config['upper_bound_usd'],
             'admitted_upper_bound_usd':self.config['upper_bound_usd']}], 'reconciliations':[]}))

    def test_auditor_closure_captures_supplemental_h1_checks_even_if_imported_before(self):
        sources=local_source_closure(ROOT,['phase1_readiness','phase2_admission'])
        for name in ('audit_calibration_provenance.py','audit_dose_response.py','audit_exp23_launch.py'):
            self.assertEqual(sources['scripts/'+name],hash_bytes(ROOT/'scripts'/name))

    def test_exact_real_decorator_profile_and_frozen_input_pass(self):
        result=audit_central_admission(self.root,self.batch,self.path,require_app_id=True)
        self.assertEqual(result['maximum_gpu_calls'],1);self.assertFalse(result['actual_spend_claimed'])
        self.assertEqual(result['input_sha256'],hash_bytes(self.root/self.path))
        value=self.batch['items'][0]['payload']
        write_exclusive(self.root/'group_calls'/f'{value["experiment"]}.json',{'job':self.batch['job'],
            'batch_sha256':self.batch['batch_sha256'],'kind':'action'})
        self.assertEqual(audit_worker_admission(self.root,value),result)

    def test_duplicate_equals_style_or_wrong_cli_value_is_rejected(self):
        original=deepcopy(self.job)
        for arguments in [original['command']+['--attempt','admission_fixture'],
                          [v if v!='--attempt' else '--attempt=admission_fixture' for v in original['command']],
                          [v if v!='admission_fixture' else 'different' for v in original['command']]]:
            self.job=deepcopy(original);self.job['command']=arguments;self.save()
            with self.assertRaises(ValueError):audit_central_admission(self.root,self.batch,self.path)

    def test_missing_frozen_payload_or_changed_package_is_rejected(self):
        self.job['frozen_files'].pop(self.path);self.save()
        with self.assertRaisesRegex(ValueError,'freeze exact'):audit_central_admission(self.root,self.batch,self.path)
        self.job=deepcopy(self.config);self.job['app_id']='ap-fixture';self.save()
        source=self.root/'scripts/scenario_contract.py';source.write_text(source.read_text()+'\n# fixture mutation\n')
        with self.assertRaisesRegex(ValueError,'freeze exact'):audit_central_admission(self.root,self.batch,self.path)

    def test_valid_historical_reduction_keeps_final_audit_stable_but_cannot_launch_again(self):
        before=audit_central_admission(self.root,self.batch,self.path,require_app_id=True)
        path=self.root/'infrastructure/budget_reservations.json';budget=json.loads(path.read_text())
        original=budget['reservations'][0]['upper_bound_usd']
        budget['reservations'][0]['upper_bound_usd']=1
        budget['reconciliations']=[{'job':self.batch['job'],'app_id':'ap-fixture',
            'previous_upper_bound_usd':original,'new_upper_bound_usd':1}]
        path.write_text(json.dumps(budget))
        self.assertEqual(audit_central_admission(self.root,self.batch,self.path,require_app_id=True),before)
        with self.assertRaisesRegex(ValueError,'cannot authorize a fresh'):audit_central_admission(self.root,self.batch,self.path)
        budget['reconciliations']=[];path.write_text(json.dumps(budget))
        with self.assertRaisesRegex(ValueError,'reconciliation history'):audit_central_admission(self.root,self.batch,self.path,require_app_id=True)

    def test_declared_call_count_and_final_app_id_must_match(self):
        self.job['resource_profile']['functions']['evaluate']['max_calls']=2;self.save()
        with self.assertRaisesRegex(ValueError,'call bounds'):audit_central_admission(self.root,self.batch,self.path)
        self.job=deepcopy(self.config);self.save()
        audit_central_admission(self.root,self.batch,self.path)
        with self.assertRaisesRegex(ValueError,'application ID'):audit_central_admission(self.root,self.batch,self.path,require_app_id=True)


if __name__=='__main__':unittest.main()
