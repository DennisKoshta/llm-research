"""Actual raw chains and resource ASTs with synthetic model/backend fixtures."""
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import json,shutil,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import structured_records_worker as worker
import structured_records_evidence as evidence
from launch_bounded_job import _decorators
from test_structured_records_execution import fixtures,Model,Tokenizer,inventory


class StructuredEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  sources={}
  def save(name,value):
   path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
   path.write_text(json.dumps(value));sources[name]=worker.hash_bytes(path)
   return sources[name]
  components={}
  for key in worker.READY_KEYS:
   name=f'records/{key}.json';h=save(name,{})
   components[key]={'status':'PASS','gate_passed':True,'receipt_path':name,'receipt_sha256':h}
  requests=fixtures();save('stimuli/requests.json',requests)
  spec=save('FINAL_RESEARCH_SPECIFICATION.md',{'synthetic_fixture':True})
  checkpoint=save('checkpoints_full_residual_v2/intact_materialization.json',{})
  tokenizers={name:save('tokenizer_source/'+name,{'fixture':name}) for name in ('tokenizer_config.json','tokenizer.json')}
  self.ready={'schema':'phase1_readiness_raw_audit_v1','primary_phase2_authorized':True,'components':components,
              'provenance':{'status':'PASS','artifact_sha256':{}}}
  code={}
  for name in (*worker.RUNTIME_FILES,'modal_structured_records.py','structured_records_evidence.py','historical_admission.py'):
   target=self.root/'scripts'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'scripts'/name,target)
   if name in worker.RUNTIME_FILES:code[name]=worker.hash_bytes(target)
  self.payload=worker.build_payload('evidence_fixture',requests,readiness=self.ready,input_sources=sources,source_hashes=code,
   specification_sha256=spec,request_source_path='stimuli/requests.json',checkpoint_sha256=checkpoint,
   tokenizer_files=tokenizers,tokenizer_source_paths={n:'tokenizer_source/'+n for n in tokenizers},component_sources={})
  self.input='stimuli/'+self.payload['experiment']+'_input.json';worker.write_exclusive(self.root/self.input,self.payload)
  source='scripts/modal_structured_records.py';limits=_decorators((self.root/source).read_text())
  self.job={'job':self.payload['central_job'],'gpus':2,'single_gpu_workers':2,'tp8_workers':0,'app_id':'ap-fixture','admitted_upper_bound_usd':20,
   'command':['modal','run','--detach',source,'--attempt',self.payload['attempt'],'--input-path',self.input,'--launch'],
   'resource_profile':{'source':source,'source_sha256':worker.hash_bytes(self.root/source),
    'functions':{n:{'max_calls':1,'limits':limits[n]} for n in ('records','coordinate')},'excluded_functions':{},
    'call_bound_basis':'One frozen record worker and coordinator, no retries.'},
   'frozen_files':{**sources,**{'scripts/'+n:h for n,h in code.items()},self.input:worker.hash_bytes(self.root/self.input),
    **{name:worker.hash_bytes(self.root/name) for name in ('scripts/modal_structured_records.py','scripts/structured_records_evidence.py','scripts/historical_admission.py')}}}
  self.save_job()
  self.ledger={'state':'SPAWNED','call_id':'fc-fixture','payload_sha256':self.payload['payload_sha256'],
   'experiment':self.payload['experiment'],'role':self.payload['model_role'],'task':self.payload['task'],
   'job':self.payload['central_job'],'worker_module':'modal_structured_records','worker_function':'records'}
  worker.write_exclusive(self.root/'group_calls'/f'{self.payload["experiment"]}.json',self.ledger)
  worker.write_exclusive(self.root/'structured_worker_starts'/f'{self.payload["experiment"]}.json',
   {'call_id':'fc-fixture','payload_sha256':self.payload['payload_sha256']})
  run=worker.RecordRun(self.root,self.payload,lambda:None);self.folder=run.directory
  (self.folder/'tokenizer').mkdir()
  for name in tokenizers:shutil.copyfile(self.root/'tokenizer_source'/name,self.folder/'tokenizer'/name)
  self.status=run.execute(Model(run),lambda cap:cap,inventory(self.payload))
 def save_job(self):
  path=self.root/'infrastructure';path.mkdir(exist_ok=True)
  (path/'resource_allocations.json').write_text(json.dumps({'jobs':[self.job]}))
  (path/'budget_reservations.json').write_text(json.dumps({'reservations':[{'job':self.job['job'],'upper_bound_usd':20,'admitted_upper_bound_usd':20}]}))
 def lookup(self,key):return {'call_id':key,'function_name':'records','module_name':'modal_structured_records','status':'SUCCESS'}
 def audit(self,**kwargs):
  with patch('phase1_readiness.recompute',return_value=self.ready):
   return evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=self.lookup,**kwargs)
 def test_complete_real_raw_replay_central_and_backend(self):
  value=self.audit()
  self.assertTrue(value['exact_prompt_replay']);self.assertTrue(value['backend_identity_verified'])
  self.assertEqual(value['full_n'],7);self.assertEqual(value['pilot_n'],5)
  self.assertEqual(value['stream_audit'],self.status['stream_audit'])
  self.assertFalse(value['phase2_executor_launch']);self.assertEqual(value['central_admission']['maximum_gpu_calls'],1)
 def test_reconciled_reservation_preserves_original_actual_profile(self):
  before=evidence.audit_central_admission(self.root,self.payload,self.input,require_app_id=True)
  path=self.root/'infrastructure/budget_reservations.json';budget=json.loads(path.read_text())
  budget['reservations'][0]['upper_bound_usd']=1
  budget['reconciliations']=[{'job':self.job['job'],'app_id':'ap-fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':1}]
  path.write_text(json.dumps(budget))
  self.assertEqual(before,evidence.audit_central_admission(self.root,self.payload,self.input,require_app_id=True))
  with self.assertRaisesRegex(ValueError,'fresh structured launch'):
   evidence.audit_central_admission(self.root,self.payload,self.input)
  budget['reconciliations']=[];path.write_text(json.dumps(budget))
  with self.assertRaisesRegex(ValueError,'reconciliation history'):
   evidence.audit_central_admission(self.root,self.payload,self.input,require_app_id=True)
 def test_wrong_worker_started_or_backend_cannot_pass(self):
  path=self.root/'structured_worker_starts'/f'{self.payload["experiment"]}.json'
  path.write_text(json.dumps({'call_id':'fc-wrong','payload_sha256':self.payload['payload_sha256']}))
  with self.assertRaisesRegex(ValueError,'worker start'):self.audit()
  path.write_text(json.dumps({'call_id':'fc-fixture','payload_sha256':self.payload['payload_sha256']}))
  with patch('phase1_readiness.recompute',return_value=self.ready):
   for change in ({'function_name':'evaluate'},{'module_name':'modal_exp23_executor'},{'status':'PENDING'}):
    with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Backend'):
     evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=lambda key:{**self.lookup(key),**change})
 def test_captured_tokenizer_tamper_rejected_before_prompt_replay(self):
  (self.folder/'tokenizer/tokenizer.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'tokenizer bytes'):self.audit()
 def test_independently_rendered_prompt_mismatch_is_rejected(self):
  class WrongTokenizer(Tokenizer):
   def apply_chat_template(self,*args,**kwargs):return super().apply_chat_template(*args,**kwargs)+'changed'
  with patch('phase1_readiness.recompute',return_value=self.ready),self.assertRaisesRegex(ValueError,'prompt/tokenizer replay'):
   evidence.audit_complete_run(self.root,self.payload,tokenizer=WrongTokenizer(),lookup_call=self.lookup)
 def test_default_replay_requires_exact_captured_versions(self):
  path=self.folder/'runtime_context.json';context=json.loads(path.read_text());context['runtime_versions']['transformers']='captured-other'
  path.write_text(json.dumps(context))
  with patch('phase1_readiness.recompute',return_value=self.ready),patch('structured_records_evidence.importlib.metadata.version',return_value='local-other'):
   with self.assertRaisesRegex(ValueError,'captured transformers'):
    evidence.audit_complete_run(self.root,self.payload,lookup_call=self.lookup)
 def test_cli_swapped_payload_duplicate_flags_and_frozen_source_fail(self):
  original=deepcopy(self.job)
  for change in ('path','duplicate','freeze','calls','app'):
   self.job=deepcopy(original)
   if change=='path':self.job['command'][self.job['command'].index('--input-path')+1]='stimuli/other.json'
   elif change=='duplicate':self.job['command']+=['--attempt',self.payload['attempt']]
   elif change=='freeze':self.job['frozen_files'][self.input]='0'*64
   elif change=='calls':self.job['resource_profile']['functions']['records']['max_calls']=2
   else:self.job.pop('app_id')
   self.save_job()
   with self.subTest(change=change),self.assertRaises(ValueError):
    evidence.audit_central_admission(self.root,self.payload,self.input,require_app_id=True)
 def test_changed_readiness_and_completion_status_rejected(self):
  with patch('phase1_readiness.recompute',return_value={}),self.assertRaisesRegex(ValueError,'Phase1'):
   evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=self.lookup)
  path=self.root/'qualification_status'/f'{self.payload["experiment"]}.json'
  value=json.loads(path.read_text());value['status']='RECORD_GATE_FAILED';path.write_text(json.dumps(value))
  with self.assertRaisesRegex(ValueError,'saved completion'):self.audit()


if __name__=='__main__':unittest.main()
