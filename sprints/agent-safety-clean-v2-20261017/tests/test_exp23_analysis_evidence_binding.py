"""Synthetic mutation tests; no empirical outcomes, backend calls, or receipts."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import exp23_statistical_analysis as analysis
from exp23_executor_worker import paths
from exp23_author_worker import ChainWriter,read_chain,hash_bytes


class EvidenceBindingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name)
  self.payload={'experiment':'exp23_SYNTHETIC_binding','executor':'primary_intact_executor','cot_mode':'cot_on',
      'payload_sha256':'a'*64,'readiness':{'primary_phase2_authorized':True,'external_validity_authorized':False},
      'design':{'schema':'SYNTHETIC_ONLY'},'input_sources':{},'source_hashes':{},'tokenizer_files':{}}
  self.write('infrastructure/exp23_analysis_plan_v1.json',analysis.plan())
  self.write('scripts/exp23_statistical_analysis.py',(ROOT/'scripts/exp23_statistical_analysis.py').read_text(),raw=True)
  for key,name in [('input_sources','stimuli/synthetic.json'),('source_hashes','synthetic_worker.py'),('tokenizer_files','tokenizer_config.json')]:
   path=self.root/name if key=='input_sources' else self.root/'scripts'/name if key=='source_hashes' else paths(self.root,self.payload,'pilot')['folder']/'tokenizer'/name
   self.write(path,{'synthetic':True});self.payload[key][name]=hash_bytes(path)
  self.write('payload.json',self.payload)
  self.rows=[{'episode_id':'SYNTHETIC_EPISODE','request_sha256':'b'*64,'stage':'full','outcome_category':'SAFE_ROUTE'}]
  self.audits={}
  self.call=f'group_calls/{self.payload["experiment"]}/full.json'
  for stage in ('pilot','full'):
   p=paths(self.root,self.payload,stage)
   for key in ('raw','journal','returns','inventory','claim'):self.write(p[key],{'synthetic':key,'stage':stage})
   p['episodes'].parent.mkdir(parents=True,exist_ok=True)
   writer=ChainWriter(p['episodes'],self.payload['payload_sha256'])
   for row in self.rows:writer.append({**row,'stage':stage})
   _,head=read_chain(p['episodes'],self.payload['payload_sha256'])
   self.audits[stage]={'complete_data':True,'H1':True,'H2':True,'H4':True,'H5':stage=='pilot',
       **{key+'_sha256':hash_bytes(p[key]) for key in ('raw','journal','returns','episodes','inventory')},
       'chain_heads':{'episodes':head}}
   self.write(p['status'],{'status':'FULL_COMPLETE' if stage=='full' else 'PILOT_PASS','audit':self.audits[stage]})
   self.write(p['folder']/'frozen_input.json',self.payload)
   self.write(p['pilot_audit'],'SYNTHETIC PILOT ONLY',raw=True)
   ledger={'schema':'exp23_executor_call_v1','state':'SPAWNED','stage':stage,'payload_path':'payload.json',
      'payload_sha256':self.payload['payload_sha256'],'worker_module':'modal_exp23_executor',
      'worker_function':'execute','gpu':'H200:1','call_id':'fc-Synthetic'+stage}
   self.write(f'group_calls/{self.payload["experiment"]}/{stage}.json',ledger)
   self.write(f'group_calls/{self.payload["experiment"]}/{stage}_worker_started.json',
      {'call_id':ledger['call_id'],'payload_sha256':self.payload['payload_sha256'],'stage':stage})

 def write(self,name,value,raw=False):
  path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(value if raw else json.dumps(value));return path

 def run_audit(self,*,during_backend=None,during_analysis=None,stage_mutation=None):
  def backend(call_id):
   if during_backend:during_backend()
   return {'call_id':call_id,'function_name':'execute','module_name':'modal_exp23_executor','status':'SUCCESS'}
  def analyze(rows,design):
   if during_analysis:during_analysis()
   return {'SYNTHETIC':True,'rows':rows,'design':design}
  def stage(root,payload,name):
   if stage_mutation:stage_mutation(name)
   return deepcopy(self.audits[name])
  with ExitStack() as stack:
   stack.enter_context(patch('audit_exp23_launch.recompute',return_value={'SYNTHETIC':True}))
   stack.enter_context(patch('audit_exp23_launch.lookup_backend',side_effect=backend))
   stack.enter_context(patch('exp23_executor_worker.audit_stage',side_effect=stage))
   stack.enter_context(patch.object(analysis,'analyze',side_effect=analyze))
   return analysis.recompute(self.root,[self.call])

 def test_unchanged_chain_is_real_read_and_all_immutable_sources_are_bound(self):
  result=self.run_audit();bound=result['immutable_evidence_sha256']
  self.assertEqual(result['rows'][0]['outcome_category'],'SAFE_ROUTE')
  self.assertEqual(len(bound),26)
  self.assertIn('scripts/synthetic_worker.py',bound)
  self.assertIn('stimuli/synthetic.json',bound)
  self.assertTrue(any(name.endswith('pilot_worker_started.json') for name in bound))

 def test_new_valid_chain_after_raw_audit_is_rejected(self):
  def swap():
   path=paths(self.root,self.payload,'full')['episodes'];path.unlink()
   writer=ChainWriter(path,self.payload['payload_sha256'])
   writer.append({**self.rows[0],'outcome_category':'SHORTCUT_VIOLATION'})
   # Demonstrate internal chain validity is insufficient: this succeeds.
   read_chain(path,self.payload['payload_sha256'])
  with self.assertRaisesRegex(ValueError,'Reloaded episode chain differs'):self.run_audit(during_backend=swap)

 def test_wrong_head_even_with_same_file_bytes_is_rejected(self):
  self.audits['full']['chain_heads']['episodes']='c'*64
  self.write(paths(self.root,self.payload,'full')['status'],{'status':'FULL_COMPLETE','audit':self.audits['full']})
  with self.assertRaisesRegex(ValueError,'Reloaded episode chain differs'):self.run_audit()

 def test_changes_during_analysis_rejected_for_both_stages_and_sources(self):
  targets=[paths(self.root,self.payload,stage)[key] for stage in ('pilot','full')
      for key in ('raw','journal','returns','episodes','inventory','claim','status')]
  targets += [self.root/'stimuli/synthetic.json',self.root/'scripts/synthetic_worker.py',
      self.root/'scripts/exp23_statistical_analysis.py',self.root/'payload.json',
      paths(self.root,self.payload,'pilot')['folder']/'tokenizer/tokenizer_config.json']
  for path in targets:
   before=path.read_bytes()
   with self.subTest(path=path.relative_to(self.root)):
    with self.assertRaisesRegex(ValueError,'Immutable evidence changed'):
     self.run_audit(during_analysis=lambda:path.write_bytes(before+b' '))
   path.write_bytes(before)

 def test_mutation_during_stage_audit_is_rejected(self):
  path=paths(self.root,self.payload,'pilot')['raw']
  def mutate(stage):
   if stage=='full':path.write_text('CHANGED SYNTHETIC RAW')
  with self.assertRaisesRegex(ValueError,'Immutable evidence changed'):self.run_audit(stage_mutation=mutate)

 def test_audit_hash_not_matching_snapshot_is_rejected(self):
  self.audits['pilot']['journal_sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'Raw stage audit binds different evidence'):self.run_audit()

 def test_missing_source_and_tokenizer_tamper_fail_before_backend(self):
  path=paths(self.root,self.payload,'pilot')['folder']/'tokenizer/tokenizer_config.json'
  path.write_text('{}')
  with self.assertRaisesRegex(ValueError,'Immutable scientific source differs'):self.run_audit()
  path.unlink()
  with self.assertRaises(FileNotFoundError):self.run_audit()

 def test_payload_swap_between_initial_read_and_snapshot_is_rejected(self):
  original=analysis.evidence_bindings
  def swap(root,payload,call):
   path=root/'payload.json';path.write_text(path.read_text()+' ')
   return original(root,payload,call)
  with patch.object(analysis,'evidence_bindings',side_effect=swap):
   with self.assertRaisesRegex(ValueError,'changed while preparing immutable evidence snapshot'):self.run_audit()


if __name__=='__main__':unittest.main()
