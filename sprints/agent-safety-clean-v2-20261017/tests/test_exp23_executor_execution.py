"""CPU-only synthetic executor, coordinator and ordering audit regressions."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
from test_exp23_design import fixture_bundle
from test_structured_records_execution import Tokenizer as BaseTokenizer,inventory
import exp23_executor_worker as worker
import audit_exp23_launch as launch_audit
from exp23_design import build_design

DESIGN=None


class Tokenizer(BaseTokenizer):
 def apply_chat_template(self,messages,**kwargs):
  return json.dumps(messages,sort_keys=True)+'\n'+('<think>\n' if kwargs.get('enable_thinking') else '<assistant>\n')


def fixture_payload(attempt='offline',executor='primary_intact_executor',mode='cot_on'):
 global DESIGN
 if DESIGN is None:DESIGN=build_design(fixture_bundle(),'a'*64,'stimuli/fixture_bundle.json')
 components={key:{'status':'PASS','gate_passed':True,'receipt_path':f'records/{key}.json','receipt_sha256':'d'*64}
             for key in ('source','post_fp8','primary','author','calibration','dose')}
 sources={row['receipt_path']:row['receipt_sha256'] for row in components.values()};sources['components.json']='e'*64
 ready={'schema':'phase1_readiness_raw_audit_v1','primary_phase2_authorized':True,'components':components,'provenance':{'status':'PASS'}}
 return worker.build_payload(attempt,DESIGN,executor=executor,mode=mode,conditions=['B1'],readiness=ready,
  input_sources=sources,source_hashes={n:'b'*64 for n in worker.RUNTIME_FILES},components_path='components.json',
  tokenizer_files={'tokenizer_config.json':'c'*64,'tokenizer.json':'c'*64},checkpoint_sha256='f'*64,
  dependency_audit={'schema':'exp23_dependency_bundle_audit_v1','all_streams_passed':True,'bundle_sha256':'a'*64,
   'boilerplate_passed':True,'bundle_path':'stimuli/fixture_bundle.json','runs_path':'stimuli/fixture_runs.json'})


class Model:
 def __init__(self,run,staged=False,unclosed=False,capped=0,shape=False):
  self.run=run;self.staged=staged;self.unclosed=unclosed;self.cap_left=capped;self.shape=shape
 def get_tokenizer(self):return Tokenizer()
 def generate(self,inputs,params,use_tqdm):
  rows,_=worker.read_chain(self.run.journal.path,self.run.payload['payload_sha256']);rows=rows[-len(inputs):];outputs=[]
  for row,inp in zip(rows,inputs):
   name=row['safe_tool'] if self.staged else row['direct_tool']
   args={'step':row['turn_index'],'operation':row['required_ordinary_operations'][row['turn_index']-1]} if self.staged else {}
   raw=('Reason.</think>' if row['cot_mode']=='cot_on' else '')+'<tool_call>'+json.dumps({'name':name,'arguments':args})+'</tool_call>'
   if self.unclosed:raw=raw.replace('</think>','')
   finish='length' if self.cap_left else 'stop';self.cap_left=max(0,self.cap_left-1)
   outputs.append(SimpleNamespace(prompt_token_ids=inp['prompt_token_ids'],outputs=[SimpleNamespace(text=raw,token_ids=[7]*20,finish_reason=finish)]))
  return outputs[:-1] if self.shape else outputs


class ExecutorTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.commits=0
 def commit(self):self.commits+=1
 def run_stage(self,payload,stage,**options):
  run=worker.ExecutorRun(self.root,payload,stage,self.commit)
  result=run.execute(Model(run,**options),lambda cap:cap,inventory(payload))
  return run,result

 def test_five_multiturn_pilots_replay_exact_prompts_observations_and_tokens(self):
  value=fixture_payload();run,result=self.run_stage(value,'pilot',staged=True)
  self.assertTrue(result['audit']['H5']);self.assertEqual(result['audit']['episodes'],5);self.assertEqual(result['audit']['turns'],35)
  audited=worker.audit_stage(self.root,value,'pilot',tokenizer=Tokenizer())
  self.assertEqual(audited,result['audit']);self.assertGreater(self.commits,10)
  self.assertIn('Status: PASS',run.p['pilot_audit'].read_text())
  with patch.object(worker,'runtime_versions',return_value={'transformers':'mismatched','tokenizers':'mismatched','jinja2':'mismatched','vllm':None}):
   with self.assertRaisesRegex(ValueError,'captured transformers version'):
    worker.audit_stage(self.root,value,'pilot')

 def test_unclosed_pilot_retains_raw_and_is_failed(self):
  value=fixture_payload();run,result=self.run_stage(value,'pilot',unclosed=True)
  self.assertEqual(result['status'],'PILOT_FAILED');self.assertFalse(result['audit']['H5'])
  rows,_=worker.read_chain(run.p['raw'],value['payload_sha256'])
  self.assertTrue(all(r['parse_status']=='TRUNCATED_IN_THINK' for r in rows))
  self.assertFalse(worker.paths(self.root,value,'full')['raw'].exists())
  self.assertFalse(result['audit']['H1']);self.assertEqual(result['audit']['capped_turns'],0)
  self.assertEqual(result['audit']['truncated_turns'],5);self.assertEqual(result['audit']['unclosed_think_turns'],5)

 def test_full_sweep_is_all96_and_h1_uses_episode_and_turn_denominators(self):
  value=fixture_payload();run,result=self.run_stage(value,'full',capped=2)
  self.assertEqual(result['audit']['episodes'],96);self.assertEqual(result['status'],'FULL_H1_FAILED')
  self.assertEqual(result['audit']['capped_episodes'],2)
  rows,_=worker.read_chain(run.p['raw'],value['payload_sha256']);self.assertEqual(len(rows),96)
  self.assertEqual({r['condition'] for r in rows},{'B1'})

 def test_stop_reason_unclosed_think_fails_full_h1_below_cap(self):
  value=fixture_payload();run,result=self.run_stage(value,'full',unclosed=True)
  audit=result['audit']
  self.assertEqual(result['status'],'FULL_H1_FAILED')
  self.assertEqual(audit['capped_turns'],0);self.assertEqual(audit['capped_episodes'],0)
  self.assertEqual(audit['truncated_turns'],96);self.assertEqual(audit['truncated_episodes'],96)
  self.assertEqual(audit['unclosed_think_turns'],96)

 def test_llama_and_cot_off_preserve_actual_native_template_path(self):
  value=fixture_payload(executor='llama_executor',mode='cot_off');run,result=self.run_stage(value,'pilot')
  self.assertTrue(result['audit']['H5'])
  rows,_=worker.read_chain(run.p['raw'],value['payload_sha256'])
  self.assertTrue(all(r['family']=='llama' and r['max_new_tokens']==1024 and r['full_prompt'].endswith('<assistant>\n') for r in rows))

 def test_unexpected_return_shape_is_durable_before_failure(self):
  value=fixture_payload();run=worker.ExecutorRun(self.root,value,'pilot',self.commit)
  with self.assertRaisesRegex(ValueError,'return shape'):
   run.execute(Model(run,shape=True),lambda cap:cap,inventory(value))
  rows,_=worker.read_chain(run.p['returns'],value['payload_sha256']);self.assertEqual(len(rows[0]['outputs']),4)
  with self.assertRaises(ValueError):worker.audit_stage(self.root,value,'pilot',tokenizer=Tokenizer())

 def test_raw_and_inventory_tampering_and_duplicate_attempt_fail(self):
  value=fixture_payload();run,_=self.run_stage(value,'pilot')
  with self.assertRaises(FileExistsError):worker.ExecutorRun(self.root,value,'pilot',self.commit)
  rows=run.p['raw'].read_text().splitlines();row=json.loads(rows[0]);row['outcome_category']='SAFE_ROUTE';rows[0]=json.dumps(row)
  run.p['raw'].write_text('\n'.join(rows)+'\n')
  with self.assertRaises(ValueError):worker.audit_stage(self.root,value,'pilot',tokenizer=Tokenizer())

 def test_missing_cells_changed_requests_or_unqualified_identity_rejected(self):
  value=fixture_payload()
  for mutation in (lambda v:v['requests'].pop(),lambda v:v['identity'].update(model_arm='refusal_ablated'),
                   lambda v:v['readiness'].update(primary_phase2_authorized=False),lambda v:v['requests'][0].update(max_new_tokens=512)):
   changed=deepcopy(value);mutation(changed);changed['payload_sha256']=worker.digest({k:v for k,v in changed.items() if k!='payload_sha256'})
   with self.assertRaises(ValueError):worker.validate_payload(changed)

 def test_durable_calls_distinguish_pilot_full_and_unresolved_intents(self):
  value=fixture_payload();base={'schema':'exp23_executor_call_v1','payload_sha256':value['payload_sha256'],
   'state':'SPAWNED','worker_function':'execute','worker_module':'modal_exp23_executor','gpu':'H200:1',
   'spawn_intent_utc':'2026-10-17T01:00:00+00:00','submitted_utc':'2026-10-17T01:00:01+00:00'}
  pilot={**base,'stage':'pilot','call_id':'fc-pilot'};full={**base,'stage':'full','call_id':'fc-full',
   'pilot_call_id':'fc-pilot','pilot_audit_sha256':'a'*64,'pilot_completed_utc':'2026-10-17T00:59:00+00:00'}
  worker.write_exclusive(launch_audit.ledger_path(self.root,value,'pilot'),pilot)
  p=launch_audit.ledger_path(self.root,value,'full');worker.write_exclusive(p,full)
  self.assertEqual(launch_audit.audit_durable_call(self.root,value,'full')['call_id'],'fc-full')
  for change in ({'call_id':'fc-pilot'},{'state':'SPAWN_INTENT'},{'worker_function':'records'},
                 {'pilot_completed_utc':'2026-10-17T02:00:00+00:00'}):
   p.write_text(json.dumps({**full,**change}))
   with self.assertRaises(ValueError):launch_audit.audit_durable_call(self.root,value,'full')

 def test_actual_coordinator_separate_calls_and_no_resubmission_after_crash_intent(self):
  value=fixture_payload();root=self.root/'volume';volume=root/worker.NS;calls=[];saved={}
  for name in value['input_sources']:
   p=volume/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
   value['input_sources'][name]=worker.hash_bytes(p)
  for comp in value['readiness']['components'].values():comp['receipt_sha256']=value['input_sources'][comp['receipt_path']]
  value['payload_sha256']=worker.digest({k:v for k,v in value.items() if k!='payload_sha256'})
  pilot_audit={'H5':True};result={'status':'PILOT_PASS','audit':pilot_audit,'finished_utc':'2026-10-17T01:00:00+00:00'}
  worker.write_exclusive(worker.paths(volume,value,'pilot')['status'],result)
  def spawn(payload,stage):
   path=launch_audit.ledger_path(volume,payload,stage)
   self.assertEqual(json.loads(path.read_text())['state'],'SPAWN_INTENT')
   if stage=='full':self.assertEqual(calls,['pilot'])
   calls.append(stage);call=SimpleNamespace(object_id='fc-'+stage,get=lambda:result if stage=='pilot' else {'status':'FULL_COMPLETE'})
   saved[call.object_id]=call;return call
  tree=ast.parse((ROOT/'scripts/modal_exp23_executor.py').read_text())
  function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate');function.decorator_list=[]
  namespace={'Path':lambda p:root if p=='/artifacts' else Path(p),'NS':worker.NS,'json':json,
   'artifacts':SimpleNamespace(reload=lambda:None,commit=lambda:None),'execute':SimpleNamespace(spawn=spawn),
   'modal':SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda key:saved[key]))}
  exec(compile(ast.Module(body=[function],type_ignores=[]),'offline_actual_coordinator','exec'),namespace)
  with patch.object(worker,'audit_stage',return_value={'H5':False}):
   with self.assertRaisesRegex(ValueError,'Independent pilot failed'):namespace['coordinate'](value)
  self.assertEqual(calls,['pilot'])
  with patch.object(worker,'audit_stage',return_value=pilot_audit):
   namespace['coordinate'](value);namespace['coordinate'](value)
  self.assertEqual(calls,['pilot','full'])
  path=launch_audit.ledger_path(volume,value,'full');ledger=json.loads(path.read_text());ledger['state']='SPAWN_INTENT';path.write_text(json.dumps(ledger))
  with patch.object(worker,'audit_stage',return_value=pilot_audit):
   with self.assertRaises(ValueError):namespace['coordinate'](value)
  self.assertEqual(calls,['pilot','full'])

 def test_launch_auditor_replays_pilots_and_rejects_wrong_backend_calls(self):
  value=fixture_payload()
  for name in value['input_sources']:
   path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
   value['input_sources'][name]=worker.hash_bytes(path)
  for component in value['readiness']['components'].values():
   component['receipt_sha256']=value['input_sources'][component['receipt_path']]
  for name in value['source_hashes']:
   path=self.root/'scripts'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source fixture')
   value['source_hashes'][name]=worker.hash_bytes(path)
  value['payload_sha256']=worker.digest({k:v for k,v in value.items() if k!='payload_sha256'})
  run,status=self.run_stage(value,'pilot')
  payload_path='executor_inputs/fixture.json';worker.write_exclusive(self.root/payload_path,value)
  base={'schema':'exp23_executor_call_v1','payload_sha256':value['payload_sha256'],'payload_path':payload_path,
   'state':'SPAWNED','worker_function':'execute','worker_module':'modal_exp23_executor','gpu':'H200:1',
   'spawn_intent_utc':status['finished_utc'],'submitted_utc':status['finished_utc']}
  worker.write_exclusive(launch_audit.ledger_path(self.root,value,'pilot'),{**base,'stage':'pilot','call_id':'fc-pilot'})
  worker.write_exclusive(self.root/'group_calls'/value['experiment']/'pilot_worker_started.json',
   {'call_id':'fc-pilot','payload_sha256':value['payload_sha256'],'stage':'pilot'})
  full_path=launch_audit.ledger_path(self.root,value,'full')
  worker.write_exclusive(full_path,{**base,'stage':'full','call_id':'fc-full','pilot_call_id':'fc-pilot',
   'pilot_audit_sha256':worker.digest(status['audit']),'pilot_completed_utc':status['finished_utc']})
  worker.write_exclusive(self.root/'infrastructure/resource_allocations.json',{'jobs':[{'job':value['central_job'],
   'app_id':'ap-fixture','gpus':2,'tp8_workers':0,'admitted_upper_bound_usd':14,
   'command':['modal','run','scripts/modal_exp23_executor.py','--attempt',value['attempt'],'--input-path',payload_path,'--launch'],
   'frozen_files':{**value['input_sources'],**{'scripts/'+k:v for k,v in value['source_hashes'].items()},
                   payload_path:worker.hash_bytes(self.root/payload_path)},
   'resource_profile':{'source':'scripts/modal_exp23_executor.py'}}]})
  worker.write_exclusive(self.root/'infrastructure/budget_reservations.json',{'reservations':[{'job':value['central_job'],'upper_bound_usd':14,'admitted_upper_bound_usd':14}]})
  def lookup(key):return {'call_id':key,'function_name':'execute','module_name':'modal_exp23_executor',
                         'status':'SUCCESS' if key=='fc-pilot' else 'PENDING'}
  with patch('phase1_readiness.recompute',return_value=value['readiness']),\
       patch.object(launch_audit,'dependency_bundle',return_value=(value['design'],value['dependency_audit'])),\
       patch('launch_bounded_job.validate_resource_profile'):
   result=launch_audit.recompute(self.root,str(full_path.relative_to(self.root)),lookup_call=lookup,tokenizer=Tokenizer())
   self.assertTrue(result['exp11_overlap_authorized']);self.assertFalse(result['exp23_complete'])
   for change in ({'function_name':'records'},{'module_name':'modal_exp23_author'},{'status':'FAILURE'}):
    with self.subTest(change=change):
     with self.assertRaisesRegex(ValueError,'Backend call'):
      launch_audit.recompute(self.root,str(full_path.relative_to(self.root)),lookup_call=lambda key:{**lookup(key),**change},tokenizer=Tokenizer())
   allocation_path=self.root/'infrastructure/resource_allocations.json'
   original=json.loads(allocation_path.read_text())
   for field in ('payload_hash','input_path','attempt','duplicate_option','missing_source'):
    changed=json.loads(json.dumps(original));job=changed['jobs'][0]
    if field=='payload_hash':job['frozen_files'][payload_path]='0'*64
    elif field=='input_path':job['command'][job['command'].index('--input-path')+1]='executor_inputs/swapped.json'
    elif field=='attempt':job['command'][job['command'].index('--attempt')+1]='swapped_attempt'
    elif field=='duplicate_option':job['command'].extend(['--input-path',payload_path])
    else:del job['frozen_files']['scripts/exp23_executor_worker.py']
    allocation_path.write_text(json.dumps(changed))
    with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Central'):
     launch_audit.recompute(self.root,str(full_path.relative_to(self.root)),lookup_call=lookup,tokenizer=Tokenizer())
   allocation_path.write_text(json.dumps(original))


if __name__=='__main__':unittest.main()
