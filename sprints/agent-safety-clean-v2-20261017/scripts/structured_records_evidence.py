"""Independent actual-call, central admission and prompt replay for record jobs.

Prepared current-sprint audit only. No generation, launch, resource mutation or
receipt persistence. Default complete audits recheck Phase1 from raw evidence.
"""
from pathlib import Path
import importlib.metadata
import json
import re


def require(value,message):
 if not value:raise ValueError(message)


def inside(root,name):
 require(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts,'Structured evidence path outside sprint')
 root=Path(root).resolve();path=(root/name).resolve()
 require(path.is_relative_to(root) and path!=root,'Structured evidence path outside sprint')
 return path


def option(command,name):
 require(isinstance(command,list) and all(isinstance(v,str) for v in command),'Central command must be argument list')
 positions=[i for i,v in enumerate(command) if v==name or v.startswith(name+'=')]
 require(len(positions)==1,'Central command must bind exactly one '+name)
 index=positions[0]
 require(command[index]==name and index+1<len(command),'Central option requires exact separate value: '+name)
 return command[index+1]


def source_bindings(root,payload):
 from structured_records_worker import hash_bytes
 root=Path(root).resolve();expected={}
 def add(name,h):
  require(isinstance(h,str) and bool(re.fullmatch('[0-9a-f]{64}',h)),'Invalid source digest')
  require(name not in expected or expected[name]==h,'Conflicting structured source bindings')
  require(hash_bytes(inside(root,name))==h,'Frozen structured source changed: '+name)
  expected[name]=h
 for name,h in payload['input_sources'].items():add(name,h)
 for name,h in payload['source_hashes'].items():
  require(Path(name).name==name,'Runtime source path not a basename');add('scripts/'+name,h)
 for name in ('scripts/modal_structured_records.py','scripts/structured_records_evidence.py','scripts/historical_admission.py'):
  add(name,hash_bytes(inside(root,name)))
 return expected


def audit_central_admission(root,payload,input_path,*,require_app_id=False):
 from structured_records_worker import validate_payload,hash_bytes,digest
 from historical_admission import validate_historical_admission
 root=Path(root).resolve();validate_payload(payload)
 canonical='stimuli/'+payload['experiment']+'_input.json'
 require(input_path==canonical,'Structured central input is not canonical immutable payload')
 path=inside(root,input_path)
 require(json.loads(path.read_text())==payload,'Central input differs from exact structured payload')
 require(payload['central_job']==payload['experiment'],'Unexpected structured central job identity')
 jobs=json.loads((root/'infrastructure/resource_allocations.json').read_text())['jobs']
 selected=[j for j in jobs if j['job']==payload['central_job']]
 require(len(selected)==1,'Missing unique central structured admission');job=selected[0]
 require(not require_app_id or isinstance(job.get('app_id'),str) and bool(re.fullmatch('ap-[A-Za-z0-9_-]+',job['app_id'])),
         'Structured central app ID not durably bound')
 require(job['gpus']==2 and job['single_gpu_workers']==2 and job['tp8_workers']==0,'Unexpected structured app topology')
 command=job['command']
 require(option(command,'--attempt')==payload['attempt'] and option(command,'--input-path')==input_path,'Central CLI binds different structured input/attempt')
 require(command.count('--launch')==1 and not any(v.startswith('--launch=') for v in command),'Ambiguous structured launch flag')
 profile=job['resource_profile']
 require(profile['source']=='scripts/modal_structured_records.py' and set(profile['functions'])=={'records','coordinate'} and
         profile['excluded_functions']=={},'Central structured resource profile targets different functions')
 require(all(profile['functions'][name]['max_calls']==1 for name in ('records','coordinate')),'Exactly one structured GPU call and one coordinator admitted')
 budget=json.loads((root/'infrastructure/budget_reservations.json').read_text());reservations=budget['reservations']
 selected=[r for r in reservations if r['job']==payload['central_job']]
 require(len(selected)==1,'Missing unique structured budget reservation')
 checked=validate_historical_admission(job,selected[0],budget.get('reconciliations',[]),root)
 if not require_app_id:
  require(selected[0]['upper_bound_usd']==job['admitted_upper_bound_usd'],'Reconciled historical funding cannot authorize a fresh structured launch')
 expected=source_bindings(root,payload);expected[input_path]=hash_bytes(path)
 frozen=job.get('frozen_files',{})
 require(isinstance(frozen,dict),'Structured frozen central source map missing')
 for name,h in expected.items():require(frozen.get(name)==h,'Central admission did not freeze exact structured input/source: '+name)
 return {'schema':'structured_central_admission_audit_v1','job':payload['central_job'],
  'app_id':job.get('app_id'),'payload_sha256':payload['payload_sha256'],'input_path':input_path,'input_sha256':hash_bytes(path),
  'profile_sha256':digest(profile),'source_sha256':dict(sorted(expected.items())),
  'maximum_gpu_calls':1,'maximum_coordinator_calls':1,'resource_profile_audit':checked,'actual_spend_claimed':False}


def audit_complete_run(root,payload,*,tokenizer=None,lookup_call=None):
 from structured_records_worker import validate_payload,audit_streams,hash_bytes,read_chain,render_request,digest
 from phase1_readiness import recompute as phase1_recompute
 from audit_exp23_launch import lookup_backend
 root=Path(root).resolve();validate_payload(payload)
 central=audit_central_admission(root,payload,'stimuli/'+payload['experiment']+'_input.json',require_app_id=True)
 sources=source_bindings(root,payload)
 require(phase1_recompute(root,payload['component_sources'])==payload['readiness'] and
         payload['readiness']['primary_phase2_authorized'] is True,'Structured Phase1 source selection no longer independently passes')
 require(json.loads(inside(root,payload['request_source_path']).read_text())==payload['requests'],'Frozen structured request source differs')
 require(set(payload['tokenizer_files'])==set(payload['tokenizer_source_paths']) and
         {'tokenizer_config.json','tokenizer.json'}<=set(payload['tokenizer_files']),'Missing exact tokenizer source map')
 folder=root/'raw_transcripts'/payload['experiment']
 for name,h in payload['tokenizer_files'].items():
  require(Path(name).name==name,'Tokenizer path not a basename')
  source=payload['tokenizer_source_paths'][name]
  require(payload['input_sources'].get(source)==h and hash_bytes(inside(root,source))==h,'Tokenizer source hash differs')
  require(hash_bytes(folder/'tokenizer'/name)==h,'Captured structured tokenizer bytes differ')
 context_path=folder/'runtime_context.json';context=json.loads(context_path.read_text())
 require(context['payload_sha256']==payload['payload_sha256'] and context['identity']==payload['identity'] and
         context['tokenizer_files']==payload['tokenizer_files'],'Structured runtime context identity differs')
 if tokenizer is None:
  for name in ('transformers','tokenizers','jinja2'):
   require(isinstance(context['runtime_versions'].get(name),str) and
           importlib.metadata.version(name)==context['runtime_versions'][name],'Exact structured prompt audit requires captured '+name+' version')
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(str(folder/'tokenizer'),local_files_only=True,trust_remote_code=False)
 ledger_path=root/'group_calls'/f'{payload["experiment"]}.json';ledger=json.loads(ledger_path.read_text())
 require(ledger['state']=='SPAWNED' and ledger['worker_module']=='modal_structured_records' and ledger['worker_function']=='records' and
         ledger['job']==payload['central_job'] and ledger['payload_sha256']==payload['payload_sha256'] and
         ledger['experiment']==payload['experiment'] and ledger['role']==payload['model_role'] and ledger['task']==payload['task'],
         'Structured worker call ledger identity differs')
 require(isinstance(ledger['call_id'],str) and bool(re.fullmatch('fc-[A-Za-z0-9_-]+',ledger['call_id'])),'Structured actual call ID absent')
 start_path=root/'structured_worker_starts'/f'{payload["experiment"]}.json';started=json.loads(start_path.read_text())
 require(started=={'payload_sha256':payload['payload_sha256'],'call_id':ledger['call_id']},'Actual structured worker start differs from durable admission')
 backend=(lookup_call or lookup_backend)(ledger['call_id'])
 require(backend['call_id']==ledger['call_id'] and backend['function_name']=='records' and
         backend['module_name'].split('.')[-1]=='modal_structured_records' and backend['status']=='SUCCESS',
         'Backend does not verify completed structured worker identity')
 audit=audit_streams(root,payload)
 require(audit['complete_data'] is True and all(audit[k] is True for k in ('H1','H2','H4','H5','all_records_valid')),
         'Structured raw run is incomplete or failed hygiene')
 rows,_=read_chain(folder/f'{payload["model_role"]}.jsonl',payload['payload_sha256'])
 requests={r['trial_id']:r for r in payload['pilots']+payload['requests']}
 for row in rows:
  messages,prompt,ids=render_request(requests[row['trial_id']],tokenizer,payload['identity']['family'])
  require(row['rendered_messages']==messages and row['full_prompt']==prompt and row['prompt_token_ids']==ids and
          row['tokenizer_file_sha256']==payload['tokenizer_files'],'Independent structured prompt/tokenizer replay differs')
 status_path=root/'qualification_status'/f'{payload["experiment"]}.json';status=json.loads(status_path.read_text())
 require(status['status']=='RECORDS_COMPLETE' and status['payload_sha256']==payload['payload_sha256'] and
         status['stream_audit']==audit,'Structured saved completion differs from independent raw audit')
 # Recheck after slow raw/backend audits: changing evidence never yields PASS.
 require(source_bindings(root,payload)==sources,'Structured immutable sources changed during audit')
 require(audit_central_admission(root,payload,central['input_path'],require_app_id=True)==central,'Structured central admission changed during audit')
 require(json.loads(ledger_path.read_text())==ledger and json.loads(start_path.read_text())==started and
         json.loads(context_path.read_text())==context,'Structured runtime/call evidence changed during audit')
 return {**audit,'schema':'structured_complete_execution_audit_v1','stream_audit':audit,
  'central_admission':central,'actual_call_id':ledger['call_id'],'backend_identity_verified':True,
  'worker_start_sha256':hash_bytes(start_path),'call_ledger_sha256':hash_bytes(ledger_path),
  'runtime_context_sha256':hash_bytes(context_path),'status_sha256':hash_bytes(status_path),
  'exact_prompt_replay':True,'phase1_prerequisites_recomputed':True,'source_sha256':sources,
  'phase2_executor_launch':False,'complete_experiment_receipt':False}
