"""Prepared two-GPU-call protocol; root alone may centrally admit this app."""
from pathlib import Path
import json,modal
ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-exp23-executor')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
RUNTIME_FILES=('exp23_executor_worker.py','audit_exp23_launch.py','exp23_design.py','exp23_author_worker.py',
 'structured_records_worker.py','sandbox_runtime.py','strict_parser.py','scenario_contract.py',
 'primary_model_loader.py','primary_fp8_inventory.py','llama_model_loader.py','llama_serving_worker.py')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12').entrypoint([])
 .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn'}))
for name in RUNTIME_FILES:image=image.add_local_file(ROOT/'scripts'/name,'/opt/cleanroom/'+name)


@app.function(image=image,gpu='H200',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
 timeout=3600,startup_timeout=300,retries=0,max_containers=2,scaledown_window=2,
 volumes={'/weights':weights,'/artifacts':artifacts})
def execute(payload,stage):
 import os,subprocess,sys,time
 sys.path.insert(0,'/opt/cleanroom')
 from exp23_executor_worker import validate_payload,require,write_exclusive
 from audit_exp23_launch import audit_durable_call
 validate_payload(payload);root=Path('/artifacts')/NS;weights.reload()
 # A spawned GPU call may start before the coordinator commits its call ID.
 # Never generate until that exact actual FunctionCall is durable.
 for _ in range(30):
  artifacts.reload()
  try:
   ledger=audit_durable_call(root,payload,stage)
   require(ledger['call_id']==modal.current_function_call_id(),'Worker call differs from durable admission')
   break
  except FileNotFoundError:time.sleep(1)
  except ValueError as exc:
   if 'Missing committed executor GPU call' not in str(exc):raise
   time.sleep(1)
 else:raise RuntimeError('Call ID admission never committed; no generation performed')
 started=root/'group_calls'/payload['experiment']/(stage+'_worker_started.json')
 write_exclusive(started,{'call_id':modal.current_function_call_id(),'payload_sha256':payload['payload_sha256'],'stage':stage})
 artifacts.commit()
 path=root/'executor_inputs'/(payload['experiment']+'.json')
 require(json.loads(path.read_text())==payload,'Worker payload differs from frozen input')
 environment=dict(os.environ,TRITON_CACHE_DIR='/tmp/exp23-executor/'+payload['experiment']+'/'+stage)
 child=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/exp23_executor_worker.py',str(path),stage],
  env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 try:
  for line in child.stdout:
   if line.strip()=='COMMIT_ARTIFACTS':artifacts.commit();child.stdin.write('COMMITTED\n');child.stdin.flush()
   else:print(line,end='',flush=True)
  code=child.wait()
 finally:
  if child.poll() is None:
   child.terminate()
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  artifacts.commit()
 if code:raise RuntimeError('Executor stage failed; preserve attempt and reconcile without a repeated spawn')
 return json.loads((root/'evaluation_status'/(payload['experiment']+'_'+stage+'.json')).read_text())


@app.function(image=image,cpu=(1,1),memory=(4096,4096),timeout=8400,startup_timeout=300,
 retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':artifacts})
def coordinate(payload):
 import sys
 sys.path.insert(0,'/opt/cleanroom')
 from exp23_executor_worker import validate_payload,require,write_exclusive,utc_now,audit_stage,digest,paths,hash_bytes
 from audit_exp23_launch import ledger_path,audit_durable_call
 validate_payload(payload);artifacts.reload();root=Path('/artifacts')/NS
 for name,h in payload['input_sources'].items():require(hash_bytes(root/name)==h,'Frozen input differs before pilot')
 target=root/'executor_inputs'/(payload['experiment']+'.json')
 if target.exists():require(json.loads(target.read_text())==payload,'Attempt was rebound')
 else:write_exclusive(target,payload)
 artifacts.commit()
 def stage_call(stage,prior=None,status=None):
  path=ledger_path(root,payload,stage)
  if path.exists():
   ledger=audit_durable_call(root,payload,stage)
   return modal.FunctionCall.from_id(ledger['call_id']),ledger
  ledger={'schema':'exp23_executor_call_v1','state':'SPAWN_INTENT','stage':stage,
   'payload_sha256':payload['payload_sha256'],'payload_path':str(target.relative_to(root)),
   'worker_function':'execute','worker_module':'modal_exp23_executor','gpu':'H200:1','spawn_intent_utc':utc_now()}
  if stage=='full':
   ledger.update(pilot_call_id=audit_durable_call(root,payload,'pilot')['call_id'],
    pilot_audit_sha256=digest(prior),pilot_completed_utc=status['finished_utc'])
  write_exclusive(path,ledger);artifacts.commit()
  call=execute.spawn(payload,stage)
  ledger.update(state='SPAWNED',call_id=call.object_id,submitted_utc=utc_now())
  temporary=path.with_suffix('.submitted.json');write_exclusive(temporary,ledger);temporary.replace(path);artifacts.commit()
  return call,ledger
 call,_=stage_call('pilot');result=call.get();artifacts.reload()
 prior=audit_stage(root,payload,'pilot')
 require(result['status']=='PILOT_PASS' and result['audit']==prior and prior['H5'],'Independent pilot failed; full call forbidden')
 status=json.loads(paths(root,payload,'pilot')['status'].read_text());require(status==result,'Durable pilot result differs')
 for name,h in payload['input_sources'].items():require(hash_bytes(root/name)==h,'Frozen input changed before full call')
 call,ledger=stage_call('full',prior,status)
 return call.get()


def prepare(root,attempt,*,executor,mode,conditions,bundle_path,runs_path,components_path,tokenizer_metadata):
 from phase1_readiness import recompute
 from audit_exp23_launch import dependency_bundle,inside
 from exp23_executor_worker import build_payload,hash_bytes,require,RUNTIME_FILES as WORKER_FILES
 from admit_parallel_calibration import ReadBindings
 import sys
 root=Path(root).resolve();bindings=ReadBindings();sys.addaudithook(bindings.hook)
 component_file=inside(root,components_path);readiness=recompute(root,json.loads(component_file.read_text()))
 require(readiness['primary_phase2_authorized'] is True,'Phase1 independent prerequisites incomplete/failed')
 design,audit=dependency_bundle(root,bundle_path,runs_path)
 folder=inside(root,tokenizer_metadata);tokenizers={}
 for path in sorted(folder.iterdir()):
  if path.is_file():
   require(path.name in {'tokenizer_config.json','tokenizer.json','special_tokens_map.json','chat_template.jinja','added_tokens.json','vocab.json','merges.txt'},'Unexpected tokenizer artifact')
   tokenizers[path.name]=hash_bytes(path)
 checkpoint_hash=None
 if executor=='primary_intact_executor':checkpoint_hash=hash_bytes(root/'checkpoints_full_residual_v2/intact_materialization.json')
 else:
  for name in ('infrastructure/nvidia__Llama-3.3-70B-Instruct-FP8_download.json','infrastructure/llama_preflight.json'):
   value=json.loads((root/name).read_text())
   require(value['repo_id']=='nvidia/Llama-3.3-70B-Instruct-FP8' and value['revision']=='fde04ee76a27704c88f569542ef023b57d4d0362','Llama pinned provenance differs')
 require(set(RUNTIME_FILES)==set(WORKER_FILES),'Image packaging differs')
 source_hashes={name:hash_bytes(root/'scripts'/name) for name in RUNTIME_FILES}
 for name in ('FINAL_RESEARCH_SPECIFICATION.md','scripts/modal_exp23_executor.py','scripts/phase1_readiness.py',
              'scripts/historical_admission.py','scripts/structured_records_evidence.py'):
  bindings.record(root/name)
 # These shared control ledgers grow when this very app is admitted. The exact
 # selected upstream central rows/profiles are independently reconstructed by
 # dependency_bundle at admission; global ledger bytes are not scientific input.
 mutable={'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}
 sources=dict(sorted((name,h) for name,h in bindings.files.items() if name not in mutable))
 # Every locally read prerequisite and source is rechecked after the audit.
 for name,h in sources.items():require(hash_bytes(root/name)==h,'Input changed during admission audit')
 return build_payload(attempt,design,executor=executor,mode=mode,conditions=conditions,
  readiness=readiness,input_sources=sources,source_hashes=source_hashes,tokenizer_files=tokenizers,
  dependency_audit=audit,components_path=str(component_file.relative_to(root)),checkpoint_sha256=checkpoint_hash)


def resource_config(root,attempt,input_path):
 from launch_bounded_job import _decorators,validate_resource_profile
 from exp23_executor_worker import hash_bytes
 root=Path(root);source='scripts/modal_exp23_executor.py';limits=_decorators((root/source).read_text())
 payload=json.loads((root/input_path).read_text())
 config={'job':'exp23_executor_'+attempt,'gpus':2,'single_gpu_workers':2,'tp8_workers':0,'upper_bound_usd':14,
  'log':f'infrastructure/exp23_executor_{attempt}.log',
  'command':[str(root/'.venv/bin/modal'),'run','--detach',source,'--attempt',attempt,'--input-path',input_path,'--launch'],
  'frozen_files':{**payload['input_sources'],input_path:hash_bytes(root/input_path)},
  'resource_profile':{'source':source,'source_sha256':hash_bytes(root/source),
   'functions':{'execute':{'max_calls':2,'limits':limits['execute']},'coordinate':{'max_calls':1,'limits':limits['coordinate']}},
   'excluded_functions':{},'call_bound_basis':'Exactly one five-episode pilot GPU call, then at most one full GPU call after independent H5; one coordinator, retries=0.'}}
 validate_resource_profile(config,root);return config


@app.local_entrypoint()
def main(attempt:str,input_path:str,launch:bool=False):
 from exp23_executor_worker import validate_payload,require,hash_bytes
 from audit_exp23_launch import inside
 from phase1_readiness import recompute
 path=inside(ROOT,input_path);payload=json.loads(path.read_text());validate_payload(payload)
 require(payload['attempt']==attempt,'Attempt differs from frozen payload')
 if not launch:
  print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','resource_config':resource_config(ROOT,attempt,input_path)}));return
 ledger=json.loads((ROOT/'infrastructure/resource_allocations.json').read_text())
 admitted=[row for row in ledger['jobs'] if row['job']==payload['central_job']]
 require(len(admitted)==1 and admitted[0].get('frozen_files',{}).get(str(path.relative_to(ROOT)))==hash_bytes(path),
         'Root central admission has not bound this exact immutable executor payload')
 require(recompute(ROOT,json.loads(inside(ROOT,payload['components_path']).read_text()))==payload['readiness'],'Independent launch-time Phase1 readiness differs')
 from audit_exp23_launch import dependency_bundle
 design,audit=dependency_bundle(ROOT,payload['dependency_audit']['bundle_path'],payload['dependency_audit']['runs_path'])
 require(design==payload['design'] and audit==payload['dependency_audit'],'Launch-time dependency audit differs')
 print(json.dumps(coordinate.remote(payload)))
