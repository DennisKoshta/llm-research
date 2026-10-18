"""Bounded Phase 1 dose supervisor, prepared for explicit parent startup only.

Materialization uses the already frozen refusal direction. Dose evaluation waits
for independently reproduced primary and Intact-calibration PASS. Paid admission
is exclusively delegated to launch_bounded_job; this script never starts Phase 2.
"""
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import argparse,errno,fcntl,hashlib,json,os,re,subprocess,sys,tempfile,time
from admit_parallel_calibration import ReadBindings,atomic_bytes,digest
from launch_bounded_job import launch,validate_resource_profile,_decorators
ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
FAMILY='checkpoint_doses_v1'
NEW_DOSES=('alpha_m0_5','alpha_p0_5','alpha_p1_5')
ARMS=('dose_m0_5','dose_0_0','dose_p0_5','dose_1_0','dose_p1_5')
MATERIALIZATION_JOB='phase1_dose_materialization_checkpoint_doses_v1'
MATERIALIZATION_RESERVATION=12
EVALUATION_RESERVATION=30
NETWORK_OPERATIONS={'sync','upload','materialization-job-state','evaluation-job-state'}
MAX_CONSECUTIVE_NETWORK_RETRIES=8


class TransientTransportError(RuntimeError):
 """A bounded remote observation failed; this never authorizes a new job."""


def transient_transport(exc):
 """Classify transport failures by type, never arbitrary scientific error text."""
 if isinstance(exc,(AssertionError,ValueError,FileNotFoundError)):return False
 if isinstance(exc,(ConnectionError,TimeoutError,subprocess.TimeoutExpired)):return True
 if isinstance(exc,OSError) and exc.errno in {errno.ECONNRESET,errno.ECONNREFUSED,errno.ETIMEDOUT,
     errno.EHOSTUNREACH,errno.ENETUNREACH,errno.EPIPE}:return True
 kind=type(exc)
 if kind.__module__.startswith(('modal.','grpclib.','grpc.')) and kind.__name__ in {
     'ResourceExhaustedError','ServiceError','ConnectionError','StreamTerminatedError'}:return True
 if kind.__module__.startswith(('grpclib.','grpc.')):
  status=getattr(exc,'status',None)
  if getattr(status,'name',None) in {'RESOURCE_EXHAUSTED','UNAVAILABLE','DEADLINE_EXCEEDED'}:return True
 # modal app list executes in a bounded child; only explicit SDK transport
 # exception lines are recognized, not general subprocess failures.
 if isinstance(exc,subprocess.CalledProcessError):
  error=(exc.stderr or '')+'\n'+(exc.output or '')
  return any(re.search(r'(?m)^'+re.escape(name)+r':',error) for name in (
      'modal.exception.ResourceExhaustedError','modal.exception.ConnectionError',
      'modal.exception.ServiceError'))
 return False


def network_retry_delay(failures,poll_seconds):
 assert 1<=failures<=MAX_CONSECUTIVE_NETWORK_RETRIES
 return min(300,max(45,poll_seconds)*2**(failures-1))
MATERIALIZATION_SOURCES=[
 'FINAL_RESEARCH_SPECIFICATION.md','ENGINEERING_CORRECTIONS.md','scripts/admit_phase1_doses_d3.py',
 'scripts/admit_parallel_calibration.py','scripts/launch_bounded_job.py','scripts/modal_materialize_doses.py',
 'scripts/checkpoint_math.py','scripts/audit_source_directions.py','scripts/prepare_directions.py']
EVALUATION_SOURCES=[
 'FINAL_RESEARCH_SPECIFICATION.md','scripts/admit_phase1_doses_d3.py','scripts/admit_parallel_calibration.py',
 'scripts/launch_bounded_job.py','scripts/modal_dose_evaluation_d3.py','scripts/prepare_dose_evaluation_d3.py',
 'scripts/scenario_contract.py','scripts/audit_authored_scenarios.py','scripts/audit_prompt_boilerplate.py',
 'scripts/phase1_evidence.py','scripts/paired_statistics.py','scripts/audit_sandbox_episodes.py',
 'scripts/prepare_calibration.py','scripts/primary_model_loader.py','scripts/primary_fp8_inventory.py',
 'scripts/strict_parser.py','scripts/sandbox_runtime.py','scripts/sandbox_evaluation_worker.py',
 'scripts/audit_dose_response_d3.py','scripts/audit_calibration_provenance.py','scripts/raw_generation_evidence.py',
 'scripts/prepare_dose_evaluation.py','scripts/modal_dose_evaluation.py','scripts/audit_dose_response.py',
 'scripts/admit_phase1_doses.py']


def now():return datetime.now(timezone.utc).isoformat()


def simple_id(value):
 if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',value):raise argparse.ArgumentTypeError('Use a lowercase simple attempt identifier')
 return value


def inside(relative):
 path=(ROOT/relative).resolve()
 if not path.is_relative_to(ROOT):raise ValueError('Path escapes clean sprint')
 return path


def save(path,value,immutable=False):
 atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),immutable=immutable)


def snapshot(args,stage):return ROOT/f'executions/phase1_dose_supervisor_{args.attempt}/{stage}'


def cache_path(args,stage):return snapshot(args,stage)/'authorization_cache.json'


def evaluation_config_path(args):
 if getattr(args,'descriptive_amendment',None):
  return f'stimuli/dose_evaluation_{args.author_attempt}_{args.attempt}.json'
 return f'stimuli/dose_evaluation_{args.author_attempt}.json'


def environment():
 import numpy,sklearn
 return {'python':sys.version,'numpy':numpy.__version__,'sklearn':sklearn.__version__}


def verify_cache(args,stage):
 cache=json.loads(cache_path(args,stage).read_text())
 assert cache['attempt']==args.attempt and cache['author_attempt']==args.author_attempt and cache['stage']==stage
 assert cache.get('descriptive_amendment')==getattr(args,'descriptive_amendment',None)
 assert cache['environment']==environment(),'Cached audit environment changed'
 for relative,expected in cache['read_prerequisite_sha256'].items():
  assert digest(inside(relative))==expected,f'Frozen prerequisite changed: {relative}'
 return cache


def freeze_cache(args,stage,bindings,result):
 files=dict(sorted(bindings.files.items()))
 cache={'schema':'phase1_dose_supervisor_authorization_v1','stage':stage,'attempt':args.attempt,
        'author_attempt':args.author_attempt,'result':result,'environment':environment(),
        'read_prerequisite_sha256':files,'phase2_authorized':False,
        'binding_scope':'Every sprint file read by the independent audit, plus explicit execution/audit sources; installed package versions recorded separately.'}
 if getattr(args,'descriptive_amendment',None):cache['descriptive_amendment']=args.descriptive_amendment
 for relative,expected in files.items():assert digest(inside(relative))==expected,f'Prerequisite changed during audit: {relative}'
 save(cache_path(args,stage),cache,immutable=True)
 return cache


def prepare_materialization(args):
 raise AssertionError('D3 does not authorize new materialization')
 if cache_path(args,'materialization').exists():return verify_cache(args,'materialization')
 bindings=ReadBindings();sys.addaudithook(bindings.hook)
 for relative in MATERIALIZATION_SOURCES:
  path=inside(relative)
  if not path.is_file():raise FileNotFoundError(path)
  bindings.record(path)
 from modal_materialize_doses import prepare_payload
 payload=prepare_payload(ROOT)
 assert payload['namespace']==NS
 save(snapshot(args,'materialization')/'frozen_payload.json',payload,immutable=True)
 return freeze_cache(args,'materialization',bindings,{'direction_sha256':payload['direction_sha256'],
     'selected_layer':payload['selected_layer'],'materialization_authorized':True,
     'primary_behavior_inspected':False,'calibration_behavior_inspected':False})


def prepare_evaluation(args):
 if cache_path(args,'evaluation').exists():return verify_cache(args,'evaluation')
 bindings=ReadBindings();sys.addaudithook(bindings.hook)
 for relative in EVALUATION_SOURCES:
  path=inside(relative)
  if not path.is_file():raise FileNotFoundError(path)
  bindings.record(path)
 from prepare_dose_evaluation_d3 import prepare_evaluation_config,DESCRIPTIVE_SCOPE
 from phase1_evidence import recompute_primary
 from audit_sandbox_episodes import calibration_receipt
 from modal_dose_evaluation_d3 import authorize,primary_mechanical_checks
 config=prepare_evaluation_config(ROOT,args.author_attempt,args.attempt,getattr(args,'descriptive_amendment',None))
 config_path=ROOT/evaluation_config_path(args)
 save(config_path,config,immutable=True)
 # Freeze the complete config bytes as an input, not merely a generated result.
 bindings.record(config_path)
 primary_path=inside(args.primary_receipt);calibration_path=inside(args.calibration_receipt)
 primary=json.loads(primary_path.read_text());calibration=json.loads(calibration_path.read_text())
 assert primary==recompute_primary(ROOT,primary['attempt'],primary['activation_attempt'],
     primary.get('logit_stage','qualification'),primary.get('logit_attempt'),primary.get('logit_arm_attempts'))
 assert calibration==calibration_receipt(ROOT,calibration['attempt'],calibration['author_attempt'])
 assert calibration['author_attempt']==args.author_attempt
 gates={'primary_qualification_passed':bool(primary['primary_checkpoints_qualified']),
        'intact_calibration_passed':bool(calibration['calibration_passed'])}
 descriptive=config.get('eligibility_scope')==DESCRIPTIVE_SCOPE
 if descriptive:gates['primary_mechanical_hygiene_passed']=all(primary_mechanical_checks(primary).values())
 required=('primary_mechanical_hygiene_passed','intact_calibration_passed') if descriptive else tuple(gates)
 if not all(gates[key] for key in required):
  return freeze_cache(args,'evaluation',bindings,{'evaluation_authorized':False,'gates':gates,
       'reason':'Independent scientific prerequisite gate failed; no dose evaluation authorized.'})
 authorization=authorize(ROOT,config,primary_path,calibration_path)
 assert authorization['phase2_authorized'] is False
 save(snapshot(args,'evaluation')/'frozen_authorization.json',authorization,immutable=True)
 return freeze_cache(args,'evaluation',bindings,{'evaluation_authorized':True,'gates':gates,'authorization':authorization})


def resource_config(args,stage):
 if stage=='materialization':
  source='scripts/modal_materialize_doses.py';counts={'materialize_dose':3,'coordinate':1}
  command=[str(ROOT/'.venv/bin/modal'),'run','--detach',source]
  job=MATERIALIZATION_JOB;workers=1;amount=MATERIALIZATION_RESERVATION
  basis='Exactly three sequential fixed-alpha materialize_dose FunctionCalls plus one durable coordinator, retries=0; no evaluation calls.'
 else:
  source='scripts/modal_dose_evaluation_d3.py';counts={'evaluate':5,'coordinate':1}
  command=[str(ROOT/'.venv/bin/modal'),'run','--detach',source,'--config',evaluation_config_path(args),
           '--attempt',args.attempt,'--primary-receipt',args.primary_receipt,'--calibration-receipt',args.calibration_receipt]
  if getattr(args,'descriptive_amendment',None):command+=['--descriptive-amendment',args.descriptive_amendment]
  job=f'phase1_dose_evaluation_{args.attempt}';workers=4;amount=EVALUATION_RESERVATION
  basis='Exactly five fixed-alpha evaluate FunctionCalls plus one durable coordinator, retries=0; max four simultaneous GPU workers after the TP8 author app stops.'
 raw=inside(source).read_bytes();decorators=_decorators(raw.decode())
 assert set(decorators)==set(counts),'Unexpected remotely executable function; review budget and call graph'
 config={'job':job,'gpus':workers,'single_gpu_workers':workers,'tp8_workers':0,'upper_bound_usd':amount,
  'log':f'infrastructure/dose_{stage}_{args.attempt}.log','command':command,
  'resource_profile':{'source':source,'source_sha256':hashlib.sha256(raw).hexdigest(),
   'functions':{name:{'max_calls':number,'limits':decorators[name]} for name,number in counts.items()},
   'excluded_functions':{},'call_bound_basis':basis}}
 if cache_path(args,stage).exists():
  cache=json.loads(cache_path(args,stage).read_text())
  config['frozen_files']=cache['read_prerequisite_sha256']
 audit=validate_resource_profile(config)
 assert audit['whole_app_allocation']['gpus']==workers
 return config,audit


def freeze_sources(args,stage,config,profile):
 files=MATERIALIZATION_SOURCES if stage=='materialization' else EVALUATION_SOURCES
 hashes={}
 for relative in files:
  content=inside(relative).read_bytes();hashes[relative]=hashlib.sha256(content).hexdigest()
  atomic_bytes(snapshot(args,stage)/'source_files'/relative,content,immutable=True)
 save(snapshot(args,stage)/'source_manifest.json',hashes,immutable=True)
 save(snapshot(args,stage)/'resource_config.json',config,immutable=True)
 save(snapshot(args,stage)/'resource_bound.json',profile,immutable=True)


def sync_once(args):
 import modal
 volume=modal.Volume.from_name('curiosity-study-artifacts')
 stem=f'phase1_dose_response_{args.attempt}'
 # A terminal status is committed only after its raw/trajectory writes finish.
 # Wait for all five immutable terminal commits before downloading JSONL, so
 # growing streams cannot race the listing/read integrity check.
 terminal={}
 for arm in ARMS:
  remote=f'{NS}/evaluation_status/{stem}_{arm}_cot_on.json'
  try:data=b''.join(volume.read_file(remote))
  except (modal.exception.NotFoundError,FileNotFoundError):continue
  row=json.loads(data)
  assert row['attempt']==args.attempt and row['arm']==arm
  assert row['status'] in {'COMPLETE','H1_FAILED','H5_FAILED'}
  terminal[arm]=row['status']
 if len(terminal)!=5:
  return {'state':'WAITING_FOR_IMMUTABLE_TERMINAL_COMMITS','terminal_worker_statuses':terminal,
          'files_synced':0,'live_raw_streams_downloaded':False}
 cache_file=ROOT/f'infrastructure/phase1_doses_{args.attempt}_sync_cache.json'
 cache=json.loads(cache_file.read_text()) if cache_file.exists() else {}
 material_names={'materialization_set.json',*[f'{dose}_materialization.json' for dose in NEW_DOSES]}
 raw_names={f'{arm}_cot_on{suffix}.jsonl' for arm in ARMS for suffix in ('','_requests')}
 trajectory_names={f'{arm}_cot_on.jsonl' for arm in ARMS}
 status_names={f'{stem}_{arm}_cot_on.json' for arm in ARMS}|{f'{stem}_group.json'}
 pilot_names={f'{stem}_{arm}_cot_on_pilot_audit.md' for arm in ARMS}
 ledger_names={'frozen_input.json',*[f'{arm}.json' for arm in ARMS]}
 payload_names={f'{stem}_{arm}_cot_on.json' for arm in ARMS}
 receipt_names=set()
 for relative in (args.primary_receipt,args.calibration_receipt):
  assert PurePosixPath(relative).parent==PurePosixPath('records'),'Prerequisite receipts must live in records/'
  receipt_names.add(PurePosixPath(relative).name)
 scopes={FAMILY:material_names,f'raw_transcripts/{stem}':raw_names,
         f'trajectories/{stem}':trajectory_names,'evaluation_status':status_names,'pilots':pilot_names,
         f'group_calls/{stem}':ledger_names,'sandbox_inputs':payload_names,'records':receipt_names}
 copied=unchanged=0
 for directory,names in scopes.items():
  try:entries=list(volume.iterdir(f'{NS}/{directory}',recursive=False))
  except (modal.exception.NotFoundError,FileNotFoundError):continue
  for entry in entries:
   remote=PurePosixPath(str(entry.path).lstrip('/'))
   assert remote.parts[0]==NS and '..' not in remote.parts,'Out-of-namespace artifact'
   relative=PurePosixPath(*remote.parts[1:])
   assert relative.parent==PurePosixPath(directory),'Unexpected nested artifact'
   if relative.name not in names:continue
   local=inside(str(relative));fingerprint={'mtime':entry.mtime,'size':entry.size}
   if local.exists() and local.stat().st_size==entry.size and cache.get(str(relative))==fingerprint:
    unchanged+=1;continue
   local.parent.mkdir(parents=True,exist_ok=True)
   descriptor,name=tempfile.mkstemp(prefix=local.name+'.',suffix='.downloading',dir=local.parent)
   temporary=Path(name)
   try:
    with os.fdopen(descriptor,'wb') as handle:
     for chunk in volume.read_file(f'{NS}/{relative}'):handle.write(chunk)
     handle.flush();os.fsync(handle.fileno())
    if temporary.stat().st_size!=entry.size:raise RuntimeError(f'Artifact changed during recovery: {relative}')
    if local.exists():
     old=local.read_bytes();new=temporary.read_bytes()
     if relative.suffix=='.jsonl':assert new.startswith(old),'Existing raw or trajectory bytes changed'
     elif relative.suffix=='.json':assert json.loads(old)==json.loads(new),'Immutable committed manifest/status changed'
     else:assert old==new,'Immutable pilot audit changed'
    temporary.replace(local);cache[str(relative)]=fingerprint;copied+=1
   finally:temporary.unlink(missing_ok=True)
 save(cache_file,cache)
 tokenizer=recover_tokenizer(args,volume)
 return {'files_synced':copied,'unchanged_files_skipped':unchanged,'tokenizer_audit':tokenizer}


def recover_tokenizer(args,artifact_volume):
 """Copy only small immutable template metadata; never retrieve model shards."""
 import modal
 stem=f'phase1_dose_response_{args.attempt}'
 payloads=[]
 for arm in ARMS:
  path=ROOT/f'sandbox_inputs/{stem}_{arm}_cot_on.json'
  if path.exists():payloads.append(json.loads(path.read_text()))
 if not payloads:return {'state':'WAITING_FOR_TOKENIZER_HASHED_PAYLOAD'}
 expected=payloads[0]['tokenizer_file_sha256']
 assert all(payload['tokenizer_file_sha256']==expected for payload in payloads)
 assert 'tokenizer_config.json' in expected
 weights=modal.Volume.from_name('curiosity-study-weights');saved={}
 for name in sorted(expected):
  assert Path(name).name==name, 'Tokenizer metadata must be a basename'
  source=f'{NS}/checkpoints_full_residual_v2/intact/{name}'
  data=b''.join(weights.read_file(source));actual=hashlib.sha256(data).hexdigest()
  assert actual==expected[name],f'Intact tokenizer metadata changed: {name}'
  relative=f'{FAMILY}/tokenizer_audit/{name}';atomic_bytes(ROOT/relative,data,immutable=True)
  remote=f'{NS}/{relative}'
  try:existing=b''.join(artifact_volume.read_file(remote))
  except (modal.exception.NotFoundError,FileNotFoundError):existing=None
  if existing is None:
   with artifact_volume.batch_upload(force=False) as batch:batch.put_file(ROOT/relative,remote)
  else:assert existing==data,'Remote immutable tokenizer audit metadata differs'
  assert b''.join(artifact_volume.read_file(remote))==data
  saved[name]=actual
 return {'state':'TOKENIZER_METADATA_VERIFIED','files':saved}


def audit_final(args):
 from audit_dose_response_d3 import recompute
 result=recompute(ROOT,args.attempt,args.author_attempt)
 assert result['schema']=='phase1_dose_response_raw_audit_v1' and result['complete_data'] is True
 assert result['phase2_authorized'] is False and type(result['hygiene_passed']) is bool
 relative=f'records/phase1_dose_response_{args.attempt}_receipt.json'
 save(ROOT/relative,result,immutable=True)
 return {'receipt':relative,'result':result}


def upload_receipt(args):
 import modal
 volume=modal.Volume.from_name('curiosity-study-artifacts')
 relative=f'records/phase1_dose_response_{args.attempt}_receipt.json';data=(ROOT/relative).read_bytes()
 remote=f'{NS}/{relative}'
 try:existing=b''.join(volume.read_file(remote))
 except (modal.exception.NotFoundError,FileNotFoundError):existing=None
 if existing is not None:assert existing==data,'Remote immutable receipt differs'
 else:
  with volume.batch_upload(force=False) as batch:batch.put_file(ROOT/relative,remote)
 assert b''.join(volume.read_file(remote))==data
 return {'receipt_uploaded':True,'receipt_sha256':hashlib.sha256(data).hexdigest(),'modal_path':'/artifacts/'+remote}


def job_status(job):
 ledger=json.loads((ROOT/'infrastructure/resource_allocations.json').read_text())
 assert ledger['limits']['gpus']<=10,'Dose supervisor requires global account ceiling <=10 GPUs'
 assert ledger['limits']['single_gpu_workers']<=6 and ledger['limits']['tp8_workers']<=1
 return next((row for row in ledger['jobs'] if row['job']==job),None)


def remote_job_state(args,job):
 row=job_status(job)
 if row is None:return {'state':'NOT_RESERVED'}
 app_id=row.get('app_id')
 if not app_id:
  path=inside(row['log']);text=path.read_text() if path.exists() else ''
  ids=re.findall(r'https://modal.com/[^\s]+/(ap-[A-Za-z0-9]+)',text)
  if ids:app_id=ids[0]
 if not app_id:
  return {'state':'LAUNCH_PENDING_APP_ID' if row.get('launch_pid') and Path(f'/proc/{row["launch_pid"]}').exists() else 'UNRESOLVED_LAUNCH','job':job}
 apps=json.loads(subprocess.check_output([str(ROOT/'.venv/bin/modal'),'app','list','--json'],cwd=ROOT,text=True,stderr=subprocess.PIPE,timeout=60))
 match=next((app for app in apps if app['app_id']==app_id),None)
 assert match and match['description'].startswith(NS),'Missing or unrelated registered app; no assumed termination'
 return {'state':'STOPPED' if match['state']=='stopped' else 'ACTIVE','app_id':app_id,'job':job,'modal_state':match['state']}


def operation(args,name,lock_fd,deadline):
 remaining=deadline-time.monotonic()
 if remaining<=0:raise TimeoutError('Dose supervisor deadline reached')
 command=[sys.executable,str(Path(__file__).resolve()),'--attempt',args.attempt,'--author-attempt',args.author_attempt,
          '--primary-receipt',args.primary_receipt,'--calibration-receipt',args.calibration_receipt,
          '--operation',name,'--lock-fd',str(lock_fd)]
 if getattr(args,'descriptive_amendment',None):command+=['--descriptive-amendment',args.descriptive_amendment]
 timeout=900 if name in {'prepare-materialization','prepare-evaluation','audit'} else 180
 result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=min(timeout,remaining),pass_fds=(lock_fd,))
 if result.returncode:
  error=(result.stderr or result.stdout)[-3000:]
  if result.returncode==75:raise FileNotFoundError(error)
  if result.returncode==76:raise TransientTransportError(error)
  raise RuntimeError(f'{name}: {error}')
 return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def admit(args,stage,lock_fd,deadline):
 cache=operation(args,'verify-'+stage,lock_fd,deadline)
 assert cache['result'][stage+'_authorized'], 'Frozen audit did not authorize this stage'
 config,profile=resource_config(args,stage);freeze_sources(args,stage,config,profile)
 assert config['frozen_files']==cache['read_prerequisite_sha256'] and config['frozen_files']
 # Central admission serializes budget and all app resources. No direct modal
 # subprocess is started by this supervisor outside that authority.
 assert job_status(config['job']) is None,'Existing reservation must be recovered instead of relaunched'
 budget=json.loads((ROOT/'infrastructure/budget_reservations.json').read_text())
 assert budget['ceiling_usd']<=500
 return launch(config)


def supervise(args,lock_fd):
 state_path=ROOT/f'infrastructure/phase1_doses_{args.attempt}_supervisor_status.json'
 deadline=time.monotonic()+args.duration_seconds;last=None;network_failures=0
 while time.monotonic()<deadline:
  state={'checked_utc':now(),'attempt':args.attempt,'author_attempt':args.author_attempt,
         'phase2_authorized':False,'global_gpu_ceiling':10,'state':'RECOVERING'}
  delay=args.poll_seconds
  try:
   state['sync']=operation(args,'sync',lock_fd,deadline)
   materialized=ROOT/FAMILY/'materialization_set.json'
   mat_job=job_status(MATERIALIZATION_JOB)
   if not materialized.exists():
    if mat_job is None:
     operation(args,'prepare-materialization',lock_fd,deadline)
     try:state.update(state='MATERIALIZATION_LAUNCHED',launch=admit(args,'materialization',lock_fd,deadline))
     except BlockingIOError:state['state']='WAITING_FOR_CENTRAL_LAUNCH_LOCK'
     except ValueError as exc:
      if str(exc).startswith('Resource capacity exceeded:'):state.update(state='WAITING_FOR_MATERIALIZATION_CAPACITY',reason=str(exc))
      else:raise
    else:
     state['materialization_job']=operation(args,'materialization-job-state',lock_fd,deadline)
     if state['materialization_job']['state']=='STOPPED':
      # The app can finish between our first recovery and the app-state read.
      # Recover its final commit before concluding that artifacts are missing.
      state['final_sync']=operation(args,'sync',lock_fd,deadline)
     if state['materialization_job']['state'] in {'STOPPED','UNRESOLVED_LAUNCH'}:
      if not materialized.exists():
       state['state']='MATERIALIZATION_INCOMPLETE_FAIL_CLOSED';save(state_path,state);print(json.dumps(state),flush=True);return 2
     state['state']='WAITING_FOR_MATERIALIZATION'
   else:
    material=json.loads(materialized.read_text())
    assert material['schema']=='phase1_alpha_materialization_set_v1' and material['alpha_grid']==[-.5,0.,.5,1.,1.5]
    assert set(material['new_doses'])==set(NEW_DOSES)
    for dose in NEW_DOSES:assert json.loads((ROOT/FAMILY/f'{dose}_materialization.json').read_text())==material['new_doses'][dose]
    evaluation_job=f'phase1_dose_evaluation_{args.attempt}';existing=job_status(evaluation_job)
    if existing is None:
     if not inside(args.primary_receipt).exists() or not inside(args.calibration_receipt).exists():
      state['state']='WAITING_FOR_INDEPENDENT_PRIMARY_AND_INTACT_CALIBRATION_RECEIPTS'
     else:
      cache=operation(args,'prepare-evaluation',lock_fd,deadline)
      if not cache['result']['evaluation_authorized']:
       state.update(state='SCIENTIFIC_PREREQUISITE_GATE_FAILED',independent_gates=cache['result']['gates'])
       save(state_path,state);print(json.dumps(state),flush=True);return 2
      try:state.update(state='DOSE_EVALUATION_LAUNCHED',launch=admit(args,'evaluation',lock_fd,deadline))
      except BlockingIOError:state['state']='WAITING_FOR_CENTRAL_LAUNCH_LOCK'
      except ValueError as exc:
       if str(exc).startswith('Resource capacity exceeded:'):state.update(state='WAITING_FOR_DOSE_EVALUATION_CAPACITY',reason=str(exc))
       else:raise
    else:
     statuses={};stem=f'phase1_dose_response_{args.attempt}'
     for arm in ARMS:
      path=ROOT/f'evaluation_status/{stem}_{arm}_cot_on.json'
      if path.exists():
       row=json.loads(path.read_text());assert row['attempt']==args.attempt and row['arm']==arm
       statuses[arm]=row
     state['worker_statuses']={arm:row['status'] for arm,row in statuses.items()}
     state['failed_worker_arms']=[arm for arm,row in statuses.items() if row['status'] in {'H1_FAILED','H5_FAILED'}]
     terminal=len(statuses)==5 and all(row['status'] in {'COMPLETE','H1_FAILED','H5_FAILED'} for row in statuses.values())
     if terminal:
      if any(row['status']=='H5_FAILED' for row in statuses.values()):
       state.update(state='DOSE_H5_FAILED',reason='Full five-arm audit cannot be fabricated from absent full sweeps; all recovered pilot evidence retained.')
       save(state_path,state);print(json.dumps(state),flush=True);return 2
      audited=operation(args,'audit',lock_fd,deadline);state['independent_audit']=audited
      state.update(operation(args,'upload',lock_fd,deadline))
      passed=audited['result']['hygiene_passed'] and not state['failed_worker_arms']
      state['state']='DOSE_RESPONSE_AUDITED' if passed else 'DOSE_HYGIENE_FAILED_AUDITED'
      save(state_path,state);print(json.dumps(state),flush=True);return 0 if passed else 2
     state['evaluation_job']=operation(args,'evaluation-job-state',lock_fd,deadline)
     if state['evaluation_job']['state']=='STOPPED':state['final_sync']=operation(args,'sync',lock_fd,deadline)
     if state['evaluation_job']['state'] in {'STOPPED','UNRESOLVED_LAUNCH'}:
      final_statuses=[ROOT/f'evaluation_status/{stem}_{arm}_cot_on.json' for arm in ARMS]
      if not all(path.exists() for path in final_statuses):
       state['state']='DOSE_EVALUATION_INCOMPLETE_FAIL_CLOSED';save(state_path,state);print(json.dumps(state),flush=True);return 2
     state['state']='RECOVERING_DOSE_EVALUATION'
   network_failures=0
  except (TransientTransportError,subprocess.TimeoutExpired) as exc:
   network_failures+=1
   exhausted=network_failures>MAX_CONSECUTIVE_NETWORK_RETRIES
   delay=network_retry_delay(network_failures,args.poll_seconds) if not exhausted else 0
   state.update(state='TRANSPORT_RETRIES_EXHAUSTED_FAIL_CLOSED' if exhausted else 'TRANSIENT_TRANSPORT_RECOVERY',
                error=type(exc).__name__+': '+str(exc)[-3000:],consecutive_transport_failures=network_failures,
                retry_limit=MAX_CONSECUTIVE_NETWORK_RETRIES,retry_after_seconds=delay,
                action='Recover existing registered jobs only; preserve immutable authorization and all scientific gates.')
   if exhausted:
    save(state_path,state);print(json.dumps(state),flush=True);return 2
  except (FileNotFoundError,json.JSONDecodeError) as exc:
   state.update(state='TRANSIENT_RECOVERY_OR_INPUT_ERROR',error=type(exc).__name__+': '+str(exc)[-3000:],
                action='Retry existing evidence recovery only; no duplicate paid job.')
  except Exception as exc:
   state.update(state='SUPERVISOR_FAILED_CLOSED',error=type(exc).__name__+': '+str(exc)[-3000:],
                action='Inspect immutable evidence; no downstream phase authorization.')
   save(state_path,state);print(json.dumps(state),flush=True);return 2
  save(state_path,state);print(json.dumps(state),flush=True);last=state
  remaining=deadline-time.monotonic()
  if remaining>0:time.sleep(min(delay,remaining))
 state={**(last or {}),'checked_utc':now(),'attempt':args.attempt,'state':'SUPERVISOR_DEADLINE_REACHED',
        'completion_implied':False,'phase2_authorized':False}
 save(state_path,state);print(json.dumps(state),flush=True);return 3


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True,type=simple_id)
 parser.add_argument('--author-attempt',required=True,type=simple_id)
 parser.add_argument('--primary-receipt',required=True);parser.add_argument('--calibration-receipt',required=True)
 parser.add_argument('--descriptive-amendment',help='Explicit source-bound administrative amendment; default retains qualified-only eligibility')
 parser.add_argument('--duration-seconds',type=int,default=14400)
 parser.add_argument('--poll-seconds',type=int,default=45)
 parser.add_argument('--operation',choices=['sync','prepare-materialization','prepare-evaluation','verify-materialization',
    'verify-evaluation','materialization-job-state','evaluation-job-state','audit','upload'],help=argparse.SUPPRESS)
 parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS);args=parser.parse_args()
 if not 0<args.duration_seconds<=14400:parser.error('Bounded duration must be between 1 and 14400 seconds')
 if not 45<=args.poll_seconds<=300:parser.error('Polling interval must be between 45 and 300 seconds')
 for name in ('primary_receipt','calibration_receipt'):
  setattr(args,name,str(inside(getattr(args,name)).relative_to(ROOT)))
 if args.descriptive_amendment:
  from prepare_dose_evaluation_d3 import AMENDMENT_PATH,DESCRIPTIVE_ATTEMPT
  assert args.descriptive_amendment==AMENDMENT_PATH and args.attempt==DESCRIPTIVE_ATTEMPT
 from prepare_dose_evaluation_d3 import DESCRIPTIVE_ATTEMPT,AMENDMENT_PATH
 assert args.attempt==DESCRIPTIVE_ATTEMPT and args.author_attempt=='v1'
 assert args.descriptive_amendment==AMENDMENT_PATH
 assert (ROOT/FAMILY/'materialization_set.json').is_file(), 'D3 may only reuse already materialized checkpoints'
 lock_path=ROOT/f'infrastructure/phase1_doses_{args.attempt}_supervisor.lock'
 global_lock_path=ROOT/'infrastructure/phase1_doses_supervisor.lock'
 if args.operation:
  if args.lock_fd is None:parser.error('Child operation requires the inherited supervisor lock')
  # Children inherit the shared family lock itself. If the parent exits while
  # recovery is still running, another supervisor remains excluded.
  actual=os.fstat(args.lock_fd);expected=global_lock_path.stat()
  assert (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)
  fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  operations={'sync':lambda:sync_once(args),'prepare-materialization':lambda:prepare_materialization(args),
    'prepare-evaluation':lambda:prepare_evaluation(args),'verify-materialization':lambda:verify_cache(args,'materialization'),
    'verify-evaluation':lambda:verify_cache(args,'evaluation'),'materialization-job-state':lambda:remote_job_state(args,MATERIALIZATION_JOB),
    'evaluation-job-state':lambda:remote_job_state(args,f'phase1_dose_evaluation_{args.attempt}'),
    'audit':lambda:audit_final(args),'upload':lambda:upload_receipt(args)}
  try:result=operations[args.operation]()
  except (FileNotFoundError,json.JSONDecodeError) as exc:
   print(json.dumps({'transient_input_error':type(exc).__name__+': '+str(exc)}),flush=True);return 75
  except Exception as exc:
   if args.operation not in NETWORK_OPERATIONS or not transient_transport(exc):raise
   print(json.dumps({'transient_transport_error':type(exc).__name__+': '+str(exc)}),flush=True);return 76
  print(json.dumps(result),flush=True);return 0
 with global_lock_path.open('a') as global_lock,lock_path.open('a') as lock:
  # All attempts share the fixed materialization family and tokenizer metadata;
  # therefore only one supervisor may write those recovery paths at a time.
  fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Re-read the current specification on every supervisor startup; its bytes are
  # subsequently bound into both immutable stage authorizations.
  (ROOT/'FINAL_RESEARCH_SPECIFICATION.md').read_text()
  return supervise(args,global_lock.fileno())

if __name__=='__main__':sys.exit(main())
