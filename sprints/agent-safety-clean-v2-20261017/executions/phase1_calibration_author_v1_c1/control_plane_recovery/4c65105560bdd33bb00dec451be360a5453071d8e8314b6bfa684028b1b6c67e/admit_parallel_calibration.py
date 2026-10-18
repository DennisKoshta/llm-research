"""Admit only fixed Intact calibration, with bounded, immutable prerequisite audits.

No Phase 2 launch or treatment-outcome tuning is authorized. Every scientific
prerequisite read is hash-bound before caching; subsequent admission polls check
those bytes instead of repeating the expensive independent numerical audit.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,tempfile,time,types
ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=[
 'FINAL_RESEARCH_SPECIFICATION.md','scripts/admit_parallel_calibration.py',
 'scripts/launch_bounded_job.py','scripts/modal_sandbox_evaluation.py',
 'scripts/sandbox_evaluation_worker.py','scripts/sandbox_runtime.py','scripts/primary_model_loader.py',
 'scripts/primary_fp8_inventory.py','scripts/strict_parser.py','scripts/prepare_calibration.py',
 'scripts/audit_sandbox_episodes.py','scripts/audit_authored_scenarios.py','scripts/scenario_contract.py',
 'scripts/audit_source_directions.py','scripts/prepare_directions.py','scripts/check_phase1_activation_gate.py']
CONTROL_PLANE_FILES={'scripts/admit_parallel_calibration.py','scripts/launch_bounded_job.py'}


class TransientAuditError(RuntimeError):pass


def now():return datetime.now(timezone.utc).isoformat()


def identifier(value):
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',value):raise argparse.ArgumentTypeError('Simple attempt identifier required')
 return value


def digest(path):
 value=hashlib.sha256()
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):value.update(block)
 return value.hexdigest()


def atomic_bytes(path,content,immutable=False):
 path.parent.mkdir(parents=True,exist_ok=True)
 if immutable and path.exists():
  if path.read_bytes()!=content:raise ValueError(f'Frozen bytes changed: {path.relative_to(ROOT)}')
  return
 descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.writing',dir=path.parent)
 temporary=Path(name)
 try:
  with os.fdopen(descriptor,'wb') as handle:
   handle.write(content);handle.flush();os.fsync(handle.fileno())
  if immutable:
   try:os.link(temporary,path)
   except FileExistsError:
    if path.read_bytes()!=content:raise ValueError(f'Concurrent frozen bytes differ: {path.relative_to(ROOT)}')
  else:temporary.replace(path)
 finally:temporary.unlink(missing_ok=True)


def save(path,value,immutable=False):
 atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),immutable)


def cache_path(attempt):return ROOT/f'executions/phase1_calibration_{attempt}/authorization_cache.json'


def fresh_launch_authority():
 """Compile current central authority bytes; never retain old imported limits."""
 path=ROOT/'scripts/launch_bounded_job.py'
 module=types.ModuleType('_calibration_current_launch_authority');module.__file__=str(path)
 exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 return module


class ReadBindings:
 """Bind first-read bytes and detect mutation across the complete audit."""
 def __init__(self):self.files={};self.busy=False
 def record(self,path):
  if self.busy:return
  self.busy=True
  try:
   path=Path(path).resolve()
   if not path.is_relative_to(ROOT) or not path.is_file():return
   relative=str(path.relative_to(ROOT))
   # Package installations are environment inputs, not sprint evidence. Their
   # versions are recorded separately; no model/scientific files are excluded.
   if relative.startswith('.venv/'):return
   if '__pycache__' in path.parts and path.suffix=='.pyc':
    import importlib.util
    source=Path(importlib.util.source_from_cache(str(path)))
    if source.exists():
     source_relative=str(source.relative_to(ROOT))
     if source_relative not in self.files:self.files[source_relative]=digest(source)
    return
   if relative not in self.files:self.files[relative]=digest(path)
  finally:self.busy=False
 def hook(self,event,args):
  if event!='open' or self.busy:return
  path,mode,flags=args
  if not isinstance(path,(str,bytes,os.PathLike)):return
  if flags & (os.O_WRONLY|os.O_TRUNC|os.O_CREAT):return
  self.record(os.fsdecode(path))


def verify_bindings(cache,recover_control_plane=False):
 if cache['schema']!='immutable_calibration_authorization_v1':raise ValueError('Unknown authorization cache schema')
 for relative,expected in cache['read_prerequisite_sha256'].items():
  path=(ROOT/relative).resolve()
  if not path.is_relative_to(ROOT):raise ValueError(f'Frozen prerequisite escapes sprint: {relative}')
  if recover_control_plane and relative in CONTROL_PLANE_FILES:
   # Preserve and verify the original authorization's exact old control source.
   # The replacement admission code does not replace scientific inputs or audits.
   original=cache_path(cache['attempt']).parent/Path(relative).name
   if digest(original)!=expected:raise ValueError(f'Original control-plane snapshot changed: {relative}')
  elif digest(path)!=expected:
   raise ValueError(f'Frozen prerequisite changed: {relative}')
 return cache


def authorize_once(attempt,author_attempt):
 path=cache_path(attempt)
 if path.exists():return verify_cached(attempt,author_attempt)
 bindings=ReadBindings();sys.addaudithook(bindings.hook)
 for relative in SOURCE_FILES:bindings.record(ROOT/relative)
 config=ROOT/f'stimuli/calibration_{author_attempt}.json'
 payload=json.loads(config.read_text())
 if payload['author_attempt']!=author_attempt:raise ValueError('Author attempt mismatch in frozen config')
 from modal_sandbox_evaluation import authorize_calibration
 authorization=authorize_calibration(ROOT,payload)
 # Copy bindings before checking them: later cache/result output is not itself
 # an input to the completed authorization.
 files=dict(sorted(bindings.files.items()))
 cache={'schema':'immutable_calibration_authorization_v1','attempt':attempt,'author_attempt':author_attempt,
        'authorization':authorization,'read_prerequisite_sha256':files,
        'environment':{'python':sys.version,'numpy':__import__('numpy').__version__,
                       'sklearn':__import__('sklearn').__version__},
        'binding_scope':'Every file read inside this sprint during authorization, plus frozen runner/audit sources. Installed .venv package files excluded; versions recorded.'}
 verify_bindings(cache)
 save(path,cache,immutable=True)
 return {'cache':str(path.relative_to(ROOT)),'bound_files':len(files),'authorization':authorization}


def verify_cached(attempt,author_attempt,recover_control_plane=False):
 cache=json.loads(cache_path(attempt).read_text())
 if cache['attempt']!=attempt or cache['author_attempt']!=author_attempt:raise ValueError('Cached authorization identity mismatch')
 environment={'python':sys.version,'numpy':__import__('numpy').__version__,'sklearn':__import__('sklearn').__version__}
 if cache['environment']!=environment:raise ValueError('Authorization environment versions changed')
 verify_bindings(cache,recover_control_plane)
 return {'cache':str(cache_path(attempt).relative_to(ROOT)),'bound_files':len(cache['read_prerequisite_sha256']),
         'authorization':cache['authorization']}


def audit_operation(args,name,lock_fd,deadline):
 remaining=deadline-time.monotonic()
 if remaining<=0:raise TimeoutError('Admission deadline reached')
 command=[sys.executable,str(Path(__file__).resolve()),'--attempt',args.attempt,
       '--author-attempt',args.author_attempt,'--operation',name,'--lock-fd',str(lock_fd)]
 if getattr(args,'recover_control_plane',False):command.append('--recover-control-plane')
 result=subprocess.run(command,cwd=ROOT,
       capture_output=True,text=True,timeout=min(600 if name=='authorize' else 120,remaining),pass_fds=(lock_fd,))
 if result.returncode==76:raise BlockingIOError('Current launch authority lock is held')
 if result.returncode==77:
  detail=json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
  raise ValueError(detail['capacity_error'])
 if result.returncode==75:raise TransientAuditError(f'Bounded {name} transient input failure: '+(result.stderr or result.stdout)[-2000:])
 if result.returncode:raise RuntimeError(f'Bounded {name} failed: '+(result.stderr or result.stdout)[-2000:])
 return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def watcher_alive(pid,attempt):
 try:command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,ProcessLookupError):return False
 return any(b'watch_calibration.py' in part for part in command) and attempt.encode() in command


def start_observer(attempt,author_attempt):
 lock_path=ROOT/f'infrastructure/calibration_{attempt}_observer.lock'
 record_path=ROOT/f'infrastructure/calibration_{attempt}_observer_process.json'
 with lock_path.open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return {'state':'OBSERVER_ALREADY_HOLDS_LOCK','record':str(record_path.relative_to(ROOT))}
  record=json.loads(record_path.read_text()) if record_path.exists() else {'launches':[]}
  for previous in record['launches']:
   if watcher_alive(previous['pid'],attempt):return {'state':'OBSERVER_ALREADY_RUNNING',**previous}
  # The observer acquires its own lock. Release our check-lock before spawning;
  # another concurrent observer wins the same lock and only one can write.
  fcntl.flock(lock,fcntl.LOCK_UN)
  stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  relative=f'infrastructure/calibration_{attempt}_watch_{stamp}.log'
  command=[sys.executable,str(ROOT/'scripts/watch_calibration.py'),'--attempt',attempt,'--author-attempt',author_attempt]
  with (ROOT/relative).open('x') as log:
   process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  entry={'pid':process.pid,'created_utc':now(),'log':relative,'command':command}
  record['launches'].append(entry);save(record_path,record)
  return {'state':'OBSERVER_STARTED',**entry}


def build_config(args):
 source='scripts/modal_sandbox_evaluation.py';raw=(ROOT/source).read_bytes();functions=fresh_launch_authority()._decorators(raw.decode())
 return {'job':f'phase1_intact_calibration_{args.attempt}','gpus':1,'single_gpu_workers':1,'tp8_workers':0,'upper_bound_usd':12,
   'log':f'infrastructure/calibration_{args.attempt}.log',
   'command':[str(ROOT/'.venv/bin/modal'),'run','--detach',source,'--config',f'stimuli/calibration_{args.author_attempt}.json','--attempt',args.attempt],
   'resource_profile':{'source':source,'source_sha256':hashlib.sha256(raw).hexdigest(),
    'functions':{name:{'max_calls':1,'limits':functions[name]} for name in ('evaluate','coordinate')},
    'excluded_functions':{},'call_bound_basis':'One fixed Intact payload; one durable coordinate call and one evaluate FunctionCall; no retries or other model arms.'}}


def freeze_snapshot(args,authorization,cfg):
 snapshot=ROOT/f'executions/phase1_calibration_{args.attempt}';snapshot.mkdir(parents=True,exist_ok=True)
 if getattr(args,'recover_control_plane',False):
  cache=json.loads(cache_path(args.attempt).read_text());verify_cached(args.attempt,args.author_attempt,True)
  original=json.loads((snapshot/'source_manifest.json').read_text())
  for relative,expected in original.items():
   if digest(snapshot/Path(relative).name)!=expected:raise ValueError(f'Original snapshot changed: {relative}')
   if relative not in CONTROL_PLANE_FILES and digest(ROOT/relative)!=expected:
    raise ValueError(f'Frozen scientific execution source changed: {relative}')
  if json.loads((snapshot/'authorization.json').read_text())!=authorization:
   raise ValueError('Recovery cannot replace the original scientific authorization')
  if json.loads((snapshot/'resource_config.json').read_text())!=cfg:
   raise ValueError('Recovery cannot change the frozen execution payload or resource profile')
  current={relative:digest(ROOT/relative) for relative in sorted(CONTROL_PLANE_FILES)}
  record={'schema':'calibration_control_plane_recovery_v1','attempt':args.attempt,'author_attempt':args.author_attempt,
    'original_authorization_cache_sha256':digest(cache_path(args.attempt)),
    'original_source_manifest_sha256':digest(snapshot/'source_manifest.json'),
    'original_resource_config_sha256':digest(snapshot/'resource_config.json'),
    'original_control_plane_sha256':{relative:cache['read_prerequisite_sha256'][relative] for relative in sorted(CONTROL_PLANE_FILES)},
    'current_control_plane_sha256':current,'scientific_execution_changed':False,
    'same_job_id':cfg['job'],'phase2_authorized':False}
  key=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
  destination=snapshot/'control_plane_recovery'/key
  save(destination/'recovery.json',record,immutable=True)
  for relative in current:atomic_bytes(destination/Path(relative).name,(ROOT/relative).read_bytes(),immutable=True)
  return
 files=SOURCE_FILES+[f'stimuli/calibration_{args.author_attempt}.json']
 hashes={}
 for relative in files:
  content=(ROOT/relative).read_bytes();hashes[relative]=hashlib.sha256(content).hexdigest()
  atomic_bytes(snapshot/Path(relative).name,content,immutable=True)
 save(snapshot/'source_manifest.json',hashes,immutable=True)
 save(snapshot/'authorization.json',authorization,immutable=True)
 save(snapshot/'resource_config.json',cfg,immutable=True)


def launch_current(args):
 """Child-process admission, same immutable job, with hashes checked under central lock."""
 recovered=getattr(args,'recover_control_plane',False)
 verify_cached(args.attempt,args.author_attempt,recovered)
 snapshot=cache_path(args.attempt).parent
 cfg=json.loads((snapshot/'resource_config.json').read_text())
 if cfg!=build_config(args):raise ValueError('Frozen execution configuration changed before admission')
 cache=json.loads(cache_path(args.attempt).read_text())
 files=dict(cache['read_prerequisite_sha256'])
 if recovered:
  files.update({relative:digest(ROOT/relative) for relative in CONTROL_PLANE_FILES})
 # This changes only the admission envelope; the Modal command, model inputs,
 # job ID, one-call resource bounds, and scientific authorization stay exact.
 cfg={**cfg,'frozen_files':files}
 authority=fresh_launch_authority()
 return authority.launch(cfg)


def supervise(args,lock_fd):
 statefile=ROOT/f'infrastructure/calibration_admission_{args.attempt}.json'
 configpath=ROOT/f'stimuli/calibration_{args.author_attempt}.json'
 deadline=time.monotonic()+10800;job=f'phase1_intact_calibration_{args.attempt}'
 while time.monotonic()<deadline:
  state={'checked_utc':now(),'author_attempt':args.author_attempt,'attempt':args.attempt,'phase2_authorized':False}
  try:
   registry=json.loads((ROOT/'infrastructure/resource_allocations.json').read_text())
   existing=next((row for row in registry['jobs'] if row['job']==job),None)
   if existing:
    # A watcher only recovers evidence; starting it is safe even when the old
    # detached launch has an unresolved app ID. Never duplicate the GPU call.
    state.update(state='ALREADY_RESERVED_OR_LAUNCHED_INSPECT_LEDGER',existing=existing,
                 observer=start_observer(args.attempt,args.author_attempt))
    save(statefile,state);print(json.dumps(state),flush=True);return 0
   if not configpath.exists():
    author_path=ROOT/f'qualification_status/author_{args.author_attempt}_flash_next_uncensored.json'
    if author_path.exists():
     worker=json.loads(author_path.read_text())
     if worker.get('status') in {'H1_FAILED','H5_FAILED','AUTHOR_CONTRACT_FAILED','AUTHORING_GATE_FAILED'}:
      state.update(state='AUTHOR_GATE_FAILED',worker_status=worker)
      save(statefile,state);print(json.dumps(state),flush=True);return 2
    state['state']='WAITING_FOR_AUDITED_AUTHORING'
   else:
    state['state']='VERIFYING_FROZEN_PREREQUISITES' if cache_path(args.attempt).exists() else 'AUTHORIZING_CALIBRATION'
    save(statefile,state)
    evidence=audit_operation(args,'verify' if cache_path(args.attempt).exists() else 'authorize',lock_fd,deadline)
    cfg=build_config(args);fresh_launch_authority().validate_resource_profile(cfg)
    freeze_snapshot(args,evidence['authorization'],cfg)
    # Recheck all audited inputs immediately before launch, including raw arrays
    # and authored data; a cached successful audit never authorizes changed bytes.
    audit_operation(args,'verify',lock_fd,deadline)
    try:result=audit_operation(args,'launch',lock_fd,deadline)
    except BlockingIOError:state['state']='WAITING_FOR_LAUNCH_LOCK'
    except ValueError as exc:
     if str(exc).startswith('Resource capacity exceeded:'):state.update(state='WAITING_FOR_GPU_CAPACITY',reason=str(exc))
     else:raise
    else:
     state.update(state='CALIBRATION_LAUNCHED',launch=result)
     save(statefile,state)
     try:state['observer']=start_observer(args.attempt,args.author_attempt)
     except Exception as exc:
      state.update(state='CALIBRATION_LAUNCHED_OBSERVER_START_FAILED',observer_error=type(exc).__name__+': '+str(exc))
     save(statefile,state);print(json.dumps(state),flush=True)
     with (ROOT/'PROGRESS_AND_GATES.md').open('a') as handle:
      handle.write(f'\n- {state["checked_utc"]}: Automatically admitted fixed Intact-only calibration {args.attempt} after immutable independent prerequisite audit and capacity release. Five fresh pilots required; no Phase 2 authorization. Snapshot: executions/phase1_calibration_{args.attempt}/.\n')
     return 0 if state['state']=='CALIBRATION_LAUNCHED' else 2
  except (FileNotFoundError,json.JSONDecodeError,subprocess.TimeoutExpired,TransientAuditError) as exc:
   state.update(state='TRANSIENT_PREREQUISITE_OR_AUDIT_ERROR',error=type(exc).__name__+': '+str(exc)[-2000:],
                action='Retry same immutable inputs; no GPU launch authorized by elapsed time.')
  except Exception as exc:
   state.update(state='ADMISSION_FAILED_CLOSED',error=type(exc).__name__+': '+str(exc)[-2000:],
                action='No gate waiver or duplicate GPU launch; inspect recorded error.')
   save(statefile,state);print(json.dumps(state),flush=True);return 2
  save(statefile,state);print(json.dumps(state),flush=True)
  remaining=deadline-time.monotonic()
  if remaining>0:time.sleep(min(45,remaining))
 state={'checked_utc':now(),'attempt':args.attempt,'author_attempt':args.author_attempt,
        'state':'ADMISSION_DEADLINE_REACHED','phase2_authorized':False,'completion_implied':False}
 save(statefile,state);print(json.dumps(state),flush=True);return 3


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--author-attempt',required=True,type=identifier)
 parser.add_argument('--attempt',required=True,type=identifier)
 parser.add_argument('--recover-control-plane',action='store_true',
   help='Reuse the exact original scientific cache/payload while recording updated controller/central-launcher sources separately')
 parser.add_argument('--operation',choices=['authorize','verify','launch'],help=argparse.SUPPRESS)
 parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS);args=parser.parse_args()
 lock_path=ROOT/f'infrastructure/calibration_admission_{args.attempt}.lock'
 if args.operation:
  if args.lock_fd is None:parser.error('Bounded audit requires inherited admission lock')
  actual=os.fstat(args.lock_fd);expected=lock_path.stat()
  if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('Wrong admission lock descriptor')
  fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   if args.operation=='authorize':
    if args.recover_control_plane:raise ValueError('Recovery requires an existing immutable authorization cache')
    result=authorize_once(args.attempt,args.author_attempt)
   elif args.operation=='verify':result=verify_cached(args.attempt,args.author_attempt,args.recover_control_plane)
   else:result=launch_current(args)
  except BlockingIOError:return 76
  except ValueError as exc:
   if str(exc).startswith('Resource capacity exceeded:'):
    print(json.dumps({'capacity_error':str(exc)}),flush=True);return 77
   raise
  except (FileNotFoundError,json.JSONDecodeError) as exc:
   print(json.dumps({'transient_error':type(exc).__name__+': '+str(exc)}),flush=True);return 75
  print(json.dumps(result),flush=True);return 0
 with lock_path.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return supervise(args,lock.fileno())

if __name__=='__main__':sys.exit(main())
