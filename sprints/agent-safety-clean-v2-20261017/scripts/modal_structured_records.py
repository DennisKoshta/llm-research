"""Prepared bounded record runner. Only the root resource coordinator may launch.

Each submitted payload contains one role/task/cot batch. Admission independently
recomputes all primary Phase1, author, calibration and dose prerequisites. Up to
two single-GPU workers may run; there is no TP8 or automatic retry pathway.
"""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
app = modal.App(NS+'-structured-records')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
RUNTIME_FILES = ('structured_records_worker.py', 'llama_model_loader.py', 'llama_serving_worker.py',
                 'primary_model_loader.py', 'primary_fp8_inventory.py', 'exp23_author_worker.py',
                 'exp23_design.py', 'scenario_contract.py', 'strict_parser.py')
image = (modal.Image.from_registry('vllm/vllm-openai:v0.31.0', add_python='3.12').entrypoint([])
         .env({'HF_HOME': f'/weights/{NS}/huggingface', 'HF_HUB_OFFLINE': '1',
               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
               'TRITON_CACHE_DIR': f'/weights/{NS}/structured_records_triton_cache'}))
coordinator_image = modal.Image.debian_slim(python_version='3.12')
for filename in RUNTIME_FILES:
    image = image.add_local_file(ROOT/'scripts'/filename, '/opt/cleanroom/'+filename)
    coordinator_image = coordinator_image.add_local_file(ROOT/'scripts'/filename, '/opt/cleanroom/'+filename)


@app.function(image=image, gpu='H200', cpu=(4, 4), memory=(65536, 65536), ephemeral_disk=524288,
              timeout=3600, startup_timeout=300, retries=0, max_containers=2, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts})
def records(payload):
    import os
    import subprocess
    import sys
    import time
    sys.path.insert(0, '/opt/cleanroom')
    from structured_records_worker import validate_payload, write_exclusive, durable_call
    validate_payload(payload); artifacts.reload(); weights.reload();root=Path('/artifacts')/NS
    # A rapidly scheduled worker can arrive before spawn()'s caller commits the
    # returned call ID. No model initialization or generation may cross that gap.
    for check in range(6):
        try:
            bound=durable_call(root,payload); break
        except (FileNotFoundError, ValueError):
            if check == 5: raise
            time.sleep(2); artifacts.reload()
    if bound['call_id']!=modal.current_function_call_id():raise ValueError('Actual structured worker call differs from durable admission')
    write_exclusive(root/'structured_worker_starts'/f'{payload["experiment"]}.json',
        {'payload_sha256':payload['payload_sha256'],'call_id':bound['call_id']});artifacts.commit()
    path = root/'structured_inputs'/f'{payload["experiment"]}.json'
    if path.exists():
        if json.loads(path.read_text()) != payload: raise ValueError('Attempt input already differs')
    else: write_exclusive(path, payload)
    artifacts.commit()
    environment = dict(os.environ, TRITON_CACHE_DIR='/tmp/structured-cache/'+payload['experiment'])
    child = subprocess.Popen(['/usr/bin/python3', '/opt/cleanroom/structured_records_worker.py', str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment)
    try:
        for line in child.stdout:
            if line.strip() == 'COMMIT_ARTIFACTS':
                artifacts.commit(); child.stdin.write('COMMITTED\n'); child.stdin.flush()
            else: print(line, end='', flush=True)
        code = child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        artifacts.commit()
    if code:
        failure=root/'structured_worker_failures'/f'{payload["experiment"]}.json'
        if not failure.exists():write_exclusive(failure,{'payload_sha256':payload['payload_sha256'],'call_id':bound['call_id'],
            'exit_code':code,'automatic_retry':False});artifacts.commit()
        raise RuntimeError(f'Structured worker failed ({code}); preserve this attempt and reconcile evidence')
    result = json.loads((Path('/artifacts')/NS/'qualification_status'/f'{payload["experiment"]}.json').read_text())
    if result['payload_sha256'] != payload['payload_sha256']: raise ValueError('Returned result differs from frozen payload')
    return result


@app.function(image=coordinator_image, cpu=(1, 1), memory=(1024, 1024), timeout=7800, startup_timeout=300,
              retries=0, max_containers=2, scaledown_window=2, volumes={'/artifacts': artifacts})
def coordinate(payload):
    import sys
    sys.path.insert(0, '/opt/cleanroom')
    from structured_records_worker import validate_payload, write_exclusive, utc_now, hash_bytes, durable_call
    validate_payload(payload); artifacts.reload()
    for name,h in payload['input_sources'].items():
        if hash_bytes(Path('/artifacts')/NS/name)!=h:raise ValueError('Structured prerequisite changed before submission')
    path = Path('/artifacts')/NS/'group_calls'/f'{payload["experiment"]}.json'
    if path.exists():
        ledger = json.loads(path.read_text())
        if ledger['payload_sha256'] != payload['payload_sha256']: raise ValueError('Immutable attempt rebound')
        if ledger.get('state') != 'SPAWNED' or not ledger.get('call_id'):
            raise RuntimeError('Unresolved spawn intent; reconcile read-only before any further paid submission')
        durable_call(Path('/artifacts')/NS,payload)
        call = modal.FunctionCall.from_id(ledger['call_id'])
    else:
        ledger = {'schema': 'structured_records_call_ledger_v1', 'state': 'SPAWN_INTENT',
            'payload_sha256': payload['payload_sha256'], 'experiment': payload['experiment'],
            'role': payload['model_role'], 'task': payload['task'], 'created_at_utc': utc_now(),
            'job':payload['central_job'],'worker_module':'modal_structured_records','worker_function':'records',
            'max_resources': {'gpu': 'H200:1', 'cpu': 4, 'memory_mib': 65536, 'timeout_seconds': 3600, 'startup_seconds': 300}}
        write_exclusive(path, ledger); artifacts.commit()
        call = records.spawn(payload)
        ledger.update(state='SPAWNED', call_id=call.object_id, submitted_at_utc=utc_now())
        temporary = path.with_suffix('.submitted.json')
        write_exclusive(temporary, ledger); temporary.replace(path); artifacts.commit()
    return call.get()


def prepare(root, attempt, requests_path, components_path, *, tokenizer_metadata):
    """Local preparation only; returns a fully audited immutable launch payload."""
    from phase1_readiness import recompute
    import sys
    from admit_parallel_calibration import ReadBindings
    from exp11_admission import local_source_closure
    from structured_records_worker import (RUNTIME_FILES as WORKER_FILES, PRIMARY_ROLES, LLAMA_REPO,
        build_payload, hash_bytes, identity,TOKENIZER_FILES)
    from strict_parser import unique_object, reject_constant
    root = Path(root).resolve();bindings=ReadBindings();sys.addaudithook(bindings.hook)
    def inside(path):
        result = (root/path).resolve(); result.relative_to(root); return result
    request_path, component_path = inside(requests_path), inside(components_path)
    requests = json.loads(request_path.read_text(), object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(requests, list): raise ValueError('Input must be the exact immutable request list, without private QA joins')
    components = json.loads(component_path.read_text(), object_pairs_hook=unique_object, parse_constant=reject_constant)
    readiness = recompute(root, components)
    if readiness['primary_phase2_authorized'] is not True:
        raise ValueError('Primary Phase1, independent Intact calibration and dose prerequisites have not passed')
    sources = {str(path.relative_to(root)): hash_bytes(path) for path in (request_path, component_path)}
    for name, component in readiness['components'].items():
        # External validity is reported by readiness, never required for primary
        # record jobs. Only required component receipts need be synced remotely.
        if name != 'external': sources[component['receipt_path']] = component['receipt_sha256']
    sources.update(readiness['provenance']['artifact_sha256'])
    sources['FINAL_RESEARCH_SPECIFICATION.md'] = hash_bytes(root/'FINAL_RESEARCH_SPECIFICATION.md')
    sources['scripts/phase1_readiness.py'] = hash_bytes(root/'scripts/phase1_readiness.py')
    if not requests:raise ValueError('No structured records requested')
    role = requests[0]['model_role']; checkpoint_hash = None; provenance = {}
    if role == 'independent_qa_judge':
        sources['scripts/qa_semantic_protocol.py'] = hash_bytes(root/'scripts/qa_semantic_protocol.py')
    if role in PRIMARY_ROLES:
        path = root/'checkpoints_full_residual_v2'/f'{PRIMARY_ROLES[role]}_materialization.json'
        checkpoint_hash = hash_bytes(path); sources[str(path.relative_to(root))] = checkpoint_hash
        manifest = json.loads(path.read_text())
        expected = identity(role, checkpoint_hash)
        if manifest['path'] != expected['model_path'] or manifest['arm'] != expected['model_arm']:
            raise ValueError('Reviewer role checkpoint manifest mismatch')
    else:
        provenance = {'download_manifest': 'infrastructure/'+LLAMA_REPO.replace('/', '__')+'_download.json',
                      'preflight': 'infrastructure/llama_preflight.json'}
        for filename in provenance.values(): sources[filename] = hash_bytes(root/filename)
    folder=inside(tokenizer_metadata);tokenizers={};tokenizer_paths={}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name in TOKENIZER_FILES:
            tokenizers[path.name]=hash_bytes(path);tokenizer_paths[path.name]=str(path.relative_to(root))
            sources[tokenizer_paths[path.name]]=tokenizers[path.name]
    for name,h in local_source_closure(root,('phase1_readiness','structured_records_evidence')).items():sources[name]=h
    for name in (*RUNTIME_FILES,'modal_structured_records.py','structured_records_evidence.py'):
        sources['scripts/'+name]=hash_bytes(root/'scripts'/name)
    mutable={'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}
    sources.update({k:v for k,v in bindings.files.items() if k not in mutable})
    for name,h in sources.items():
        if hash_bytes(root/name)!=h:raise ValueError('Structured prerequisite changed during preparation')
    if set(RUNTIME_FILES) != set(WORKER_FILES): raise ValueError('Image/worker source packaging differs')
    return build_payload(attempt, requests, readiness=readiness, input_sources=sources,
        source_hashes={name: hash_bytes(root/'scripts'/name) for name in RUNTIME_FILES},
        specification_sha256=hash_bytes(root/'FINAL_RESEARCH_SPECIFICATION.md'),
        request_source_path=str(request_path.relative_to(root)),
        tokenizer_files=tokenizers,tokenizer_source_paths=tokenizer_paths,component_sources=components,
        checkpoint_sha256=checkpoint_hash, model_provenance=provenance)


def authorize(root,payload):
    from phase1_readiness import recompute
    from structured_records_worker import validate_payload,hash_bytes
    root=Path(root);validate_payload(payload)
    if recompute(root,payload['component_sources'])!=payload['readiness']:
        raise ValueError('Structured Phase1 prerequisites no longer independently recompute')
    for name,h in payload['input_sources'].items():
        if hash_bytes(root/name)!=h:raise ValueError('Frozen structured input source changed')
    for name,h in payload['source_hashes'].items():
        if hash_bytes(root/'scripts'/name)!=h:raise ValueError('Packaged structured runtime changed')
    if json.loads((root/payload['request_source_path']).read_text())!=payload['requests']:
        raise ValueError('Immutable structured request source differs')
    return True


def resource_config(root,payload,input_path):
    from launch_bounded_job import _decorators,validate_resource_profile
    from structured_records_worker import validate_payload,hash_bytes
    root=Path(root);validate_payload(payload);source='scripts/modal_structured_records.py'
    expected=f'stimuli/{payload["experiment"]}_input.json'
    if input_path!=expected or json.loads((root/input_path).read_text())!=payload:raise ValueError('Canonical structured payload input differs')
    limits=_decorators((root/source).read_text())
    config={'job':payload['central_job'],'gpus':2,'single_gpu_workers':2,'tp8_workers':0,'upper_bound_usd':20,
        'log':f'infrastructure/{payload["central_job"]}.log','command':[str(root/'.venv/bin/modal'),'run','--detach',source,
            '--attempt',payload['attempt'],'--input-path',input_path,'--launch'],
        'frozen_files':{**payload['input_sources'],**{'scripts/'+n:h for n,h in payload['source_hashes'].items()},
            source:hash_bytes(root/source),'scripts/structured_records_evidence.py':hash_bytes(root/'scripts/structured_records_evidence.py'),
            input_path:hash_bytes(root/input_path)},
        'resource_profile':{'source':source,'source_sha256':hash_bytes(root/source),'functions':{
            'records':{'max_calls':1,'limits':limits['records']},'coordinate':{'max_calls':1,'limits':limits['coordinate']}},
            'excluded_functions':{},'call_bound_basis':'One immutable role/task/mode payload, one coordinator and one single-GPU call, five fresh pilots plus the full bounded request list, zero retries.'}}
    checked=validate_resource_profile(config,root);config['upper_bound_usd']=checked['minimum_compute_reservation_usd']
    validate_resource_profile(config,root);return config


@app.local_entrypoint()
def main(attempt:str,input_path:str,launch:bool=False):
    """Frozen-input launch only after exact central and independent admission."""
    from structured_records_worker import write_exclusive,validate_payload
    path=(ROOT/input_path).resolve();path.relative_to(ROOT);payload=json.loads(path.read_text());validate_payload(payload)
    if payload['attempt']!=attempt or input_path!=f'stimuli/{payload["experiment"]}_input.json':raise ValueError('Exact structured attempt/input differs')
    if not launch:
        print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','resource_config':resource_config(ROOT,payload,input_path)}));return
    from structured_records_evidence import audit_central_admission
    audit_central_admission(ROOT,payload,input_path);authorize(ROOT,payload)
    result=coordinate.remote(payload)
    destination=ROOT/'qualification_status'/f'{payload["experiment"]}.json'
    if destination.exists():
        if json.loads(destination.read_text())!=result:raise ValueError('Immutable local structured status differs')
    else:write_exclusive(destination,result)
    print(json.dumps(result))
