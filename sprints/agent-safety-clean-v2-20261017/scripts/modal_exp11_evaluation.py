"""Prepared bounded Exp1.1 action/QA execution; root is sole paid-launch authority."""
from pathlib import Path
import json
import modal

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-exp11-evaluation')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
PACKAGE_FILES=('exp11_action_worker.py','exp11_probe_worker.py','exp11_design.py','qa_semantic_protocol.py',
    'exp23_executor_worker.py','audit_exp23_launch.py','exp23_author_worker.py','exp23_design.py',
    'structured_records_worker.py','scenario_contract.py','strict_parser.py','sandbox_runtime.py',
    'primary_model_loader.py','primary_fp8_inventory.py','llama_model_loader.py','llama_serving_worker.py')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12').entrypoint([])
       .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn'}))
for filename in PACKAGE_FILES:image=image.add_local_file(ROOT/'scripts'/filename,'/opt/cleanroom/'+filename)


def validate_batch(batch):
    import re
    from exp11_action_worker import validate_payload as validate_action,require,digest
    from exp11_probe_worker import validate_payload as validate_probe
    require(batch['schema']=='exp11_batch_payload_v1' and bool(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,31}',batch['attempt'])),'Invalid batch identity')
    require(batch['job']=='exp11_evaluation_'+batch['attempt'] and 1<=len(batch['items'])<=2,'At most two fixed single-GPU items per admitted job')
    keys=[]
    for item in batch['items']:
        require(item['kind'] in ('action','probe'),'Unknown Exp1.1 response family')
        payload=item['payload'];(validate_action if item['kind']=='action' else validate_probe)(payload)
        require(payload['ordering_audit'].get('exp11_overlap_authorized') is True,'Actual Exp2.3 executor start is required')
        require({'tokenizer_config.json','tokenizer.json'}<=set(item['tokenizer_files']),'Frozen tokenizer source required')
        if item['kind']=='action':require(item['tokenizer_files']==payload['tokenizer_files'],'Action tokenizer binding differs')
        for name,h in item['tokenizer_files'].items():require(Path(name).name==name and bool(re.fullmatch('[0-9a-f]{64}',h)),'Invalid tokenizer hash')
        keys.append((item['kind'],payload['arm'],payload.get('cot_mode','cot_on'),payload.get('chunk_index',payload.get('probe_kind'))))
    require(len(keys)==len(set(keys)) and len({i['payload']['experiment'] for i in batch['items']})==len(keys),'Duplicate item/writer in batch')
    require(len({i['payload']['design_sha256'] for i in batch['items']})==1,'Mixed full-study designs in batch')
    require(set(batch['package_sha256'])==set(PACKAGE_FILES),'Packaged worker source set changed')
    for item in batch['items']:
        require(all(batch['package_sha256'].get(n)==h for n,h in item['payload']['source_hashes'].items()),
                'Payload runtime differs from packaged source bytes')
    require(all(bool(re.fullmatch('[0-9a-f]{64}',h)) for h in batch['package_sha256'].values()),'Invalid packaged source digest')
    require(batch['batch_sha256']==digest({k:v for k,v in batch.items() if k!='batch_sha256'}),'Batch digest changed')


def build_batch(attempt,items,package_sha256):
    from exp11_action_worker import digest
    batch={'schema':'exp11_batch_payload_v1','attempt':attempt,'job':'exp11_evaluation_'+attempt,
           'items':items,'package_sha256':package_sha256,'launch_authorized_by_preparation':False}
    batch['batch_sha256']=digest(batch);validate_batch(batch);return batch


def item_paths(root,item):
    experiment=item['payload']['experiment']
    return {'ledger':Path(root)/'group_calls'/(experiment+'.json'),
            'input':Path(root)/'exp11_inputs'/(experiment+'.json'),
            'tokenizers':Path(root)/'exp11_inputs'/(experiment+'_tokenizers.json')}


def validate_ledger(root,batch,item):
    from exp11_action_worker import require
    ledger=json.loads(item_paths(root,item)['ledger'].read_text())
    require(ledger['state']=='SPAWNED' and ledger['batch_sha256']==batch['batch_sha256'] and
            ledger['payload_sha256']==item['payload']['payload_sha256'] and ledger['experiment']==item['payload']['experiment'],
            'Missing committed exact Exp1.1 item call')
    require(ledger['worker_module']=='modal_exp11_evaluation' and ledger['worker_function']=='evaluate' and
            ledger['kind']==item['kind'] and ledger['job']==batch['job'],'Item call targets a different runner/job')
    require(isinstance(ledger.get('call_id'),str) and ledger['call_id'].startswith('fc-'),'No actual item call ID')
    return ledger


@app.function(image=image,gpu='H200',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
              timeout=7200,startup_timeout=300,retries=0,max_containers=2,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def evaluate(batch,index):
    import os,subprocess,sys,time
    sys.path.insert(0,'/opt/cleanroom')
    from exp11_action_worker import require,hash_bytes,write_exclusive
    validate_batch(batch);require(type(index) is int and 0<=index<len(batch['items']),'Unknown immutable batch index')
    item=batch['items'][index];payload=item['payload'];root=Path('/artifacts')/NS;weights.reload()
    for check in range(30):
        artifacts.reload()
        try:
            ledger=validate_ledger(root,batch,item)
            require(ledger['call_id']==modal.current_function_call_id(),'Remote worker differs from durable call ID');break
        except FileNotFoundError:
            if check==29:raise
            time.sleep(1)
        except ValueError as exc:
            if check==29 or 'Missing committed exact' not in str(exc):raise
            time.sleep(1)
    for name,h in batch['package_sha256'].items():require(hash_bytes(Path('/opt/cleanroom')/name)==h,'Packaged runtime differs from frozen batch')
    paths=item_paths(root,item)
    require(json.loads(paths['input'].read_text())==payload and json.loads(paths['tokenizers'].read_text())==item['tokenizer_files'],'Worker input/tokenizer file differs')
    write_exclusive(root/'exp11_worker_starts'/f'{payload["experiment"]}.json',{
        'call_id':ledger['call_id'],'payload_sha256':payload['payload_sha256'],'batch_sha256':batch['batch_sha256']});artifacts.commit()
    environment=dict(os.environ,TRITON_CACHE_DIR='/tmp/exp11-evaluation/'+payload['experiment'])
    child=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/exp11_action_worker.py',str(paths['input']),item['kind'],str(paths['tokenizers'])],
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
    if code:
        failure=root/'exp11_worker_failures'/f'{payload["experiment"]}.json'
        if not failure.exists():write_exclusive(failure,{'status':'WORKER_FAILED','exit_code':code,
            'payload_sha256':payload['payload_sha256'],'call_id':ledger['call_id'],'automatic_retry':False});artifacts.commit()
        raise RuntimeError('Exp1.1 item failed; preserve exact attempt and all returned evidence')
    status=root/('evaluation_status' if item['kind']=='action' else 'qualification_status')/f'{payload["experiment"]}.json'
    return json.loads(status.read_text())


@app.function(image=image,cpu=(1,1),memory=(4096,4096),timeout=15600,startup_timeout=300,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':artifacts})
def coordinate(batch):
    import sys
    sys.path.insert(0,'/opt/cleanroom')
    from exp11_action_worker import write_exclusive,hash_bytes,require,utc_now,audit_streams,audit_probe_evidence
    validate_batch(batch);artifacts.reload();root=Path('/artifacts')/NS
    calls=[]
    for index,item in enumerate(batch['items']):
        payload=item['payload'];paths=item_paths(root,item)
        for name,h in payload['input_sources'].items():require(hash_bytes(root/name)==h,'Audited prerequisite changed before Exp1.1 submission')
        for path,value in ((paths['input'],payload),(paths['tokenizers'],item['tokenizer_files'])):
            if path.exists():require(json.loads(path.read_text())==value,'Immutable Exp1.1 input changed')
            else:write_exclusive(path,value)
        if paths['ledger'].exists():
            ledger=validate_ledger(root,batch,item);call=modal.FunctionCall.from_id(ledger['call_id'])
        else:
            ledger={'schema':'exp11_evaluation_call_v1','state':'SPAWN_INTENT','kind':item['kind'],'job':batch['job'],
                'experiment':payload['experiment'],'payload_sha256':payload['payload_sha256'],'batch_sha256':batch['batch_sha256'],
                'worker_module':'modal_exp11_evaluation','worker_function':'evaluate','gpu':'H200:1','spawn_intent_utc':utc_now()}
            write_exclusive(paths['ledger'],ledger);artifacts.commit()
            call=evaluate.spawn(batch,index)
            ledger.update(state='SPAWNED',call_id=call.object_id,submitted_utc=utc_now())
            temporary=paths['ledger'].with_suffix('.submitted.json');write_exclusive(temporary,ledger);temporary.replace(paths['ledger']);artifacts.commit()
        calls.append((item,call))
    results=[]
    for item,call in calls:
        result=call.get();artifacts.reload();payload=item['payload']
        audit=(audit_streams(root,payload) if item['kind']=='action' else audit_probe_evidence(root,payload,item['tokenizer_files']))
        require(result['payload_sha256']==payload['payload_sha256'],'Returned item status identity changed')
        if item['kind']=='action':require(result['audit']==audit,'Action status differs from independent full replay')
        else:require(result['stream_audit']=={k:v for k,v in audit.items() if k not in ('exact_prompt_replay','runtime_context_sha256')},'Probe status differs from independent full replay')
        results.append({'experiment':payload['experiment'],'kind':item['kind'],'status':result['status'],'audit':audit})
    output={'schema':'exp11_batch_result_v1','job':batch['job'],'batch_sha256':batch['batch_sha256'],'items':results,
            'full_experiment_complete':False}
    target=root/'evaluation_status'/f'{batch["job"]}_batch.json'
    if target.exists():require(json.loads(target.read_text())==output,'Immutable batch result changed')
    else:write_exclusive(target,output);artifacts.commit()
    return output


def prepare_item(root,*,kind,attempt,arm,mode='cot_on',chunk_index=0,probe_kind='consequence_awareness',
                 bases_path,design_path,components_path,ordering_evidence_path,tokenizer_metadata):
    """Preparation can run now; missing/failed launch evidence fails closed."""
    import sys
    from admit_parallel_calibration import ReadBindings
    from exp11_admission import local_source_closure
    from exp11_probe_worker import prepare_group
    from exp11_action_worker import build_payload,hash_bytes,require,digest,RUNTIME_FILES
    root=Path(root).resolve();bindings=ReadBindings();sys.addaudithook(bindings.hook)
    baseline=prepare_group(root,attempt,arm,probe_kind,bases_path=bases_path,design_path=design_path,
        components_path=components_path,ordering_evidence_path=ordering_evidence_path)
    folder=(root/tokenizer_metadata).resolve();folder.relative_to(root)
    allowed={'tokenizer_config.json','tokenizer.json','chat_template.jinja','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','generation_config.json'}
    tokenizers={}
    for path in sorted(folder.iterdir()):
        if path.is_file():require(path.name in allowed,'Unexpected tokenizer metadata artifact');tokenizers[path.name]=hash_bytes(path)
    require({'tokenizer_config.json','tokenizer.json'}<=set(tokenizers),'Missing exact tokenizer source files')
    for name in PACKAGE_FILES:bindings.record(root/'scripts'/name)
    for name in local_source_closure(root,('phase1_readiness','phase2_admission','audit_exp23_launch','exp11_admission')):
        bindings.record(root/name)
    for name in ('scripts/modal_exp11_evaluation.py','scripts/exp11_admission.py','scripts/phase1_readiness.py','scripts/phase2_admission.py'):
        bindings.record(root/name)
    # Control-plane ledgers legitimately change when this very job is reserved.
    # Their relevant Exp2.3 job admission is independently rechecked at launch;
    # scientific sources, per-call evidence, and frozen resource profiles remain bound.
    mutable={'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}
    sources={**baseline['input_sources'],**{k:v for k,v in bindings.files.items() if k not in mutable}}
    for name,h in sources.items():require(hash_bytes(root/name)==h,'Prerequisite changed during preparation audit')
    if kind=='action':
        payload=build_payload(attempt,arm,mode,chunk_index,bases=baseline['bases'],manifest_sha256=baseline['identity']['model_revision'],
            design_path=baseline['design_path'],design_sha256=baseline['design_sha256'],readiness=baseline['readiness'],
            component_sources=baseline['component_sources'],ordering_evidence_path=baseline['ordering_evidence_path'],
            ordering_audit=baseline['ordering_audit'],input_sources=sources,source_hashes={n:hash_bytes(root/'scripts'/n) for n in RUNTIME_FILES},
            tokenizer_files=tokenizers)
    else:
        require(kind=='probe' and mode=='cot_on','Unknown/nonexecuting response scope')
        payload={**baseline,'input_sources':sources};payload['payload_sha256']=digest({k:v for k,v in payload.items() if k!='payload_sha256'})
    return {'kind':kind,'payload':payload,'tokenizer_files':tokenizers}


def authorize_batch(root,batch):
    from phase1_readiness import recompute
    from phase2_admission import require_exp11_ordering
    from audit_authored_scenarios import reconstruct
    from exp11_design import build_design,validate_bases
    from exp11_action_worker import require,hash_bytes
    validate_batch(batch);root=Path(root)
    for name,h in batch['package_sha256'].items():require(hash_bytes(root/'scripts'/name)==h,'Packaged runtime source changed')
    for item in batch['items']:
        value=item['payload']
        ready=recompute(root,value['component_sources']);require(ready==value['readiness'] and ready['primary_phase2_authorized'],'Phase1 readiness no longer independently passes')
        require(require_exp11_ordering(root,value['ordering_evidence_path'],expected_readiness=ready)==value['ordering_audit'],'Actual Exp2.3 start no longer independently passes')
        bases,_=reconstruct(root,ready['provenance']['same_author_attempt']);require(validate_bases(bases)==value['bases'],'Unified-author bases changed')
        design=json.loads((root/value['design_path']).read_text())
        require(design==build_design(bases,design['source_sha256'],design['source_path']),'Complete design reconstruction differs')
        for name,h in value['input_sources'].items():require(hash_bytes(root/name)==h,'Frozen prerequisite bytes changed before launch')
    return True


def resource_config(root,attempt,input_path):
    import math
    from launch_bounded_job import _decorators,validate_resource_profile
    from exp11_action_worker import hash_bytes
    root=Path(root);source='scripts/modal_exp11_evaluation.py';batch=json.loads((root/input_path).read_text());validate_batch(batch)
    limits=_decorators((root/source).read_text())
    config={'job':batch['job'],'gpus':2,'single_gpu_workers':2,'tp8_workers':0,'upper_bound_usd':30,
        'log':f'infrastructure/{batch["job"]}.log','command':[str(root/'.venv/bin/modal'),'run','--detach',source,
            '--attempt',attempt,'--input-path',input_path,'--launch'],
        'frozen_files':{**{name:h for item in batch['items'] for name,h in item['payload']['input_sources'].items()},
            **{'scripts/'+n:h for n,h in batch['package_sha256'].items()},
            'scripts/exp11_admission.py':hash_bytes(root/'scripts/exp11_admission.py'),input_path:hash_bytes(root/input_path)},
        'resource_profile':{'source':source,'source_sha256':hash_bytes(root/source),'functions':{
            'evaluate':{'max_calls':len(batch['items']),'limits':limits['evaluate']},
            'coordinate':{'max_calls':1,'limits':limits['coordinate']}},'excluded_functions':{},
            'call_bound_basis':'One coordinator submits exactly the frozen one or two item GPU calls; each contains five fresh pilots then its fixed full chunk/group, with no retries or additional children.'}}
    checked=validate_resource_profile(config,root);config['upper_bound_usd']=checked['minimum_compute_reservation_usd']
    validate_resource_profile(config,root);return config


@app.local_entrypoint()
def main(attempt:str,input_path:str,launch:bool=False):
    from exp11_action_worker import require
    path=(ROOT/input_path).resolve();path.relative_to(ROOT);batch=json.loads(path.read_text());validate_batch(batch)
    require(batch['attempt']==attempt,'Batch attempt differs from frozen input')
    if not launch:
        print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','resource_config':resource_config(ROOT,attempt,input_path)}));return
    from exp11_admission import audit_central_admission
    audit_central_admission(ROOT,batch,input_path)
    authorize_batch(ROOT,batch)
    print(json.dumps(coordinate.remote(batch)))
