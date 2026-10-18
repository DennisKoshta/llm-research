"""Prepared Monitor5 app; root central admission is the sole paid-launch path."""
from pathlib import Path
import json
import modal

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-exp22-monitor')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
RUNTIME_FILES=('exp22_monitor_worker.py','exp22_monitor_evidence.py','exp22_design.py','exp22_evaluation.py',
 'exp23_design.py','exp23_author_worker.py','scenario_contract.py','strict_parser.py','structured_records_worker.py',
 'primary_model_loader.py','primary_fp8_inventory.py','llama_model_loader.py','llama_serving_worker.py')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12').entrypoint([])
    .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn'}))
for name in RUNTIME_FILES:image=image.add_local_file(ROOT/'scripts'/name,'/opt/cleanroom/'+name)


@app.function(image=image,gpu='H200',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
    timeout=3600,startup_timeout=300,retries=0,max_containers=2,scaledown_window=2,
    volumes={'/weights':weights,'/artifacts':artifacts})
def monitor(payload,stage):
    import os,subprocess,sys,time
    sys.path.insert(0,'/opt/cleanroom')
    from exp22_monitor_worker import validate_payload,require,write_exclusive,paths
    from exp22_monitor_evidence import durable_call,ledger_path
    validate_payload(payload);root=Path('/artifacts')/NS;weights.reload()
    for check in range(30):
        artifacts.reload()
        try:
            call=durable_call(root,payload,stage);break
        except FileNotFoundError:
            if check==29:raise
            time.sleep(1)
        except ValueError as exc:
            if 'Committed concrete monitor call required' not in str(exc) or check==29:raise
            time.sleep(1)
    require(call['call_id']==modal.current_function_call_id(),'Actual monitor call differs from durable admission')
    write_exclusive(ledger_path(root,payload,stage).parent/(stage+'_worker_started.json'),
        {'call_id':call['call_id'],'payload_sha256':payload['payload_sha256'],'stage':stage});artifacts.commit()
    path=root/'monitor_inputs'/(payload['experiment']+'.json')
    child=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/exp22_monitor_worker.py',str(path),stage],
        env=dict(os.environ,TRITON_CACHE_DIR='/tmp/exp22-monitor/'+payload['experiment']+'/'+stage),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
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
    require(code==0,'Monitor worker failed; retain attempt, no automatic retry')
    return json.loads(paths(root,payload,stage)['status'].read_text())


@app.function(image=image,cpu=(1,1),memory=(4096,4096),timeout=8400,startup_timeout=300,
    retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':artifacts})
def coordinate(payload):
    import sys
    sys.path.insert(0,'/opt/cleanroom')
    from exp22_monitor_worker import validate_payload,require,hash_bytes,write_exclusive,utc_now,digest,safe_path
    from exp22_monitor_evidence import ledger_path,durable_call,audit_pilot_completion
    validate_payload(payload);root=Path('/artifacts')/NS;artifacts.reload()
    def unchanged():
        for name,h in payload['input_sources'].items():require(hash_bytes(safe_path(root,name))==h,'Frozen monitor prerequisite changed')
    unchanged();target=root/'monitor_inputs'/(payload['experiment']+'.json')
    if target.exists():require(json.loads(target.read_text())==payload,'Immutable attempt was rebound')
    else:write_exclusive(target,payload)
    artifacts.commit()
    def stage_call(stage,prior=None,status=None):
        path=ledger_path(root,payload,stage)
        if path.exists():
            call=durable_call(root,payload,stage)
            return modal.FunctionCall.from_id(call['call_id'])
        record={'schema':'exp22_monitor_call_v1','state':'SPAWN_INTENT','stage':stage,'job':payload['central_job'],
            'payload_path':str(target.relative_to(root)),'payload_sha256':payload['payload_sha256'],
            'worker_module':'modal_exp22_monitor','worker_function':'monitor','gpu':'H200:1','spawn_intent_utc':utc_now()}
        if stage=='full':record.update(pilot_call_id=durable_call(root,payload,'pilot')['call_id'],
            pilot_audit_sha256=digest(prior),pilot_completed_utc=status['finished_utc'])
        write_exclusive(path,record);artifacts.commit()
        call=monitor.spawn(payload,stage)
        record.update(state='SPAWNED',call_id=call.object_id,submitted_utc=utc_now())
        temporary=path.with_suffix('.submitted.json');write_exclusive(temporary,record);temporary.replace(path);artifacts.commit()
        return call
    result=stage_call('pilot').get();artifacts.reload()
    prior,status=audit_pilot_completion(root,payload)
    require(result==status,'Actual pilot result differs from independent durable evidence')
    unchanged();return stage_call('full',prior,status).get()



class ReadBindings:
    """Scope actual prerequisite reads to this root, without mutable ledger bytes."""
    def __init__(self,root):self.root=Path(root).resolve();self.files={};self.active=False;self.busy=False
    def record(self,path):
        if self.busy:return
        self.busy=True
        try:
            from exp22_monitor_worker import hash_bytes,require
            path=Path(path).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():return
            name=str(path.relative_to(self.root))
            if name.startswith(('.venv/','.omnibus-venv/')) or name in {'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}:return
            if '__pycache__' in path.parts and path.suffix=='.pyc':
                import importlib.util
                path=Path(importlib.util.source_from_cache(str(path)))
                if not path.is_file():return
                name=str(path.relative_to(self.root))
            h=hash_bytes(path);require(name not in self.files or self.files[name]==h,'Prerequisite mutated during audit')
            self.files[name]=h
        finally:self.busy=False
    def hook(self,event,args):
        import os
        if not self.active or self.busy or event!='open':return
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and not flags&(os.O_WRONLY|os.O_CREAT|os.O_TRUNC):self.record(os.fsdecode(path))
    def __enter__(self):
        import sys
        self.active=True;sys.addaudithook(self.hook);return self
    def __exit__(self,*args):self.active=False


def prepare(root,attempt,*,sweep_id,bundle_path,producer_selection_path,phase3_inputs,tokenizer_metadata):
    from exp22_monitor_worker import build_payload,require,hash_bytes,safe_path,TOKENIZER_NAMES
    from exp22_monitor_evidence import dependency_bundle
    from phase3_readiness import recompute
    from exp11_admission import local_source_closure
    root=Path(root).resolve()
    with ReadBindings(root) as bindings:
        gate=recompute(root,phase3_inputs)
        require(gate['phase3_authorized'] is True,'Complete fresh Phase3 gate did not pass; monitor generation forbidden')
        design,dependency=dependency_bundle(root,bundle_path,producer_selection_path,phase3_inputs,gate)
        from exp22_design import sweeps,ROLES
        selected=[s for s in sweeps(design) if s['sweep_id']==sweep_id];require(len(selected)==1,'Exact prospective sweep required')
        folder=safe_path(root,tokenizer_metadata);tokenizers={};tokenizer_sources={}
        for path in sorted(folder.iterdir()):
            if path.is_file():
                require(path.name in TOKENIZER_NAMES,'Unexpected tokenizer capture file')
                tokenizers[path.name]=hash_bytes(path);tokenizer_sources[path.name]=str(path.relative_to(root));bindings.record(path)
        primary=selected[0]['model_role']==ROLES[0];checkpoint=None;model_provenance=None
        if primary:
            path=root/'checkpoints_full_residual_v2/intact_materialization.json';checkpoint=hash_bytes(path);bindings.record(path)
        else:
            model_provenance={}
            for key,name in [('download','infrastructure/nvidia__Llama-3.3-70B-Instruct-FP8_download.json'),('preflight','infrastructure/llama_preflight.json')]:
                value=json.loads((root/name).read_text())
                require(value['repo_id']=='nvidia/Llama-3.3-70B-Instruct-FP8' and value['revision']=='fde04ee76a27704c88f569542ef023b57d4d0362','Pinned Llama identity differs')
                model_provenance[key]={'path':name,'sha256':hash_bytes(root/name)};bindings.record(root/name)
        for name in ('FINAL_RESEARCH_SPECIFICATION.md','scripts/modal_exp22_monitor.py','scripts/historical_admission.py'):
            bindings.record(root/name)
        for name in local_source_closure(root,('phase3_readiness','exp22_monitor_evidence','exp22_producer_evidence')):bindings.record(root/name)
        for name in RUNTIME_FILES:bindings.record(root/'scripts'/name)
        mutable={'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}
        sources={k:v for k,v in bindings.files.items() if k not in mutable}
        sources.update(dependency['source_sha256'])
        for name,h in sources.items():require(hash_bytes(safe_path(root,name))==h,'Source changed during independent preparation')
        return build_payload(attempt,design,sweep_id,phase3_inputs=phase3_inputs,phase3_audit=gate,dependency_audit=dependency,
            input_sources=sources,source_hashes={name:hash_bytes(root/'scripts'/name) for name in RUNTIME_FILES},
            tokenizer_files=tokenizers,tokenizer_source_paths=tokenizer_sources,checkpoint_sha256=checkpoint,model_provenance=model_provenance)


def resource_config(root,attempt,input_path):
    from launch_bounded_job import _decorators,validate_resource_profile
    from exp22_monitor_worker import hash_bytes,validate_payload,require
    root=Path(root);payload=json.loads((root/input_path).read_text());validate_payload(payload)
    require(payload['attempt']==attempt and input_path==f'stimuli/{payload["experiment"]}_input.json','Exact prospective input required')
    source='scripts/modal_exp22_monitor.py';limits=_decorators((root/source).read_text())
    frozen={**payload['input_sources'],**{'scripts/'+n:h for n,h in payload['source_hashes'].items()},
        source:hash_bytes(root/source),'scripts/historical_admission.py':hash_bytes(root/'scripts/historical_admission.py'),
        input_path:hash_bytes(root/input_path)}
    config={'job':payload['central_job'],'gpus':2,'single_gpu_workers':2,'tp8_workers':0,'upper_bound_usd':14,
        'log':f'infrastructure/{payload["experiment"]}.log',
        'command':[str(root/'.venv/bin/modal'),'run','--detach',source,'--attempt',attempt,'--input-path',input_path,'--launch'],
        'frozen_files':frozen,'resource_profile':{'source':source,'source_sha256':hash_bytes(root/source),
            'functions':{'monitor':{'max_calls':2,'limits':limits['monitor']},'coordinate':{'max_calls':1,'limits':limits['coordinate']}},
            'excluded_functions':{},'call_bound_basis':'One five-trial pilot, one full call only after independent completed H5; one CPU coordinator; retries zero.'}}
    validate_resource_profile(config,root);return config


@app.local_entrypoint()
def main(attempt:str,input_path:str,launch:bool=False):
    from exp22_monitor_worker import validate_payload,require,safe_path
    from exp22_monitor_evidence import authorize_payload,audit_central_admission
    payload=json.loads(safe_path(ROOT,input_path).read_text());validate_payload(payload);require(payload['attempt']==attempt,'Attempt mismatch')
    if not launch:
        print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','resource_config':resource_config(ROOT,attempt,input_path)}));return
    audit_central_admission(ROOT,payload,input_path)
    authorize_payload(ROOT,payload)
    audit_central_admission(ROOT,payload,input_path)
    print(json.dumps(coordinate.remote(payload)))
