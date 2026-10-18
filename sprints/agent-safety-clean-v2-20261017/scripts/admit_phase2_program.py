"""Bounded Phase2 preparation controller; root remains the only paid-launch authority.

Default: inspect explicit selections and emit a state/intent document. --freeze-local
may create immutable inputs/configs, never invoke Modal, launch_bounded_job.launch,
or change a live ledger, raw stream, receipt, source, or prior preparation.
Every invocation re-audits prerequisites. A missing or unresolved existing job is
recovered, never retried under a fresh ID. This is not full-program completion.
"""
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
SCHEMA='phase2_program_selection_v1'
TOKENIZERS={'tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json',
            'added_tokens.json','vocab.json','merges.txt'}


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(value):return sha256(canonical(value)).hexdigest()


def inside(root,name):
    require(isinstance(name,str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts,'Path outside current sprint')
    path=(Path(root)/name).resolve();path.relative_to(Path(root).resolve());return path


def read(root,name):
    from phase1_readiness import strict_json
    return strict_json(inside(root,name).read_text())


def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()


def default_selection():
    primary='infrastructure/calibration_provenance/author_v1_c2/cpu_v1'
    return {'schema':SCHEMA,'program_id':'p2v1','components_path':'infrastructure/phase1_component_paths_q4.json',
        'author_attempt':'v3_cap16384','author_payload_path':'stimuli/exp23_author_v3_cap16384_input.json','structured_payload_paths':[],
        'executor_payload_paths':[],'exp11_batch_paths':[],'qa_judge_payload_paths':[],'exp23_start_evidence':None,
        'runtime_metadata_path':'infrastructure/runtime_metadata/stock_vllm031_v1.json',
        'tokenizer_sources':{'primary':{'folder':primary+'/checkpoint_doses_v1/tokenizer_audit','capture_path':primary+'/capture.json'},
            'llama':{'folder':'infrastructure/tokenizer_metadata/llama_pinned_v1','capture_path':'infrastructure/tokenizer_metadata/llama_pinned_v1/capture.json'}},
        'exp11_sources':{'bases_path':'stimuli/authored_scenarios_v1.jsonl','design_path':'stimuli/exp11_design_v1.json'}}


def validate_selection(value):
    require(isinstance(value,dict) and set(value)==set(default_selection()) and value['schema']==SCHEMA,'Exact program selection schema required')
    require(bool(re.fullmatch('[a-z0-9][a-z0-9_]{0,11}',value['program_id'])),'Invalid program ID')
    require(bool(re.fullmatch('[a-z0-9][a-z0-9_-]{0,47}',value['author_attempt'])) and
        value['author_payload_path']==f'stimuli/exp23_author_{value["author_attempt"]}_input.json','Explicit canonical author attempt required')
    require(set(value['tokenizer_sources'])=={'primary','llama'},'Both pinned model tokenizer sources required')
    for name in ('structured_payload_paths','executor_payload_paths','exp11_batch_paths','qa_judge_payload_paths'):
        require(isinstance(value[name],list) and len(value[name])==len(set(value[name])),'Duplicate selected payload paths')
    for descriptor in value['tokenizer_sources'].values():
        require(isinstance(descriptor,dict) and set(descriptor)=={'folder','capture_path'},'Tokenizer source descriptor differs')
    if value['exp11_sources'] is not None:
        require(set(value['exp11_sources'])=={'bases_path','design_path'},'Exp1.1 requires exact bases and full design')


def write_once(path,data):
    """Atomic publication with exact-byte idempotence; never replace frozen bytes."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==data,'Frozen preparation already exists with different bytes: '+str(path));return
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        try:os.link(tmp,path)
        except FileExistsError:require(path.read_bytes()==data,'Concurrent frozen preparation differs')
    finally:os.unlink(tmp)


@contextmanager
def preparation_lock(root,program):
    folder=Path(root)/'infrastructure/phase2_program'/program;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'preparation.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


class Preparation:
    def __init__(self,root,selection,freeze):
        self.root=Path(root).resolve();self.selection=selection;self.freeze=freeze
        self.folder='infrastructure/phase2_program/'+selection['program_id']
        self.proposed=[];self.source_hashes={}
    def bind(self,name):
        path=inside(self.root,name);value=file_hash(path)
        require(name not in self.source_hashes or self.source_hashes[name]==value,'Prerequisite changed during preparation')
        self.source_hashes[name]=value;return value
    def document(self,tag,value):
        data=canonical(value);name=f'{self.folder}/{tag}_{sha256(data).hexdigest()}.json'
        return self.file(name,data)
    def file(self,name,data):
        path=inside(self.root,name);fingerprint=sha256(data).hexdigest()
        exists=path.exists()
        if exists:require(path.read_bytes()==data,'Existing immutable preparation differs')
        elif self.freeze:write_once(path,data)
        self.proposed.append({'path':name,'sha256':fingerprint,'bytes':len(data),'present':exists or self.freeze})
        return name
    def recheck(self):
        for name,h in self.source_hashes.items():require(file_hash(inside(self.root,name))==h,'Prerequisite changed while preparing intents')
    @contextmanager
    def capture_reads(self):
        """Bind raw/source reads during independent audits; ignore mutable control ledgers."""
        active=True;busy=False
        def hook(event,args):
            nonlocal busy
            if not active or busy or event!='open':return
            path,mode,flags=args
            if not isinstance(path,(str,bytes,os.PathLike)) or flags&(os.O_WRONLY|os.O_TRUNC|os.O_CREAT):return
            busy=True
            try:
                value=Path(os.fsdecode(path)).resolve()
                if not value.is_relative_to(self.root) or not value.is_file():return
                name=str(value.relative_to(self.root))
                if name.startswith(('.venv/','.omnibus-venv/')) or name in {
                    'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}:return
                if '__pycache__' in value.parts and value.suffix=='.pyc':
                    import importlib.util
                    value=Path(importlib.util.source_from_cache(str(value)))
                    if not value.is_file():return
                    name=str(value.relative_to(self.root))
                self.bind(name)
            finally:busy=False
        sys.addaudithook(hook)
        try:yield
        finally:active=False


def resource_snapshot(root):
    """Read-only conservative hint; central launcher must refresh actual cloud state."""
    from launch_bounded_job import budget_total,validate_counts
    names=('infrastructure/resource_allocations.json','infrastructure/budget_reservations.json')
    before={name:inside(root,name).read_bytes() for name in names}
    allocations,budget=(json.loads(before[name]) for name in names)
    reserved=budget_total(budget);limits=allocations['limits']
    require(set(limits)=={'gpus','single_gpu_workers','tp8_workers'} and all(type(n) is int and n>=0 for n in limits.values()),
            'Resource limits require nonnegative integers')
    require(limits['gpus']<=10 and limits['single_gpu_workers']<=6 and limits['tp8_workers']<=1,'Authorized global limits exceeded')
    jobs=allocations['jobs'];require(len({j['job'] for j in jobs})==len(jobs),'Duplicate central job IDs')
    reservations={r['job']:r for r in budget['reservations']}
    require({j['job'] for j in jobs}==set(reservations),'Allocation/budget crash window requires central recovery')
    allocated=dict.fromkeys(limits,0);unresolved=[]
    for job in jobs:
        counts=validate_counts(job,per_app=True)
        if job.get('allocation_status')!='released_after_modal_stop':
            for key in allocated:allocated[key]+=counts[key]
        if not job.get('app_id'):unresolved.append(job['job'])
    for name,data in before.items():require(inside(root,name).read_bytes()==data,'Central ledgers changed during snapshot')
    require(all(allocated[k]<=limits[k] for k in limits),'Current allocations exceed global capacity')
    return {'reserved_usd':reserved,'ceiling_usd':budget['ceiling_usd'],
        'available_budget_usd':budget['ceiling_usd']-reserved,'limits':limits,'allocated_upper_bound':allocated,
        'available_gpu_bound':max(0,limits['gpus']-allocated['gpus']),
        'unresolved_jobs':unresolved,'jobs':jobs,'ledger_sha256':{n:sha256(b).hexdigest() for n,b in before.items()},
        'cloud_state_verified':False,'fresh_central_launcher_admission_required':True}


def tokenizer_metadata(prep,family):
    source=prep.selection['tokenizer_sources'][family];capture=read(prep.root,source['capture_path']);prep.bind(source['capture_path'])
    if family=='llama':
        require(capture['schema']=='pinned_tokenizer_metadata_capture_v1' and
            capture['model_id']=='nvidia/Llama-3.3-70B-Instruct-FP8' and capture['revision']=='fde04ee76a27704c88f569542ef023b57d4d0362',
            'Llama tokenizer capture identity differs')
        hashes={n:r['sha256'] for n,r in capture['files'].items() if n in TOKENIZERS}
        for name in hashes:
            require(capture['files'][name]['volume_path']==f'{NS}/source_models/nvidia__Llama-3.3-70B-Instruct-FP8/{name}',
                    'Tokenizer capture escaped the pinned weights namespace')
    else:
        require(capture['schema']=='calibration_checkpoint_capture_v1' and
            capture['checkpoint_path']==f'/weights/{NS}/checkpoints_full_residual_v2/intact','Primary tokenizer capture identity differs')
        manifest='checkpoints_full_residual_v2/intact_materialization.json'
        require(prep.bind(manifest)==capture['manifest_sha256'],'Primary capture checkpoint manifest differs')
        hashes={n:h for n,h in capture['tokenizer_file_sha256'].items() if n in TOKENIZERS}
    require({'tokenizer.json','tokenizer_config.json'}<=set(hashes),'Missing exact tokenizer metadata files')
    folder=f'{prep.folder}/tokenizers/{family}_{digest(hashes)}'
    for name,h in hashes.items():
        source_name=source['folder']+'/'+name
        require(prep.bind(source_name)==h,'Captured tokenizer bytes differ')
        prep.file(folder+'/'+name,inside(prep.root,source_name).read_bytes())
    return folder


def audit_environment(prep):
    name=prep.selection['runtime_metadata_path'];metadata=read(prep.root,name);prep.bind(name)
    require(metadata['schema']=='serving_package_metadata_v1' and metadata['image_reference']=='vllm/vllm-openai:v0.31.0',
            'Require observed pinned stock serving package metadata')
    pending=[]
    for package in ('transformers','tokenizers','Jinja2'):
        expected=metadata['packages'][package]
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=expected:pending.append({'package':package,'required':expected,'observed':actual})
    return pending


def producer_records(root,payload_path,kind,author_attempt=None):
    """Inspect full raw records only after durable completion, never partial effects."""
    payload=read(root,payload_path);experiment=payload['experiment']
    require(payload_path==f'stimuli/{experiment}_input.json','Canonical producer payload required')
    if kind=='author':
        require(payload['attempt']==author_attempt and experiment=='exp23_author_'+author_attempt,'Recover only the explicitly selected author attempt')
        status_path=f'qualification_status/{experiment}_flash_next_uncensored.json';expected='AUTHORING_COMPLETE_PENDING_SEMANTIC_REVIEW'
        stem='flash_next_uncensored'
    else:
        require(kind=='structured','Unknown producer kind');status_path=f'qualification_status/{experiment}.json';expected='RECORDS_COMPLETE';stem=payload['model_role']
    if not inside(root,status_path).exists():
        return payload,None,{'state':'WAITING_PRODUCER_RECOVERY','payload_path':payload_path,'experiment':experiment,
            'remote_namespace':NS,'required_prefixes':[f'raw_transcripts/{experiment}',f'qualification_status/{experiment}',
                f'group_calls/{experiment}',f'pilots/{experiment}',f'{"author_artifacts" if kind=="author" else "structured_artifacts"}/{experiment}']+
                ([f'structured_worker_starts/{experiment}.json'] if kind=='structured' else []),
            'resubmit':False}
    status=read(root,status_path)
    require(status['status']==expected,'Producer completed with failed/incompatible status; no selective retry: '+experiment)
    if kind=='author':
        from exp23_author_worker import audit_streams
        checked=audit_streams(root,payload)
        require(checked['pilot_n']==5 and checked['pilot_pass'] and checked['authoring_H1_pass'] and
                checked['all_author_contracts_valid'] and not checked['pending_writeahead_trial_ids'],'Author mechanical audit failed')
    else:
        from structured_records_evidence import audit_complete_run
        checked=audit_complete_run(root,payload)
        require(checked['complete_data'] and all(checked[k] for k in ('H1','H2','H4','H5','all_records_valid')),'Producer raw audit failed')
    from exp23_author_worker import read_chain
    rows,_=read_chain(inside(root,f'raw_transcripts/{experiment}/{stem}.jsonl'),payload['payload_sha256'])
    return payload,[r for r in rows if r['split']=='artifact_generation'],None


def executor_nodes():
    from exp23_design import CORE_CONDITIONS,CLAUSE_CONDITIONS,COT_OFF_CONDITIONS,EXECUTORS
    nodes=[]
    for executor in EXECUTORS:
        for mode in ('cot_on','cot_off'):
            conditions=list(CORE_CONDITIONS+CLAUSE_CONDITIONS if mode=='cot_on' else COT_OFF_CONDITIONS)
            for offset in range(0,len(conditions),4):
                selected=conditions[offset:offset+4]
                nodes.append({'kind':'executor','executor':executor,'mode':mode,'conditions':selected,
                    'key':f'{executor}:{mode}:{offset//4}','full_n':96*len(selected),'fresh_pilot_n':5})
    return nodes


def exp11_nodes():
    from exp11_action_worker import execution_plan
    groups=execution_plan()['groups'];nodes=[]
    for index in range(0,len(groups),2):
        nodes.append({'kind':'exp11','key':'exp11_batch_'+str(index//2),'items':groups[index:index+2],
                      'full_n':sum(r['evaluation_n'] for r in groups[index:index+2]),'fresh_pilot_n':10})
    return nodes


def node_attempt(program,node):
    prefix={'structured':'s','qa_grader':'g','executor':'e','exp11':'q'}[node['kind']]
    return program+'_'+prefix+'_'+digest(node)[:12]


class ContextPreflightError(ValueError):
    def __init__(self,report_path,report):
        self.report_path=report_path;self.report=report
        super().__init__('BLOCKED_CONTEXT_PREFLIGHT: '+report_path+'; no clipping or payload publication')


def context_preflight(prep,node,payload,tokenizers,*,tokenizer_loader=None):
    """Render all immutable initial prompts; unknown action histories stay runtime guarded."""
    require(not audit_environment(prep),'Exact stock tokenizer environment required for context preflight')
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader=lambda folder:AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False)
    if node['kind']=='exp11':
        groups=[(item['kind'],item['payload'],'primary',item['tokenizer_files']) for item in payload['items']]
    else:
        groups=[('record' if node['kind'] in {'structured','qa_grader'} else 'action',payload,
                 payload['identity']['family'],payload['tokenizer_files'])]
    rows=[];loaded={};files={}
    for kind,group,family,hashes in groups:
        folder=inside(prep.root,tokenizers[family])
        require({'tokenizer.json','tokenizer_config.json'}<=set(hashes),'Missing preflight tokenizer files')
        require({p.name for p in folder.iterdir() if p.is_file()}==set(hashes),'Preflight tokenizer file membership differs')
        for name,h in hashes.items():
            require(name in TOKENIZERS,'Unexpected preflight tokenizer filename')
            path=str((folder/name).relative_to(prep.root))
            require(prep.bind(path)==h,'Preflight tokenizer bytes changed');files[path]=h
        key=(family,digest(hashes))
        if key not in loaded:loaded[key]=tokenizer_loader(folder)
        tokenizer=loaded[key];seen=set()
        require(group['max_model_len']==16384,'Preflight must match current immutable worker context')
        for split in ('pilots','requests'):
            require(group[split],'Empty preflight request partition')
            for request in group[split]:
                require(request['trial_id'] not in seen,'Duplicate preflight request ID');seen.add(request['trial_id'])
                messages=deepcopy(request['messages'])
                if kind!='action' and family=='llama':
                    from llama_model_loader import rendered_messages
                    messages=rendered_messages(request)
                options={'enable_thinking':request['cot_mode']=='cot_on'} if family=='primary' else {}
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**options)
                require(isinstance(prompt,str) and prompt,'Invalid preflight rendered prompt')
                if kind!='action' and family=='llama':require(not prompt.rstrip().endswith('<think>'),'Forced Llama reasoning prefix')
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                require(ids and all(type(t) is int and t>=0 for t in ids),'Invalid preflight prompt token IDs')
                cap=request['max_new_tokens'];require(type(cap) is int and cap>0,'Invalid generation cap')
                fits=len(ids)+cap<=group['max_model_len']
                if fits:
                    if kind=='action':
                        from exp23_executor_worker import render
                        actual_prompt,actual_ids=render(request['messages'],request,tokenizer,family)
                    else:
                        from structured_records_worker import render_request
                        actual_messages,actual_prompt,actual_ids=render_request(request,tokenizer,family)
                        require(actual_messages==messages,'CPU/runtime rendered messages differ')
                    require(actual_prompt==prompt and list(actual_ids)==list(ids),'CPU/runtime prompt replay differs')
                rows.append({'experiment':group['experiment'],'split':split,'trial_id':request['trial_id'],
                    'request_sha256':digest(request),'family':family,'prompt_sha256':sha256(prompt.encode()).hexdigest(),
                    'prompt_token_ids_sha256':digest(list(ids)),'prompt_tokens':len(ids),'max_new_tokens':cap,
                    'required_context_tokens':len(ids)+cap,'max_model_len':group['max_model_len'],'fits':fits})
    prep.recheck()
    report={'schema':'phase2_all_request_context_preflight_v1','node':node['key'],'payload_sha256':digest(payload),
        'runtime_metadata_path':prep.selection['runtime_metadata_path'],'tokenizer_files':files,
        'all_initial_requests_checked':True,'unknown_generated_history_preflighted':False,
        'future_action_history_guard':'Existing runtime render checks before every generated turn; never clip',
        'all_fit':all(r['fits'] for r in rows),'minimum_initial_context_tokens':max(r['required_context_tokens'] for r in rows),
        'requests':rows,'request_count':len(rows)}
    report_path=prep.document('context_preflight',report)
    if not report['all_fit']:raise ContextPreflightError(report_path,report)
    return report_path


def prepare_node(prep,node,*,bundle_path,runs_path,tokenizers,ordering_path):
    """Exact prepared APIs; no .remote(), .spawn(), Modal CLI, or launcher call."""
    root=prep.root;program=prep.selection['program_id'];attempt=node_attempt(program,node)
    if node['kind'] in {'structured','qa_grader'}:
        import modal_structured_records as runner
        request_path=prep.document('requests',node['requests'])
        payload=runner.prepare(root,attempt,request_path,prep.selection['components_path'],tokenizer_metadata=tokenizers[node['family']])
        preflight_path=context_preflight(prep,node,payload,tokenizers)
        input_path=f'stimuli/{payload["experiment"]}_input.json';prep.file(input_path,canonical(payload))
        config=runner.resource_config(root,payload,input_path)
    elif node['kind']=='executor':
        import modal_exp23_executor as runner
        family='primary' if node['executor']=='primary_intact_executor' else 'llama'
        payload=runner.prepare(root,attempt,executor=node['executor'],mode=node['mode'],conditions=node['conditions'],
            bundle_path=bundle_path,runs_path=runs_path,components_path=prep.selection['components_path'],tokenizer_metadata=tokenizers[family])
        preflight_path=context_preflight(prep,node,payload,tokenizers)
        input_path=f'stimuli/{payload["experiment"]}_input.json';prep.file(input_path,canonical(payload))
        config=runner.resource_config(root,attempt,input_path)
    else:
        import modal_exp11_evaluation as runner
        sources=prep.selection['exp11_sources'];require(sources is not None,'Missing independently reconstructed Exp1.1 bases/full design paths')
        items=[]
        for index,item in enumerate(node['items']):
            items.append(runner.prepare_item(root,kind=item['kind'],attempt=attempt+'_'+str(index),arm=item['arm'],mode=item['cot_mode'],
                chunk_index=item.get('chunk_index',0),probe_kind=item.get('probe_kind','consequence_awareness'),
                bases_path=sources['bases_path'],design_path=sources['design_path'],components_path=prep.selection['components_path'],
                ordering_evidence_path=ordering_path,tokenizer_metadata=tokenizers['primary']))
        payload=runner.build_batch(attempt,items,{n:file_hash(root/'scripts'/n) for n in runner.PACKAGE_FILES})
        preflight_path=context_preflight(prep,node,payload,tokenizers)
        input_path=f'stimuli/{payload["job"]}_input.json';prep.file(input_path,canonical(payload))
        config=runner.resource_config(root,attempt,input_path)
    config['frozen_files'].update(prep.source_hashes)
    config['frozen_files'][preflight_path]=file_hash(inside(root,preflight_path))
    # Retain the precise preparation recipe rather than mutable global selections.
    recipe_path=prep.document('recipe',{'node':node,'components_path':prep.selection['components_path'],
        'bundle_path':bundle_path,'runs_path':runs_path,'ordering_evidence_path':ordering_path})
    config['frozen_files'][recipe_path]=file_hash(inside(root,recipe_path))
    from launch_bounded_job import validate_resource_profile,verify_frozen_files
    require(config['gpus']==2 and config['single_gpu_workers']==2 and config['tp8_workers']==0,'Prepared app exceeds two-single-GPU bound')
    validate_resource_profile(config,root);verify_frozen_files(config,root)
    config_path=prep.document('launch_config',config)
    manifest={'schema':'phase2_prepared_job_v1','node':node,'attempt':attempt,'job':config['job'],
        'input_path':input_path,'input_sha256':file_hash(inside(root,input_path)),
        'config_path':config_path,'config_sha256':file_hash(inside(root,config_path)),
        'context_preflight_path':preflight_path,'context_preflight_sha256':file_hash(inside(root,preflight_path))}
    prep.file(f'{prep.folder}/jobs/{config["job"]}.json',canonical(manifest))
    return manifest,config


def registered_jobs(prep):
    """Immutable preparation manifests are leases, even before central admission."""
    result=[]
    for path in sorted((prep.root/prep.folder/'jobs').glob('*.json')):
        row=read(prep.root,str(path.relative_to(prep.root)))
        require(row['schema']=='phase2_prepared_job_v1','Unknown prepared job manifest')
        require(row['attempt']==node_attempt(prep.selection['program_id'],row['node']),'Prepared node identity differs')
        require(file_hash(inside(prep.root,row['input_path']))==row['input_sha256'],'Frozen prepared payload bytes changed')
        require(file_hash(inside(prep.root,row['config_path']))==row['config_sha256'],'Frozen resource configuration changed')
        preflight=row['context_preflight_path'];require(file_hash(inside(prep.root,preflight))==row['context_preflight_sha256'],'Context preflight changed')
        report=read(prep.root,preflight);require(report['all_fit'] and report['all_initial_requests_checked'],'Missing passing complete initial-context preflight')
        require(read(prep.root,row['config_path'])['frozen_files'].get(preflight)==row['context_preflight_sha256'],'Context preflight not centrally frozen')
        result.append(row)
    require(len({j['job'] for j in result})==len(result),'Duplicate prepared job')
    return result


def existing_node_keys(root,selection,prepared):
    """Explicit payloads or owned manifests reserve whole fixed groups, never rows."""
    keys=set();structured=list(selection['structured_payload_paths']);executors=list(selection['executor_payload_paths'])
    batches=list(selection['exp11_batch_paths']);judges=list(selection['qa_judge_payload_paths'])
    for job in prepared:
        kind=job['node']['kind'];key=(kind,job['node']['key'])
        require(key not in keys,'Duplicate prepared scientific node');keys.add(key)
        if kind=='structured':structured.append(job['input_path'])
        if kind=='executor':executors.append(job['input_path'])
        if kind=='exp11':batches.append(job['input_path'])
        if kind=='qa_grader':judges.append(job['input_path'])
    require(len(set(structured))==len(structured),'Duplicate producer selection; do not list owned manifests twice')
    for path in selection['executor_payload_paths']:
        from exp23_executor_worker import validate_payload
        value=read(root,path);validate_payload(value)
        matches=[node for node in executor_nodes() if (node['executor'],node['mode'],node['conditions'])==
                 (value['executor'],value['cot_mode'],value['conditions'])]
        require(len(matches)==1,'Selected executor is not a whole prospectively fixed partition')
        key=('executor',matches[0]['key']);require(key not in keys,'Duplicate executor partition');keys.add(key)
    for path in selection['exp11_batch_paths']:
        from modal_exp11_evaluation import validate_batch
        value=read(root,path);validate_batch(value)
        actual=[(i['kind'],i['payload']['arm'],i['payload'].get('cot_mode','cot_on'),
                 i['payload'].get('chunk_index',i['payload'].get('probe_kind'))) for i in value['items']]
        def node_items(node):return [(i['kind'],i['arm'],i['cot_mode'],i.get('chunk_index',i.get('probe_kind'))) for i in node['items']]
        matches=[node for node in exp11_nodes() if node_items(node)==actual]
        require(len(matches)==1,'Selected Exp1.1 batch is not a complete fixed partition')
        key=('exp11',matches[0]['key']);require(key not in keys,'Duplicate Exp1.1 partition');keys.add(key)
    require(len(set(batches))==len(batches) and len(set(judges))==len(judges),'Duplicate action/QA payload selection')
    return keys,structured,executors,batches,judges


def audit_respondent_item(prep,batch,item):
    """Completed nonexecuting QA only: actual call/source and exact prompt replay."""
    from modal_exp11_evaluation import validate_ledger,authorize_batch,item_paths
    from exp11_admission import audit_worker_admission
    from audit_exp23_launch import lookup_backend
    from exp11_action_worker import audit_probe_evidence
    root=prep.root;payload=item['payload'];experiment=payload['experiment']
    with prep.capture_reads():
        authorize_batch(root,batch)
        ledger=validate_ledger(root,batch,item);audit_worker_admission(root,payload)
        started=read(root,f'exp11_worker_starts/{experiment}.json')
        require(started=={'call_id':ledger['call_id'],'payload_sha256':payload['payload_sha256'],
            'batch_sha256':batch['batch_sha256']},'Actual respondent worker-start binding differs')
        backend=lookup_backend(ledger['call_id'])
        require(backend['call_id']==ledger['call_id'] and backend['status']=='SUCCESS' and backend['function_name']=='evaluate' and
            backend['module_name'].split('.')[-1]=='modal_exp11_evaluation','Respondent actual backend completion is unverified')
        paths=item_paths(root,item)
        require(json.loads(paths['input'].read_text())==payload and json.loads(paths['tokenizers'].read_text())==item['tokenizer_files'],
            'Respondent canonical worker input/tokenizer mirror differs')
        audit=audit_probe_evidence(root,payload,item['tokenizer_files'])
        require(audit['complete_data'] and all(audit[k] for k in ('H1','H2','H4','H5')),'Respondent group incomplete or failed hygiene')
        status=read(root,f'qualification_status/{experiment}.json')
        require(status['payload_sha256']==payload['payload_sha256'] and status['status']=='PROBES_COMPLETE_PENDING_GRADING' and
            status['stream_audit']=={k:v for k,v in audit.items() if k not in {'exact_prompt_replay','runtime_context_sha256'}},
            'Respondent durable status differs from independent raw replay')
    return audit


def qa_intake(prep,batch_paths,judge_paths,readiness):
    """Reconstruct every grader request from whole completed respondent groups.

    Invalid answers stay in the forty-scenario denominator as failed probes.
    No pass/fail labels or private joins enter the public grader request list.
    """
    from modal_exp11_evaluation import validate_batch,item_paths
    from structured_records_worker import validate_payload
    from structured_records_evidence import audit_complete_run
    from exp23_author_worker import read_chain
    from qa_semantic_protocol import grading_request
    from exp11_probe_worker import scored_probe_rows
    from exp11_admission import local_source_closure
    for name,h in local_source_closure(prep.root,('exp11_evidence','qa_semantic_protocol','structured_records_evidence')).items():
        require(prep.bind(name)==h,'QA intake implementation changed')
    root=prep.root;actions=[];respondents=[];expected={};invalid=[];pending=[];seen=set();audits=[]
    for batch_path in batch_paths:
        batch=read(root,batch_path);validate_batch(batch);prep.bind(batch_path)
        for item in batch['items']:
            value=item['payload'];key=(item['kind'],value['arm'],value.get('cot_mode','cot_on'),value.get('chunk_index',value.get('probe_kind')))
            require(key not in seen,'Duplicate action/respondent group selection');seen.add(key)
            require(value['readiness']==readiness,'QA source uses a different independently qualified Phase1 selection')
            paths=item_paths(root,item);descriptor={'payload_path':str(paths['input'].relative_to(root))}
            status_path=('evaluation_status/' if item['kind']=='action' else 'qualification_status/')+value['experiment']+'.json'
            if not inside(root,status_path).is_file():
                pending.append({'state':'WAITING_EXP11_COMPLETE_STREAM','kind':item['kind'],'experiment':value['experiment'],'resubmit':False});continue
            status=read(root,status_path)
            required_status='ACTION_CHUNK_COMPLETE' if item['kind']=='action' else 'PROBES_COMPLETE_PENDING_GRADING'
            require(status['status']==required_status,'Exp1.1 group failed; preserve all trials without selective retry')
            if item['kind']=='action':
                # Full action replay occurs as one complete 26-chunk aggregate below.
                require(read(root,descriptor['payload_path'])==value,'Canonical action worker payload changed')
                actions.append({'payload':value,**descriptor});continue
            audit=audit_respondent_item(prep,batch,item);audits.append({'experiment':value['experiment'],'audit':audit})
            descriptor['tokenizer_path']=str(paths['tokenizers'].relative_to(root))
            respondents.append({'payload':value,'tokenizer_files':item['tokenizer_files'],**descriptor})
            raw_path=f'raw_transcripts/{value["experiment"]}/{value["arm"]}.jsonl'
            with prep.capture_reads():rows,_=read_chain(inside(root,raw_path),value['payload_sha256'])
            full=rows[5:];require(len(full)==len(value['requests'])==40,'Require the entire fixed respondent stream')
            for probe,row in zip(value['requests'],full):
                require(row['trial_id']==probe['trial_id'],'Respondent ordering changed')
                request,private=grading_request(probe,row,row['source_manifest'])
                if request is None:
                    invalid.append({'source_trial_id':row['trial_id'],'valid_source_answer':False,'passed':False,
                                    'probe_kind':value['probe_kind'],'reason':'Invalid source answer; retained in full denominator.'})
                else:
                    require('private_join' not in request and request['trial_id'] not in expected,'Private/duplicate grader request')
                    expected[request['trial_id']]=request
    leased={};judges=[];judge_audits=[]
    for path in judge_paths:
        value=read(root,path);validate_payload(value);prep.bind(path)
        require(value['model_role']=='independent_qa_judge' and value['readiness']==readiness,'Judge is unrelated or bound to different Phase1 sources')
        for request in value['requests']:
            key=request['trial_id'];require(key in expected and request==expected[key],'Judge request was not reconstructed from its entire authenticated source group')
            require(key not in leased,'Duplicate independent judgment lease');leased[key]=request
        status_path=f'qualification_status/{value["experiment"]}.json'
        if not inside(root,status_path).is_file():
            pending.append({'state':'WAITING_QA_JUDGE_RECOVERY','payload_path':path,'experiment':value['experiment'],'resubmit':False});continue
        require(read(root,status_path)['status']=='RECORDS_COMPLETE','QA grader failed; no selective replacement')
        with prep.capture_reads():audit=audit_complete_run(root,value)
        require(audit['complete_data'] and all(audit[k] for k in ('H1','H2','H4','H5','all_records_valid')),'QA judge stream is not independently complete')
        judges.append({'payload':value,'payload_path':path});judge_audits.append(audit)
    nodes=[]
    for kind in ('consequence_awareness','rule_identification'):
        requests=[r for key,r in sorted(expected.items()) if r['probe_kind']==kind and key not in leased]
        for offset in range(0,len(requests),192):
            chunk=requests[offset:offset+192]
            nodes.append({'kind':'qa_grader','key':'qa:'+kind+':'+digest(chunk),'role':'independent_qa_judge',
                'task':kind,'mode':'cot_on','family':'llama','requests':chunk,'full_n':len(chunk),'fresh_pilot_n':5})
    complete_coverage=len(actions)==26 and len(respondents)==12
    complete_judges={r['trial_id'] for j in judges for r in j['payload']['requests']}
    ready=complete_coverage and not pending and set(expected)==complete_judges
    result={'state':'READY_FOR_COMPLETE_EXECUTION_AUDIT' if ready else 'WAITING_COMPLETE_ACTION_QA_AND_JUDGMENTS',
        'completed_action_chunks':len(actions),'completed_respondent_groups':len(respondents),
        'source_answer_n':40*len(respondents),'valid_source_answer_n':len(expected),'invalid_source_answer_n':len(invalid),
        'invalid_answers_retained_as_failed':invalid,'required_judgments':len(expected),'completed_judgments':len(complete_judges),
        'pending':pending,'respondent_audits':audits,'judge_audits':judge_audits,
        'all_valid_answers_judged_exactly_once':ready,'complete_experiment_receipt':False}
    if ready:
        # Use the existing public scorer, then require its exact projection in
        # the complete execution auditor. No caller-supplied labels are accepted.
        with prep.capture_reads():
            scores=[]
            for item in respondents:
                scores.extend(scored_probe_rows(root,item['payload'],[j['payload'] for j in judges if j['payload']['task']==item['payload']['probe_kind']]))
            from exp11_evidence import complete_evidence
            evidence=complete_evidence(root,[a['payload'] for a in actions],respondents,[j['payload'] for j in judges])
        require(sorted(scores,key=lambda r:(r['model_arm'],r['scenario_id'],r['probe_kind']))==
                sorted(evidence['probes'],key=lambda r:(r['model_arm'],r['scenario_id'],r['probe_kind'])),
                'Complete evidence/scored-probe reconstruction differs')
        selection={'schema':'exp11_execution_selection_v1','action_payload_paths':[a['payload_path'] for a in actions],
            'respondent_items':[{k:i[k] for k in ('payload_path','tokenizer_path')} for i in respondents],
            'judge_payload_paths':[j['payload_path'] for j in judges]}
        result['complete_selection_path']=prep.document('exp11_complete_selection',selection)
        result['complete_selection_sha256']=digest(selection)
        result['evidence_sha256']=digest(evidence)
        return nodes,result,evidence
    return nodes,result,None


STATISTICS_DRIVER = r'''"""Frozen CPU handoff; fresh raw replay before analysis, no GPU or phase admission."""
from pathlib import Path
import argparse, hashlib, json, os, signal, subprocess, sys

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    with path.open('x') as handle:json.dump(value,handle,sort_keys=True,allow_nan=False)
def inside(root,name):
    require(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts,'Escaping path')
    path=(root/name).resolve();path.relative_to(root);return path

def check(root,request):
    for name,expected in request['frozen_files'].items():
        require(sha(inside(root,name))==expected,'Changed frozen evidence/source: '+name)
    evidence=read(inside(root,request['evidence_path']))
    require(evidence['execution_evidence_complete'] is True and evidence['phase3_authorized'] is False and
        evidence['coverage']['action_episodes']==4520 and evidence['coverage']['respondent_answers']==480 and
        evidence['coverage']['action_chunks']==26 and evidence['coverage']['respondent_groups']==12 and
        len(evidence['episodes'])==4520 and len(evidence['probes'])==480,'Incomplete statistical denominator')
    return evidence

def bounded(command,root,env,log,timeout):
    with log.open('xb') as handle:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=timeout)
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait();raise
    require(code==0,'Analysis/audit child failed; inspect '+str(log))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    parser.add_argument('--operation',choices=('binary','omnibus'),required=True);parser.add_argument('--child',choices=('audit','analyze'))
    args=parser.parse_args();root=Path(args.root).resolve();request_path=inside(root,args.request);request=read(request_path)
    require(request_path.name=='exp11_statistics_request_'+sha(request_path)+'.json','Immutable handoff request bytes changed')
    require(request['schema']=='exp11_statistics_handoff_v1','Unexpected handoff schema')
    require(sha(Path(__file__))==request['driver_sha256'],'Driver binding differs')
    sys.path.insert(0,str(root/'scripts'));evidence=check(root,request);folder=inside(root,request['destinations'][args.operation])
    if args.child=='audit':
        require(Path(sys.executable).absolute()==(root/'.venv/bin/python').absolute(),'Source replay interpreter differs')
        from exp11_evidence import recompute
        require(recompute(root,request['selection_path'])==evidence,'Fresh complete source replay differs')
        check(root,request);write(folder/'fresh_evidence_audit.json',{'passed':True,'evidence_sha256':sha(inside(root,request['evidence_path']))});return
    if args.child=='analyze':
        require(read(folder/'fresh_evidence_audit.json')=={'passed':True,'evidence_sha256':sha(inside(root,request['evidence_path']))},'Missing fresh source audit')
        if args.operation=='binary':
            require(Path(sys.executable).absolute()==(root/'.venv/bin/python').absolute(),'Binary interpreter differs')
            from exp11_binary_analysis import analyze
            result=analyze(evidence['episodes'],evidence['probes']);check(root,request);write(folder/'component.json',result)
        else:
            require(Path(sys.executable).absolute()==(root/'.omnibus-venv/bin/python').absolute(),'Isolated omnibus interpreter required')
            require(os.sched_getaffinity(0)=={0,1},'Omnibus CPU affinity differs')
            from exp11_omnibus_subprocess import environment
            expected=environment()
            for key in request['isolated_environment_keys']:require(os.environ.get(key)==expected.get(key),'Environment differs: '+key)
            require('PYTHONPATH' not in os.environ and 'PYTHONHOME' not in os.environ,'Injected Python path')
            import exp11_omnibus_analysis as scientific
            scientific.environment()
            require(scientific.environment()==read(inside(root,request['environment_baseline']))['environment'],'Initialized environment differs')
            check(root,request);scientific.fit(evidence['episodes'],folder/'posterior');check(root,request)
        return
    require(not folder.exists(),'Existing invocation: no automatic overwrite or refit');folder.mkdir(parents=True)
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env['PYTHONNOUSERSITE']='1'
    base=[str(Path(__file__).resolve()),'--root',str(root),'--request',args.request,'--operation',args.operation]
    bounded([str(root/'.venv/bin/python'),*base,'--child','audit'],root,env,folder/'audit.log',900)
    if args.operation=='omnibus':
        from exp11_omnibus_subprocess import environment
        env=environment();command=['taskset','-c','0,1',str(root/'.omnibus-venv/bin/python'),*base,'--child','analyze']
    else:command=[str(root/'.venv/bin/python'),*base,'--child','analyze']
    bounded(command,root,env,folder/'analysis.log',3600 if args.operation=='omnibus' else 900);check(root,request)
    write(folder/'execution.json',{'completed':True,'operation':args.operation,'request_sha256':sha(request_path),
        'complete_experiment_receipt':False,'phase3_authorized':False,'actual_sampler_replay_still_required':args.operation=='omnibus'})

if __name__=='__main__':main()
'''


def statistics_handoff(prep,intake,evidence):
    """Freeze exact invocations only; never calculate estimates or run a sampler."""
    require(intake.get('all_valid_answers_judged_exactly_once') is True and
        evidence.get('execution_evidence_complete') is True,'Complete raw execution/judgment audit required before statistics')
    selection_path=intake['complete_selection_path']
    if not inside(prep.root,selection_path).is_file():
        return {'state':'WAITING_COMPLETE_SELECTION_LOCAL_FREEZE','launch_authorized':False,'statistics_executed':False}
    require(prep.bind(selection_path)==intake['complete_selection_sha256'],'Complete selection bytes changed')
    from exp11_evidence import recompute
    with prep.capture_reads():fresh=recompute(prep.root,selection_path)
    require({k:v for k,v in fresh.items() if k!='selection_file_sha256'}==evidence,'Public complete evidence hook differs from intake')
    for name,h in fresh['selection_file_sha256'].items():require(prep.bind(name)==h,'Complete selection manifest changed')
    require(fresh['coverage']['action_episodes']==4520 and fresh['coverage']['respondent_answers']==480 and
        fresh['coverage']['action_chunks']==26 and fresh['coverage']['respondent_groups']==12 and
        len(fresh['episodes'])==4520 and len(fresh['probes'])==480,'All fixed statistical denominators required')
    baseline='infrastructure/omnibus_validation_v1/initialized_environment_for_production.json'
    require(prep.bind(baseline)=='a75a3c7d0b4b2d74d4fd3c2dc7641dbb571a710fd7e5c641ff354fa2616ea8ab','Validated environment-only baseline differs')
    from exp11_admission import local_source_closure
    for name,h in local_source_closure(prep.root,('exp11_evidence','exp11_binary_analysis','exp11_omnibus_analysis','exp11_omnibus_subprocess')).items():
        require(prep.bind(name)==h,'Statistical implementation closure changed')
    for name in ('scripts/exp11_binary_analysis.py','scripts/exp11_omnibus_analysis.py','scripts/exp11_omnibus_subprocess.py',
                 'scripts/exp11_evidence.py','infrastructure/exp11_omnibus_plan_v1.json',
                 'infrastructure/exp11_binary_analysis_plan_v1.json',
                 'infrastructure/omnibus_validation_v1/requirements_lock_v2.txt'):
        prep.bind(name)
    evidence_path=prep.document('exp11_complete_evidence',fresh)
    driver_bytes=STATISTICS_DRIVER.encode();driver_sha=sha256(driver_bytes).hexdigest()
    driver_path=prep.file(prep.folder+'/statistics_driver_'+driver_sha+'.py',driver_bytes)
    if not inside(prep.root,evidence_path).exists() or not inside(prep.root,driver_path).exists():
        return {'state':'WAITING_STATISTICS_LOCAL_FREEZE','launch_authorized':False,'statistics_executed':False}
    prep.bind(evidence_path);prep.bind(driver_path)
    key=digest({'evidence':file_hash(inside(prep.root,evidence_path)),'sources':prep.source_hashes})
    request={'schema':'exp11_statistics_handoff_v1','selection_path':selection_path,'evidence_path':evidence_path,
        'environment_baseline':baseline,'driver_sha256':driver_sha,'frozen_files':dict(sorted(prep.source_hashes.items())),
        'destinations':{kind:prep.folder+'/statistics/'+key+'/'+kind for kind in ('binary','omnibus')},
        'isolated_environment_keys':['PYTHONHASHSEED','PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','MPLCONFIGDIR','NUMBA_CACHE_DIR','PYTENSOR_FLAGS','TMPDIR']}
    request_path=prep.document('exp11_statistics_request',request)
    return {'state':'READY_FOR_ROOT_CPU_STATISTICS','statistics_executed':False,'launch_authorized':False,
        'selection_path':selection_path,'evidence_path':evidence_path,'request_path':request_path,
        'invocations':[{'operation':kind,'command':[str(prep.root/'.venv/bin/python'),driver_path,'--root',str(prep.root),
            '--request',request_path,'--operation',kind],
            'fresh_complete_source_audit_timeout_seconds':900,'analysis_timeout_seconds':3600 if kind=='omnibus' else 900,
            'destination':request['destinations'][kind]} for kind in ('binary','omnibus')],
        'final_verification':{'binary':'exp11_binary_analysis.analyze(episodes, probes) == saved_component',
            'omnibus':'exp11_omnibus_subprocess.recompute(episodes, posterior, audit_directory=<fresh>, timeout_seconds=3600)',
            'actual_sampler_replay_required':True},'complete_experiment_receipt':False,'phase3_authorized':False}


def queue_intents(prep,prepared,resources):
    """Emit no launchable command except the one centralized config admission."""
    from launch_bounded_job import validate_resource_profile,verify_frozen_files
    central={row['job']:row for row in resources['jobs']};queue=[]
    available_gpus=resources['available_gpu_bound'];available_single=max(0,resources['limits']['single_gpu_workers']-resources['allocated_upper_bound']['single_gpu_workers'])
    dollars=resources['available_budget_usd']
    for job in prepared:
        config=read(prep.root,job['config_path'])
        require(config['job']==job['job'] and config['gpus']==2 and config['single_gpu_workers']==2 and config['tp8_workers']==0,
                'Prepared whole-app resource shape changed')
        validate_resource_profile(config,prep.root);verify_frozen_files(config,prep.root)
        row={'job':job['job'],'kind':job['node']['kind'],'input_path':job['input_path'],'resource_config_path':job['config_path'],
             'upper_bound_usd':config['upper_bound_usd'],'gpus':2,'launch_authorized':False}
        if job['job'] in central:
            existing=central[job['job']]
            require(existing['command']==config['command'] and existing['resource_profile']==config['resource_profile'] and
                    existing.get('frozen_files')==config['frozen_files'],'Central admitted job differs from immutable preparation')
            row.update(state='RECOVER_EXISTING_ADMISSION',app_id=existing.get('app_id'),resubmit=False)
        elif resources['unresolved_jobs']:row['state']='WAITING_CENTRAL_UNRESOLVED_ADMISSION'
        elif config['upper_bound_usd']>dollars:row['state']='WAITING_BUDGET_CAPACITY'
        elif available_gpus<2 or available_single<2:row['state']='WAITING_GPU_CAPACITY'
        else:
            dollars-=config['upper_bound_usd'];available_gpus-=2;available_single-=2
            row.update(state='READY_FOR_ROOT_CENTRAL_ADMISSION',
                central_command=[str(prep.root/'.venv/bin/python'),'scripts/launch_bounded_job.py','--config',job['config_path']],
                requires_live_cloud_recheck=True)
        queue.append(row)
    return queue


def inspect_program(root,selection,*,freeze_local=False,max_new_jobs=2):
    """One bounded state transition; no daemon, retry policy, or paid launch."""
    validate_selection(selection);require(type(max_new_jobs) is int and 0<=max_new_jobs<=2,'Prepare at most two new apps per transition')
    root=Path(root).resolve();prep=Preparation(root,selection,freeze_local)
    result={'schema':'phase2_program_state_v1','program_id':selection['program_id'],'selection_sha256':digest(selection),
        'state':'INITIALIZING','launch_authorized':False,'phase2_complete':False,'phase3_authorized':False,
        'paid_launch_authority':'root via scripts/launch_bounded_job.py only','pending':[],'queue':[],
        'proposed_files':prep.proposed,'scientific_filters_applied':False,
        'remaining_completion_contracts':['All required Exp2.3 conditions and full raw audits','All Exp1.1 action/QA/semantic-grading data',
            'Independent binary/clustered/omnibus statistics','Complete final-verifier registration; no current component grants Phase3']}
    def finish(state):
        prep.recheck();result['state']=state;result['bound_preparation_sources']=dict(sorted(prep.source_hashes.items()));return result
    try:
        from phase1_readiness import recompute
        from phase2_admission import external_scope,ordering_state,record_request_batches
        components=read(root,selection['components_path']);prep.bind(selection['components_path'])
        readiness=recompute(root,components)
        result['phase1_readiness']=readiness;result['external_scope']=external_scope(readiness)
        if readiness['primary_phase2_authorized'] is not True:
            return finish('BLOCKED_PHASE1_GATE' if readiness['status']=='FAIL_PRIMARY_PREREQUISITES' else 'WAITING_PHASE1_COMPLETE_EVIDENCE')
        for name in ('FINAL_RESEARCH_SPECIFICATION.md','scripts/admit_phase2_program.py'):
            prep.bind(name)
        identity={k:selection[k] for k in ('program_id','components_path','author_attempt','author_payload_path',
                                          'runtime_metadata_path','tokenizer_sources')}
        identity_path=prep.file(prep.folder+'/program_identity.json',canonical(identity))
        if inside(root,identity_path).exists():prep.bind(identity_path)
        # Qualified Role1 would require a separate, unimplemented external slice.
        if readiness['external_validity_authorized']:
            result['pending'].append({'state':'BLOCKED_QUALIFIED_EXTERNAL_SLICE_IMPLEMENTATION',
                'scope':'Primary work may prepare, but Phase2 cannot be declared complete without the qualified separate Role1 slice.'})
        resources=resource_snapshot(root);result['resource_snapshot']={k:v for k,v in resources.items() if k!='jobs'}
        prepared=registered_jobs(prep)
        require(not prepared or inside(root,identity_path).is_file(),'Prepared jobs have no immutable program identity')
        keys,producer_paths,executor_paths,exp11_batch_paths,qa_judge_paths=existing_node_keys(root,selection,prepared)
        runs=[{'kind':'author','payload_path':selection['author_payload_path']}]
        runs += [{'kind':'structured','payload_path':name} for name in producer_paths]
        records=[];adopted=[];leased={};waiting=[]
        for descriptor in runs:
            if not inside(root,descriptor['payload_path']).is_file():
                result['pending'].append({'state':'WAITING_PRODUCER_PAYLOAD_RECOVERY','payload_path':descriptor['payload_path'],'resubmit':False})
                return finish('WAITING_PRODUCER_RECOVERY')
            payload,rows,pending=producer_records(root,descriptor['payload_path'],descriptor['kind'],selection['author_attempt'])
            prep.bind(descriptor['payload_path'])
            if descriptor['kind']=='structured':
                for request in payload['requests']:
                    key=request['trial_id'];require(key not in leased,'Duplicate structured request lease across attempts');leased[key]=request
            if pending:
                waiting.append(pending)
                if descriptor['kind']=='author':result['pending']+=waiting;return finish('WAITING_AUTHOR_RECOVERY')
            else:records.extend(rows);adopted.append(descriptor)
        # Exact ordered full-stream membership, never selected valid records.
        require(len({r['trial_id'] for r in records})==len(records),'Duplicate full producer membership')
        bundle={'raw_artifacts':records};bundle_path=prep.document('dependency_bundle',bundle);runs_path=prep.document('dependency_runs',adopted)
        if not inside(root,bundle_path).exists() or not inside(root,runs_path).exists():
            result['pending'].append({'state':'WAITING_LOCAL_IMMUTABLE_FREEZE','bundle_path':bundle_path,'runs_path':runs_path,
                'next':'Run the same selection with --freeze-local; these are CPU preparation writes only.'})
            return finish('WAITING_LOCAL_FREEZE')
        from audit_exp23_launch import dependency_bundle
        design,dependency_audit=dependency_bundle(root,bundle_path,runs_path)
        # These exact raw/source prerequisites become central frozen inputs for
        # every newly prepared downstream job. Global control ledgers are instead
        # independently reconstructed by the runner at each actual admission.
        mutable={'infrastructure/resource_allocations.json','infrastructure/budget_reservations.json'}
        for name,h in dependency_audit['input_sources'].items():
            if name not in mutable:require(prep.bind(name)==h,'Audited producer source changed')
        result['dependency_summary']={'bundle_path':bundle_path,'runs_path':runs_path,'design_sha256':digest(design),
            'audit_sha256':digest(dependency_audit),'ready_conditions':design['ready_conditions'],
            'blocked_conditions':design['blocked_conditions'],'counts':design['actual_preparation_counts']}
        require(not design['semantic_rejections'],'Semantic stimulus rejection: preserve full attempted family; no selective replacement')
        require(not design['unreferenced_artifact_ids'],'Unreferenced producer data cannot be silently discarded')
        available={r['trial_id']:r for r in design['pending_requests']}
        for key,request in leased.items():
            require(key in available or key in design['validated_artifact_hashes'],'Producer lease is outside current exact design')
            if key in available:require(available[key]==request,'Previously leased request changed')
        pending_design=deepcopy(design);pending_design['pending_requests']=[r for r in design['pending_requests'] if r['trial_id'] not in leased]
        batches=record_request_batches(pending_design)
        require(not batches['pending_author_request_ids'],'Author produced incomplete fixed 48-scenario set')
        require(not batches['blocked_small_batches'],'Incomplete structured role/task batch; no invented padding')
        nodes=[]
        for batch in batches['batches']:
            nodes.append({'kind':'structured','key':batch['role']+':'+batch['task']+':'+batch['requests_sha256'],
                'role':batch['role'],'task':batch['task'],'mode':batch['cot_mode'],'requests':batch['requests'],
                'family':'primary' if batch['role'].endswith('_reviewer') else 'llama','full_n':len(batch['requests']),'fresh_pilot_n':5})
        result['pending']+=waiting
        ordering_path=selection['exp23_start_evidence']
        if not ordering_path and executor_paths:
            # The fixed first partition, never a successful outcome-selected run.
            first=next((p for p in executor_paths if read(root,p)['conditions']==executor_nodes()[0]['conditions'] and
                        read(root,p)['executor']=='primary_intact_executor' and read(root,p)['cot_mode']=='cot_on'),None)
            if first:ordering_path='group_calls/'+read(root,first)['experiment']+'/full.json'
        ordering=ordering_state(root,ordering_path,expected_readiness=readiness)
        result['exp11_ordering']=ordering
        if not design['pending_requests']:
            require(not design['blocked_conditions'],'Mandatory clause/control grid unsupported; no skipped conditions or invented source clauses')
            full_executor_nodes=[n for n in executor_nodes() if ('executor',n['key']) not in keys]
            if ordering['exp11_overlap_authorized']:
                full_exp11_nodes=[n for n in exp11_nodes() if ('exp11',n['key']) not in keys]
                # Alternate the independent studies after the real first-full-call gate.
                for index in range(max(len(full_executor_nodes),len(full_exp11_nodes))):
                    if index<len(full_executor_nodes):nodes.append(full_executor_nodes[index])
                    if index<len(full_exp11_nodes):nodes.append(full_exp11_nodes[index])
            else:nodes+=full_executor_nodes
        result['exp11_fixed_coverage']={'action_episodes':4520,'respondent_answers':480,'action_chunks':26,'respondent_groups':12,
            'fresh_action_and_respondent_pilots':190,'all_six_arms_required':True,'semantic_grader_pilots_and_complete_joins_still_required':True}
        environment=audit_environment(prep)
        if environment:
            result['pending'].append({'state':'WAITING_MATCHED_CPU_TOKENIZER_ENVIRONMENT','packages':environment,'automatic_install':False})
            return finish('WAITING_AUDIT_ENVIRONMENT')
        if exp11_batch_paths or qa_judge_paths:
            require(ordering['exp11_overlap_authorized'],'Existing Exp1.1 streams lack freshly audited real Exp2.3 ordering')
            qa_nodes,intake,evidence=qa_intake(prep,exp11_batch_paths,qa_judge_paths,readiness)
            result['exp11_qa_intake']=intake
            nodes=qa_nodes+nodes
            if evidence is not None:
                result['exp11_statistics_handoff']=statistics_handoff(prep,intake,evidence)
        result['planned_nodes']=[{k:v for k,v in n.items() if k not in {'requests','items'}} for n in nodes]
        tokenizers={family:tokenizer_metadata(prep,family) for family in ('primary','llama')}
        if not freeze_local:
            result['queue']=queue_intents(prep,prepared,resources)
            return finish('PREPARATION_PLAN_ONLY')
        unresolved=[j for j in prepared if j['job'] not in {r['job'] for r in resources['jobs']}]
        slots=max(0,max_new_jobs-len(unresolved))
        for node in nodes[:slots]:
            if node['kind']=='exp11' and selection['exp11_sources'] is None:
                result['pending'].append({'state':'WAITING_EXP11_BASES_AND_FULL_DESIGN','node':node['key']});continue
            manifest,_=prepare_node(prep,node,bundle_path=bundle_path,runs_path=runs_path,tokenizers=tokenizers,ordering_path=ordering_path)
            prepared.append(manifest)
        result['queue']=queue_intents(prep,prepared,resources)
        if any(r['state']=='READY_FOR_ROOT_CENTRAL_ADMISSION' for r in result['queue']):state='READY_FOR_ROOT_CENTRAL_ADMISSION'
        elif nodes:state='WAITING_RESOURCES_OR_EXISTING_ADMISSION'
        elif waiting:state='WAITING_PRODUCER_RECOVERY'
        else:state='WAITING_FULL_EXECUTION_QA_AND_STATISTICAL_AUDITS'
        return finish(state)
    except ContextPreflightError as exc:
        result['pending'].append({'state':'BLOCKED_CONTEXT_PREFLIGHT','report_path':exc.report_path,
            'minimum_initial_context_tokens':exc.report['minimum_initial_context_tokens'],
            'oversized_request_ids':[r['trial_id'] for r in exc.report['requests'] if not r['fits']]})
        return finish('BLOCKED_CONTEXT_PREFLIGHT')
    except FileNotFoundError as exc:
        result['pending'].append({'state':'WAITING_MISSING_LOCAL_EVIDENCE','path':str(exc.filename),'resubmit':False})
        return finish('WAITING_MISSING_LOCAL_EVIDENCE')
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        result['state']='BLOCKED_INDEPENDENT_AUDIT_OR_IMMUTABLE_INPUT';return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection');parser.add_argument('--print-selection-template',action='store_true')
    parser.add_argument('--freeze-local',action='store_true');parser.add_argument('--max-new-jobs',type=int,default=2)
    parser.add_argument('--timeout-seconds',type=int,default=600);parser.add_argument('--output')
    parser.add_argument('--inspect-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.print_selection_template:print(json.dumps(default_selection(),indent=2));return 0
    parser.error('--selection is required') if not args.selection else None
    require(30<=args.timeout_seconds<=900,'Controller audit deadline must be 30–900 seconds')
    selection=read(ROOT,args.selection);validate_selection(selection)
    if args.inspect_child:
        if args.freeze_local:
            with preparation_lock(ROOT,selection['program_id']),redirect_stdout(sys.stderr):
                result=inspect_program(ROOT,selection,freeze_local=True,max_new_jobs=args.max_new_jobs)
        else:
            with redirect_stdout(sys.stderr):result=inspect_program(ROOT,selection,max_new_jobs=args.max_new_jobs)
        print(json.dumps(result,allow_nan=False));return 0
    command=[sys.executable,str(Path(__file__).resolve()),'--selection',args.selection,'--inspect-child',
             '--max-new-jobs',str(args.max_new_jobs),'--timeout-seconds',str(args.timeout_seconds)]
    if args.freeze_local:command.append('--freeze-local')
    try:
        process=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=args.timeout_seconds)
        require(process.returncode==0,'Controller subprocess failed: '+process.stderr[-4000:])
        result=json.loads(process.stdout)
    except subprocess.TimeoutExpired:
        result={'schema':'phase2_program_state_v1','state':'WAITING_BOUNDED_AUDIT_TIMEOUT','launch_authorized':False,
            'phase2_complete':False,'phase3_authorized':False,'resubmit':False,
            'reason':'Audit deadline exceeded; immutable partial preparations are retained and must be recovered, never relaunched.'}
    if args.output:
        require(args.output.startswith('infrastructure/phase2_program/'+selection['program_id']+'/states/') and args.output.endswith('.json'),
                'State output must be a new JSON file under this program states directory')
        write_once(inside(ROOT,args.output),canonical(result))
    print(json.dumps(result,indent=2,allow_nan=False));return 0 if not result['state'].startswith('BLOCKED') else 2


if __name__=='__main__':raise SystemExit(main())
