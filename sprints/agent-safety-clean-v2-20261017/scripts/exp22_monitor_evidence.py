"""Independent Monitor5 source, actual-call, central and temporal H5 evidence.

Offline sweeps do not substitute for the separately required live monitor data.
The producer bridge is deliberately mandatory and presently not implemented.
"""
from datetime import datetime
from pathlib import Path
import json
import re

from exp22_monitor_worker import (safe_path,validate_payload,hash_bytes,digest,require,paths,audit_stage)


def dependency_bundle(root,bundle_path,producer_selection_path,phase3_inputs,phase3_audit):
    """Require actual Phase3 author/reviewer provenance, never saved PASS flags.

    Future exp22_producer_evidence.recompute must return the exact complete raw
    artifact population, fresh global gate, and all immutable upstream sources.
    Current Exp2.3-only structured runners cannot supply this evidence.
    """
    try:from exp22_producer_evidence import recompute as audit_producers
    except ModuleNotFoundError as exc:
        if exc.name!='exp22_producer_evidence':raise
        raise RuntimeError('BLOCKED_EXP22_PRODUCER_PROVENANCE: Phase3-aware unified author, semantic/clause judge runners and independent full-call auditor are required') from exc
    root=Path(root).resolve();bundle_file=safe_path(root,bundle_path);selection=safe_path(root,producer_selection_path)
    source=audit_producers(root,producer_selection_path,phase3_inputs=phase3_inputs)
    require(source['schema']=='exp22_producer_execution_evidence_v1' and source['all_streams_passed'] is True and
        source['phase3_audit']==phase3_audit,'Independent source production/global gate differs')
    bundle=json.loads(bundle_file.read_text())
    require(bundle=={'raw_artifacts':source['raw_artifacts']},'Full producer population differs; no selected record subset')
    from exp22_design import build_design
    design=build_design(bundle,bundle_path,hash_bytes(bundle_file))
    require(design['complete_core_design'] and not design['pending_requests'] and not design['semantic_rejections'] and
        not design['unreferenced_artifact_ids'],'Unresolved/rejected source family blocks monitor execution')
    sources={**source['source_sha256'],bundle_path:hash_bytes(bundle_file),producer_selection_path:hash_bytes(selection)}
    for name,h in sources.items():require(hash_bytes(safe_path(root,name))==h,'Producer source changed')
    return design,{'schema':'exp22_monitor_dependencies_v1','bundle_path':bundle_path,'bundle_sha256':hash_bytes(bundle_file),
        'producer_selection_path':producer_selection_path,'producer_audit_sha256':digest(source),
        'design_sha256':digest(design),'phase3_admission_sha256':digest(phase3_audit),'source_sha256':dict(sorted(sources.items())),
        'full_clause_factorial_supported':design['full_clause_factorial_supported'],
        'missing_claim_policy':'Use the frozen design support mask; never invent claims or fill after monitor outcomes.'}


def source_bindings(root,payload):
    result=dict(payload['input_sources'])
    for name,h in payload['source_hashes'].items():
        key='scripts/'+name
        require(key not in result or result[key]==h,'Conflicting runtime/input source binding')
        result[key]=h
    for name in ('scripts/modal_exp22_monitor.py','scripts/exp22_monitor_evidence.py','scripts/historical_admission.py'):
        actual=hash_bytes(safe_path(root,name));require(name not in result or result[name]==actual,'Source binding differs');result[name]=actual
    for name,h in result.items():require(hash_bytes(safe_path(root,name))==h,'Monitor prerequisite/source changed: '+name)
    return dict(sorted(result.items()))


def audit_central_admission(root,payload,input_path,*,require_app_id=False):
    from exp11_admission import option
    from historical_admission import validate_historical_admission
    validate_payload(payload);root=Path(root).resolve()
    require(input_path==f'stimuli/{payload["experiment"]}_input.json','Canonical immutable monitor input required')
    require(json.loads(safe_path(root,input_path).read_text())==payload,'Central input differs')
    rows=json.loads((root/'infrastructure/resource_allocations.json').read_text())['jobs']
    selected=[r for r in rows if r['job']==payload['central_job']];require(len(selected)==1,'Unique root central admission required');job=selected[0]
    require(not require_app_id or bool(re.fullmatch('ap-[A-Za-z0-9_-]+',job.get('app_id') or '')),'Actual central app ID missing')
    require(job['gpus']==2 and job['single_gpu_workers']==2 and job['tp8_workers']==0,'Unexpected monitor app resource shape')
    command=job['command'];require(option(command,'--attempt')==payload['attempt'] and option(command,'--input-path')==input_path and
        command.count('--launch')==1 and not any(c.startswith('--launch=') for c in command),'Exact central CLI binding differs')
    profile=job['resource_profile'];require(profile['source']=='scripts/modal_exp22_monitor.py' and
        set(profile['functions'])=={'monitor','coordinate'} and profile['excluded_functions']=={} and
        profile['functions']['monitor']['max_calls']==2 and profile['functions']['coordinate']['max_calls']==1,
        'Exactly pilot/full GPU calls and one coordinator must be bounded')
    budget=json.loads((root/'infrastructure/budget_reservations.json').read_text())
    selected=[r for r in budget['reservations'] if r['job']==job['job']];require(len(selected)==1,'Unique monitor budget reservation required')
    checked=validate_historical_admission(job,selected[0],budget.get('reconciliations',[]),root)
    if not require_app_id:require(selected[0]['upper_bound_usd']==job['admitted_upper_bound_usd'],'Reduced historical budget cannot authorize fresh launch')
    sources=source_bindings(root,payload);sources[input_path]=hash_bytes(safe_path(root,input_path))
    for name,h in sources.items():require(job.get('frozen_files',{}).get(name)==h,'Central admission omitted exact source/input: '+name)
    return {'schema':'exp22_monitor_central_admission_v1','job':job['job'],'app_id':job.get('app_id'),
        'payload_sha256':payload['payload_sha256'],'input_path':input_path,'input_sha256':sources[input_path],
        'source_sha256':sources,'profile_sha256':digest(profile),'resource_profile_audit':checked,
        'maximum_gpu_calls':2,'maximum_coordinator_calls':1,'actual_spend_claimed':False}


def ledger_path(root,payload,stage):
    require(stage in ('pilot','full'),'Unknown monitor stage')
    return Path(root)/'group_calls'/payload['experiment']/(stage+'.json')


def durable_call(root,payload,stage):
    row=json.loads(ledger_path(root,payload,stage).read_text())
    require(row['schema']=='exp22_monitor_call_v1' and row['state']=='SPAWNED' and
        bool(re.fullmatch('fc-[A-Za-z0-9_-]+',row.get('call_id') or '')),'Committed concrete monitor call required')
    require(row['payload_sha256']==payload['payload_sha256'] and row['stage']==stage and row['job']==payload['central_job'] and
        row['payload_path']==f'monitor_inputs/{payload["experiment"]}.json' and
        row['worker_module']=='modal_exp22_monitor' and row['worker_function']=='monitor' and row['gpu']=='H200:1',
        'Durable monitor call is unrelated to exact payload/stage')
    require(json.loads(safe_path(root,row['payload_path']).read_text())==payload,'Worker input mirror differs')
    return row


def audit_pilot_completion(root,payload,*,tokenizer=None):
    audit=audit_stage(root,payload,'pilot',tokenizer=tokenizer);p=paths(root,payload,'pilot')
    require(audit['trials']==5 and audit['H5'],'Five independent raw pilots must pass before full call')
    status=json.loads(p['status'].read_text())
    require(status['status']=='PILOT_PASS' and status['stage']=='pilot' and status['payload_sha256']==payload['payload_sha256'] and
        status['audit']==audit,'Durable pilot completion differs from raw replay')
    from exp23_author_worker import read_chain
    raw,_=read_chain(p['raw'],payload['payload_sha256']);text=p['pilot_audit'].read_text()
    require(re.findall(r'^Status: (PASS|FAIL)$',text,re.M)==['PASS'] and
        [json.loads(t) for t in re.findall(r'^```json\n(.*?)\n```$',text,re.M|re.S)]==raw,'Pilot Markdown evidence differs')
    return audit,status


def authorize_payload(root,payload):
    """Local admission always replays the global gate and exact producer evidence."""
    validate_payload(payload);source_bindings(root,payload)
    from phase3_readiness import recompute
    gate=recompute(root,payload['phase3_inputs'])
    require(gate==payload['phase3_audit'] and gate['phase3_authorized'] is True,'Fresh global Phase3 recomputation differs or fails')
    dep=payload['dependency_audit'];design,audit=dependency_bundle(root,dep['bundle_path'],dep['producer_selection_path'],payload['phase3_inputs'],gate)
    require(design==payload['design'] and audit==dep,'Fresh producer/design audit differs')
    source_bindings(root,payload)
    return gate


def audit_complete_run(root,payload,*,tokenizer=None,lookup_call=None):
    """Actual completed offline sweep; no full Exp2.2 or Phase3 completion claim."""
    from audit_exp23_launch import lookup_backend
    root=Path(root).resolve();authorize_payload(root,payload)
    central=audit_central_admission(root,payload,f'stimuli/{payload["experiment"]}_input.json',require_app_id=True)
    pilot,status=audit_pilot_completion(root,payload,tokenizer=tokenizer)
    full=audit_stage(root,payload,'full',tokenizer=tokenizer)
    full_status=json.loads(paths(root,payload,'full')['status'].read_text())
    require(full_status['status']==('FULL_COMPLETE' if full['H1'] else 'FULL_H1_FAILED') and full_status['audit']==full and
        full_status['payload_sha256']==payload['payload_sha256'] and full_status['stage']=='full','Full completion differs from replay')
    calls={s:durable_call(root,payload,s) for s in ('pilot','full')}
    require(calls['pilot']['call_id']!=calls['full']['call_id'],'Pilot and full must be separate GPU calls')
    require(calls['full']['pilot_call_id']==calls['pilot']['call_id'] and calls['full']['pilot_audit_sha256']==digest(pilot) and
        calls['full']['pilot_completed_utc']==status['finished_utc'],'Full launch not bound to exact independently completed pilot')
    def timestamp(text):
        stamp=datetime.fromisoformat(text);require(stamp.tzinfo is not None,'Missing timezone');return stamp
    require(timestamp(calls['pilot']['spawn_intent_utc'])<=timestamp(calls['pilot']['submitted_utc'])<=timestamp(status['finished_utc'])<=
        timestamp(calls['full']['spawn_intent_utc'])<=timestamp(calls['full']['submitted_utc'])<=timestamp(full_status['finished_utc']),
        'Full-call admission did not follow completed pilot')
    backend={}
    for stage,call in calls.items():
        start=json.loads((ledger_path(root,payload,stage).parent/(stage+'_worker_started.json')).read_text())
        require(start=={'call_id':call['call_id'],'payload_sha256':payload['payload_sha256'],'stage':stage},'Actual worker-start binding differs')
        actual=(lookup_call or lookup_backend)(call['call_id'])
        require(actual['call_id']==call['call_id'] and actual['status']=='SUCCESS' and actual['function_name']=='monitor' and
            actual['module_name'].split('.')[-1]=='modal_exp22_monitor','Actual backend completed monitor call unverified')
        backend[stage]=actual
    source_bindings(root,payload)
    return {'schema':'exp22_monitor_complete_sweep_evidence_v1','payload_sha256':payload['payload_sha256'],
        'sweep_id':payload['sweep']['sweep_id'],'pilot':pilot,'full':full,'backend_calls':backend,'central_admission':central,
        'complete_data':True,'hygiene_passed':all(full[k] for k in ('H1','H2','H3','H4')) and pilot['H5'],
        'H5_before_full_call_verified':True,'execution_provenance_verified':True,
        'full_clause_factorial_supported':payload['design']['full_clause_factorial_supported'],
        'live_monitor_stream_verified':False,'complete_experiment_receipt':False,'phase3_authorized':False}
