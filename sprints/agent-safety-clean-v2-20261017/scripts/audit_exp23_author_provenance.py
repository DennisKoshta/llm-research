"""Independent complete author provenance, with honest post-launch replay scope.

Never launches inference. The immutable historical worker did not attest prompt
IDs or installed package versions per generation. Reconstruction does not fill
that historical gap; every result preserves the limitation explicitly.
"""
from pathlib import Path
from datetime import datetime
import json,os,subprocess
from capture_exp23_author_provenance import require,read,relative,hash_bytes,canonical


def central_admission(root,payload_path,payload,capture):
    from historical_admission import validate_historical_admission
    from structured_records_evidence import option
    from exp23_author_worker import verify_runtime_sources
    root=Path(root).resolve();job_id='exp23_unified_author_'+payload['attempt']
    allocations=read(root/'infrastructure/resource_allocations.json');budget=read(root/'infrastructure/budget_reservations.json')
    jobs=[r for r in allocations['jobs'] if r['job']==job_id]
    reservations=[r for r in budget['reservations'] if r['job']==job_id]
    require(len(jobs)==len(reservations)==1,'Unique author central admission/reservation required')
    job=jobs[0];require(job.get('app_id')==capture['source_app_id'],'Actual author app ID must be durably reconciled')
    require(job['gpus']==8 and job['tp8_workers']==1 and job['single_gpu_workers']==0,'Wrong author topology')
    require(option(job['command'],'--attempt')==payload['attempt'],'Central author attempt differs')
    receipt_path=option(job['command'],'--external-receipt')
    require(hash_bytes(relative(root,receipt_path).read_bytes())==payload['external_mechanical_receipt_sha256'],'Author receipt CLI differs')
    profile=job['resource_profile']
    require(profile['source']=='scripts/modal_exp23_author.py' and set(profile['functions'])=={'author','coordinate'} and
        profile['excluded_functions']=={} and all(v['max_calls']==1 for v in profile['functions'].values()),'Wrong author resource/call profile')
    checked=validate_historical_admission(job,reservations[0],budget.get('reconciliations',[]),root)
    expected={payload_path:hash_bytes(relative(root,payload_path).read_bytes()),
        'scripts/modal_exp23_author.py':profile['source_sha256'],'FINAL_RESEARCH_SPECIFICATION.md':payload['specification_sha256'],
        'infrastructure/preflight.json':payload['preflight_sha256'],receipt_path:payload['external_mechanical_receipt_sha256']}
    expected.update({'scripts/'+k:v for k,v in payload['source_hashes'].items()})
    from audit_exp23_launch import audit_author_download_manifest
    expected.update(audit_author_download_manifest(root,payload))
    for name,h in expected.items():
        require(job['frozen_files'].get(name)==h and hash_bytes(relative(root,name).read_bytes())==h,'Author central source/payload binding differs: '+name)
    verify_runtime_sources(payload,root/'scripts',root,manifest_location='local')
    require(datetime.fromisoformat(capture['captured_utc'])>=datetime.fromisoformat(job['created_utc']),'Post-launch capture predates admission')
    return {'job':job_id,'app_id':job['app_id'],'profile_audit':checked,'source_sha256':expected}


def isolated_replay(root,payload_path,capture_path):
    """Run exact captured CPU package stack, returning fresh output without caching PASS."""
    # Keep the stdlib-only replay module in the static immutable source closure.
    from replay_exp23_author_tokenizer import replay as cpu_replay_contract
    require(callable(cpu_replay_contract),'Missing isolated replay implementation')
    root=Path(root).resolve();env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    code="import json,sys;sys.path.insert(0,sys.argv[1]);from replay_exp23_author_tokenizer import replay;print(json.dumps(replay(sys.argv[2],sys.argv[3],sys.argv[4])))"
    result=subprocess.run([str(root/'.author-audit-venv/bin/python'),'-c',code,str(root/'scripts'),str(root),payload_path,capture_path],
        env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180,check=True)
    return json.loads(result.stdout)


def recompute(root,payload_path,capture_path,*,lookup_call=None,replay_runner=None):
    from exp23_author_worker import validate_payload,audit_streams,read_chain
    from audit_exp23_launch import lookup_backend
    root=Path(root).resolve();payload=read(relative(root,payload_path));validate_payload(payload)
    require(payload_path=='stimuli/'+payload['experiment']+'_input.json','Canonical author payload required')
    capture=read(relative(root,capture_path))
    require(capture['capture_timing']=='POST_LAUNCH' and capture['runtime_source_sha256']==payload['source_hashes'],'Capture identity differs')
    require(all(capture[k]==payload[k] for k in ('model_id','revision','model_path')),'Captured author checkpoint identity differs')
    admission=central_admission(root,payload_path,payload,capture)
    ledger_path='group_calls/'+payload['experiment']+'.json';ledger=read(relative(root,ledger_path))
    require(ledger['schema']=='exp23_author_call_ledger_v1' and ledger['state']=='SPAWNED' and
        ledger['attempt']==payload['attempt'] and ledger['payload_sha256']==payload['payload_sha256'],'Actual author call ledger differs')
    require(datetime.fromisoformat(ledger['created_at_utc'])<=datetime.fromisoformat(ledger['submitted_at_utc'])<=
        datetime.fromisoformat(capture['captured_utc']),'Author call/capture chronology differs')
    backend=(lookup_call or lookup_backend)(ledger['call_id'])
    require(backend['call_id']==ledger['call_id'] and backend['function_name']=='author' and
        backend['module_name'].split('.')[-1]=='modal_exp23_author' and backend['status']=='SUCCESS','Actual author GPU call not successfully complete')
    status_path='qualification_status/'+payload['experiment']+'_flash_next_uncensored.json';status=read(relative(root,status_path))
    require(status['status']=='AUTHORING_COMPLETE_PENDING_SEMANTIC_REVIEW' and status['payload_sha256']==payload['payload_sha256'],'Author completion unavailable')
    folder='raw_transcripts/'+payload['experiment'];raw_path=folder+'/flash_next_uncensored.jsonl';journal_path=folder+'/flash_next_uncensored_requests.jsonl'
    evidence_names=(ledger_path,status_path,raw_path,journal_path,'author_artifacts/'+payload['experiment']+'/model_returns.jsonl',
                    folder+'/frozen_input.json',folder+'/attempt_claim.json')
    evidence_before={name:hash_bytes(relative(root,name).read_bytes()) for name in evidence_names}
    audit=audit_streams(root,payload)
    require(audit['pilot_n']==5 and audit['pilot_pass'] and audit['authoring_H1_pass'] and audit['all_author_contracts_valid'] and
        not audit['pending_writeahead_trial_ids'],'Author raw hygiene incomplete or failed')
    replay=(replay_runner or isolated_replay)(root,payload_path,capture_path)
    require(replay['schema']=='exp23_author_postlaunch_prompt_replay_v1' and replay['payload_sha256']==payload['payload_sha256'] and
        replay['generation_time_token_ids_attested'] is False and replay['generation_time_package_versions_attested'] is False,'Invalid replay provenance scope')
    folder='raw_transcripts/'+payload['experiment'];raw_path=folder+'/flash_next_uncensored.jsonl';journal_path=folder+'/flash_next_uncensored_requests.jsonl'
    raw,_=read_chain(relative(root,raw_path),payload['payload_sha256']);journal,_=read_chain(relative(root,journal_path),payload['payload_sha256'])
    rows={r['trial_id']:r for r in raw};written={r['trial_id']:r for r in journal};expected={r['trial_id'] for r in payload['pilots']+payload['requests']}
    require(len(raw)==len(journal)==len(replay['requests'])==53 and set(rows)==set(written)=={r['trial_id'] for r in replay['requests']}==expected,'Full53-request replay membership differs')
    for row in replay['requests']:
        key=row['trial_id'];require(rows[key]['full_prompt']==written[key]['full_prompt']==row['full_prompt'] and
            written[key]['prompt_tokens']==row['prompt_tokens'],'Independent author prompt/count replay differs')
    sources={**admission['source_sha256'],**replay['source_sha256']}
    sources.update(evidence_before)
    for name in ('capture_exp23_author_provenance.py','replay_exp23_author_tokenizer.py','audit_exp23_author_provenance.py'):
        sources['scripts/'+name]=hash_bytes((root/'scripts'/name).read_bytes())
    for name,h in sources.items():require(hash_bytes(relative(root,name).read_bytes())==h,'Author provenance input changed during audit')
    return {'schema':'exp23_author_complete_postlaunch_provenance_audit_v1','complete_data':True,'author_hygiene':audit,
        'central_admission':admission,'backend_call':backend,'postlaunch_prompt_replay_verified':True,'request_count':53,
        'capture_path':capture_path,'capture_timing':'POST_LAUNCH','generation_time_token_ids_attested':False,
        'generation_time_package_versions_attested':False,'source_sha256':dict(sorted(sources.items())),
        'limitations':capture['limitations'],'phase2_execution_authorized':False}
