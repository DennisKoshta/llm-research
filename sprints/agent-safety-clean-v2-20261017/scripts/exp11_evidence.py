"""Complete Exp1.1 execution evidence and an immutable QA-scored projection.

This adapter calculates no effect estimates, writes no receipts, and cannot
launch inference. The exact selected 26 action chunks, 12 respondent groups,
and all necessary independent judgments are recomputed from raw evidence.
"""
from copy import deepcopy
from pathlib import Path
import json

from exp11_action_worker import (ARMS, audit_complete_actions, audit_probe_evidence,
                                digest, hash_bytes, read_chain, require)
from exp11_probe_worker import KINDS, scored_probe_rows
from qa_semantic_protocol import grading_request


def join_projection(episodes, probes):
    """Pure join after audit; never modify the original raw episode dictionaries."""
    from exp11_binary_analysis import validate_complete
    validate_complete(episodes, probes)
    by_key={(p['scenario_id'],p['model_arm'],p['probe_kind']):p for p in probes}
    projected=[]
    for episode in episodes:
        consequence=by_key[episode['scenario_id'],episode['model_arm'],'consequence_awareness']
        rule=by_key[episode['scenario_id'],episode['model_arm'],'rule_identification']
        projected.append({**deepcopy(episode),'original_audited_episode_sha256':digest(episode),
            'passed_consequence_probe':consequence['passed'],'passed_rule_probe':rule['passed'],
            'consequence_probe_score_sha256':digest(consequence),'rule_probe_score_sha256':digest(rule),
            'qa_join_scope':'Same base scenario and checkpoint; cost-invariant probes, separate immutable scoring projection.'})
    return projected


def _sources(root,payloads):
    """Recheck immutable scientific sources; fail if selected attempts conflict."""
    root=Path(root);sources={}
    for value in payloads:
        for name,h in value['input_sources'].items():
            require(name not in sources or sources[name]==h,'Selected attempts bind conflicting source bytes')
            path=(root/name).resolve();path.relative_to(root.resolve())
            require(hash_bytes(path)==h,'Selected prerequisite source no longer matches frozen bytes')
            sources[name]=h
        for name,h in value['source_hashes'].items():
            require(Path(name).name==name and hash_bytes(root/'scripts'/name)==h,'Selected worker implementation changed')
    return dict(sorted(sources.items()))


def _worker_start(root,payload):
    ledger_path=Path(root)/'group_calls'/f'{payload["experiment"]}.json'
    start_path=Path(root)/'exp11_worker_starts'/f'{payload["experiment"]}.json'
    ledger=json.loads(ledger_path.read_text());started=json.loads(start_path.read_text())
    require(ledger['state']=='SPAWNED' and ledger['worker_module']=='modal_exp11_evaluation' and
            ledger['worker_function']=='evaluate','Unexpected execution call boundary')
    require(started=={'call_id':ledger['call_id'],'payload_sha256':payload['payload_sha256'],
                     'batch_sha256':ledger['batch_sha256']},'Actual worker-start/call binding differs')
    from exp11_admission import audit_worker_admission
    central=audit_worker_admission(root,payload)
    from audit_exp23_launch import lookup_backend
    actual=lookup_backend(ledger['call_id'])
    require(actual['call_id']==ledger['call_id'] and actual['function_name']=='evaluate' and
            actual['module_name'].split('.')[-1]=='modal_exp11_evaluation' and actual['status']=='SUCCESS',
            'Backend does not confirm the actual completed Exp1.1 worker')
    return {'experiment':payload['experiment'],'call_id':ledger['call_id'],'central_admission':central,
            'ledger_sha256':hash_bytes(ledger_path),'worker_start_sha256':hash_bytes(start_path)}


def complete_evidence(root,action_payloads,respondent_items,judge_payloads,*,tokenizers=None):
    """Return full execution evidence, never a completed scientific experiment."""
    root=Path(root);keys=[(i['payload']['arm'],i['payload']['probe_kind']) for i in respondent_items]
    require(len(keys)==12 and len(set(keys))==12 and set(keys)=={(a,k) for a in ARMS for k in KINDS},
            'Require all twelve fixed respondent groups exactly once')
    expected_actions={(a,m,i) for a in ARMS for m in ('cot_on','cot_off') if m=='cot_on' or a in ARMS[:2]
                      for i in range(4 if m=='cot_on' else 1)}
    action_keys=[(p['arm'],p['cot_mode'],p['chunk_index']) for p in action_payloads]
    require(len(action_keys)==26 and len(set(action_keys))==26 and set(action_keys)==expected_actions,'Require all 26 fixed action payloads exactly once')
    respondents=[i['payload'] for i in respondent_items];all_scientific=action_payloads+respondents
    for field in ('design_sha256','bases','readiness','ordering_audit'):
        require(len({digest(p[field]) for p in all_scientific})==1,'Action and QA scientific sources differ: '+field)
    identities={}
    for value in all_scientific:
        arm=value['arm'];require(arm not in identities or identities[arm]==value['identity'],'Action and QA checkpoints differ')
        identities[arm]=value['identity']
    tokenizer_hashes={digest(p['tokenizer_files']) for p in action_payloads}|{digest(i['tokenizer_files']) for i in respondent_items}
    require(len(tokenizer_hashes)==1,'Paired action/QA tokenizer bytes differ')
    require(len({p['experiment'] for p in judge_payloads})==len(judge_payloads),'Duplicate independent grader attempt')
    for value in judge_payloads:
        require(value['model_role']=='independent_qa_judge' and value['task'] in KINDS,'Unrelated judge records included')
        require(value['readiness']==all_scientific[0]['readiness'],'Independent grader Phase1 source selection differs')
    from phase1_readiness import recompute as phase1_recompute
    from phase2_admission import require_exp11_ordering
    baseline=all_scientific[0];ready=phase1_recompute(root,baseline['component_sources'])
    require(ready==baseline['readiness'] and ready['primary_phase2_authorized'],'Final evidence Phase1 admission no longer independently passes')
    require(require_exp11_ordering(root,baseline['ordering_evidence_path'],expected_readiness=ready)==baseline['ordering_audit'],
            'Final evidence actual Exp2.3 ordering no longer independently passes')
    sources=_sources(root,all_scientific+judge_payloads)
    from structured_records_evidence import audit_complete_run as audit_judge_run
    judge_audits=[{'experiment':p['experiment'],'audit':audit_judge_run(root,p)} for p in judge_payloads]
    starts=[_worker_start(root,p) for p in all_scientific]
    action=audit_complete_actions(root,action_payloads,tokenizers=tokenizers)
    probe_audits=[];expected_judgments={}
    for item in sorted(respondent_items,key=lambda i:(ARMS.index(i['payload']['arm']),i['payload']['probe_kind'])):
        value=item['payload'];audit=audit_probe_evidence(root,value,item['tokenizer_files'],
            tokenizer=(tokenizers or {}).get(value['experiment']))
        require(audit['complete_data'] and all(audit[k] for k in ('H1','H2','H4','H5')),'An independent respondent group failed hygiene')
        probe_audits.append({'experiment':value['experiment'],**audit})
        rows,_=read_chain(root/'raw_transcripts'/value['experiment']/f'{value["arm"]}.jsonl',value['payload_sha256'])
        for probe,raw in zip(value['requests'],rows[5:]):
            request,_=grading_request(probe,raw,raw['source_manifest'])
            if request is not None:
                require(request['trial_id'] not in expected_judgments,'Duplicate reconstructed public judgment request')
                expected_judgments[request['trial_id']]=request
    actual={}
    for value in judge_payloads:
        for request in value['requests']:
            require(request['trial_id'] not in actual,'Duplicate full judgment cannot be selected or averaged')
            actual[request['trial_id']]=request
    require(actual==expected_judgments,'Independent grading must cover exactly every valid source answer, with no fabricated or omitted judgment')
    probes=[]
    for item in sorted(respondent_items,key=lambda i:(ARMS.index(i['payload']['arm']),i['payload']['probe_kind'])):
        value=item['payload']
        probes.extend(scored_probe_rows(root,value,[p for p in judge_payloads if p['task']==value['probe_kind']]))
    projection=join_projection(action['episodes'],probes)
    result={'schema':'exp11_complete_execution_evidence_v1','episodes':projection,'probes':probes,
        'action_chunk_audits':action['chunk_audits'],'respondent_audits':probe_audits,'independent_grader_audits':judge_audits,
        'actual_worker_calls':starts,'source_sha256':sources,
        'payload_sha256':{p['experiment']:p['payload_sha256'] for p in all_scientific+judge_payloads},
        'coverage':{'action_episodes':4520,'respondent_answers':480,'full_independent_judgments':len(expected_judgments),
            'action_chunks':26,'respondent_groups':12,'fresh_action_and_respondent_pilots':190,
            'fresh_independent_judge_pilots':5*len(judge_payloads),'original_design_pilot_templates':70},
        'unjoined_episode_sha256':digest(action['episodes']),'scored_probe_sha256':digest(probes),
        'scored_episode_projection_sha256':digest(projection),'execution_evidence_complete':True,
        'raw_transcripts_modified':False,'complete_experiment_receipt':False,'phase3_authorized':False,
        'remaining':'Registered binary and omnibus statistics; complete Exp2.3 and global Phase3 gate audit.'}
    return result


def recompute(root,selection_path):
    """Final-verifier hook selecting exact immutable manifests, without discovery."""
    root=Path(root).resolve();paths={}
    def read(relative):
        require(isinstance(relative,str) and not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Selection path outside sprint')
        path=(root/relative).resolve();path.relative_to(root)
        raw=path.read_bytes();paths[relative]=hash_bytes(path);return json.loads(raw)
    selection=read(selection_path)
    require(set(selection)=={'schema','action_payload_paths','respondent_items','judge_payload_paths'} and
            selection['schema']=='exp11_execution_selection_v1','Unsupported complete evidence selection')
    action=[read(p) for p in selection['action_payload_paths']]
    respondents=[]
    for item in selection['respondent_items']:
        require(set(item)=={'payload_path','tokenizer_path'},'Respondent selection requires both immutable manifests')
        respondents.append({'payload':read(item['payload_path']),'tokenizer_files':read(item['tokenizer_path'])})
    judges=[read(p) for p in selection['judge_payload_paths']]
    result=complete_evidence(root,action,respondents,judges)
    result['selection_file_sha256']=dict(sorted(paths.items()))
    return result
