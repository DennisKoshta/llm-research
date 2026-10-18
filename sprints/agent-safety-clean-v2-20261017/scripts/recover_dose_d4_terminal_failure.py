"""Recover and independently audit the stopped D4 hygiene failure, never complete it."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
ATTEMPT='author_v1_d4_cap8192_format_v1'
APP='ap-VC17PiSHqn38pWtc4RQuA4'
STEM='phase1_dose_response_'+ATTEMPT
ARMS=('dose_m0_5','dose_0_0','dose_p0_5','dose_1_0','dose_p1_5')
BASE='infrastructure/dose_d4_terminal_failure_provenance'
EXEC=f'executions/phase1_dose_supervisor_{ATTEMPT}/evaluation'


def digest(data):return hashlib.sha256(data).hexdigest()
def file_hash(path):return digest(Path(path).read_bytes())
def read_json(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def immutable(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==data, f'Existing evidence differs; no overwrite: {path}'
    else:
        with path.open('xb') as handle:handle.write(data)


def observe_stopped(root):
    data=subprocess.check_output([str(root/'.venv/bin/modal'),'app','info',APP,'--json'],
                                 cwd=root,text=True,timeout=60)
    value=json.loads(data)
    assert value['app_id']==APP and value['description']==NS+'-dose-evaluation-d4'
    assert value['lifecycle']['state']=='stopped' and value['lifecycle']['stopped_at'], 'D4 app is not verified stopped'
    return value


def inventory(volume):
    import modal
    from modal.volume import FileEntryType
    scopes={f'raw_transcripts/{STEM}':None,f'trajectories/{STEM}':None,
        f'group_calls/{STEM}':None,'evaluation_status':STEM+'_','pilots':STEM+'_','sandbox_inputs':STEM+'_'}
    result={}
    for folder,prefix in scopes.items():
        try:entries=list(volume.iterdir(NS+'/'+folder,recursive=True))
        except (modal.exception.NotFoundError,FileNotFoundError):continue
        for entry in entries:
            remote=PurePosixPath(str(entry.path).lstrip('/'))
            assert remote.parts[0]==NS and '..' not in remote.parts
            name=str(PurePosixPath(*remote.parts[1:]))
            if prefix and not PurePosixPath(name).name.startswith(prefix):continue
            if entry.type==FileEntryType.DIRECTORY:continue
            assert entry.type==FileEntryType.FILE and PurePosixPath(name).suffix in {'.json','.jsonl','.md'}
            assert name.startswith(folder+'/')
            result[name]={'remote_path':str(remote),'size_bytes':entry.size,'mtime':str(entry.mtime)}
    return dict(sorted(result.items()))


def recover(root):
    import modal
    before=observe_stopped(root)
    volume=modal.Volume.from_name('curiosity-study-artifacts')
    listed=inventory(volume);assert listed, 'No D4 artifacts found'
    recovered={}
    for name,entry in listed.items():
        data=b''.join(volume.read_file(entry['remote_path']))
        assert len(data)==entry['size_bytes'],'Terminal volume listing/read size differs'
        path=root/name
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        existed=path.exists();immutable(path,data)
        recovered[name]={**entry,'sha256':digest(data),'local_disposition':'EXACT_MATCH_PRESERVED' if existed else 'RECOVERED_NEW'}
    assert inventory(volume)==listed,'Terminal namespace changed during recovery'
    after=observe_stopped(root)
    for name,entry in recovered.items():
        assert file_hash(root/name)==entry['sha256']
    return {'app_stopped_before':before,'app_stopped_after':after,'recovery_files':recovered}


def audit(root,recovery):
    from strict_parser import parse_output,validate_decoding
    from raw_generation_evidence import audit_request_journal,requires_return_evidence
    from sandbox_runtime import Sandbox
    from audit_dose_response_d4 import prompt_renderer,token_id_renderer
    from modal_dose_evaluation_d4 import arm_payload,require_evaluation_authorization
    from prepare_dose_evaluation_d4 import digest as config_digest
    root=Path(root)
    source=read_json(root/EXEC/'source_manifest.json')
    cache=read_json(root/EXEC/'authorization_cache.json')
    frozen=cache['read_prerequisite_sha256']
    assert len(frozen)==149, 'Expected the exact admitted149-input closure'
    assert all(file_hash(root/name)==value for name,value in source.items())
    assert all(file_hash(root/name)==value for name,value in frozen.items())
    frozen_group=root/f'group_calls/{STEM}/frozen_input.json'
    value=read_json(frozen_group);config=value['config'];authorization=value['authorization']
    assert value['payload_sha256']==config_digest([config,ATTEMPT,authorization])
    assert config_digest(config)==authorization['config_sha256']
    assert authorization==cache['result']['authorization']
    require_evaluation_authorization(authorization,ATTEMPT)
    assert config['format_correction']['no_further_wording_search'] is True
    expected={row['trial_id']:(split,row) for split in ('pilot','evaluation')
              for row in config['pilots' if split=='pilot' else 'trials']}
    assert len(expected)==53
    bound=set(source)|set(frozen)|set(recovery['recovery_files'])|{
        EXEC+'/source_manifest.json',EXEC+'/authorization_cache.json',
        'infrastructure/dose_d4_first_pilot_failure.json','infrastructure/dose_d4_h5_stop_decision.json',
        'infrastructure/dose_d4_stop_action_result.json',
        'scripts/recover_dose_d4_terminal_failure.py'}
    before={name:file_hash(root/name) for name in bound}
    results={};render=encode=None;tokenizers=None
    for arm in ARMS:
        payload_path=root/f'sandbox_inputs/{STEM}_{arm}_cot_on.json'
        raw_path=root/f'raw_transcripts/{STEM}/{arm}_cot_on.jsonl'
        request_path=raw_path.with_name(arm+'_cot_on_requests.jsonl')
        return_path=raw_path.with_name(arm+'_cot_on_returns.jsonl')
        trajectory_path=root/f'trajectories/{STEM}/{arm}_cot_on.jsonl'
        status_path=root/f'evaluation_status/{STEM}_{arm}_cot_on.json'
        pilot_path=root/f'pilots/{STEM}_{arm}_cot_on_pilot_audit.md'
        raw=rows(raw_path);requests=rows(request_path);saved=rows(trajectory_path)
        assert not raw or requests
        journal=audit_request_journal(request_path,root) if request_path.exists() else {
            'requested':0,'returned':0,'pending_trial_ids':[]}
        if payload_path.exists():
            payload=read_json(payload_path)
            model=next(item for item in config['arms'] if item['arm']==arm)
            hashes=payload['tokenizer_file_sha256']
            assert payload=={**arm_payload(config,model,ATTEMPT,authorization),'tokenizer_file_sha256':hashes}
            if tokenizers is None:
                tokenizers=hashes
                render,prompt_paths=prompt_renderer(root,hashes)
                encode,token_paths=token_id_renderer(root,hashes)
                for path in prompt_paths+token_paths:
                    name=str(path.relative_to(root));bound.add(name);before[name]=file_hash(path)
            assert hashes==tokenizers
        else:
            assert not raw and not requests and not saved
        raw_by_id={row['trial_id']:row for row in raw};assert len(raw_by_id)==len(raw)
        states={};histories={};turn_ids=defaultdict(list);split_by_id={};pilot_valid={}
        parser_counts=Counter();capped=unclosed=0;failures=[]
        for request in requests:
            episode=request['episode_id'];split,spec=expected[episode]
            assert request['split']==split and request['model_arm']==arm and request['cot_mode']=='cot_on'
            for key,item in spec.items():
                if key!='trial_id':assert request[key]==item
            assert request['max_new_tokens']==8192 and request['temperature']==0 and request['forced_prefix'] is None
            assert (request['context_limit'],request['max_num_seqs'],request['max_num_batched_tokens'])==(65536,4,8192)
            if episode not in states:
                states[episode]=Sandbox(spec);histories[episode]=[dict(message) for message in spec['messages']]
                split_by_id[episode]=split;pilot_valid[episode]=True
            assert not states[episode].completed
            turn=len(turn_ids[episode])+1
            assert request['turn_index']==turn and request['trial_id']==episode+f':turn{turn}'
            assert request['full_prompt']==render(histories[episode]),'Exact prompt/history replay differs'
            ids=encode(request['full_prompt'])
            assert request['prompt_token_ids']==ids and len(ids)+8192<=65536
            row=raw_by_id.get(request['trial_id'])
            if row is None:continue
            for key,item in spec.items():
                if key!='trial_id':assert row[key]==item
            assert row['checkpoint_path']==payload['model_path']
            assert row['prompt_token_ids']==ids and row['turn_index']==turn
            assert row['generated_tokens']==len(row['generated_token_ids'])
            assert all(type(token) is int and token>=0 for token in row['generated_token_ids'])
            validate_decoding(row['cot_mode'],row['max_new_tokens'],row['temperature'],row['forced_prefix'])
            parsed=parse_output(row['raw_output'],row['cot_mode'],row['declared_names'])
            for key,item in parsed.items():assert row[key]==item,'Strict parser differs'
            parser_counts[parsed['parse_status']]+=1
            capped_here=row['generated_tokens']>=8192 or row['finish_reason']=='length'
            capped+=capped_here;unclosed+=not parsed['think_closed']
            valid=parsed['parse_status']=='VALID' and not capped_here and parsed['think_closed']
            pilot_valid[episode]&=valid
            observation=states[episode].step(parsed,capped_here)
            if turn==10 and not states[episode].completed:observation=states[episode].exhaust()
            assert row['sandbox_observation']==observation
            assert row['episode_terminal']==states[episode].completed
            assert row['outcome_category']==(states[episode].outcome or ('SAFE_ROUTE' if observation['status']=='OK' else 'TRUNCATED_OR_MALFORMED'))
            turn_ids[episode].append(row['trial_id'])
            histories[episode].extend([{'role':'assistant','content':row['raw_output']},
                {'role':'user','content':'Sandbox observation: '+json.dumps(observation)}])
            if not valid:
                failures.append({key:row[key] for key in ('trial_id','episode_id','split','generated_tokens',
                    'max_new_tokens','finish_reason','think_closed','parse_status')})
        derived={}
        for episode,state in states.items():
            if not state.completed:continue
            split,spec=expected[episode]
            derived[episode]={key:spec[key] for key in ('scenario_id','template_id','cost_regime','consequence','boundary','control','domain')}
            derived[episode].update(episode_id=episode,model_arm=arm,cot_mode='cot_on',split=split,
                raw_turn_ids=turn_ids[episode],outcome_category=state.outcome,turn_count=len(turn_ids[episode]))
        assert len({row['episode_id'] for row in saved})==len(saved)
        for row in saved:assert row==derived[row['episode_id']],'Saved trajectory differs'
        complete_pilots=[key for key in derived if split_by_id[key]=='pilot']
        h5=(all(pilot_valid[key] for key in complete_pilots) if len(complete_pilots)==5 else None)
        evaluation_turns=sum(row['split']=='evaluation' for row in raw)
        if evaluation_turns:assert h5 is True,'Full evaluation launched despite failed/incomplete pilots'
        pilot_rows=[row for row in raw if row['split']=='pilot']
        if pilot_path.exists():
            assert h5 is not None
            expected_markdown='# Multi-step sandbox pilot\n\nStatus: '+('PASS' if h5 else 'FAIL')+'\n\n'+'\n\n'.join('```json\n'+json.dumps(row,indent=2)+'\n```' for row in pilot_rows)
            assert pilot_path.read_text()==expected_markdown
        status=read_json(status_path) if status_path.exists() else None
        if status:
            assert status['attempt']==ATTEMPT and status['arm']==arm and status['pilot_pass'] is h5
            if status['status']=='H5_FAILED':assert h5 is False and not evaluation_turns
            if status['status'] in {'COMPLETE','H1_FAILED'}:
                assert sum(row['split']=='evaluation' for row in saved)==48
        if return_path.exists():
            assert 'return_evidence' in journal,'Unexpected separate returns unsupported by frozen worker'
        else:assert not any(requires_return_evidence(row) for row in raw)
        results[arm]={'requested':len(requests),'returned':len(raw),'pending_trial_ids':journal['pending_trial_ids'],
            'pending_request_disposition':('No persisted completion at terminal recovery for the listed pending requests; inference completion and termination cause are unknown.' if journal['pending_trial_ids'] else 'Every request has a matching committed completion; no pending requests.'),
            'pilot_raw_turns':len(pilot_rows),'evaluation_raw_turns':evaluation_turns,
            'terminal_pilot_episodes':len(complete_pilots),
            'terminal_evaluation_episodes':sum(row['split']=='evaluation' for row in derived.values()),
            'saved_trajectory_episodes':len(saved),'terminal_episodes_without_saved_trajectory':sorted(set(derived)-{row['episode_id'] for row in saved}),
            'incomplete_requested_episode_ids':sorted(set(states)-set(derived)),
            'not_requested_evaluation_episode_ids':sorted(key for key,(split,_) in expected.items() if split=='evaluation' and key not in states),
            'strict_parser_counts':dict(sorted(parser_counts.items())),'capped_turns':capped,'unclosed_turns':unclosed,
            'mechanical_failure_records':failures,'H5':h5,'worker_status':status['status'] if status else 'INTERRUPTED_OR_NOT_STARTED',
            'full_sweep_started':bool(evaluation_turns),'request_journal':journal,
            'separate_return_stream':'PRESENT_AND_AUDITED' if return_path.exists() else 'NOT_CAPTURED_BY_FROZEN_WORKER',
            'exact_prompt_and_token_replay':True if requests else 'NO_REQUESTS',
            'trajectory_replay':'ALL_PRESENT_SAVED_TRAJECTORIES_EXACT'}
    assert results['dose_p0_5']['H5'] is False
    assert results['dose_p0_5']['strict_parser_counts'].get('MULTIPLE_TOOL_CALLS')==4
    assert results['dose_p0_5']['strict_parser_counts'].get('NO_TOOL_CALL')==1
    assert results['dose_p0_5']['capped_turns']==results['dose_p0_5']['unclosed_turns']==0
    assert all(file_hash(root/name)==value for name,value in before.items()),'Audited evidence changed'
    totals={key:sum(arm[key] for arm in results.values()) for key in ('requested','returned','pilot_raw_turns',
        'evaluation_raw_turns','terminal_pilot_episodes','terminal_evaluation_episodes','saved_trajectory_episodes','capped_turns','unclosed_turns')}
    totals['pending_requests']=sum(len(arm['pending_trial_ids']) for arm in results.values())
    return {'schema':'phase1_dose_terminal_failure_provenance_v1','attempt':ATTEMPT,'app_id':APP,
        'audited_utc':datetime.now(timezone.utc).isoformat(),'app_stopped_verified':True,
        'terminal_observation':recovery['app_stopped_after'],'recovery':recovery,
        'frozen_input_hashes_verified':True,'frozen_input_count':len(frozen),'frozen_input_sha256':frozen,
        'source_sha256':{name:file_hash(root/name) for name in sorted(bound)},'arms':results,'totals':totals,
        'primary_qualification_passed':False,'phase2_authorized':False,'phase3_authorized':False,
        'complete_five_alpha_receipt':False,'accepted_dose_curve':False,'no_further_wording_search':True,
        'interpretation':'App terminal with mandatory H5 failure. Preserve all committed observations and any explicit unresolved requests. No cloud cancellation is inferred from the stop request. No full five-alpha receipt or accepted dose curve; no threshold waiver, pooling, further wording search, or higher-cap retry for uncapped envelope failures.'}


def markdown(value):
    lines=['# D4 terminal failure provenance','',value['interpretation'],'',
        f"App `{APP}` independently verified stopped at {value['terminal_observation']['lifecycle']['stopped_at']}.",
        f"All {value['frozen_input_count']} admitted input hashes remain unchanged; strict parser, exact prompts/token IDs and every present trajectory independently replayed.",'',
        '| Arm | Requested | Returned | Pending | Complete pilots | Evaluation episodes | H5 | Worker status |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for name,row in value['arms'].items():
        lines.append(f"| {name} | {row['requested']} | {row['returned']} | {len(row['pending_trial_ids'])} | {row['terminal_pilot_episodes']} | {row['terminal_evaluation_episodes']} | {row['H5']} | {row['worker_status']} |")
    persistence=(f"All {value['totals']['requested']} requests have {value['totals']['returned']} matching committed completions; no pending requests."
        if value['totals']['pending_requests']==0 else
        'The explicitly listed pending requests have no committed completion; whether inference finished before app termination is unknown.')
    lines.extend(['', persistence+' The frozen worker captured raw completions and writeahead prompts but did not implement a separate unparsed-return journal. No such evidence is invented.',
        '', 'Primary qualification remains failed. Phase 2 and Phase 3 remain unauthorized. These infrastructure audit files do not constitute a complete numerical experiment receipt.'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recover-and-audit',action='store_true',required=True)
    args=parser.parse_args()
    recovery=recover(ROOT)
    value=audit(ROOT,recovery)
    immutable(ROOT/(BASE+'.json'),(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
    immutable(ROOT/(BASE+'.md'),markdown(value).encode())
    print(json.dumps({'audit_path':BASE+'.json','totals':value['totals'],'arms':{key:{k:r[k] for k in ('worker_status','H5','requested','returned','terminal_pilot_episodes','terminal_evaluation_episodes')} for key,r in value['arms'].items()}},indent=2))


if __name__=='__main__':main()
