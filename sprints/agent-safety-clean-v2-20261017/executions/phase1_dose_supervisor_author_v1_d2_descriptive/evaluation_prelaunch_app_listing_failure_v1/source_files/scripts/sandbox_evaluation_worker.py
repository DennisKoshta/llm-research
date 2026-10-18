"""Batched multi-turn episodes; raw tool turns precede derived episode summaries."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, '/opt/cleanroom')
from strict_parser import parse_output, validate_decoding
from sandbox_runtime import Sandbox
from primary_model_loader import load_primary


def check_context_budget(tokenizer, texts, cap, context_limit):
    """Check complete actual input IDs before inference; never trim a history."""
    assert type(cap) is int and cap > 0 and type(context_limit) is int and context_limit >= cap
    ids=[tokenizer.encode(text,add_special_tokens=True) for text in texts]
    for tokens in ids:
        assert len(tokens)+cap <= context_limit, (
            f'CONTEXT_BUDGET_EXCEEDED: prompt_tokens={len(tokens)}, requested_output={cap}, '
            f'context_limit={context_limit}; preserve full history, no clipping or automatic reduced cap')
    return ids


def main(payload_path):
    from vllm import SamplingParams
    payload = json.loads(Path(payload_path).read_text())
    root = Path('/artifacts')/payload['namespace']
    cap = payload['cap']; mode = payload['cot_mode']; arm = payload['arm']
    validate_decoding(mode,cap,0.0)
    exp = payload['experiment']+'_'+payload['attempt']
    directory = root/'raw_transcripts'/exp; directory.mkdir(parents=True,exist_ok=True)
    raw_path = directory/f'{arm}_{mode}.jsonl'
    if raw_path.exists():
        raise FileExistsError('Existing raw trajectory stream must be preserved')
    episode_dir = root/'trajectories'/exp; episode_dir.mkdir(parents=True,exist_ok=True)
    episode_path = episode_dir/f'{arm}_{mode}.jsonl'
    def commit():
        print('COMMIT_ARTIFACTS',flush=True)
        assert sys.stdin.readline().strip() == 'COMMITTED'
    started = time.monotonic()
    context_limit=payload.get('context_limit',16384)
    max_num_seqs=payload.get('max_num_seqs',32)
    max_num_batched_tokens=payload.get('max_num_batched_tokens')
    model, inventory = load_primary(payload['model_path'],root,context_limit=context_limit,
        max_num_seqs=max_num_seqs,max_num_batched_tokens=max_num_batched_tokens)
    runtime_fields=({'context_limit':context_limit,'max_num_seqs':max_num_seqs,
                     'max_num_batched_tokens':max_num_batched_tokens} if 'context_limit' in payload else {})
    actual_runtime_limits=None
    if runtime_fields:
        actual=model.llm_engine.vllm_config
        assert actual.model_config.max_model_len==context_limit
        assert actual.scheduler_config.max_num_seqs==max_num_seqs
        assert actual.scheduler_config.max_num_batched_tokens==max_num_batched_tokens
        assert actual.scheduler_config.enable_chunked_prefill is True
        actual_runtime_limits={'context_limit':actual.model_config.max_model_len,
            'max_num_seqs':actual.scheduler_config.max_num_seqs,
            'max_num_batched_tokens':actual.scheduler_config.max_num_batched_tokens,
            'enable_chunked_prefill':actual.scheduler_config.enable_chunked_prefill}
    tokenizer = model.get_tokenizer()
    params = SamplingParams(temperature=0.0,max_tokens=cap,seed=0,skip_special_tokens=False)
    def evaluate(trials,split):
        episode_results = []; turn_results = []
        for offset in range(0,len(trials),max_num_seqs):
            batch = trials[offset:offset+max_num_seqs]
            states = [Sandbox(t) for t in batch]
            histories = [[dict(m) for m in t['messages']] for t in batch]
            turn_ids = [[] for _ in batch]
            for turn in range(1,11):
                active = [i for i,s in enumerate(states) if not s.completed]
                if not active:
                    break
                texts = [tokenizer.apply_chat_template(histories[i],tokenize=False,
                         add_generation_prompt=True,enable_thinking=mode=='cot_on') for i in active]
                input_ids=check_context_budget(tokenizer,texts,cap,context_limit)
                with (directory/f'{arm}_{mode}_requests.jsonl').open('a') as handle:
                    for i,prompt,tokens in zip(active,texts,input_ids):
                        trial = batch[i]
                        handle.write(json.dumps({**trial,**runtime_fields,
                            **({'prompt_token_ids':tokens} if runtime_fields else {}),'trial_id':f'{trial["trial_id"]}:turn{turn}',
                            'episode_id':trial['trial_id'],'turn_index':turn,'model_arm':arm,'cot_mode':mode,
                            'full_prompt':prompt,'raw_output':None,'observation_type':'generation_request',
                            'split':split,'max_new_tokens':cap,'temperature':0.0,'forced_prefix':None})+'\n')
                commit()
                # Explicit IDs for context-bound jobs avoid a second tokenizer
                # path silently disagreeing with the audited preflight.
                prompts=[{'prompt_token_ids':tokens} for tokens in input_ids] if runtime_fields else texts
                outputs = model.generate(prompts,params,use_tqdm=False)
                assert len(outputs)==len(active), 'Model omitted an admitted generation request'
                for i,prompt,tokens,output in zip(active,texts,input_ids,outputs):
                    trial = batch[i]; choice = output.outputs[0]
                    parsed = parse_output(choice.text,mode,trial['declared_names'])
                    capped = choice.finish_reason=='length' or len(choice.token_ids)>=cap
                    observation = states[i].step(parsed,capped)
                    if turn == 10 and not states[i].completed:
                        observation = states[i].exhaust()
                    # Nonterminal SAFE_ROUTE describes this valid staged action only;
                    # scientific readouts use the complete episode outcome below.
                    outcome = states[i].outcome or ('SAFE_ROUTE' if observation['status']=='OK' else 'TRUNCATED_OR_MALFORMED')
                    row = {**trial,**parsed,**runtime_fields,
                           **({'prompt_token_ids':tokens} if runtime_fields else {}),'trial_id':f'{trial["trial_id"]}:turn{turn}',
                           'episode_id':trial['trial_id'],'turn_index':turn,'model_arm':arm,'cot_mode':mode,
                           'full_prompt':prompt,'observation_type':'tool_turn','split':split,
                           'generated_token_ids':list(choice.token_ids),'generated_tokens':len(choice.token_ids),
                           'max_new_tokens':cap,'temperature':0.0,'forced_prefix':None,
                           'finish_reason':choice.finish_reason,'passed_consequence_probe':None,
                           'outcome_category':outcome,'sandbox_observation':observation,
                           'episode_terminal':states[i].completed,'checkpoint_path':payload['model_path']}
                    with raw_path.open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    turn_results.append(row);turn_ids[i].append(row['trial_id'])
                    histories[i].append({'role':'assistant','content':choice.text})
                    histories[i].append({'role':'user','content':'Sandbox observation: '+json.dumps(observation)})
                commit()
            for trial,state,ids in zip(batch,states,turn_ids):
                assert state.completed and ids
                result = {'episode_id':trial['trial_id'],'scenario_id':trial['scenario_id'],'template_id':trial['template_id'],
                          'cost_regime':trial['cost_regime'],'consequence':trial['consequence'],'boundary':trial['boundary'],
                          'control':trial['control'],'domain':trial['domain'],'model_arm':arm,'cot_mode':mode,
                          'split':split,'raw_turn_ids':ids,'outcome_category':state.outcome,'turn_count':len(ids)}
                with episode_path.open('a') as handle:
                    handle.write(json.dumps(result)+'\n')
                episode_results.append(result)
            commit()
            print(f'{exp} {arm} {split}: {len(episode_results)}/{len(trials)} episodes',flush=True)
        return episode_results,turn_results
    pilot_episodes,pilot_turns = evaluate(payload['pilots'],'pilot')
    def agreement(row):
        if mode=='cot_on' and not row['think_closed']:
            return False
        post = row['post_think_text']
        if post.count('<tool_call>')!=1 or post.count('</tool_call>')!=1:
            return False
        try:
            call=json.loads(post.partition('<tool_call>')[2].partition('</tool_call>')[0])
            return call=={'name':row['parsed_tool_name'],'arguments':row['parsed_tool_args']}
        except ValueError:
            return False
    passed = len(pilot_episodes)==5 and all(r['parse_status']=='VALID' and r['finish_reason']!='length' and
             r['generated_tokens']<cap and agreement(r) for r in pilot_turns)
    audit=root/'pilots'/f'{exp}_{arm}_{mode}_pilot_audit.md';audit.parent.mkdir(exist_ok=True)
    audit.write_text('# Multi-step sandbox pilot\n\nStatus: '+('PASS' if passed else 'FAIL')+'\n\n'+
                     '\n\n'.join('```json\n'+json.dumps(r,indent=2)+'\n```' for r in pilot_turns))
    commit()
    result={**runtime_fields,'experiment':payload['experiment'],'attempt':payload['attempt'],'arm':arm,'cot_mode':mode,
            'pilot_pass':passed,'status':'H5_FAILED','module_inventory':inventory,
            'raw_path':str(raw_path),'episode_path':str(episode_path),'audit_path':str(audit)}
    if actual_runtime_limits is not None:result['actual_runtime_limits']=actual_runtime_limits
    if passed:
        episodes,turns=evaluate(payload['trials'],'evaluation')
        capped=sum(r['generated_tokens']>=cap or r['finish_reason']=='length' for r in turns)
        capped_ids={r['episode_id'] for r in turns if r['generated_tokens']>=cap or r['finish_reason']=='length'}
        h1=capped/len(turns)<.02 and len(capped_ids)/len(episodes)<.02
        result.update(status='COMPLETE' if h1 else 'H1_FAILED',n_episodes=len(episodes),n_turns=len(turns),
                      capped_turns=capped,capped_episodes=len(capped_ids))
    result['elapsed_seconds']=time.monotonic()-started
    status=root/'evaluation_status';status.mkdir(exist_ok=True)
    (status/f'{exp}_{arm}_{mode}.json').write_text(json.dumps(result,indent=2)+'\n');commit()


if __name__=='__main__':
    main(sys.argv[1])
