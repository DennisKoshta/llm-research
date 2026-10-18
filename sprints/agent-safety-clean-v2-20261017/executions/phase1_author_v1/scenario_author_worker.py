"""Unified, paired scenario authorship with raw evidence and fail-closed pilots."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, '/opt/cleanroom')
from strict_parser import parse_output, validate_decoding
from scenario_contract import validate_author_pair


def main(payload_path):
    from vllm import LLM, SamplingParams
    payload = json.loads(Path(payload_path).read_text())
    cap = payload['cap']; validate_decoding('cot_on', cap, 0.0)
    root = Path('/artifacts')/payload['namespace']
    exp = 'phase1_author_'+payload['attempt']
    directory = root/'raw_transcripts'/exp; directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory/'flash_next_uncensored.jsonl'
    if raw_path.exists():
        raise FileExistsError('Preserve existing author stream; a partial attempt requires a new ID')
    def commit():
        print('COMMIT_ARTIFACTS', flush=True)
        assert sys.stdin.readline().strip() == 'COMMITTED'
    started = time.monotonic()
    model = LLM(model=payload['model_path'], tensor_parallel_size=8, enable_expert_parallel=True,
                max_model_len=8192, max_num_seqs=16, gpu_memory_utilization=.88,
                enable_prefix_caching=False, enforce_eager=True, trust_remote_code=False,
                safetensors_load_strategy='eager')
    tokenizer = model.get_tokenizer()
    parameters = SamplingParams(temperature=0.0, max_tokens=cap, seed=0, skip_special_tokens=False)
    def evaluate(trials, split):
        records = []
        for offset in range(0, len(trials), 16):
            batch = trials[offset:offset+16]
            prompts = [tokenizer.apply_chat_template(t['messages'], tokenize=False,
                       add_generation_prompt=True, enable_thinking=True) for t in batch]
            with (directory/'flash_next_uncensored_requests.jsonl').open('a') as handle:
                for trial, prompt in zip(batch, prompts):
                    handle.write(json.dumps({**trial, 'model_arm': 'flash_next_uncensored', 'full_prompt': prompt,
                        'raw_output': None, 'observation_type': 'generation_request', 'split': split,
                        'max_new_tokens': cap, 'temperature': 0.0, 'forced_prefix': None,
                        'revision': payload['revision']})+'\n')
            commit()
            outputs = model.generate(prompts, parameters, use_tqdm=False)
            for trial, prompt, output in zip(batch, prompts, outputs):
                choice = output.outputs[0]
                parsed = parse_output(choice.text, 'cot_on', trial['declared_names'])
                valid = False; validation_error = None
                if parsed['parse_status'] == 'VALID' and choice.finish_reason != 'length' and len(choice.token_ids) < cap:
                    try:
                        validate_author_pair(parsed['parsed_tool_args'], trial)
                        valid = True
                    except (AssertionError, TypeError, ValueError) as exc:
                        validation_error = type(exc).__name__+': '+str(exc)
                row = {**trial, **parsed, 'observation_type': 'scenario_authoring',
                       'model_arm': 'flash_next_uncensored', 'full_prompt': prompt,
                       'generated_tokens': len(choice.token_ids), 'generated_token_ids': list(choice.token_ids),
                       'max_new_tokens': cap, 'temperature': 0.0, 'forced_prefix': None,
                       'finish_reason': choice.finish_reason, 'split': split, 'revision': payload['revision'],
                       'model_path': payload['model_path'], 'cost_regime': 'not_applicable',
                       'consequence': 'paired_authoring', 'boundary': 'not_applicable',
                       'passed_consequence_probe': None, 'author_contract_valid': valid,
                       'author_contract_error': validation_error,
                       'outcome_category': 'SAFE_ROUTE' if valid else 'TRUNCATED_OR_MALFORMED'}
                with raw_path.open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                records.append(row)
            commit()
            print(f'author {split}: {len(records)}/{len(trials)}', flush=True)
        return records
    pilots = evaluate(payload['pilots'], 'pilot')
    def independently_valid(row):
        if not row['think_closed'] or not row['author_contract_valid']:
            return False
        post = row['raw_output'].split('</think>', 1)[1]
        if post.count('<tool_call>') != 1 or post.count('</tool_call>') != 1:
            return False
        try:
            call = json.loads(post.partition('<tool_call>')[2].partition('</tool_call>')[0])
            return call == {'name': row['parsed_tool_name'], 'arguments': row['parsed_tool_args']}
        except ValueError:
            return False
    passed = len(pilots) == 5 and all(independently_valid(r) for r in pilots)
    audit = root/'pilots'/f'{exp}_pilot_audit.md'; audit.parent.mkdir(exist_ok=True)
    audit.write_text('# Paired scenario author pilot\n\nStatus: '+('PASS' if passed else 'FAIL')+'\n\n'+
                     '\n\n'.join('```json\n'+json.dumps(r, indent=2)+'\n```' for r in pilots))
    commit()
    result = {'attempt': payload['attempt'], 'pilot_pass': passed, 'status': 'H5_FAILED',
              'raw_path': str(raw_path), 'audit_path': str(audit)}
    if passed:
        rows = evaluate(payload['requests'], 'authoring')
        capped = sum(r['generated_tokens'] >= cap or r['finish_reason'] == 'length' for r in rows)
        result.update(n=len(rows), capped=capped, all_contracts_valid=all(r['author_contract_valid'] for r in rows),
                      status='COMPLETE' if len(rows) == 40 and capped/len(rows) < .02 and
                      all(r['author_contract_valid'] for r in rows) else 'AUTHORING_GATE_FAILED')
    result['elapsed_seconds'] = time.monotonic()-started
    status = root/'qualification_status'; status.mkdir(exist_ok=True)
    (status/f'author_{payload["attempt"]}_flash_next_uncensored.json').write_text(json.dumps(result, indent=2)+'\n')
    commit()


if __name__ == '__main__':
    main(sys.argv[1])
