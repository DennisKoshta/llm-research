"""Bounded structured-record inference; importing this module performs no inference.

Natural reviews, paraphrases and factual judgments are records, not sandbox action
outcomes. False semantic judgments are retained as valid records. Every immutable
role/task/mode attempt has five fresh pilots and three durable evidence chains.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time

from exp23_author_worker import ChainWriter, digest, hash_bytes, read_chain, utc_now, write_exclusive
from strict_parser import parse_output, unique_object, reject_constant

NS = 'agent-safety-clean-v2-20261017'
PRIMARY_REPO = 'Qwen/Qwen3.8-27B'
PRIMARY_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
LLAMA_REPO = 'nvidia/Llama-3.3-70B-Instruct-FP8'
LLAMA_REVISION = 'fde04ee76a27704c88f569542ef023b57d4d0362'
PRIMARY_ROLES = {'intact_reviewer': 'intact', 'pca_101_reviewer': 'pca_101',
                 'refusal_ablated_reviewer': 'refusal_ablated'}
LLAMA_ROLES = {'llama_paraphraser', 'independent_semantic_judge', 'independent_qa_judge'}
RUNTIME_FILES = ('structured_records_worker.py', 'llama_model_loader.py', 'llama_serving_worker.py',
                 'primary_model_loader.py', 'primary_fp8_inventory.py', 'exp23_author_worker.py',
                 'exp23_design.py', 'scenario_contract.py', 'strict_parser.py')
READY_KEYS = ('source', 'post_fp8', 'primary', 'author', 'calibration', 'dose')
TOKENIZER_FILES={'tokenizer_config.json','tokenizer.json','chat_template.jinja','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','generation_config.json'}


def runtime_versions():
    result={}
    for name in ('transformers','tokenizers','jinja2','vllm','torch'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def schema_definition(schema):
    """Validate the supported strict schema dialect; unknown constraints fail closed."""
    if not isinstance(schema, dict):
        raise ValueError('Schema must be an object')
    kind = schema.get('type')
    allowed = {'object': {'type', 'properties', 'required', 'additionalProperties'},
               'array': {'type', 'items', 'minItems', 'maxItems'},
               'string': {'type', 'minLength', 'maxLength'},
               'integer': {'type', 'minimum', 'maximum'},
               'number': {'type', 'minimum', 'maximum'}, 'boolean': {'type'}}
    if kind not in allowed or set(schema)-allowed[kind]:
        raise ValueError('Unsupported schema type or constraint')
    if kind == 'object':
        properties, required = schema.get('properties'), schema.get('required')
        if (not isinstance(properties, dict) or not isinstance(required, list) or
                any(not isinstance(key, str) for key in required) or
                len(set(required)) != len(required) or set(required) != set(properties) or
                schema.get('additionalProperties') is not False):
            raise ValueError('Objects require exactly all declared properties')
        for child in properties.values(): schema_definition(child)
    if kind == 'array': schema_definition(schema.get('items'))
    for key in set(schema) & {'minLength', 'maxLength', 'minItems', 'maxItems'}:
        if type(schema[key]) is not int or schema[key] < 0: raise ValueError('Invalid nonnegative schema bound')
    for key in set(schema) & {'minimum', 'maximum'}:
        if type(schema[key]) not in (int, float) or not math.isfinite(schema[key]): raise ValueError('Invalid numeric schema bound')
    for lower, upper in [('minLength', 'maxLength'), ('minItems', 'maxItems'), ('minimum', 'maximum')]:
        if lower in schema and upper in schema and schema[lower] > schema[upper]: raise ValueError('Inverted schema bounds')


def typed_value(value, schema):
    schema_definition(schema)
    kind = schema['type']
    if kind == 'object':
        return (isinstance(value, dict) and set(value) == set(schema['required']) and
                all(typed_value(value[key], child) for key, child in schema['properties'].items()))
    if kind == 'array':
        return (isinstance(value, list) and schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', math.inf)
                and all(typed_value(item, schema['items']) for item in value))
    if kind == 'string':
        return (isinstance(value, str) and schema.get('minLength', 0) <= len(value.strip()) <= schema.get('maxLength', math.inf))
    if kind == 'boolean': return type(value) is bool
    if kind == 'integer' and type(value) is not int: return False
    if kind == 'number' and (type(value) not in (int, float) or not math.isfinite(value)): return False
    return schema.get('minimum', -math.inf) <= value <= schema.get('maximum', math.inf)


def request_task(request):
    role, key = request['model_role'], request['trial_id']
    if role in PRIMARY_ROLES and key.startswith('exp23:review:'):
        expected = {'intact_reviewer': 'B1', 'pca_101_reviewer': 'B2', 'refusal_ablated_reviewer': 'R1'}[role]
        if key.rsplit(':', 1)[-1] != expected: raise ValueError('Reviewer condition does not match checkpoint role')
        return 'natural_review'
    if role == 'llama_paraphraser' and key.startswith('exp23:paraphrase:'): return 'paraphrase'
    if role == 'independent_semantic_judge':
        for task in ('author_judge', 'paraphrase_judge', 'clause_judge', 'clauses'):
            if key.startswith('exp23:'+task+':'): return task
    if role == 'independent_qa_judge' and key.startswith('qa_grade:'):
        if request['probe_kind'] not in {'consequence_awareness', 'rule_identification'}: raise ValueError('Unknown QA task')
        return request['probe_kind']
    raise ValueError('Only frozen Exp2.3 record roles and arm-blinded Exp1.1 QA grading are supported')


def identity(role, checkpoint_sha256=None):
    if role in PRIMARY_ROLES:
        if not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256 or ''): raise ValueError('Primary checkpoint requires manifest hash')
        return {'family': 'primary', 'model_id': PRIMARY_REPO, 'model_revision': checkpoint_sha256,
                'base_revision': PRIMARY_REVISION, 'model_arm': PRIMARY_ROLES[role],
                'model_path': f'/weights/{NS}/checkpoints_full_residual_v2/{PRIMARY_ROLES[role]}'}
    if role not in LLAMA_ROLES: raise ValueError('Unsupported record role')
    return {'family': 'llama', 'model_id': LLAMA_REPO, 'model_revision': LLAMA_REVISION,
            'base_revision': LLAMA_REVISION, 'model_arm': 'llama_intact',
            'model_path': f'/weights/{NS}/source_models/{LLAMA_REPO.replace("/", "__")}'}


def pilot_requests(requests, experiment):
    pilots = []
    # Five fresh generation trials with unique IDs; sources are evenly spaced.
    # QA batches with fewer than five valid source answers repeat pilot content
    # only, preserving every unique full answer without inventing evaluations.
    for index in range(5):
        source = requests[index*(len(requests)-1)//4]
        row = deepcopy(source)
        row.update(trial_id=f'{experiment}:pilot:{index}', split='pilot', source_request_id=source['trial_id'])
        pilots.append(row)
    return pilots


def build_payload(attempt, requests, *, readiness, input_sources, source_hashes,
                  specification_sha256, request_source_path, tokenizer_files, tokenizer_source_paths, component_sources,
                  checkpoint_sha256=None, model_provenance=None):
    requests = deepcopy(requests)
    if not requests: raise ValueError('Bounded batches require immutable full requests')
    minimum=1 if requests[0].get('model_role')=='independent_qa_judge' else 5
    if not minimum <= len(requests) <= 192: raise ValueError('Bounded batches require 5–192 requests, or 1–192 existing QA judgments')
    role, task, mode = requests[0]['model_role'], request_task(requests[0]), requests[0]['cot_mode']
    experiment = f'structured_{attempt}_{role}_{task}_{mode}'
    value = {'schema': 'structured_records_payload_v1', 'namespace': NS, 'attempt': attempt,
             'experiment': experiment, 'model_role': role, 'task': task, 'cot_mode': mode,
             'identity': identity(role, checkpoint_sha256), 'central_job':experiment,
             'tokenizer_files':tokenizer_files,'tokenizer_source_paths':tokenizer_source_paths,'component_sources':component_sources,
             'requests': requests,
             'pilots': pilot_requests(requests, experiment), 'batch_size': 5, 'max_model_len': 16384,
             'tensor_parallel_size': 1, 'readiness': deepcopy(readiness), 'input_sources': input_sources,
             'source_hashes': source_hashes, 'specification_sha256': specification_sha256,
             'request_source_path': request_source_path,
             'model_provenance': model_provenance or {}, 'purpose': 'Structured records; no executor evaluation',
             'phase2_executor_launch': False}
    value['request_manifest_sha256'] = digest({'pilots': value['pilots'], 'requests': requests})
    value['payload_sha256'] = digest(value)
    validate_payload(value)
    return value


def validate_payload(value):
    json.dumps(value, allow_nan=False)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}', value.get('attempt', '')): raise ValueError('Unsafe attempt ID')
    role, task, mode = value['model_role'], value['task'], value['cot_mode']
    if value['namespace'] != NS or value['experiment'] != f'structured_{value["attempt"]}_{role}_{task}_{mode}': raise ValueError('Namespace mismatch')
    if value['central_job']!=value['experiment'] or not isinstance(value['component_sources'],dict): raise ValueError('Structured central job/Phase1 source selection changed')
    if not {'tokenizer_config.json','tokenizer.json'}<=set(value['tokenizer_files'])<=TOKENIZER_FILES or set(value['tokenizer_files'])!=set(value['tokenizer_source_paths']):
        raise ValueError('Missing complete tokenizer source binding')
    for name,h in value['tokenizer_files'].items():
        path=value['tokenizer_source_paths'][name]
        if Path(path).name!=name or not re.fullmatch('[0-9a-f]{64}',h) or value['input_sources'].get(path)!=h:
            raise ValueError('Exact source tokenizer bytes not bound')
    expected = identity(role, value['identity']['model_revision'] if role in PRIMARY_ROLES else None)
    if value['identity'] != expected: raise ValueError('Pinned role/checkpoint mapping mismatch')
    if (value['batch_size'] != 5 or value['max_model_len'] != 16384 or value['tensor_parallel_size'] != 1 or
            value['phase2_executor_launch'] is not False): raise ValueError('Unsupported runtime or executor scope')
    readiness = value['readiness']
    if readiness.get('schema') != 'phase1_readiness_raw_audit_v1' or readiness.get('primary_phase2_authorized') is not True:
        raise ValueError('Independent Phase1/calibration/dose prerequisites are not authorized')
    for name in READY_KEYS:
        component = readiness['components'][name]
        if component['status'] != 'PASS' or component['gate_passed'] is not True: raise ValueError('Required prerequisite did not pass')
        if value['input_sources'].get(component['receipt_path']) != component['receipt_sha256']: raise ValueError('Missing prerequisite source binding')
    if readiness.get('provenance', {}).get('status') != 'PASS': raise ValueError('Independent aggregate provenance did not pass')
    for filename, fingerprint in readiness['provenance']['artifact_sha256'].items():
        if value['input_sources'].get(filename) != fingerprint: raise ValueError('Readiness provenance artifact is not bound')
    if (value['request_source_path'] not in value['input_sources'] or
            value['input_sources'].get('FINAL_RESEARCH_SPECIFICATION.md') != value['specification_sha256']):
        raise ValueError('Exact request/specification source is not bound')
    if role in PRIMARY_ROLES:
        filename = f'checkpoints_full_residual_v2/{PRIMARY_ROLES[role]}_materialization.json'
        if value['input_sources'].get(filename) != expected['model_revision']: raise ValueError('Role materialization manifest not bound')
    else:
        expected_provenance = {'download_manifest': 'infrastructure/'+LLAMA_REPO.replace('/', '__')+'_download.json',
                               'preflight': 'infrastructure/llama_preflight.json'}
        if value['model_provenance'] != expected_provenance or any(p not in value['input_sources'] for p in expected_provenance.values()):
            raise ValueError('Pinned Llama source provenance not bound')
    if set(value['source_hashes']) != set(RUNTIME_FILES): raise ValueError('Incomplete runtime source binding')
    for source, fingerprint in value['input_sources'].items():
        if Path(source).is_absolute() or '..' in Path(source).parts: raise ValueError('Input source outside sprint')
        if not re.fullmatch('[0-9a-f]{64}', fingerprint): raise ValueError('Invalid input hash')
    if any(not re.fullmatch('[0-9a-f]{64}', h) for h in [*value['source_hashes'].values(), value['specification_sha256']]): raise ValueError('Invalid source hash')
    requests = value['requests']
    minimum=1 if role=='independent_qa_judge' else 5
    if not minimum <= len(requests) <= 192 or len({r['trial_id'] for r in requests}) != len(requests): raise ValueError('Request cardinality/IDs invalid')
    for row in requests:
        if row['model_role'] != role or request_task(row) != task or row['cot_mode'] != mode: raise ValueError('Mixed role/task/cot batch')
        if (type(row['max_new_tokens']) is not int or row['max_new_tokens'] < (4096 if mode == 'cot_on' else 1024) or row['max_new_tokens'] > 8192 or
                type(row['temperature']) not in (int, float) or row['temperature'] != 0.0 or row['forced_prefix'] is not None or
                type(row['rollouts']) is not int or row['rollouts'] != 1 or mode not in {'cot_on', 'cot_off'}):
            raise ValueError('Frozen decoding budget/prefix mismatch')
        if row['split'] not in {'artifact_generation', 'qa_grading'}: raise ValueError('Only record-generation requests allowed')
        if not row['messages'] or row['messages'][0]['role'] != 'system': raise ValueError('Missing system message')
        if any(set(m) != {'role', 'content'} or m['role'] not in {'system', 'user'} or not isinstance(m['content'], str) for m in row['messages']):
            raise ValueError('No assistant prefix, hidden message fields, or unsupported role allowed')
        tool = 'record_review_note' if task in {'natural_review', 'paraphrase'} else 'record_clause_spans' if task == 'clauses' else 'record_assessment'
        if row['declared_names'] != [tool]: raise ValueError('Record tool differs from fixed task')
        schema_definition(row['arguments_schema'])
        for key in ('model_id', 'model_revision'):
            if key in row and row[key] != expected[key]: raise ValueError('Request model identity differs from fixed role')
        if role == 'independent_qa_judge':
            allowed = {'trial_id','grading_id','probe_kind','schema','model_role','model_id','model_revision','messages',
                       'declared_names','arguments_schema','cot_mode','max_new_tokens','temperature','rollouts','forced_prefix','split'}
            if set(row) != allowed: raise ValueError('QA grader received private join or unrecognized metadata')
            if row['trial_id'] != 'qa_grade:'+row['grading_id']: raise ValueError('QA opaque join identifier changed')
    if value['pilots'] != pilot_requests(requests, value['experiment']): raise ValueError('Fresh fixed pilot selection changed')
    if value['request_manifest_sha256'] != digest({'pilots': value['pilots'], 'requests': requests}): raise ValueError('Request bytes changed')
    if value['payload_sha256'] != digest({k:v for k,v in value.items() if k != 'payload_sha256'}): raise ValueError('Payload changed')


def independent_call(raw, mode):
    if mode == 'cot_on' and '</think>' not in raw: raise ValueError('TRUNCATED_IN_THINK')
    post = raw.split('</think>', 1)[1] if '</think>' in raw else raw
    if post.count('<tool_call>') != 1 or post.count('</tool_call>') != 1: raise ValueError('Require one complete tool block')
    _, _, tail = post.partition('<tool_call>'); body, closing, _ = tail.partition('</tool_call>')
    if not closing: raise ValueError('Missing closing block')
    data = body.lstrip()
    call, end = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant).raw_decode(data)
    if data[end:].strip() or not isinstance(call, dict) or set(call) != {'name', 'arguments'}: raise ValueError('Invalid record envelope')
    return call


def parsed_record(raw, token_ids, finish, request):
    parsed = parse_output(raw, request['cot_mode'], request['declared_names'])
    agreement, valid, error = False, False, None
    try:
        call = independent_call(raw, request['cot_mode'])
        agreement = call == {'name': parsed['parsed_tool_name'], 'arguments': parsed['parsed_tool_args']}
        if not agreement or parsed['parse_status'] != 'VALID' or not typed_value(call['arguments'], request['arguments_schema']): raise ValueError('Invalid typed structured record')
        if not token_ids or any(type(t) is not int or t < 0 for t in token_ids) or finish == 'length' or len(token_ids) >= request['max_new_tokens']: raise ValueError('Empty or capped generation')
        valid = True
    except (ValueError, TypeError, KeyError) as exc:
        error = type(exc).__name__+': '+str(exc)
    return {**parsed, 'record_valid': valid, 'record_error': error, 'independent_parser_agreement': agreement,
            'outcome_category': 'SAFE_ROUTE' if valid else 'TRUNCATED_OR_MALFORMED',
            'outcome_category_scope': 'STRUCTURED_RECORD_VALIDITY_ONLY_NOT_BEHAVIORAL_OUTCOME'}


def render_request(request, tokenizer, family):
    if family == 'llama':
        from llama_model_loader import rendered_messages
        messages = rendered_messages(request)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt.rstrip().endswith('<think>'): raise ValueError('Llama may not receive a forced reasoning prefix')
    else:
        messages = deepcopy(request['messages'])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=request['cot_mode'] == 'cot_on')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or any(type(t) is not int or t < 0 for t in ids): raise ValueError('Invalid prompt token IDs')
    if len(ids)+request['max_new_tokens'] > 16384: raise ValueError('Prompt plus complete generation budget exceeds context; never truncate')
    return messages, prompt, ids


def validate_inventory(inventory, family):
    loaded = inventory if family == 'primary' else inventory['loaded']
    if not isinstance(loaded, list) or len(loaded) != 1: raise ValueError('Require one independently audited TP1 worker')
    entry = loaded[0]
    if family == 'primary':
        if entry['expected_primary_unfused_modules'] != 400 or entry['loaded_primary_fused_fp8_modules'] != 256:
            raise ValueError('Primary FP8 module coverage differs')
        count = 256
    else:
        if (entry['loaded_model_class'] != 'LlamaForCausalLM' or entry['expected_unfused_fp8_modules'] != 560 or
                entry['loaded_fused_fp8_modules'] != 320 or inventory['serialized']['serialized_fp8_modules'] != 560):
            raise ValueError('Llama FP8 module coverage differs')
        if set(entry['weight_scales']) != set(entry['fp8_module_names']): raise ValueError('Llama scale inventory differs')
        count = 320
    if len(entry['fp8_module_names']) != count or len(set(entry['fp8_module_names'])) != count:
        raise ValueError('Duplicate or missing loaded FP8 module names')


def durable_call(root, payload):
    path = Path(root)/'group_calls'/f'{payload["experiment"]}.json'
    ledger = json.loads(path.read_text())
    if (ledger.get('state') != 'SPAWNED' or not isinstance(ledger.get('call_id'), str) or
            not ledger['call_id'].startswith('fc-') or ledger['payload_sha256'] != payload['payload_sha256'] or
            ledger['experiment'] != payload['experiment'] or ledger['role'] != payload['model_role'] or
            ledger['task'] != payload['task'] or ledger.get('job')!=payload['central_job'] or
            ledger.get('worker_module')!='modal_structured_records' or ledger.get('worker_function')!='records'):
        raise ValueError('Generation requires its durable, exact coordinator call ID')
    return {'call_id': ledger['call_id'], 'call_ledger_sha256': hash_bytes(path)}


class RecordRun:
    def __init__(self, root, payload, commit):
        validate_payload(payload)
        self.root, self.payload, self.commit = Path(root), payload, commit
        self.exp, self.role = payload['experiment'], payload['model_role']
        self.directory = self.root/'raw_transcripts'/self.exp
        self.started = time.monotonic()
        self.call_binding = durable_call(self.root, payload)
        write_exclusive(self.directory/'attempt_claim.json', {'payload_sha256': payload['payload_sha256'],
                        **self.call_binding, 'claimed_utc': utc_now()})
        write_exclusive(self.directory/'frozen_input.json', payload)
        self.raw = ChainWriter(self.directory/f'{self.role}.jsonl', payload['payload_sha256'])
        self.requests = ChainWriter(self.directory/f'{self.role}_requests.jsonl', payload['payload_sha256'])
        self.returned = ChainWriter(self.root/'structured_artifacts'/self.exp/'model_returns.jsonl', payload['payload_sha256'])
        self.commit()

    def evaluate(self, trials, model, sampling):
        records = []; tokenizer = model.get_tokenizer()
        for offset in range(0, len(trials), 5):
            batch = trials[offset:offset+5]; journals = []
            for trial in batch:
                messages, prompt, ids = render_request(trial, tokenizer, self.payload['identity']['family'])
                journals.append(self.requests.append({**trial, **self.payload['identity'],
                    'rendered_messages': messages, 'full_prompt': prompt, 'prompt_token_ids': ids,
                    'prompt_tokens': len(ids), 'prompt_token_ids_sha256': digest(ids),
                    'request_sha256': digest(trial), 'payload_sha256': self.payload['payload_sha256'],
                    'tokenizer_file_sha256':self.payload['tokenizer_files'],
                    'observation_type': 'generation_request', 'recorded_utc': utc_now()}))
            self.commit()
            outputs = model.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in journals],
                [sampling(t['max_new_tokens']) for t in batch], use_tqdm=False)
            returned = self.returned.append({'observation_type': 'generation_return_batch',
                'trial_ids': [t['trial_id'] for t in batch], 'request_record_sha256': [j['chain_sha256'] for j in journals],
                'outputs': [{'prompt_token_ids': list(o.prompt_token_ids),
                             'choices': [{'raw_output': c.text, 'token_ids': list(c.token_ids), 'finish_reason': c.finish_reason} for c in o.outputs]} for o in outputs],
                'payload_sha256': self.payload['payload_sha256'], 'recorded_utc': utc_now()})
            self.commit()  # Every returned output is durable before parsing or shape assertions.
            if len(outputs) != len(batch) or any(len(o.outputs) != 1 for o in outputs): raise ValueError('Unexpected returned batch shape; preserved without filtering')
            for index, (trial, journal, output) in enumerate(zip(batch, journals, outputs)):
                if list(output.prompt_token_ids) != journal['prompt_token_ids']: raise ValueError('Runtime prompt token IDs differ; returned evidence preserved')
                choice = output.outputs[0]
                row = {**trial, **self.payload['identity'], **parsed_record(choice.text, list(choice.token_ids), choice.finish_reason, trial),
                    'observation_type': 'structured_record', 'full_prompt': journal['full_prompt'],
                    'rendered_messages': journal['rendered_messages'], 'prompt_token_ids': journal['prompt_token_ids'],
                    'prompt_token_ids_sha256': journal['prompt_token_ids_sha256'],
                    'tokenizer_file_sha256':self.payload['tokenizer_files'],
                    'generated_token_ids': list(choice.token_ids), 'generated_tokens': len(choice.token_ids), 'finish_reason': choice.finish_reason,
                    'request_record_sha256': journal['chain_sha256'], 'generation_return_record_sha256': returned['chain_sha256'],
                    'return_index': index, 'request_sha256': digest(trial), 'payload_sha256': self.payload['payload_sha256'],
                    'tokenizer_file_sha256':self.payload['tokenizer_files'],
                    'raw_source_path': str(self.raw.path.relative_to(self.root)), 'passed_consequence_probe': None,
                    'scenario_id': trial.get('scenario_id', trial.get('grading_id', trial.get('source_request_id', trial['trial_id']))),
                    'template_id': 'structured_record', 'domain': 'synthetic_sandbox_records', 'cost_regime': 'not_applicable',
                    'consequence': 'not_applicable', 'boundary': 'not_applicable', 'tensor_parallel_size': 1,
                    'semantic_acceptance_inferred': False}
                records.append(self.raw.append(row))
            self.commit()
            print(f'{self.exp}: {batch[0]["split"]} {len(records)}/{len(trials)}', flush=True)
        return records

    def execute(self, model, sampling, inventory):
        write_exclusive(self.directory/'runtime_context.json',{'payload_sha256':self.payload['payload_sha256'],
            'identity':self.payload['identity'],'tokenizer_files':self.payload['tokenizer_files'],'runtime_versions':runtime_versions()})
        self.commit()
        validate_inventory(inventory, self.payload['identity']['family'])
        write_exclusive(self.directory/'loaded_model_inventory.json', {'identity': self.payload['identity'], 'inventory': inventory,
                        'payload_sha256': self.payload['payload_sha256']}); self.commit()
        pilots = self.evaluate(self.payload['pilots'], model, sampling)
        pilot_audit = audit_streams(self.root, self.payload)
        path = self.root/'pilots'/f'{self.exp}_pilot_audit.md'; path.parent.mkdir(exist_ok=True)
        with path.open('x') as handle:
            handle.write('# Structured-record five-trial pilot\n\nStatus: '+('PASS' if pilot_audit['H5'] else 'FAIL')+
                '\n\nRecord validity only; false semantic verdicts remain valid.\n\n'+
                '\n\n'.join('```json\n'+json.dumps(row, indent=2)+'\n```' for row in pilots)+'\n')
        self.commit()
        rows = self.evaluate(self.payload['requests'], model, sampling) if pilot_audit['H5'] else []
        audit = audit_streams(self.root, self.payload)
        complete = audit['complete_data'] and all(audit[k] for k in ('H1', 'H2', 'H4', 'H5', 'all_records_valid'))
        bundle_path = None
        if complete:
            bundle_path = self.root/'structured_artifacts'/f'{self.exp}_bundle.json'
            write_exclusive(bundle_path, {'schema': 'structured_record_artifact_bundle_v1', 'raw_artifacts': rows,
                'source_stream_audit': audit, 'payload_sha256': self.payload['payload_sha256'], 'semantic_acceptance_inferred': False})
        result = {'schema': 'structured_records_status_v1', 'status': 'RECORDS_COMPLETE' if complete else 'H5_FAILED' if not audit['H5'] else 'RECORD_GATE_FAILED',
            'experiment': self.exp, 'model_role': self.role, 'task': self.payload['task'], 'payload_sha256': self.payload['payload_sha256'],
            'stream_audit': audit, 'artifact_bundle_path': str(bundle_path) if bundle_path else None,
            'raw_path': str(self.raw.path), 'writeahead_path': str(self.requests.path), 'model_returns_path': str(self.returned.path),
            'pilot_audit_path': str(path), 'elapsed_seconds': time.monotonic()-self.started,
            'finished_at_utc': utc_now(), 'phase2_executor_launch': False}
        self.save_status(result)
        return result

    def save_status(self, result):
        write_exclusive(self.root/'qualification_status'/f'{self.exp}.json', result); self.commit()


def audit_streams(root, payload):
    validate_payload(payload); root = Path(root); directory = root/'raw_transcripts'/payload['experiment']
    if json.loads((directory/'frozen_input.json').read_text()) != payload: raise ValueError('Frozen input changed')
    claim = json.loads((directory/'attempt_claim.json').read_text())
    if claim['payload_sha256'] != payload['payload_sha256']: raise ValueError('Writer claim changed')
    call_binding = durable_call(root, payload)
    if any(claim[key] != value for key, value in call_binding.items()): raise ValueError('Writer claim call binding changed')
    inventory_path = directory/'loaded_model_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory['identity'] != payload['identity'] or inventory['payload_sha256'] != payload['payload_sha256']: raise ValueError('Loaded model inventory identity differs')
    validate_inventory(inventory['inventory'], payload['identity']['family'])
    raw_path = directory/f'{payload["model_role"]}.jsonl'; journal_path = directory/f'{payload["model_role"]}_requests.jsonl'
    return_path = root/'structured_artifacts'/payload['experiment']/'model_returns.jsonl'
    rows, head = read_chain(raw_path, payload['payload_sha256']); journals, jhead = read_chain(journal_path, payload['payload_sha256'])
    returned, rhead = read_chain(return_path, payload['payload_sha256'])
    expected = payload['pilots']+payload['requests']; sequence = [t['trial_id'] for t in expected]
    for stream in (rows, journals):
        if [r['trial_id'] for r in stream] != sequence[:len(stream)]: raise ValueError('Missing, repeated, reordered or selected request')
    expected_map = {r['trial_id']:r for r in expected}; journal_map = {r['trial_id']:r for r in journals}
    returned_map = {r['chain_sha256']:r for r in returned}
    returned_sequence = [key for batch in returned for key in batch['trial_ids']]
    if returned_sequence != [r['trial_id'] for r in journals][:len(returned_sequence)]: raise ValueError('Returned batches repeat or reorder requests')
    return_shapes_valid = all(len(batch['outputs']) == len(batch['trial_ids']) and
                             all(len(output['choices']) == 1 for output in batch['outputs']) for batch in returned)
    for row in rows:
        request = expected_map[row['trial_id']]; written = journal_map[row['trial_id']]
        for key, value in request.items():
            if row[key] != value or written[key] != value: raise ValueError('Request differed between input, writeahead and raw')
        for key, value in payload['identity'].items():
            if row[key] != value or written[key] != value: raise ValueError('Pinned model identity changed')
        if row['request_record_sha256'] != written['chain_sha256']: raise ValueError('Writeahead link changed')
        batch = returned_map[row['generation_return_record_sha256']]; index = row['return_index']
        if batch['trial_ids'][index] != row['trial_id'] or batch['request_record_sha256'][index] != written['chain_sha256']: raise ValueError('Returned request binding changed')
        result = batch['outputs'][index]
        if result != {'prompt_token_ids': row['prompt_token_ids'], 'choices': [{'raw_output': row['raw_output'], 'token_ids': row['generated_token_ids'], 'finish_reason': row['finish_reason']}]}: raise ValueError('Raw record differs from pre-parser returned output')
        for key in ('full_prompt', 'rendered_messages', 'prompt_token_ids', 'prompt_token_ids_sha256', 'request_sha256', 'payload_sha256', 'tokenizer_file_sha256'):
            if row[key] != written[key]: raise ValueError('Prompt/input provenance changed')
        if row['tokenizer_file_sha256']!=payload['tokenizer_files']:raise ValueError('Structured tokenizer binding changed')
        if payload['identity']['family'] == 'llama':
            from llama_model_loader import rendered_messages
            expected_messages = rendered_messages(request)
        else: expected_messages = request['messages']
        if row['rendered_messages'] != expected_messages: raise ValueError('Model-specific prompt rendering changed')
        if (row['prompt_token_ids_sha256'] != digest(row['prompt_token_ids']) or row['request_sha256'] != digest(request) or
                row['payload_sha256'] != payload['payload_sha256'] or row['generated_tokens'] != len(row['generated_token_ids']) or
                row['raw_source_path'] != str(raw_path.relative_to(root))): raise ValueError('Raw hashes/counts/source differ')
        for key, value in parsed_record(row['raw_output'], row['generated_token_ids'], row['finish_reason'], request).items():
            if row[key] != value: raise ValueError('Independent raw reparse differs')
    pilot, full = rows[:5], rows[5:]
    h5 = len(pilot) == 5 and all(r['record_valid'] and r['independent_parser_agreement'] and
             (r['cot_mode'] == 'cot_off' or r['think_closed']) for r in pilot)
    if full and not h5: raise ValueError('Full record sweep began before its own H5 passed')
    if full:
        markdown=(root/'pilots'/f'{payload["experiment"]}_pilot_audit.md').read_text()
        if re.findall(r'^Status: (PASS|FAIL)$',markdown,re.M)!=['PASS'] or [json.loads(v) for v in re.findall(r'^```json\n(.*?)\n```$',markdown,re.M|re.S)]!=pilot:
            raise ValueError('Structured pilot Markdown differs from raw evidence')
    capped = sum(r['finish_reason'] == 'length' or r['generated_tokens'] >= r['max_new_tokens'] for r in full)
    truncated = sum(r['finish_reason'] == 'length' or r['generated_tokens'] >= r['max_new_tokens'] or r['parse_status']=='TRUNCATED_IN_THINK' for r in full)
    evidence_complete = len(rows) == len(journals) and returned_sequence == sequence[:len(rows)] and return_shapes_valid
    complete = len(full) == len(payload['requests']) and evidence_complete
    return {'schema': 'structured_records_stream_audit_v1', 'pilot_n': len(pilot), 'full_n': len(full), 'expected_n': len(payload['requests']),
            'complete_data': complete, 'capped_n': capped, 'truncated_n':truncated, 'H1': complete and truncated/len(payload['requests']) < .02,
            'H2': all(r['independent_parser_agreement'] for r in rows), 'H4': evidence_complete, 'H5': h5,
            'all_records_valid': complete and all(r['record_valid'] for r in full),
            'pending_writeahead_trial_ids': [r['trial_id'] for r in journals[len(rows):]],
            'raw_sha256': hash_bytes(raw_path), 'writeahead_sha256': hash_bytes(journal_path), 'model_returns_sha256': hash_bytes(return_path),
            'loaded_model_inventory_sha256': hash_bytes(inventory_path),
            'raw_chain_head': head, 'writeahead_chain_head': jhead, 'model_returns_chain_head': rhead,
            'payload_sha256': payload['payload_sha256'], 'semantic_acceptance_inferred': False}


def verify_sources(payload, code, root):
    for filename, expected in payload['source_hashes'].items():
        if hash_bytes(Path(code)/filename) != expected: raise ValueError('Runtime source hash changed: '+filename)
    for filename, expected in payload['input_sources'].items():
        if hash_bytes(Path(root)/filename) != expected: raise ValueError('Frozen prerequisite/input hash changed: '+filename)
    if json.loads((Path(root)/payload['request_source_path']).read_text()) != payload['requests']:
        raise ValueError('Exact request source bytes no longer describe the payload requests')
    for name,h in payload['tokenizer_files'].items():
        if hash_bytes(Path(payload['identity']['model_path'])/name)!=h:raise ValueError('Model tokenizer differs from exact pinned input metadata')
    if payload['identity']['family'] == 'primary':
        from primary_model_loader import checkpoint_manifest
        path = Path(payload['identity']['model_path']); manifest = checkpoint_manifest(path, root)
        manifest_path = Path(root)/path.parent.name/f'{path.name}_materialization.json'
        if hash_bytes(manifest_path) != payload['identity']['model_revision']: raise ValueError('Role checkpoint manifest changed')
        actual_shards = {p.name for p in path.glob('*.safetensors')}
        indexed_shards = set(json.loads((path/'model.safetensors.index.json').read_text())['weight_map'].values())
        if actual_shards != indexed_shards or actual_shards != set(manifest['shard_sha256']): raise ValueError('Checkpoint shard coverage mismatch')
        for filename, expected in {**manifest['shard_sha256'], **manifest.get('support_file_sha256', {})}.items():
            if Path(filename).name != filename: raise ValueError('Checkpoint manifest escapes model directory')
            with (path/filename).open('rb') as handle: actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != expected: raise ValueError('Checkpoint bytes changed')
    else:
        provenance = payload['model_provenance']
        for filename in ('download_manifest', 'preflight'):
            value = json.loads((Path(root)/provenance[filename]).read_text())
            if value['repo_id'] != LLAMA_REPO or value['revision'] != LLAMA_REVISION: raise ValueError('Llama pinned source provenance mismatch')
            if filename == 'download_manifest' and value['volume_path'] != payload['identity']['model_path']:
                raise ValueError('Llama download path differs from pinned namespace')


def main(payload_path):
    payload = json.loads(Path(payload_path).read_text()); validate_payload(payload)
    root = Path('/artifacts')/NS
    verify_sources(payload, Path(__file__).parent, root)
    def commit():
        print('COMMIT_ARTIFACTS', flush=True)
        if sys.stdin.readline().strip() != 'COMMITTED': raise RuntimeError('Durable commit not acknowledged')
    run = RecordRun(root, payload, commit)
    try:
        captured=run.directory/'tokenizer';captured.mkdir(exist_ok=True)
        for name in payload['tokenizer_files']:
            with (captured/name).open('xb') as handle:handle.write((Path(payload['identity']['model_path'])/name).read_bytes())
        commit()
        from vllm import SamplingParams
        if payload['identity']['family'] == 'primary':
            from primary_model_loader import load_primary
            model, inventory = load_primary(payload['identity']['model_path'], root)
        else:
            from llama_model_loader import load_llama
            model, inventory = load_llama(payload['identity']['model_path'])
        result = run.execute(model, lambda cap: SamplingParams(temperature=0.0, max_tokens=cap, seed=0, skip_special_tokens=False), inventory)
        print(json.dumps({'status': result['status'], 'experiment': payload['experiment']}), flush=True)
    except BaseException as exc:
        if not (root/'qualification_status'/f'{payload["experiment"]}.json').exists():
            run.save_status({'schema': 'structured_records_status_v1', 'status': 'WORKER_FAILED', 'error_type': type(exc).__name__,
                'experiment': payload['experiment'], 'payload_sha256': payload['payload_sha256'], 'finished_at_utc': utc_now(),
                'phase2_executor_launch': False})
        raise


if __name__ == '__main__':
    main(sys.argv[1])
