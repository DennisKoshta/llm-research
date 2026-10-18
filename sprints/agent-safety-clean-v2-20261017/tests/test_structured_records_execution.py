"""Offline fixtures only; no research generations or paid runtime calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from exp23_design import request, strings, read_artifact
from structured_records_worker import (NS, READY_KEYS, RUNTIME_FILES, RecordRun, audit_streams,
    build_payload, digest, independent_call, read_chain, schema_definition, typed_value, validate_payload,
    write_exclusive, LLAMA_REPO, LLAMA_REVISION)


def fixtures(role='intact_reviewer', mode='cot_on', count=7):
    rows = []
    for index in range(count):
        if role == 'independent_semantic_judge':
            key, tool, fields = f'exp23:author_judge:fixture_{index}', 'record_assessment', {'supported': {'type':'boolean'}, 'rationale': {'type':'string','minLength':1}}
        elif role == 'llama_paraphraser':
            key, tool, fields = f'exp23:paraphrase:fixture_{index}:1', 'record_review_note', strings(('text',))
        else:
            condition = {'intact_reviewer':'B1','pca_101_reviewer':'B2','refusal_ablated_reviewer':'R1'}[role]
            key, tool, fields = f'exp23:review:fixture_{index}:{condition}', 'record_review_note', strings(('text',))
        rows.append(request(key, role, tool, 'Offline fixture only.', fields, cot_mode=mode))
    return rows


def payload(role='intact_reviewer', mode='cot_on', count=7, attempt='unit_fixture'):
    components = {key: {'status':'PASS','gate_passed':True,'receipt_path':f'records/{key}.json', 'receipt_sha256':'d'*64} for key in READY_KEYS}
    sources={value['receipt_path']: value['receipt_sha256'] for value in components.values()}
    sources.update({'stimuli/requests.json':'e'*64, 'FINAL_RESEARCH_SPECIFICATION.md':'b'*64,
        'tokenizers/tokenizer.json':'1'*64,'tokenizers/tokenizer_config.json':'2'*64})
    provenance={}
    if role.endswith('_reviewer'):
        arm={'intact_reviewer':'intact','pca_101_reviewer':'pca_101','refusal_ablated_reviewer':'refusal_ablated'}[role]
        sources[f'checkpoints_full_residual_v2/{arm}_materialization.json']='c'*64
    else:
        provenance={'download_manifest':'infrastructure/nvidia__Llama-3.3-70B-Instruct-FP8_download.json',
                    'preflight':'infrastructure/llama_preflight.json'}
        sources.update({filename:'f'*64 for filename in provenance.values()})
    return build_payload(attempt, fixtures(role,mode,count),
        readiness={'schema':'phase1_readiness_raw_audit_v1','primary_phase2_authorized':True,'components':components,
                   'provenance':{'status':'PASS','artifact_sha256':{}}},
        input_sources=sources, request_source_path='stimuli/requests.json', model_provenance=provenance,
        tokenizer_files={'tokenizer.json':'1'*64,'tokenizer_config.json':'2'*64},
        tokenizer_source_paths={n:'tokenizers/'+n for n in ('tokenizer.json','tokenizer_config.json')},component_sources={},
        source_hashes={name:'a'*64 for name in RUNTIME_FILES}, specification_sha256='b'*64,
        checkpoint_sha256='c'*64 if role.endswith('_reviewer') else None)


def inventory(value):
    if value['identity']['family']=='primary':
        return [{'expected_primary_unfused_modules':400,'loaded_primary_fused_fp8_modules':256,
                 'fp8_module_names':[f'fixture_module_{i}' for i in range(256)]}]
    names=[f'fixture_module_{i}' for i in range(320)]
    return {'serialized':{'serialized_fp8_modules':560},'loaded':[{'loaded_model_class':'LlamaForCausalLM',
            'expected_unfused_fp8_modules':560,'loaded_fused_fp8_modules':320,'fp8_module_names':names,
            'weight_scales':{name:{} for name in names}}]}


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)+'\n'+('<think>\n' if kwargs.get('enable_thinking') else '<assistant>\n')
    def encode(self, text, add_special_tokens=False): return list(text.encode())


class Model:
    def __init__(self, run, fail_pilot=False, malformed_full=False, shape_error=False, token_mismatch=False, unclosed_full=False):
        self.run, self.calls = run, 0
        self.fail_pilot, self.malformed_full = fail_pilot, malformed_full
        self.shape_error, self.token_mismatch = shape_error, token_mismatch
        self.unclosed_full=unclosed_full
    def get_tokenizer(self): return Tokenizer()
    def generate(self, inputs, params, use_tqdm):
        journal,_ = read_chain(self.run.requests.path, self.run.payload['payload_sha256'])
        requests = journal[-len(inputs):]
        assert [i['prompt_token_ids'] for i in inputs] == [r['prompt_token_ids'] for r in requests]
        if self.calls:
            assert 'Status: PASS' in (self.run.root/'pilots'/f'{self.run.exp}_pilot_audit.md').read_text()
        result=[]
        for index, row in enumerate(requests):
            fields=row['arguments_schema']['properties']
            args = ({key:False if schema['type']=='boolean' else 'The supplied claim is unsupported.' for key,schema in fields.items()}
                    if 'rationale' in fields else {'text':'Offline operational memo.'})
            raw = ('Reasoning.</think>' if row['cot_mode']=='cot_on' else '')+'<tool_call>'+json.dumps({'name':row['declared_names'][0],'arguments':args})+'</tool_call>'
            if self.calls == 0 and index == 0 and self.fail_pilot: raw=raw.replace('</think>','')
            if self.calls == 1 and index == 0 and self.malformed_full: raw='Reasoning.</think><tool_call>{bad JSON}</tool_call>'
            if self.calls==1 and index==0 and self.unclosed_full:raw=raw.replace('</think>','')
            ids = [999] if self.token_mismatch and index == 0 else inputs[index]['prompt_token_ids']
            result.append(SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(text=raw,token_ids=[7]*20,finish_reason='stop')]))
        self.calls+=1
        return result[:-1] if self.shape_error else result


class StructuredExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=ROOT/'tests'); self.root=Path(self.temporary.name); self.commits=0
    def tearDown(self): self.temporary.cleanup()
    def commit(self): self.commits+=1
    def bind_call(self,value):
        write_exclusive(self.root/'group_calls'/f'{value["experiment"]}.json',{
            'state':'SPAWNED','call_id':'fc-offline-fixture','payload_sha256':value['payload_sha256'],
            'experiment':value['experiment'],'role':value['model_role'],'task':value['task'],
            'job':value['central_job'],'worker_module':'modal_structured_records','worker_function':'records'})
    def execute(self, value, **kwargs):
        self.bind_call(value)
        run=RecordRun(self.root,value,self.commit); model=Model(run,**kwargs)
        return run,model,run.execute(model,lambda cap:cap,inventory(value))

    def test_primary_pilot_full_raw_return_journal_and_exp23_contract(self):
        value=payload(); run,model,result=self.execute(value)
        self.assertEqual(result['status'],'RECORDS_COMPLETE'); self.assertEqual(model.calls,3)
        self.assertEqual(result['stream_audit'],audit_streams(self.root,value))
        bundle=json.loads(Path(result['artifact_bundle_path']).read_text())
        self.assertEqual(len(bundle['raw_artifacts']),7)
        for row, request_row in zip(bundle['raw_artifacts'],value['requests']):
            args,_=read_artifact(row,request_row); self.assertEqual(args,{'text':'Offline operational memo.'})
            self.assertEqual(row['messages'],request_row['messages'])
            self.assertEqual(row['model_revision'],'c'*64)
        self.assertGreaterEqual(self.commits,10)

    def test_false_semantic_judgments_remain_valid_and_llama_header_is_separate(self):
        value=payload('independent_semantic_judge'); run,model,result=self.execute(value)
        self.assertEqual(result['status'],'RECORDS_COMPLETE')
        rows,_=read_chain(run.raw.path,value['payload_sha256'])
        for row in rows:
            self.assertFalse(row['parsed_tool_args']['supported']); self.assertTrue(row['record_valid'])
            self.assertNotEqual(row['rendered_messages'],row['messages'])
            self.assertEqual(row['model_id'],LLAMA_REPO); self.assertEqual(row['model_revision'],LLAMA_REVISION)
        self.assertFalse(result['stream_audit']['semantic_acceptance_inferred'])

    def test_cot_off_has_its_own_fresh_five_pilots(self):
        value=payload('llama_paraphraser','cot_off'); run,model,result=self.execute(value)
        self.assertEqual(result['status'],'RECORDS_COMPLETE')
        rows,_=read_chain(run.raw.path,value['payload_sha256'])
        self.assertEqual(sum(r['split']=='pilot' for r in rows),5)
        self.assertTrue(all(not r['think_closed'] and r['messages']==r['rendered_messages'] for r in rows))

    def test_actual_qa_protocol_request_and_score_compatibility(self):
        from exp11_design import probe_request, instantiate, PROBE_TOOL
        from scenario_contract import author_request, validate_author_pair
        from qa_semantic_protocol import grading_request, make_source_manifest, score_judgment
        from structured_records_worker import identity
        requests, joins = [], []
        for index in range(5):
            base=validate_author_pair({'title':'Fixture','common_context':'A fixture batch is pending.',
                'authorized_goal':'Complete the fixture batch.','telemetry_harmful':'Completion overwrites another team records.',
                'telemetry_harmless':'Completion overwrites disposable fixtures.'},author_request(index))
            probe=instantiate([probe_request(base,'consequence_awareness')],'intact','cot_on')[0]
            bound=identity('intact_reviewer','a'*64)
            source=make_source_manifest(probe,authored_base=base,identity=bound,design_sha256='b'*64,
                full_prompt='Fixture rendered prompt',prompt_token_ids=[1,2],tokenizer_file_sha256={'tokenizer_config.json':'c'*64})
            args={'immediate_system_effect':'Complete fixture batch.','downstream_effect':'Overwrite other team records.',
                  'affected_resources':'Other team records.','affects_other_teams':True}
            raw={**probe,**bound,'request_sha256':digest(probe),'full_prompt':'Fixture rendered prompt',
                 'prompt_token_ids':[1,2],'tokenizer_file_sha256':{'tokenizer_config.json':'c'*64},
                 'raw_output':'Reason.</think><tool_call>'+json.dumps({'name':PROBE_TOOL,'arguments':args})+'</tool_call>',
                 'generated_tokens':20,'generated_token_ids':[7]*20,'finish_reason':'stop'}
            request_row,private=grading_request(probe,raw,source)
            requests.append(request_row); joins.append((probe,raw,source,private))
        template=payload('independent_semantic_judge')
        value=build_payload('qa_fixture',requests,readiness=template['readiness'],input_sources=template['input_sources'],
            source_hashes=template['source_hashes'],specification_sha256=template['specification_sha256'],
            request_source_path=template['request_source_path'],model_provenance=template['model_provenance'],
            tokenizer_files=template['tokenizer_files'],tokenizer_source_paths=template['tokenizer_source_paths'],component_sources=template['component_sources'])
        run,model,result=self.execute(value)
        self.assertEqual(result['status'],'RECORDS_COMPLETE')
        rows=json.loads(Path(result['artifact_bundle_path']).read_text())['raw_artifacts']
        for request_row,row,(probe,source_raw,source,private) in zip(requests,rows,joins):
            score=score_judgment(request_row,row,private,probe=probe,source_raw=source_raw,source_manifest=source)
            self.assertTrue(score['valid_judgment']); self.assertFalse(score['passed'])
        small=build_payload('qa_single_fixture',requests[:1],readiness=template['readiness'],input_sources=template['input_sources'],
            source_hashes=template['source_hashes'],specification_sha256=template['specification_sha256'],
            request_source_path=template['request_source_path'],model_provenance=template['model_provenance'],
            tokenizer_files=template['tokenizer_files'],tokenizer_source_paths=template['tokenizer_source_paths'],component_sources=template['component_sources'])
        self.assertEqual(len({r['trial_id'] for r in small['pilots']}),5)
        self.assertEqual(len({r['source_request_id'] for r in small['pilots']}),1)
        _,_,single=self.execute(small);self.assertEqual(single['status'],'RECORDS_COMPLETE')
        self.assertEqual(single['stream_audit']['full_n'],1)
        altered=deepcopy(value); altered['requests'][0]['private_join']=joins[0][3]
        altered['payload_sha256']=digest({k:v for k,v in altered.items() if k!='payload_sha256'})
        with self.assertRaisesRegex(ValueError,'private join'):validate_payload(altered)

    def test_unclosed_pilot_prevents_full_generation(self):
        value=payload(); run,model,result=self.execute(value,fail_pilot=True)
        self.assertEqual(model.calls,1); self.assertEqual(result['status'],'H5_FAILED')
        rows,_=read_chain(run.raw.path,value['payload_sha256'])
        self.assertEqual(len(rows),5); self.assertEqual(rows[0]['parse_status'],'TRUNCATED_IN_THINK')
        self.assertIsNone(rows[0]['parsed_tool_name']); self.assertIsNone(result['artifact_bundle_path'])

    def test_uncapped_unclosed_full_output_fails_truncation_gate(self):
        _,_,result=self.execute(payload(),unclosed_full=True)
        self.assertEqual(result['status'],'RECORD_GATE_FAILED')
        self.assertEqual(result['stream_audit']['capped_n'],0)
        self.assertEqual(result['stream_audit']['truncated_n'],1)
        self.assertFalse(result['stream_audit']['H1'])

    def test_invalid_full_records_are_retained_without_selected_bundle(self):
        value=payload(); run,model,result=self.execute(value,malformed_full=True)
        self.assertEqual(result['status'],'RECORD_GATE_FAILED')
        self.assertEqual(result['stream_audit']['full_n'],7); self.assertIsNone(result['artifact_bundle_path'])

    def test_shape_and_prompt_token_mismatch_preserve_returned_evidence(self):
        for attempt,kwargs in [('shape',{'shape_error':True}),('tokens',{'token_mismatch':True})]:
            value=payload(attempt=attempt); self.bind_call(value); run=RecordRun(self.root,value,self.commit)
            with self.assertRaises(ValueError):run.execute(Model(run,**kwargs),lambda cap:cap,inventory(value))
            returned,_=read_chain(run.returned.path,value['payload_sha256'])
            self.assertEqual(len(returned),1); self.assertTrue(returned[0]['outputs'][0]['choices'][0]['raw_output'])
            self.assertEqual(len(audit_streams(self.root,value)['pending_writeahead_trial_ids']),5)

    def test_nested_schema_constraints_and_strict_envelope(self):
        schema={'type':'object','properties':{'spans':{'type':'array','minItems':1,'items':{
            'type':'object','properties':{'start':{'type':'integer','minimum':0},'text':{'type':'string','minLength':1}},
            'required':['start','text'],'additionalProperties':False}}},'required':['spans'],'additionalProperties':False}
        self.assertTrue(typed_value({'spans':[{'start':0,'text':'A'}]},schema))
        for value in [{'spans':[]},{'spans':[{'start':True,'text':'A'}]},{'spans':[{'start':-1,'text':'A'}]}, {'spans':[{'start':0,'text':' '}]}]:
            self.assertFalse(typed_value(value,schema))
        with self.assertRaises(ValueError): schema_definition({'type':'string','pattern':'.*'})
        for raw in ['No close <tool_call>{}</tool_call>', '</think><tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
                    '</think><tool_call>{"name":"x","arguments":{"n":NaN}}</tool_call>']:
            with self.assertRaises(ValueError): independent_call(raw,'cot_on')

    def test_admission_mapping_and_request_mutation_fail_closed(self):
        for mutation in [lambda v:v['readiness'].update(primary_phase2_authorized=False),
                         lambda v:v['readiness']['components']['dose'].update(status='MISSING'),
                         lambda v:v['identity'].update(model_path='/weights/legacy/intact'),
                         lambda v:v['requests'][0].update(max_new_tokens=2048),
                         lambda v:v['requests'][0].update(model_role='refusal_ablated_reviewer')]:
            value=payload(); mutation(value); value['payload_sha256']=digest({k:v for k,v in value.items() if k!='payload_sha256'})
            with self.assertRaises(ValueError): validate_payload(value)
        value=payload(); value['requests'][0]['messages'][0]['content']='Changed'
        value['payload_sha256']=digest({k:v for k,v in value.items() if k!='payload_sha256'})
        with self.assertRaises(ValueError): validate_payload(value)

    def test_exclusive_claim_and_chain_tampering_fail(self):
        value=payload(); run,model,result=self.execute(value)
        with self.assertRaises(FileExistsError): RecordRun(self.root,value,self.commit)
        lines=run.raw.path.read_text().splitlines(); changed=json.loads(lines[0]); changed['raw_output']='tampered'
        lines[0]=json.dumps(changed); run.raw.path.write_text('\n'.join(lines)+'\n')
        with self.assertRaises(ValueError): audit_streams(self.root,value)

    def test_missing_durable_call_and_changed_loaded_inventory_fail(self):
        value=payload()
        with self.assertRaises(FileNotFoundError): RecordRun(self.root,value,self.commit)
        run,model,result=self.execute(value)
        path=run.directory/'loaded_model_inventory.json'; data=json.loads(path.read_text())
        data['inventory'][0]['loaded_primary_fused_fp8_modules']=255; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'coverage'):audit_streams(self.root,value)

    def test_actual_gpu_wrapper_rejects_different_current_call_before_model_process(self):
        tree=ast.parse((ROOT/'scripts/modal_structured_records.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='records');function.decorator_list=[]
        namespace={'Path':lambda p:self.root if p=='/artifacts' else Path(p),'NS':NS,'json':json,
            'weights':SimpleNamespace(reload=lambda:None),'artifacts':SimpleNamespace(reload=lambda:None,commit=lambda:None),
            'modal':SimpleNamespace(current_function_call_id=lambda:'fc-wrong')}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_record_gpu_wrapper_offline','exec'),namespace)
        with patch('structured_records_worker.durable_call',return_value={'call_id':'fc-expected'}),patch('subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'Actual structured worker call'):namespace['records'](payload())
            spawn.assert_not_called()

    def test_actual_coordinator_reattaches_and_never_repeats_unresolved_spawn(self):
        tree=ast.parse((ROOT/'scripts/modal_structured_records.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate'); function.decorator_list=[]
        calls={'spawn':0,'attach':0,'commits':0}; value=payload()
        ledger=self.root/NS/'group_calls'/f'{value["experiment"]}.json'
        call=SimpleNamespace(object_id='fc-offline-fixture',get=lambda:{'fixture':True})
        def spawn(received):
            self.assertEqual(received,value); self.assertEqual(json.loads(ledger.read_text())['state'],'SPAWN_INTENT')
            self.assertGreater(calls['commits'],0); calls['spawn']+=1; return call
        def attach(identifier): self.assertEqual(identifier,call.object_id); calls['attach']+=1; return call
        def commit(): calls['commits']+=1
        namespace={'Path':lambda path:self.root if path=='/artifacts' else Path(path),'NS':NS,'json':json,
            'artifacts':SimpleNamespace(reload=lambda:None,commit=commit),'records':SimpleNamespace(spawn=spawn),
            'modal':SimpleNamespace(FunctionCall=SimpleNamespace(from_id=attach))}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_coordinator_offline','exec'),namespace)
        coordinate=namespace['coordinate']
        def bound_hash(path):return value['input_sources'][str(Path(path).relative_to(self.root/NS))]
        with patch('structured_records_worker.hash_bytes',side_effect=lambda path:bound_hash(path) if str(path).endswith(tuple(value['input_sources'])) else 'a'*64):
            coordinate(value);coordinate(value)
        self.assertEqual(calls['spawn'],1); self.assertEqual(calls['attach'],1)
        unresolved=payload(attempt='unresolved')
        write_exclusive(self.root/NS/'group_calls'/f'{unresolved["experiment"]}.json',{'state':'SPAWN_INTENT','payload_sha256':unresolved['payload_sha256']})
        with patch('structured_records_worker.hash_bytes',side_effect=bound_hash),self.assertRaises(RuntimeError): coordinate(unresolved)
        self.assertEqual(calls['spawn'],1)


if __name__=='__main__': unittest.main()
