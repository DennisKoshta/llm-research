"""Synthetic backend/admission fixtures; never monitor observations or live calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from exp22_design import build_design,sweeps,digest,ROLES
from exp22_monitor_worker import (build_payload,validate_payload,MonitorRun,audit_stage,paths,read_chain,hash_bytes,
                                  RUNTIME_FILES,write_exclusive,strip_chain)
import exp22_monitor_evidence as evidence
import modal_exp22_monitor as wrapper
from test_exp22_design import fixture_bundle
from test_exp22_evaluation import Tokenizer


def inventory(family='primary'):
    if family=='primary':return [{'expected_primary_unfused_modules':400,'loaded_primary_fused_fp8_modules':256,
        'fp8_module_names':['fixture_'+str(i) for i in range(256)]}]
    names=['fixture_'+str(i) for i in range(320)]
    return {'serialized':{'serialized_fp8_modules':560},'loaded':[{'loaded_model_class':'LlamaForCausalLM',
        'expected_unfused_fp8_modules':560,'loaded_fused_fp8_modules':320,'fp8_module_names':names,'weight_scales':dict.fromkeys(names,1)}]}


class Model:
    def __init__(self,run,unclosed=0,shape=False):self.run=run;self.calls=0;self.unclosed=unclosed;self.shape=shape
    def get_tokenizer(self):return Tokenizer()
    def generate(self,inputs,parameters,use_tqdm):
        rows,_=read_chain(self.run.p['journal'],self.run.value['payload_sha256']);rows=rows[-len(inputs):]
        assert parameters==4096 and len(inputs)<=5
        result=[]
        for i,(row,inputs) in enumerate(zip(rows,inputs)):
            assert row['prompt_token_ids']==inputs['prompt_token_ids']
            request=row['request'];label=next(iter(request['label_mapping']))
            text='Reason.</think>VERDICT: '+label
            if self.calls==0 and i<self.unclosed:text=text.replace('</think>','')
            result.append(SimpleNamespace(prompt_token_ids=inputs['prompt_token_ids'],outputs=[
                SimpleNamespace(text=text,token_ids=[1,2,3],finish_reason='stop')]))
        self.calls+=1
        return result+result[:1] if self.shape else result


class MonitorExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=fixture_bundle()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for name in {*RUNTIME_FILES,'modal_exp22_monitor.py','historical_admission.py'}:
            path=self.root/'scripts'/name;path.parent.mkdir(exist_ok=True);path.write_bytes((ROOT/'scripts'/name).read_bytes())
        self.put('FINAL_RESEARCH_SPECIFICATION.md',{'fixture_only':True})
        self.put('fixture/bundle.json',self.bundle);self.put('fixture/producers.json',{'fixture_only':True})
        self.put('fixture/tokenizers/tokenizer.json',{});self.put('fixture/tokenizers/tokenizer_config.json',{})
        self.put('checkpoints_full_residual_v2/intact_materialization.json',{'fixture_only':True})
        self.design=build_design(self.bundle,'fixture/bundle.json',hash_bytes(self.root/'fixture/bundle.json'))
        self.inputs={'phase1_components':'fixture/phase1.json','exp11_selection':'fixture/exp11.json',
            'exp11_sampler_invocation':'fixture/invocation.json','exp23_full_calls':['fixture/full.json']}
        for name in ('fixture/phase1.json','fixture/exp11.json','fixture/invocation.json','fixture/full.json'):self.put(name,{})
        self.gate={'schema':'phase3_readiness_raw_audit_v1','state':'PASS','phase3_authorized':True,
            'complete_phase2_primary_scope':True,'omnibus_replay_verified':True,'post_sampler_upstream_reaudited_exactly':True,
            'inputs':self.inputs,'input_sha256':{p:hash_bytes(self.root/p) for p in ('fixture/phase1.json','fixture/exp11.json','fixture/invocation.json','fixture/full.json')},
            'specification_sha256':hash_bytes(self.root/'FINAL_RESEARCH_SPECIFICATION.md')}
        source={p:hash_bytes(self.root/p) for p in ('fixture/bundle.json','fixture/producers.json','FINAL_RESEARCH_SPECIFICATION.md',
            'fixture/tokenizers/tokenizer.json','fixture/tokenizers/tokenizer_config.json','checkpoints_full_residual_v2/intact_materialization.json')}
        source.update(self.gate['input_sha256'])
        self.dependency={'schema':'exp22_monitor_dependencies_v1','bundle_path':'fixture/bundle.json','bundle_sha256':self.design['source_sha256'],
            'producer_selection_path':'fixture/producers.json','design_sha256':digest(self.design),'phase3_admission_sha256':digest(self.gate),
            'source_sha256':{'fixture/bundle.json':self.design['source_sha256']}}
        self.payload=build_payload('fixture',self.design,sweeps(self.design)[0]['sweep_id'],phase3_inputs=self.inputs,phase3_audit=self.gate,
            dependency_audit=self.dependency,input_sources=source,source_hashes={n:hash_bytes(self.root/'scripts'/n) for n in RUNTIME_FILES},
            tokenizer_files={n:hash_bytes(self.root/'fixture/tokenizers'/n) for n in ('tokenizer.json','tokenizer_config.json')},
            tokenizer_source_paths={n:'fixture/tokenizers/'+n for n in ('tokenizer.json','tokenizer_config.json')},
            checkpoint_sha256=source['checkpoints_full_residual_v2/intact_materialization.json'])
    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def execute(self,stage,**kwargs):
        commits=[];run=MonitorRun(self.root,self.payload,stage,lambda:commits.append(True))
        folder=run.p['folder']/'tokenizer';folder.mkdir()
        for name in self.payload['tokenizer_files']:(folder/name).write_bytes((self.root/'fixture/tokenizers'/name).read_bytes())
        model=Model(run,**kwargs);status=run.execute(model,lambda cap:cap,inventory(self.payload['identity']['family']))
        return run,status,commits
    def test_complete_pilot_and_full_save_exact_return_journal_and_prompt_evidence(self):
        _,pilot,_=self.execute('pilot');_,full,commits=self.execute('full')
        self.assertEqual(pilot['status'],'PILOT_PASS');self.assertEqual(full['status'],'FULL_COMPLETE')
        self.assertEqual(full['audit']['trials'],96);self.assertGreater(len(commits),20)
        audit,status=evidence.audit_pilot_completion(self.root,self.payload,tokenizer=Tokenizer())
        self.assertEqual(audit,pilot['audit']);self.assertEqual(status,pilot)
        self.assertFalse(full['audit']['execution_provenance_verified'])
    def test_missing_closure_stop_below_cap_counts_truncation_and_fails_h5(self):
        _,pilot,_=self.execute('pilot',unclosed=1);_,full,_=self.execute('full',unclosed=2)
        self.assertEqual(pilot['status'],'H5_FAILED');self.assertEqual(full['audit']['truncation_count'],2)
        self.assertFalse(full['audit']['H1']);self.assertEqual(full['status'],'FULL_H1_FAILED')
        with self.assertRaisesRegex(ValueError,'Five independent raw pilots'):evidence.audit_pilot_completion(self.root,self.payload,tokenizer=Tokenizer())
    def test_preparser_extra_model_return_is_saved_before_shape_rejection(self):
        run=MonitorRun(self.root,self.payload,'pilot',lambda:None)
        with self.assertRaisesRegex(ValueError,'return shape'):run.execute(Model(run,shape=True),lambda c:c,inventory())
        returned,_=read_chain(run.p['returns'],self.payload['payload_sha256'])
        self.assertEqual(len(returned[0]['outputs']),6);self.assertEqual(read_chain(run.p['raw'],self.payload['payload_sha256'])[0],[])
    def test_tokenizer_and_complete_source_metadata_tampering_fail(self):
        run,_,_=self.execute('pilot');(run.p['folder']/'tokenizer/tokenizer.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Captured tokenizer'):audit_stage(self.root,self.payload,'pilot',tokenizer=Tokenizer())
        changed=deepcopy(self.payload);changed['phase3_audit']['phase3_authorized']=False
        with self.assertRaises(ValueError):validate_payload(changed)
        changed=deepcopy(self.payload);changed['sweep']['full_requests']=changed['sweep']['full_requests'][:-1]
        with self.assertRaisesRegex(ValueError,'Selective'):validate_payload(changed)
    def test_missing_phase3_aware_producer_backend_explicitly_blocks_admission(self):
        with self.assertRaisesRegex(RuntimeError,'BLOCKED_EXP22_PRODUCER_PROVENANCE'):
            evidence.dependency_bundle(self.root,'fixture/bundle.json','fixture/producers.json',self.inputs,self.gate)
    def test_preparation_requires_fresh_global_gate_before_any_producer_work(self):
        with patch('phase3_readiness.recompute',return_value={'phase3_authorized':False}),patch.object(evidence,'dependency_bundle') as deps:
            with self.assertRaisesRegex(ValueError,'fresh Phase3 gate'):
                wrapper.prepare(self.root,'fixture',sweep_id=self.payload['sweep']['sweep_id'],bundle_path='fixture/bundle.json',
                    producer_selection_path='fixture/producers.json',phase3_inputs=self.inputs,tokenizer_metadata='fixture/tokenizers')
            deps.assert_not_called()
    def central(self):
        input_path=f'stimuli/{self.payload["experiment"]}_input.json';self.put(input_path,self.payload)
        config=wrapper.resource_config(self.root,'fixture',input_path)
        job={**config,'app_id':'ap-fixture','admitted_upper_bound_usd':config['upper_bound_usd']}
        self.put('infrastructure/resource_allocations.json',{'jobs':[job]})
        self.put('infrastructure/budget_reservations.json',{'reservations':[{'job':job['job'],'upper_bound_usd':14,'admitted_upper_bound_usd':14}], 'reconciliations':[]})
        return input_path,config
    def test_resource_ast_max_calls_budget_and_exact_central_payload_binding(self):
        name,config=self.central();checked=evidence.audit_central_admission(self.root,self.payload,name,require_app_id=True)
        self.assertEqual(checked['maximum_gpu_calls'],2);self.assertEqual(config['gpus'],2)
        rows=json.loads((self.root/'infrastructure/resource_allocations.json').read_text());rows['jobs'][0]['frozen_files'][name]='a'*64
        self.put('infrastructure/resource_allocations.json',rows)
        with self.assertRaisesRegex(ValueError,'omitted exact'):evidence.audit_central_admission(self.root,self.payload,name,require_app_id=True)
    def calls(self,pilot,full):
        exp=self.payload['experiment'];self.put(f'monitor_inputs/{exp}.json',self.payload)
        for stage,callid in [('pilot','fc-pilot'),('full','fc-full')]:
            record={'schema':'exp22_monitor_call_v1','state':'SPAWNED','stage':stage,'job':exp,'payload_path':f'monitor_inputs/{exp}.json',
                'payload_sha256':self.payload['payload_sha256'],'worker_module':'modal_exp22_monitor','worker_function':'monitor','gpu':'H200:1',
                'call_id':callid,'spawn_intent_utc':'2026-01-01T00:00:00+00:00','submitted_utc':'2026-01-01T00:00:01+00:00'}
            if stage=='full':record.update(spawn_intent_utc=pilot['finished_utc'],submitted_utc=pilot['finished_utc'],
                pilot_call_id='fc-pilot',pilot_audit_sha256=digest(pilot['audit']),pilot_completed_utc=pilot['finished_utc'])
            self.put(f'group_calls/{exp}/{stage}.json',record)
            self.put(f'group_calls/{exp}/{stage}_worker_started.json',{'call_id':callid,'payload_sha256':self.payload['payload_sha256'],'stage':stage})
        return lambda cid:{'call_id':cid,'function_name':'monitor','module_name':'modal_exp22_monitor','status':'SUCCESS'}
    def test_complete_independent_actual_call_audit_and_before_pilot_full_rejection(self):
        _,pilot,_=self.execute('pilot');_,full,_=self.execute('full');self.central();lookup=self.calls(pilot,full)
        with patch.object(evidence,'authorize_payload',return_value=self.gate):
            result=evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=lookup)
            self.assertTrue(result['execution_provenance_verified']);self.assertTrue(result['H5_before_full_call_verified'])
            self.assertFalse(result['live_monitor_stream_verified']);self.assertFalse(result['complete_experiment_receipt'])
            path=evidence.ledger_path(self.root,self.payload,'full');row=json.loads(path.read_text());row['spawn_intent_utc']='2026-01-01T00:00:00+00:00';path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'did not follow'):evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=lookup)
    def test_backend_must_confirm_real_monitor_completion_and_worker_start(self):
        _,pilot,_=self.execute('pilot');_,full,_=self.execute('full');self.central();lookup=self.calls(pilot,full)
        with patch.object(evidence,'authorize_payload',return_value=self.gate):
            with self.assertRaisesRegex(ValueError,'Actual backend'):
                evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=lambda c:{**lookup(c),'function_name':'records'})
            path=evidence.ledger_path(self.root,self.payload,'pilot').parent/'pilot_worker_started.json';path.unlink()
            with self.assertRaises(FileNotFoundError):evidence.audit_complete_run(self.root,self.payload,tokenizer=Tokenizer(),lookup_call=lookup)

    def runner_function(self,name,extra):
        import ast
        tree=ast.parse((ROOT/'scripts/modal_exp22_monitor.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);function.decorator_list=[]
        namespace={'Path':lambda p:self.root.parent,'NS':self.root.name,'json':json,
            'artifacts':SimpleNamespace(reload=lambda:None,commit=lambda:None),
            'weights':SimpleNamespace(reload=lambda:None),**extra}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_prepared_runner','exec'),namespace)
        return namespace[name]

    def test_coordinator_requires_pilot_audit_and_recovers_without_duplicate_spawn(self):
        from unittest.mock import Mock
        _,pilot,_=self.execute('pilot');spawned=[];calls={}
        def spawn(payload,stage):
            spawned.append(stage);call=SimpleNamespace(object_id='fc-'+stage,get=lambda:pilot if stage=='pilot' else {'fixture_full':True})
            calls[call.object_id]=call;return call
        coordinate=self.runner_function('coordinate',{'monitor':SimpleNamespace(spawn=spawn),
            'modal':SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda cid:calls[cid]))})
        with patch.object(evidence,'audit_pilot_completion',side_effect=ValueError('Pilot failed')):
            with self.assertRaisesRegex(ValueError,'Pilot failed'):coordinate(self.payload)
        self.assertEqual(spawned,['pilot'])
        with patch.object(evidence,'audit_pilot_completion',return_value=(pilot['audit'],pilot)):
            self.assertEqual(coordinate(self.payload),{'fixture_full':True});coordinate(self.payload)
            self.assertEqual(spawned,['pilot','full'])
            path=evidence.ledger_path(self.root,self.payload,'full');row=json.loads(path.read_text());row['state']='SPAWN_INTENT';path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'Committed concrete'):coordinate(self.payload)
        self.assertEqual(spawned,['pilot','full'])

    def test_actual_worker_call_mismatch_cannot_start_model_subprocess(self):
        monitor=self.runner_function('monitor',{'modal':SimpleNamespace(current_function_call_id=lambda:'fc-wrong')})
        with patch.object(evidence,'durable_call',return_value={'call_id':'fc-pilot'}),patch('subprocess.Popen') as process:
            with self.assertRaisesRegex(ValueError,'Actual monitor call'):monitor(self.payload,'pilot')
            process.assert_not_called()

    def test_llama_pinned_monitor_uses_verdict_prompt_without_tool_header(self):
        provenance={};sources=dict(self.payload['input_sources'])
        for key in ('download','preflight'):
            path='fixture/'+key+'.json';self.put(path,{});sources[path]=hash_bytes(self.root/path);provenance[key]={'path':path,'sha256':sources[path]}
        self.payload=build_payload('llama_fixture',self.design,sweeps(self.design)[12]['sweep_id'],phase3_inputs=self.inputs,phase3_audit=self.gate,
            dependency_audit=self.dependency,input_sources=sources,source_hashes=self.payload['source_hashes'],
            tokenizer_files=self.payload['tokenizer_files'],tokenizer_source_paths=self.payload['tokenizer_source_paths'],model_provenance=provenance)
        run,status,_=self.execute('pilot');self.assertTrue(status['audit']['H5'])
        rows,_=read_chain(run.p['raw'],self.payload['payload_sha256'])
        self.assertEqual(rows[0]['identity']['family'],'llama');self.assertNotIn('enable_thinking',rows[0]['full_prompt'])
        self.assertNotIn('<tool_call>',rows[0]['full_prompt']);self.assertIn('VERDICT:',rows[0]['full_prompt'])

if __name__=='__main__':unittest.main()
