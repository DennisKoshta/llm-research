"""CPU-only uniform format-clarification tests; no inference or paid calls."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import modal_dose_evaluation_d4 as runner
import sandbox_evaluation_worker as worker
from launch_bounded_job import _decorators, validate_resource_profile
from test_dose_evaluation import fixture_design
from test_dose_context_budget import Tokenizer, Model


def retry_fixture():
    return {'policy':{'path':'infrastructure/dose_uniform_budget_retry_rule_v1.json','sha256':'1'*64},
        'previous_receipt':{'path':'records/phase1_dose_response_author_v1_d2_descriptive_receipt.json','sha256':'2'*64},
        'previous_config':{'path':'stimuli/dose_evaluation_v1_author_v1_d2_descriptive.json','sha256':'3'*64},
        'previous_attempt':'author_v1_d2_descriptive','old_cap':4096,'new_cap':8192,
        'unchanged_design_sha256':'4'*64,'pooling_allowed':False,'scientific_effect_used_for_retry_selection':False}


def fixtures():
    config=fixture_design()[2];config.update(cap=8192,descriptive_attempt=runner.DESCRIPTIVE_ATTEMPT,
        schema='phase1_descriptive_dose_evaluation_design_v3',eligibility_scope=runner.DESCRIPTIVE_SCOPE,
        mechanical_retry=retry_fixture())
    correction={'instruction':runner.FORMAT_INSTRUCTION,'placement':'system_message_suffix',
        'old_prompt_rule_explicitly_superseded':True,'generated_prefix_injected':False,
        'semantic_fields_unchanged':True,'format_may_change_behavior':True,'no_further_wording_search':True}
    for field,kind in [('trials','trial'),('pilots','pilot')]:
        correction[f'original_{kind}_messages_sha256']=runner.digest([row['messages'] for row in config[field]])
        for row in config[field]:row['messages'][0]['content']+=' '+runner.FORMAT_INSTRUCTION
        config[f'shared_{kind}_messages_sha256']=runner.digest([row['messages'] for row in config[field]])
        correction[f'amended_{kind}_messages_sha256']=config[f'shared_{kind}_messages_sha256']
    config['format_correction']=correction
    config['administrative_amendment']={'path':'infrastructure/phase1_descriptive_dose_format_retry_v1.json',
        'sha256':'5'*64,'definition':{'schema':'phase1_descriptive_dose_format_retry_amendment_v1',
        'attempt':runner.DESCRIPTIVE_ATTEMPT,'maximum_gpu_workers':4,'mechanical_retry':retry_fixture(),
        'format_correction':deepcopy(correction),
        'runtime':{'cap':8192,'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,
                   'temperature':0.,'seed':0,'forced_prefix':None}}}
    auth={'phase2_authorized':False,'intact_calibration_passed':True,'admission_scope':runner.DESCRIPTIVE_SCOPE,
        'descriptive_attempt':runner.DESCRIPTIVE_ATTEMPT,'primary_qualification_passed':False,
        'descriptive_phase1_authorized':True,'dose_results_may_replace_qualification':False,
        'primary_mechanical_checks':{key:True for key in ('all_six_raw_generation_streams','all_six_generation_hygiene',
            'all_six_complete_counts','separate_forward_pilots','post_fp8_projection_passed')},
        'administrative_amendment':deepcopy(config['administrative_amendment']),
        'format_correction':deepcopy(correction),
        'mechanical_retry':retry_fixture(),'checkpoints':{arm['arm']:{} for arm in config['arms']},
        'config_sha256':runner.digest(config)}
    return config,auth


class RetryModel(Model):
    def __init__(self,mode='valid'):
        super().__init__();self.mode=mode;self.parameters=[];self.seen=0
    def generate(self,prompts,params,use_tqdm=False):
        self.parameters.append(vars(params));outputs=super().generate(prompts,params,use_tqdm)
        for output in outputs:
            if self.mode=='unclosed_pilot' and self.seen==0:
                output.outputs[0].text='Unclosed reasoning'
            if self.mode=='capped_full' and self.seen==5:
                output.outputs[0].token_ids=list(range(8192));output.outputs[0].finish_reason='length'
            self.seen+=1
        return outputs


class D4RunnerTests(unittest.TestCase):
    def test_uniform_cap_preserves_all_shared_inputs_and_checkpoint_paths(self):
        config,auth=fixtures();old=fixture_design()[2]
        jobs=[runner.arm_payload(config,a,runner.DESCRIPTIVE_ATTEMPT,auth) for a in config['arms']]
        self.assertEqual([p['alpha'] for p in jobs],[-.5,0.,.5,1.,1.5])
        self.assertTrue(all((p['cap'],p['context_limit'],p['max_num_seqs'],p['max_num_batched_tokens'])==
                            (8192,65536,4,8192) for p in jobs))
        for field in ('trials','pilots'):
            for before,after in zip(old[field],jobs[0][field]):
                restored=deepcopy(after)
                self.assertEqual(restored['messages'][0]['content'],before['messages'][0]['content']+' '+runner.FORMAT_INSTRUCTION)
                restored['messages'][0]['content']=before['messages'][0]['content']
                self.assertEqual(restored,before)
            self.assertTrue(all(p[field]==jobs[0][field] for p in jobs))
        self.assertTrue(all(p['format_correction']==config['format_correction'] for p in jobs))
        self.assertEqual([p['model_path'] for p in jobs],[a['model_path'] for a in old['arms']])
        with self.assertRaises(AssertionError):runner.arm_payload(config,config['arms'][0],'author_v1_d2_descriptive',auth)
        config['cap']=4096
        with self.assertRaises(AssertionError):runner.arm_payload(config,config['arms'][0],runner.DESCRIPTIVE_ATTEMPT,auth)

    def test_retry_cannot_waive_failure_change_caps_or_pool_attempts(self):
        _,auth=fixtures();runner.require_evaluation_authorization(auth,runner.DESCRIPTIVE_ATTEMPT)
        for key,value in [('primary_qualification_passed',True),('phase2_authorized',True),
                          ('dose_results_may_replace_qualification',True),('primary_mechanical_checks',{})]:
            bad=deepcopy(auth);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                runner.require_evaluation_authorization(bad,runner.DESCRIPTIVE_ATTEMPT)
        for key,value in [('old_cap',2048),('new_cap',16384),('pooling_allowed',True),
                          ('scientific_effect_used_for_retry_selection',True),('previous_attempt','different')]:
            bad=deepcopy(auth);bad['mechanical_retry'][key]=value
            bad['administrative_amendment']['definition']['mechanical_retry']=deepcopy(bad['mechanical_retry'])
            with self.subTest(key=key),self.assertRaises(AssertionError):
                runner.require_evaluation_authorization(bad,runner.DESCRIPTIVE_ATTEMPT)
        with self.assertRaises(AssertionError):runner.require_evaluation_authorization(auth,'author_v1_d2_descriptive')

    def test_exact_instruction_and_disclosed_confound_cannot_be_relaxed(self):
        _,auth=fixtures()
        for key,value in [('instruction',runner.FORMAT_INSTRUCTION+' Modified.'),('placement','assistant_prefix'),
            ('old_prompt_rule_explicitly_superseded',False),('generated_prefix_injected',True),
            ('semantic_fields_unchanged',False),('format_may_change_behavior',False),('no_further_wording_search',False)]:
            bad=deepcopy(auth);bad['format_correction'][key]=value
            bad['administrative_amendment']['definition']['format_correction']=deepcopy(bad['format_correction'])
            with self.subTest(key=key),self.assertRaises(AssertionError):
                runner.require_evaluation_authorization(bad,runner.DESCRIPTIVE_ATTEMPT)
        bad=deepcopy(auth);bad['format_correction']['extra_allowance']=True
        bad['administrative_amendment']['definition']['format_correction']=deepcopy(bad['format_correction'])
        with self.assertRaises(AssertionError):runner.require_evaluation_authorization(bad,runner.DESCRIPTIVE_ATTEMPT)

    def test_all53_exact_suffixes_and_original_hashes_are_required(self):
        config,_=fixtures();runner.verify_format_inputs(config)
        for field,index in [('pilots',0),('trials',47)]:
            for change in ('missing','newline','duplicated','user_change'):
                bad=deepcopy(config);row=bad[field][index]
                if change=='missing':row['messages'][0]['content']=row['messages'][0]['content'][:-len(' '+runner.FORMAT_INSTRUCTION)]
                elif change=='newline':row['messages'][0]['content']=row['messages'][0]['content'].replace(' '+runner.FORMAT_INSTRUCTION,'\n'+runner.FORMAT_INSTRUCTION)
                elif change=='duplicated':row['messages'][0]['content']+=' '+runner.FORMAT_INSTRUCTION
                else:row['messages'][1]['content']+=' Modified.'
                with self.subTest(field=field,change=change),self.assertRaises(AssertionError):runner.verify_format_inputs(bad)
        for key in ('original_trial_messages_sha256','amended_trial_messages_sha256','original_pilot_messages_sha256','amended_pilot_messages_sha256'):
            bad=deepcopy(config);bad['format_correction'][key]='0'*64
            with self.subTest(key=key),self.assertRaises(AssertionError):runner.verify_format_inputs(bad)

    def test_payload_rejects_a_different_format_correction_or_stale_prompt_hash(self):
        config,auth=fixtures();bad=deepcopy(auth);bad['format_correction']['original_trial_messages_sha256']='0'*64
        with self.assertRaises(AssertionError):runner.arm_payload(config,config['arms'][0],runner.DESCRIPTIVE_ATTEMPT,bad)
        bad=deepcopy(config);bad['shared_pilot_messages_sha256']='0'*64
        with self.assertRaises(AssertionError):runner.arm_payload(bad,bad['arms'][0],runner.DESCRIPTIVE_ATTEMPT,auth)

    def test_four_worker_resource_bound_is_five_calls_without_retries(self):
        path=Path(runner.__file__);raw=path.read_bytes();decorators=_decorators(raw.decode())
        self.assertEqual(set(decorators),{'evaluate','coordinate'})
        limits=decorators['evaluate']
        self.assertEqual((limits['gpu'],limits['max_containers'],limits['timeout'],limits['startup_timeout'],limits['retries']),
                         ('H100',4,3600,300,0))
        config={'gpus':4,'single_gpu_workers':4,'tp8_workers':0,'upper_bound_usd':30,
            'command':['modal','run','scripts/modal_dose_evaluation_d4.py'],
            'resource_profile':{'source':'scripts/modal_dose_evaluation_d4.py',
                'source_sha256':hashlib.sha256(raw).hexdigest(),
                'functions':{name:{'max_calls':n,'limits':decorators[name]} for name,n in [('evaluate',5),('coordinate',1)]},
                'excluded_functions':{},'call_bound_basis':'Five fixed alpha calls; one coordinator; no retries.'}}
        audited=validate_resource_profile(config,runner.ROOT)
        self.assertEqual(audited['whole_app_allocation']['gpus'],4)
        self.assertLessEqual(audited['minimum_compute_reservation_usd'],30)

    def test_coordinator_spawns_exactly_five_then_reattaches_without_new_calls(self):
        config,auth=fixtures();calls=[];saved={}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder)
            def spawn(payload):
                calls.append(deepcopy(payload));ident='fc-synthetic-'+str(len(calls))
                result={'status':'COMPLETE','arm':payload['arm']}
                call=SimpleNamespace(object_id=ident,get=lambda:result);saved[ident]=call;return call
            namespace=dict(vars(runner));namespace.update(
                Path=lambda p:root if p=='/artifacts' else Path(p),
                artifacts=SimpleNamespace(reload=lambda:None,commit=lambda:None),
                weights=SimpleNamespace(reload=lambda:None),evaluate=SimpleNamespace(spawn=spawn),
                tokenizer_hashes=lambda p:{'tokenizer.json':'f'*64,'tokenizer_config.json':'e'*64},
                modal=SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda ident:saved[ident])))
            node=next(n for n in ast.parse(Path(runner.__file__).read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='coordinate');node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),'synthetic_coordinator','exec'),namespace)
            first=namespace['coordinate'](config,runner.DESCRIPTIVE_ATTEMPT,auth)
            self.assertEqual(namespace['coordinate'](config,runner.DESCRIPTIVE_ATTEMPT,auth),first)
            self.assertEqual(len(calls),5);self.assertFalse(first['phase2_authorized'])
            self.assertEqual(len(list((root/runner.NS/'group_calls'/f'{runner.EXPERIMENT}_{runner.DESCRIPTIVE_ATTEMPT}').glob('dose_*.json'))),5)
            self.assertTrue(all(p['cap']==8192 and p['attempt']==runner.DESCRIPTIVE_ATTEMPT for p in calls))

    def run_worker(self,model):
        config,auth=fixtures();prototype=config['trials'][0]
        payload=runner.arm_payload(config,config['arms'][0],runner.DESCRIPTIVE_ATTEMPT,auth)
        payload['pilots']=[{**prototype,'trial_id':f'synthetic_pilot_{i}'} for i in range(5)]
        payload['trials']=[{**prototype,'trial_id':f'synthetic_full_{i}'} for i in range(48)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder);source=root/'input.json';source.write_text(json.dumps(payload))
            with patch.object(worker,'Path',side_effect=lambda p:root/'artifacts' if str(p)=='/artifacts' else Path(p)), \
                 patch.object(worker,'load_primary',return_value=(model,[{'synthetic_only':True}])), \
                 patch.dict('sys.modules',{'vllm':SimpleNamespace(SamplingParams=lambda **kw:SimpleNamespace(**kw))}), \
                 patch.object(worker.sys,'stdin',io.StringIO('COMMITTED\n'*300)),redirect_stdout(io.StringIO()):
                worker.main(str(source))
            stem=runner.EXPERIMENT+'_'+runner.DESCRIPTIVE_ATTEMPT
            base=root/'artifacts'/runner.NS
            raw=[json.loads(line) for line in (base/f'raw_transcripts/{stem}/dose_m0_5_cot_on.jsonl').read_text().splitlines()]
            requests=[json.loads(line) for line in (base/f'raw_transcripts/{stem}/dose_m0_5_cot_on_requests.jsonl').read_text().splitlines()]
            status=json.loads((base/f'evaluation_status/{stem}_dose_m0_5_cot_on.json').read_text())
            return raw,requests,status

    def test_frozen_worker_uses8192_and_runs_five_pilots_before_full48(self):
        model=RetryModel();raw,requests,status=self.run_worker(model)
        self.assertEqual([r['split'] for r in raw],['pilot']*5+['evaluation']*48)
        self.assertEqual(status['status'],'COMPLETE');self.assertEqual(status['n_episodes'],48)
        self.assertTrue(all(p=={'temperature':0.0,'max_tokens':8192,'seed':0,'skip_special_tokens':False} for p in model.parameters))
        self.assertTrue(all(r['max_new_tokens']==8192 and r['forced_prefix'] is None for r in raw+requests))
        self.assertTrue(all(r['prompt_token_ids']==q['prompt_token_ids'] and len(r['prompt_token_ids'])+8192<=65536
                            for r,q in zip(raw,requests)))

    def test_frozen_worker_keeps_h5_and_one_of48_h1_failures(self):
        raw,_,status=self.run_worker(RetryModel('unclosed_pilot'))
        self.assertEqual(status['status'],'H5_FAILED');self.assertEqual(len(raw),5)
        raw,_,status=self.run_worker(RetryModel('capped_full'))
        self.assertEqual(status['status'],'H1_FAILED');self.assertEqual(status['capped_episodes'],1)
        self.assertEqual(len(raw),53)

    def test_context_guard_accounts_for_full8192_without_clipping(self):
        ids=worker.check_context_budget(Tokenizer(),['x'*57344],8192,65536)
        self.assertEqual(len(ids[0]),57344)
        with self.assertRaisesRegex(AssertionError,'no clipping'):
            worker.check_context_budget(Tokenizer(),['x'*57345],8192,65536)

    def test_authorization_reconstructs_new_config_and_retains_independent_gates(self):
        from test_descriptive_dose_eligibility import DescriptiveDoseTests
        fixture=DescriptiveDoseTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=fixture.config();prototype,auth=fixtures()
        config.update({key:prototype[key] for key in ('cap','schema','eligibility_scope','descriptive_attempt','mechanical_retry',
            'format_correction','trials','pilots','shared_trial_messages_sha256','shared_pilot_messages_sha256')})
        definition=config['administrative_amendment']['definition']
        definition.update(prototype['administrative_amendment']['definition'])
        config['administrative_amendment'].update(path=prototype['administrative_amendment']['path'],sha256='5'*64)
        with patch('prepare_dose_evaluation_d4.prepare_evaluation_config',return_value=config) as prepared:
            authorization=runner.authorize(fixture.root,config,fixture.primary_path,fixture.calibration_path)
            prepared.assert_called_once_with(fixture.root,'v1',runner.DESCRIPTIVE_ATTEMPT,
                'infrastructure/phase1_descriptive_dose_format_retry_v1.json')
            runner.require_evaluation_authorization(authorization,runner.DESCRIPTIVE_ATTEMPT)
            self.assertFalse(authorization['phase2_authorized'] or authorization['primary_qualification_passed'])
            self.assertEqual(authorization['mechanical_retry'],retry_fixture())
            with patch('audit_calibration_provenance.recompute',return_value={'provenance_passed':False}):
                with self.assertRaisesRegex(AssertionError,'Calibration prompt'):
                    runner.authorize(fixture.root,config,fixture.primary_path,fixture.calibration_path)


if __name__=='__main__':unittest.main()
