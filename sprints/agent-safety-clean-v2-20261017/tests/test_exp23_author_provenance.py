"""Synthetic only; no cloud calls or empirical outputs."""
from copy import deepcopy
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import capture_exp23_author_provenance as capture
import replay_exp23_author_tokenizer as replay
import audit_exp23_author_provenance as audit
from test_exp23_author_execution import payload,Tokenizer,Model,AuthorRun

class AuthorProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.payload=payload();self.pp='stimuli/'+self.payload['experiment']+'_input.json'
        self.put(self.pp,self.payload);self.payload=json.loads((self.root/self.pp).read_text());self.cp='infrastructure/capture/capture.json'
        self.files={}
        for name in ('tokenizer.json','tokenizer_config.json'):
            self.put('infrastructure/capture/tokenizer/'+name,{})
            self.files[name]={'sha256':capture.hash_bytes((self.root/'infrastructure/capture/tokenizer'/name).read_bytes())}
        self.value={'schema':'exp23_author_postlaunch_capture_v1','capture_timing':'POST_LAUNCH',
            'generation_time_token_ids_attested':False,'generation_time_package_versions_attested':False,
            'local_payload_path':self.pp,'local_payload_sha256':capture.hash_bytes((self.root/self.pp).read_bytes()),
            'payload_sha256':self.payload['payload_sha256'],'files':self.files,
            'packages':{n:'fixture-version' for n in ('transformers','tokenizers','Jinja2','huggingface-hub')},
            'runtime_source_sha256':self.payload['source_hashes'],'captured_utc':'2026-10-18T02:00:00+00:00',
            'source_app_id':'ap-fixture','limitations':['Post-launch fixture, no original prompt ID attestation'],
            **{k:self.payload[k] for k in ('model_id','revision','model_path')}}
        self.put(self.cp,self.value)
    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(capture.canonical(value));return path
    def run_replay(self):
        with patch('replay_exp23_author_tokenizer.importlib.metadata.version',return_value='fixture-version'):
            return replay.replay(self.root,self.pp,self.cp,tokenizer=Tokenizer())
    def test_all53_request_reconstruction_and_explicit_postlaunch_limit(self):
        value=self.run_replay();self.assertEqual(len(value['requests']),53)
        self.assertFalse(value['generation_time_token_ids_attested']);self.assertFalse(value['generation_time_package_versions_attested'])
        self.assertTrue(all(r['prompt_tokens']>0 for r in value['requests']))
    def test_capture_tokenizer_or_scope_tamper_fails(self):
        self.put('infrastructure/capture/tokenizer/tokenizer.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.run_replay()
        self.value['generation_time_token_ids_attested']=True;self.put(self.cp,self.value)
        with self.assertRaisesRegex(ValueError,'generation-time'):self.run_replay()
    def test_exact_package_versions_required_even_with_fixture_tokenizer(self):
        with patch('replay_exp23_author_tokenizer.importlib.metadata.version',return_value='other'):
            with self.assertRaisesRegex(ValueError,'package versions differ'):
                replay.replay(self.root,self.pp,self.cp,tokenizer=Tokenizer())
    def test_capture_remote_program_has_no_mutations_or_inference(self):
        text=capture.remote_program(self.payload)
        compile(text,'capture','exec')
        for forbidden in ('write_text','write_bytes','model.generate','subprocess','os.environ'):
            self.assertNotIn(forbidden,text)
        self.assertIn('read_bytes',text);self.assertIn('importlib.metadata.version',text)
    def completed_fixture(self):
        run=AuthorRun(self.root,self.payload,lambda:None);run.execute(Model(run),object())
        self.put('group_calls/'+self.payload['experiment']+'.json',{'schema':'exp23_author_call_ledger_v1','state':'SPAWNED',
            'attempt':self.payload['attempt'],'payload_sha256':self.payload['payload_sha256'],'call_id':'fc-fixture',
            'created_at_utc':'2026-10-18T00:00:00+00:00','submitted_at_utc':'2026-10-18T00:00:01+00:00'})
        for name in ('capture_exp23_author_provenance.py','replay_exp23_author_tokenizer.py','audit_exp23_author_provenance.py'):
            path=self.root/'scripts'/name;path.parent.mkdir(exist_ok=True);path.write_bytes((ROOT/'scripts'/name).read_bytes())
        return self.run_replay()
    def complete(self,value,backend=None):
        with patch.object(audit,'central_admission',return_value={'source_sha256':{}}):
            return audit.recompute(self.root,self.pp,self.cp,replay_runner=lambda *args:value,
                lookup_call=lambda _:backend or {'call_id':'fc-fixture','function_name':'author','module_name':'modal_exp23_author','status':'SUCCESS'})
    def test_complete_raw_prompt_count_replay_and_backend_provenance(self):
        value=self.completed_fixture();result=self.complete(value)
        self.assertTrue(result['complete_data']);self.assertEqual(result['request_count'],53)
        self.assertFalse(result['phase2_execution_authorized']);self.assertFalse(result['generation_time_token_ids_attested'])
        changed=deepcopy(value);changed['requests'][-1]['prompt_tokens']+=1
        with self.assertRaisesRegex(ValueError,'prompt/count'):self.complete(changed)
        with self.assertRaisesRegex(ValueError,'not successfully complete'):
            self.complete(value,{'call_id':'fc-fixture','function_name':'records','module_name':'modal_structured_records','status':'SUCCESS'})
    def test_full_membership_cannot_omit_invalid_or_last_response(self):
        value=self.completed_fixture();value['requests'].pop()
        with self.assertRaisesRegex(ValueError,'53-request'):self.complete(value)
    def test_durable_central_app_binding_required_before_history_validation(self):
        self.put('infrastructure/resource_allocations.json',{'jobs':[{'job':'exp23_unified_author_'+self.payload['attempt']}]})
        self.put('infrastructure/budget_reservations.json',{'reservations':[{'job':'exp23_unified_author_'+self.payload['attempt']}]})
        with self.assertRaisesRegex(ValueError,'durably reconciled'):
            audit.central_admission(self.root,self.pp,self.payload,self.value)

    def test_dependency_bridge_requires_fresh_complete_honest_provenance(self):
        from audit_exp23_launch import audit_author_postlaunch
        expected={'complete_data':True,'postlaunch_prompt_replay_verified':True,'capture_timing':'POST_LAUNCH',
            'generation_time_token_ids_attested':False,'generation_time_package_versions_attested':False}
        with patch.object(audit,'recompute',return_value=expected) as checked:
            self.assertEqual(audit_author_postlaunch(self.root,self.pp,self.payload),expected)
            checked.assert_called_once_with(self.root,self.pp,
                'infrastructure/author_provenance/'+self.payload['attempt']+'/postlaunch_v1/capture.json')
        for field in ('complete_data','postlaunch_prompt_replay_verified'):
            bad={**expected,field:False}
            with patch.object(audit,'recompute',return_value=bad):
                with self.assertRaisesRegex(ValueError,'honest post-launch'):audit_author_postlaunch(self.root,self.pp,self.payload)
        with patch.object(audit,'recompute',side_effect=FileNotFoundError('capture missing')):
            with self.assertRaises(FileNotFoundError):audit_author_postlaunch(self.root,self.pp,self.payload)
        with patch.object(audit,'recompute',return_value={**expected,'generation_time_token_ids_attested':True}):
            with self.assertRaisesRegex(ValueError,'honest post-launch'):audit_author_postlaunch(self.root,self.pp,self.payload)

if __name__=='__main__':unittest.main()
