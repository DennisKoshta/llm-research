"""CPU-only dose context guard tests; no GPU or scientific source data."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from primary_model_loader import runtime_limits, load_primary
import sandbox_evaluation_worker as worker
from modal_dose_evaluation import arm_payload
from test_dose_evaluation import fixture_design


class Tokenizer:
    def encode(self,text,add_special_tokens=True):
        assert add_special_tokens is True
        return [ord(c) for c in text]
    def apply_chat_template(self,history,**kwargs):
        return json.dumps(history,sort_keys=True)+'\n<think>\n'


class Model:
    def __init__(self,context=65536):
        self.llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=context),
            scheduler_config=SimpleNamespace(max_num_seqs=4,max_num_batched_tokens=8192,enable_chunked_prefill=True)))
        self.batch_sizes=[];self.prompts=[]
    def get_tokenizer(self):return Tokenizer()
    def generate(self,prompts,params,use_tqdm=False):
        self.batch_sizes.append(len(prompts));self.prompts.extend(deepcopy(prompts))
        assert len(prompts)<=4
        assert all(set(p)=={'prompt_token_ids'} for p in prompts)
        raw='UNIT reasoning.</think><tool_call>{"name":"commit_ledger_direct","arguments":{}}</tool_call>'
        return [SimpleNamespace(outputs=[SimpleNamespace(text=raw,token_ids=list(range(len(raw))),finish_reason='stop')]) for _ in prompts]


class DoseContextTests(unittest.TestCase):
    def test_legacy_loader_defaults_and_explicit_chunked_prefill(self):
        self.assertEqual(runtime_limits(),{'max_model_len':16384,'max_num_seqs':32})
        expected={'max_model_len':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,'enable_chunked_prefill':True}
        self.assertEqual(runtime_limits(65536,4,8192),expected)
        for args in [(65536,0,8192),(True,4,8192),(65536,4,70000)]:
            with self.assertRaises(AssertionError):runtime_limits(*args)
        calls=[]
        def llm(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(collective_rpc=lambda *a,**k:['UNIT'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder);(root/'config.json').write_text(json.dumps({'text_config':{'max_position_embeddings':262144},
                'quantization_config':{'modules_to_not_convert':[]}}))
            with patch('primary_model_loader.checkpoint_manifest',return_value={'quantized_modules':[f'unit{i}' for i in range(400)]}), \
                 patch.dict('sys.modules',{'torch':SimpleNamespace(),'vllm':SimpleNamespace(LLM=llm)}):
                load_primary(root,root,context_limit=65536,max_num_seqs=4,max_num_batched_tokens=8192)
                load_primary(root,root)
            self.assertEqual({key:calls[0][key] for key in expected},expected)
            self.assertEqual(calls[1]['max_model_len'],16384)
            self.assertEqual(calls[1]['max_num_seqs'],32)
            self.assertNotIn('max_num_batched_tokens',calls[1])
            self.assertNotIn('enable_chunked_prefill',calls[1])

    def test_exact_boundary_and_no_clipping_on_long_history(self):
        text='x'*61440
        ids=worker.check_context_budget(Tokenizer(),[text],4096,65536)
        self.assertEqual(ids,[list(map(ord,text))])
        with self.assertRaisesRegex(AssertionError,'no clipping'):
            worker.check_context_budget(Tokenizer(),[text+'x'],4096,65536)
        self.assertEqual(len(text),61440)
        with self.assertRaisesRegex(AssertionError,'CONTEXT_BUDGET_EXCEEDED'):
            worker.check_context_budget(Tokenizer(),['x'*12289],4096,16384)

    def test_all_five_dose_payloads_freeze_same_limits(self):
        _,_,config=fixture_design();auth={'checkpoints':{a['arm']:{} for a in config['arms']}}
        payloads=[arm_payload(config,a,'unit',auth) for a in config['arms']]
        self.assertTrue(all((p['context_limit'],p['max_num_seqs'],p['max_num_batched_tokens'],p['cap'])==(65536,4,8192,4096) for p in payloads))

    def run_worker(self,model):
        _,_,design=fixture_design();prototype=design['trials'][0]
        pilots=[{**prototype,'trial_id':f'unit_pilot_{i}'} for i in range(5)]
        trials=[{**prototype,'trial_id':f'unit_full_{i}'} for i in range(9)]
        payload={'namespace':'UNIT_ONLY','experiment':'unit','attempt':'fixture','arm':'dose_0_0','cot_mode':'cot_on',
                 'cap':4096,'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,
                 'model_path':'UNIT_MODEL','pilots':pilots,'trials':trials}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder);source=root/'payload.json';source.write_text(json.dumps(payload))
            def mapped(path):return root/'artifacts' if str(path)=='/artifacts' else Path(path)
            with patch.object(worker,'Path',side_effect=mapped), \
                 patch.object(worker,'load_primary',return_value=(model,[{'UNIT':True}])) as loader, \
                 patch.dict('sys.modules',{'vllm':SimpleNamespace(SamplingParams=lambda **k:SimpleNamespace(**k))}), \
                 patch.object(worker.sys,'stdin',io.StringIO('COMMITTED\n'*200)),redirect_stdout(io.StringIO()):
                worker.main(str(source))
            folder=root/'artifacts/UNIT_ONLY'
            raw=[json.loads(line) for line in (folder/'raw_transcripts/unit_fixture/dose_0_0_cot_on.jsonl').read_text().splitlines()]
            journal=[json.loads(line) for line in (folder/'raw_transcripts/unit_fixture/dose_0_0_cot_on_requests.jsonl').read_text().splitlines()]
            status=json.loads((folder/'evaluation_status/unit_fixture_dose_0_0_cot_on.json').read_text())
            return raw,journal,status,loader.call_args

    def test_actual_runtime_checked_and_full_token_ids_persisted(self):
        model=Model();raw,journal,status,call=self.run_worker(model)
        self.assertEqual(len(raw),14)
        self.assertEqual(status['status'],'COMPLETE')
        self.assertEqual(status['actual_runtime_limits'],{'context_limit':65536,'max_num_seqs':4,
            'max_num_batched_tokens':8192,'enable_chunked_prefill':True})
        self.assertEqual(call.kwargs,{'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192})
        self.assertEqual(model.batch_sizes,[4,1,4,4,1])
        for row,request,prompt in zip(raw,journal,model.prompts):
            self.assertEqual(row['prompt_token_ids'],Tokenizer().encode(row['full_prompt']))
            self.assertEqual(request['prompt_token_ids'],row['prompt_token_ids'])
            self.assertEqual(prompt['prompt_token_ids'],row['prompt_token_ids'])
            self.assertEqual(row['context_limit'],65536)
            self.assertLessEqual(len(row['prompt_token_ids'])+4096,65536)

    def test_wrong_actual_runtime_fails_before_any_generation(self):
        model=Model(context=16384)
        with self.assertRaises(AssertionError):self.run_worker(model)
        self.assertFalse(model.prompts)


if __name__=='__main__':unittest.main()
